"""
多模态大模型交叉验证模块

实现双模型（Gemini 3.1 Pro + Opus 4.6）交叉验证机制
"""
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResponse:
    """单个模型的响应"""
    model_name: str
    response_text: str
    parsed_data: Optional[Dict[str, Any]]
    parse_success: bool
    error_message: Optional[str] = None


@dataclass
class CrossValidationResult:
    """交叉验证结果"""
    gemini_response: ModelResponse
    opus_response: ModelResponse
    has_differences: bool
    differences: List[Dict[str, Any]]
    final_data: Dict[str, Any]
    consensus_reached: bool
    validation_rounds: int


class CrossValidator:
    """
    交叉验证器

    使用两个模型（Gemini 3.1 Pro + Opus 4.6）进行交叉验证
    """

    def __init__(
        self,
        gemini_client,
        opus_client,
        max_validation_rounds: int = 3
    ):
        """
        初始化交叉验证器

        Args:
            gemini_client: Gemini LLM 客户端
            opus_client: Opus LLM 客户端
            max_validation_rounds: 最大验证轮数
        """
        self.gemini_client = gemini_client
        self.opus_client = opus_client
        self.max_validation_rounds = max_validation_rounds

    def validate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        expected_fields: Optional[List[str]] = None
    ) -> CrossValidationResult:
        """
        执行交叉验证

        Args:
            prompt: 提示词
            image_path: 图片路径
            expected_fields: 期望的字段列表（用于验证）

        Returns:
            CrossValidationResult: 交叉验证结果
        """
        logger.info("="*60)
        logger.info("开始交叉验证")
        logger.info("="*60)

        validation_round = 0
        gemini_response = None
        opus_response = None

        while validation_round < self.max_validation_rounds:
            validation_round += 1
            logger.info(f"验证轮次: {validation_round}/{self.max_validation_rounds}")

            # 第一轮：两个模型独立识别
            if validation_round == 1:
                logger.info("阶段 1: 两个模型独立识别")
                gemini_response = self._call_model(
                    self.gemini_client,
                    "Gemini 3.1 Pro",
                    prompt,
                    image_path
                )
                opus_response = self._call_model(
                    self.opus_client,
                    "Qwen3.5 (辅助)",
                    prompt,
                    image_path
                )

                # 如果两个模型都解析失败，直接返回
                if not gemini_response.parse_success and not opus_response.parse_success:
                    logger.error("两个模型都无法解析出有效的 JSON")
                    return CrossValidationResult(
                        gemini_response=gemini_response,
                        opus_response=opus_response,
                        has_differences=True,
                        differences=[],
                        final_data={},
                        consensus_reached=False,
                        validation_rounds=validation_round
                    )

                # 如果只有一个模型成功，使用成功的那个
                if gemini_response.parse_success and not opus_response.parse_success:
                    logger.warning("只有 Gemini 成功解析，使用 Gemini 的结果")
                    return CrossValidationResult(
                        gemini_response=gemini_response,
                        opus_response=opus_response,
                        has_differences=False,
                        differences=[],
                        final_data=gemini_response.parsed_data,
                        consensus_reached=False,
                        validation_rounds=validation_round
                    )

                if opus_response.parse_success and not gemini_response.parse_success:
                    logger.warning("只有辅助模型成功解析，使用辅助模型的结果")
                    return CrossValidationResult(
                        gemini_response=gemini_response,
                        opus_response=opus_response,
                        has_differences=False,
                        differences=[],
                        final_data=opus_response.parsed_data,
                        consensus_reached=False,
                        validation_rounds=validation_round
                    )

                # 两个模型都成功，比较结果
                logger.info("阶段 2: 比较两个模型的输出")
                differences = self._compare_results(
                    gemini_response.parsed_data,
                    opus_response.parsed_data
                )

                if not differences:
                    logger.info("✓ 两个模型输出一致，验证通过")
                    return CrossValidationResult(
                        gemini_response=gemini_response,
                        opus_response=opus_response,
                        has_differences=False,
                        differences=[],
                        final_data=gemini_response.parsed_data,
                        consensus_reached=True,
                        validation_rounds=validation_round
                    )

                # 有差异，进入交叉验证
                logger.warning(f"✗ 发现 {len(differences)} 处差异")
                for i, diff in enumerate(differences, 1):
                    logger.info(f"  差异 {i}: {diff['field']}")
                    logger.info(f"    Gemini: {diff['gemini_value']}")
                    logger.info(f"    Qwen3.5: {diff['opus_value']}")

            # 后续轮次：仅让 Gemini 作为主判断方复审（不对称策略）
            # Gemini 评测得分 0.901 >> Qwen 0.838，辅助模型仅提供参考，不再重新作答
            else:
                logger.info(f"阶段 3: Gemini 复审（第 {validation_round - 1} 轮，Gemini 为主判断方）")

                # 仅构建 Gemini 复审 prompt，Opus 不再重新调用
                gemini_review_prompt = self._build_gemini_review_prompt(
                    prompt,
                    opus_response.parsed_data,
                    differences
                )

                prev_gemini_response = gemini_response
                gemini_response = self._call_model(
                    self.gemini_client,
                    "Gemini 3.1 Pro",
                    gemini_review_prompt,
                    image_path
                )

                # Gemini 复审失败则沿用上一轮结果
                if not gemini_response.parse_success:
                    logger.warning("Gemini 复审解析失败，沿用上一轮 Gemini 结果")
                    gemini_response = prev_gemini_response
                    break

                # 直接以 Gemini 复审结果为最终答案，无需再次比较
                logger.info("✓ Gemini 复审完成，以 Gemini 结果为最终输出")
                return CrossValidationResult(
                    gemini_response=gemini_response,
                    opus_response=opus_response,
                    has_differences=True,
                    differences=differences,
                    final_data=gemini_response.parsed_data,
                    consensus_reached=True,
                    validation_rounds=validation_round
                )

        # 达到最大轮数，使用投票或合并策略
        logger.warning(f"达到最大验证轮数 ({self.max_validation_rounds})，使用合并策略")
        final_data = self._merge_results(
            gemini_response.parsed_data if gemini_response.parse_success else {},
            opus_response.parsed_data if opus_response.parse_success else {},
            differences
        )

        return CrossValidationResult(
            gemini_response=gemini_response,
            opus_response=opus_response,
            has_differences=True,
            differences=differences,
            final_data=final_data,
            consensus_reached=False,
            validation_rounds=validation_round
        )

    def _call_model(
        self,
        client,
        model_name: str,
        prompt: str,
        image_path: Optional[str]
    ) -> ModelResponse:
        """
        调用单个模型

        Args:
            client: LLM 客户端
            model_name: 模型名称
            prompt: 提示词
            image_path: 图片路径

        Returns:
            ModelResponse: 模型响应
        """
        logger.info(f"调用 {model_name}...")

        try:
            response_text = client.chat(prompt, image_path=image_path)

            # 尝试解析 JSON
            try:
                parsed_data = self._parse_json_response(response_text)
                logger.info(f"✓ {model_name} 解析成功")
                return ModelResponse(
                    model_name=model_name,
                    response_text=response_text,
                    parsed_data=parsed_data,
                    parse_success=True
                )
            except Exception as e:
                logger.warning(f"✗ {model_name} JSON 解析失败: {e}")
                return ModelResponse(
                    model_name=model_name,
                    response_text=response_text,
                    parsed_data=None,
                    parse_success=False,
                    error_message=str(e)
                )

        except Exception as e:
            logger.error(f"✗ {model_name} 调用失败: {e}")
            return ModelResponse(
                model_name=model_name,
                response_text="",
                parsed_data=None,
                parse_success=False,
                error_message=str(e)
            )

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析 JSON 响应（支持多种格式）

        Args:
            response: 响应字符串

        Returns:
            解析后的字典
        """
        import re

        # 策略 1: 直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 JSON 代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 策略 3: 提取 {} 包裹的内容
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        raise ValueError(f"无法解析 JSON: {response[:200]}")

    def _compare_results(
        self,
        gemini_data: Dict[str, Any],
        opus_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        比较两个模型的输出

        Args:
            gemini_data: Gemini 的输出
            opus_data: Opus 的输出

        Returns:
            差异列表
        """
        differences = []

        # 获取所有字段
        all_fields = set(gemini_data.keys()) | set(opus_data.keys())

        for field in all_fields:
            gemini_value = gemini_data.get(field)
            opus_value = opus_data.get(field)

            # 跳过元数据字段
            if field in ["notes", "reasoning", "description"]:
                continue

            # 比较值
            if gemini_value != opus_value:
                differences.append({
                    "field": field,
                    "gemini_value": gemini_value,
                    "opus_value": opus_value,
                    "type": type(gemini_value).__name__ if gemini_value is not None else "None"
                })

        return differences

    def _build_cross_validation_prompt(
        self,
        original_prompt: str,
        other_model_name: str,
        other_model_data: Dict[str, Any],
        differences: List[Dict[str, Any]]
    ) -> str:
        """
        构建交叉验证 prompt

        Args:
            original_prompt: 原始 prompt
            other_model_name: 另一个模型的名称
            other_model_data: 另一个模型的输出
            differences: 差异列表

        Returns:
            交叉验证 prompt
        """
        # 构建差异说明
        diff_text = "\n".join([
            f"- 字段 `{diff['field']}`:\n"
            f"  - {other_model_name} 识别为: {json.dumps(diff['opus_value'] if 'Opus' in other_model_name else diff['gemini_value'], ensure_ascii=False)}\n"
            f"  - 你识别为: {json.dumps(diff['gemini_value'] if 'Opus' in other_model_name else diff['opus_value'], ensure_ascii=False)}"
            for diff in differences
        ])

        cross_prompt = f"""{original_prompt}

**重要提示：交叉验证**

{other_model_name} 模型也识别了这张图纸，但在以下字段上与你的识别结果有所不同：

{diff_text}

{other_model_name} 的完整识别结果：
```json
{json.dumps(other_model_data, ensure_ascii=False, indent=2)}
```

请你：
1. 仔细查看图纸，重新确认这些有差异的字段
2. 结合 {other_model_name} 的识别结果，判断哪个更准确
3. 如果你认为自己的识别是正确的，请保持原结果
4. 如果你认为 {other_model_name} 的识别更准确，请采纳其结果
5. 如果两者都有道理，请选择更符合工程图纸规范的那个

请返回你重新确认后的完整 JSON 结果（格式与之前相同）。"""

        return cross_prompt

    def _build_gemini_review_prompt(
        self,
        original_prompt: str,
        opus_data: Optional[Dict[str, Any]],
        differences: List[Dict[str, Any]]
    ) -> str:
        """
        构建 Gemini 专用复审 prompt（不对称策略）

        Gemini 作为主判断方，Opus 的输出仅作参考。
        Gemini 根据自己对图纸的判断给出最终答案，不必向 Opus 妥协。

        Args:
            original_prompt: 原始 prompt
            opus_data: Opus 第一轮的输出（供参考）
            differences: 第一轮的差异列表

        Returns:
            Gemini 复审 prompt
        """
        diff_text = "\n".join([
            f"- 字段 `{diff['field']}`:\n"
            f"  - 你的识别结果: {json.dumps(diff['gemini_value'], ensure_ascii=False)}\n"
            f"  - 辅助模型的识别结果: {json.dumps(diff['opus_value'], ensure_ascii=False)}"
            for diff in differences
        ])

        aux_json = json.dumps(opus_data, ensure_ascii=False, indent=2) if opus_data else "{}"

        review_prompt = f"""{original_prompt}

**复审指示（你是主判断方）**

另一个辅助模型也识别了这张图纸，与你的结果存在以下差异：

{diff_text}

辅助模型的完整识别结果（仅供参考）：
```json
{aux_json}
```

**请你重新仔细查看图纸，给出你的最终判断。**

要求：
1. 以图纸上的实际内容为唯一依据，不要因为辅助模型的结果而改变你正确的判断
2. 如果你认为自己原来的识别是正确的，保持原结果
3. 如果查看图纸后发现辅助模型的某个字段确实更准确，可以采纳
4. 你是最终裁决方，请返回你确认后的完整 JSON 结果（格式与之前相同）"""

        return review_prompt

    def _merge_results(
        self,
        gemini_data: Dict[str, Any],
        opus_data: Dict[str, Any],
        differences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        合并两个模型的结果（当无法达成一致时）

        策略：以 Gemini 为主，Opus 仅在 Gemini 缺失字段时补充。
        评测数据显示 Gemini (0.901) 显著优于 Opus (0.720)，
        取平均/取并集/优先 Opus 均会降低最终质量。

        Args:
            gemini_data: Gemini 的输出
            opus_data: Opus 的输出
            differences: 差异列表

        Returns:
            合并后的结果
        """
        logger.info("使用合并策略处理差异（Gemini 优先）")

        # 以 Gemini 为基础
        merged = gemini_data.copy()

        # 仅在 Gemini 缺失该字段时才采用 Opus 的值
        for diff in differences:
            field = diff["field"]
            gemini_value = diff["gemini_value"]
            opus_value = diff["opus_value"]

            if gemini_value is None and opus_value is not None:
                merged[field] = opus_value
                logger.info(f"  字段 '{field}': 使用 Opus 的值（Gemini 缺失）")
            else:
                logger.info(f"  字段 '{field}': 保持 Gemini 的值（Gemini 优先）")

        return merged


__all__ = [
    "ModelResponse",
    "CrossValidationResult",
    "CrossValidator"
]
