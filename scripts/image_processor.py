"""
图纸处理器 - 单张图纸独立处理流程

本模块实现对每张图纸的独立处理流程：
1. 图像预处理（为 OCR 优化）
2. OCR 识别并筛选
3. 第一次 LLM 调用：识别图纸类型（立面图/平面图）
4. 第二次 LLM 调用：根据类型提取信息（带 OCR 结果）
5. 生成 JSON 文件
"""
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import base64
from pathlib import Path
from logger import get_logger
from image_preprocessor import ImagePreprocessor

# 获取日志记录器
logger = get_logger(__name__)


@dataclass
class OCRResult:
    """OCR 识别结果"""
    text: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox
        }


@dataclass
class DrawingTypeResult:
    """图纸类型识别结果"""
    drawing_type: str  # "elevation" 或 "plan"
    confidence: float
    reasoning: str  # LLM 的推理过程

    def to_dict(self) -> Dict:
        return {
            "drawing_type": self.drawing_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class ExtractionResult:
    """信息提取结果"""
    drawing_type: str  # "elevation" 或 "plan"
    floor_id: Optional[str]  # 楼层标识（如 "1F"）或 "立面图"
    data: Dict[str, Any]  # 提取的结构化数据
    ocr_used: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        result = {
            "drawing_type": self.drawing_type,
            "floor_id": self.floor_id,
            "data": self.data,
            "ocr_used": self.ocr_used,
            "metadata": self.metadata
        }
        # 为平面图添加构件连接关系说明
        if self.drawing_type == "plan" and self.floor_id:
            result["structural_note"] = (
                f"本图纸（{self.floor_id} 平面图）中的构件连接关系："
                f"柱和墙从 {self.floor_id} 楼板面向上延伸至上一层楼板面；"
                f"梁位于上一层楼板底面。"
                f"具体的 z 坐标需结合立面图的标高信息确定。"
            )
        return result


class ImageProcessor:
    """
    单张图纸处理器

    处理流程：
    1. OCR 识别
    2. 识别图纸类型
    3. 提取结构信息
    4. 保存 JSON
    """

    def __init__(
        self,
        ocr_client,  # OCR 客户端（由 Skill 提供）
        llm_client,  # LLM 客户端（由 Skill 提供）
        ocr_confidence_threshold: float = 0.85,
        output_dir: str = "./output",
        preprocessing_config: dict = None
    ):
        """
        初始化处理器

        Args:
            ocr_client: OCR 客户端，需要实现 recognize(image_path) 方法
            llm_client: LLM 客户端，需要实现 chat(prompt, image_path) 方法
            ocr_confidence_threshold: OCR 置信度阈值
            output_dir: 输出目录
            preprocessing_config: 图像预处理配置
        """
        self.ocr_client = ocr_client
        self.llm_client = llm_client
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.output_dir = output_dir
        self.preprocessing_config = preprocessing_config or {}

        os.makedirs(output_dir, exist_ok=True)

        # 初始化图像预处理器
        if self.preprocessing_config.get("enabled", False):
            preprocess_output_dir = self.preprocessing_config.get("output_dir", "./preprocessed")
            save_intermediate = self.preprocessing_config.get("save_intermediate_steps", False)
            self.preprocessor = ImagePreprocessor(
                output_dir=preprocess_output_dir,
                save_intermediate=save_intermediate
            )
            logger.info("图像预处理已启用")
        else:
            self.preprocessor = None
            logger.info("图像预处理已禁用")

    def process_image(self, image_path: str) -> ExtractionResult:
        """
        处理单张图纸

        Args:
            image_path: 图片路径

        Returns:
            ExtractionResult: 提取结果
        """
        logger.info("="*60)
        logger.info(f"处理图纸: {os.path.basename(image_path)}")
        logger.info("="*60)

        # 步骤 0: 图像预处理（如果启用）
        ocr_image_path = image_path  # 默认使用原图
        if self.preprocessor and self.preprocessing_config.get("enabled", False):
            logger.info("[步骤 0/3] 图像预处理...")
            try:
                methods = self.preprocessing_config.get("methods")
                ocr_image_path = self.preprocessor.preprocess_for_ocr(image_path, methods=methods)
                logger.info(f"预处理图片: {os.path.basename(ocr_image_path)}")
            except Exception as e:
                logger.warning(f"图像预处理失败，使用原图: {e}")
                ocr_image_path = image_path

        # 步骤 1: OCR 识别（使用预处理图）
        logger.info("[步骤 1/3] OCR 识别...")
        ocr_results = self._run_ocr(ocr_image_path)
        logger.info(f"识别到 {len(ocr_results)} 个文本区域")

        # 步骤 2: 识别图纸类型（使用原图）
        logger.info("[步骤 2/3] 识别图纸类型...")
        drawing_type_result = self._identify_drawing_type(image_path, ocr_results)
        logger.info(f"图纸类型: {drawing_type_result.drawing_type}")
        logger.info(f"置信度: {drawing_type_result.confidence:.2f}")

        # 步骤 3: 提取结构信息（使用原图）
        logger.info("[步骤 3/3] 提取结构信息...")
        extraction_data = self._extract_information(
            image_path,
            drawing_type_result.drawing_type,
            ocr_results
        )

        # 构建结果
        result = ExtractionResult(
            drawing_type=drawing_type_result.drawing_type,
            floor_id=extraction_data.get("floor_id"),
            data=extraction_data,
            ocr_used=len(ocr_results) > 0,
            metadata={
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "ocr_text_count": len(ocr_results),
                "type_confidence": drawing_type_result.confidence,
                "type_reasoning": drawing_type_result.reasoning
            }
        )

        # 保存 JSON
        self._save_result(result, image_path)

        logger.info(f"处理完成: {os.path.basename(image_path)}")
        return result

    def _run_ocr(self, image_path: str) -> List[OCRResult]:
        """
        运行 OCR 识别并筛选结果

        Args:
            image_path: 图片路径

        Returns:
            筛选后的 OCR 结果列表
        """
        try:
            # 调用 OCR 客户端
            raw_results = self.ocr_client.recognize(image_path)

            # 筛选高置信度结果
            filtered_results = []
            for item in raw_results:
                if item.get("confidence", 0) >= self.ocr_confidence_threshold:
                    filtered_results.append(OCRResult(
                        text=item["text"],
                        confidence=item["confidence"],
                        bbox=item.get("bbox", [0, 0, 0, 0])
                    ))

            return filtered_results

        except Exception as e:
            logger.warning(f"OCR 识别失败: {e}")
            return []

    def _identify_drawing_type(
        self,
        image_path: str,
        ocr_results: List[OCRResult]
    ) -> DrawingTypeResult:
        """
        第一次 LLM 调用：识别图纸类型

        Args:
            image_path: 图片路径
            ocr_results: OCR 识别结果

        Returns:
            DrawingTypeResult: 图纸类型识别结果
        """
        # 构建 OCR 文本摘要
        ocr_text_summary = self._build_ocr_summary(ocr_results)

        # 构建 Prompt
        prompt = self._build_type_identification_prompt(ocr_text_summary)

        # 调用 LLM
        try:
            response = self.llm_client.chat(prompt, image_path=image_path)
            result_dict = self._parse_llm_response(response)

            return DrawingTypeResult(
                drawing_type=result_dict.get("drawing_type", "unknown"),
                confidence=result_dict.get("confidence", 0.0),
                reasoning=result_dict.get("reasoning", "")
            )

        except Exception as e:
            logger.warning(f"图纸类型识别失败: {e}")
            # 返回默认值
            return DrawingTypeResult(
                drawing_type="unknown",
                confidence=0.0,
                reasoning=f"识别失败: {str(e)}"
            )

    def _extract_information(
        self,
        image_path: str,
        drawing_type: str,
        ocr_results: List[OCRResult]
    ) -> Dict[str, Any]:
        """
        第二次 LLM 调用：根据图纸类型提取信息（支持重试）

        Args:
            image_path: 图片路径
            drawing_type: 图纸类型
            ocr_results: OCR 识别结果

        Returns:
            提取的结构化数据
        """
        # 构建 OCR 文本摘要
        ocr_text_summary = self._build_ocr_summary(ocr_results)

        # 根据图纸类型选择 Prompt
        if drawing_type == "elevation":
            prompt = self._build_elevation_extraction_prompt(ocr_text_summary)
        elif drawing_type == "plan":
            prompt = self._build_plan_extraction_prompt(ocr_text_summary)
        else:
            raise ValueError(f"未知的图纸类型: {drawing_type}")

        # 尝试调用 LLM，最多重试 3 次
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"LLM 调用尝试 {attempt + 1}/{max_retries}")
                response = self.llm_client.chat(prompt, image_path=image_path)

                # 尝试解析响应
                extraction_data = self._parse_llm_response(response)

                # 验证必需字段
                if drawing_type == "elevation":
                    if "floor_levels" not in extraction_data:
                        raise ValueError("立面图提取结果缺少 floor_levels 字段")
                elif drawing_type == "plan":
                    if "components_above" not in extraction_data:
                        raise ValueError("平面图提取结果缺少 components_above 字段")
                    # 后处理：将梁的轴线交点标签推算为数值坐标
                    extraction_data = self._resolve_beam_coordinates(extraction_data)

                logger.info(f"LLM 调用成功（尝试 {attempt + 1}）")
                return extraction_data

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(f"LLM 响应解析失败（尝试 {attempt + 1}/{max_retries}）: {e}")

                if attempt < max_retries - 1:
                    # 如果不是最后一次尝试，修改 prompt 要求更严格的 JSON 格式
                    prompt = self._build_retry_prompt(prompt, str(e))
                    logger.info("使用更严格的 prompt 重试...")
                else:
                    logger.error(f"LLM 调用失败，已达到最大重试次数")

            except Exception as e:
                last_error = e
                logger.error(f"LLM 调用异常（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt == max_retries - 1:
                    break

        # 所有重试都失败，返回错误信息
        return {"error": str(last_error)}

    def _build_retry_prompt(self, original_prompt: str, error_msg: str) -> str:
        """
        构建重试 Prompt，要求更严格的 JSON 格式

        Args:
            original_prompt: 原始 Prompt
            error_msg: 错误信息

        Returns:
            重试 Prompt
        """
        retry_instruction = f"""

**重要提示：上一次响应的 JSON 格式有误（{error_msg}），请务必：**
1. 只返回纯 JSON，不要包含任何 markdown 代码块标记（```json 或 ```）
2. 确保所有字段之间都有逗号分隔
3. 确保所有字符串都用双引号包裹
4. 确保最后一个字段后面没有多余的逗号
5. 确保所有括号都正确配对
6. 不要在 JSON 中添加任何注释

请严格按照 JSON 标准格式返回，直接以 {{ 开头，以 }} 结尾。"""

        return original_prompt + retry_instruction

    def _build_ocr_summary(self, ocr_results: List[OCRResult]) -> str:
        """
        构建 OCR 文本摘要

        Args:
            ocr_results: OCR 识别结果

        Returns:
            文本摘要字符串
        """
        if not ocr_results:
            return "（无 OCR 识别结果）"

        # 按置信度排序
        sorted_results = sorted(ocr_results, key=lambda x: x.confidence, reverse=True)

        # 构建摘要（最多取前 50 个）
        summary_lines = []
        for i, result in enumerate(sorted_results[:50]):
            summary_lines.append(
                f"  [{i+1}] {result.text} (置信度: {result.confidence:.2f})"
            )

        return "\n".join(summary_lines)

    def _build_type_identification_prompt(self, ocr_summary: str) -> str:
        """
        构建图纸类型识别 Prompt

        Args:
            ocr_summary: OCR 文本摘要

        Returns:
            Prompt 字符串
        """
        return f"""你是一名结构工程专家，请分析这张建筑结构图纸，判断它是**立面图**还是**平面图**。

**判断依据：**

**立面图（elevation）的特征：**
- 显示建筑的**侧面/剖面**视图
- 包含标高符号（▽）和标高数值
- 可以看到水平楼层线（代表楼板）
- 显示竖向构件（柱、墙）的侧面
- 通常标题中含有"立面"、"剖面"等字样

**平面图（plan）的特征：**
- 显示建筑某一楼层的**俯视图**
- 包含轴网：数字轴（1、2、3...）和字母轴（A、B、C...）
- 显示柱（填充矩形/圆形）、梁（双线）、墙（粗线）的平面布置
- 通常标题中含有楼层名称（如"1F"、"二层"等）和"平面图"字样

**OCR 识别的文本信息：**
{ocr_summary}

**请返回 JSON 格式：**
{{
    "drawing_type": "elevation" 或 "plan",
    "confidence": 0.0-1.0 之间的置信度,
    "reasoning": "你的判断理由（简短说明）"
}}

注意：只返回 JSON，不要有其他内容。"""

    def _build_elevation_extraction_prompt(self, ocr_summary: str) -> str:
        """
        构建立面图信息提取 Prompt

        Args:
            ocr_summary: OCR 文本摘要

        Returns:
            Prompt 字符串
        """
        return f"""你是一名结构工程专家，请从这张建筑立面图中提取楼层标高信息。

**结构工程背景知识：**
立面图展示建筑的侧面剖面，其中：
- 每条水平楼层线代表一块**楼板（floor slab）**的位置
- 标高符号（▽）标注的是该楼板面的绝对高度
- 相邻两块楼板之间的空间就是一个"层"，柱、墙等竖向构件就在这个空间内
- 例如：1F 楼板标高 0mm，2F 楼板标高 3600mm，则 1F 的层高（floor_height）= 3600mm
  这意味着 1F 平面图中的柱从 0mm 延伸到 3600mm

**提取内容：**
1. 识别所有标高符号（▽）及其旁边的数值（单位：mm）
2. 识别对应的楼层名称（如 1F、2F、3F、RF 等）
3. 计算相邻楼层之间的层高（floor_height）
4. 按从下到上的顺序排列

**OCR 识别的文本信息（可作为参考）：**
{ocr_summary}

**请返回 JSON 格式：**
{{
    "floor_id": "立面图",
    "floor_levels": [
        {{
            "floor": "1F",
            "elevation": 0.0,
            "floor_height": 3600.0,
            "description": "一层楼板面标高"
        }},
        {{
            "floor": "2F",
            "elevation": 3600.0,
            "floor_height": 3600.0,
            "description": "二层楼板面标高"
        }},
        {{
            "floor": "3F",
            "elevation": 7200.0,
            "floor_height": 3200.0,
            "description": "三层楼板面标高"
        }},
        {{
            "floor": "RF",
            "elevation": 10400.0,
            "floor_height": null,
            "description": "屋面标高"
        }}
    ],
    "total_height": 10400.0,
    "floor_count": 3,
    "notes": "其他备注信息（可选）"
}}

**字段说明：**
- floor: 楼层名称，使用标准格式（1F, 2F, ..., RF）
- elevation: 楼板面标高（单位：mm），即标高符号旁的数值
- floor_height: 该层的层高（单位：mm），等于上一层楼板标高减去本层楼板标高。最顶层（如 RF）的 floor_height 为 null
- total_height: 建筑总高度（最高标高，单位：mm）
- floor_count: 楼层数量（不含屋面层）

**注意事项：**
1. 标高单位统一为毫米（mm），如图纸标注的是米（m），请转换为 mm
2. 按从下到上的顺序排列
3. floor_height = 上一层 elevation - 本层 elevation
4. **【标高基准】**：图纸上的 ±0.000 即为一层楼板面（elevation = 0）；
   地下室楼板标高为负值（如 -3600）；
   如图纸标注的是绝对标高（如 ±0.000 = 某个固定高度），
   请将所有标高**转换为相对于±0.000 的相对值**再输出
5. **【自查步骤】**：输出前检查 floor_height 的合理性：常见层高范围 2700mm ~ 5000mm；
   如有楼层的 floor_height 明显偏小（<1000mm）或偏大（>10000mm），请重新核对图纸标注
6. 只返回 JSON，不要有其他内容"""

    def _build_grid_extraction_prompt(self, ocr_summary: str) -> str:
        """
        构建平面图轴网专项提取 Prompt（第一阶段）

        只提取轴网信息，作为第二阶段构件识别的基础上下文。

        Args:
            ocr_summary: OCR 文本摘要

        Returns:
            Prompt 字符串
        """
        return f"""你是一名结构工程专家，请从这张结构平面图中**只提取轴网信息**。

**轴网说明：**
- 数字轴（X 方向）：轴线编号为 1、2、3...，从左到右坐标递增
- 字母轴（Y 方向）：轴线编号为 A、B、C...，从下到上坐标递增
- 坐标通过图纸上的轴间距标注（如 6000、3000 等数字）累加推算
- 原点：编号最小的数字轴与编号最小的字母轴的交点，坐标为 (0, 0)，单位：毫米（mm）

**【重要】双轴网（附加轴）概念：**
复杂平面图中常出现**两套轴网并存**的情况：
- 主轴网：整数编号，如 1、2、3...（X 方向）和 A、B、C...（Y 方向）
- 附加轴（副轴）：在两条主轴之间插入的附加轴线，编号规则为"前一主轴编号 + 分隔符 + 序号"，
  例如 X 方向的 1/1、1/2（位于 1 轴和 2 轴之间）；Y 方向的 A/1、B/1（位于相邻字母轴之间）
- 附加轴坐标 = 前一主轴坐标 + 该段标注的附加间距
- 提取时**必须包含所有主轴和附加轴**，不能遗漏
- label 字段严格按图纸标注填写（如 "1/1"、"A/1"），不要自行改名

**轴线顺序与方向规则（必须遵守）：**
1. 数字轴：按图纸从左到右的方向，编号递增排列（1, 2, 3, ... 或包含附加轴 1, 1/1, 2, 2/1, 3...）
2. 字母轴：按图纸从下到上的方向，编号递增排列（A, B, C, ... 或包含附加轴）
3. 坐标通过**累加相邻轴间距**得到，不得倒序

**OCR 识别的文本信息（可作为参考）：**
{ocr_summary}

**请返回 JSON 格式：**
{{
    "x_axes": [
        {{"label": "1",   "coordinate": 0}},
        {{"label": "1/1", "coordinate": 3000}},
        {{"label": "2",   "coordinate": 6000}},
        {{"label": "3",   "coordinate": 12000}}
    ],
    "y_axes": [
        {{"label": "A",   "coordinate": 0}},
        {{"label": "B",   "coordinate": 4000}},
        {{"label": "B/1", "coordinate": 6000}},
        {{"label": "C",   "coordinate": 8000}}
    ]
}}

**注意事项：**
1. 只返回轴网信息，不要提取其他构件
2. 坐标单位为毫米（mm），如图纸标注为米请转换
3. 按轴线在图纸上从左到右（X）、从下到上（Y）的顺序排列，坐标严格递增
4. 图纸上每一条带标注的轴线都必须包含，附加轴不得遗漏
5. 只返回 JSON，不要有其他内容"""

    def _build_plan_extraction_prompt(self, ocr_summary: str, grid_info: Dict[str, Any] = None) -> str:
        """
        构建平面图信息提取 Prompt（第二阶段）

        Args:
            ocr_summary: OCR 文本摘要
            grid_info: 已知的轴网信息（由第一阶段提取，可为 None）

        Returns:
            Prompt 字符串
        """
        # 构建轴网上下文块
        if grid_info:
            grid_context = f"""
**【已知轴网信息】（第一阶段提取结果，请直接使用，无需重新识别）：**
```json
{json.dumps(grid_info, ensure_ascii=False, indent=2)}
```
请严格基于以上轴网坐标来定位所有构件，在输出的 grid_info 字段中直接复用此轴网数据。
"""
        else:
            grid_context = ""

        return f"""你是一名结构工程专家，请从这张结构平面图中提取构件信息。
{grid_context}
**结构工程背景知识（非常重要，请仔细理解）：**

结构平面图（如"1F 结构平面图"）是从上往下的俯视图，它展示的是**从该楼层楼板向上延伸**的所有结构构件。具体来说：

1. **柱（Column）**：
   - 平面图上的柱表示的是**竖向构件**，从**当前楼板面**向上延伸到**上一层楼板面**
   - 例如：1F 平面图中的柱，是从 1F 楼板延伸到 2F 楼板的柱
   - 柱在平面图上显示为填充的矩形或圆形
   - **柱的定位策略（按优先级）：**
     1. 绝大多数柱位于轴线交点处：**必须逐行逐列扫描每一个轴线交点**（x轴 × y轴的笛卡尔积），
        逐一判断该交点处是否存在柱，若有则以该交点坐标作为柱的坐标，并在 grid_location 中记录交点标签（如 "A-1"）
     2. 少数柱不落在轴线交点（如悬挑柱、错位柱）：直接从图纸尺寸标注推算其世界坐标，grid_location 填 null
   - **扫描要求**：若图纸有 m 条 X 轴和 n 条 Y 轴，则最多有 m×n 个交点，请逐一检查，不得跳过
   - 需要提取：柱的平面位置（x, y）、截面尺寸（如 400x400、500x500）、柱编号

2. **梁（Beam）**：
   - 平面图上的梁是**水平构件**，位于**上一层楼板的底部**
   - 例如：1F 平面图中的梁，位于 2F 楼板底面
   - 梁在平面图上显示为两条平行线之间的区域
   - 梁的两端通常搭在柱子或墙上，因此**梁的起止点由两端所在的轴线交点决定**
   - 需要提取：梁两端连接的轴线交点标签（如 "A-1"、"A-2"）、截面标注（如 KL1 250x500）、梁编号

3. **剪力墙（Shear Wall）**：
   - 剪力墙是**竖向面状构件**，从**当前楼板面**向上延伸到**上一层楼板面**
   - 例如：1F 平面图中的墙，从 1F 楼板延伸到 2F 楼板
   - 需要提取：墙的起终点坐标、墙厚、墙编号

4. **楼板（Slab）**：
   - 如果图纸上标注了楼板信息，提取楼板的范围和厚度

**总结：N层平面图中的构件连接关系：**
- 柱和墙：底部在 N 层楼板面，顶部在 N+1 层楼板面
- 梁：位于 N+1 层楼板底面
- 构件的实际高度（z方向）需要结合立面图的标高信息来确定

**OCR 识别的文本信息（可作为参考）：**
{ocr_summary}

**坐标系定义：**
- 原点：1 轴与 A 轴的交点
- X 轴方向：数字轴增大方向（1→2→3...）
- Y 轴方向：字母轴增大方向（A→B→C...）
- 单位：毫米（mm）
- 坐标必须是基于轴网间距计算的世界坐标，不是图纸上的像素坐标

**请返回 JSON 格式：**
{{
    "floor_id": "1F",
    "components_above": {{
        "columns": [
            {{
                "x": 0,
                "y": 0,
                "label": "KZ1",
                "grid_location": "A-1",
                "section": "400x400"
            }},
            {{
                "x": 6000,
                "y": 0,
                "label": "KZ1",
                "grid_location": "A-2",
                "section": "400x400"
            }}
        ],
        "beams": [
            {{
                "start_grid": "A-1",
                "end_grid": "A-2",
                "label": "KL1",
                "section": "250x500"
            }},
            {{
                "start_grid": "A-1",
                "end_grid": "B-1",
                "label": "KL2",
                "section": "200x400"
            }}
        ],
        "walls": [
            {{
                "start": [0, 0],
                "end": [0, 6000],
                "thickness": 200,
                "label": "Q1"
            }}
        ],
        "slabs": [
            {{
                "boundary": [[0, 0], [6000, 0], [6000, 6000], [0, 6000]],
                "thickness": 120,
                "label": "板1"
            }}
        ]
    }},
    "grid_info": {{
        "x_axes": [
            {{"label": "1", "coordinate": 0}},
            {{"label": "2", "coordinate": 6000}},
            {{"label": "3", "coordinate": 12000}}
        ],
        "y_axes": [
            {{"label": "A", "coordinate": 0}},
            {{"label": "B", "coordinate": 6000}},
            {{"label": "C", "coordinate": 9000}}
        ]
    }},
    "connection_note": "本层构件连接关系：柱和墙从 1F 楼板延伸至 2F 楼板，梁位于 2F 楼板底面",
    "notes": "其他备注信息（可选）"
}}

**字段说明：**
- floor_id: 楼层名称（如 1F, 2F），从图纸标题中识别
- components_above: 从该楼层向上延伸的构件
  - columns: 柱列表
    - x, y: 柱中心点的世界坐标（mm）
    - label: 柱编号（如 KZ1、KZ2）
    - grid_location: 轴网位置（如 A-1 表示 A 轴和 1 轴交点）
    - section: 截面尺寸（如 "400x400"，单位 mm，如图纸未标注则为 null）
  - beams: 梁列表
    - start_grid: 梁起点所在的轴线交点标签，格式为 "字母轴-数字轴"（如 "A-1"）
    - end_grid: 梁终点所在的轴线交点标签，格式为 "字母轴-数字轴"（如 "A-2"）
    - label: 梁编号（如 KL1、LL1）
    - section: 截面尺寸（如 "250x500"，宽x高，单位 mm，如图纸未标注则为 null）
    - 注意：start/end 数值坐标将由后处理程序根据 grid_info 自动推算，无需 LLM 输出
  - walls: 剪力墙列表
    - start, end: 墙中心线的起终点世界坐标（mm）
    - thickness: 墙厚（mm）
    - label: 墙编号
  - slabs: 楼板列表（如果图纸上有标注）
    - boundary: 楼板边界坐标点列表
    - thickness: 板厚（mm）
    - label: 板编号
- grid_info: 轴网信息
  - x_axes, y_axes: 各轴线的编号和对应的世界坐标（mm）
- connection_note: 构件连接关系说明

**注意事项：**
1. 坐标必须是基于轴网间距的世界坐标（mm），不是像素坐标
2. 通过识别轴网间距标注（如 6000、3000 等数值）来确定世界坐标
3. 柱的坐标是柱截面中心点
4. 墙的 start/end 坐标是其中心线轴线位置
5. **梁只需输出 start_grid 和 end_grid（轴线交点标签），不要输出数值坐标**，数值坐标由后处理自动推算
6. 如果截面尺寸在图纸上未标注，设为 null
7. 如果没有楼板信息，slabs 可以为空列表 []
8. **【自查步骤，输出前必须执行】**
   - 统计 grid_info 中 x_axes 数量（记为 m）和 y_axes 数量（记为 n），最多有 m×n 个轴网交叉点
   - 检查图纸上每个交叉点位置是否有柱，逐行逐列确认
   - 若你提取的柱数量远少于图纸上可见的柱总数，请补充遗漏的柱，再输出
   - 确认无遗漏后方可输出 JSON
9. 只返回 JSON，不要有其他内容"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON 响应，支持多种修复策略

        Args:
            response: LLM 响应字符串

        Returns:
            解析后的字典
        """
        import re

        # 策略 1: 尝试直接解析
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
            json_str = json_match.group(0)

            # 策略 3.1: 直接解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # 策略 3.2: 尝试修复常见的 JSON 格式错误
                try:
                    fixed_json = self._fix_json_errors(json_str, e)
                    return json.loads(fixed_json)
                except:
                    pass

        raise ValueError(f"无法解析 LLM 响应为 JSON: {response[:200]}")

    def _fix_json_errors(self, json_str: str, error: json.JSONDecodeError) -> str:
        """
        尝试修复常见的 JSON 格式错误

        Args:
            json_str: JSON 字符串
            error: JSON 解析错误

        Returns:
            修复后的 JSON 字符串
        """
        import re

        # 修复 1: 移除尾部逗号（trailing comma）
        # 例如: {"a": 1,} -> {"a": 1}
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # 修复 2: 修复缺失的逗号（在对象/数组元素之间）
        # 这个比较复杂，只处理简单情况
        # 例如: {"a": 1 "b": 2} -> {"a": 1, "b": 2}
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
        json_str = re.sub(r']\s*\n\s*\[', '],\n[', json_str)

        # 修复 3: 修复单引号（应该用双引号）
        # 注意：这个可能会误伤字符串内容中的单引号
        # json_str = json_str.replace("'", '"')

        # 修复 4: 移除注释（JSON 不支持注释）
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str

    def _resolve_beam_coordinates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理：将梁的轴线交点标签（如 "A-1"）转换为世界坐标。

        LLM 输出 start_grid / end_grid，本函数根据 grid_info 推算
        出对应的 start / end 数值坐标（mm），并将两者都保留在结果中。

        Args:
            data: LLM 解析后的平面图数据字典

        Returns:
            补充了 start / end 坐标的数据字典
        """
        grid_info = data.get("grid_info", {})
        x_axes = {ax["label"]: ax["coordinate"] for ax in grid_info.get("x_axes", [])}
        y_axes = {ax["label"]: ax["coordinate"] for ax in grid_info.get("y_axes", [])}

        if not x_axes or not y_axes:
            logger.warning("grid_info 为空，无法推算梁坐标，跳过后处理")
            return data

        def grid_label_to_coord(label: str):
            """将 'A-1' 解析为 (x, y)，字母轴→Y，数字轴→X"""
            if not label or "-" not in label:
                return None
            parts = label.split("-", 1)
            y_label, x_label = parts[0].strip(), parts[1].strip()
            x = x_axes.get(x_label)
            y = y_axes.get(y_label)
            if x is None or y is None:
                logger.warning(f"轴线标签 '{label}' 在 grid_info 中找不到对应坐标")
                return None
            return [x, y]

        beams = data.get("components_above", {}).get("beams", [])
        resolved = 0
        for beam in beams:
            start_grid = beam.get("start_grid")
            end_grid = beam.get("end_grid")
            if start_grid and end_grid:
                start_coord = grid_label_to_coord(start_grid)
                end_coord = grid_label_to_coord(end_grid)
                if start_coord is not None:
                    beam["start"] = start_coord
                if end_coord is not None:
                    beam["end"] = end_coord
                if start_coord is not None and end_coord is not None:
                    resolved += 1

        if beams:
            logger.info(f"梁坐标后处理：{resolved}/{len(beams)} 条梁已推算坐标")

        return data

    def _save_result(self, result: ExtractionResult, image_path: str):
        """
        保存提取结果为 JSON 文件

        Args:
            result: 提取结果
            image_path: 原始图片路径
        """
        # 生成输出文件名
        image_name = Path(image_path).stem
        output_filename = f"{image_name}_extraction.json"
        output_path = os.path.join(self.output_dir, output_filename)

        # 保存 JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"保存结果: {output_filename}")


class BatchImageProcessor:
    """
    批量图纸处理器

    按顺序处理多张图纸，每张图纸独立处理
    """

    def __init__(
        self,
        ocr_client,
        llm_client,
        ocr_confidence_threshold: float = 0.85,
        output_dir: str = "./output",
        preprocessing_config: dict = None
    ):
        """
        初始化批量处理器

        Args:
            ocr_client: OCR 客户端
            llm_client: LLM 客户端
            ocr_confidence_threshold: OCR 置信度阈值
            output_dir: 输出目录
            preprocessing_config: 图像预处理配置
        """
        self.processor = ImageProcessor(
            ocr_client=ocr_client,
            llm_client=llm_client,
            ocr_confidence_threshold=ocr_confidence_threshold,
            output_dir=output_dir,
            preprocessing_config=preprocessing_config
        )

    def process_images(self, image_paths: List[str]) -> List[ExtractionResult]:
        """
        批量处理图纸

        Args:
            image_paths: 图片路径列表

        Returns:
            提取结果列表
        """
        logger.info("="*60)
        logger.info(f"开始批量处理 {len(image_paths)} 张图纸")
        logger.info("="*60)

        results = []
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n进度: {i}/{len(image_paths)}")

            try:
                result = self.processor.process_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"处理失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        logger.info("="*60)
        logger.info("批量处理完成！")
        logger.info(f"成功: {len(results)}/{len(image_paths)}")
        logger.info(f"输出目录: {self.processor.output_dir}")
        logger.info("="*60)

        return results


# ============================================================================
# 工具函数
# ============================================================================

def load_config(config_path: str = "../config.json") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_processor_from_config(
    ocr_client,
    llm_client,
    config_path: str = "../config.json"
) -> ImageProcessor:
    """
    从配置文件创建处理器

    Args:
        ocr_client: OCR 客户端
        llm_client: LLM 客户端
        config_path: 配置文件路径

    Returns:
        ImageProcessor 实例
    """
    config = load_config(config_path)

    return ImageProcessor(
        ocr_client=ocr_client,
        llm_client=llm_client,
        ocr_confidence_threshold=config["ocr"]["confidence_threshold"],
        output_dir="./output"
    )


__all__ = [
    "OCRResult",
    "DrawingTypeResult",
    "ExtractionResult",
    "ImageProcessor",
    "BatchImageProcessor",
    "load_config",
    "create_processor_from_config"
]
