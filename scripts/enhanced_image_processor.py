"""
增强的图像处理器 - 支持交叉验证和上下文管理

在原有的 ImageProcessor 基础上，添加：
1. 双模型交叉验证
2. 上下文管理
"""
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from logger import get_logger
from image_processor import (
    OCRResult,
    DrawingTypeResult,
    ExtractionResult,
    ImageProcessor
)
from cross_validation import CrossValidator, CrossValidationResult
from context_manager import ConversationContext, ContextAwareLLMClient
from client_interfaces import create_gemini_client, create_opus_client, create_qwen_client

logger = get_logger(__name__)


class EnhancedImageProcessor(ImageProcessor):
    """
    增强的图像处理器

    在原有功能基础上添加：
    1. 双模型交叉验证（Gemini + Opus）
    2. 上下文管理
    """

    def __init__(
        self,
        ocr_client,
        llm_client,  # 主 LLM 客户端（Gemini）
        ocr_confidence_threshold: float = 0.85,
        output_dir: str = "./output",
        preprocessing_config: dict = None,
        cross_validation_enabled: bool = True,
        context_enabled: bool = True,
        context_dir: str = "./context",
        session_id: Optional[str] = None
    ):
        """
        初始化增强处理器

        Args:
            ocr_client: OCR 客户端
            llm_client: 主 LLM 客户端（Gemini）
            ocr_confidence_threshold: OCR 置信度阈值
            output_dir: 输出目录
            preprocessing_config: 图像预处理配置
            cross_validation_enabled: 是否启用交叉验证
            context_enabled: 是否启用上下文管理
            context_dir: 上下文存储目录
            session_id: 会话 ID
        """
        # 调用父类初始化
        super().__init__(
            ocr_client=ocr_client,
            llm_client=llm_client,
            ocr_confidence_threshold=ocr_confidence_threshold,
            output_dir=output_dir,
            preprocessing_config=preprocessing_config
        )

        self.cross_validation_enabled = cross_validation_enabled
        self.context_enabled = context_enabled

        # 初始化上下文管理器
        if self.context_enabled:
            self.context = ConversationContext(
                session_id=session_id,
                context_dir=context_dir,
                auto_save=True
            )
            logger.info(f"上下文管理已启用: session_id={self.context.session_id}")

            # 包装 LLM 客户端为上下文感知客户端
            self.gemini_client = ContextAwareLLMClient(
                llm_client=llm_client,
                context=self.context,
                model_name="Gemini 3.1 Pro"
            )
        else:
            self.context = None
            self.gemini_client = llm_client

        # 初始化交叉验证器
        if self.cross_validation_enabled:
            try:
                # 创建 Qwen3.5 辅助模型客户端（替换 Opus，评测数据显示 Qwen 0.838 > Opus 0.720）
                aux_client = create_qwen_client(
                    max_tokens=32768,
                    temperature=0.1,
                    reasoning_enabled=True
                )

                # Qwen 辅助模型不注入跨图纸历史上下文：
                # 1. Qwen 只需看当前图纸，不需要前序图纸的对话记录
                # 2. 避免累积历史导致请求体超过 Qwen 262144 token 上限
                self.opus_client = aux_client

                # 创建交叉验证器
                self.cross_validator = CrossValidator(
                    gemini_client=self.gemini_client,
                    opus_client=self.opus_client,
                    max_validation_rounds=3
                )
                logger.info("交叉验证已启用: Gemini 3.1 Pro (主) + Qwen3.5-397B (辅)")
            except Exception as e:
                logger.error(f"交叉验证初始化失败: {e}")
                self.cross_validation_enabled = False
                self.cross_validator = None
        else:
            self.cross_validator = None
            self.opus_client = None

    def process_image(self, image_path: str) -> ExtractionResult:
        """
        处理单张图纸（增强版本）

        Args:
            image_path: 图片路径

        Returns:
            ExtractionResult: 提取结果
        """
        logger.info("="*60)
        logger.info(f"处理图纸: {os.path.basename(image_path)}")
        logger.info("="*60)

        # 添加系统消息到上下文
        if self.context:
            self.context.add_system_message(f"开始处理图纸: {os.path.basename(image_path)}")

        # 步骤 0: 图像预处理（如果启用）
        ocr_image_path = image_path
        if self.preprocessor and self.preprocessing_config.get("enabled", False):
            logger.info("[步骤 0/4] 图像预处理...")
            try:
                methods = self.preprocessing_config.get("methods")
                ocr_image_path = self.preprocessor.preprocess_for_ocr(image_path, methods=methods)
                logger.info(f"预处理图片: {os.path.basename(ocr_image_path)}")
            except Exception as e:
                logger.warning(f"图像预处理失败，使用原图: {e}")
                ocr_image_path = image_path

        # 步骤 1: OCR 识别（使用预处理图）
        logger.info("[步骤 1/4] OCR 识别...")
        ocr_results = self._run_ocr(ocr_image_path)
        logger.info(f"识别到 {len(ocr_results)} 个文本区域")

        # 步骤 2: 识别图纸类型
        logger.info("[步骤 2/4] 识别图纸类型...")
        if self.cross_validation_enabled and self.cross_validator:
            drawing_type_result = self._identify_drawing_type_with_validation(image_path, ocr_results)
        else:
            drawing_type_result = self._identify_drawing_type(image_path, ocr_results)

        logger.info(f"图纸类型: {drawing_type_result.drawing_type}")
        logger.info(f"置信度: {drawing_type_result.confidence:.2f}")

        # 步骤 3: 提取结构信息（平面图：阶段1轴网+阶段2构件；立面图：单阶段）
        is_plan = drawing_type_result.drawing_type == "plan"
        step_label = "[步骤 3/4] 提取结构信息（两阶段：轴网→构件）..." if is_plan else "[步骤 3/4] 提取结构信息..."
        logger.info(step_label)
        if self.cross_validation_enabled and self.cross_validator:
            extraction_data, validation_result = self._extract_information_with_validation(
                image_path,
                drawing_type_result.drawing_type,
                ocr_results
            )
        else:
            extraction_data = self._extract_information(
                image_path,
                drawing_type_result.drawing_type,
                ocr_results
            )
            validation_result = None

        # 步骤 4: 构建结果
        logger.info("[步骤 4/4] 构建结果...")
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
                "type_reasoning": drawing_type_result.reasoning,
                "cross_validation_enabled": self.cross_validation_enabled,
                "context_enabled": self.context_enabled
            }
        )

        # 如果有交叉验证结果，添加到元数据
        if validation_result:
            result.metadata["cross_validation"] = {
                "consensus_reached": validation_result.consensus_reached,
                "validation_rounds": validation_result.validation_rounds,
                "has_differences": validation_result.has_differences,
                "differences_count": len(validation_result.differences)
            }

        # 保存 JSON
        self._save_result(result, image_path)

        # 添加完成消息到上下文
        if self.context:
            self.context.add_system_message(f"完成处理图纸: {os.path.basename(image_path)}")

        logger.info(f"处理完成: {os.path.basename(image_path)}")
        return result

    def _identify_drawing_type_with_validation(
        self,
        image_path: str,
        ocr_results: List[OCRResult]
    ) -> DrawingTypeResult:
        """
        使用交叉验证识别图纸类型

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

        # 使用交叉验证
        validation_result = self.cross_validator.validate(
            prompt=prompt,
            image_path=image_path,
            expected_fields=["drawing_type", "confidence", "reasoning"]
        )

        # 使用最终结果
        final_data = validation_result.final_data

        return DrawingTypeResult(
            drawing_type=final_data.get("drawing_type", "unknown"),
            confidence=final_data.get("confidence", 0.0),
            reasoning=final_data.get("reasoning", "")
        )

    def _extract_grid_with_validation(
        self,
        image_path: str,
        ocr_results: List[OCRResult]
    ) -> Dict[str, Any]:
        """
        第一阶段：使用交叉验证专项提取平面图轴网信息

        Args:
            image_path: 图片路径
            ocr_results: OCR 识别结果

        Returns:
            轴网数据字典 {"x_axes": [...], "y_axes": [...]}
        """
        ocr_text_summary = self._build_ocr_summary(ocr_results)
        prompt = self._build_grid_extraction_prompt(ocr_text_summary)

        logger.info("[阶段 1/2] 交叉验证提取轴网信息...")
        validation_result = self.cross_validator.validate(
            prompt=prompt,
            image_path=image_path,
            expected_fields=["x_axes", "y_axes"]
        )

        grid_info = validation_result.final_data
        x_count = len(grid_info.get("x_axes", []))
        y_count = len(grid_info.get("y_axes", []))
        logger.info(f"轴网提取完成：X 轴 {x_count} 条，Y 轴 {y_count} 条")
        return grid_info

    def _extract_information_with_validation(
        self,
        image_path: str,
        drawing_type: str,
        ocr_results: List[OCRResult]
    ) -> tuple[Dict[str, Any], CrossValidationResult]:
        """
        使用交叉验证提取结构信息。

        平面图采用两阶段流程：
          阶段 1 — 交叉验证提取轴网信息
          阶段 2 — 将轴网注入 Prompt，交叉验证提取构件信息

        立面图仍为单阶段流程。

        Args:
            image_path: 图片路径
            drawing_type: 图纸类型
            ocr_results: OCR 识别结果

        Returns:
            (提取的结构化数据, 交叉验证结果)
        """
        ocr_text_summary = self._build_ocr_summary(ocr_results)

        if drawing_type == "elevation":
            prompt = self._build_elevation_extraction_prompt(ocr_text_summary)
            expected_fields = ["floor_levels", "total_height", "floor_count"]

            validation_result = self.cross_validator.validate(
                prompt=prompt,
                image_path=image_path,
                expected_fields=expected_fields
            )
            return validation_result.final_data, validation_result

        elif drawing_type == "plan":
            # 阶段 1：交叉验证提取轴网
            grid_info = self._extract_grid_with_validation(image_path, ocr_results)

            # 阶段 2：将轴网作为已知上下文，交叉验证提取构件
            logger.info("[阶段 2/2] 交叉验证提取构件信息（注入已知轴网）...")
            prompt = self._build_plan_extraction_prompt(ocr_text_summary, grid_info=grid_info)
            expected_fields = ["components_above", "grid_info"]

            validation_result = self.cross_validator.validate(
                prompt=prompt,
                image_path=image_path,
                expected_fields=expected_fields
            )

            extraction_data = validation_result.final_data
            # 后处理：推算梁的数值坐标
            extraction_data = self._resolve_beam_coordinates(extraction_data)
            return extraction_data, validation_result

        else:
            raise ValueError(f"未知的图纸类型: {drawing_type}")

    def get_context_summary(self) -> Optional[Dict[str, Any]]:
        """
        获取上下文摘要

        Returns:
            上下文摘要（如果启用了上下文管理）
        """
        if self.context:
            return self.context.get_summary()
        return None

    def save_context(self, file_path: Optional[str] = None):
        """
        保存上下文

        Args:
            file_path: 文件路径
        """
        if self.context:
            self.context.save(file_path)
            logger.info(f"上下文已保存")

    def load_context(self, file_path: Optional[str] = None):
        """
        加载上下文

        Args:
            file_path: 文件路径
        """
        if self.context:
            self.context.load(file_path)
            logger.info(f"上下文已加载")


class EnhancedBatchImageProcessor:
    """
    增强的批量图纸处理器

    支持交叉验证和上下文管理
    """

    def __init__(
        self,
        ocr_client,
        llm_client,
        ocr_confidence_threshold: float = 0.85,
        output_dir: str = "./output",
        preprocessing_config: dict = None,
        cross_validation_enabled: bool = True,
        context_enabled: bool = True,
        context_dir: str = "./context"
    ):
        """
        初始化批量处理器

        Args:
            ocr_client: OCR 客户端
            llm_client: LLM 客户端
            ocr_confidence_threshold: OCR 置信度阈值
            output_dir: 输出目录
            preprocessing_config: 图像预处理配置
            cross_validation_enabled: 是否启用交叉验证
            context_enabled: 是否启用上下文管理
            context_dir: 上下文存储目录
        """
        self.processor = EnhancedImageProcessor(
            ocr_client=ocr_client,
            llm_client=llm_client,
            ocr_confidence_threshold=ocr_confidence_threshold,
            output_dir=output_dir,
            preprocessing_config=preprocessing_config,
            cross_validation_enabled=cross_validation_enabled,
            context_enabled=context_enabled,
            context_dir=context_dir
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

        # 打印上下文摘要
        if self.processor.context:
            summary = self.processor.get_context_summary()
            logger.info(f"上下文摘要: {summary['total_messages']} 条消息, {summary['images_processed']} 张图片")

        logger.info("="*60)

        return results


__all__ = [
    "EnhancedImageProcessor",
    "EnhancedBatchImageProcessor"
]
