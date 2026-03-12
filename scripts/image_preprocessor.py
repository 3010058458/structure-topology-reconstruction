"""
图像预处理模块

为 OCR 识别优化图像质量，提高识别准确率
支持多种预处理方法：灰度化、二值化、去噪、增强对比度等
"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional
from logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    图像预处理器

    专门为 OCR 识别优化图像，提高文字识别准确率
    """

    def __init__(
        self,
        output_dir: str = "./preprocessed",
        save_intermediate: bool = False
    ):
        """
        初始化预处理器

        Args:
            output_dir: 预处理图片输出目录
            save_intermediate: 是否保存中间步骤的图片（用于调试）
        """
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        os.makedirs(output_dir, exist_ok=True)

        if save_intermediate:
            os.makedirs(os.path.join(output_dir, "steps"), exist_ok=True)

    def preprocess_for_ocr(
        self,
        image_path: str,
        methods: dict = None
    ) -> str:
        """
        为 OCR 预处理图片（完整流程）

        Args:
            image_path: 原始图片路径
            methods: 预处理方法配置
                {
                    "grayscale": True,           # 灰度化
                    "adaptive_threshold": True,  # 自适应二值化
                    "denoise": True,             # 去噪
                    "enhance_contrast": True,    # 增强对比度
                    "sharpen": False             # 锐化
                }

        Returns:
            预处理后的图片路径
        """
        logger.info(f"开始预处理图片: {os.path.basename(image_path)}")

        # 默认配置
        if methods is None:
            methods = {
                "grayscale": True,
                "adaptive_threshold": False,  # 不做二值化，保留灰度信息
                "denoise": True,
                "enhance_contrast": True,
                "sharpen": False
            }

        try:
            # 读取图片（支持中文路径）
            img = self._read_image_with_chinese_path(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            logger.debug(f"原始图片尺寸: {img.shape[1]}x{img.shape[0]}")

            # 步骤 1: 灰度化
            if methods.get("grayscale", True):
                img = self._convert_to_grayscale(img, image_path)
                logger.debug("完成灰度化")

            # 步骤 2: 去噪
            if methods.get("denoise", True):
                img = self._denoise(img, image_path)
                logger.debug("完成去噪")

            # 步骤 3: 增强对比度
            if methods.get("enhance_contrast", True):
                img = self._enhance_contrast(img, image_path)
                logger.debug("完成对比度增强")

            # 步骤 4: 自适应二值化（可选）
            if methods.get("adaptive_threshold", False):
                img = self._adaptive_threshold(img, image_path)
                logger.debug("完成自适应二值化")

            # 步骤 5: 锐化（可选）
            if methods.get("sharpen", False):
                img = self._sharpen(img, image_path)
                logger.debug("完成锐化")

            # 保存预处理后的图片（支持中文路径）
            output_path = self._get_output_path(image_path)
            self._write_image_with_chinese_path(img, output_path)

            logger.info(f"预处理完成: {os.path.basename(output_path)}")
            return output_path

        except Exception as e:
            logger.error(f"图片预处理失败: {e}")
            raise

    def _convert_to_grayscale(self, img: np.ndarray, original_path: str) -> np.ndarray:
        """
        转换为灰度图

        Args:
            img: 输入图片
            original_path: 原始图片路径（用于保存中间结果）

        Returns:
            灰度图
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if self.save_intermediate:
            self._save_step(gray, original_path, "1_grayscale")

        return gray

    def _denoise(self, img: np.ndarray, original_path: str) -> np.ndarray:
        """
        去噪处理（针对工程图纸优化）

        使用非局部均值去噪算法，保留边缘细节
        对于工程图纸，使用较弱的去噪强度以保留细节

        Args:
            img: 输入图片
            original_path: 原始图片路径

        Returns:
            去噪后的图片
        """
        # 使用 fastNlMeansDenoising 进行去噪
        # h 参数控制去噪强度，对于工程图纸使用较小值（5-8）以保留细节
        # 标准文档推荐 10，但工程图纸需要保留更多细节
        denoised = cv2.fastNlMeansDenoising(img, h=6)

        if self.save_intermediate:
            self._save_step(denoised, original_path, "2_denoised")

        return denoised

    def _enhance_contrast(self, img: np.ndarray, original_path: str) -> np.ndarray:
        """
        增强对比度（针对工程图纸优化）

        使用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
        自适应直方图均衡化，增强局部对比度
        对于工程图纸，使用适中的参数以避免过度增强

        Args:
            img: 输入图片
            original_path: 原始图片路径

        Returns:
            对比度增强后的图片
        """
        # 创建 CLAHE 对象
        # clipLimit: 1.8 适合工程图纸，避免过度增强导致噪点放大
        # tileGridSize: 8x8 网格，平衡局部与全局对比度
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        if self.save_intermediate:
            self._save_step(enhanced, original_path, "3_enhanced")

        return enhanced

    def _adaptive_threshold(self, img: np.ndarray, original_path: str) -> np.ndarray:
        """
        自适应二值化

        根据局部区域自动计算阈值，适应不同光照条件

        Args:
            img: 输入图片
            original_path: 原始图片路径

        Returns:
            二值化后的图片
        """
        # 自适应阈值二值化
        # blockSize: 邻域大小，使用 15 以适应工程图纸较粗的线条与文字
        # C: 常数，设为 4 使背景更干净，文字更突出
        binary = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=4
        )

        if self.save_intermediate:
            self._save_step(binary, original_path, "4_binary")

        return binary

    def _sharpen(self, img: np.ndarray, original_path: str) -> np.ndarray:
        """
        锐化处理

        增强边缘，使文字更清晰

        Args:
            img: 输入图片
            original_path: 原始图片路径

        Returns:
            锐化后的图片
        """
        # 锐化卷积核
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        sharpened = cv2.filter2D(img, -1, kernel)

        if self.save_intermediate:
            self._save_step(sharpened, original_path, "5_sharpened")

        return sharpened

    def _get_output_path(self, original_path: str) -> str:
        """
        生成输出文件路径

        Args:
            original_path: 原始图片路径

        Returns:
            输出文件路径
        """
        filename = Path(original_path).stem
        ext = Path(original_path).suffix
        output_filename = f"{filename}_preprocessed{ext}"
        return os.path.join(self.output_dir, output_filename)

    def _save_step(self, img: np.ndarray, original_path: str, step_name: str):
        """
        保存中间步骤的图片（用于调试）

        Args:
            img: 图片数据
            original_path: 原始图片路径
            step_name: 步骤名称
        """
        filename = Path(original_path).stem
        ext = Path(original_path).suffix
        step_filename = f"{filename}_{step_name}{ext}"
        step_path = os.path.join(self.output_dir, "steps", step_filename)
        self._write_image_with_chinese_path(img, step_path)

    def _read_image_with_chinese_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        读取图片（支持中文路径）

        Args:
            image_path: 图片路径

        Returns:
            图片数据，如果失败返回 None
        """
        try:
            # 使用 numpy 读取文件，支持中文路径
            with open(image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"读取图片失败: {e}")
            return None

    def _write_image_with_chinese_path(self, img: np.ndarray, output_path: str) -> bool:
        """
        保存图片（支持中文路径）

        Args:
            img: 图片数据
            output_path: 输出路径

        Returns:
            是否成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 编码图片
            ext = Path(output_path).suffix
            success, encoded_img = cv2.imencode(ext, img)

            if success:
                # 写入文件，支持中文路径
                with open(output_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                return True
            else:
                logger.error(f"图片编码失败: {output_path}")
                return False

        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            return False

    def get_image_quality_score(self, image_path: str) -> float:
        """
        评估图片质量

        基于清晰度、对比度等指标评分

        Args:
            image_path: 图片路径

        Returns:
            质量分数 (0-100)
        """
        try:
            # 读取图片（支持中文路径）
            img = self._read_image_with_chinese_path(image_path)
            if img is None:
                return 0.0

            # 转灰度
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 计算拉普拉斯方差（衡量清晰度）
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

            # 计算对比度（标准差）
            contrast = img.std()

            # 综合评分（归一化到 0-100）
            # 清晰度权重 0.6，对比度权重 0.4
            score = min(100, (laplacian_var / 100) * 0.6 + (contrast / 2.55) * 0.4)

            logger.debug(f"图片质量评分: {score:.2f} (清晰度: {laplacian_var:.2f}, 对比度: {contrast:.2f})")
            return score

        except Exception as e:
            logger.warning(f"图片质量评估失败: {e}")
            return 50.0  # 返回中等分数


def preprocess_image_for_ocr(
    image_path: str,
    output_dir: str = "./preprocessed",
    config: dict = None
) -> str:
    """
    便捷函数：为 OCR 预处理图片

    Args:
        image_path: 原始图片路径
        output_dir: 输出目录
        config: 预处理配置

    Returns:
        预处理后的图片路径
    """
    preprocessor = ImagePreprocessor(output_dir=output_dir)

    # 从配置中提取预处理方法
    methods = None
    if config and "image_preprocessing" in config:
        methods = config["image_preprocessing"].get("methods")

    return preprocessor.preprocess_for_ocr(image_path, methods=methods)


__all__ = [
    "ImagePreprocessor",
    "preprocess_image_for_ocr"
]
