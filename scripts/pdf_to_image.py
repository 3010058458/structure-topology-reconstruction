"""
PDF/图像转换模块 (v2 Vision-LLM 方案)

将 PDF 工程图纸转换为高分辨率图像，或直接处理图像文件，供 Vision-LLM 分析使用
支持格式：PDF, PNG, JPG, JPEG, BMP, TIFF, GIF, WEBP
"""
import os
from typing import List, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image


class PDFToImageConverter:
    """PDF/图像转换器 - 支持 PDF 和常见图像格式"""

    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    def __init__(self, dpi: int = 600, output_format: str = "png", quality: int = 95):
        """
        初始化转换器

        Args:
            dpi: 输出图像分辨率（默认 600，工程图纸最高质量）
            output_format: 输出格式 "png" 或 "jpeg"
            quality: JPEG 质量（1-100，默认 95，仅对 JPEG 有效）
        """
        self.dpi = dpi
        self.output_format = output_format.lower()
        self.quality = quality

        if self.output_format not in ["png", "jpeg", "jpg"]:
            raise ValueError(f"不支持的输出格式: {output_format}")

        if not (1 <= quality <= 100):
            raise ValueError(f"质量参数必须在 1-100 之间，当前值: {quality}")

    def is_image_file(self, file_path: str) -> bool:
        """
        判断文件是否为支持的图像格式

        Args:
            file_path: 文件路径

        Returns:
            是否为图像文件
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_IMAGE_FORMATS

    def process_image_file(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        normalize: bool = True,
        max_dimension: Optional[int] = None
    ) -> str:
        """
        处理图像文件，可选择标准化为统一格式（保持高质量）

        Args:
            image_path: 图像文件路径
            output_dir: 输出目录（如果为 None 且 normalize=True，则保存在图像同目录下）
            normalize: 是否标准化为统一格式
            max_dimension: 最大尺寸限制（像素），None 表示不限制。如果图像超过此尺寸，会等比例缩放

        Returns:
            处理后的图像文件路径（如果 normalize=False，返回原路径）
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        if not self.is_image_file(image_path):
            raise ValueError(f"不支持的图像格式: {image_path}")

        # 如果不需要标准化，直接返回原路径
        if not normalize:
            return image_path

        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)

        # 读取图像
        img = Image.open(image_path)

        # 记录原始尺寸
        original_size = (img.width, img.height)

        # 转换为 RGB（如果需要）
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # 如果设置了最大尺寸限制，进行高质量缩放
        if max_dimension and (img.width > max_dimension or img.height > max_dimension):
            # 计算缩放比例
            scale = min(max_dimension / img.width, max_dimension / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))

            # 使用 LANCZOS 高质量重采样
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"图像已缩放: {original_size} -> {new_size}")

        # 生成输出文件名
        image_name = Path(image_path).stem
        output_filename = f"{image_name}_normalized.{self.output_format}"
        output_path = os.path.join(output_dir, output_filename)

        # 保存标准化图像（高质量）
        if self.output_format == "png":
            # PNG 使用最高压缩级别但不损失质量
            img.save(output_path, format="PNG", optimize=True)
        else:
            # JPEG 使用指定质量
            img.save(output_path, format="JPEG", quality=self.quality, optimize=True)

        print(f"已标准化: {output_filename} ({img.width}x{img.height})")

        return output_path

    def convert_pdf_to_images(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None,
        alpha: bool = False
    ) -> List[str]:
        """
        将 PDF 转换为高质量图像文件

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录（如果为 None，则保存在 PDF 同目录下）
            page_range: 页面范围 (start, end)，如果为 None 则转换所有页面
            alpha: 是否保留透明通道（PNG 格式）

        Returns:
            生成的图像文件路径列表
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        os.makedirs(output_dir, exist_ok=True)

        # 打开 PDF
        doc = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).stem

        # 确定页面范围
        start_page = 0
        end_page = len(doc)
        if page_range:
            start_page = max(0, page_range[0])
            end_page = min(len(doc), page_range[1])

        # 转换每一页
        image_paths = []
        zoom = self.dpi / 72.0  # PDF 默认 72 DPI
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(start_page, end_page):
            page = doc[page_num]

            # 使用高质量渲染参数
            pix = page.get_pixmap(
                matrix=mat,
                alpha=alpha,  # 是否保留透明通道
                colorspace=fitz.csRGB  # 使用 RGB 色彩空间
            )

            # 生成输出文件名
            if len(doc) == 1:
                output_filename = f"{pdf_name}.{self.output_format}"
            else:
                output_filename = f"{pdf_name}_page{page_num + 1}.{self.output_format}"

            output_path = os.path.join(output_dir, output_filename)

            # 保存图像（高质量）
            if self.output_format == "png":
                # PNG 无损压缩
                pix.save(output_path)
            else:
                # JPEG 使用指定质量
                # PyMuPDF 的 JPEG 保存不支持 quality 参数，需要通过 PIL 转换
                from PIL import Image
                import io

                # 将 pixmap 转换为 PIL Image
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))

                # 保存为高质量 JPEG
                img.save(output_path, "JPEG", quality=self.quality, optimize=True)

            image_paths.append(output_path)
            print(f"已转换: {output_filename} ({pix.width}x{pix.height}, DPI: {self.dpi})")

        doc.close()
        return image_paths

    def convert_file_to_images(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None,
        normalize_images: bool = False
    ) -> List[str]:
        """
        统一接口：将 PDF 或图像文件转换为标准格式

        Args:
            file_path: PDF 或图像文件路径
            output_dir: 输出目录
            page_range: 页面范围（仅对 PDF 有效）
            normalize_images: 是否标准化图像文件

        Returns:
            生成的图像文件路径列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        # 处理 PDF 文件
        if file_ext == '.pdf':
            return self.convert_pdf_to_images(file_path, output_dir, page_range)

        # 处理图像文件
        elif self.is_image_file(file_path):
            processed_path = self.process_image_file(file_path, output_dir, normalize_images)
            return [processed_path]

        else:
            raise ValueError(f"不支持的文件格式: {file_ext}。支持的格式: PDF, {', '.join(self.SUPPORTED_IMAGE_FORMATS)}")

    def batch_convert_pdfs(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False,
        include_images: bool = True
    ) -> dict:
        """
        批量转换目录下的所有 PDF 和图像文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            recursive: 是否递归处理子目录
            include_images: 是否包含图像文件

        Returns:
            转换结果字典 {file_path: [image_paths]}
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # 查找所有支持的文件
        supported_files = []
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file).suffix.lower()
                    if file_ext == '.pdf' or (include_images and file_ext in self.SUPPORTED_IMAGE_FORMATS):
                        supported_files.append(file_path)
        else:
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path):
                    file_ext = Path(file).suffix.lower()
                    if file_ext == '.pdf' or (include_images and file_ext in self.SUPPORTED_IMAGE_FORMATS):
                        supported_files.append(file_path)

        # 批量转换
        results = {}
        for file_path in supported_files:
            try:
                print(f"\n处理: {os.path.basename(file_path)}")
                image_paths = self.convert_file_to_images(file_path, output_dir)
                results[file_path] = image_paths
            except Exception as e:
                print(f"转换失败 {file_path}: {e}")
                results[file_path] = []

        return results


# 便捷函数
def convert_file_to_images(
    file_path: str,
    dpi: int = 600,
    output_format: str = "png",
    output_dir: Optional[str] = None,
    normalize_images: bool = False,
    quality: int = 95
) -> List[str]:
    """
    便捷函数：将 PDF 或图像文件转换为标准格式

    Args:
        file_path: PDF 或图像文件路径
        dpi: 输出分辨率（仅对 PDF 有效，默认 600，工程图纸最高质量）
        output_format: 输出格式
        output_dir: 输出目录
        normalize_images: 是否标准化图像文件
        quality: JPEG 质量（1-100，默认 95）

    Returns:
        生成的图像文件路径列表
    """
    converter = PDFToImageConverter(dpi=dpi, output_format=output_format, quality=quality)
    return converter.convert_file_to_images(file_path, output_dir, normalize_images=normalize_images)


def convert_pdf_to_images(
    pdf_path: str,
    dpi: int = 600,
    output_format: str = "png",
    output_dir: Optional[str] = None,
    quality: int = 95
) -> List[str]:
    """
    便捷函数：将 PDF 转换为高质量图像（保持向后兼容）

    Args:
        pdf_path: PDF 文件路径
        dpi: 输出分辨率（默认 600，工程图纸最高质量）
        output_format: 输出格式
        output_dir: 输出目录
        quality: JPEG 质量（1-100，默认 95）

    Returns:
        生成的图像文件路径列表
    """
    converter = PDFToImageConverter(dpi=dpi, output_format=output_format, quality=quality)
    return converter.convert_pdf_to_images(pdf_path, output_dir)


def batch_convert_pdfs(
    input_dir: str,
    output_dir: str,
    dpi: int = 600,
    output_format: str = "png",
    recursive: bool = False,
    include_images: bool = True,
    quality: int = 95
) -> dict:
    """
    便捷函数：批量转换 PDF 和图像文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        dpi: 输出分辨率（默认 600，工程图纸最高质量）
        output_format: 输出格式
        recursive: 是否递归处理子目录
        include_images: 是否包含图像文件
        quality: JPEG 质量（1-100，默认 95）

    Returns:
        转换结果字典
    """
    converter = PDFToImageConverter(dpi=dpi, output_format=output_format, quality=quality)
    return converter.batch_convert_pdfs(input_dir, output_dir, recursive, include_images)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF/图像转换工具 - 支持 PDF, PNG, JPG, BMP, TIFF, GIF, WEBP")
    parser.add_argument("--input", required=True, help="输入 PDF/图像文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--dpi", type=int, default=600, help="输出分辨率 (默认 600, 仅对 PDF 有效)")
    parser.add_argument("--format", default="png", choices=["png", "jpeg"], help="输出格式")
    parser.add_argument("--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("--no-images", action="store_true", help="批量处理时排除图像文件")
    parser.add_argument("--normalize", action="store_true", help="标准化图像文件格式")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        # 单个文件
        images = convert_file_to_images(
            args.input,
            dpi=args.dpi,
            output_format=args.format,
            output_dir=args.output,
            normalize_images=args.normalize
        )
        print(f"\n转换完成，生成 {len(images)} 个图像文件")
    else:
        # 批量处理
        results = batch_convert_pdfs(
            args.input,
            args.output,
            dpi=args.dpi,
            output_format=args.format,
            recursive=args.recursive,
            include_images=not args.no_images
        )
        total_images = sum(len(imgs) for imgs in results.values())
        print(f"\n批量转换完成，处理 {len(results)} 个文件，生成 {total_images} 个图像文件")


# ============================================================================
# Coze 平台专用函数
# ============================================================================

def convert_pdf_for_coze(
    pdf_path: str,
    output_dir: str = "./temp",
    dpi: int = 600
) -> List[str]:
    """
    Coze 专用：将 PDF 转换为图像

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        dpi: 分辨率（默认 600，工程图纸最高质量）

    Returns:
        图像文件路径列表
    """
    converter = PDFToImageConverter(dpi=dpi, output_format="png")
    return converter.convert_pdf_to_images(pdf_path, output_dir)

