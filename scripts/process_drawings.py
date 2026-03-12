"""
主入口脚本 - 图纸处理流程

使用方法：
    python process_drawings.py --images image1.png image2.jpg --output ./output

环境变量：
    OPENROUTER_API_KEY: OpenRouter API 密钥（如果使用 OpenRouter）
    ANTHROPIC_API_KEY: Anthropic API 密钥（如果使用 Claude）
"""
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import List

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processor import BatchImageProcessor, load_config
from enhanced_image_processor import EnhancedBatchImageProcessor
from client_interfaces import create_ocr_client, create_llm_client
from config_validator import validate_config
from logger import setup_logger, get_logger
from pdf_to_image import PDFToImageConverter


def expand_pdfs_to_images(file_paths: List[str], pdf_output_dir: str) -> List[str]:
    """
    将文件列表中的 PDF 转换为图片，返回展开后的纯图片路径列表

    Args:
        file_paths: 文件路径列表（可混合 PDF 和图片）
        pdf_output_dir: PDF 转图片的临时输出目录

    Returns:
        全为图片的路径列表
    """
    import os
    converter = PDFToImageConverter(dpi=600, output_format="png")
    result = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            print(f"[PDF] 正在转换: {os.path.basename(path)} (600 DPI)...", flush=True)
            images = converter.convert_pdf_to_images(path, output_dir=pdf_output_dir)
            print(f"[PDF] 转换完成: {len(images)} 页 → {[os.path.basename(p) for p in images]}", flush=True)
            result.extend(images)
        else:
            result.append(path)
    return result


def find_images_in_directory(directory: str) -> List[str]:
    """
    在目录中查找所有支持的图片文件

    Args:
        directory: 目录路径

    Returns:
        图片路径列表
    """
    supported_formats = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"]
    image_paths = []

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in supported_formats:
            image_paths.append(str(file_path))

    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(
        description="结构工程图纸处理 - 单张图纸独立处理流程"
    )

    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--images",
        nargs="+",
        help="图片文件路径列表"
    )
    input_group.add_argument(
        "--input-dir",
        help="输入目录（自动查找所有图片）"
    )

    # 输出参数
    parser.add_argument(
        "--output",
        default="./output",
        help="输出目录（默认: ./output）"
    )

    # 配置参数
    parser.add_argument(
        "--config",
        default="../config.json",
        help="配置文件路径（默认: ../config.json）"
    )

    # OCR 参数
    parser.add_argument(
        "--ocr-threshold",
        type=float,
        help="OCR 置信度阈值（覆盖配置文件）"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="禁用 OCR（仅使用 Vision LLM）"
    )

    # LLM 参数
    parser.add_argument(
        "--llm-provider",
        choices=["openrouter", "claude"],
        help="LLM 提供商（覆盖配置文件）"
    )
    parser.add_argument(
        "--llm-model",
        help="LLM 模型名称（覆盖配置文件）"
    )

    # 交叉验证参数
    parser.add_argument(
        "--no-cross-validation",
        action="store_true",
        help="禁用交叉验证（仅使用单个模型）"
    )

    # 上下文管理参数
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="禁用上下文管理"
    )
    parser.add_argument(
        "--session-id",
        help="指定会话 ID（用于恢复之前的上下文）"
    )

    args = parser.parse_args()

    # 初始化日志系统
    logger = setup_logger(
        name="structure-topology",
        level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
        log_dir="../logs"
    )

    logger.info("="*60)
    logger.info("结构工程图纸处理 - 启动")
    logger.info("="*60)

    # 加载配置
    logger.info("加载配置...")
    try:
        config = load_config(args.config)
        logger.info(f"配置文件: {args.config}")
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        sys.exit(1)

    # 验证配置
    logger.info("验证配置...")
    try:
        validate_config(config)
        logger.info("配置验证通过")
    except ValueError as e:
        logger.error(f"配置验证失败:\n{e}")
        sys.exit(1)

    # 覆盖配置
    if args.ocr_threshold is not None:
        config["ocr"]["confidence_threshold"] = args.ocr_threshold
        logger.info(f"覆盖 OCR 置信度阈值: {args.ocr_threshold}")
    if args.llm_provider:
        config["llm"]["provider"] = args.llm_provider
        logger.info(f"覆盖 LLM 提供商: {args.llm_provider}")
    if args.llm_model:
        config["llm"]["model"] = args.llm_model
        logger.info(f"覆盖 LLM 模型: {args.llm_model}")

    # 获取图片列表
    if args.images:
        image_paths = args.images
        logger.info(f"处理 {len(image_paths)} 张指定图片")
    else:
        logger.info(f"扫描目录: {args.input_dir}")
        image_paths = find_images_in_directory(args.input_dir)
        logger.info(f"找到 {len(image_paths)} 张图片")

    if not image_paths:
        logger.error("没有找到图片文件")
        sys.exit(1)

    # PDF 转图片（如有）
    pdf_temp_dir = os.path.join(args.output, "_pdf_pages")
    image_paths = expand_pdfs_to_images(image_paths, pdf_temp_dir)
    if not image_paths:
        logger.error("PDF 转换后没有可处理的图片")
        sys.exit(1)

    # 创建客户端
    logger.info("初始化客户端...")

    try:
        if args.no_ocr:
            logger.info("OCR: 已禁用")
            ocr_client = None
        else:
            ocr_client = create_ocr_client(config)
            logger.info(f"OCR: {config['ocr']['engine']} @ {config['ocr']['server_url']}")
            logger.info(f"OCR 超时: {config['ocr']['timeout']} 秒")

        llm_client = create_llm_client(config)
        logger.info(f"LLM: {config['llm']['provider']} / {config['llm']['model']}")

    except Exception as e:
        logger.error(f"客户端初始化失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # 创建处理器
    preprocessing_config = config.get("image_preprocessing", )
    cross_validation_config = config.get("cross_validation", {})
    context_config = config.get("context_management", {})

    # 确定是否启用交叉验证和上下文管理
    cross_validation_enabled = cross_validation_config.get("enabled", True) and not args.no_cross_validation
    context_enabled = context_config.get("enabled", True) and not args.no_context

    if cross_validation_enabled or context_enabled:
        logger.info("使用增强处理器（支持交叉验证和上下文管理）")
        processor = EnhancedBatchImageProcessor(
            ocr_client=ocr_client,
            llm_client=llm_client,
            ocr_confidence_threshold=config["ocr"]["confidence_threshold"],
            output_dir=args.output,
            preprocessing_config=preprocessing_config,
            cross_validation_enabled=cross_validation_enabled,
            context_enabled=context_enabled,
            context_dir=context_config.get("context_dir", "./context")
        )

        if cross_validation_enabled:
            logger.info("交叉验证: 已启用（Gemini 3.1 Pro + Opus 4.6）")
        else:
            logger.info("交叉验证: 已禁用")

        if context_enabled:
            logger.info("上下文管理: 已启用")
            if args.session_id:
                logger.info(f"会话 ID: {args.session_id}")
        else:
            logger.info("上下文管理: 已禁用")
    else:
        logger.info("使用标准处理器")
        processor = BatchImageProcessor(
            ocr_client=ocr_client,
            llm_client=llm_client,
            ocr_confidence_threshold=config["ocr"]["confidence_threshold"],
            output_dir=args.output,
            preprocessing_config=preprocessing_config
        )

    # 处理图片
    try:
        results = processor.process_images(image_paths)

        # 打印统计信息
        logger.info("="*60)
        logger.info("处理统计")
        logger.info("="*60)

        elevation_count = sum(1 for r in results if r.drawing_type == "elevation")
        plan_count = sum(1 for r in results if r.drawing_type == "plan")
        unknown_count = sum(1 for r in results if r.drawing_type == "unknown")

        logger.info(f"立面图: {elevation_count}")
        logger.info(f"平面图: {plan_count}")
        logger.info(f"未识别: {unknown_count}")
        logger.info(f"总计: {len(results)}")

        if args.no_ocr:
            logger.info("OCR: 已禁用")
        else:
            ocr_used_count = sum(1 for r in results if r.ocr_used)
            logger.info(f"使用 OCR: {ocr_used_count}/{len(results)}")

        logger.info(f"输出目录: {args.output}")
        logger.info("="*60)
        logger.info("处理完成！")

    except KeyboardInterrupt:
        logger.warning("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
