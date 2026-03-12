"""配置验证模块"""
from typing import Dict
import os


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate(config: Dict) -> None:
        """
        验证配置文件

        Args:
            config: 配置字典

        Raises:
            ValueError: 配置无效时抛出
        """
        # 检查必需的顶层键
        required_keys = ["project", "ocr", "llm", "processing", "output"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置缺少必需项: {key}")

        # 验证 OCR 配置
        ConfigValidator._validate_ocr_config(config["ocr"])

        # 验证 LLM 配置
        ConfigValidator._validate_llm_config(config["llm"])

        # 验证处理配置
        ConfigValidator._validate_processing_config(config["processing"])

        # 验证输出配置
        ConfigValidator._validate_output_config(config["output"])

        print("[OK] 配置验证通过")

    @staticmethod
    def _validate_ocr_config(ocr_config: Dict) -> None:
        """验证 OCR 配置"""
        # 检查必需字段
        required_fields = ["server_url", "confidence_threshold", "enabled", "engine", "language"]
        for field in required_fields:
            if field not in ocr_config:
                raise ValueError(f"OCR 配置缺少必需字段: {field}")

        # 验证超时时间
        timeout = ocr_config.get("timeout", 0)
        if not isinstance(timeout, (int, float)):
            raise ValueError(
                f"OCR 超时时间必须是数字类型\n"
                f"当前类型: {type(timeout).__name__}\n"
                f"建议值: 1200 秒（20分钟）"
            )

        if timeout < 60:
            raise ValueError(
                f"OCR 超时时间不能小于 60 秒\n"
                f"当前值: {timeout} 秒\n"
                f"建议值: 1200 秒（20分钟）"
            )

        # 验证置信度阈值
        threshold = ocr_config.get("confidence_threshold", 0)
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"OCR 置信度阈值必须是数字类型\n"
                f"当前类型: {type(threshold).__name__}"
            )

        if not (0 <= threshold <= 1):
            raise ValueError(
                f"OCR 置信度阈值必须在 0-1 之间\n"
                f"当前值: {threshold}\n"
                f"建议值: 0.85"
            )

        # 验证服务器 URL
        if ocr_config.get("enabled"):
            server_url = ocr_config.get("server_url", "")
            if not server_url or not isinstance(server_url, str):
                raise ValueError(
                    "启用 OCR 时必须配置有效的 server_url\n"
                    "示例: http://localhost:5000"
                )

    @staticmethod
    def _validate_llm_config(llm_config: Dict) -> None:
        """验证 LLM 配置"""
        # 检查必需字段
        required_fields = ["provider", "api_url", "model"]
        for field in required_fields:
            if field not in llm_config:
                raise ValueError(f"LLM 配置缺少必需字段: {field}")

        # 验证 API URL
        api_url = llm_config.get("api_url", "")
        if not api_url or not isinstance(api_url, str):
            raise ValueError("LLM API URL 不能为空")

        if not api_url.startswith(("http://", "https://")):
            raise ValueError(
                f"LLM API URL 格式无效\n"
                f"当前值: {api_url}\n"
                f"必须以 http:// 或 https:// 开头"
            )

        # 验证 max_tokens
        max_tokens = llm_config.get("max_tokens", 0)
        if not isinstance(max_tokens, int):
            raise ValueError(
                f"LLM max_tokens 必须是整数类型\n"
                f"当前类型: {type(max_tokens).__name__}"
            )

        if max_tokens < 100:
            raise ValueError(
                f"LLM max_tokens 不能小于 100\n"
                f"当前值: {max_tokens}\n"
                f"建议值: 4096"
            )

        # 验证 temperature
        temperature = llm_config.get("temperature", 0)
        if not isinstance(temperature, (int, float)):
            raise ValueError(
                f"LLM temperature 必须是数字类型\n"
                f"当前类型: {type(temperature).__name__}"
            )

        if not (0 <= temperature <= 2):
            raise ValueError(
                f"LLM temperature 必须在 0-2 之间\n"
                f"当前值: {temperature}\n"
                f"建议值: 0.1"
            )

        # 检查 API Key（根据 provider）
        # 优先级：配置中的 api_key > 硬编码常量 > 环境变量
        provider = llm_config.get("provider", "").lower()
        has_api_key_in_config = "api_key" in llm_config and llm_config["api_key"]

        if provider == "openrouter":
            # 检查硬编码常量
            try:
                from client_interfaces import HARD_CODED_OPENROUTER_API_KEY
                has_hardcoded_key = HARD_CODED_OPENROUTER_API_KEY is not None
            except ImportError:
                has_hardcoded_key = False

            if not os.environ.get("OPENROUTER_API_KEY") and not has_api_key_in_config and not has_hardcoded_key:
                raise ValueError(
                    "使用 OpenRouter 时必须设置 OPENROUTER_API_KEY 环境变量\n"
                    "或在配置中提供 api_key 字段\n"
                    "或在 client_interfaces.py 中设置 HARD_CODED_OPENROUTER_API_KEY"
                )
        elif provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY") and not has_api_key_in_config:
                raise ValueError(
                    "使用 Anthropic 时必须设置 ANTHROPIC_API_KEY 环境变量\n"
                    "或在配置中提供 api_key 字段"
                )

    @staticmethod
    def _validate_processing_config(processing_config: Dict) -> None:
        """验证处理配置"""
        # 验证 PDF DPI
        pdf_dpi = processing_config.get("pdf_dpi", 0)
        if not isinstance(pdf_dpi, int):
            raise ValueError(
                f"PDF DPI 必须是整数类型\n"
                f"当前类型: {type(pdf_dpi).__name__}"
            )

        if pdf_dpi < 72:
            raise ValueError(
                f"PDF DPI 不能小于 72\n"
                f"当前值: {pdf_dpi}\n"
                f"建议值: 200"
            )

        # 验证支持的文件格式
        supported_formats = processing_config.get("supported_file_formats", [])
        if not isinstance(supported_formats, list):
            raise ValueError("supported_file_formats 必须是列表类型")

        if not supported_formats:
            raise ValueError("supported_file_formats 不能为空")

        # 验证容差值
        tolerance = processing_config.get("tolerance_mm", 0)
        if not isinstance(tolerance, (int, float)):
            raise ValueError(
                f"tolerance_mm 必须是数字类型\n"
                f"当前类型: {type(tolerance).__name__}"
            )

        if tolerance < 0:
            raise ValueError(
                f"tolerance_mm 不能为负数\n"
                f"当前值: {tolerance}"
            )

    @staticmethod
    def _validate_output_config(output_config: Dict) -> None:
        """验证输出配置"""
        # 检查必需字段
        required_fields = ["json_format", "include_metadata", "pretty_print"]
        for field in required_fields:
            if field not in output_config:
                raise ValueError(f"输出配置缺少必需字段: {field}")

        # 验证 json_format
        json_format = output_config.get("json_format", "")
        if json_format not in ["structured", "flat"]:
            raise ValueError(
                f"json_format 必须是 'structured' 或 'flat'\n"
                f"当前值: {json_format}"
            )


def validate_config(config: Dict) -> None:
    """
    验证配置（便捷函数）

    Args:
        config: 配置字典

    Raises:
        ValueError: 配置无效时抛出
    """
    ConfigValidator.validate(config)
