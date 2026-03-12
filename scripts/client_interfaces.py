"""
客户端接口定义

定义 OCR 和 LLM 客户端的标准接口，供 Skill 实现
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
import json
import base64

import os as _os

# API Keys — 从环境变量读取，不在代码中硬编码
# 请在 .env 文件或系统环境变量中配置（参见 .env.example）
HARD_CODED_OPENROUTER_API_KEY: Optional[str] = _os.environ.get("OPENROUTER_API_KEY")
HARD_CODED_OPUS_API_KEY: Optional[str] = _os.environ.get("OPENROUTER_API_KEY")
HARD_CODED_QWEN_API_KEY: Optional[str] = _os.environ.get("OPENROUTER_API_KEY")


class OCRClientInterface(ABC):
    """OCR 客户端接口"""

    @abstractmethod
    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """
        识别图片中的文字

        Args:
            image_path: 图片路径

        Returns:
            识别结果列表，每个元素包含：
            {
                "text": str,           # 识别的文字
                "confidence": float,   # 置信度 (0-1)
                "bbox": [x1, y1, x2, y2]  # 边界框坐标
            }
        """
        pass


class LLMClientInterface(ABC):
    """LLM 客户端接口"""

    @abstractmethod
    def chat(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        调用 LLM 进行对话

        Args:
            prompt: 提示词
            image_path: 图片路径（可选，用于 Vision 模型）

        Returns:
            LLM 响应字符串
        """
        pass


# ============================================================================
# 默认实现
# ============================================================================

class PaddleOCRClient(OCRClientInterface):
    """
    PaddleOCR 客户端实现

    通过 HTTP API 调用 OCR 服务
    遇到 500 错误时自动重启服务并重试一次
    """

    def __init__(self, server_url: str = "http://localhost:5000"):
        """
        初始化客户端

        Args:
            server_url: OCR 服务器地址
        """
        import time
        self.server_url = server_url.rstrip("/")
        # 记录最近一次服务重启完成的时间戳（含模型加载等待），避免连续重启
        self._last_restart_time: float = 0.0

    def _restart_service(self) -> bool:
        """
        终止当前 OCR 服务进程并重新启动

        使用 sys.executable（当前 Python 解释器）启动服务，保证与运行时环境一致。
        stderr 重定向到日志文件（ocr_service/ocr_server_stderr.log）便于排查崩溃。

        Returns:
            重启是否成功
        """
        import subprocess
        import sys
        import time
        import os
        import re

        print("OCR 服务不可用，正在重启...", flush=True)

        # 找到占用端口的进程 PID（Windows: netstat，Linux/Mac: lsof）
        try:
            port = self.server_url.split(":")[-1].split("/")[0]
            if os.name == "nt":
                out = subprocess.check_output(
                    f'netstat -ano | findstr ":{port} "',
                    shell=True, text=True, stderr=subprocess.DEVNULL
                )
                pids = set(re.findall(r'\s+(\d+)\s*$', out, re.MULTILINE))
                for pid in pids:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
            else:
                out = subprocess.check_output(
                    f"lsof -ti :{port}", shell=True, text=True, stderr=subprocess.DEVNULL
                )
                for pid in out.strip().split():
                    subprocess.run(
                        ["kill", "-9", pid],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
            print(f"已终止端口 {port} 上的 OCR 进程", flush=True)
        except subprocess.CalledProcessError:
            print("未找到占用端口的进程，直接尝试启动", flush=True)
        except Exception as e:
            print(f"终止进程时出错: {e}", flush=True)

        time.sleep(2)

        # 重新启动 ocr_server.py
        ocr_server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_service", "ocr_server.py"
        )
        if not os.path.exists(ocr_server_path):
            print(f"找不到 OCR 服务脚本: {ocr_server_path}", flush=True)
            return False

        # stderr 写入日志文件，方便排查崩溃
        log_path = os.path.join(os.path.dirname(ocr_server_path), "ocr_server_stderr.log")
        log_file = open(log_path, "a", encoding="utf-8", errors="replace")

        # 使用 sys.executable 保证与当前运行环境一致（避免路径或 conda env 不匹配）
        subprocess.Popen(
            [sys.executable, ocr_server_path],
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            cwd=os.path.dirname(ocr_server_path)
        )

        # 等待 Flask 启动（最多 30 秒）
        print("OCR 服务已重新启动，等待 Flask 就绪（最多 30 秒）...", flush=True)
        flask_ready_at = None
        for i in range(30):
            time.sleep(1)
            try:
                resp = requests.get(f"{self.server_url}/health", timeout=3,
                                    proxies={"http": None, "https": None})
                if resp.status_code == 200:
                    flask_ready_at = i + 1
                    print(f"OCR Flask 已就绪（{flask_ready_at}秒），额外等待 20 秒让模型加载...", flush=True)
                    break
            except Exception:
                pass

        if flask_ready_at is None:
            print(f"OCR 服务重启超时（Flask 未响应），详见日志: {log_path}", flush=True)
            return False

        # 额外等待以确保 PaddleOCR 模型完全加载（Flask 就绪 ≠ 模型就绪）
        time.sleep(20)
        self._last_restart_time = time.time()
        print("OCR 服务模型加载等待完成，开始使用", flush=True)
        return True

    def _do_recognize(self, image_data: str) -> List[Dict[str, Any]]:
        """发起一次 OCR HTTP 请求并解析结果"""
        response = requests.post(
            f"{self.server_url}/ocr",
            json={"image": image_data},
            timeout=1200,  # 20 分钟
            proxies={"http": None, "https": None}
        )

        if response.status_code == 500:
            raise RuntimeError("OCR_500")

        if response.status_code != 200:
            raise RuntimeError(f"OCR 服务调用失败: {response.status_code}")

        result = response.json()

        ocr_results = []
        items = result.get("results", result.get("result", []))

        for item in items:
            bbox = item.get("bbox", item.get("box", [0, 0, 0, 0]))
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

            ocr_results.append({
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "bbox": bbox
            })

        return ocr_results

    def _ensure_service_running(self) -> bool:
        """
        检查 OCR 服务健康状态，若未运行则自动启动。

        每张图纸识别前调用，确保服务始终可用。
        注意：health 接口只表示 Flask 存活；PaddleOCR 模型加载需要额外 ~20 秒。
        _restart_service() 内部已含模型等待，此处无需重复等待。

        Returns:
            服务是否可用
        """
        import time

        # 若刚完成重启（60秒内），跳过 health 检查，直接认为可用，避免重复重启
        if time.time() - self._last_restart_time < 60:
            return True

        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5,
                                proxies={"http": None, "https": None})
            if resp.status_code == 200:
                return True
        except Exception:
            pass

        # 服务不可用，尝试启动
        print("OCR 服务未运行或不健康，正在启动...", flush=True)
        return self._restart_service()

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """
        识别图片中的文字

        每次调用前先确保 OCR 服务正在运行（自动启动）。
        遇到 500 错误或连接失败时自动重启并重试一次。

        Args:
            image_path: 图片路径

        Returns:
            识别结果列表
        """
        # 每张图纸前确保服务运行
        if not self._ensure_service_running():
            raise RuntimeError("OCR 服务不可用且无法启动")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        try:
            return self._do_recognize(image_data)
        except RuntimeError as e:
            if str(e) != "OCR_500":
                # 非 500 错误：也尝试重启一次
                print(f"OCR 识别异常: {e}，尝试重启服务...", flush=True)
                restarted = self._restart_service()
                if not restarted:
                    raise
                print("重启完成，重新识别...", flush=True)
                return self._do_recognize(image_data)
            # 500 错误：重启服务后重试一次
            restarted = self._restart_service()
            if not restarted:
                raise RuntimeError("OCR 服务重启失败，无法继续识别")
            print("重启完成，重新识别...", flush=True)
            return self._do_recognize(image_data)
        except requests.exceptions.RequestException as e:
            # 网络连接类错误（ConnectionError、Timeout 等）：重启服务后重试
            print(f"OCR 连接失败: {e}，尝试重启服务...", flush=True)
            restarted = self._restart_service()
            if not restarted:
                raise RuntimeError(f"OCR 服务连接失败且重启失败: {e}")
            print("重启完成，重新识别...", flush=True)
            return self._do_recognize(image_data)


# Anthropic 的 5MB 限制针对 base64 字符串长度，原始文件 base64 后增大约 33%
# 因此原始文件必须 < 5MB * 3/4 ≈ 3.75MB 才能安全传输，留余量取 3.5MB
IMAGE_SIZE_LIMIT = int(3.5 * 1024 * 1024)  # ~3.5MB 原始文件 → base64 约 4.7MB

# Anthropic 对图片的像素尺寸限制（任意一边不超过此值）
IMAGE_MAX_DIMENSION = 8000


class OpenRouterLLMClient(LLMClientInterface):
    """
    OpenRouter LLM 客户端实现

    支持通过 OpenRouter 调用各种 LLM 模型
    支持 Gemini 3.1 Pro Preview 的推理模式
    对超过 5MB 的图片，自动上传到临时公共 URL（绕过 Anthropic base64 限制）
    """

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-3.1-pro-preview",
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        reasoning_enabled: bool = False
    ):
        """
        初始化客户端

        Args:
            api_key: OpenRouter API 密钥
            model: 模型名称
            api_url: API 地址
            max_tokens: 最大 token 数
            temperature: 温度参数
            reasoning_enabled: 是否启用推理模式（仅 Gemini 3.1 Pro Preview）
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_enabled = reasoning_enabled

        # 存储对话历史（用于推理模式）
        self.conversation_history = []

        # 缓存已上传图片的 URL，避免重复上传
        self._url_cache: Dict[str, str] = {}
        # URL 上传服务是否可用（首次失败后跳过）
        self._upload_service_available: bool = True
        # 缓存已压缩图片的字节，避免重复压缩
        self._compressed_cache: Dict[str, tuple] = {}

    def _upload_image_for_url(self, image_path: str) -> str:
        """
        将图片上传到 0x0.st 临时公共存储，返回可访问的 URL

        Args:
            image_path: 本地图片路径

        Returns:
            公共 URL 字符串

        Raises:
            RuntimeError: 上传失败（网络不通或服务错误）
        """
        import os

        # 检查缓存
        if image_path in self._url_cache:
            print(f"使用缓存 URL: {self._url_cache[image_path]}", flush=True)
            return self._url_cache[image_path]

        size_mb = os.path.getsize(image_path) / 1024 / 1024
        print(f"[URL-UPLOAD] 正在上传 {os.path.basename(image_path)} ({size_mb:.1f}MB) 到 0x0.st...", flush=True)

        with open(image_path, "rb") as f:
            response = requests.post(
                "https://0x0.st",
                files={"file": (os.path.basename(image_path), f)},
                timeout=30
            )

        if response.status_code != 200:
            raise RuntimeError(f"图片上传失败: {response.status_code}\n{response.text}")

        url = response.text.strip()
        self._url_cache[image_path] = url
        print(f"[URL-UPLOAD] 上传成功: {url}", flush=True)
        return url

    def _compress_image_to_bytes(self, image_path: str) -> tuple:
        """
        将图片调整至 Anthropic 限制以内：
          - 任意一边 ≤ IMAGE_MAX_DIMENSION（8000px）
          - 原始字节 ≤ IMAGE_SIZE_LIMIT（3.5MB → base64 < 5MB）

        Args:
            image_path: 本地图片路径

        Returns:
            (bytes, mime_type) 压缩后的图片字节和 MIME 类型
        """
        import os
        from PIL import Image
        import io

        mime_type = "image/jpeg"
        img = Image.open(image_path).convert("RGB")
        original_size = os.path.getsize(image_path)
        original_dims = (img.width, img.height)

        # Step 1: 若任意边超过像素限制，先等比缩放
        if img.width > IMAGE_MAX_DIMENSION or img.height > IMAGE_MAX_DIMENSION:
            scale = min(IMAGE_MAX_DIMENSION / img.width, IMAGE_MAX_DIMENSION / img.height)
            new_w, new_h = int(img.width * scale), int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(
                f"[COMPRESS] 像素缩放: {original_dims[0]}x{original_dims[1]} → {new_w}x{new_h}",
                flush=True
            )

        # Step 2: 逐步降低 JPEG 质量直到满足文件大小限制
        for quality in range(85, 25, -10):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= IMAGE_SIZE_LIMIT:
                print(
                    f"[COMPRESS] {os.path.basename(image_path)} "
                    f"{original_size/1024/1024:.1f}MB → {len(data)/1024/1024:.1f}MB "
                    f"(quality={quality}, {img.width}x{img.height}px)",
                    flush=True
                )
                return data, mime_type

        # Step 3: 质量仍不够，继续缩小分辨率
        scale = 0.75
        while scale >= 0.3:
            w, h = int(img.width * scale), int(img.height * scale)
            resized = img.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=70, optimize=True)
            data = buf.getvalue()
            if len(data) <= IMAGE_SIZE_LIMIT:
                print(
                    f"[COMPRESS] {os.path.basename(image_path)} "
                    f"{original_size/1024/1024:.1f}MB → {len(data)/1024/1024:.1f}MB "
                    f"(scale={scale:.0%}, {w}x{h}px)",
                    flush=True
                )
                return data, mime_type
            scale -= 0.15

        # 兜底
        buf = io.BytesIO()
        img.resize((int(img.width * 0.3), int(img.height * 0.3)), Image.LANCZOS).save(
            buf, format="JPEG", quality=60
        )
        return buf.getvalue(), mime_type

    def chat(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        调用 LLM 进行对话

        Args:
            prompt: 提示词
            image_path: 图片路径（可选）

        Returns:
            LLM 响应字符串
        """
        import os

        # 构建消息
        if image_path:
            file_size = os.path.getsize(image_path)

            # 检查像素尺寸（避免完整加载，只读 header）
            from PIL import Image as _PILImage
            with _PILImage.open(image_path) as _img:
                _w, _h = _img.size
            _exceeds_dim = _w > IMAGE_MAX_DIMENSION or _h > IMAGE_MAX_DIMENSION

            if file_size > IMAGE_SIZE_LIMIT or _exceeds_dim:
                # 文件超出 base64 安全限制，先尝试 URL 上传，失败则压缩
                use_compression = not self._upload_service_available
                if not use_compression:
                    try:
                        public_url = self._upload_image_for_url(image_path)
                        image_content = {
                            "type": "image_url",
                            "image_url": {"url": public_url}
                        }
                        use_compression = False
                    except Exception as upload_err:
                        print(
                            f"[COMPRESS-FALLBACK] URL 上传失败，标记服务不可用，改用本地压缩...",
                            flush=True
                        )
                        self._upload_service_available = False
                        use_compression = True

                if use_compression:
                    # 使用缓存的压缩结果
                    if image_path not in self._compressed_cache:
                        self._compressed_cache[image_path] = self._compress_image_to_bytes(image_path)
                    image_data_bytes, mime_type = self._compressed_cache[image_path]
                    image_data = base64.b64encode(image_data_bytes).decode("utf-8")
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                    }
            else:
                # 文件在限制内：直接 base64 编码
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp"
                }.get(ext, "image/png")

                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                }

            # 构建多模态消息
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        else:
            # 纯文本消息
            user_message = {
                "role": "user",
                "content": prompt
            }

        # 添加到对话历史
        messages = self.conversation_history + [user_message]

        # 调用 API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        # 如果启用推理模式（仅 Gemini 3.1 Pro Preview）
        if self.reasoning_enabled and "gemini-3.1-pro-preview" in self.model.lower():
            payload["reasoning"] = {"enabled": True}

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=(30, 600)  # (连接超时30s, 读取超时600s) — 防止服务器挂起不响应
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM API 调用失败: {response.status_code}\n{response.text}"
            )

        result = response.json()
        assistant_message = result["choices"][0]["message"]

        # 保存对话历史（包含推理细节）
        if self.reasoning_enabled:
            self.conversation_history.append(user_message)
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.get("content"),
                "reasoning_details": assistant_message.get("reasoning_details")
            })

        return assistant_message["content"]

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []



# ============================================================================
# 工厂函数
# ============================================================================

def create_ocr_client(config: Dict[str, Any]) -> OCRClientInterface:
    """
    从配置创建 OCR 客户端

    Args:
        config: 配置字典

    Returns:
        OCR 客户端实例
    """
    ocr_config = config.get("ocr", {})
    engine = ocr_config.get("engine", "PaddleOCR")

    if engine == "PaddleOCR":
        return PaddleOCRClient(
            server_url=ocr_config.get("server_url", "http://localhost:5000")
        )
    else:
        raise ValueError(f"不支持的 OCR 引擎: {engine}")


def create_llm_client(config: Dict[str, Any]) -> LLMClientInterface:
    """
    从配置创建 LLM 客户端

    Args:
        config: 配置字典

    Returns:
        LLM 客户端实例
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openrouter")

    # 目前仅支持 openrouter 作为 LLM 提供商
    if provider != "openrouter":
        raise ValueError(f"当前仅支持 openrouter 作为 LLM 提供商，收到: {provider}")

    # 读取 API 密钥：优先使用环境变量，其次配置文件，最后硬编码常量
    import os
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or llm_config.get("api_key")
        or HARD_CODED_OPENROUTER_API_KEY
    )
    if not api_key:
        raise ValueError("请通过环境变量 OPENROUTER_API_KEY 或配置文件 llm.api_key 提供 API Key（参见 .env.example）")

    return OpenRouterLLMClient(
        api_key=api_key,
        model=llm_config.get("model", "google/gemini-3.1-pro-preview"),
        api_url=llm_config.get("api_url", "https://openrouter.ai/api/v1/chat/completions"),
        max_tokens=llm_config.get("max_tokens", 4096),
        temperature=llm_config.get("temperature", 0.1),
        reasoning_enabled=llm_config.get("reasoning_enabled", False)
    )


def create_opus_client(
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    reasoning_enabled: bool = True
) -> LLMClientInterface:
    """
    创建 Opus 4.6 客户端

    Args:
        api_key: API 密钥（如果为 None，从环境变量 OPUS_API_KEY / OPENROUTER_API_KEY 读取）
        max_tokens: 最大 token 数
        temperature: 温度参数
        reasoning_enabled: 是否启用推理模式

    Returns:
        Opus LLM 客户端实例
    """
    import os

    if api_key is None:
        api_key = (
            os.environ.get("OPUS_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or HARD_CODED_OPUS_API_KEY
        )

    if not api_key:
        raise ValueError("请通过环境变量 OPENROUTER_API_KEY 提供 API Key（参见 .env.example）")

    return OpenRouterLLMClient(
        api_key=api_key,
        model="anthropic/claude-opus-4.6",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled
    )


def create_qwen_client(
    api_key: Optional[str] = None,
    max_tokens: int = 32768,
    temperature: float = 0.1,
    reasoning_enabled: bool = True
) -> LLMClientInterface:
    """
    创建 Qwen3.5-397B 客户端（用作交叉验证辅助模型）

    Args:
        api_key: API 密钥（如果为 None，使用硬编码的密钥）
        max_tokens: 最大 token 数
        temperature: 温度参数
        reasoning_enabled: 是否启用推理模式

    Returns:
        Qwen LLM 客户端实例
    """
    import os

    if api_key is None:
        api_key = HARD_CODED_QWEN_API_KEY or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("请通过环境变量 OPENROUTER_API_KEY 提供 API Key（参见 .env.example）")

    return OpenRouterLLMClient(
        api_key=api_key,
        model="qwen/qwen3.5-397b-a17b",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled
    )


def create_gemini_client(
    api_key: Optional[str] = None,
    max_tokens: int = 32768,
    temperature: float = 0.1,
    reasoning_enabled: bool = True
) -> LLMClientInterface:
    """
    创建 Gemini 3.1 Pro 客户端

    Args:
        api_key: API 密钥（如果为 None，使用硬编码的密钥）
        max_tokens: 最大 token 数
        temperature: 温度参数
        reasoning_enabled: 是否启用推理模式

    Returns:
        Gemini LLM 客户端实例
    """
    import os

    # 使用硬编码的 Gemini API Key
    if api_key is None:
        api_key = HARD_CODED_OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("请通过环境变量 OPENROUTER_API_KEY 提供 API Key（参见 .env.example）")

    return OpenRouterLLMClient(
        api_key=api_key,
        model="google/gemini-3.1-pro-preview",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled
    )


__all__ = [
    "OCRClientInterface",
    "LLMClientInterface",
    "PaddleOCRClient",
    "OpenRouterLLMClient",
    "create_ocr_client",
    "create_llm_client",
    "create_opus_client",
    "create_qwen_client",
    "create_gemini_client"
]
