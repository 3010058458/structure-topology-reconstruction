"""
上下文管理模块

管理 LLM 对话的上下文，支持多轮对话和上下文持久化
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from logger import get_logger

logger = get_logger(__name__)


class ConversationContext:
    """
    对话上下文管理器

    存储和管理 LLM 对话的上下文信息
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        context_dir: str = "./context",
        auto_save: bool = True
    ):
        """
        初始化上下文管理器

        Args:
            session_id: 会话 ID（如果为 None，自动生成）
            context_dir: 上下文存储目录
            auto_save: 是否自动保存上下文
        """
        self.session_id = session_id or self._generate_session_id()
        self.context_dir = context_dir
        self.auto_save = auto_save

        # 创建上下文目录
        os.makedirs(context_dir, exist_ok=True)

        # 对话历史
        self.messages: List[Dict[str, Any]] = []

        # 元数据
        self.metadata = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_turns": 0,
            "images_processed": []
        }

        logger.info(f"初始化上下文管理器: session_id={self.session_id}")

    def add_user_message(
        self,
        content: str,
        image_path: Optional[str] = None
    ):
        """
        添加用户消息

        Args:
            content: 消息内容
            image_path: 图片路径（可选）
        """
        message = {
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        if image_path:
            message["image_path"] = image_path
            if image_path not in self.metadata["images_processed"]:
                self.metadata["images_processed"].append(image_path)

        self.messages.append(message)
        self.metadata["total_turns"] += 1
        self.metadata["last_updated"] = datetime.now().isoformat()

        if self.auto_save:
            self.save()

        logger.debug(f"添加用户消息: {content[:50]}...")

    def add_assistant_message(
        self,
        content: str,
        model_name: str = "unknown",
        reasoning_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加助手消息

        Args:
            content: 消息内容
            model_name: 模型名称
            reasoning_details: 推理细节（用于支持推理模式）
            metadata: 额外的元数据
        """
        message = {
            "role": "assistant",
            "content": content,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }

        if reasoning_details:
            message["reasoning_details"] = reasoning_details

        if metadata:
            message["metadata"] = metadata

        self.messages.append(message)
        self.metadata["last_updated"] = datetime.now().isoformat()

        if self.auto_save:
            self.save()

        logger.debug(f"添加助手消息: {model_name} - {content[:50]}...")

    def add_system_message(self, content: str):
        """
        添加系统消息

        Args:
            content: 消息内容
        """
        message = {
            "role": "system",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.messages.append(message)
        self.metadata["last_updated"] = datetime.now().isoformat()

        if self.auto_save:
            self.save()

        logger.debug(f"添加系统消息: {content[:50]}...")

    def get_messages(
        self,
        include_images: bool = False,
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取消息列表

        Args:
            include_images: 是否包含图片路径
            last_n: 只返回最后 N 条消息

        Returns:
            消息列表
        """
        messages = self.messages.copy()

        if not include_images:
            # 移除图片路径
            messages = [
                {k: v for k, v in msg.items() if k != "image_path"}
                for msg in messages
            ]

        if last_n:
            messages = messages[-last_n:]

        return messages

    def get_conversation_history_for_llm(
        self,
        include_reasoning: bool = True
    ) -> List[Dict[str, Any]]:
        """
        获取适合传递给 LLM 的对话历史

        Args:
            include_reasoning: 是否包含推理细节

        Returns:
            LLM 格式的消息列表
        """
        llm_messages = []

        for msg in self.messages:
            llm_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }

            # 如果有推理细节且需要包含
            if include_reasoning and "reasoning_details" in msg:
                llm_msg["reasoning_details"] = msg["reasoning_details"]

            llm_messages.append(llm_msg)

        return llm_messages

    def save(self, file_path: Optional[str] = None):
        """
        保存上下文到文件

        Args:
            file_path: 文件路径（如果为 None，使用默认路径）
        """
        if file_path is None:
            file_path = os.path.join(
                self.context_dir,
                f"context_{self.session_id}.json"
            )

        context_data = {
            "metadata": self.metadata,
            "messages": self.messages
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(context_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"上下文已保存: {file_path}")

    def load(self, file_path: Optional[str] = None):
        """
        从文件加载上下文

        Args:
            file_path: 文件路径（如果为 None，使用默认路径）
        """
        if file_path is None:
            file_path = os.path.join(
                self.context_dir,
                f"context_{self.session_id}.json"
            )

        if not os.path.exists(file_path):
            logger.warning(f"上下文文件不存在: {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)

        self.metadata = context_data.get("metadata", self.metadata)
        self.messages = context_data.get("messages", [])

        logger.info(f"上下文已加载: {file_path} ({len(self.messages)} 条消息)")

    def clear(self):
        """清空上下文"""
        self.messages = []
        self.metadata["total_turns"] = 0
        self.metadata["images_processed"] = []
        self.metadata["last_updated"] = datetime.now().isoformat()

        if self.auto_save:
            self.save()

        logger.info("上下文已清空")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取上下文摘要

        Returns:
            摘要信息
        """
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "total_turns": self.metadata["total_turns"],
            "images_processed": len(self.metadata["images_processed"]),
            "created_at": self.metadata["created_at"],
            "last_updated": self.metadata["last_updated"]
        }

    def _generate_session_id(self) -> str:
        """
        生成会话 ID

        Returns:
            会话 ID
        """
        from datetime import datetime
        import random
        import string

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{timestamp}_{random_suffix}"

    @classmethod
    def list_sessions(cls, context_dir: str = "./context") -> List[Dict[str, Any]]:
        """
        列出所有会话

        Args:
            context_dir: 上下文存储目录

        Returns:
            会话列表
        """
        if not os.path.exists(context_dir):
            return []

        sessions = []
        for file_name in os.listdir(context_dir):
            if file_name.startswith("context_") and file_name.endswith(".json"):
                file_path = os.path.join(context_dir, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        context_data = json.load(f)
                    sessions.append({
                        "file_path": file_path,
                        "session_id": context_data["metadata"]["session_id"],
                        "created_at": context_data["metadata"]["created_at"],
                        "total_messages": len(context_data["messages"])
                    })
                except Exception as e:
                    logger.warning(f"无法读取会话文件 {file_name}: {e}")

        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


class ContextAwareLLMClient:
    """
    支持上下文的 LLM 客户端包装器

    将普通的 LLM 客户端包装为支持上下文管理的客户端
    """

    def __init__(
        self,
        llm_client,
        context: ConversationContext,
        model_name: str = "unknown"
    ):
        """
        初始化上下文感知的 LLM 客户端

        Args:
            llm_client: 原始 LLM 客户端
            context: 对话上下文
            model_name: 模型名称
        """
        self.llm_client = llm_client
        self.context = context
        self.model_name = model_name

    def chat(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        use_context: bool = True
    ) -> str:
        """
        调用 LLM 进行对话（支持上下文）

        Args:
            prompt: 提示词
            image_path: 图片路径
            use_context: 是否使用上下文

        Returns:
            LLM 响应
        """
        # 先将历史（不含本次消息）注入 LLM 客户端，再调用 chat()
        # 注意：必须在 add_user_message 之前获取历史，否则当前消息会重复
        if use_context and hasattr(self.llm_client, 'conversation_history'):
            self.llm_client.conversation_history = self.context.get_conversation_history_for_llm()

        # 调用 LLM（chat() 内部会构建含图片的 user_message）
        response = self.llm_client.chat(prompt, image_path)

        # 调用完成后，将本次对话记录到上下文
        self.context.add_user_message(prompt, image_path)

        reasoning_details = None
        if hasattr(self.llm_client, 'conversation_history') and len(self.llm_client.conversation_history) > 0:
            last_msg = self.llm_client.conversation_history[-1]
            if isinstance(last_msg, dict) and "reasoning_details" in last_msg:
                reasoning_details = last_msg["reasoning_details"]

        self.context.add_assistant_message(
            response,
            model_name=self.model_name,
            reasoning_details=reasoning_details
        )

        return response

    def reset_context(self):
        """重置上下文"""
        self.context.clear()
        if hasattr(self.llm_client, 'reset_conversation'):
            self.llm_client.reset_conversation()


__all__ = [
    "ConversationContext",
    "ContextAwareLLMClient"
]
