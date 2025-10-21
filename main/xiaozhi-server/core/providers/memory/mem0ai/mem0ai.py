from datetime import datetime
from typing import List, Optional, Dict, Any

from .prompt import FACT_PROMPT
from ..base import MemoryProviderBase, logger
from mem0 import Memory
from mem0.configs.base import LlmConfig, EmbedderConfig, MemoryConfig, VectorStoreConfig
from core.utils.util import check_model_key

TAG = __name__

# 配置常量
DEFAULT_API_VERSION = "v1.1"
DEFAULT_METADATA = {"category": "preferences"}
MIN_MESSAGES_FOR_SAVE = 2

# LLM配置
LLM_CONFIG = {
    "provider": "litellm",
    "config": {
        "model": "dashscope/qwen-turbo",
        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
}

# Embedder配置
EMBEDDER_CONFIG = {
    "provider": "ollama",
    "config": {
        "model": "bge-large:latest",
        "ollama_base_url": "http://localhost:11434"
    }
}

# Vector Store配置
VECTOR_CONFIG = {
    "provider": "qdrant",
    "config": {
        "collection_name": "xiaozhi_memories",
        "embedding_model_dims": 1024,
        "host": "127.0.0.1",
        "port": 6333
    }
}


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory=None):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.api_version = config.get("api_version", DEFAULT_API_VERSION)
        
        # 检查API密钥
        model_key_msg = check_model_key("Mem0ai", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
            self.use_mem0 = False
            return
        
        self.use_mem0 = True
        self._initialize_memory_client()

    def _initialize_memory_client(self) -> None:
        """初始化Memory客户端"""
        try:
            llm_config = LlmConfig(**LLM_CONFIG)
            embedder_config = EmbedderConfig(**EMBEDDER_CONFIG)
            vector_config = VectorStoreConfig(**VECTOR_CONFIG)
            
            memory_config = MemoryConfig(
                llm=llm_config,
                embedder=embedder_config,
                vector_store=vector_config,
                custom_prompt=FACT_PROMPT,
            )
            
            self.m: Memory = Memory(config=memory_config)
            logger.bind(tag=TAG).info("Memory client initialized successfully")
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to initialize memory client: {str(e)}")
            self.use_mem0 = False

    def _format_timestamp(self, timestamp: Optional[str]) -> str:
        """格式化时间戳"""
        if not timestamp:
            return ""
        
        try:
            # 移除毫秒并格式化时间
            dt_part = timestamp.split(".")[0]
            return dt_part.replace("T", " ")
        except Exception:
            return timestamp

    def _filter_system_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤掉系统消息"""
        return [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]

    async def save_memory(self, msgs) -> Optional[Any]:
        """保存记忆"""
        if not self.use_mem0 or len(msgs) < MIN_MESSAGES_FOR_SAVE:
            return None

        try:
            messages = self._filter_system_messages(msgs)
            if not messages:
                return None

            logger.bind(tag=TAG).info(f"Saving memory for role_id: {self.role_id}")
            
            result = self.m.add(
                messages, 
                user_id=self.role_id, 
                metadata=DEFAULT_METADATA
            )
            
            logger.bind(tag=TAG).info(f"Memory saved successfully for role_id: {self.role_id}")
            return result
            
        except Exception as e:
            logger.bind(tag=TAG).error(
                f"Failed to save memory for role_id {self.role_id}: {str(e)}"
            )
            return None

    async def query_memory(self, query: str) -> str:
        """查询记忆"""
        if not self.use_mem0:
            return ""

        try:
            results = self.m.search(query, user_id=self.role_id)
            logger.bind(tag=TAG).info(f"Search completed for role_id: {self.role_id}")
            
            if not results or "results" not in results:
                return ""

            memories = self._process_search_results(results["results"])
            return self._format_memories_output(memories)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to query memory: {str(e)}")
            return ""

    def _process_search_results(self, results: List[Dict[str, Any]]) -> List[tuple]:
        """处理搜索结果"""
        memories = []
        
        for entry in results:
            memory_content = entry.get("memory", "")
            if not memory_content:
                continue
                
            timestamp = entry.get("updated_at", "")
            formatted_time = self._format_timestamp(timestamp)
            
            if formatted_time:
                display_text = f"[{formatted_time}] {memory_content}"
            else:
                display_text = f"[无时间] {memory_content}"
                
            memories.append((timestamp or "", display_text))
        
        # 按时间戳降序排序
        memories.sort(key=lambda x: x[0], reverse=True)
        return memories

    def _format_memories_output(self, memories: List[tuple]) -> str:
        """格式化记忆输出"""
        if not memories:
            return ""
            
        memories_str = "\n".join(f"- {memory[1]}" for memory in memories)
        logger.bind(tag=TAG).debug(f"Formatted memories: {memories_str}")
        return memories_str