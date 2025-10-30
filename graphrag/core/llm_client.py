"""
LLM客户端模块

提供与大语言模型交互的统一接口，支持多种LLM提供商和重试机制。
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
import openai
from openai import AsyncOpenAI
import tiktoken
from dataclasses import dataclass
from ..config.settings import LLMConfig
from ..utils.exceptions import LLMError, TimeoutError as GraphRAGTimeoutError
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应数据类
    
    封装LLM API的响应结果。
    """
    
    content: str  # 生成的内容
    tokens_used: int  # 使用的token数量
    model: str  # 使用的模型
    finish_reason: str  # 完成原因
    response_time: float  # 响应时间(秒)
    metadata: Dict[str, Any]  # 额外的元数据


@dataclass
class LLMRequest:
    """LLM请求数据类
    
    封装发送给LLM的请求参数。
    """
    
    messages: List[Dict[str, str]]  # 消息列表
    temperature: Optional[float] = None  # 生成温度
    max_tokens: Optional[int] = None  # 最大token数
    system_prompt: Optional[str] = None  # 系统提示
    functions: Optional[List[Dict[str, Any]]] = None  # 函数调用定义
    metadata: Dict[str, Any] = None  # 请求元数据
    
    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}


class BaseLLMClient(ABC):
    """LLM客户端基类
    
    定义所有LLM客户端必须实现的接口。
    """
    
    def __init__(self, config: LLMConfig):
        """初始化客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        self._token_encoder = tiktoken.get_encoding(self.config.encoding_name 
                                                   if hasattr(self.config, 'encoding_name')
                                                   else 'cl100k_base')
    
    @abstractmethod
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """异步生成响应
        
        Args:
            request: LLM请求
            
        Returns:
            LLM响应
        """
        pass
    
    @abstractmethod
    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """同步生成响应
        
        Args:
            request: LLM请求
            
        Returns:
            LLM响应
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            token数量
        """
        try:
            return len(self._token_encoder.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # 使用粗略估计：1 token ≈ 4 个字符
            return len(text) // 4
    
    def estimate_request_tokens(self, request: LLMRequest) -> int:
        """估计请求的token数量
        
        Args:
            request: LLM请求
            
        Returns:
            估计的token数量
        """
        total_tokens = 0
        
        # 计算系统提示token
        if request.system_prompt:
            total_tokens += self.count_tokens(request.system_prompt)
        
        # 计算消息token
        for message in request.messages:
            content = message.get('content', '')
            total_tokens += self.count_tokens(content)
            # 为消息格式添加少量overhead
            total_tokens += 10
        
        return total_tokens


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端实现
    
    提供OpenAI API的异步和同步访问接口。
    """
    
    def __init__(self, config: LLMConfig):
        """初始化OpenAI客户端
        
        Args:
            config: LLM配置
        """
        super().__init__(config)
        
        # 初始化异步客户端
        self._async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.request_timeout
        )
        
        # 初始化同步客户端
        self._sync_client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.request_timeout
        )
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """异步生成响应
        
        Args:
            request: LLM请求
            
        Returns:
            LLM响应
            
        Raises:
            LLMError: 当API调用失败时
        """
        start_time = time.time()
        
        # 构建请求参数
        api_request = self._build_api_request(request)
        
        # 重试机制
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # 发送API请求
                response = await self._async_client.chat.completions.create(**api_request)
                
                # 解析响应
                return self._parse_response(response, start_time)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM API call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    break
        
        # 所有重试都失败了
        raise LLMError(
            f"LLM API call failed after {self.config.max_retries + 1} attempts: {last_exception}",
            model_name=self.config.model_name,
            prompt_type=request.metadata.get('prompt_type'),
            token_count=self.estimate_request_tokens(request)
        )
    
    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """同步生成响应
        
        Args:
            request: LLM请求
            
        Returns:
            LLM响应
        """
        start_time = time.time()
        
        # 构建请求参数
        api_request = self._build_api_request(request)
        
        # 重试机制
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # 发送API请求
                response = self._sync_client.chat.completions.create(**api_request)
                
                # 解析响应
                return self._parse_response(response, start_time)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM API call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    break
        
        # 所有重试都失败了
        raise LLMError(
            f"LLM API call failed after {self.config.max_retries + 1} attempts: {last_exception}",
            model_name=self.config.model_name,
            prompt_type=request.metadata.get('prompt_type'),
            token_count=self.estimate_request_tokens(request)
        )
    
    def _build_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """构建API请求参数
        
        Args:
            request: LLM请求
            
        Returns:
            API请求参数字典
        """
        # 构建消息列表
        messages = []
        
        # 添加系统提示
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # 添加用户消息
        messages.extend(request.messages)
        
        # 构建基础请求参数
        api_request = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": request.temperature or self.config.temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
        }
        
        # 添加函数调用支持
        if request.functions:
            api_request["functions"] = request.functions
            api_request["function_call"] = "auto"
        
        return api_request
    
    def _parse_response(self, response, start_time: float) -> LLMResponse:
        """解析API响应
        
        Args:
            response: OpenAI API响应
            start_time: 请求开始时间
            
        Returns:
            解析后的LLM响应
        """
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 提取token使用信息
        tokens_used = 0
        if hasattr(response, 'usage') and response.usage:
            tokens_used = response.usage.total_tokens
        
        return LLMResponse(
            content=content,
            tokens_used=tokens_used,
            model=response.model,
            finish_reason=choice.finish_reason,
            response_time=response_time,
            metadata={
                "response_id": response.id,
                "created": response.created,
                "function_call": getattr(choice.message, 'function_call', None)
            }
        )


class LLMClientFactory:
    """LLM客户端工厂类
    
    根据配置创建相应的LLM客户端实例。
    """
    
    _clients = {
        "openai": OpenAIClient,
        # 可以在这里添加其他提供商的客户端
        # "anthropic": AnthropicClient,
        # "azure": AzureOpenAIClient,
    }
    
    @classmethod
    def create_client(cls, config: LLMConfig) -> BaseLLMClient:
        """创建LLM客户端
        
        Args:
            config: LLM配置
            
        Returns:
            LLM客户端实例
            
        Raises:
            LLMError: 当不支持的提供商时
        """
        provider = config.provider.lower()
        
        if provider not in cls._clients:
            raise LLMError(
                f"Unsupported LLM provider: {provider}. "
                f"Available providers: {list(cls._clients.keys())}",
                model_name=config.model_name
            )
        
        client_class = cls._clients[provider]
        return client_class(config)
    
    @classmethod
    def register_client(cls, provider: str, client_class: type):
        """注册新的LLM客户端类
        
        Args:
            provider: 提供商名称
            client_class: 客户端类
        """
        if not issubclass(client_class, BaseLLMClient):
            raise ValueError("Client class must inherit from BaseLLMClient")
        
        cls._clients[provider.lower()] = client_class


class LLMManager:
    """LLM管理器
    
    提供高级的LLM操作接口，包括批量处理、并发控制等功能。
    """
    
    def __init__(self, config: LLMConfig):
        """初始化LLM管理器
        
        Args:
            config: LLM配置
        """
        self.config = config
        self.client = LLMClientFactory.create_client(config)
        self._semaphore = asyncio.Semaphore(10)  # 默认最大并发数
    
    async def generate_async(
        self, 
        request: LLMRequest,
        timeout: Optional[float] = None
    ) -> LLMResponse:
        """异步生成单个响应
        
        Args:
            request: LLM请求
            timeout: 超时时间(秒)
            
        Returns:
            LLM响应
        """
        timeout = timeout or self.config.request_timeout
        
        try:
            return await asyncio.wait_for(
                self.client.generate_async(request),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise GraphRAGTimeoutError(
                f"LLM request timeout after {timeout} seconds",
                timeout_seconds=timeout,
                operation="llm_generation"
            )
    
    async def generate_batch_async(
        self, 
        requests: List[LLMRequest],
        max_concurrent: Optional[int] = None
    ) -> List[LLMResponse]:
        """异步批量生成响应
        
        Args:
            requests: LLM请求列表
            max_concurrent: 最大并发数
            
        Returns:
            LLM响应列表
        """
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)
        else:
            semaphore = self._semaphore
        
        async def _generate_with_semaphore(request: LLMRequest) -> LLMResponse:
            """带信号量的生成函数"""
            async with semaphore:
                return await self.generate_async(request)
        
        # 并发执行所有请求
        tasks = [_generate_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """同步生成响应
        
        Args:
            request: LLM请求
            
        Returns:
            LLM响应
        """
        return self.client.generate_sync(request)
    
    def count_tokens(self, text: str) -> int:
        """计算文本token数量
        
        Args:
            text: 输入文本
            
        Returns:
            token数量
        """
        return self.client.count_tokens(text)
    
    def validate_request(self, request: LLMRequest) -> bool:
        """验证请求的有效性
        
        Args:
            request: LLM请求
            
        Returns:
            请求是否有效
        """
        # 检查token限制
        estimated_tokens = self.client.estimate_request_tokens(request)
        max_tokens = request.max_tokens or self.config.max_tokens
        
        if estimated_tokens > max_tokens:
            logger.warning(
                f"Request tokens ({estimated_tokens}) exceed limit ({max_tokens})"
            )
            return False
        
        # 检查消息格式
        for message in request.messages:
            if not isinstance(message, dict) or 'content' not in message:
                logger.warning("Invalid message format in request")
                return False
        
        return True