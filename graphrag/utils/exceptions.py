"""
异常处理模块

定义GraphRAG系统中使用的所有自定义异常类。
"""

from typing import Optional, Any, Dict


class GraphRAGException(Exception):
    """GraphRAG基础异常类
    
    所有GraphRAG特定异常的基类，提供统一的异常处理接口。
    """
    
    def __init__(
        self, 
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            context: 额外的上下文信息
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GRAPHRAG_ERROR"
        self.context = context or {}
    
    def __str__(self) -> str:
        """返回异常的字符串表示"""
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


class ConfigurationError(GraphRAGException):
    """配置错误异常
    
    当配置参数无效或缺失时抛出。
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        """初始化配置错误
        
        Args:
            message: 错误消息
            config_key: 相关的配置键
        """
        context = {"config_key": config_key} if config_key else {}
        super().__init__(message, "CONFIG_ERROR", context)


class DocumentProcessingError(GraphRAGException):
    """文档处理错误异常
    
    当文档处理过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, document_path: Optional[str] = None):
        """初始化文档处理错误
        
        Args:
            message: 错误消息
            document_path: 相关的文档路径
        """
        context = {"document_path": document_path} if document_path else {}
        super().__init__(message, "DOC_PROCESSING_ERROR", context)


class ExtractionError(GraphRAGException):
    """实体关系提取错误异常
    
    当实体、关系或主张提取失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        chunk_id: Optional[str] = None,
        extraction_type: Optional[str] = None
    ):
        """初始化提取错误
        
        Args:
            message: 错误消息
            chunk_id: 相关的文档块ID
            extraction_type: 提取类型(entity/relationship/claim)
        """
        context = {}
        if chunk_id:
            context["chunk_id"] = chunk_id
        if extraction_type:
            context["extraction_type"] = extraction_type
        
        super().__init__(message, "EXTRACTION_ERROR", context)


class KnowledgeGraphError(GraphRAGException):
    """知识图构建错误异常
    
    当知识图构建过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, operation: Optional[str] = None):
        """初始化知识图错误
        
        Args:
            message: 错误消息
            operation: 相关的操作类型
        """
        context = {"operation": operation} if operation else {}
        super().__init__(message, "KNOWLEDGE_GRAPH_ERROR", context)


class CommunityDetectionError(GraphRAGException):
    """社区检测错误异常
    
    当社区检测算法失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        algorithm: Optional[str] = None,
        level: Optional[int] = None
    ):
        """初始化社区检测错误
        
        Args:
            message: 错误消息
            algorithm: 使用的算法
            level: 相关的层级
        """
        context = {}
        if algorithm:
            context["algorithm"] = algorithm
        if level is not None:
            context["level"] = level
        
        super().__init__(message, "COMMUNITY_DETECTION_ERROR", context)


class SummarizationError(GraphRAGException):
    """摘要生成错误异常
    
    当社区摘要生成失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        community_id: Optional[str] = None,
        level: Optional[int] = None
    ):
        """初始化摘要生成错误
        
        Args:
            message: 错误消息
            community_id: 相关的社区ID
            level: 相关的层级
        """
        context = {}
        if community_id:
            context["community_id"] = community_id
        if level is not None:
            context["level"] = level
        
        super().__init__(message, "SUMMARIZATION_ERROR", context)


class QueryProcessingError(GraphRAGException):
    """查询处理错误异常
    
    当查询处理过程中发生错误时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        query: Optional[str] = None,
        stage: Optional[str] = None
    ):
        """初始化查询处理错误
        
        Args:
            message: 错误消息
            query: 相关的查询
            stage: 处理阶段(map/reduce)
        """
        context = {}
        if query:
            context["query"] = query
        if stage:
            context["stage"] = stage
        
        super().__init__(message, "QUERY_PROCESSING_ERROR", context)


class LLMError(GraphRAGException):
    """LLM调用错误异常
    
    当大语言模型调用失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        prompt_type: Optional[str] = None,
        token_count: Optional[int] = None
    ):
        """初始化LLM错误
        
        Args:
            message: 错误消息
            model_name: 模型名称
            prompt_type: 提示类型
            token_count: token数量
        """
        context = {}
        if model_name:
            context["model_name"] = model_name
        if prompt_type:
            context["prompt_type"] = prompt_type
        if token_count:
            context["token_count"] = token_count
        
        super().__init__(message, "LLM_ERROR", context)


class StorageError(GraphRAGException):
    """存储错误异常
    
    当数据存储操作失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        storage_type: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """初始化存储错误
        
        Args:
            message: 错误消息
            storage_type: 存储类型
            operation: 操作类型(read/write/delete)
        """
        context = {}
        if storage_type:
            context["storage_type"] = storage_type
        if operation:
            context["operation"] = operation
        
        super().__init__(message, "STORAGE_ERROR", context)


class ValidationError(GraphRAGException):
    """验证错误异常
    
    当数据验证失败时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        field_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None
    ):
        """初始化验证错误
        
        Args:
            message: 错误消息
            field_name: 字段名称
            expected_type: 期望的类型
            actual_value: 实际值
        """
        context = {}
        if field_name:
            context["field_name"] = field_name
        if expected_type:
            context["expected_type"] = expected_type
        if actual_value is not None:
            context["actual_value"] = str(actual_value)
        
        super().__init__(message, "VALIDATION_ERROR", context)


class TimeoutError(GraphRAGException):
    """超时错误异常
    
    当操作超时时抛出。
    """
    
    def __init__(
        self, 
        message: str, 
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None
    ):
        """初始化超时错误
        
        Args:
            message: 错误消息
            timeout_seconds: 超时时间(秒)
            operation: 相关操作
        """
        context = {}
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        if operation:
            context["operation"] = operation
        
        super().__init__(message, "TIMEOUT_ERROR", context)


# 异常处理工具函数

def handle_graphrag_exception(exception: Exception) -> GraphRAGException:
    """将通用异常转换为GraphRAG异常
    
    Args:
        exception: 原始异常
        
    Returns:
        转换后的GraphRAG异常
    """
    if isinstance(exception, GraphRAGException):
        return exception
    
    # 根据异常类型进行转换
    if isinstance(exception, FileNotFoundError):
        return DocumentProcessingError(
            f"File not found: {str(exception)}",
            document_path=getattr(exception, 'filename', None)
        )
    elif isinstance(exception, ValueError):
        return ValidationError(f"Value error: {str(exception)}")
    elif isinstance(exception, TimeoutError):
        return TimeoutError(f"Operation timeout: {str(exception)}")
    else:
        return GraphRAGException(
            f"Unexpected error: {str(exception)}",
            error_code="UNKNOWN_ERROR",
            context={"original_exception": type(exception).__name__}
        )


def log_exception(exception: GraphRAGException, logger) -> None:
    """记录异常信息
    
    Args:
        exception: GraphRAG异常
        logger: 日志记录器
    """
    logger.error(
        f"GraphRAG Exception: {exception}",
        extra={
            "error_code": exception.error_code,
            "context": exception.context
        }
    )