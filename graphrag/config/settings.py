"""
配置管理模块

该模块定义了GraphRAG系统的所有配置选项，包括模型参数、处理选项和系统设置。
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path
import yaml
import os
from dataclasses import dataclass

class LLMConfig(BaseModel):
    """LLM配置类
    
    定义大语言模型的相关配置参数，包括模型选择、API配置等。
    """
    
    provider: str = Field(default="openai", description="LLM服务提供商")
    model_name: str = Field(default="gpt-4-turbo", description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    api_base: Optional[str] = Field(default=None, description="API基础URL")
    temperature: float = Field(default=0.0, description="生成温度")
    max_tokens: int = Field(default=8000, description="最大token数")
    request_timeout: int = Field(default=60, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟(秒)")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """验证温度参数范围"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        """验证最大token数"""
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        return v


class DocumentProcessingConfig(BaseModel):
    """文档处理配置类
    
    定义文档切块、重叠等处理参数。
    """
    
    chunk_size: int = Field(default=600, description="文档切块大小(tokens)")
    chunk_overlap: int = Field(default=100, description="切块重叠大小(tokens)")
    encoding_name: str = Field(default="cl100k_base", description="编码方式")
    supported_formats: List[str] = Field(
        default=["txt", "md", "pdf", "docx", "html"],
        description="支持的文档格式"
    )
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        """验证重叠大小不能超过切块大小"""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class ExtractionConfig(BaseModel):
    """实体关系提取配置类
    
    定义实体、关系、主张提取的相关参数。
    """
    
    entity_types: List[str] = Field(
        default=["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"],
        description="实体类型列表"
    )
    max_gleaning_iterations: int = Field(default=3, description="自反思最大迭代次数")
    enable_claim_extraction: bool = Field(default=True, description="是否启用主张提取")
    extraction_prompt_template: Optional[str] = Field(default=None, description="提取提示模板")
    claim_prompt_template: Optional[str] = Field(default=None, description="主张提取提示模板")
    
    # Few-shot示例配置
    few_shot_examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Few-shot学习示例"
    )


class CommunityDetectionConfig(BaseModel):
    """社区检测配置类
    
    定义图社区检测算法的相关参数。
    """
    
    algorithm: str = Field(default="leiden", description="社区检测算法")
    resolution: float = Field(default=1.0, description="分辨率参数")
    max_levels: int = Field(default=4, description="最大层级数")
    min_community_size: int = Field(default=3, description="最小社区大小")
    random_seed: Optional[int] = Field(default=42, description="随机种子")
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        """验证算法选择"""
        if v not in ['leiden', 'louvain']:
            raise ValueError("Algorithm must be 'leiden' or 'louvain'")
        return v


class SummarizationConfig(BaseModel):
    """摘要生成配置类
    
    定义社区摘要生成的相关参数。
    """
    
    max_summary_tokens: int = Field(default=2000, description="单个摘要最大token数")
    summary_prompt_template: Optional[str] = Field(default=None, description="摘要提示模板")
    prioritize_by_degree: bool = Field(default=True, description="是否按节点度优先排序")
    include_claims: bool = Field(default=True, description="是否包含主张信息")


class QueryProcessingConfig(BaseModel):
    """查询处理配置类
    
    定义查询处理和答案生成的相关参数。
    """
    
    max_context_tokens: int = Field(default=8000, description="最大上下文token数")
    min_helpfulness_score: float = Field(default=0.0, description="最小有用度分数阈值")
    map_prompt_template: Optional[str] = Field(default=None, description="Map阶段提示模板")
    reduce_prompt_template: Optional[str] = Field(default=None, description="Reduce阶段提示模板")
    enable_parallel_processing: bool = Field(default=True, description="是否启用并行处理")
    max_concurrent_tasks: int = Field(default=10, description="最大并发任务数")


class StorageConfig(BaseModel):
    """存储配置类
    
    定义数据存储的相关配置。
    """
    
    storage_type: str = Field(default="local", description="存储类型")
    base_path: Path = Field(default=Path("./graphrag_data"), description="基础存储路径")
    cache_enabled: bool = Field(default=True, description="是否启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    # 数据库配置(可选)
    database_url: Optional[str] = Field(default=None, description="数据库连接URL")
    
    @validator('storage_type')
    def validate_storage_type(cls, v):
        """验证存储类型"""
        if v not in ['local', 'database', 'redis']:
            raise ValueError("Storage type must be 'local', 'database', or 'redis'")
        return v


class GraphRAGConfig(BaseModel):
    """GraphRAG主配置类
    
    集成所有子配置，提供统一的配置管理接口。
    """
    
    # 子配置模块
    llm: LLMConfig = Field(default_factory=LLMConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    community_detection: CommunityDetectionConfig = Field(default_factory=CommunityDetectionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    query_processing: QueryProcessingConfig = Field(default_factory=QueryProcessingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # 全局配置
    debug_mode: bool = Field(default=False, description="是否启用调试模式")
    log_level: str = Field(default="INFO", description="日志级别")
    enable_telemetry: bool = Field(default=False, description="是否启用遥测")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "GraphRAGConfig":
        """从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            GraphRAGConfig实例
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    @classmethod
    def from_env(cls) -> "GraphRAGConfig":
        """从环境变量加载配置
        
        Returns:
            GraphRAGConfig实例
        """
        config = cls()
        
        # LLM配置
        if os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_API_BASE"):
            config.llm.api_base = os.getenv("OPENAI_API_BASE")
        if os.getenv("LLM_MODEL"):
            config.llm.model_name = os.getenv("LLM_MODEL")
        
        # 其他环境变量配置...
        
        return config
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """导出配置到YAML文件
        
        Args:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, allow_unicode=True)
    
    def validate_config(self) -> bool:
        """验证配置的完整性和有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 验证API密钥是否设置
            if not self.llm.api_key:
                raise ValueError("LLM API key is required")
            
            # 验证存储路径
            if self.storage.storage_type == "local":
                self.storage.base_path.mkdir(parents=True, exist_ok=True)
            
            # 验证token限制的一致性
            if self.query_processing.max_context_tokens > self.llm.max_tokens:
                raise ValueError("Query context tokens cannot exceed LLM max tokens")
            
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"Configuration validation failed: {e}")
            return False


# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> GraphRAGConfig:
    """加载配置的便捷函数
    
    Args:
        config_path: 配置文件路径，如果为None则从环境变量加载
        
    Returns:
        GraphRAGConfig实例
    """
    if config_path:
        return GraphRAGConfig.from_yaml(config_path)
    else:
        return GraphRAGConfig.from_env()