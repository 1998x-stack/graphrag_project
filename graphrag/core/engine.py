"""
GraphRAG核心引擎模块

整合所有组件，提供完整的GraphRAG管道执行功能。
"""

import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import logging

from ..config.settings import GraphRAGConfig
from ..core.llm_client import LLMManager
from ..document_processing.document_processor import DocumentProcessor, DocumentChunk
from ..knowledge_graph.extraction import ExtractionManager, ExtractionResult
from ..knowledge_graph.graph_builder import KnowledgeGraphBuilder, KnowledgeGraph
from ..community_detection.community_detector import CommunityDetectionManager, CommunityStructure
from ..core.summarizer import SummarizationManager, CommunitySummary
from ..query_processing.query_processor import QueryProcessor, GlobalAnswer
from ..utils.exceptions import GraphRAGException

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGIndex:
    """GraphRAG索引数据类
    
    封装构建完成的GraphRAG索引。
    """
    
    knowledge_graph: KnowledgeGraph  # 知识图
    community_structure: CommunityStructure  # 社区结构
    community_summaries: Dict[str, CommunitySummary]  # 社区摘要
    extraction_results: List[ExtractionResult]  # 提取结果
    index_metadata: Dict[str, Any]  # 索引元数据
    
    def save_to_directory(self, output_dir: Union[str, Path]):
        """保存索引到目录
        
        Args:
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存知识图
        kg_path = output_dir / "knowledge_graph.json"
        self.knowledge_graph.save_to_file(str(kg_path))
        
        # 保存社区结构
        community_path = output_dir / "community_structure.json"
        with open(community_path, 'w', encoding='utf-8') as f:
            json.dump(self.community_structure.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 保存社区摘要
        summaries_path = output_dir / "community_summaries.json"
        summaries_data = {
            comm_id: summary.to_dict() 
            for comm_id, summary in self.community_summaries.items()
        }
        with open(summaries_path, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, ensure_ascii=False, indent=2)
        
        # 保存提取结果
        extraction_path = output_dir / "extraction_results.json"
        extraction_data = []
        for result in self.extraction_results:
            extraction_data.append({
                "chunk_id": result.chunk_id,
                "entities": [entity.to_dict() for entity in result.entities],
                "relationships": [rel.to_dict() for rel in result.relationships],
                "claims": [claim.to_dict() for claim in result.claims],
                "metadata": result.extraction_metadata
            })
        with open(extraction_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=2)
        
        # 保存索引元数据
        metadata_path = output_dir / "index_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"GraphRAG index saved to {output_dir}")
    
    @classmethod
    def load_from_directory(cls, input_dir: Union[str, Path]) -> 'GraphRAGIndex':
        """从目录加载索引
        
        Args:
            input_dir: 输入目录路径
            
        Returns:
            GraphRAG索引实例
        """
        input_dir = Path(input_dir)
        
        # 加载知识图
        kg_path = input_dir / "knowledge_graph.json"
        knowledge_graph = KnowledgeGraph.load_from_file(str(kg_path))
        
        # 加载社区结构（简化实现）
        community_path = input_dir / "community_structure.json"
        with open(community_path, 'r', encoding='utf-8') as f:
            community_data = json.load(f)
        # 注意：这里需要完整的社区结构重建逻辑
        community_structure = CommunityStructure()  # 简化处理
        
        # 加载社区摘要
        summaries_path = input_dir / "community_summaries.json"
        with open(summaries_path, 'r', encoding='utf-8') as f:
            summaries_data = json.load(f)
        
        community_summaries = {}
        for comm_id, summary_data in summaries_data.items():
            summary = CommunitySummary(
                community_id=summary_data["community_id"],
                level=summary_data["level"],
                title=summary_data["title"],
                summary=summary_data["summary"],
                rating=summary_data["rating"],
                rating_explanation=summary_data["rating_explanation"],
                findings=summary_data["findings"],
                full_content=summary_data["full_content"],
                num_nodes=summary_data["num_nodes"],
                token_count=summary_data["token_count"],
                metadata=summary_data["metadata"]
            )
            community_summaries[comm_id] = summary
        
        # 加载提取结果（简化处理）
        extraction_results = []
        
        # 加载索引元数据
        metadata_path = input_dir / "index_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            index_metadata = json.load(f)
        
        return cls(
            knowledge_graph=knowledge_graph,
            community_structure=community_structure,
            community_summaries=community_summaries,
            extraction_results=extraction_results,
            index_metadata=index_metadata
        )


class GraphRAGEngine:
    """GraphRAG引擎主类
    
    提供完整的GraphRAG功能，包括索引构建和查询处理。
    """
    
    def __init__(self, config: GraphRAGConfig):
        """初始化GraphRAG引擎
        
        Args:
            config: GraphRAG配置
        """
        self.config = config
        
        # 验证配置
        if not config.validate_config():
            raise GraphRAGException("Invalid configuration")
        
        # 初始化核心组件
        self.llm_manager = LLMManager(config.llm)
        self.document_processor = DocumentProcessor(config.document_processing)
        self.extraction_manager = ExtractionManager(self.llm_manager, config.extraction)
        self.graph_builder = KnowledgeGraphBuilder(self.llm_manager, config)
        self.community_detector = CommunityDetectionManager(config.community_detection)
        self.summarization_manager = SummarizationManager(self.llm_manager, config.summarization)
        self.query_processor = QueryProcessor(self.llm_manager, config.query_processing)
        
        # 索引缓存
        self._current_index: Optional[GraphRAGIndex] = None
        
        logger.info("GraphRAG engine initialized successfully")
    
    async def build_index_async(
        self, 
        documents: Union[List[str], str],
        output_dir: Optional[Union[str, Path]] = None
    ) -> GraphRAGIndex:
        """异步构建GraphRAG索引
        
        Args:
            documents: 文档列表或目录路径
            output_dir: 输出目录（可选）
            
        Returns:
            构建的GraphRAG索引
        """
        try:
            start_time = time.time()
            logger.info("Starting GraphRAG index construction")
            
            # 阶段1：文档处理
            logger.info("Stage 1: Document processing")
            stage_start = time.time()
            
            if isinstance(documents, str):
                # 处理目录
                chunks_dict = self.document_processor.process_directory(documents)
                all_chunks = []
                for chunks in chunks_dict.values():
                    all_chunks.extend(chunks)
            else:
                # 处理文件列表
                all_chunks = []
                for doc_path in documents:
                    chunks = self.document_processor.process_file(doc_path)
                    all_chunks.extend(chunks)
            
            logger.info(f"Document processing completed: {len(all_chunks)} chunks in {time.time() - stage_start:.1f}s")
            
            # 阶段2：实体关系提取
            logger.info("Stage 2: Entity and relationship extraction")
            stage_start = time.time()
            
            extraction_results = await self.extraction_manager.extract_chunks_async(all_chunks)
            
            logger.info(f"Extraction completed: {len(extraction_results)} results in {time.time() - stage_start:.1f}s")
            
            # 阶段3：知识图构建
            logger.info("Stage 3: Knowledge graph construction")
            stage_start = time.time()
            
            knowledge_graph = await self.graph_builder.build_graph_async(extraction_results)
            
            logger.info(f"Knowledge graph built in {time.time() - stage_start:.1f}s")
            
            # 阶段4：社区检测
            logger.info("Stage 4: Community detection")
            stage_start = time.time()
            
            community_structure = self.community_detector.detect_communities(knowledge_graph)
            
            logger.info(f"Community detection completed in {time.time() - stage_start:.1f}s")
            
            # 阶段5：社区摘要生成
            logger.info("Stage 5: Community summarization")
            stage_start = time.time()
            
            community_summaries = await self.summarization_manager.generate_all_summaries_async(
                community_structure, knowledge_graph
            )
            
            logger.info(f"Community summarization completed in {time.time() - stage_start:.1f}s")
            
            # 创建索引对象
            index = GraphRAGIndex(
                knowledge_graph=knowledge_graph,
                community_structure=community_structure,
                community_summaries=community_summaries,
                extraction_results=extraction_results,
                index_metadata={
                    "created_at": str(pd.Timestamp.now()),
                    "total_build_time": time.time() - start_time,
                    "num_documents": len(documents) if isinstance(documents, list) else "directory",
                    "num_chunks": len(all_chunks),
                    "num_entities": len(knowledge_graph.nodes),
                    "num_relationships": len(knowledge_graph.edges),
                    "num_communities": community_structure.total_communities,
                    "config": self.config.dict()
                }
            )
            
            # 保存索引
            if output_dir:
                index.save_to_directory(output_dir)
            
            # 缓存索引
            self._current_index = index
            
            total_time = time.time() - start_time
            logger.info(f"GraphRAG index construction completed in {total_time:.1f}s")
            
            return index
            
        except Exception as e:
            raise GraphRAGException(f"Index construction failed: {e}")
    
    def build_index_sync(
        self, 
        documents: Union[List[str], str],
        output_dir: Optional[Union[str, Path]] = None
    ) -> GraphRAGIndex:
        """同步构建GraphRAG索引"""
        try:
            start_time = time.time()
            logger.info("Starting GraphRAG index construction")
            
            # 阶段1：文档处理
            logger.info("Stage 1: Document processing")
            stage_start = time.time()
            
            if isinstance(documents, str):
                chunks_dict = self.document_processor.process_directory(documents)
                all_chunks = []
                for chunks in chunks_dict.values():
                    all_chunks.extend(chunks)
            else:
                all_chunks = []
                for doc_path in documents:
                    chunks = self.document_processor.process_file(doc_path)
                    all_chunks.extend(chunks)
            
            logger.info(f"Document processing completed: {len(all_chunks)} chunks in {time.time() - stage_start:.1f}s")
            
            # 阶段2：实体关系提取
            logger.info("Stage 2: Entity and relationship extraction")
            stage_start = time.time()
            
            extraction_results = []
            for chunk in all_chunks:
                try:
                    result = self.extraction_manager.extract_chunk_sync(chunk)
                    extraction_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to extract from chunk {chunk.chunk_id}: {e}")
                    continue
            
            logger.info(f"Extraction completed: {len(extraction_results)} results in {time.time() - stage_start:.1f}s")
            
            # 阶段3：知识图构建
            logger.info("Stage 3: Knowledge graph construction")
            stage_start = time.time()
            
            knowledge_graph = self.graph_builder.build_graph_sync(extraction_results)
            
            logger.info(f"Knowledge graph built in {time.time() - stage_start:.1f}s")
            
            # 阶段4：社区检测
            logger.info("Stage 4: Community detection")
            stage_start = time.time()
            
            community_structure = self.community_detector.detect_communities(knowledge_graph)
            
            logger.info(f"Community detection completed in {time.time() - stage_start:.1f}s")
            
            # 阶段5：社区摘要生成
            logger.info("Stage 5: Community summarization")
            stage_start = time.time()
            
            community_summaries = self.summarization_manager.generate_all_summaries_sync(
                community_structure, knowledge_graph
            )
            
            logger.info(f"Community summarization completed in {time.time() - stage_start:.1f}s")
            
            # 创建索引对象
            index = GraphRAGIndex(
                knowledge_graph=knowledge_graph,
                community_structure=community_structure,
                community_summaries=community_summaries,
                extraction_results=extraction_results,
                index_metadata={
                    "created_at": str(pd.Timestamp.now()),
                    "total_build_time": time.time() - start_time,
                    "num_documents": len(documents) if isinstance(documents, list) else "directory",
                    "num_chunks": len(all_chunks),
                    "num_entities": len(knowledge_graph.nodes),
                    "num_relationships": len(knowledge_graph.edges),
                    "num_communities": community_structure.total_communities,
                    "config": self.config.dict()
                }
            )
            
            # 保存索引
            if output_dir:
                index.save_to_directory(output_dir)
            
            # 缓存索引
            self._current_index = index
            
            total_time = time.time() - start_time
            logger.info(f"GraphRAG index construction completed in {total_time:.1f}s")
            
            return index
            
        except Exception as e:
            raise GraphRAGException(f"Index construction failed: {e}")
    
    def load_index(self, index_dir: Union[str, Path]) -> GraphRAGIndex:
        """加载已构建的索引
        
        Args:
            index_dir: 索引目录路径
            
        Returns:
            加载的GraphRAG索引
        """
        try:
            index = GraphRAGIndex.load_from_directory(index_dir)
            self._current_index = index
            logger.info(f"GraphRAG index loaded from {index_dir}")
            return index
            
        except Exception as e:
            raise GraphRAGException(f"Failed to load index: {e}")
    
    async def query_async(
        self, 
        query: str,
        community_level: int = 0,
        answer_type: str = "comprehensive",
        index: Optional[GraphRAGIndex] = None
    ) -> GlobalAnswer:
        """异步查询GraphRAG索引
        
        Args:
            query: 用户查询
            community_level: 使用的社区层级
            answer_type: 答案类型
            index: 使用的索引（可选，默认使用当前索引）
            
        Returns:
            全局答案
        """
        try:
            # 确定使用的索引
            current_index = index or self._current_index
            if not current_index:
                raise GraphRAGException("No index available. Please build or load an index first.")
            
            # 获取指定层级的社区摘要
            community_summaries = self.summarization_manager.get_summaries_by_level(
                current_index.community_summaries, community_level
            )
            
            if not community_summaries:
                # 尝试使用叶子社区
                community_summaries = list(current_index.community_summaries.values())
                logger.warning(f"No summaries found for level {community_level}, using all available summaries")
            
            # 处理查询
            answer = await self.query_processor.process_query_async(
                query, community_summaries, answer_type
            )
            
            logger.info(f"Query processed successfully: {len(answer.used_communities)} communities used")
            return answer
            
        except Exception as e:
            raise GraphRAGException(f"Query processing failed: {e}")
    
    def query_sync(
        self, 
        query: str,
        community_level: int = 0,
        answer_type: str = "comprehensive",
        index: Optional[GraphRAGIndex] = None
    ) -> GlobalAnswer:
        """同步查询GraphRAG索引"""
        try:
            # 确定使用的索引
            current_index = index or self._current_index
            if not current_index:
                raise GraphRAGException("No index available. Please build or load an index first.")
            
            # 获取指定层级的社区摘要
            community_summaries = self.summarization_manager.get_summaries_by_level(
                current_index.community_summaries, community_level
            )
            
            if not community_summaries:
                community_summaries = list(current_index.community_summaries.values())
                logger.warning(f"No summaries found for level {community_level}, using all available summaries")
            
            # 处理查询
            answer = self.query_processor.process_query_sync(
                query, community_summaries, answer_type
            )
            
            logger.info(f"Query processed successfully: {len(answer.used_communities)} communities used")
            return answer
            
        except Exception as e:
            raise GraphRAGException(f"Query processing failed: {e}")
    
    def get_index_statistics(self, index: Optional[GraphRAGIndex] = None) -> Dict[str, Any]:
        """获取索引统计信息
        
        Args:
            index: 索引对象（可选）
            
        Returns:
            统计信息字典
        """
        current_index = index or self._current_index
        if not current_index:
            return {"error": "No index available"}
        
        kg_stats = current_index.knowledge_graph.get_statistics()
        community_stats = current_index.community_structure.get_statistics()
        
        return {
            "index_metadata": current_index.index_metadata,
            "knowledge_graph": kg_stats,
            "community_structure": community_stats,
            "summaries": {
                "total_summaries": len(current_index.community_summaries),
                "total_tokens": sum(
                    summary.token_count 
                    for summary in current_index.community_summaries.values()
                ),
                "avg_rating": sum(
                    summary.rating 
                    for summary in current_index.community_summaries.values()
                ) / len(current_index.community_summaries) if current_index.community_summaries else 0
            }
        }
    
    def validate_index(self, index: Optional[GraphRAGIndex] = None) -> bool:
        """验证索引的完整性
        
        Args:
            index: 索引对象（可选）
            
        Returns:
            索引是否有效
        """
        current_index = index or self._current_index
        if not current_index:
            logger.error("No index to validate")
            return False
        
        try:
            # 验证知识图
            if not current_index.knowledge_graph.nodes:
                logger.error("Knowledge graph has no nodes")
                return False
            
            # 验证社区结构
            if not self.community_detector.validate_communities(current_index.community_structure):
                logger.error("Community structure validation failed")
                return False
            
            # 验证摘要
            if not current_index.community_summaries:
                logger.error("No community summaries available")
                return False
            
            logger.info("Index validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False
    
    def update_config(self, new_config: GraphRAGConfig):
        """更新引擎配置
        
        Args:
            new_config: 新配置
        """
        if not new_config.validate_config():
            raise GraphRAGException("Invalid new configuration")
        
        self.config = new_config
        
        # 重新初始化需要配置的组件
        self.llm_manager = LLMManager(new_config.llm)
        self.document_processor = DocumentProcessor(new_config.document_processing)
        self.extraction_manager = ExtractionManager(self.llm_manager, new_config.extraction)
        self.graph_builder = KnowledgeGraphBuilder(self.llm_manager, new_config)
        self.community_detector = CommunityDetectionManager(new_config.community_detection)
        self.summarization_manager = SummarizationManager(self.llm_manager, new_config.summarization)
        self.query_processor = QueryProcessor(self.llm_manager, new_config.query_processing)
        
        logger.info("GraphRAG engine configuration updated")