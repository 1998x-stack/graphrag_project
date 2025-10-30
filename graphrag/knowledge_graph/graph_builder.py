"""
知识图构建模块

将提取的实体、关系和主张聚合成统一的知识图结构。
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
import json
import logging

from .extraction import Entity, Relationship, Claim, ExtractionResult
from ..core.llm_client import LLMManager, LLMRequest
from ..config.settings import GraphRAGConfig
from ..utils.exceptions import KnowledgeGraphError

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphNode:
    """知识图节点类
    
    表示知识图中的一个实体节点。
    """
    
    name: str  # 实体名称(唯一标识)
    entity_type: str  # 实体类型
    description: str  # 合并后的描述
    degree: int = 0  # 节点度数
    source_chunks: Set[str] = field(default_factory=set)  # 来源文档块
    raw_entities: List[Entity] = field(default_factory=list)  # 原始实体列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def add_entity(self, entity: Entity):
        """添加原始实体到节点"""
        self.raw_entities.append(entity)
        self.source_chunks.add(entity.source_chunk_id)
        
        # 更新实体类型(选择最常见的类型)
        type_counts = Counter(e.entity_type for e in self.raw_entities)
        self.entity_type = type_counts.most_common(1)[0][0]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "degree": self.degree,
            "source_chunks": list(self.source_chunks),
            "metadata": self.metadata
        }


@dataclass
class KnowledgeGraphEdge:
    """知识图边类
    
    表示知识图中的一个关系边。
    """
    
    source: str  # 源节点名称
    target: str  # 目标节点名称
    relationship_type: str  # 关系类型
    description: str  # 合并后的描述
    weight: float  # 边权重(基于出现频次和强度)
    strength: float  # 关系强度
    source_chunks: Set[str] = field(default_factory=set)  # 来源文档块
    raw_relationships: List[Relationship] = field(default_factory=list)  # 原始关系列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def add_relationship(self, relationship: Relationship):
        """添加原始关系到边"""
        self.raw_relationships.append(relationship)
        self.source_chunks.add(relationship.source_chunk_id)
        
        # 更新权重和强度
        self.weight = len(self.raw_relationships)
        self.strength = sum(r.strength for r in self.raw_relationships) / len(self.raw_relationships)
        
        # 更新关系类型(选择最常见的类型)
        type_counts = Counter(r.relationship_type for r in self.raw_relationships)
        self.relationship_type = type_counts.most_common(1)[0][0]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relationship_type,
            "description": self.description,
            "weight": self.weight,
            "strength": self.strength,
            "source_chunks": list(self.source_chunks),
            "metadata": self.metadata
        }


@dataclass
class KnowledgeGraphClaim:
    """知识图主张类
    
    表示知识图中的事实性陈述。
    """
    
    subject: str  # 主语实体
    predicate: str  # 谓语
    object: str  # 宾语
    description: str  # 描述
    source_chunks: Set[str] = field(default_factory=set)  # 来源文档块
    raw_claims: List[Claim] = field(default_factory=list)  # 原始主张列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def add_claim(self, claim: Claim):
        """添加原始主张"""
        self.raw_claims.append(claim)
        self.source_chunks.add(claim.source_chunk_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "source_chunks": list(self.source_chunks),
            "metadata": self.metadata
        }


class KnowledgeGraph:
    """知识图类
    
    表示完整的知识图结构。
    """
    
    def __init__(self):
        """初始化知识图"""
        self.nodes: Dict[str, KnowledgeGraphNode] = {}  # 节点字典
        self.edges: Dict[Tuple[str, str], KnowledgeGraphEdge] = {}  # 边字典
        self.claims: List[KnowledgeGraphClaim] = []  # 主张列表
        self.graph: nx.Graph = nx.Graph()  # NetworkX图对象
        self.metadata: Dict[str, Any] = {}  # 图级别元数据
    
    def add_node(self, entity: Entity) -> KnowledgeGraphNode:
        """添加或更新节点
        
        Args:
            entity: 实体对象
            
        Returns:
            知识图节点
        """
        name = entity.name
        
        if name in self.nodes:
            # 更新现有节点
            self.nodes[name].add_entity(entity)
        else:
            # 创建新节点
            node = KnowledgeGraphNode(
                name=name,
                entity_type=entity.entity_type,
                description=entity.description
            )
            node.add_entity(entity)
            self.nodes[name] = node
            
            # 添加到NetworkX图
            self.graph.add_node(name, **node.to_dict())
        
        return self.nodes[name]
    
    def add_edge(self, relationship: Relationship) -> Optional[KnowledgeGraphEdge]:
        """添加或更新边
        
        Args:
            relationship: 关系对象
            
        Returns:
            知识图边，如果源或目标节点不存在则返回None
        """
        source = relationship.source_entity
        target = relationship.target_entity
        
        # 检查节点是否存在
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"Cannot add edge {source}->{target}: missing nodes")
            return None
        
        edge_key = (source, target)
        reverse_key = (target, source)
        
        # 检查是否已有相同或反向边
        if edge_key in self.edges:
            # 更新现有边
            self.edges[edge_key].add_relationship(relationship)
            edge = self.edges[edge_key]
        elif reverse_key in self.edges:
            # 更新反向边
            self.edges[reverse_key].add_relationship(relationship)
            edge = self.edges[reverse_key]
        else:
            # 创建新边
            edge = KnowledgeGraphEdge(
                source=source,
                target=target,
                relationship_type=relationship.relationship_type,
                description=relationship.description,
                weight=1.0,
                strength=relationship.strength
            )
            edge.add_relationship(relationship)
            self.edges[edge_key] = edge
            
            # 添加到NetworkX图
            self.graph.add_edge(
                source, target, 
                weight=edge.weight,
                **edge.to_dict()
            )
        
        # 更新节点度数
        self.nodes[source].degree = self.graph.degree[source]
        self.nodes[target].degree = self.graph.degree[target]
        
        return edge
    
    def add_claim(self, claim: Claim):
        """添加主张
        
        Args:
            claim: 主张对象
        """
        # 查找是否已有相同主张
        for existing_claim in self.claims:
            if (existing_claim.subject == claim.subject and
                existing_claim.predicate == claim.predicate and
                existing_claim.object == claim.object):
                existing_claim.add_claim(claim)
                return
        
        # 创建新主张
        kg_claim = KnowledgeGraphClaim(
            subject=claim.subject,
            predicate=claim.predicate,
            object=claim.object,
            description=claim.description
        )
        kg_claim.add_claim(claim)
        self.claims.append(kg_claim)
    
    def get_node_neighbors(self, node_name: str) -> List[str]:
        """获取节点的邻居
        
        Args:
            node_name: 节点名称
            
        Returns:
            邻居节点名称列表
        """
        if node_name in self.graph:
            return list(self.graph.neighbors(node_name))
        return []
    
    def get_connected_subgraph(self, node_names: List[str]) -> nx.Graph:
        """获取指定节点的连通子图
        
        Args:
            node_names: 节点名称列表
            
        Returns:
            连通子图
        """
        return self.graph.subgraph(node_names).copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_claims": len(self.claims),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "num_connected_components": nx.number_connected_components(self.graph)
        }
        
        if self.nodes:
            degrees = [node.degree for node in self.nodes.values()]
            stats.update({
                "avg_degree": sum(degrees) / len(degrees),
                "max_degree": max(degrees),
                "min_degree": min(degrees)
            })
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "claims": [claim.to_dict() for claim in self.claims],
            "statistics": self.get_statistics(),
            "metadata": self.metadata
        }
    
    def save_to_file(self, file_path: str):
        """保存到文件
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'KnowledgeGraph':
        """从文件加载
        
        Args:
            file_path: 文件路径
            
        Returns:
            知识图实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls()
        
        # 重建节点
        for node_data in data["nodes"]:
            node = KnowledgeGraphNode(
                name=node_data["name"],
                entity_type=node_data["type"],
                description=node_data["description"],
                degree=node_data["degree"],
                source_chunks=set(node_data["source_chunks"]),
                metadata=node_data["metadata"]
            )
            kg.nodes[node.name] = node
            kg.graph.add_node(node.name, **node.to_dict())
        
        # 重建边
        for edge_data in data["edges"]:
            edge = KnowledgeGraphEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                relationship_type=edge_data["type"],
                description=edge_data["description"],
                weight=edge_data["weight"],
                strength=edge_data["strength"],
                source_chunks=set(edge_data["source_chunks"]),
                metadata=edge_data["metadata"]
            )
            kg.edges[(edge.source, edge.target)] = edge
            kg.graph.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                **edge.to_dict()
            )
        
        # 重建主张
        for claim_data in data["claims"]:
            claim = KnowledgeGraphClaim(
                subject=claim_data["subject"],
                predicate=claim_data["predicate"],
                object=claim_data["object"],
                description=claim_data["description"],
                source_chunks=set(claim_data["source_chunks"]),
                metadata=claim_data["metadata"]
            )
            kg.claims.append(claim)
        
        kg.metadata = data.get("metadata", {})
        
        return kg


class EntityLinker:
    """实体链接器
    
    负责将相似的实体名称链接到同一个节点。
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """初始化实体链接器
        
        Args:
            llm_manager: LLM管理器，用于高级实体链接
        """
        self.llm_manager = llm_manager
    
    def link_entities(self, entities: List[Entity]) -> Dict[str, str]:
        """链接实体名称
        
        Args:
            entities: 实体列表
            
        Returns:
            原始名称到标准名称的映射
        """
        # 简单的字符串匹配实现
        # 在实际应用中可能需要更复杂的相似度计算或LLM辅助
        
        name_mapping = {}
        canonical_names = {}
        
        for entity in entities:
            name = entity.name.strip().upper()
            
            # 查找相似的名称
            best_match = None
            best_similarity = 0.0
            
            for canonical_name in canonical_names:
                similarity = self._calculate_similarity(name, canonical_name)
                if similarity > best_similarity and similarity > 0.8:  # 相似度阈值
                    best_match = canonical_name
                    best_similarity = similarity
            
            if best_match:
                name_mapping[name] = best_match
            else:
                canonical_names[name] = entity
                name_mapping[name] = name
        
        return name_mapping
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """计算两个名称的相似度
        
        Args:
            name1: 名称1
            name2: 名称2
            
        Returns:
            相似度分数(0-1)
        """
        # 简单的Jaccard相似度
        set1 = set(name1.lower().split())
        set2 = set(name2.lower().split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union


class DescriptionSummarizer:
    """描述摘要器
    
    使用LLM合并多个实体或关系的描述。
    """
    
    def __init__(self, llm_manager: LLMManager):
        """初始化描述摘要器
        
        Args:
            llm_manager: LLM管理器
        """
        self.llm_manager = llm_manager
    
    async def summarize_entity_descriptions_async(
        self, 
        entities: List[Entity]
    ) -> str:
        """异步合并实体描述
        
        Args:
            entities: 实体列表
            
        Returns:
            合并后的描述
        """
        if not entities:
            return ""
        
        if len(entities) == 1:
            return entities[0].description
        
        descriptions = [e.description for e in entities]
        entity_name = entities[0].name
        
        prompt = f"""Given multiple descriptions of the same entity "{entity_name}", please create a comprehensive, coherent summary that captures all the key information without redundancy.

Descriptions:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Summarized description:"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            metadata={"prompt_type": "entity_description_summary"}
        )
        
        response = await self.llm_manager.generate_async(request)
        return response.content.strip()
    
    def summarize_entity_descriptions_sync(self, entities: List[Entity]) -> str:
        """同步合并实体描述"""
        if not entities:
            return ""
        
        if len(entities) == 1:
            return entities[0].description
        
        descriptions = [e.description for e in entities]
        entity_name = entities[0].name
        
        prompt = f"""Given multiple descriptions of the same entity "{entity_name}", please create a comprehensive, coherent summary that captures all the key information without redundancy.

Descriptions:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Summarized description:"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            metadata={"prompt_type": "entity_description_summary"}
        )
        
        response = self.llm_manager.generate_sync(request)
        return response.content.strip()
    
    async def summarize_relationship_descriptions_async(
        self, 
        relationships: List[Relationship]
    ) -> str:
        """异步合并关系描述
        
        Args:
            relationships: 关系列表
            
        Returns:
            合并后的描述
        """
        if not relationships:
            return ""
        
        if len(relationships) == 1:
            return relationships[0].description
        
        descriptions = [r.description for r in relationships]
        source = relationships[0].source_entity
        target = relationships[0].target_entity
        
        prompt = f"""Given multiple descriptions of the relationship between "{source}" and "{target}", please create a comprehensive, coherent summary that captures all the key information without redundancy.

Descriptions:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Summarized description:"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            metadata={"prompt_type": "relationship_description_summary"}
        )
        
        response = await self.llm_manager.generate_async(request)
        return response.content.strip()
    
    def summarize_relationship_descriptions_sync(
        self, 
        relationships: List[Relationship]
    ) -> str:
        """同步合并关系描述"""
        if not relationships:
            return ""
        
        if len(relationships) == 1:
            return relationships[0].description
        
        descriptions = [r.description for r in relationships]
        source = relationships[0].source_entity
        target = relationships[0].target_entity
        
        prompt = f"""Given multiple descriptions of the relationship between "{source}" and "{target}", please create a comprehensive, coherent summary that captures all the key information without redundancy.

Descriptions:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Summarized description:"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            metadata={"prompt_type": "relationship_description_summary"}
        )
        
        response = self.llm_manager.generate_sync(request)
        return response.content.strip()


class KnowledgeGraphBuilder:
    """知识图构建器
    
    从提取结果构建完整的知识图。
    """
    
    def __init__(self, llm_manager: LLMManager, config: GraphRAGConfig):
        """初始化知识图构建器
        
        Args:
            llm_manager: LLM管理器
            config: GraphRAG配置
        """
        self.llm_manager = llm_manager
        self.config = config
        self.entity_linker = EntityLinker(llm_manager)
        self.description_summarizer = DescriptionSummarizer(llm_manager)
    
    async def build_graph_async(
        self, 
        extraction_results: List[ExtractionResult]
    ) -> KnowledgeGraph:
        """异步构建知识图
        
        Args:
            extraction_results: 提取结果列表
            
        Returns:
            构建的知识图
        """
        try:
            # 收集所有实体、关系和主张
            all_entities = []
            all_relationships = []
            all_claims = []
            
            for result in extraction_results:
                all_entities.extend(result.entities)
                all_relationships.extend(result.relationships)
                all_claims.extend(result.claims)
            
            logger.info(
                f"Building knowledge graph from {len(all_entities)} entities, "
                f"{len(all_relationships)} relationships, {len(all_claims)} claims"
            )
            
            # 实体链接
            name_mapping = self.entity_linker.link_entities(all_entities)
            
            # 创建知识图
            kg = KnowledgeGraph()
            
            # 按链接后的名称分组实体
            entity_groups = defaultdict(list)
            for entity in all_entities:
                canonical_name = name_mapping[entity.name]
                entity_groups[canonical_name].append(entity)
            
            # 构建节点
            await self._build_nodes_async(kg, entity_groups)
            
            # 按链接后的名称分组关系
            relationship_groups = defaultdict(list)
            for rel in all_relationships:
                source_canonical = name_mapping.get(rel.source_entity)
                target_canonical = name_mapping.get(rel.target_entity)
                
                if source_canonical and target_canonical:
                    key = (source_canonical, target_canonical)
                    relationship_groups[key].append(rel)
            
            # 构建边
            await self._build_edges_async(kg, relationship_groups)
            
            # 添加主张
            for claim in all_claims:
                kg.add_claim(claim)
            
            # 添加构建元数据
            kg.metadata = {
                "built_at": str(pd.Timestamp.now()),
                "num_extraction_results": len(extraction_results),
                "num_raw_entities": len(all_entities),
                "num_raw_relationships": len(all_relationships),
                "num_raw_claims": len(all_claims),
                "entity_linking_ratio": len(kg.nodes) / len(all_entities) if all_entities else 0
            }
            
            stats = kg.get_statistics()
            logger.info(f"Knowledge graph built: {stats}")
            
            return kg
            
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to build knowledge graph: {e}")
    
    def build_graph_sync(
        self, 
        extraction_results: List[ExtractionResult]
    ) -> KnowledgeGraph:
        """同步构建知识图"""
        try:
            # 收集所有实体、关系和主张
            all_entities = []
            all_relationships = []
            all_claims = []
            
            for result in extraction_results:
                all_entities.extend(result.entities)
                all_relationships.extend(result.relationships)
                all_claims.extend(result.claims)
            
            logger.info(
                f"Building knowledge graph from {len(all_entities)} entities, "
                f"{len(all_relationships)} relationships, {len(all_claims)} claims"
            )
            
            # 实体链接
            name_mapping = self.entity_linker.link_entities(all_entities)
            
            # 创建知识图
            kg = KnowledgeGraph()
            
            # 按链接后的名称分组实体
            entity_groups = defaultdict(list)
            for entity in all_entities:
                canonical_name = name_mapping[entity.name]
                entity_groups[canonical_name].append(entity)
            
            # 构建节点
            self._build_nodes_sync(kg, entity_groups)
            
            # 按链接后的名称分组关系
            relationship_groups = defaultdict(list)
            for rel in all_relationships:
                source_canonical = name_mapping.get(rel.source_entity)
                target_canonical = name_mapping.get(rel.target_entity)
                
                if source_canonical and target_canonical:
                    key = (source_canonical, target_canonical)
                    relationship_groups[key].append(rel)
            
            # 构建边
            self._build_edges_sync(kg, relationship_groups)
            
            # 添加主张
            for claim in all_claims:
                kg.add_claim(claim)
            
            # 添加构建元数据
            kg.metadata = {
                "built_at": str(pd.Timestamp.now()),
                "num_extraction_results": len(extraction_results),
                "num_raw_entities": len(all_entities),
                "num_raw_relationships": len(all_relationships),
                "num_raw_claims": len(all_claims),
                "entity_linking_ratio": len(kg.nodes) / len(all_entities) if all_entities else 0
            }
            
            stats = kg.get_statistics()
            logger.info(f"Knowledge graph built: {stats}")
            
            return kg
            
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to build knowledge graph: {e}")
    
    async def _build_nodes_async(
        self, 
        kg: KnowledgeGraph, 
        entity_groups: Dict[str, List[Entity]]
    ):
        """异步构建图节点"""
        for canonical_name, entities in entity_groups.items():
            # 创建节点
            node = KnowledgeGraphNode(
                name=canonical_name,
                entity_type=entities[0].entity_type,
                description=""
            )
            
            # 添加所有实体
            for entity in entities:
                node.add_entity(entity)
            
            # 合并描述
            if len(entities) > 1:
                node.description = await self.description_summarizer.summarize_entity_descriptions_async(entities)
            else:
                node.description = entities[0].description
            
            # 添加到知识图
            kg.nodes[canonical_name] = node
            kg.graph.add_node(canonical_name, **node.to_dict())
    
    def _build_nodes_sync(
        self, 
        kg: KnowledgeGraph, 
        entity_groups: Dict[str, List[Entity]]
    ):
        """同步构建图节点"""
        for canonical_name, entities in entity_groups.items():
            # 创建节点
            node = KnowledgeGraphNode(
                name=canonical_name,
                entity_type=entities[0].entity_type,
                description=""
            )
            
            # 添加所有实体
            for entity in entities:
                node.add_entity(entity)
            
            # 合并描述
            if len(entities) > 1:
                node.description = self.description_summarizer.summarize_entity_descriptions_sync(entities)
            else:
                node.description = entities[0].description
            
            # 添加到知识图
            kg.nodes[canonical_name] = node
            kg.graph.add_node(canonical_name, **node.to_dict())
    
    async def _build_edges_async(
        self, 
        kg: KnowledgeGraph, 
        relationship_groups: Dict[Tuple[str, str], List[Relationship]]
    ):
        """异步构建图边"""
        for (source, target), relationships in relationship_groups.items():
            if source not in kg.nodes or target not in kg.nodes:
                continue
            
            # 创建边
            edge = KnowledgeGraphEdge(
                source=source,
                target=target,
                relationship_type=relationships[0].relationship_type,
                description="",
                weight=len(relationships),
                strength=sum(r.strength for r in relationships) / len(relationships)
            )
            
            # 添加所有关系
            for rel in relationships:
                edge.add_relationship(rel)
            
            # 合并描述
            if len(relationships) > 1:
                edge.description = await self.description_summarizer.summarize_relationship_descriptions_async(relationships)
            else:
                edge.description = relationships[0].description
            
            # 添加到知识图
            kg.edges[(source, target)] = edge
            kg.graph.add_edge(source, target, weight=edge.weight, **edge.to_dict())
            
            # 更新节点度数
            kg.nodes[source].degree = kg.graph.degree[source]
            kg.nodes[target].degree = kg.graph.degree[target]
    
    def _build_edges_sync(
        self, 
        kg: KnowledgeGraph, 
        relationship_groups: Dict[Tuple[str, str], List[Relationship]]
    ):
        """同步构建图边"""
        for (source, target), relationships in relationship_groups.items():
            if source not in kg.nodes or target not in kg.nodes:
                continue
            
            # 创建边
            edge = KnowledgeGraphEdge(
                source=source,
                target=target,
                relationship_type=relationships[0].relationship_type,
                description="",
                weight=len(relationships),
                strength=sum(r.strength for r in relationships) / len(relationships)
            )
            
            # 添加所有关系
            for rel in relationships:
                edge.add_relationship(rel)
            
            # 合并描述
            if len(relationships) > 1:
                edge.description = self.description_summarizer.summarize_relationship_descriptions_sync(relationships)
            else:
                edge.description = relationships[0].description
            
            # 添加到知识图
            kg.edges[(source, target)] = edge
            kg.graph.add_edge(source, target, weight=edge.weight, **edge.to_dict())
            
            # 更新节点度数
            kg.nodes[source].degree = kg.graph.degree[source]
            kg.nodes[target].degree = kg.graph.degree[target]