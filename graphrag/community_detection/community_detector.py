"""
社区检测模块

使用Leiden算法对知识图进行层次化社区检测，为后续的摘要生成提供结构化的节点分组。
"""

import asyncio
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx
import logging

# 图算法库导入
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

from ..knowledge_graph.graph_builder import KnowledgeGraph
from ..config.settings import CommunityDetectionConfig
from ..utils.exceptions import CommunityDetectionError

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """社区数据类
    
    表示图中的一个社区。
    """
    
    community_id: str  # 社区ID
    level: int  # 层级
    nodes: Set[str] = field(default_factory=set)  # 社区节点
    sub_communities: List['Community'] = field(default_factory=list)  # 子社区
    parent_community: Optional['Community'] = None  # 父社区
    size: int = 0  # 社区大小
    density: float = 0.0  # 社区密度
    modularity: float = 0.0  # 模块度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后处理"""
        self.size = len(self.nodes)
    
    def add_node(self, node: str):
        """添加节点到社区"""
        self.nodes.add(node)
        self.size = len(self.nodes)
    
    def add_sub_community(self, sub_community: 'Community'):
        """添加子社区"""
        sub_community.parent_community = self
        self.sub_communities.append(sub_community)
    
    def get_all_nodes(self) -> Set[str]:
        """获取社区及其所有子社区的所有节点"""
        all_nodes = self.nodes.copy()
        for sub_comm in self.sub_communities:
            all_nodes.update(sub_comm.get_all_nodes())
        return all_nodes
    
    def get_leaf_communities(self) -> List['Community']:
        """获取所有叶子社区"""
        if not self.sub_communities:
            return [self]
        
        leaf_communities = []
        for sub_comm in self.sub_communities:
            leaf_communities.extend(sub_comm.get_leaf_communities())
        
        return leaf_communities
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "nodes": list(self.nodes),
            "sub_communities": [sub.to_dict() for sub in self.sub_communities],
            "size": self.size,
            "density": self.density,
            "modularity": self.modularity,
            "metadata": self.metadata
        }


@dataclass
class CommunityStructure:
    """社区结构数据类
    
    表示完整的层次化社区结构。
    """
    
    levels: Dict[int, List[Community]] = field(default_factory=dict)  # 各层级的社区列表
    root_communities: List[Community] = field(default_factory=list)  # 根级社区
    max_level: int = 0  # 最大层级
    total_communities: int = 0  # 总社区数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def add_community(self, community: Community):
        """添加社区到结构"""
        level = community.level
        if level not in self.levels:
            self.levels[level] = []
        
        self.levels[level].append(community)
        
        if level == 0:
            self.root_communities.append(community)
        
        self.max_level = max(self.max_level, level)
        self.total_communities += 1
    
    def get_communities_by_level(self, level: int) -> List[Community]:
        """获取指定层级的社区"""
        return self.levels.get(level, [])
    
    def get_all_leaf_communities(self) -> List[Community]:
        """获取所有叶子社区"""
        leaf_communities = []
        for community in self.root_communities:
            leaf_communities.extend(community.get_leaf_communities())
        return leaf_communities
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取社区结构统计信息"""
        stats = {
            "total_communities": self.total_communities,
            "max_level": self.max_level,
            "num_root_communities": len(self.root_communities),
            "communities_by_level": {
                level: len(communities) for level, communities in self.levels.items()
            }
        }
        
        # 计算叶子社区统计
        leaf_communities = self.get_all_leaf_communities()
        if leaf_communities:
            leaf_sizes = [comm.size for comm in leaf_communities]
            stats.update({
                "num_leaf_communities": len(leaf_communities),
                "avg_leaf_community_size": sum(leaf_sizes) / len(leaf_sizes),
                "max_leaf_community_size": max(leaf_sizes),
                "min_leaf_community_size": min(leaf_sizes)
            })
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "levels": {
                str(level): [comm.to_dict() for comm in communities]
                for level, communities in self.levels.items()
            },
            "statistics": self.get_statistics(),
            "metadata": self.metadata
        }


class BaseCommunityDetector(ABC):
    """社区检测器基类
    
    定义所有社区检测器必须实现的接口。
    """
    
    def __init__(self, config: CommunityDetectionConfig):
        """初始化检测器
        
        Args:
            config: 社区检测配置
        """
        self.config = config
    
    @abstractmethod
    def detect_communities(self, graph: nx.Graph) -> CommunityStructure:
        """检测社区结构
        
        Args:
            graph: NetworkX图对象
            
        Returns:
            社区结构
        """
        pass
    
    def _calculate_community_metrics(
        self, 
        graph: nx.Graph, 
        community: Community
    ) -> Tuple[float, float]:
        """计算社区度量指标
        
        Args:
            graph: 图对象
            community: 社区对象
            
        Returns:
            (密度, 模块度)
        """
        if community.size < 2:
            return 0.0, 0.0
        
        # 提取社区子图
        subgraph = graph.subgraph(community.nodes)
        
        # 计算密度
        density = nx.density(subgraph)
        
        # 计算模块度（简化版本）
        total_edges = graph.number_of_edges()
        internal_edges = subgraph.number_of_edges()
        total_degree = sum(graph.degree[node] for node in community.nodes)
        expected_internal_edges = (total_degree ** 2) / (4 * total_edges) if total_edges > 0 else 0
        
        modularity = (internal_edges - expected_internal_edges) / total_edges if total_edges > 0 else 0
        
        return density, modularity


class LeidenCommunityDetector(BaseCommunityDetector):
    """Leiden算法社区检测器
    
    使用Leiden算法进行层次化社区检测。
    """
    
    def __init__(self, config: CommunityDetectionConfig):
        """初始化Leiden检测器"""
        super().__init__(config)
        
        if not HAS_LEIDEN:
            raise CommunityDetectionError(
                "Leiden algorithm dependencies not available. "
                "Please install with: pip install igraph leidenalg",
                algorithm="leiden"
            )
    
    def detect_communities(self, graph: nx.Graph) -> CommunityStructure:
        """使用Leiden算法检测社区"""
        try:
            # 转换为igraph格式
            ig_graph = self._networkx_to_igraph(graph)
            
            # 执行层次化社区检测
            community_structure = self._hierarchical_leiden(ig_graph)
            
            # 计算社区度量指标
            self._calculate_all_metrics(graph, community_structure)
            
            return community_structure
            
        except Exception as e:
            raise CommunityDetectionError(
                f"Leiden community detection failed: {e}",
                algorithm="leiden"
            )
    
    def _networkx_to_igraph(self, nx_graph: nx.Graph) -> ig.Graph:
        """转换NetworkX图为igraph格式
        
        Args:
            nx_graph: NetworkX图
            
        Returns:
            igraph图对象
        """
        # 创建节点映射
        nodes = list(nx_graph.nodes())
        node_to_id = {node: i for i, node in enumerate(nodes)}
        
        # 创建边列表
        edges = [(node_to_id[u], node_to_id[v]) for u, v in nx_graph.edges()]
        
        # 创建igraph对象
        ig_graph = ig.Graph(edges)
        ig_graph.vs['name'] = nodes
        
        # 添加边权重
        if nx_graph.size() > 0:
            weights = [nx_graph[u][v].get('weight', 1.0) for u, v in nx_graph.edges()]
            ig_graph.es['weight'] = weights
        
        return ig_graph
    
    def _hierarchical_leiden(self, graph: ig.Graph) -> CommunityStructure:
        """执行层次化Leiden社区检测
        
        Args:
            graph: igraph图对象
            
        Returns:
            社区结构
        """
        community_structure = CommunityStructure()
        node_names = graph.vs['name']
        
        # 当前图和社区映射
        current_graph = graph
        current_node_mapping = {i: node_names[i] for i in range(len(node_names))}
        parent_communities = {}
        
        level = 0
        
        while level < self.config.max_levels:
            logger.info(f"Detecting communities at level {level}")
            
            # 执行Leiden算法
            if hasattr(current_graph.es, 'weight'):
                partition = leidenalg.find_partition(
                    current_graph,
                    leidenalg.ModularityVertexPartition,
                    weights=current_graph.es['weight'],
                    resolution_parameter=self.config.resolution,
                    seed=self.config.random_seed
                )
            else:
                partition = leidenalg.find_partition(
                    current_graph,
                    leidenalg.ModularityVertexPartition,
                    resolution_parameter=self.config.resolution,
                    seed=self.config.random_seed
                )
            
            # 创建当前层级的社区
            level_communities = []
            community_to_nodes = {}
            
            for comm_id, community_nodes in enumerate(partition):
                if len(community_nodes) < self.config.min_community_size:
                    continue
                
                # 获取真实节点名称
                real_nodes = {current_node_mapping[node_id] for node_id in community_nodes}
                
                # 创建社区对象
                community = Community(
                    community_id=f"level_{level}_comm_{comm_id}",
                    level=level,
                    nodes=real_nodes
                )
                
                level_communities.append(community)
                community_to_nodes[comm_id] = community_nodes
                community_structure.add_community(community)
                
                # 建立与父社区的关系
                if level > 0:
                    for node_name in real_nodes:
                        if node_name in parent_communities:
                            parent_communities[node_name].add_sub_community(community)
                            break
            
            logger.info(f"Level {level}: detected {len(level_communities)} communities")
            
            # 检查是否需要继续
            if len(level_communities) <= 1 or level == self.config.max_levels - 1:
                break
            
            # 为下一层级准备
            if len(level_communities) > 1:
                # 创建新的图，每个社区成为一个节点
                next_graph, next_node_mapping = self._create_community_graph(
                    current_graph, community_to_nodes, current_node_mapping
                )
                
                # 更新父社区映射
                parent_communities = {}
                for community in level_communities:
                    for node_name in community.nodes:
                        parent_communities[node_name] = community
                
                current_graph = next_graph
                current_node_mapping = next_node_mapping
                level += 1
            else:
                break
        
        community_structure.metadata = {
            "algorithm": "leiden",
            "resolution": self.config.resolution,
            "max_levels_reached": level + 1,
            "total_nodes": len(node_names)
        }
        
        return community_structure
    
    def _create_community_graph(
        self, 
        graph: ig.Graph, 
        community_to_nodes: Dict[int, List[int]],
        node_mapping: Dict[int, str]
    ) -> Tuple[ig.Graph, Dict[int, str]]:
        """创建社区级别的图
        
        Args:
            graph: 原始图
            community_to_nodes: 社区ID到节点列表的映射
            node_mapping: 节点ID到名称的映射
            
        Returns:
            (社区图, 新的节点映射)
        """
        # 计算社区间的边权重
        community_edges = {}
        
        for edge in graph.es:
            source_id = edge.source
            target_id = edge.target
            weight = edge['weight'] if 'weight' in edge.attributes() else 1.0
            
            # 找到源和目标节点所属的社区
            source_comm = None
            target_comm = None
            
            for comm_id, nodes in community_to_nodes.items():
                if source_id in nodes:
                    source_comm = comm_id
                if target_id in nodes:
                    target_comm = comm_id
            
            # 如果是跨社区的边，累加权重
            if source_comm is not None and target_comm is not None and source_comm != target_comm:
                edge_key = tuple(sorted([source_comm, target_comm]))
                community_edges[edge_key] = community_edges.get(edge_key, 0) + weight
        
        # 创建新图
        community_ids = list(community_to_nodes.keys())
        community_id_mapping = {comm_id: i for i, comm_id in enumerate(community_ids)}
        
        edges = []
        weights = []
        
        for (comm1, comm2), weight in community_edges.items():
            if comm1 in community_id_mapping and comm2 in community_id_mapping:
                idx1 = community_id_mapping[comm1]
                idx2 = community_id_mapping[comm2]
                edges.append((idx1, idx2))
                weights.append(weight)
        
        new_graph = ig.Graph(edges)
        if weights:
            new_graph.es['weight'] = weights
        
        # 创建新的节点映射（使用第一个节点的名称作为社区代表）
        new_node_mapping = {}
        for i, comm_id in enumerate(community_ids):
            representative_node_id = community_to_nodes[comm_id][0]
            representative_name = node_mapping[representative_node_id]
            new_node_mapping[i] = f"comm_{comm_id}_{representative_name}"
        
        new_graph.vs['name'] = [new_node_mapping[i] for i in range(len(community_ids))]
        
        return new_graph, new_node_mapping
    
    def _calculate_all_metrics(self, graph: nx.Graph, structure: CommunityStructure):
        """计算所有社区的度量指标"""
        for level_communities in structure.levels.values():
            for community in level_communities:
                density, modularity = self._calculate_community_metrics(graph, community)
                community.density = density
                community.modularity = modularity


class LouvainCommunityDetector(BaseCommunityDetector):
    """Louvain算法社区检测器
    
    作为Leiden算法的替代方案。
    """
    
    def __init__(self, config: CommunityDetectionConfig):
        """初始化Louvain检测器"""
        super().__init__(config)
        
        if not HAS_LOUVAIN:
            raise CommunityDetectionError(
                "Louvain algorithm dependencies not available. "
                "Please install with: pip install python-louvain",
                algorithm="louvain"
            )
    
    def detect_communities(self, graph: nx.Graph) -> CommunityStructure:
        """使用Louvain算法检测社区"""
        try:
            # 执行Louvain算法
            partition = community_louvain.best_partition(
                graph, 
                resolution=self.config.resolution,
                random_state=self.config.random_seed
            )
            
            # 创建社区结构
            community_structure = CommunityStructure()
            
            # 按社区ID分组节点
            communities_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = set()
                communities_dict[comm_id].add(node)
            
            # 创建社区对象
            for comm_id, nodes in communities_dict.items():
                if len(nodes) >= self.config.min_community_size:
                    community = Community(
                        community_id=f"louvain_comm_{comm_id}",
                        level=0,
                        nodes=nodes
                    )
                    
                    # 计算度量指标
                    density, modularity = self._calculate_community_metrics(graph, community)
                    community.density = density
                    community.modularity = modularity
                    
                    community_structure.add_community(community)
            
            community_structure.metadata = {
                "algorithm": "louvain",
                "resolution": self.config.resolution,
                "total_nodes": len(graph.nodes())
            }
            
            return community_structure
            
        except Exception as e:
            raise CommunityDetectionError(
                f"Louvain community detection failed: {e}",
                algorithm="louvain"
            )


class CommunityDetectorFactory:
    """社区检测器工厂类
    
    根据配置创建相应的社区检测器。
    """
    
    _detectors = {
        "leiden": LeidenCommunityDetector,
        "louvain": LouvainCommunityDetector,
    }
    
    @classmethod
    def create_detector(cls, config: CommunityDetectionConfig) -> BaseCommunityDetector:
        """创建社区检测器
        
        Args:
            config: 社区检测配置
            
        Returns:
            社区检测器实例
            
        Raises:
            CommunityDetectionError: 当不支持的算法时
        """
        algorithm = config.algorithm.lower()
        
        if algorithm not in cls._detectors:
            raise CommunityDetectionError(
                f"Unsupported community detection algorithm: {algorithm}. "
                f"Available algorithms: {list(cls._detectors.keys())}",
                algorithm=algorithm
            )
        
        detector_class = cls._detectors[algorithm]
        return detector_class(config)
    
    @classmethod
    def register_detector(cls, algorithm: str, detector_class: type):
        """注册新的社区检测器类
        
        Args:
            algorithm: 算法名称
            detector_class: 检测器类
        """
        if not issubclass(detector_class, BaseCommunityDetector):
            raise ValueError("Detector class must inherit from BaseCommunityDetector")
        
        cls._detectors[algorithm.lower()] = detector_class


class CommunityDetectionManager:
    """社区检测管理器
    
    提供高级的社区检测接口。
    """
    
    def __init__(self, config: CommunityDetectionConfig):
        """初始化社区检测管理器
        
        Args:
            config: 社区检测配置
        """
        self.config = config
        self.detector = CommunityDetectorFactory.create_detector(config)
    
    def detect_communities(self, knowledge_graph: KnowledgeGraph) -> CommunityStructure:
        """检测知识图的社区结构
        
        Args:
            knowledge_graph: 知识图对象
            
        Returns:
            社区结构
            
        Raises:
            CommunityDetectionError: 当检测失败时
        """
        try:
            logger.info("Starting community detection")
            
            # 检查图是否为空
            if knowledge_graph.graph.number_of_nodes() == 0:
                logger.warning("Knowledge graph is empty, creating single community")
                structure = CommunityStructure()
                return structure
            
            # 检查图的连通性
            if not nx.is_connected(knowledge_graph.graph):
                logger.info(
                    f"Graph is not connected. "
                    f"Found {nx.number_connected_components(knowledge_graph.graph)} components"
                )
            
            # 执行社区检测
            structure = self.detector.detect_communities(knowledge_graph.graph)
            
            # 记录统计信息
            stats = structure.get_statistics()
            logger.info(f"Community detection completed: {stats}")
            
            return structure
            
        except Exception as e:
            raise CommunityDetectionError(f"Community detection failed: {e}")
    
    def validate_communities(self, structure: CommunityStructure) -> bool:
        """验证社区结构的有效性
        
        Args:
            structure: 社区结构
            
        Returns:
            结构是否有效
        """
        try:
            # 检查基本约束
            if structure.total_communities == 0:
                logger.warning("No communities detected")
                return False
            
            # 检查层级一致性
            for level, communities in structure.levels.items():
                for community in communities:
                    if community.level != level:
                        logger.error(f"Level mismatch in community {community.community_id}")
                        return False
                    
                    if community.size < self.config.min_community_size:
                        logger.warning(
                            f"Community {community.community_id} size ({community.size}) "
                            f"below minimum ({self.config.min_community_size})"
                        )
            
            # 检查节点覆盖（同一层级的社区不应有重叠节点）
            for level, communities in structure.levels.items():
                all_nodes = set()
                for community in communities:
                    overlap = all_nodes.intersection(community.nodes)
                    if overlap:
                        logger.error(f"Node overlap detected in level {level}: {overlap}")
                        return False
                    all_nodes.update(community.nodes)
            
            logger.info("Community structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Community validation failed: {e}")
            return False
    
    def get_community_summary(self, structure: CommunityStructure) -> Dict[str, Any]:
        """获取社区结构摘要
        
        Args:
            structure: 社区结构
            
        Returns:
            摘要信息
        """
        stats = structure.get_statistics()
        
        summary = {
            "algorithm": self.config.algorithm,
            "configuration": {
                "resolution": self.config.resolution,
                "max_levels": self.config.max_levels,
                "min_community_size": self.config.min_community_size
            },
            "statistics": stats,
            "metadata": structure.metadata
        }
        
        return summary