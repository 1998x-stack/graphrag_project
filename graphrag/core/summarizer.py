"""
社区摘要生成模块

为社区检测得到的各层级社区生成报告式摘要，实现GraphRAG的核心"预摘要"功能。
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
import logging

from ..core.llm_client import LLMManager, LLMRequest, LLMResponse
from ..config.settings import SummarizationConfig
from ..community_detection.community_detector import Community, CommunityStructure
from ..knowledge_graph.graph_builder import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge, KnowledgeGraphClaim
from ..utils.exceptions import SummarizationError

logger = logging.getLogger(__name__)


@dataclass
class ElementSummary:
    """元素摘要数据类
    
    表示图元素（节点、边、主张）的摘要信息。
    """
    
    element_type: str  # 元素类型(node/edge/claim)
    element_id: str  # 元素ID
    summary: str  # 摘要内容
    importance_score: float  # 重要性分数
    token_count: int  # token数量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class CommunitySummary:
    """社区摘要数据类
    
    表示单个社区的完整摘要报告。
    """
    
    community_id: str  # 社区ID
    level: int  # 层级
    title: str  # 摘要标题
    summary: str  # 摘要内容
    rating: float  # 影响严重性评分(0-10)
    rating_explanation: str  # 评分说明
    findings: List[Dict[str, str]]  # 关键发现列表
    full_content: str  # 完整摘要内容(JSON格式)
    
    # 元素统计
    num_nodes: int = 0
    num_edges: int = 0
    num_claims: int = 0
    token_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "rating": self.rating,
            "rating_explanation": self.rating_explanation,
            "findings": self.findings,
            "full_content": self.full_content,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_claims": self.num_claims,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


class ElementSummarizer:
    """元素摘要器
    
    为图中的节点、边和主张生成摘要。
    """
    
    def __init__(self, llm_manager: LLMManager, config: SummarizationConfig):
        """初始化元素摘要器
        
        Args:
            llm_manager: LLM管理器
            config: 摘要配置
        """
        self.llm_manager = llm_manager
        self.config = config
    
    def summarize_node(self, node: KnowledgeGraphNode) -> ElementSummary:
        """为节点生成摘要
        
        Args:
            node: 知识图节点
            
        Returns:
            元素摘要
        """
        # 节点摘要主要基于其描述和连接度
        summary = f"{node.name} ({node.entity_type}): {node.description}"
        
        # 计算重要性分数（基于度数）
        importance_score = min(node.degree / 10.0, 1.0)  # 归一化到0-1
        
        # 计算token数量
        token_count = self.llm_manager.count_tokens(summary)
        
        return ElementSummary(
            element_type="node",
            element_id=node.name,
            summary=summary,
            importance_score=importance_score,
            token_count=token_count,
            metadata={
                "degree": node.degree,
                "entity_type": node.entity_type,
                "source_chunks": len(node.source_chunks)
            }
        )
    
    def summarize_edge(self, edge: KnowledgeGraphEdge) -> ElementSummary:
        """为边生成摘要
        
        Args:
            edge: 知识图边
            
        Returns:
            元素摘要
        """
        # 边摘要基于关系描述和强度
        summary = f"{edge.source} -> {edge.target} ({edge.relationship_type}): {edge.description}"
        
        # 计算重要性分数（基于权重和强度）
        importance_score = min((edge.weight * edge.strength) / 50.0, 1.0)  # 归一化
        
        # 计算token数量
        token_count = self.llm_manager.count_tokens(summary)
        
        return ElementSummary(
            element_type="edge",
            element_id=f"{edge.source}-{edge.target}",
            summary=summary,
            importance_score=importance_score,
            token_count=token_count,
            metadata={
                "weight": edge.weight,
                "strength": edge.strength,
                "relationship_type": edge.relationship_type,
                "source_chunks": len(edge.source_chunks)
            }
        )
    
    def summarize_claim(self, claim: KnowledgeGraphClaim) -> ElementSummary:
        """为主张生成摘要
        
        Args:
            claim: 知识图主张
            
        Returns:
            元素摘要
        """
        # 主张摘要基于其描述
        summary = f"CLAIM: {claim.description}"
        
        # 主张的重要性分数较低
        importance_score = 0.3
        
        # 计算token数量
        token_count = self.llm_manager.count_tokens(summary)
        
        return ElementSummary(
            element_type="claim",
            element_id=f"{claim.subject}-{claim.predicate}-{claim.object}",
            summary=summary,
            importance_score=importance_score,
            token_count=token_count,
            metadata={
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object": claim.object,
                "source_chunks": len(claim.source_chunks)
            }
        )


class CommunitySummarizer:
    """社区摘要器
    
    为社区生成报告式摘要。
    """
    
    def __init__(self, llm_manager: LLMManager, config: SummarizationConfig):
        """初始化社区摘要器
        
        Args:
            llm_manager: LLM管理器
            config: 摘要配置
        """
        self.llm_manager = llm_manager
        self.config = config
        self.element_summarizer = ElementSummary(llm_manager, config)
        
        # 准备提示模板
        self.summary_prompt = self._get_summary_prompt()
    
    async def summarize_community_async(
        self, 
        community: Community,
        knowledge_graph: KnowledgeGraph,
        sub_community_summaries: Optional[List[CommunitySummary]] = None
    ) -> CommunitySummary:
        """异步为社区生成摘要
        
        Args:
            community: 社区对象
            knowledge_graph: 知识图
            sub_community_summaries: 子社区摘要列表（用于高层级社区）
            
        Returns:
            社区摘要
        """
        try:
            logger.info(f"Generating summary for community {community.community_id}")
            
            # 构建摘要输入
            summary_input = await self._build_summary_input_async(
                community, knowledge_graph, sub_community_summaries
            )
            
            # 生成摘要
            response = await self._generate_summary_async(summary_input)
            
            # 解析响应
            community_summary = self._parse_summary_response(
                response, community, summary_input
            )
            
            logger.info(
                f"Generated summary for {community.community_id}: "
                f"{community_summary.token_count} tokens, "
                f"rating {community_summary.rating}"
            )
            
            return community_summary
            
        except Exception as e:
            raise SummarizationError(
                f"Failed to generate community summary: {e}",
                community_id=community.community_id,
                level=community.level
            )
    
    def summarize_community_sync(
        self, 
        community: Community,
        knowledge_graph: KnowledgeGraph,
        sub_community_summaries: Optional[List[CommunitySummary]] = None
    ) -> CommunitySummary:
        """同步为社区生成摘要"""
        try:
            logger.info(f"Generating summary for community {community.community_id}")
            
            # 构建摘要输入
            summary_input = self._build_summary_input_sync(
                community, knowledge_graph, sub_community_summaries
            )
            
            # 生成摘要
            response = self._generate_summary_sync(summary_input)
            
            # 解析响应
            community_summary = self._parse_summary_response(
                response, community, summary_input
            )
            
            logger.info(
                f"Generated summary for {community.community_id}: "
                f"{community_summary.token_count} tokens, "
                f"rating {community_summary.rating}"
            )
            
            return community_summary
            
        except Exception as e:
            raise SummarizationError(
                f"Failed to generate community summary: {e}",
                community_id=community.community_id,
                level=community.level
            )
    
    async def _build_summary_input_async(
        self,
        community: Community,
        knowledge_graph: KnowledgeGraph,
        sub_community_summaries: Optional[List[CommunitySummary]] = None
    ) -> str:
        """异步构建摘要输入"""
        if sub_community_summaries:
            # 高层级社区：使用子社区摘要
            return self._build_hierarchical_input(sub_community_summaries)
        else:
            # 叶子社区：使用元素摘要
            return await self._build_element_input_async(community, knowledge_graph)
    
    def _build_summary_input_sync(
        self,
        community: Community,
        knowledge_graph: KnowledgeGraph,
        sub_community_summaries: Optional[List[CommunitySummary]] = None
    ) -> str:
        """同步构建摘要输入"""
        if sub_community_summaries:
            # 高层级社区：使用子社区摘要
            return self._build_hierarchical_input(sub_community_summaries)
        else:
            # 叶子社区：使用元素摘要
            return self._build_element_input_sync(community, knowledge_graph)
    
    def _build_hierarchical_input(
        self, 
        sub_community_summaries: List[CommunitySummary]
    ) -> str:
        """构建基于子社区摘要的输入"""
        input_parts = []
        
        for i, sub_summary in enumerate(sub_community_summaries):
            input_parts.append(f"## Sub-community {i+1}: {sub_summary.title}")
            input_parts.append(f"Rating: {sub_summary.rating}/10")
            input_parts.append(f"Summary: {sub_summary.summary}")
            
            if sub_summary.findings:
                input_parts.append("Key Findings:")
                for finding in sub_summary.findings:
                    input_parts.append(f"- {finding.get('summary', '')}: {finding.get('explanation', '')}")
            
            input_parts.append("")  # 空行分隔
        
        return "\n".join(input_parts)
    
    async def _build_element_input_async(
        self, 
        community: Community, 
        knowledge_graph: KnowledgeGraph
    ) -> str:
        """异步构建基于元素摘要的输入"""
        # 收集社区相关的元素
        element_summaries = []
        
        # 收集节点摘要
        for node_name in community.nodes:
            if node_name in knowledge_graph.nodes:
                node = knowledge_graph.nodes[node_name]
                summary = self.element_summarizer.summarize_node(node)
                element_summaries.append(summary)
        
        # 收集边摘要（社区内部边和重要的跨社区边）
        community_edges = []
        for edge_key, edge in knowledge_graph.edges.items():
            source, target = edge_key
            if source in community.nodes and target in community.nodes:
                # 社区内部边
                summary = self.element_summarizer.summarize_edge(edge)
                element_summaries.append(summary)
            elif (source in community.nodes or target in community.nodes) and edge.weight > 2:
                # 重要的跨社区边
                summary = self.element_summarizer.summarize_edge(edge)
                element_summaries.append(summary)
        
        # 收集相关主张
        for claim in knowledge_graph.claims:
            if claim.subject in community.nodes:
                summary = self.element_summarizer.summarize_claim(claim)
                element_summaries.append(summary)
        
        # 按重要性排序并限制token数量
        return self._prioritize_and_format_elements(element_summaries)
    
    def _build_element_input_sync(
        self, 
        community: Community, 
        knowledge_graph: KnowledgeGraph
    ) -> str:
        """同步构建基于元素摘要的输入"""
        # 收集社区相关的元素
        element_summaries = []
        
        # 收集节点摘要
        for node_name in community.nodes:
            if node_name in knowledge_graph.nodes:
                node = knowledge_graph.nodes[node_name]
                summary = self.element_summarizer.summarize_node(node)
                element_summaries.append(summary)
        
        # 收集边摘要
        for edge_key, edge in knowledge_graph.edges.items():
            source, target = edge_key
            if source in community.nodes and target in community.nodes:
                summary = self.element_summarizer.summarize_edge(edge)
                element_summaries.append(summary)
            elif (source in community.nodes or target in community.nodes) and edge.weight > 2:
                summary = self.element_summarizer.summarize_edge(edge)
                element_summaries.append(summary)
        
        # 收集相关主张
        for claim in knowledge_graph.claims:
            if claim.subject in community.nodes:
                summary = self.element_summarizer.summarize_claim(claim)
                element_summaries.append(summary)
        
        # 按重要性排序并限制token数量
        return self._prioritize_and_format_elements(element_summaries)
    
    def _prioritize_and_format_elements(
        self, 
        element_summaries: List[ElementSummary]
    ) -> str:
        """按重要性排序元素并格式化"""
        
        # 按重要性分数降序排序
        if self.config.prioritize_by_degree:
            element_summaries.sort(key=lambda x: x.importance_score, reverse=True)
        
        # 逐步添加元素，直到达到token限制
        formatted_parts = []
        total_tokens = 0
        max_tokens = self.config.max_summary_tokens // 2  # 为LLM生成留出空间
        
        # 按类型分组
        nodes = [e for e in element_summaries if e.element_type == "node"]
        edges = [e for e in element_summaries if e.element_type == "edge"]
        claims = [e for e in element_summaries if e.element_type == "claim"]
        
        # 优先添加重要的边（根据论文，按边的度数降序）
        if edges:
            formatted_parts.append("# Relationships")
            for edge_summary in edges:
                if total_tokens + edge_summary.token_count > max_tokens:
                    break
                formatted_parts.append(f"- {edge_summary.summary}")
                total_tokens += edge_summary.token_count
        
        # 添加相关节点
        if nodes:
            formatted_parts.append("\n# Entities")
            for node_summary in nodes:
                if total_tokens + node_summary.token_count > max_tokens:
                    break
                formatted_parts.append(f"- {node_summary.summary}")
                total_tokens += node_summary.token_count
        
        # 添加主张（如果启用）
        if claims and self.config.include_claims:
            formatted_parts.append("\n# Claims")
            for claim_summary in claims:
                if total_tokens + claim_summary.token_count > max_tokens:
                    break
                formatted_parts.append(f"- {claim_summary.summary}")
                total_tokens += claim_summary.token_count
        
        return "\n".join(formatted_parts)
    
    async def _generate_summary_async(self, summary_input: str) -> LLMResponse:
        """异步生成摘要"""
        prompt = self._build_summary_prompt(summary_input)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_summary_tokens,
            metadata={"prompt_type": "community_summary"}
        )
        
        return await self.llm_manager.generate_async(request)
    
    def _generate_summary_sync(self, summary_input: str) -> LLMResponse:
        """同步生成摘要"""
        prompt = self._build_summary_prompt(summary_input)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_summary_tokens,
            metadata={"prompt_type": "community_summary"}
        )
        
        return self.llm_manager.generate_sync(request)
    
    def _get_summary_prompt(self) -> str:
        """获取摘要生成提示模板"""
        if self.config.summary_prompt_template:
            return self.config.summary_prompt_template
        
        return """---Role---
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

---Goal---
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

---Report Structure---
The report should include the following sections:
- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community. IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{{
    "title": <report title>,
    "summary": <executive summary>,
    "rating": <impact severity rating>,
    "rating_explanation": <rating explanation>,
    "findings": [
        {{
            "summary": <insight 1 summary>,
            "explanation": <insight 1 explanation>
        }},
        {{
            "summary": <insight 2 summary>,
            "explanation": <insight 2 explanation>
        }}
    ]
}}

---Grounding Rules---
Points supported by data should list their data references as follows:
"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

---Real Data---
Use the following text for your answer. Do not make anything up in your answer.

Input:
{input_text}

Output:"""
    
    def _build_summary_prompt(self, summary_input: str) -> str:
        """构建具体的摘要提示"""
        return self.summary_prompt.format(input_text=summary_input)
    
    def _parse_summary_response(
        self, 
        response: LLMResponse, 
        community: Community,
        summary_input: str
    ) -> CommunitySummary:
        """解析摘要响应"""
        try:
            # 尝试解析JSON响应
            response_text = response.content.strip()
            
            # 清理可能的markdown代码块
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # 解析JSON
            summary_data = json.loads(response_text)
            
            # 创建社区摘要对象
            community_summary = CommunitySummary(
                community_id=community.community_id,
                level=community.level,
                title=summary_data.get("title", f"Community {community.community_id}"),
                summary=summary_data.get("summary", ""),
                rating=float(summary_data.get("rating", 0.0)),
                rating_explanation=summary_data.get("rating_explanation", ""),
                findings=summary_data.get("findings", []),
                full_content=response_text,
                num_nodes=len(community.nodes),
                token_count=response.tokens_used,
                metadata={
                    "response_time": response.response_time,
                    "model": response.model,
                    "input_length": len(summary_input)
                }
            )
            
            return community_summary
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # 创建简化的摘要
            return CommunitySummary(
                community_id=community.community_id,
                level=community.level,
                title=f"Community {community.community_id}",
                summary=response.content[:500] + "..." if len(response.content) > 500 else response.content,
                rating=5.0,
                rating_explanation="Unable to parse structured response",
                findings=[],
                full_content=response.content,
                num_nodes=len(community.nodes),
                token_count=response.tokens_used,
                metadata={"parse_error": str(e)}
            )


class SummarizationManager:
    """摘要生成管理器
    
    协调整个社区结构的摘要生成过程。
    """
    
    def __init__(self, llm_manager: LLMManager, config: SummarizationConfig):
        """初始化摘要生成管理器
        
        Args:
            llm_manager: LLM管理器
            config: 摘要配置
        """
        self.llm_manager = llm_manager
        self.config = config
        self.community_summarizer = CommunitySummarizer(llm_manager, config)
    
    async def generate_all_summaries_async(
        self,
        community_structure: CommunityStructure,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, CommunitySummary]:
        """异步为所有社区生成摘要
        
        Args:
            community_structure: 社区结构
            knowledge_graph: 知识图
            
        Returns:
            社区ID到摘要的映射
        """
        try:
            logger.info("Starting community summarization")
            
            all_summaries = {}
            
            # 从叶子社区开始，逐层向上生成摘要
            for level in sorted(community_structure.levels.keys(), reverse=True):
                level_communities = community_structure.get_communities_by_level(level)
                
                logger.info(f"Generating summaries for level {level}: {len(level_communities)} communities")
                
                # 并行生成当前层级的摘要
                tasks = []
                for community in level_communities:
                    if community.sub_communities:
                        # 高层级社区：使用子社区摘要
                        sub_summaries = [all_summaries[sub.community_id] for sub in community.sub_communities 
                                       if sub.community_id in all_summaries]
                        task = self.community_summarizer.summarize_community_async(
                            community, knowledge_graph, sub_summaries
                        )
                    else:
                        # 叶子社区：使用元素摘要
                        task = self.community_summarizer.summarize_community_async(
                            community, knowledge_graph
                        )
                    
                    tasks.append(task)
                
                # 等待当前层级的所有摘要完成
                level_summaries = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                for i, result in enumerate(level_summaries):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to summarize community {level_communities[i].community_id}: {result}")
                    else:
                        all_summaries[result.community_id] = result
            
            logger.info(f"Generated {len(all_summaries)} community summaries")
            return all_summaries
            
        except Exception as e:
            raise SummarizationError(f"Failed to generate community summaries: {e}")
    
    def generate_all_summaries_sync(
        self,
        community_structure: CommunityStructure,
        knowledge_graph: KnowledgeGraph
    ) -> Dict[str, CommunitySummary]:
        """同步为所有社区生成摘要"""
        try:
            logger.info("Starting community summarization")
            
            all_summaries = {}
            
            # 从叶子社区开始，逐层向上生成摘要
            for level in sorted(community_structure.levels.keys(), reverse=True):
                level_communities = community_structure.get_communities_by_level(level)
                
                logger.info(f"Generating summaries for level {level}: {len(level_communities)} communities")
                
                # 顺序生成当前层级的摘要
                for community in level_communities:
                    try:
                        if community.sub_communities:
                            # 高层级社区：使用子社区摘要
                            sub_summaries = [all_summaries[sub.community_id] for sub in community.sub_communities 
                                           if sub.community_id in all_summaries]
                            summary = self.community_summarizer.summarize_community_sync(
                                community, knowledge_graph, sub_summaries
                            )
                        else:
                            # 叶子社区：使用元素摘要
                            summary = self.community_summarizer.summarize_community_sync(
                                community, knowledge_graph
                            )
                        
                        all_summaries[summary.community_id] = summary
                        
                    except Exception as e:
                        logger.error(f"Failed to summarize community {community.community_id}: {e}")
                        continue
            
            logger.info(f"Generated {len(all_summaries)} community summaries")
            return all_summaries
            
        except Exception as e:
            raise SummarizationError(f"Failed to generate community summaries: {e}")
    
    def get_summaries_by_level(
        self, 
        summaries: Dict[str, CommunitySummary], 
        level: int
    ) -> List[CommunitySummary]:
        """获取指定层级的摘要
        
        Args:
            summaries: 所有摘要
            level: 层级
            
        Returns:
            该层级的摘要列表
        """
        return [summary for summary in summaries.values() if summary.level == level]
    
    def save_summaries(self, summaries: Dict[str, CommunitySummary], output_path: str):
        """保存摘要到文件
        
        Args:
            summaries: 摘要字典
            output_path: 输出文件路径
        """
        summary_data = {
            "summaries": {
                comm_id: summary.to_dict() 
                for comm_id, summary in summaries.items()
            },
            "statistics": {
                "total_summaries": len(summaries),
                "levels": list(set(summary.level for summary in summaries.values())),
                "total_tokens": sum(summary.token_count for summary in summaries.values())
            },
            "metadata": {
                "config": self.config.dict(),
                "generated_at": str(pd.Timestamp.now())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(summaries)} summaries to {output_path}")
    
    @classmethod
    def load_summaries(cls, input_path: str) -> Dict[str, CommunitySummary]:
        """从文件加载摘要
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            摘要字典
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summaries = {}
        for comm_id, summary_data in data["summaries"].items():
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
                num_edges=summary_data.get("num_edges", 0),
                num_claims=summary_data.get("num_claims", 0),
                token_count=summary_data["token_count"],
                metadata=summary_data["metadata"]
            )
            summaries[comm_id] = summary
        
        logger.info(f"Loaded {len(summaries)} summaries from {input_path}")
        return summaries