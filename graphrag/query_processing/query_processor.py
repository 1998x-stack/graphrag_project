"""
查询处理模块

实现GraphRAG的Map-Reduce查询处理引擎，支持全局意义构建查询。
"""

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..core.llm_client import LLMManager, LLMRequest, LLMResponse
from ..config.settings import QueryProcessingConfig
from ..core.summarizer import CommunitySummary, SummarizationManager
from ..utils.exceptions import QueryProcessingError

logger = logging.getLogger(__name__)


@dataclass
class PartialAnswer:
    """部分答案数据类
    
    表示Map阶段单个社区摘要生成的部分答案。
    """
    
    community_id: str  # 社区ID
    answer: str  # 部分答案内容
    helpfulness_score: float  # 有用度评分(0-100)
    token_count: int  # token数量
    source_summaries: List[str] = field(default_factory=list)  # 使用的源摘要ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def is_useful(self, min_score: float = 0.0) -> bool:
        """判断答案是否有用"""
        return self.helpfulness_score > min_score


@dataclass
class GlobalAnswer:
    """全局答案数据类
    
    表示Reduce阶段合并后的最终答案。
    """
    
    query: str  # 原始查询
    answer: str  # 最终答案
    answer_type: str  # 答案类型(comprehensive/focused等)
    total_token_count: int  # 总token数量
    used_communities: List[str] = field(default_factory=list)  # 使用的社区ID
    partial_answers: List[PartialAnswer] = field(default_factory=list)  # 部分答案列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "total_token_count": self.total_token_count,
            "used_communities": self.used_communities,
            "num_partial_answers": len(self.partial_answers),
            "metadata": self.metadata
        }


class MapProcessor:
    """Map阶段处理器
    
    负责从社区摘要生成部分答案。
    """
    
    def __init__(self, llm_manager: LLMManager, config: QueryProcessingConfig):
        """初始化Map处理器
        
        Args:
            llm_manager: LLM管理器
            config: 查询处理配置
        """
        self.llm_manager = llm_manager
        self.config = config
        
        # 准备提示模板
        self.map_prompt = self._get_map_prompt()
    
    async def process_communities_async(
        self, 
        query: str,
        community_summaries: List[CommunitySummary]
    ) -> List[PartialAnswer]:
        """异步处理社区摘要生成部分答案
        
        Args:
            query: 用户查询
            community_summaries: 社区摘要列表
            
        Returns:
            部分答案列表
        """
        try:
            logger.info(f"Map phase: processing {len(community_summaries)} community summaries")
            
            # 随机打乱摘要顺序，避免位置偏差
            shuffled_summaries = community_summaries.copy()
            random.shuffle(shuffled_summaries)
            
            # 将摘要分组为chunks
            summary_chunks = self._chunk_summaries(shuffled_summaries)
            
            # 并行处理每个chunk
            if self.config.enable_parallel_processing:
                tasks = []
                semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
                
                for chunk in summary_chunks:
                    task = self._process_chunk_async(query, chunk, semaphore)
                    tasks.append(task)
                
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # 顺序处理
                chunk_results = []
                for chunk in summary_chunks:
                    result = await self._process_chunk_async(query, chunk)
                    chunk_results.append(result)
            
            # 合并所有部分答案
            all_partial_answers = []
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"Chunk processing failed: {result}")
                else:
                    all_partial_answers.extend(result)
            
            # 过滤低质量答案
            useful_answers = [
                answer for answer in all_partial_answers
                if answer.is_useful(self.config.min_helpfulness_score)
            ]
            
            logger.info(
                f"Map phase completed: {len(useful_answers)}/{len(all_partial_answers)} "
                f"useful answers (threshold: {self.config.min_helpfulness_score})"
            )
            
            return useful_answers
            
        except Exception as e:
            raise QueryProcessingError(
                f"Map phase processing failed: {e}",
                query=query,
                stage="map"
            )
    
    def process_communities_sync(
        self, 
        query: str,
        community_summaries: List[CommunitySummary]
    ) -> List[PartialAnswer]:
        """同步处理社区摘要生成部分答案"""
        try:
            logger.info(f"Map phase: processing {len(community_summaries)} community summaries")
            
            # 随机打乱摘要顺序
            shuffled_summaries = community_summaries.copy()
            random.shuffle(shuffled_summaries)
            
            # 将摘要分组为chunks
            summary_chunks = self._chunk_summaries(shuffled_summaries)
            
            # 顺序处理每个chunk
            all_partial_answers = []
            for chunk in summary_chunks:
                try:
                    chunk_answers = self._process_chunk_sync(query, chunk)
                    all_partial_answers.extend(chunk_answers)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    continue
            
            # 过滤低质量答案
            useful_answers = [
                answer for answer in all_partial_answers
                if answer.is_useful(self.config.min_helpfulness_score)
            ]
            
            logger.info(
                f"Map phase completed: {len(useful_answers)}/{len(all_partial_answers)} "
                f"useful answers (threshold: {self.config.min_helpfulness_score})"
            )
            
            return useful_answers
            
        except Exception as e:
            raise QueryProcessingError(
                f"Map phase processing failed: {e}",
                query=query,
                stage="map"
            )
    
    def _chunk_summaries(
        self, 
        summaries: List[CommunitySummary]
    ) -> List[List[CommunitySummary]]:
        """将摘要分组为适合处理的chunks
        
        Args:
            summaries: 摘要列表
            
        Returns:
            摘要chunk列表
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        max_chunk_tokens = self.config.max_context_tokens // 2  # 为答案生成预留空间
        
        for summary in summaries:
            summary_tokens = summary.token_count
            
            # 检查是否需要开始新chunk
            if (current_tokens + summary_tokens > max_chunk_tokens and current_chunk):
                chunks.append(current_chunk)
                current_chunk = [summary]
                current_tokens = summary_tokens
            else:
                current_chunk.append(summary)
                current_tokens += summary_tokens
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.debug(f"Created {len(chunks)} summary chunks")
        return chunks
    
    async def _process_chunk_async(
        self, 
        query: str,
        summaries: List[CommunitySummary],
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[PartialAnswer]:
        """异步处理单个摘要chunk"""
        if semaphore:
            async with semaphore:
                return await self._generate_partial_answers_async(query, summaries)
        else:
            return await self._generate_partial_answers_async(query, summaries)
    
    def _process_chunk_sync(
        self, 
        query: str,
        summaries: List[CommunitySummary]
    ) -> List[PartialAnswer]:
        """同步处理单个摘要chunk"""
        return self._generate_partial_answers_sync(query, summaries)
    
    async def _generate_partial_answers_async(
        self, 
        query: str,
        summaries: List[CommunitySummary]
    ) -> List[PartialAnswer]:
        """异步生成部分答案"""
        # 构建提示
        prompt = self._build_map_prompt(query, summaries)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_context_tokens // 2,
            metadata={"prompt_type": "map_generation", "query": query}
        )
        
        response = await self.llm_manager.generate_async(request)
        
        # 解析响应
        return self._parse_map_response(response, summaries)
    
    def _generate_partial_answers_sync(
        self, 
        query: str,
        summaries: List[CommunitySummary]
    ) -> List[PartialAnswer]:
        """同步生成部分答案"""
        # 构建提示
        prompt = self._build_map_prompt(query, summaries)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_context_tokens // 2,
            metadata={"prompt_type": "map_generation", "query": query}
        )
        
        response = self.llm_manager.generate_sync(request)
        
        # 解析响应
        return self._parse_map_response(response, summaries)
    
    def _get_map_prompt(self) -> str:
        """获取Map阶段提示模板"""
        if self.config.map_prompt_template:
            return self.config.map_prompt_template
        
        return """---Role---
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset, and incorporate any relevant general knowledge.

Note that the analysts' reports provided below are ranked in the **descending order of helpfulness**.
If you don't know the answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

At the beginning of your response, generate an integer score between 0-100 that indicates how **helpful** is this response in answering the user's question. Return the score in this format: <ANSWER HELPFULNESS>score value</ANSWER HELPFULNESS>.

---Target response length and format---
Multiple paragraphs

---Analyst Reports---
{report_data}

---Target response length and format---
Multiple paragraphs

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

Question: {query}
Output:"""
    
    def _build_map_prompt(
        self, 
        query: str, 
        summaries: List[CommunitySummary]
    ) -> str:
        """构建Map阶段的具体提示"""
        # 格式化社区摘要为报告数据
        report_data = []
        for i, summary in enumerate(summaries):
            report_data.append(f"Report {i+1}:")
            report_data.append(f"Title: {summary.title}")
            report_data.append(f"Summary: {summary.summary}")
            if summary.findings:
                report_data.append("Key Findings:")
                for finding in summary.findings:
                    report_data.append(f"- {finding.get('summary', '')}: {finding.get('explanation', '')}")
            report_data.append("")  # 空行分隔
        
        report_text = "\n".join(report_data)
        
        return self.map_prompt.format(
            query=query,
            report_data=report_text
        )
    
    def _parse_map_response(
        self, 
        response: LLMResponse, 
        summaries: List[CommunitySummary]
    ) -> List[PartialAnswer]:
        """解析Map阶段响应"""
        content = response.content.strip()
        
        # 提取有用度分数
        helpfulness_score = 0.0
        score_match = content.find("<ANSWER HELPFULNESS>")
        if score_match != -1:
            end_match = content.find("</ANSWER HELPFULNESS>", score_match)
            if end_match != -1:
                score_text = content[score_match + len("<ANSWER HELPFULNESS>"):end_match].strip()
                try:
                    helpfulness_score = float(score_text)
                except ValueError:
                    logger.warning(f"Failed to parse helpfulness score: {score_text}")
                
                # 移除分数标记
                content = content[:score_match] + content[end_match + len("</ANSWER HELPFULNESS>"):].strip()
        
        # 创建部分答案对象
        partial_answer = PartialAnswer(
            community_id=f"chunk_{len(summaries)}_communities",
            answer=content,
            helpfulness_score=helpfulness_score,
            token_count=response.tokens_used,
            source_summaries=[summary.community_id for summary in summaries],
            metadata={
                "response_time": response.response_time,
                "model": response.model,
                "num_source_summaries": len(summaries)
            }
        )
        
        return [partial_answer]


class ReduceProcessor:
    """Reduce阶段处理器
    
    负责将部分答案合并为最终答案。
    """
    
    def __init__(self, llm_manager: LLMManager, config: QueryProcessingConfig):
        """初始化Reduce处理器
        
        Args:
            llm_manager: LLM管理器
            config: 查询处理配置
        """
        self.llm_manager = llm_manager
        self.config = config
        
        # 准备提示模板
        self.reduce_prompt = self._get_reduce_prompt()
    
    async def combine_answers_async(
        self, 
        query: str,
        partial_answers: List[PartialAnswer],
        answer_type: str = "comprehensive"
    ) -> GlobalAnswer:
        """异步合并部分答案为最终答案
        
        Args:
            query: 用户查询
            partial_answers: 部分答案列表
            answer_type: 答案类型
            
        Returns:
            全局答案
        """
        try:
            logger.info(f"Reduce phase: combining {len(partial_answers)} partial answers")
            
            if not partial_answers:
                # 没有有用的部分答案
                return GlobalAnswer(
                    query=query,
                    answer="I don't have enough information to answer this question based on the available data.",
                    answer_type=answer_type,
                    total_token_count=0,
                    metadata={"no_useful_answers": True}
                )
            
            # 按有用度分数降序排序
            sorted_answers = sorted(
                partial_answers, 
                key=lambda x: x.helpfulness_score, 
                reverse=True
            )
            
            # 选择最有用的答案进行合并
            selected_answers = self._select_answers_for_context(sorted_answers)
            
            # 生成最终答案
            final_answer = await self._generate_final_answer_async(
                query, selected_answers, answer_type
            )
            
            logger.info(
                f"Reduce phase completed: {final_answer.total_token_count} tokens, "
                f"used {len(selected_answers)} partial answers"
            )
            
            return final_answer
            
        except Exception as e:
            raise QueryProcessingError(
                f"Reduce phase processing failed: {e}",
                query=query,
                stage="reduce"
            )
    
    def combine_answers_sync(
        self, 
        query: str,
        partial_answers: List[PartialAnswer],
        answer_type: str = "comprehensive"
    ) -> GlobalAnswer:
        """同步合并部分答案为最终答案"""
        try:
            logger.info(f"Reduce phase: combining {len(partial_answers)} partial answers")
            
            if not partial_answers:
                return GlobalAnswer(
                    query=query,
                    answer="I don't have enough information to answer this question based on the available data.",
                    answer_type=answer_type,
                    total_token_count=0,
                    metadata={"no_useful_answers": True}
                )
            
            # 按有用度分数降序排序
            sorted_answers = sorted(
                partial_answers, 
                key=lambda x: x.helpfulness_score, 
                reverse=True
            )
            
            # 选择最有用的答案进行合并
            selected_answers = self._select_answers_for_context(sorted_answers)
            
            # 生成最终答案
            final_answer = self._generate_final_answer_sync(
                query, selected_answers, answer_type
            )
            
            logger.info(
                f"Reduce phase completed: {final_answer.total_token_count} tokens, "
                f"used {len(selected_answers)} partial answers"
            )
            
            return final_answer
            
        except Exception as e:
            raise QueryProcessingError(
                f"Reduce phase processing failed: {e}",
                query=query,
                stage="reduce"
            )
    
    def _select_answers_for_context(
        self, 
        sorted_answers: List[PartialAnswer]
    ) -> List[PartialAnswer]:
        """选择适合放入上下文的答案
        
        Args:
            sorted_answers: 按有用度排序的答案列表
            
        Returns:
            选中的答案列表
        """
        selected = []
        total_tokens = 0
        max_tokens = self.config.max_context_tokens // 2  # 为最终答案生成预留空间
        
        for answer in sorted_answers:
            if total_tokens + answer.token_count <= max_tokens:
                selected.append(answer)
                total_tokens += answer.token_count
            else:
                break
        
        return selected
    
    async def _generate_final_answer_async(
        self, 
        query: str,
        selected_answers: List[PartialAnswer],
        answer_type: str
    ) -> GlobalAnswer:
        """异步生成最终答案"""
        # 构建提示
        prompt = self._build_reduce_prompt(query, selected_answers, answer_type)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_context_tokens // 2,
            metadata={"prompt_type": "reduce_generation", "query": query}
        )
        
        response = await self.llm_manager.generate_async(request)
        
        # 创建全局答案对象
        return GlobalAnswer(
            query=query,
            answer=response.content.strip(),
            answer_type=answer_type,
            total_token_count=response.tokens_used,
            used_communities=[answer.community_id for answer in selected_answers],
            partial_answers=selected_answers,
            metadata={
                "response_time": response.response_time,
                "model": response.model,
                "num_selected_answers": len(selected_answers),
                "total_available_answers": len(selected_answers)
            }
        )
    
    def _generate_final_answer_sync(
        self, 
        query: str,
        selected_answers: List[PartialAnswer],
        answer_type: str
    ) -> GlobalAnswer:
        """同步生成最终答案"""
        # 构建提示
        prompt = self._build_reduce_prompt(query, selected_answers, answer_type)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.config.max_context_tokens // 2,
            metadata={"prompt_type": "reduce_generation", "query": query}
        )
        
        response = self.llm_manager.generate_sync(request)
        
        # 创建全局答案对象
        return GlobalAnswer(
            query=query,
            answer=response.content.strip(),
            answer_type=answer_type,
            total_token_count=response.tokens_used,
            used_communities=[answer.community_id for answer in selected_answers],
            partial_answers=selected_answers,
            metadata={
                "response_time": response.response_time,
                "model": response.model,
                "num_selected_answers": len(selected_answers),
                "total_available_answers": len(selected_answers)
            }
        )
    
    def _get_reduce_prompt(self) -> str:
        """获取Reduce阶段提示模板"""
        if self.config.reduce_prompt_template:
            return self.config.reduce_prompt_template
        
        return """---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarize all relevant information in the input data tables appropriate for the response length and format, and incorporate any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

---Target response length and format---
{answer_type}

---Data tables---
{context_data}

---Target response length and format---
{answer_type}

Question: {query}
Output:"""
    
    def _build_reduce_prompt(
        self, 
        query: str, 
        selected_answers: List[PartialAnswer],
        answer_type: str
    ) -> str:
        """构建Reduce阶段的具体提示"""
        # 格式化部分答案为上下文数据
        context_data = []
        for i, answer in enumerate(selected_answers):
            context_data.append(f"Report {i+1}:")
            context_data.append(f"Helpfulness Score: {answer.helpfulness_score}")
            context_data.append(f"Content: {answer.answer}")
            context_data.append("")  # 空行分隔
        
        context_text = "\n".join(context_data)
        
        return self.reduce_prompt.format(
            query=query,
            answer_type=answer_type,
            context_data=context_text
        )


class QueryProcessor:
    """查询处理器主类
    
    协调整个Map-Reduce查询处理流程。
    """
    
    def __init__(self, llm_manager: LLMManager, config: QueryProcessingConfig):
        """初始化查询处理器
        
        Args:
            llm_manager: LLM管理器
            config: 查询处理配置
        """
        self.llm_manager = llm_manager
        self.config = config
        self.map_processor = MapProcessor(llm_manager, config)
        self.reduce_processor = ReduceProcessor(llm_manager, config)
    
    async def process_query_async(
        self, 
        query: str,
        community_summaries: List[CommunitySummary],
        answer_type: str = "comprehensive"
    ) -> GlobalAnswer:
        """异步处理查询
        
        Args:
            query: 用户查询
            community_summaries: 社区摘要列表
            answer_type: 答案类型
            
        Returns:
            全局答案
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Map阶段：生成部分答案
            partial_answers = await self.map_processor.process_communities_async(
                query, community_summaries
            )
            
            # Reduce阶段：合并最终答案
            global_answer = await self.reduce_processor.combine_answers_async(
                query, partial_answers, answer_type
            )
            
            logger.info(f"Query processing completed successfully")
            return global_answer
            
        except Exception as e:
            raise QueryProcessingError(f"Query processing failed: {e}", query=query)
    
    def process_query_sync(
        self, 
        query: str,
        community_summaries: List[CommunitySummary],
        answer_type: str = "comprehensive"
    ) -> GlobalAnswer:
        """同步处理查询"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Map阶段：生成部分答案
            partial_answers = self.map_processor.process_communities_sync(
                query, community_summaries
            )
            
            # Reduce阶段：合并最终答案
            global_answer = self.reduce_processor.combine_answers_sync(
                query, partial_answers, answer_type
            )
            
            logger.info(f"Query processing completed successfully")
            return global_answer
            
        except Exception as e:
            raise QueryProcessingError(f"Query processing failed: {e}", query=query)
    
    def get_processing_statistics(
        self, 
        global_answer: GlobalAnswer
    ) -> Dict[str, Any]:
        """获取查询处理统计信息
        
        Args:
            global_answer: 全局答案
            
        Returns:
            统计信息字典
        """
        partial_answers = global_answer.partial_answers
        
        stats = {
            "query": global_answer.query,
            "answer_type": global_answer.answer_type,
            "total_token_count": global_answer.total_token_count,
            "num_communities_used": len(global_answer.used_communities),
            "num_partial_answers": len(partial_answers),
            "helpfulness_scores": [answer.helpfulness_score for answer in partial_answers],
            "processing_metadata": global_answer.metadata
        }
        
        if partial_answers:
            scores = [answer.helpfulness_score for answer in partial_answers]
            stats.update({
                "avg_helpfulness_score": sum(scores) / len(scores),
                "max_helpfulness_score": max(scores),
                "min_helpfulness_score": min(scores)
            })
        
        return stats