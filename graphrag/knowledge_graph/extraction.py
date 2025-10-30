"""
实体关系提取模块

使用LLM从文档块中提取实体、关系和主张，是知识图构建的关键步骤。
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from ..core.llm_client import LLMManager, LLMRequest, LLMResponse
from ..config.settings import ExtractionConfig
from ..document_processing.document_processor import DocumentChunk
from ..utils.exceptions import ExtractionError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体数据类
    
    表示从文档中提取的实体。
    """
    
    name: str  # 实体名称
    entity_type: str  # 实体类型
    description: str  # 实体描述
    source_chunk_id: str  # 源文档块ID
    confidence: float = 1.0  # 提取置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后处理"""
        # 标准化实体名称
        self.name = self.name.strip().upper()
        self.entity_type = self.entity_type.strip().upper()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class Relationship:
    """关系数据类
    
    表示两个实体之间的关系。
    """
    
    source_entity: str  # 源实体名称
    target_entity: str  # 目标实体名称
    relationship_type: str  # 关系类型
    description: str  # 关系描述
    strength: float  # 关系强度(0-10)
    source_chunk_id: str  # 源文档块ID
    confidence: float = 1.0  # 提取置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后处理"""
        # 标准化实体名称
        self.source_entity = self.source_entity.strip().upper()
        self.target_entity = self.target_entity.strip().upper()
        
        # 验证关系强度
        if not 0 <= self.strength <= 10:
            self.strength = max(0, min(10, self.strength))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "strength": self.strength,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class Claim:
    """主张数据类
    
    表示从文档中提取的事实性陈述。
    """
    
    subject: str  # 主语实体
    predicate: str  # 谓语/关系
    object: str  # 宾语/对象
    description: str  # 详细描述
    source_chunk_id: str  # 源文档块ID
    confidence: float = 1.0  # 提取置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后处理"""
        # 标准化主语
        self.subject = self.subject.strip().upper()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ExtractionResult:
    """提取结果数据类
    
    封装从单个文档块提取的所有信息。
    """
    
    chunk_id: str  # 文档块ID
    entities: List[Entity] = field(default_factory=list)  # 提取的实体
    relationships: List[Relationship] = field(default_factory=list)  # 提取的关系
    claims: List[Claim] = field(default_factory=list)  # 提取的主张
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)  # 提取元数据
    
    def get_entity_names(self) -> Set[str]:
        """获取所有实体名称集合"""
        return {entity.name for entity in self.entities}
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """根据名称获取实体"""
        name = name.strip().upper()
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None


class BaseExtractor(ABC):
    """提取器基类
    
    定义所有提取器必须实现的接口。
    """
    
    def __init__(self, llm_manager: LLMManager, config: ExtractionConfig):
        """初始化提取器
        
        Args:
            llm_manager: LLM管理器
            config: 提取配置
        """
        self.llm_manager = llm_manager
        self.config = config
    
    @abstractmethod
    async def extract_async(self, chunk: DocumentChunk) -> ExtractionResult:
        """异步提取信息
        
        Args:
            chunk: 文档块
            
        Returns:
            提取结果
        """
        pass
    
    @abstractmethod
    def extract_sync(self, chunk: DocumentChunk) -> ExtractionResult:
        """同步提取信息
        
        Args:
            chunk: 文档块
            
        Returns:
            提取结果
        """
        pass


class EntityRelationshipExtractor(BaseExtractor):
    """实体关系提取器
    
    使用LLM从文档块中提取实体和关系。
    """
    
    def __init__(self, llm_manager: LLMManager, config: ExtractionConfig):
        """初始化提取器"""
        super().__init__(llm_manager, config)
        
        # 准备提示模板
        self.extraction_prompt = self._get_extraction_prompt()
        self.gleaning_prompt = self._get_gleaning_prompt()
    
    async def extract_async(self, chunk: DocumentChunk) -> ExtractionResult:
        """异步提取实体和关系"""
        try:
            # 第一轮提取
            initial_result = await self._extract_entities_relationships_async(chunk)
            
            # 自反思增强(Gleaning)
            if self.config.max_gleaning_iterations > 0:
                enhanced_result = await self._enhance_with_gleaning_async(
                    chunk, initial_result
                )
                return enhanced_result
            
            return initial_result
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract entities and relationships: {e}",
                chunk_id=chunk.chunk_id,
                extraction_type="entity_relationship"
            )
    
    def extract_sync(self, chunk: DocumentChunk) -> ExtractionResult:
        """同步提取实体和关系"""
        try:
            # 第一轮提取
            initial_result = self._extract_entities_relationships_sync(chunk)
            
            # 自反思增强(Gleaning)
            if self.config.max_gleaning_iterations > 0:
                enhanced_result = self._enhance_with_gleaning_sync(
                    chunk, initial_result
                )
                return enhanced_result
            
            return initial_result
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract entities and relationships: {e}",
                chunk_id=chunk.chunk_id,
                extraction_type="entity_relationship"
            )
    
    async def _extract_entities_relationships_async(
        self, 
        chunk: DocumentChunk
    ) -> ExtractionResult:
        """执行初始的实体关系提取"""
        
        # 构建提取请求
        prompt = self._build_extraction_prompt(chunk.content)
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            metadata={"prompt_type": "entity_relationship_extraction"}
        )
        
        # 调用LLM
        response = await self.llm_manager.generate_async(request)
        
        # 解析响应
        entities, relationships = self._parse_extraction_response(
            response.content, chunk.chunk_id
        )
        
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=entities,
            relationships=relationships,
            extraction_metadata={
                "extraction_method": "initial",
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
        )
    
    def _extract_entities_relationships_sync(
        self, 
        chunk: DocumentChunk
    ) -> ExtractionResult:
        """同步版本的实体关系提取"""
        
        # 构建提取请求
        prompt = self._build_extraction_prompt(chunk.content)
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            metadata={"prompt_type": "entity_relationship_extraction"}
        )
        
        # 调用LLM
        response = self.llm_manager.generate_sync(request)
        
        # 解析响应
        entities, relationships = self._parse_extraction_response(
            response.content, chunk.chunk_id
        )
        
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=entities,
            relationships=relationships,
            extraction_metadata={
                "extraction_method": "initial",
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
        )
    
    async def _enhance_with_gleaning_async(
        self, 
        chunk: DocumentChunk, 
        initial_result: ExtractionResult
    ) -> ExtractionResult:
        """使用自反思机制增强提取结果"""
        
        enhanced_result = initial_result
        
        for iteration in range(self.config.max_gleaning_iterations):
            # 构建自反思提示
            gleaning_prompt = self._build_gleaning_prompt(
                chunk.content, enhanced_result
            )
            
            request = LLMRequest(
                messages=[{"role": "user", "content": gleaning_prompt}],
                temperature=0.0,
                metadata={"prompt_type": "gleaning", "iteration": iteration + 1}
            )
            
            # 调用LLM
            response = await self.llm_manager.generate_async(request)
            
            # 检查是否发现了新的实体或关系
            if "CONTINUE" in response.content.upper():
                # 解析新发现的实体和关系
                new_entities, new_relationships = self._parse_extraction_response(
                    response.content, chunk.chunk_id
                )
                
                # 合并结果
                enhanced_result = self._merge_extraction_results(
                    enhanced_result, new_entities, new_relationships
                )
                
                logger.debug(
                    f"Gleaning iteration {iteration + 1} found "
                    f"{len(new_entities)} new entities, "
                    f"{len(new_relationships)} new relationships"
                )
            else:
                # LLM认为没有遗漏的实体，结束gleaning
                logger.debug(f"Gleaning completed after {iteration + 1} iterations")
                break
        
        # 更新元数据
        enhanced_result.extraction_metadata["gleaning_iterations"] = iteration + 1
        
        return enhanced_result
    
    def _enhance_with_gleaning_sync(
        self, 
        chunk: DocumentChunk, 
        initial_result: ExtractionResult
    ) -> ExtractionResult:
        """同步版本的自反思增强"""
        
        enhanced_result = initial_result
        
        for iteration in range(self.config.max_gleaning_iterations):
            # 构建自反思提示
            gleaning_prompt = self._build_gleaning_prompt(
                chunk.content, enhanced_result
            )
            
            request = LLMRequest(
                messages=[{"role": "user", "content": gleaning_prompt}],
                temperature=0.0,
                metadata={"prompt_type": "gleaning", "iteration": iteration + 1}
            )
            
            # 调用LLM
            response = self.llm_manager.generate_sync(request)
            
            # 检查是否发现了新的实体或关系
            if "CONTINUE" in response.content.upper():
                # 解析新发现的实体和关系
                new_entities, new_relationships = self._parse_extraction_response(
                    response.content, chunk.chunk_id
                )
                
                # 合并结果
                enhanced_result = self._merge_extraction_results(
                    enhanced_result, new_entities, new_relationships
                )
                
                logger.debug(
                    f"Gleaning iteration {iteration + 1} found "
                    f"{len(new_entities)} new entities, "
                    f"{len(new_relationships)} new relationships"
                )
            else:
                # LLM认为没有遗漏的实体，结束gleaning
                logger.debug(f"Gleaning completed after {iteration + 1} iterations")
                break
        
        # 更新元数据
        enhanced_result.extraction_metadata["gleaning_iterations"] = iteration + 1
        
        return enhanced_result
    
    def _get_extraction_prompt(self) -> str:
        """获取实体关系提取提示模板"""
        if self.config.extraction_prompt_template:
            return self.config.extraction_prompt_template
        
        entity_types_str = ", ".join(self.config.entity_types)
        few_shot_examples = self._build_few_shot_examples()
        
        return f"""---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity name: Name of the entity, capitalized
- entity type: One of the following types: [{entity_types_str}]
- entity description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity name><|><entity type><|><entity description>)

2. From the entities identified in step 1, identify all pairs of (source entity, target entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source entity: name of the source entity, as identified in step 1
- target entity: name of the target entity, as identified in step 1  
- relationship description: explanation as to why you think the source entity and the target entity are related to each other
- relationship strength: a numeric score indicating strength of the relationship between the source entity and target entity (0-10)
Format each relationship as ("relationship"<|><source entity><|><target entity><|><relationship description><|><relationship strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **<|RECORD|>** as the list delimiter.

4. When finished, output <|COMPLETE|>

{few_shot_examples}

---Real Data---
Entity types: {entity_types_str}
Text: {{input_text}}
Output:"""
    
    def _get_gleaning_prompt(self) -> str:
        """获取自反思提示模板"""
        return """---Goal---
You are a helpful assistant responsible for generating a comprehensive list of entities and relationships from a given text. You will be given an input text and entities/relationships already identified from that text. Please read the input text again and add any additional entities or relationships you may have missed in the previous extraction.

---Steps---
1. Review the input text and the list of entities and relationships already identified.
2. Determine if all entities and relationships have been identified. If not, identify the missed entities and relationships following the same format as the initial extraction.
3. If you find additional entities or relationships, output them using the same format: ("entity"<|>...) or ("relationship"<|>...)
4. If no additional entities or relationships are found, output "DONE"
5. If additional entities or relationships are found, output "CONTINUE" followed by the new entities and relationships.

---Real Data---
Text: {input_text}

Previously identified entities and relationships:
{previous_extractions}

Additional entities and relationships:"""
    
    def _build_few_shot_examples(self) -> str:
        """构建few-shot学习示例"""
        if not self.config.few_shot_examples:
            return ""
        
        examples_text = "\n---Examples---\n"
        for i, example in enumerate(self.config.few_shot_examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Text: {example.get('input', '')}\n"
            examples_text += f"Output:\n{example.get('output', '')}\n"
        
        return examples_text
    
    def _build_extraction_prompt(self, text: str) -> str:
        """构建具体的提取提示"""
        return self.extraction_prompt.format(input_text=text)
    
    def _build_gleaning_prompt(
        self, 
        text: str, 
        previous_result: ExtractionResult
    ) -> str:
        """构建自反思提示"""
        # 格式化之前的提取结果
        previous_extractions = []
        
        for entity in previous_result.entities:
            previous_extractions.append(
                f'("entity"<|>{entity.name}<|>{entity.entity_type}<|>{entity.description})'
            )
        
        for rel in previous_result.relationships:
            previous_extractions.append(
                f'("relationship"<|>{rel.source_entity}<|>{rel.target_entity}<|>'
                f'{rel.description}<|>{rel.strength})'
            )
        
        previous_text = "\n".join(previous_extractions)
        
        return self.gleaning_prompt.format(
            input_text=text,
            previous_extractions=previous_text
        )
    
    def _parse_extraction_response(
        self, 
        response: str, 
        chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """解析LLM的提取响应"""
        entities = []
        relationships = []
        
        # 分割记录
        records = response.split('<|RECORD|>')
        
        for record in records:
            record = record.strip()
            if not record or '<|COMPLETE|>' in record:
                continue
            
            try:
                if record.startswith('("entity"'):
                    entity = self._parse_entity_record(record, chunk_id)
                    if entity:
                        entities.append(entity)
                        
                elif record.startswith('("relationship"'):
                    relationship = self._parse_relationship_record(record, chunk_id)
                    if relationship:
                        relationships.append(relationship)
                        
            except Exception as e:
                logger.warning(f"Failed to parse record: {record}. Error: {e}")
                continue
        
        return entities, relationships
    
    def _parse_entity_record(self, record: str, chunk_id: str) -> Optional[Entity]:
        """解析实体记录"""
        try:
            # 提取括号内的内容
            match = re.search(r'\("entity"<\|>(.*?)\)', record, re.DOTALL)
            if not match:
                return None
            
            content = match.group(1)
            parts = content.split('<|>')
            
            if len(parts) >= 3:
                name = parts[0].strip()
                entity_type = parts[1].strip()
                description = parts[2].strip()
                
                return Entity(
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    source_chunk_id=chunk_id
                )
        
        except Exception as e:
            logger.warning(f"Failed to parse entity record: {record}. Error: {e}")
        
        return None
    
    def _parse_relationship_record(
        self, 
        record: str, 
        chunk_id: str
    ) -> Optional[Relationship]:
        """解析关系记录"""
        try:
            # 提取括号内的内容
            match = re.search(r'\("relationship"<\|>(.*?)\)', record, re.DOTALL)
            if not match:
                return None
            
            content = match.group(1)
            parts = content.split('<|>')
            
            if len(parts) >= 4:
                source_entity = parts[0].strip()
                target_entity = parts[1].strip()
                description = parts[2].strip()
                
                # 解析强度值
                try:
                    strength = float(parts[3].strip())
                except (ValueError, IndexError):
                    strength = 5.0  # 默认强度
                
                return Relationship(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    relationship_type="RELATED",  # 可以根据需要扩展
                    description=description,
                    strength=strength,
                    source_chunk_id=chunk_id
                )
        
        except Exception as e:
            logger.warning(f"Failed to parse relationship record: {record}. Error: {e}")
        
        return None
    
    def _merge_extraction_results(
        self,
        original: ExtractionResult,
        new_entities: List[Entity],
        new_relationships: List[Relationship]
    ) -> ExtractionResult:
        """合并提取结果，去除重复项"""
        
        # 合并实体（去重）
        existing_entity_names = {entity.name for entity in original.entities}
        merged_entities = original.entities.copy()
        
        for entity in new_entities:
            if entity.name not in existing_entity_names:
                merged_entities.append(entity)
                existing_entity_names.add(entity.name)
        
        # 合并关系（去重）
        existing_relationships = {
            (rel.source_entity, rel.target_entity, rel.description) 
            for rel in original.relationships
        }
        merged_relationships = original.relationships.copy()
        
        for rel in new_relationships:
            rel_key = (rel.source_entity, rel.target_entity, rel.description)
            if rel_key not in existing_relationships:
                merged_relationships.append(rel)
                existing_relationships.add(rel_key)
        
        return ExtractionResult(
            chunk_id=original.chunk_id,
            entities=merged_entities,
            relationships=merged_relationships,
            claims=original.claims,
            extraction_metadata=original.extraction_metadata
        )


class ClaimExtractor(BaseExtractor):
    """主张提取器
    
    使用LLM从文档块中提取事实性陈述。
    """
    
    def __init__(self, llm_manager: LLMManager, config: ExtractionConfig):
        """初始化主张提取器"""
        super().__init__(llm_manager, config)
        self.claim_prompt = self._get_claim_prompt()
    
    async def extract_async(self, chunk: DocumentChunk) -> List[Claim]:
        """异步提取主张"""
        if not self.config.enable_claim_extraction:
            return []
        
        try:
            prompt = self._build_claim_prompt(chunk.content)
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                metadata={"prompt_type": "claim_extraction"}
            )
            
            response = await self.llm_manager.generate_async(request)
            claims = self._parse_claim_response(response.content, chunk.chunk_id)
            
            return claims
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract claims: {e}",
                chunk_id=chunk.chunk_id,
                extraction_type="claim"
            )
    
    def extract_sync(self, chunk: DocumentChunk) -> List[Claim]:
        """同步提取主张"""
        if not self.config.enable_claim_extraction:
            return []
        
        try:
            prompt = self._build_claim_prompt(chunk.content)
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                metadata={"prompt_type": "claim_extraction"}
            )
            
            response = self.llm_manager.generate_sync(request)
            claims = self._parse_claim_response(response.content, chunk.chunk_id)
            
            return claims
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract claims: {e}",
                chunk_id=chunk.chunk_id,
                extraction_type="claim"
            )
    
    def _get_claim_prompt(self) -> str:
        """获取主张提取提示模板"""
        if self.config.claim_prompt_template:
            return self.config.claim_prompt_template
        
        return """---Goal---
Given a text passage, identify and extract all factual claims made in the text. A factual claim is a statement that can be verified as true or false.

---Steps---
1. Read through the text carefully
2. Identify statements that assert facts about entities, events, or relationships
3. For each factual claim, extract:
   - subject: The main entity or subject of the claim
   - predicate: The action, state, or relationship being asserted
   - object: The target entity or value (if applicable)
   - description: A clear, concise description of the full claim

4. Format each claim as: ("claim"<|><subject><|><predicate><|><object><|><description>)
5. Use **<|RECORD|>** as the delimiter between claims
6. When finished, output <|COMPLETE|>

---Real Data---
Text: {input_text}
Output:"""
    
    def _build_claim_prompt(self, text: str) -> str:
        """构建具体的主张提取提示"""
        return self.claim_prompt.format(input_text=text)
    
    def _parse_claim_response(self, response: str, chunk_id: str) -> List[Claim]:
        """解析主张提取响应"""
        claims = []
        
        # 分割记录
        records = response.split('<|RECORD|>')
        
        for record in records:
            record = record.strip()
            if not record or '<|COMPLETE|>' in record:
                continue
            
            try:
                if record.startswith('("claim"'):
                    claim = self._parse_claim_record(record, chunk_id)
                    if claim:
                        claims.append(claim)
                        
            except Exception as e:
                logger.warning(f"Failed to parse claim record: {record}. Error: {e}")
                continue
        
        return claims
    
    def _parse_claim_record(self, record: str, chunk_id: str) -> Optional[Claim]:
        """解析单个主张记录"""
        try:
            # 提取括号内的内容
            match = re.search(r'\("claim"<\|>(.*?)\)', record, re.DOTALL)
            if not match:
                return None
            
            content = match.group(1)
            parts = content.split('<|>')
            
            if len(parts) >= 4:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                obj = parts[2].strip()
                description = parts[3].strip()
                
                return Claim(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    description=description,
                    source_chunk_id=chunk_id
                )
        
        except Exception as e:
            logger.warning(f"Failed to parse claim record: {record}. Error: {e}")
        
        return None


class ExtractionManager:
    """提取管理器
    
    协调实体、关系和主张的提取过程。
    """
    
    def __init__(self, llm_manager: LLMManager, config: ExtractionConfig):
        """初始化提取管理器"""
        self.llm_manager = llm_manager
        self.config = config
        
        # 初始化各个提取器
        self.entity_relationship_extractor = EntityRelationshipExtractor(
            llm_manager, config
        )
        self.claim_extractor = ClaimExtractor(llm_manager, config)
    
    async def extract_chunk_async(self, chunk: DocumentChunk) -> ExtractionResult:
        """异步提取单个文档块的信息"""
        try:
            # 提取实体和关系
            result = await self.entity_relationship_extractor.extract_async(chunk)
            
            # 提取主张
            if self.config.enable_claim_extraction:
                claims = await self.claim_extractor.extract_async(chunk)
                result.claims = claims
            
            logger.info(
                f"Extracted from chunk {chunk.chunk_id}: "
                f"{len(result.entities)} entities, "
                f"{len(result.relationships)} relationships, "
                f"{len(result.claims)} claims"
            )
            
            return result
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract from chunk: {e}",
                chunk_id=chunk.chunk_id
            )
    
    def extract_chunk_sync(self, chunk: DocumentChunk) -> ExtractionResult:
        """同步提取单个文档块的信息"""
        try:
            # 提取实体和关系
            result = self.entity_relationship_extractor.extract_sync(chunk)
            
            # 提取主张
            if self.config.enable_claim_extraction:
                claims = self.claim_extractor.extract_sync(chunk)
                result.claims = claims
            
            logger.info(
                f"Extracted from chunk {chunk.chunk_id}: "
                f"{len(result.entities)} entities, "
                f"{len(result.relationships)} relationships, "
                f"{len(result.claims)} claims"
            )
            
            return result
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract from chunk: {e}",
                chunk_id=chunk.chunk_id
            )
    
    async def extract_chunks_async(
        self, 
        chunks: List[DocumentChunk],
        max_concurrent: int = 5
    ) -> List[ExtractionResult]:
        """异步提取多个文档块的信息"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _extract_with_semaphore(chunk: DocumentChunk) -> ExtractionResult:
            """带信号量的提取函数"""
            async with semaphore:
                return await self.extract_chunk_async(chunk)
        
        # 并发执行所有提取任务
        tasks = [_extract_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract chunk {chunks[i].chunk_id}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results