# GraphRAG: Graph-based Retrieval Augmented Generation

## 项目概述

GraphRAG是一个基于图的检索增强生成系统，专门用于处理需要全局理解的"意义构建"查询。相比传统的向量RAG，GraphRAG能够回答需要整个文档集合全局理解的问题。

## 核心特性

- **实体关系知识图构建**: 使用LLM从文档中提取实体、关系和主张
- **层次化社区检测**: 使用Leiden算法进行图社区划分
- **智能摘要生成**: 为各层级社区预生成摘要，支持自适应token管理
- **Map-Reduce查询处理**: 并行生成部分答案后智能合并
- **可配置领域适应**: 支持针对特定领域的few-shot定制

## 系统架构

```
文档 → 切块 → 实体关系提取 → 知识图构建 → 社区检测 → 社区摘要 → 查询处理
```

## 安装使用

```bash
pip install -r requirements.txt
python -m graphrag.cli.main --help
```

## 项目结构

```
graphrag_project/
├── graphrag/              # 核心模块
│   ├── core/              # 核心引擎
│   ├── document_processing/    # 文档处理
│   ├── knowledge_graph/   # 知识图构建
│   ├── community_detection/    # 社区检测
│   ├── query_processing/  # 查询处理
│   ├── config/           # 配置管理
│   ├── utils/            # 工具模块
│   ├── evaluation/       # 评估模块
│   └── cli/              # 命令行接口
├── tests/                # 单元测试
├── examples/             # 示例代码
├── docs/                 # 文档
└── data/                 # 数据文件
```

## 许可证

MIT License