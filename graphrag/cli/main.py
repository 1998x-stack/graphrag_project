"""
GraphRAG命令行接口模块

提供便捷的命令行工具来使用GraphRAG功能。
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import json
import logging

from ..config.settings import GraphRAGConfig, load_config
from ..core.engine import GraphRAGEngine
from ..utils.exceptions import GraphRAGException


def setup_logging(debug: bool = False):
    """设置日志记录
    
    Args:
        debug: 是否启用调试级别日志
    """
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器
    
    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(
        description="GraphRAG: Graph-based Retrieval Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 构建索引
  python -m graphrag.cli.main build-index ./documents --output ./index
  
  # 从配置文件构建索引
  python -m graphrag.cli.main build-index ./documents --config config.yaml
  
  # 查询索引
  python -m graphrag.cli.main query "What are the main themes?" --index ./index
  
  # 批量查询
  python -m graphrag.cli.main batch-query queries.txt --index ./index --output results.json
  
  # 验证索引
  python -m graphrag.cli.main validate-index ./index
        """
    )
    
    # 全局参数
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置文件路径 (YAML格式)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='启用调试模式'
    )
    parser.add_argument(
        '--async-mode',
        action='store_true',
        help='使用异步模式（更快但需要更多资源）'
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 构建索引命令
    build_parser = subparsers.add_parser(
        'build-index',
        help='构建GraphRAG索引'
    )
    build_parser.add_argument(
        'documents',
        type=str,
        help='文档路径（文件或目录）'
    )
    build_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='索引输出目录'
    )
    build_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制覆盖现有索引'
    )
    
    # 查询命令
    query_parser = subparsers.add_parser(
        'query',
        help='查询GraphRAG索引'
    )
    query_parser.add_argument(
        'query',
        type=str,
        help='查询文本'
    )
    query_parser.add_argument(
        '--index', '-i',
        type=str,
        required=True,
        help='索引目录路径'
    )
    query_parser.add_argument(
        '--level', '-l',
        type=int,
        default=0,
        help='社区层级 (默认: 0)'
    )
    query_parser.add_argument(
        '--type', '-t',
        type=str,
        default='comprehensive',
        choices=['comprehensive', 'focused', 'detailed'],
        help='答案类型 (默认: comprehensive)'
    )
    query_parser.add_argument(
        '--output', '-o',
        type=str,
        help='结果输出文件路径'
    )
    
    # 批量查询命令
    batch_parser = subparsers.add_parser(
        'batch-query',
        help='批量查询GraphRAG索引'
    )
    batch_parser.add_argument(
        'queries_file',
        type=str,
        help='查询文件路径（每行一个查询）'
    )
    batch_parser.add_argument(
        '--index', '-i',
        type=str,
        required=True,
        help='索引目录路径'
    )
    batch_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='结果输出文件路径'
    )
    batch_parser.add_argument(
        '--level', '-l',
        type=int,
        default=0,
        help='社区层级 (默认: 0)'
    )
    batch_parser.add_argument(
        '--type', '-t',
        type=str,
        default='comprehensive',
        choices=['comprehensive', 'focused', 'detailed'],
        help='答案类型 (默认: comprehensive)'
    )
    
    # 验证索引命令
    validate_parser = subparsers.add_parser(
        'validate-index',
        help='验证GraphRAG索引'
    )
    validate_parser.add_argument(
        'index_dir',
        type=str,
        help='索引目录路径'
    )
    
    # 统计信息命令
    stats_parser = subparsers.add_parser(
        'stats',
        help='显示索引统计信息'
    )
    stats_parser.add_argument(
        'index_dir',
        type=str,
        help='索引目录路径'
    )
    stats_parser.add_argument(
        '--format', '-f',
        type=str,
        default='text',
        choices=['text', 'json'],
        help='输出格式 (默认: text)'
    )
    
    # 配置验证命令
    config_parser = subparsers.add_parser(
        'validate-config',
        help='验证配置文件'
    )
    config_parser.add_argument(
        'config_file',
        type=str,
        help='配置文件路径'
    )
    
    return parser


async def build_index_command(args, config: GraphRAGConfig) -> int:
    """执行构建索引命令
    
    Args:
        args: 命令行参数
        config: GraphRAG配置
        
    Returns:
        退出代码
    """
    try:
        # 检查输出目录
        output_dir = Path(args.output)
        if output_dir.exists() and not args.force:
            print(f"错误: 输出目录 {output_dir} 已存在。使用 --force 强制覆盖。")
            return 1
        
        # 检查文档路径
        docs_path = Path(args.documents)
        if not docs_path.exists():
            print(f"错误: 文档路径 {docs_path} 不存在。")
            return 1
        
        # 创建引擎并构建索引
        engine = GraphRAGEngine(config)
        
        print(f"开始构建GraphRAG索引...")
        print(f"文档路径: {docs_path}")
        print(f"输出目录: {output_dir}")
        
        start_time = time.time()
        
        if args.async_mode:
            index = await engine.build_index_async(str(docs_path), output_dir)
        else:
            index = engine.build_index_sync(str(docs_path), output_dir)
        
        build_time = time.time() - start_time
        
        print(f"\n✅ 索引构建完成！")
        print(f"构建时间: {build_time:.1f}秒")
        print(f"实体数量: {len(index.knowledge_graph.nodes)}")
        print(f"关系数量: {len(index.knowledge_graph.edges)}")
        print(f"社区数量: {index.community_structure.total_communities}")
        print(f"摘要数量: {len(index.community_summaries)}")
        print(f"索引保存到: {output_dir}")
        
        return 0
        
    except GraphRAGException as e:
        print(f"❌ GraphRAG错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1


async def query_command(args, config: GraphRAGConfig) -> int:
    """执行查询命令
    
    Args:
        args: 命令行参数
        config: GraphRAG配置
        
    Returns:
        退出代码
    """
    try:
        # 检查索引目录
        index_dir = Path(args.index)
        if not index_dir.exists():
            print(f"错误: 索引目录 {index_dir} 不存在。")
            return 1
        
        # 创建引擎并加载索引
        engine = GraphRAGEngine(config)
        
        print(f"加载索引: {index_dir}")
        index = engine.load_index(index_dir)
        
        print(f"处理查询: {args.query}")
        
        start_time = time.time()
        
        if args.async_mode:
            answer = await engine.query_async(
                args.query,
                community_level=args.level,
                answer_type=args.type
            )
        else:
            answer = engine.query_sync(
                args.query,
                community_level=args.level,
                answer_type=args.type
            )
        
        query_time = time.time() - start_time
        
        # 输出结果
        print(f"\n🔍 查询结果:")
        print(f"查询时间: {query_time:.2f}秒")
        print(f"使用社区: {len(answer.used_communities)}")
        print(f"Token消耗: {answer.total_token_count}")
        print(f"\n答案:")
        print("-" * 50)
        print(answer.answer)
        print("-" * 50)
        
        # 保存结果到文件
        if args.output:
            result_data = answer.to_dict()
            result_data['query_time'] = query_time
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n结果已保存到: {args.output}")
        
        return 0
        
    except GraphRAGException as e:
        print(f"❌ GraphRAG错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1


async def batch_query_command(args, config: GraphRAGConfig) -> int:
    """执行批量查询命令
    
    Args:
        args: 命令行参数
        config: GraphRAG配置
        
    Returns:
        退出代码
    """
    try:
        # 检查文件
        queries_file = Path(args.queries_file)
        if not queries_file.exists():
            print(f"错误: 查询文件 {queries_file} 不存在。")
            return 1
        
        index_dir = Path(args.index)
        if not index_dir.exists():
            print(f"错误: 索引目录 {index_dir} 不存在。")
            return 1
        
        # 读取查询
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        if not queries:
            print(f"错误: 查询文件为空。")
            return 1
        
        # 创建引擎并加载索引
        engine = GraphRAGEngine(config)
        index = engine.load_index(index_dir)
        
        print(f"开始批量查询: {len(queries)} 个查询")
        
        results = []
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            print(f"处理查询 {i}/{len(queries)}: {query[:50]}...")
            
            try:
                if args.async_mode:
                    answer = await engine.query_async(
                        query,
                        community_level=args.level,
                        answer_type=args.type
                    )
                else:
                    answer = engine.query_sync(
                        query,
                        community_level=args.level,
                        answer_type=args.type
                    )
                
                result = answer.to_dict()
                result['query_index'] = i
                results.append(result)
                
            except Exception as e:
                print(f"  ❌ 查询失败: {e}")
                results.append({
                    'query_index': i,
                    'query': query,
                    'error': str(e),
                    'answer': None
                })
        
        total_time = time.time() - start_time
        
        # 保存结果
        output_data = {
            'total_queries': len(queries),
            'successful_queries': len([r for r in results if 'error' not in r]),
            'total_time': total_time,
            'avg_time_per_query': total_time / len(queries),
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 批量查询完成！")
        print(f"总查询数: {len(queries)}")
        print(f"成功查询: {output_data['successful_queries']}")
        print(f"总时间: {total_time:.1f}秒")
        print(f"平均时间: {total_time/len(queries):.2f}秒/查询")
        print(f"结果保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1


def validate_index_command(args, config: GraphRAGConfig) -> int:
    """执行索引验证命令
    
    Args:
        args: 命令行参数
        config: GraphRAG配置
        
    Returns:
        退出代码
    """
    try:
        index_dir = Path(args.index_dir)
        if not index_dir.exists():
            print(f"错误: 索引目录 {index_dir} 不存在。")
            return 1
        
        # 创建引擎并加载索引
        engine = GraphRAGEngine(config)
        
        print(f"验证索引: {index_dir}")
        
        try:
            index = engine.load_index(index_dir)
            is_valid = engine.validate_index(index)
            
            if is_valid:
                print("✅ 索引验证通过！")
                
                # 显示统计信息
                stats = engine.get_index_statistics(index)
                print(f"\n📊 索引统计:")
                print(f"知识图节点: {stats['knowledge_graph']['num_nodes']}")
                print(f"知识图边: {stats['knowledge_graph']['num_edges']}")
                print(f"社区总数: {stats['community_structure']['total_communities']}")
                print(f"摘要总数: {stats['summaries']['total_summaries']}")
                print(f"平均评分: {stats['summaries']['avg_rating']:.2f}")
                
                return 0
            else:
                print("❌ 索引验证失败！")
                return 1
                
        except Exception as e:
            print(f"❌ 索引加载失败: {e}")
            return 1
        
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1


def stats_command(args, config: GraphRAGConfig) -> int:
    """执行统计信息命令
    
    Args:
        args: 命令行参数
        config: GraphRAG配置
        
    Returns:
        退出代码
    """
    try:
        index_dir = Path(args.index_dir)
        if not index_dir.exists():
            print(f"错误: 索引目录 {index_dir} 不存在。")
            return 1
        
        # 创建引擎并加载索引
        engine = GraphRAGEngine(config)
        index = engine.load_index(index_dir)
        stats = engine.get_index_statistics(index)
        
        if args.format == 'json':
            # JSON格式输出
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            # 文本格式输出
            print("📊 GraphRAG索引统计信息")
            print("=" * 50)
            
            print(f"\n🗂️  索引元数据:")
            metadata = stats['index_metadata']
            print(f"创建时间: {metadata.get('created_at', 'Unknown')}")
            print(f"构建时间: {metadata.get('total_build_time', 0):.1f}秒")
            print(f"文档数量: {metadata.get('num_documents', 'Unknown')}")
            print(f"文档块数: {metadata.get('num_chunks', 0)}")
            
            print(f"\n🕸️  知识图:")
            kg_stats = stats['knowledge_graph']
            print(f"节点数量: {kg_stats['num_nodes']}")
            print(f"边数量: {kg_stats['num_edges']}")
            print(f"图密度: {kg_stats.get('density', 0):.4f}")
            print(f"连通性: {'连通' if kg_stats.get('is_connected', False) else '非连通'}")
            print(f"连通组件: {kg_stats.get('num_connected_components', 0)}")
            if 'avg_degree' in kg_stats:
                print(f"平均度数: {kg_stats['avg_degree']:.2f}")
                print(f"最大度数: {kg_stats['max_degree']}")
            
            print(f"\n🏘️  社区结构:")
            comm_stats = stats['community_structure']
            print(f"总社区数: {comm_stats['total_communities']}")
            print(f"最大层级: {comm_stats['max_level']}")
            print(f"根社区数: {comm_stats['num_root_communities']}")
            if 'num_leaf_communities' in comm_stats:
                print(f"叶子社区数: {comm_stats['num_leaf_communities']}")
                print(f"平均社区大小: {comm_stats['avg_leaf_community_size']:.1f}")
            
            print(f"\n📄 社区摘要:")
            summary_stats = stats['summaries']
            print(f"摘要总数: {summary_stats['total_summaries']}")
            print(f"总Token数: {summary_stats['total_tokens']}")
            print(f"平均评分: {summary_stats['avg_rating']:.2f}/10")
        
        return 0
        
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1


def validate_config_command(args) -> int:
    """执行配置验证命令
    
    Args:
        args: 命令行参数
        
    Returns:
        退出代码
    """
    try:
        config_file = Path(args.config_file)
        if not config_file.exists():
            print(f"错误: 配置文件 {config_file} 不存在。")
            return 1
        
        print(f"验证配置文件: {config_file}")
        
        try:
            config = GraphRAGConfig.from_yaml(config_file)
            is_valid = config.validate_config()
            
            if is_valid:
                print("✅ 配置文件验证通过！")
                
                # 显示配置摘要
                print(f"\n📋 配置摘要:")
                print(f"LLM提供商: {config.llm.provider}")
                print(f"LLM模型: {config.llm.model_name}")
                print(f"文档块大小: {config.document_processing.chunk_size}")
                print(f"社区检测算法: {config.community_detection.algorithm}")
                print(f"存储类型: {config.storage.storage_type}")
                
                return 0
            else:
                print("❌ 配置文件验证失败！")
                return 1
                
        except Exception as e:
            print(f"❌ 配置文件解析失败: {e}")
            return 1
        
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1


async def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.debug)
    
    # 加载配置
    try:
        if args.config:
            config = GraphRAGConfig.from_yaml(args.config)
        else:
            config = GraphRAGConfig.from_env()
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return 1
    
    # 执行命令
    try:
        if args.command == 'build-index':
            return await build_index_command(args, config)
        elif args.command == 'query':
            return await query_command(args, config)
        elif args.command == 'batch-query':
            return await batch_query_command(args, config)
        elif args.command == 'validate-index':
            return validate_index_command(args, config)
        elif args.command == 'stats':
            return stats_command(args, config)
        elif args.command == 'validate-config':
            return validate_config_command(args)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ 操作被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cli_main():
    """命令行入口点"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())