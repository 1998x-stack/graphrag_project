"""
GraphRAG: Graph-based Retrieval Augmented Generation

This module provides the main entry point for the GraphRAG system.
"""

__version__ = "1.0.0"
__author__ = "GraphRAG Team"
__email__ = "team@graphrag.com"

from .core.engine import GraphRAGEngine
from .config.settings import GraphRAGConfig
from .core.exceptions import GraphRAGException

__all__ = [
    "GraphRAGEngine",
    "GraphRAGConfig", 
    "GraphRAGException"
]