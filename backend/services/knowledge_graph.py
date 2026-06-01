"""
services/knowledge_graph.py — Knowledge Graph Service Wrapper
==============================================================
Re-exports the existing KnowledgeGraph from the new module path.
The actual implementation stays in the original knowledge_graph.py
for backward compatibility with the scraper and migration scripts.
"""

import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add backend dir to path so we can import the original module
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Re-export from original module
try:
    from knowledge_graph import KnowledgeGraph, get_knowledge_graph
except ImportError as e:
    logger.warning(f"Could not import knowledge_graph module: {e}")

    class KnowledgeGraph:
        """Stub when original module is not available."""
        def node_count(self): return 0
        def edge_count(self): return 0
        def search_relevant(self, *a, **kw): return []
        def extract_and_add_from_text(self, *a, **kw): pass
        def save(self): pass
        def get_stats(self): return {"total_nodes": 0, "total_edges": 0}

    def get_knowledge_graph():
        return KnowledgeGraph()

__all__ = ["KnowledgeGraph", "get_knowledge_graph"]
