"""
autocomplete/trie.py — Trie Data Structure for Prefix Search
=============================================================
Fast prefix-based autocomplete with frequency weighting.
Supports case-insensitive search and multi-word prefixes.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """A node in the trie."""
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_end: bool = False
    value: str = ""          # The full original string
    frequency: int = 0       # Usage frequency for ranking
    category: str = ""       # Type: faculty, department, program, etc.


class Trie:
    """
    Trie data structure for fast prefix search.

    Features:
      - Case-insensitive search
      - Frequency-based ranking
      - Category support (faculty, department, etc.)
      - Multi-word prefix matching
    """

    def __init__(self):
        self.root = TrieNode()
        self._count = 0

    def insert(self, text: str, frequency: int = 1, category: str = ""):
        """Insert a string into the trie."""
        if not text or len(text.strip()) < 2:
            return

        text = text.strip()
        key = text.lower()
        node = self.root

        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_end:
            self._count += 1
        node.is_end = True
        node.value = text
        node.frequency = max(node.frequency, frequency)
        node.category = category or node.category

    def search_prefix(self, prefix: str, limit: int = 8) -> List[Tuple[str, str, int]]:
        """
        Find all completions for a given prefix.

        Returns list of (value, category, frequency) tuples
        sorted by frequency (highest first).
        """
        if not prefix:
            return []

        prefix = prefix.lower().strip()
        node = self.root

        # Navigate to the prefix node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all completions from this node
        results = []
        self._collect(node, results, limit * 3)  # Collect more than needed for sorting

        # Sort by frequency and return top results
        results.sort(key=lambda x: x[2], reverse=True)
        
        # ── Fuzzy Search Fallback ──
        if not results:
            import difflib
            all_entries = []
            self._collect(self.root, all_entries, 1000)
            words = [val for val, cat, freq in all_entries]
            
            # Find close matches to the user's typo prefix
            close_matches = difflib.get_close_matches(prefix, words, n=limit, cutoff=0.5)
            
            # Reconstruct the results format
            for match in close_matches:
                for val, cat, freq in all_entries:
                    if val == match:
                        results.append((val, cat, freq))
                        break

        return results[:limit]

    def _collect(
        self,
        node: TrieNode,
        results: List[Tuple[str, str, int]],
        limit: int,
    ):
        """Recursively collect all completions from a node."""
        if len(results) >= limit:
            return

        if node.is_end:
            results.append((node.value, node.category, node.frequency))

        for child in node.children.values():
            if len(results) >= limit:
                break
            self._collect(child, results, limit)

    def count(self) -> int:
        """Return total number of entries in the trie."""
        return self._count
