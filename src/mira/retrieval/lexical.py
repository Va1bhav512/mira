"""
Lexical retrieval: page-level BM25 over native/OCR page text.

Pages are keyed by (document_id, page_num), the same as Qdrant points, so
Phase 4 can fuse this ranking with the visual one.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import bm25s
import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

from .qdrant_store import SearchResult

_STEMMER = Stemmer.Stemmer("english")
_STOPWORDS = frozenset(STOPWORDS_EN)

# Runs of letters/digits, optionally joined by _ - . / ("vdd_io", "3.3v", "pa0-pa15")
_TOKEN = re.compile(r"[a-z0-9]+(?:[_\-./][a-z0-9]+)*")
_PART = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """NFKC (ligatures, superscripts: 'ﬁ' -> 'fi', 'I²C' -> 'I2C'), rejoin words hyphenated across lines."""
    text = unicodedata.normalize("NFKC", text)
    # ponytail: also merges real compounds broken at a line end ("high-\nspeed" -> "highspeed")
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _word(token: str) -> Optional[str]:
    """Stem plain words, drop stopwords; anything with a digit is an identifier, kept as-is."""
    if not token.isalpha():
        return token
    if token in _STOPWORDS or len(token) == 1:
        return None
    return _STEMMER.stemWord(token)


def tokenize(text: str) -> List[str]:
    """
    Tokenize for BM25, keeping identifiers intact.

    "STM32F401RE" -> ["stm32f401re"]
    "VDD_IO"      -> ["vdd_io", "vdd", "io"]   (compound kept whole, plus its parts)
    "registers"   -> ["regist"]
    """
    tokens = []
    for match in _TOKEN.findall(normalize(text).lower()):
        parts = _PART.findall(match)
        candidates = [match] + parts if len(parts) > 1 else [match]
        tokens.extend(w for w in map(_word, candidates) if w)
    return tokens


class BM25Index:
    """
    Page-level BM25 index persisted as one JSONL of page texts.

    The BM25 matrix is rebuilt in memory on first search after a change.
    ponytail: full rebuild, ~seconds for a few thousand pages; incremental if the corpus grows 100x
    """

    def __init__(self, path: str = ".cache/bm25"):
        """
        Args:
            path: Directory holding pages.jsonl (loaded if it exists)
        """
        self.file = Path(path) / "pages.jsonl"
        self.pages: dict[Tuple[str, int], str] = {}
        self._bm25 = None
        self._keys: List[Tuple[str, int]] = []

        if self.file.exists():
            with open(self.file) as f:
                for line in f:
                    row = json.loads(line)
                    self.pages[(row["document_id"], row["page_num"])] = row["text"]

    def add_pages(self, document_id: str, pages: List[Tuple[int, str]]):
        """Add or replace pages: list of (page_num, text)."""
        for page_num, text in pages:
            self.pages[(document_id, page_num)] = text
        self._bm25 = None

    def delete_document(self, document_id: str):
        """Remove all pages of a document."""
        self.pages = {k: v for k, v in self.pages.items() if k[0] != document_id}
        self._bm25 = None

    def save(self):
        """Write pages.jsonl (via temp file so a crash can't leave it half-written)."""
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for (document_id, page_num), text in self.pages.items():
                f.write(json.dumps({"document_id": document_id, "page_num": page_num, "text": text}) + "\n")
        tmp.replace(self.file)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        BM25 search over pages.

        Args:
            query: Text query
            top_k: Number of results
            document_filter: Optional document ID to restrict results to

        Returns:
            SearchResults (same type as visual search), best first; pages
            with no matching terms are omitted
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.pages:
            return []

        if self._bm25 is None:
            self._keys = list(self.pages)
            self._bm25 = bm25s.BM25()
            self._bm25.index([tokenize(self.pages[k]) for k in self._keys], show_progress=False)

        scores = self._bm25.get_scores(query_tokens)
        if document_filter:
            scores = np.where([k[0] == document_filter for k in self._keys], scores, 0)

        results = []
        for i in np.argsort(-scores)[:top_k]:
            if scores[i] <= 0:
                break
            document_id, page_num = self._keys[i]
            results.append(SearchResult(
                document_id=document_id,
                page_num=page_num,
                score=float(scores[i]),
                payload={"native_text": self.pages[self._keys[i]]},
            ))
        return results
