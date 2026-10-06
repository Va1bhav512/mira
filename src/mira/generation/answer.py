"""
End-to-end answering: hybrid retrieval -> evidence regions -> crops -> VLM -> cited answer.

Citations come from the retriever, not the VLM: each evidence id maps to the
(document, page, box) it was cropped from, and the answer cites the ids the
VLM says it used.
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from mira.evidence import Box, evidence_regions, heatmap, max_patch_region, render_region
from mira.retrieval import HybridRetriever, PageEmbedding
from mira.retrieval.hybrid import HybridResult

# What the VLM sees: the whole page (no cropping), a box around the hottest patch
# (single-patch baseline), or Query-Adaptive Evidence Cropping
STRATEGIES = ("page", "max_patch", "heatmap")
FULL_PAGE: Box = (0.0, 0.0, 1.0, 1.0)

# (document_id, page_num, normalized box) -> image of that region
PageImageFn = Callable[[str, int, Box], Image.Image]


def pdf_page_images(pdf_dir: str, dpi: int = 300) -> PageImageFn:
    """Region renderer for a directory of PDFs named <document_id>.pdf (scripts/index_corpus.py's ids)."""
    return lambda doc, page, box: render_region(str(Path(pdf_dir) / f"{doc}.pdf"), page, box, dpi=dpi)


@dataclass
class Evidence:
    id: str  # E1, E2, ... as shown to the VLM
    document_id: str
    page_num: int
    box: Box  # normalized (x0, y0, x1, y1)
    image: Image.Image


@dataclass
class Answer:
    answer: str
    sources: List[Dict]  # {"document", "page", "bbox"} for each cited evidence region
    evidence: List[Evidence]
    raw: str  # unparsed VLM reply
    timings: Dict[str, float] = field(default_factory=dict)
    hits: List[HybridResult] = field(default_factory=list)  # retrieved pages, with each channel's rank


def page_boxes(query_embedding: np.ndarray, page: Optional[PageEmbedding], strategy: str) -> List[Box]:
    """Evidence boxes on one page under a context strategy (page may be None for "page")."""
    if strategy == "page":
        return [FULL_PAGE]
    heat = heatmap(query_embedding, page.patch_embeddings)
    if strategy == "max_patch":
        return [max_patch_region(heat).box]
    if strategy == "heatmap":
        return [r.box for r in evidence_regions(heat)] or [FULL_PAGE]
    raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")


def answer_query(
    query: str,
    retriever: HybridRetriever,
    generator,
    page_image: PageImageFn,
    top_pages: int = 3,
    strategy: str = "heatmap",
    mode: str = "adaptive",
    document_filter: Optional[str] = None,
) -> Answer:
    """
    Args:
        query: The question
        retriever: Hybrid retriever (its embedder and store are reused for heatmaps)
        generator: VLMGenerator (or anything with its generate(query, [(id, image)]) signature)
        page_image: Region renderer, e.g. pdf_page_images("data/samples")
        top_pages: Retrieved pages to take evidence from
        strategy: One of STRATEGIES
        mode: Retrieval mode (see mira.retrieval.hybrid.MODES)
        document_filter: Optional document ID to restrict retrieval to

    Returns:
        Answer with deterministic citations
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    t0 = time()
    query_embedding = retriever.embedder.embed_query(query)
    hits = retriever.search_modes(
        query, top_k=top_pages, document_filter=document_filter, modes=(mode,), query_embedding=query_embedding
    )[mode]
    t1 = time()

    evidence = []
    for hit in hits:
        page = retriever.store.get_page(hit.document_id, hit.page_num) if strategy != "page" else None
        for box in page_boxes(query_embedding, page, strategy):
            evidence.append(Evidence(
                f"E{len(evidence) + 1}", hit.document_id, hit.page_num, box,
                page_image(hit.document_id, hit.page_num, box),
            ))
    t2 = time()

    answer, cited, raw = generator.generate(query, [(e.id, e.image) for e in evidence])
    t3 = time()

    sources = [
        {"document": e.document_id, "page": e.page_num, "bbox": [round(v, 4) for v in e.box]}
        for e in evidence if e.id in cited
    ]
    return Answer(answer, sources, evidence, raw, {"retrieve": t1 - t0, "crop": t2 - t1, "generate": t3 - t2}, hits)
