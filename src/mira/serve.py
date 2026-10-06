"""
Serving (Phase 6) and demo: one process with a JSON API and a Gradio UI over the same pipeline.

    POST /query   {"question": ..., "mode": "adaptive", "strategy": "heatmap", "top_pages": 3,
                   "document": null, "include_images": false}
                  -> answer, cited sources (document, page, box), evidence regions, the retrieved
                     pages with each channel's rank, the fusion weights and timings
    /ui           Gradio demo: answer, retrieved pages with heatmap + evidence boxes, the crops
                  the VLM read, and why each page was retrieved

Run on a GPU box with an indexed corpus (scripts/index_corpus.py):
    python -m mira.serve --pdf-dir data/samples --qdrant-url http://localhost:6333 --bm25-path .cache/bm25
    python -m mira.serve ... --share     # Gradio UI only, behind a public gradio.live link (e.g. on Kaggle)
"""

import argparse
import base64
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from mira.evidence import heatmap, render_region
from mira.generation import STRATEGIES, answer_query, pdf_page_images
from mira.generation.answer import FULL_PAGE
from mira.retrieval.hybrid import MODES, query_weights

DISPLAY_DPI = 100


def _png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def overlay(page: Image.Image, heat: Optional[np.ndarray], boxes: List[tuple]) -> Image.Image:
    """Page with the query heatmap in red (alpha = heat) and evidence boxes in green."""
    page = page.convert("RGBA")
    if heat is not None and heat.max() > 0:
        alpha = Image.fromarray((heat / heat.max() * 150).astype(np.uint8)).resize(page.size, Image.BILINEAR)
        red = Image.new("RGBA", page.size, (230, 30, 30, 0))
        red.putalpha(alpha)
        page = Image.alpha_composite(page, red)
    draw = ImageDraw.Draw(page)
    w, h = page.size
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(20, 160, 60, 255), width=4)
    return page.convert("RGB")


class Mira:
    """The pipeline shared by the API and the UI."""

    def __init__(self, retriever, generator, pdf_dir: str, page_image=None):
        self.retriever = retriever
        self.generator = generator
        self.pdf_dir = Path(pdf_dir)
        self.page_image = page_image or pdf_page_images(pdf_dir)

    def documents(self) -> List[str]:
        return sorted({doc for doc, _ in self.retriever.text_index.pages})

    def query(self, question: str, mode: str = "adaptive", strategy: str = "heatmap",
              top_pages: int = 3, document: Optional[str] = None, include_images: bool = False) -> dict:
        ans = answer_query(question, self.retriever, self.generator, self.page_image,
                           top_pages=top_pages, strategy=strategy, mode=mode, document_filter=document)
        w = query_weights(question)
        out = {
            "answer": ans.answer,
            "sources": ans.sources,
            "evidence": [{"id": e.id, "document": e.document_id, "page": e.page_num,
                          "bbox": [round(v, 4) for v in e.box]} for e in ans.evidence],
            "retrieved": [{"document": h.document_id, "page": h.page_num, "score": round(h.score, 5),
                           "visual_rank": h.visual_rank, "lexical_rank": h.lexical_rank} for h in ans.hits],
            "weights": {"visual": w.visual, "lexical": w.lexical, "identifiers": w.identifiers,
                        "quoted": w.quoted, "visual_cues": w.visual_cues} if mode == "adaptive" else None,
            "timings": {k: round(v, 2) for k, v in ans.timings.items()},
        }
        if include_images:
            for item, e in zip(out["evidence"], ans.evidence):
                item["image_png_b64"] = _png_b64(e.image)
        out["_answer"] = ans  # for the UI; dropped from API responses
        return out

    def page_views(self, question: str, result: dict) -> List[tuple]:
        """(overlay image, caption) per retrieved page, for the UI."""
        query_embedding = self.retriever.embedder.embed_query(question)
        views = []
        for hit in result["retrieved"]:
            doc, page = hit["document"], hit["page"]
            image = render_region(str(self.pdf_dir / f"{doc}.pdf"), page, FULL_PAGE, dpi=DISPLAY_DPI)
            heat = heatmap(query_embedding, self.retriever.store.get_page(doc, page).patch_embeddings)
            boxes = [e["bbox"] for e in result["evidence"] if (e["document"], e["page"]) == (doc, page)]
            ranks = f"visual #{hit['visual_rank'] or '-'}, BM25 #{hit['lexical_rank'] or '-'}"
            views.append((overlay(image, heat, boxes), f"{doc} p.{page + 1} ({ranks})"))
        return views


def build_ui(mira: Mira, examples: List[str]):
    import gradio as gr

    def run(question, mode, strategy, top_pages, document):
        result = mira.query(question, mode, strategy, int(top_pages), None if document == "All documents" else document)
        cited = "\n".join(f"- {s['document']}, page {s['page'] + 1}" for s in result["sources"]) or "- (none)"
        answer = f"### {result['answer']}\n\n**Cited evidence**\n{cited}"
        crops = [(e.image, f"{e.id}: {e.document_id} p.{e.page_num + 1}") for e in result["_answer"].evidence]
        table = [[h["document"], h["page"] + 1, h["score"], h["visual_rank"], h["lexical_rank"]] for h in result["retrieved"]]
        details = {"weights": result["weights"], "timings_s": result["timings"]}
        return answer, mira.page_views(question, result), crops, table, details

    with gr.Blocks(title="Mira") as ui:
        gr.Markdown("## Mira: multimodal RAG over technical PDFs\nRetrieval (ColQwen + BM25, fused), "
                    "evidence cropping, and a cited answer from Qwen2.5-VL.")
        with gr.Row():
            question = gr.Textbox(label="Question", scale=4)
            ask = gr.Button("Ask", variant="primary", scale=1)
        with gr.Row():
            mode = gr.Dropdown(list(MODES), value="adaptive", label="Retrieval mode")
            strategy = gr.Dropdown(list(STRATEGIES), value="heatmap", label="Context for the VLM")
            top_pages = gr.Slider(1, 5, value=3, step=1, label="Pages")
            document = gr.Dropdown(["All documents"] + mira.documents(), value="All documents", label="Document")
        answer = gr.Markdown()
        pages = gr.Gallery(label="Retrieved pages: heatmap (red) and evidence boxes (green)", columns=3, height=520)
        crops = gr.Gallery(label="What the VLM read", columns=4, height=260)
        table = gr.Dataframe(headers=["document", "page", "fused score", "visual rank", "BM25 rank"], label="Why these pages")
        details = gr.JSON(label="Fusion weights and timings")
        inputs = [question, mode, strategy, top_pages, document]
        outputs = [answer, pages, crops, table, details]
        ask.click(run, inputs, outputs, api_name="query")
        question.submit(run, inputs, outputs)
        if examples:
            gr.Examples([[q] for q in examples], [question])
    return ui


def build_app(mira: Mira, examples: List[str]):
    import gradio as gr
    from fastapi import FastAPI
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        question: str
        mode: str = "adaptive"
        strategy: str = "heatmap"
        top_pages: int = 3
        document: Optional[str] = None
        include_images: bool = False

    app = FastAPI(title="Mira")

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        result = mira.query(**req.model_dump())
        result.pop("_answer")
        return result

    return gr.mount_gradio_app(app, build_ui(mira, examples), path="/ui")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data/samples")
    parser.add_argument("--qdrant-url", help="Qdrant server URL (default: cloud cluster from .env)")
    parser.add_argument("--collection", default="mira_pages")
    parser.add_argument("--bm25-path", default=".cache/bm25")
    parser.add_argument("--examples", default="data/eval/queries.jsonl", help="JSONL with a 'query' field")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio UI via a public share link (no /query API)")
    args = parser.parse_args()

    import json

    import uvicorn

    from mira.generation import VLMGenerator
    from mira.retrieval import BM25Index, HybridRetriever, QdrantMultivectorStore

    retriever = HybridRetriever(
        store=QdrantMultivectorStore(url=args.qdrant_url, collection_name=args.collection),
        text_index=BM25Index(args.bm25_path),
    )
    retriever.embedder  # load ColQwen now, not on the first request
    mira = Mira(retriever, VLMGenerator(), args.pdf_dir)
    examples = [json.loads(line)["query"] for line in open(args.examples)][::4] if Path(args.examples).exists() else []

    if args.share:
        build_ui(mira, examples).launch(share=True, server_name=args.host, server_port=args.port)
    else:
        uvicorn.run(build_app(mira, examples), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
