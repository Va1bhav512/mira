import gc
from PIL import Image
import numpy as np
from typing import Callable, List, Optional, Tuple
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ColQwenEmbedder:
    """
    Wrapper for ColQwen2.5 visual embedding model.
    Runs in fp16 on GPU (~7 GB for the 3B model, fits a T4) and fp32 on CPU.

    fp16 occasionally overflows to NaN on a page (1 in 256 pages of the scanned NASA
    report, deterministic). Those inputs alone are re-embedded in bf16, which doesn't
    overflow but is ~6x slower on a T4 (no native bf16), so it isn't the default.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: Optional[str] = None,
    ):
        """
        Initialize ColQwen embedder.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', None=auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = None
        self._load(self.dtype)
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

    def _load(self, dtype: torch.dtype):
        """(Re)load the model in dtype, freeing the current copy first."""
        self.model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.model = ColQwen2_5.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()

    def _redo_in_bf16(self, items: list, embed_one: Callable) -> list:
        """
        Re-embed inputs whose output overflowed to NaN, with a temporary bf16 model.

        A T4 can't hold fp16 and bf16 copies at once, and casting fp16 -> bf16 -> fp16
        would lose precision, so reload from the HF cache: ~20-30 s per call.
        """
        if self.dtype != torch.float16:
            raise ValueError(f"ColQwen produced NaN embeddings in {self.dtype}")
        print(f"fp16 overflow (NaN) on {len(items)} input(s); re-embedding them in bf16")
        self._load(torch.bfloat16)
        try:
            return [embed_one(item) for item in items]
        finally:
            self._load(self.dtype)

    def _embed_image_batch(self, batch: List[Image.Image]) -> List[Tuple[np.ndarray, Tuple[int, int], int]]:
        merge = self.model.spatial_merge_size
        image_token_id = self.model.config.image_token_id

        inputs = self.processor.process_images(batch).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs)

        results = []
        for emb, mask, ids, (_, h, w) in zip(
            embeddings, inputs["attention_mask"], inputs["input_ids"], inputs["image_grid_thw"]
        ):
            # Drop left padding; padded rows are zero vectors
            keep = mask.bool()
            emb, ids = emb[keep], ids[keep]

            rows, cols = int(h) // merge, int(w) // merge
            image_positions = (ids == image_token_id).nonzero().flatten()
            assert len(image_positions) == rows * cols, "image tokens don't match grid"

            results.append((
                emb.float().cpu().numpy(),
                (rows, cols),
                int(image_positions[0]),
            ))

        if self.device == "cuda":
            torch.cuda.empty_cache()
        return results

    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = 4
    ) -> List[Tuple[np.ndarray, Tuple[int, int], int]]:
        """
        Embed batch of page images into patch embeddings.

        Args:
            images: List of PIL Images
            batch_size: Batch size for processing (GPU memory limit)

        Returns:
            One (embeddings, patch_grid, image_token_start) tuple per image:
            - embeddings: [num_tokens, 128], padding removed, never NaN. Includes
              the prompt's text tokens, which ColQwen also scores with.
            - patch_grid: (rows, cols) of the image tokens
            - image_token_start: row index where the image tokens begin, so
              embeddings[start:start + rows*cols] reshapes to the grid

        Raises:
            ValueError: if an image still embeds to NaN after the bf16 retry
        """
        results = []
        for i in range(0, len(images), batch_size):
            results += self._embed_image_batch(images[i:i + batch_size])

        bad = [i for i, (emb, _, _) in enumerate(results) if np.isnan(emb).any()]
        if bad:
            redone = self._redo_in_bf16([images[i] for i in bad], lambda img: self._embed_image_batch([img])[0])
            if any(np.isnan(emb).any() for emb, _, _ in redone):
                raise ValueError("ColQwen produced NaN embeddings even in bf16")
            for i, result in zip(bad, redone):
                results[i] = result

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed text query into token embeddings.

        Args:
            query: Text query

        Returns:
            Query token embeddings, shape [num_tokens, 128]
        """
        return self.embed_queries([query])[0]

    def _embed_query_batch(self, queries: List[str]) -> List[np.ndarray]:
        inputs = self.processor.process_queries(queries).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs)
        return [
            emb[mask.bool()].float().cpu().numpy()
            for emb, mask in zip(embeddings, inputs["attention_mask"])
        ]

    def embed_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Embed multiple queries.

        Args:
            queries: List of text queries

        Returns:
            List of query embeddings, padding removed, never NaN
        """
        results = self._embed_query_batch(queries)

        bad = [i for i, emb in enumerate(results) if np.isnan(emb).any()]
        if bad:
            redone = self._redo_in_bf16([queries[i] for i in bad], lambda q: self._embed_query_batch([q])[0])
            if any(np.isnan(emb).any() for emb in redone):
                raise ValueError("ColQwen produced NaN embeddings even in bf16")
            for i, emb in zip(bad, redone):
                results[i] = emb

        return results
