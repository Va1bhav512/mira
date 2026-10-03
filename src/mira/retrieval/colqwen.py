from PIL import Image
import numpy as np
from typing import List, Optional, Tuple
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ColQwenEmbedder:
    """
    Wrapper for ColQwen2.5 visual embedding model.
    Runs in fp16 on GPU (~7 GB for the 3B model, fits a T4) and fp32 on CPU.
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

        self.device = device

        # T4 has no bf16 support, so fp16 on GPU
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = ColQwen2_5.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

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
            - embeddings: [num_tokens, 128], padding removed. Includes the
              prompt's text tokens, which ColQwen also scores with.
            - patch_grid: (rows, cols) of the image tokens
            - image_token_start: row index where the image tokens begin, so
              embeddings[start:start + rows*cols] reshapes to the grid
        """
        merge = self.model.spatial_merge_size
        image_token_id = self.model.config.image_token_id
        results = []

        # Process in batches to avoid OOM
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            inputs = self.processor.process_images(batch).to(self.device)

            with torch.no_grad():
                embeddings = self.model(**inputs)

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

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed text query into token embeddings.

        Args:
            query: Text query

        Returns:
            Query token embeddings, shape [num_tokens, 128]
        """
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Embed multiple queries.

        Args:
            queries: List of text queries

        Returns:
            List of query embeddings, padding removed
        """
        inputs = self.processor.process_queries(queries).to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs)

        return [
            emb[mask.bool()].float().cpu().numpy()
            for emb, mask in zip(embeddings, inputs["attention_mask"])
        ]
