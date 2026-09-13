from PIL import Image
import numpy as np
from typing import List, Optional
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ColQwenEmbedder:
    """
    Wrapper for ColQwen2.5 visual embedding model.
    Optimized for free-tier GPUs (T4, Colab/Kaggle).
    """
    
    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: Optional[str] = None,
        use_4bit: bool = True
    ):
        """
        Initialize ColQwen embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', None=auto)
            use_4bit: Use 4-bit quantization for memory efficiency
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load model with memory optimizations
        if use_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = ColQwen2_5.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = ColQwen2_5.from_pretrained(model_name)
            self.model = self.model.to(device)
        
        self.model.eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
    
    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = 4
    ) -> List[np.ndarray]:
        """
        Embed batch of page images into patch embeddings.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing (GPU memory limit)
        
        Returns:
            List of embeddings, each shape [num_patches, 128]
        """
        all_embeddings = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process images
            inputs = self.processor.process_images(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs)
            
            # Convert to numpy
            for emb in embeddings:
                all_embeddings.append(emb.cpu().numpy())
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed text query into token embeddings.
        
        Args:
            query: Text query
        
        Returns:
            Query token embeddings, shape [num_tokens, 128]
        """
        inputs = self.processor.process_queries([query])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = self.model(**inputs)
        
        return embeddings[0].cpu().numpy()
    
    def embed_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Embed multiple queries.
        
        Args:
            queries: List of text queries
        
        Returns:
            List of query embeddings
        """
        inputs = self.processor.process_queries(queries)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = self.model(**inputs)
        
        return [emb.cpu().numpy() for emb in embeddings]

