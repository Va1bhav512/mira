"""
Grounded answer generation (Phase 6) with Qwen2.5-VL.

The VLM sees labelled evidence images ([E1], [E2], ...) and returns JSON
{"answer": ..., "evidence_ids": [...]}. It never produces coordinates: the
retriever already knows each evidence id's page and box, so citations are
built deterministically from the ids it picks (see answer.py).
"""

import json
import re
from typing import List, Optional, Tuple

import torch
from PIL import Image

PROMPT = """Answer the question using only the evidence images above. Each image is labelled [E1], [E2], ...
First list the labels of every image that contains information you use, then give the answer.
Reply with JSON only: {{"evidence_ids": [<labels, e.g. "E1">], "answer": "<answer>"}}
Use an empty evidence_ids list only if no image contains the answer, and then say so in "answer".

Question: {query}"""

# ponytail: caps image tokens per evidence image (~1000) so 3 crops fit a T4 / 4 GB GPU; raise with VRAM
MAX_PIXELS = 1024 * 28 * 28


def parse_reply(reply: str, known_ids: List[str]) -> Tuple[str, List[str]]:
    """
    (answer, evidence_ids) from the model's reply. Unknown ids are dropped; a reply
    that isn't the requested JSON becomes the answer verbatim, with no ids.

    ponytail: parse-and-fallback instead of constrained decoding; add a JSON grammar if
    the fallback rate on eval runs is material.
    """
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return reply.strip(), []
    ids = data.get("evidence_ids")
    ids = ids if isinstance(ids, list) else []
    return data["answer"].strip(), [i for i in ids if i in known_ids]


class VLMGenerator:
    """Qwen2.5-VL answering over evidence images; 4-bit on GPU so it fits beside ColQwen on a T4."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = None,
        quantize: bool = True,
        max_pixels: int = MAX_PIXELS,
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None for auto
            quantize: 4-bit NF4 weights via bitsandbytes (GPU only): ~2.5 GB instead of ~7.5 GB
            max_pixels: Per-image pixel budget for the processor
        """
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if device == "cuda":
            # fp16 Qwen2.5-VL is known to overflow; bf16 needs Ampere+ (a T4 would only emulate it)
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        kwargs = {"device_map": device} if device == "cuda" else {}
        if quantize and device == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)

    def generate(
        self, query: str, evidence: List[Tuple[str, Image.Image]], max_new_tokens: int = 256
    ) -> Tuple[str, List[str], str]:
        """
        Args:
            query: The question
            evidence: (evidence_id, image) pairs, e.g. [("E1", crop), ...]
            max_new_tokens: Generation budget

        Returns:
            (answer, cited evidence ids, raw model reply)
        """
        content = []
        for evidence_id, image in evidence:
            content += [{"type": "text", "text": f"[{evidence_id}]"}, {"type": "image", "image": image}]
        content.append({"type": "text", "text": PROMPT.format(query=query)})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if self.device == "cuda":
            torch.cuda.empty_cache()

        answer, ids = parse_reply(reply, [evidence_id for evidence_id, _ in evidence])
        return answer, ids, reply
