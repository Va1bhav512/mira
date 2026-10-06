"""
Answer-correctness judge: a local instruction model grades a generated answer against
a reference answer (text only, no images). Verdicts: correct (1), partial (0.5), incorrect (0).

ponytail: one local 7B judge, unvalidated. Before quoting its numbers, hand-grade ~30
answers and report agreement with it.
"""

import json
import re
from typing import List, Tuple

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

PROMPT = """You are grading an answer to a question about documents, against a reference answer written by a human.

Question: {question}

Reference answer: {reference}

Answer to grade: {answer}

Grade the answer:
- "correct": it states the reference's key facts and contradicts none of them (extra correct detail is fine)
- "partial": it states some of the key facts but misses or gets others wrong
- "incorrect": it misses the key facts, contradicts the reference, or says the information is not available

Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""


def parse_verdict(reply: str) -> Tuple[str, str]:
    """(verdict, reason) from the judge's reply; 'invalid' if no known verdict can be read."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    if verdict not in SCORES:
        # Fall back to the first verdict word in the reply ("incorrect" before "correct": it contains it)
        found = re.search(r"\b(incorrect|partial|correct)\b", reply.lower())
        verdict = found.group(1) if found else "invalid"
    reason = data.get("reason", "") if isinstance(data, dict) else ""
    return verdict, str(reason)


class Judge:
    """Batched greedy grading with a 4-bit instruction model on GPU."""

    def __init__(self, model_name: str = JUDGE_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda", dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype),
        ).eval()

    def grade(self, items: List[dict], batch_size: int = 8) -> List[Tuple[str, str]]:
        """items: dicts with question, reference, answer. Returns (verdict, reason) per item."""
        import torch

        out = []
        for i in range(0, len(items), batch_size):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT.format(**item)}], tokenize=False, add_generation_prompt=True
                )
                for item in items[i:i + batch_size]
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            replies = self.tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [parse_verdict(r) for r in replies]
        return out
