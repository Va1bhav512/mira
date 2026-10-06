"""Qwen2.5-VL generation on a rendered image (GPU + bitsandbytes; ~3 GB VRAM in 4-bit)."""

import pytest
from PIL import Image, ImageDraw, ImageFont


@pytest.fixture(scope="module")
def generator():
    import torch
    # ~3 GB of 4-bit weights plus activations; also keeps a default local run on a small
    # GPU from downloading the ~7.5 GB checkpoint
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 6e9:
        pytest.skip("VLM test needs a GPU with >=6 GB VRAM")
    pytest.importorskip("bitsandbytes")
    from mira.generation import VLMGenerator
    return VLMGenerator()


def text_image(text):
    img = Image.new("RGB", (800, 200), "white")
    ImageDraw.Draw(img).text((20, 70), text, fill="black", font=ImageFont.load_default(size=40))
    return img


@pytest.mark.slow
def test_answers_from_cited_evidence(generator):
    evidence = [("E1", text_image("Lunch menu: soup, bread")), ("E2", text_image("Supply voltage: 3.3 V"))]
    answer, ids, raw = generator.generate("What is the supply voltage?", evidence)
    assert "3.3" in answer, raw
    assert ids == ["E2"], raw
