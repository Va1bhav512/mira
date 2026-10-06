#!/usr/bin/env python
"""
End-to-end check of a running `python -m mira.serve` (no --share): calls POST /query and the
UI's query endpoint, and saves the JSON plus the page overlays and crops the UI would show.

Usage:
    python scripts/demo_smoke.py --url http://localhost:7860 --out $OUT/demo_smoke
"""

import argparse
import json
import shutil
from pathlib import Path

import requests
from gradio_client import Client

QUESTIONS = [
    "what does register 0x2D control on the accelerometer",
    "internal block diagram of the microcontroller with two DMA controllers",
    "how can the accelerometer save power by itself when nothing is moving",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    api = [requests.post(f"{args.url}/query", json={"question": q}, timeout=300).json() for q in QUESTIONS]
    (out / "api.json").write_text(json.dumps(api, indent=1))
    for q, r in zip(QUESTIONS, api):
        print(f"\nQ: {q}\nA: {r['answer']}\nsources: {r['sources']}\nretrieved: {r['retrieved']}\ntimings: {r['timings']}")

    ui = Client(f"{args.url}/ui/")
    for i, q in enumerate(QUESTIONS):
        answer, pages, crops, _, _ = ui.predict(q, "adaptive", "heatmap", 3, "All documents", api_name="/query")
        for kind, gallery in (("page", pages), ("crop", crops)):
            for j, item in enumerate(gallery):
                shutil.copy(item["image"]["path"] if isinstance(item["image"], dict) else item["image"],
                            out / f"q{i}_{kind}{j}.png")
        print(f"UI q{i}: {len(pages)} page overlays, {len(crops)} crops")


if __name__ == "__main__":
    main()
