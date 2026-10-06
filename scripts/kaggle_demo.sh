#!/usr/bin/env bash
# Start the Mira demo in a Kaggle notebook, from the index saved by the full-eval kernel.
#
# One-time notebook setup (Kaggle UI): Accelerator "GPU T4", Internet on, and under
# "Add Input" add the dataset <you>/mira-samples and the notebook output of <you>/mira-full.
# Then in a cell:
#   !git clone -q --depth 1 https://github.com/Va1bhav512/mira /kaggle/working/mira
#   !bash /kaggle/working/mira/scripts/kaggle_demo.sh
# It prints a public https://*.gradio.live link (valid while the notebook runs; ~3 min to start).
#
# --smoke: serve without the share link and run scripts/demo_smoke.py instead (used by the
# batch runner to test this script: test_kaggle.sh --kernel-output <you>/mira-full --run ...).
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE=${CACHE:-/tmp/cache}
mkdir -p "$CACHE"

snapshot=$(find /kaggle/input -name mira_pages.snapshot | head -1)
bm25=$(dirname "$(find /kaggle/input -path '*index/bm25/pages.jsonl' | head -1)")
samples=$(dirname "$(find /kaggle/input -name test.pdf | head -1)")
: "${snapshot:?add the mira-full notebook output as an input}" "${samples:?add the mira-samples dataset as an input}"

if [[ ! -e data/samples ]]; then ln -s "$samples" data/samples; fi
if ! python -c "import mira.serve" 2>/dev/null; then
  pip install -q -e '.[generation,demo]'
  pip uninstall -q -y torchao  # transformers 5 refuses old torchao
fi

if ! curl -sf localhost:6333/readyz >/dev/null; then
  mkdir -p "$CACHE/qdrant-bin"
  [[ -x $CACHE/qdrant-bin/qdrant ]] || curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C "$CACHE/qdrant-bin"
  QDRANT__STORAGE__STORAGE_PATH="$CACHE/qdrant" nohup "$CACHE/qdrant-bin/qdrant" > "$CACHE/qdrant.log" 2>&1 &
  for _ in $(seq 60); do curl -sf localhost:6333/readyz >/dev/null && break; sleep 1; done
fi
if ! curl -sf localhost:6333/collections/mira_pages >/dev/null; then
  echo "Restoring the index snapshot ($(du -h "$snapshot" | cut -f1))..."
  curl -sf -X POST "localhost:6333/collections/mira_pages/snapshots/upload?priority=snapshot" -F "snapshot=@$snapshot" >/dev/null
fi

if [[ ${1:-} == --smoke ]]; then
  python -m mira.serve --qdrant-url http://localhost:6333 --bm25-path "$bm25" > "$CACHE/serve.log" 2>&1 &
  server=$!
  for _ in $(seq 120); do curl -sf localhost:7860/docs >/dev/null && break; sleep 5; done
  python scripts/demo_smoke.py --out "${OUT:-$CACHE}/demo_smoke"
  kill $server
else
  python -m mira.serve --qdrant-url http://localhost:6333 --bm25-path "$bm25" --share
fi
