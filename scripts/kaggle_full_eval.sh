#!/usr/bin/env bash
# The full evaluation in one Kaggle run, plus the demo index:
#   scripts/test_kaggle.sh --dataset <owner>/mira-custom-embeddings \
#     --run 'bash scripts/kaggle_full_eval.sh' mira-full
# Runs inside test_kaggle.sh's --run mode: $OUT is kept (downloaded to .cache/kaggle/mira-full/),
# $CACHE is scratch, Qdrant is up on localhost:6333.
#
# 1. Custom corpus: index (from the embedding checkpoint dataset if mounted), retrieval eval,
#    and a Qdrant snapshot + BM25 index for the demo.
# 2. ViDoRe V3 subsets: index, retrieval eval, cropping eval, generation on GEN_LIMIT queries.
# 3. Judge every generated answer.
# Every retrieval eval also dumps per-query channel rankings, so fusion can be re-scored on CPU.
set -euo pipefail
URL="--qdrant-url http://localhost:6333"
SUBSETS=${SUBSETS:-hr computer_science pharmaceuticals}
GEN_LIMIT=${GEN_LIMIT:-60}
mkdir -p "$CACHE" "$OUT/index"

# Kaggle extracts uploaded tarballs, so the checkpoint dataset mounts as an embeddings/ folder
emb=$(find /kaggle/input -type d -name embeddings 2>/dev/null | head -1)
if [[ -n $emb ]]; then cp -r "$emb" "$CACHE/embeddings"; fi
python scripts/index_corpus.py $URL --bm25-path "$OUT/index/bm25" --cache-dir "$CACHE/embeddings"
python scripts/eval_retrieval.py $URL --bm25-path "$OUT/index/bm25" --dump-rankings "$OUT/rankings_custom.jsonl"
name=$(curl -sf -X POST localhost:6333/collections/mira_pages/snapshots | python -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')
curl -sf "localhost:6333/collections/mira_pages/snapshots/$name" -o "$OUT/index/mira_pages.snapshot"
ls -la "$OUT/index"

for s in $SUBSETS; do
  bm25="$CACHE/bm25_vidore/$s"
  python scripts/index_vidore.py --subset "$s" $URL --bm25-path "$bm25" --cache-dir "$CACHE/embeddings_vidore/$s"
  python scripts/eval_retrieval.py --vidore "$s" $URL --bm25-path-vidore "$bm25" --dump-rankings "$OUT/rankings_$s.jsonl"
  python scripts/eval_cropping.py --vidore "$s" $URL
  python scripts/eval_generation.py --vidore "$s" $URL --bm25-path "$bm25" --limit "$GEN_LIMIT" --out "$OUT/generation_$s.jsonl"
done

python scripts/judge_answers.py "$OUT"/generation_*.jsonl

# Demo end to end on the GPU: API + UI over the custom corpus (overlays saved for inspection)
python -m mira.serve --qdrant-url http://localhost:6333 --bm25-path "$OUT/index/bm25" > "$OUT/serve.log" 2>&1 &
for _ in $(seq 120); do curl -sf localhost:7860/docs >/dev/null && break; sleep 5; done
python scripts/demo_smoke.py --out "$OUT/demo_smoke"
kill %1
