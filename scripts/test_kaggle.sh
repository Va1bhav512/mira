#!/usr/bin/env bash
# Run Mira's GPU tests on Kaggle against the LOCAL working tree
# (uncommitted changes included), so you can test before pushing.
#
# Usage: scripts/test_kaggle.sh [--eval] [notebook-name]     (default name: mira)
#   Default: pytest + index/search example on test.pdf (cloud Qdrant).
#   --eval:  pytest, then index all of data/samples into a Qdrant server + BM25 on the VM
#            and run eval_retrieval.py on all modes. Output saved to data/eval/kaggle_eval.log.
#            After indexing, the embedding cache (~500 MB) is downloaded to
#            .cache/kaggle/embeddings.tar and re-uploaded on later runs, so a reclaimed VM
#            costs an upload + re-render/OCR (~20 min) instead of ~1 h of re-embedding.
#
# Needs: kaggle.json in ~/.kaggle/ or KAGGLE_KEY/KAGGLE_USERNAME env vars,
#        .env with QDRANT_CLUSTER_ENDPOINT/_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
EVAL=0
if [[ ${1:-} == --eval ]]; then EVAL=1; shift; fi
N=${1:-mira-test}
CHECKPOINT=.cache/kaggle/embeddings.tar

set -a; source .env; set +a
: "${QDRANT_CLUSTER_ENDPOINT:?missing in .env}" "${QDRANT_CLUSTER_API_KEY:?missing in .env}"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# Tracked + untracked-but-not-ignored files, plus sample PDFs.
# Exclude .env (secrets) and .cache (large local files).
rm -rf data/samples 2>/dev/null || true
mkdir -p data/samples
if (( EVAL )); then  
  cp -L /home/vaibhav/g/mira/data/samples/*.pdf data/samples/
else
  cp -L /home/vaibhav/g/mira/data/samples/test.pdf data/samples/
fi
samples=(data/samples/*.pdf)
{ git ls-files -c --exclude-standard; git ls-files -o --exclude-standard; printf '%s\n' "${samples[@]}"; } | grep -v '\.env$' | grep -v '\.cache/' | tar czf "$tmp/mira.tgz" -T -

KAGGLE_USER=$(kaggle config view | grep username | awk '{print $3}')
: "${KAGGLE_USER:?Could not get Kaggle username from config}"

# Create kernel metadata with dataset dependency
cat > "$tmp/kernel-metadata.json" <<METADATA
{
  "title": "$N",
  "id": "$KAGGLE_USER/$N",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$KAGGLE_USER/$N-tarball"],
  "competition_sources": [],
  "kernel_sources": []
}
METADATA

# Create dataset metadata for tarball
cat > "$tmp/dataset-metadata.json" <<DATAMETA
{
  "title": "$N-tarball",
  "id": "$KAGGLE_USER/$N-tarball",
  "licenses": [{"name": "CC0-1.0"}]
}
DATAMETA

# Upload tarball as dataset (create or update)
if kaggle datasets list --user "$KAGGLE_USER" --search "$N-tarball" -v 2>/dev/null | grep -q "$N-tarball"; then
  echo "Updating dataset '$N-tarball'..."
  cp "$tmp/dataset-metadata.json" "$tmp/mira.tgz" .
  kaggle datasets version -p . -m "Working tree $(date +%Y%m%d-%H%M%S)" --quiet
else
  echo "Creating dataset '$N-tarball'..."
  cp "$tmp/dataset-metadata.json" "$tmp/mira.tgz" .
  kaggle datasets create -p . --quiet
fi

# Upload checkpoint if it exists
if (( EVAL )) && [[ -f $CHECKPOINT ]]; then
  cat > "$tmp/checkpoint-metadata.json" <<CHKMETA
{
  "title": "$N-embeddings",
  "id": "$KAGGLE_USER/$N-embeddings",
  "licenses": [{"name": "CC0-1.0"}]
}
CHKMETA
  if kaggle datasets list --user "$KAGGLE_USER" --search "$N-embeddings" -v 2>/dev/null | grep -q "$N-embeddings"; then
    kaggle datasets version -p "$CHECKPOINT" -m "Embeddings checkpoint $(date +%Y%m%d-%H%M%S)" --quiet 2>/dev/null || true
  else
    kaggle datasets create -p "$CHECKPOINT" --quiet
  fi
fi

# Shared helpers for both stages
cat > "$tmp/common.py" <<'PY'
import os, subprocess, sys

def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd="/kaggle/working/mira", text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}")

PATHS = "--qdrant-url http://localhost:6333 --bm25-path /kaggle/working/cache/bm25"
PY

# Stage 1: install, test, and (--eval) index the corpus, then pack the embedding cache
{ cat "$tmp/common.py"; cat <<'PY'
import tarfile, glob, os
# Find and extract the uploaded tarball (dataset slug replaces spaces with dashes)
tarball = glob.glob("/kaggle/input/*/mira.tgz")[0]
subprocess.run(f"rm -rf /kaggle/working/mira && mkdir -p /kaggle/working/mira && tar xzf {tarball} -C /kaggle/working/mira", shell=True, check=True)
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run(f"{sys.executable} -m pip install -q -e '/kaggle/working/mira[dev]'")
run(f"{sys.executable} -m pip uninstall -q -y torchao")
run(f"{sys.executable} -m pytest -v -rs /kaggle/working/mira/tests")
if os.environ.get("MIRA_EVAL") == "1":
    # Restore checkpoint if available
    checkpoint_files = glob.glob("/kaggle/input/*embeddings*/embeddings.tar")
    if checkpoint_files and not os.path.exists("/kaggle/working/cache/embeddings"):
        run(f"mkdir -p /kaggle/working/cache && tar xf {checkpoint_files[0]} -C /kaggle/working/cache")
    # Start Qdrant server
    if subprocess.run("curl -sf localhost:6333/readyz", shell=True).returncode:
        run("mkdir -p /kaggle/working/cache/qdrant-bin && cd /kaggle/working/cache/qdrant-bin && { [ -x qdrant ] || "
            "curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz; }")
        subprocess.Popen("QDRANT__STORAGE__STORAGE_PATH=/kaggle/working/cache/qdrant-server /kaggle/working/cache/qdrant-bin/qdrant"
                         " > /kaggle/working/cache/qdrant.log 2>&1", shell=True, start_new_session=True)
        run("for i in $(seq 60); do curl -sf localhost:6333/readyz && exit 0; sleep 1; done; exit 1")
    run(f"{sys.executable} /kaggle/working/mira/scripts/index_corpus.py {PATHS} --cache-dir /kaggle/working/cache/embeddings")
    run("tar cf /kaggle/working/embeddings.tar -C /kaggle/working/cache embeddings && ls -la /kaggle/working/embeddings.tar")
    print("\nSTAGE 1 DONE")
else:
    run(f"{sys.executable} /kaggle/working/mira/examples/index_and_search.py /kaggle/working/mira/data/samples/test.pdf")
    print("\nALL PASSED")
PY
} > "$tmp/script.py"

# Push kernel
echo "Pushing kernel '$N'..."
mkdir -p "$tmp/kernel"
cp "$tmp/kernel-metadata.json" "$tmp/script.py" "$tmp/kernel/"
kaggle kernels push -p "$tmp/kernel"

# Poll for completion (kernel push starts it automatically)
echo "Waiting for kernel completion..."
STATUS=""
for i in $(seq 120); do
  sleep 30
  STATUS=$(kaggle kernels status "$KAGGLE_USER/$N" 2>&1 || echo "unknown")
  echo "[$i/120] Status: $STATUS"
  if grep -qi "complete\|success\|fail\|error" <<<"$STATUS"; then
    break
  fi
done

# Get output logs
echo "Fetching output..."
kaggle kernels output "$KAGGLE_USER/$N" -p "$tmp" > "$tmp/out.log" 2>&1 || true

if (( EVAL )); then
  mkdir -p data/eval
  cp "$tmp/out.log" data/eval/kaggle_eval.log
  echo "Saved data/eval/kaggle_eval.log"
fi

grep -q "ALL PASSED" "$tmp/out.log" || { echo "KAGGLE RUN FAILED"; exit 1; }
echo "Success. Output saved to: $tmp/out.log"
