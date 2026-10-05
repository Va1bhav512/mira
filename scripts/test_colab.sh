#!/usr/bin/env bash
# Run Mira's GPU tests on a Colab T4 against the LOCAL working tree
# (uncommitted changes included), so you can test before pushing.
#
# Usage: scripts/test_colab.sh [--eval] [session-name]     (default session: mira)
#   Default: pytest + index/search example on test.pdf (cloud Qdrant).
#   --eval:  pytest, then index all of data/samples into an on-VM Qdrant + BM25 under
#            /content/cache (resumable; ~1 h first run) and run eval_retrieval.py on all
#            modes. Output saved to data/eval/colab_eval.log.
#   Reuses the session if it exists, so re-runs skip VM allocation (and, with --eval,
#   reuse the on-VM index). Stop it when done: colab stop -s mira
#
# Needs: colab CLI authenticated, .env with QDRANT_CLUSTER_ENDPOINT/_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
EVAL=0
if [[ ${1:-} == --eval ]]; then EVAL=1; shift; fi
S=${1:-mira}

set -a; source .env; set +a
: "${QDRANT_CLUSTER_ENDPOINT:?missing in .env}" "${QDRANT_CLUSTER_API_KEY:?missing in .env}"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# Tracked + untracked-but-not-ignored files, plus sample PDFs (data/samples/ is gitignored):
# just test.pdf normally, the whole corpus (~72 MB) for --eval
if (( EVAL )); then samples=(data/samples/*.pdf); else samples=(data/samples/test.pdf); fi
{ git ls-files -co --exclude-standard; printf '%s\n' "${samples[@]}"; } | tar czf "$tmp/mira.tgz" -T -

# `colab status` exits 0 even for a missing session, so check its message
if colab status -s "$S" 2>&1 | grep -q "not found"; then
  colab new -s "$S" --gpu T4
fi
colab upload -s "$S" "$tmp/mira.tgz" /content/mira.tgz

# The kernel is long-lived and won't see a fresh `pip install -e`, and subprocess
# output bypasses it, so everything runs as subprocesses with output relayed.
cat > "$tmp/run.py" <<'PY'
import os, subprocess, sys

def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd="/content/mira", text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}")

subprocess.run("rm -rf /content/mira && mkdir /content/mira && tar xzf /content/mira.tgz -C /content/mira",
               shell=True, check=True)
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run(f"{sys.executable} -m pip install -q -e '.[dev]'")
# Colab ships torchao 0.10; transformers 5 refuses to load any model with torchao < 0.16 present
run(f"{sys.executable} -m pip uninstall -q -y torchao")
run(f"{sys.executable} -m pytest -v -rs tests")
if os.environ.get("MIRA_EVAL") == "1":
    # Index lives outside /content/mira (wiped each run) so re-runs resume instead of re-embedding
    # Qdrant server on the VM (single binary, no Docker); storage persists under /content/cache.
    # Not embedded mode: that unpickles every point into RAM and OOMs a 12 GB VM at full corpus.
    if subprocess.run("curl -sf localhost:6333/readyz", shell=True).returncode:
        run("mkdir -p /content/cache/qdrant-bin && cd /content/cache/qdrant-bin && { [ -x qdrant ] || "
            "curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz; }")
        subprocess.Popen("QDRANT__STORAGE__STORAGE_PATH=/content/cache/qdrant-server /content/cache/qdrant-bin/qdrant"
                         " > /content/cache/qdrant.log 2>&1", shell=True, start_new_session=True)
        run("for i in $(seq 60); do curl -sf localhost:6333/readyz && exit 0; sleep 1; done; exit 1")
    paths = "--qdrant-url http://localhost:6333 --bm25-path /content/cache/bm25"
    run(f"{sys.executable} scripts/index_corpus.py {paths} --cache-dir /content/cache/embeddings")
    run(f"{sys.executable} scripts/eval_retrieval.py {paths}")
else:
    run(f"{sys.executable} examples/index_and_search.py data/samples/test.pdf")
print("\nALL PASSED")
PY

# MIRA_EVAL is always passed: kernel env persists, so a stale "1" would leak into a later plain run
colab exec -s "$S" -f "$tmp/run.py" --timeout $(( EVAL ? 14400 : 3600 )) \
  --env MIRA_EVAL="$EVAL" \
  --env QDRANT_CLUSTER_ENDPOINT="$QDRANT_CLUSTER_ENDPOINT" \
  --env QDRANT_CLUSTER_API_KEY="$QDRANT_CLUSTER_API_KEY" | tee "$tmp/out.log"

if (( EVAL )); then cp "$tmp/out.log" data/eval/colab_eval.log; echo "Saved data/eval/colab_eval.log"; fi

echo "Session '$S' still running. Stop it: colab stop -s $S"

# The kernel swallows SystemExit, so colab exec exits 0 even on failure; check the marker
grep -q "^ALL PASSED" "$tmp/out.log" || { echo "COLAB RUN FAILED"; exit 1; }
