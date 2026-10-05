#!/usr/bin/env bash
# Run Mira's GPU tests on a Colab T4 against the LOCAL working tree
# (uncommitted changes included), so you can test before pushing.
#
# Usage: scripts/test_colab.sh [--eval] [session-name]     (default session: mira)
#   Default: pytest + index/search example on test.pdf (cloud Qdrant).
#   --eval:  pytest, then index all of data/samples into a Qdrant server + BM25 on the VM
#            and run eval_retrieval.py on all modes. Output saved to data/eval/colab_eval.log.
#            After indexing, the embedding cache (~500 MB) is downloaded to
#            .cache/colab/embeddings.tar and re-uploaded on later runs, so a reclaimed VM
#            costs an upload + re-render/OCR (~20 min) instead of ~1 h of re-embedding.
#   Reuses the session if it exists. Stop it when done: colab stop -s mira
#
# Needs: colab CLI authenticated, .env with QDRANT_CLUSTER_ENDPOINT/_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
EVAL=0
if [[ ${1:-} == --eval ]]; then EVAL=1; shift; fi
S=${1:-mira}
CHECKPOINT=.cache/colab/embeddings.tar

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
if (( EVAL )) && [[ -f $CHECKPOINT ]]; then
  colab upload -s "$S" "$CHECKPOINT" /content/embeddings.tar
fi

# Each stage runs as its own detached process on the VM (see remote_stage), so a fresh
# `pip install -e` is visible to the steps after it. Shared helpers for both stages:
cat > "$tmp/common.py" <<'PY'
import os, subprocess, sys

def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd="/content/mira", text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}")

PATHS = "--qdrant-url http://localhost:6333 --bm25-path /content/cache/bm25"
PY

# Stage 1: install, test, and (--eval) index the corpus, then pack the embedding cache
{ cat "$tmp/common.py"; cat <<'PY'
subprocess.run("rm -rf /content/mira && mkdir /content/mira && tar xzf /content/mira.tgz -C /content/mira",
               shell=True, check=True)
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run(f"{sys.executable} -m pip install -q -e '.[dev]'")
# Colab ships torchao 0.10; transformers 5 refuses to load any model with torchao < 0.16 present
run(f"{sys.executable} -m pip uninstall -q -y torchao")
run(f"{sys.executable} -m pytest -v -rs tests")
if os.environ.get("MIRA_EVAL") == "1":
    # Restore a checkpoint uploaded from a previous VM (index lives outside the wiped /content/mira)
    if os.path.exists("/content/embeddings.tar") and not os.path.exists("/content/cache/embeddings"):
        run("mkdir -p /content/cache && tar xf /content/embeddings.tar -C /content/cache")
    # Qdrant server on the VM (single binary, no Docker); storage persists under /content/cache.
    # Not embedded mode: that unpickles every point into RAM and OOMs a 12 GB VM at full corpus.
    if subprocess.run("curl -sf localhost:6333/readyz", shell=True).returncode:
        run("mkdir -p /content/cache/qdrant-bin && cd /content/cache/qdrant-bin && { [ -x qdrant ] || "
            "curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz; }")
        subprocess.Popen("QDRANT__STORAGE__STORAGE_PATH=/content/cache/qdrant-server /content/cache/qdrant-bin/qdrant"
                         " > /content/cache/qdrant.log 2>&1", shell=True, start_new_session=True)
        run("for i in $(seq 60); do curl -sf localhost:6333/readyz && exit 0; sleep 1; done; exit 1")
    run(f"{sys.executable} scripts/index_corpus.py {PATHS} --cache-dir /content/cache/embeddings")
    run("tar cf /content/embeddings.tar -C /content/cache embeddings && ls -la /content/embeddings.tar")
    print("\nSTAGE 1 DONE")
else:
    run(f"{sys.executable} examples/index_and_search.py data/samples/test.pdf")
    print("\nALL PASSED")
PY
} > "$tmp/stage1.py"

# Stage 2 (--eval): retrieval eval over the indexed corpus
{ cat "$tmp/common.py"; cat <<'PY'
run(f"{sys.executable} scripts/eval_retrieval.py {PATHS}")
print("\nALL PASSED")
PY
} > "$tmp/stage2.py"

# Run a stage detached on the VM and follow its log by polling. A single long `colab exec`
# isn't safe: its output stream stalled mid-run twice while the job kept going, and the
# client never returned. Polls are short, so a stalled one just times out and retries.
remote_stage() {  # $1 = local stage file, $2 = stage name
  local name=$2 offset=0 status=running out
  colab upload -s "$S" "$1" "/content/$name.py" >/dev/null
  # MIRA_EVAL always passed: kernel env persists, and the detached job inherits it
  printf '%s\n' "import subprocess" \
    "subprocess.Popen('cd /content && rm -f $name.status && python3 -u $name.py > $name.log 2>&1; echo \$? > $name.status', shell=True, start_new_session=True)" \
    > "$tmp/launch.py"
  colab exec -s "$S" -f "$tmp/launch.py" --timeout 60 \
    --env MIRA_EVAL="$EVAL" \
    --env QDRANT_CLUSTER_ENDPOINT="$QDRANT_CLUSTER_ENDPOINT" \
    --env QDRANT_CLUSTER_API_KEY="$QDRANT_CLUSTER_API_KEY" >/dev/null
  while [[ $status == running ]]; do
    sleep 30
    printf '%s\n' "import os" \
      "d = open('/content/$name.log', 'rb').read() if os.path.exists('/content/$name.log') else b''" \
      "print(d[$offset:].decode(errors='replace'), end='')" \
      "print('\n@@OFFSET', len(d))" \
      "print('@@STATUS', open('/content/$name.status').read().strip() if os.path.exists('/content/$name.status') else 'running')" \
      > "$tmp/poll.py"
    out=$(timeout 120 colab exec -s "$S" -f "$tmp/poll.py" --timeout 60 2>&1) || continue
    if grep -q "not found" <<<"$out" && ! grep -q "^@@STATUS" <<<"$out"; then
      echo "Colab session '$S' is gone"; return 1
    fi
    grep -q "^@@STATUS" <<<"$out" || continue
    { grep -v -e "^@@OFFSET" -e "^@@STATUS" <<<"$out" || true; } | tee -a "$tmp/out.log"
    offset=$(sed -n 's/^@@OFFSET //p' <<<"$out")
    status=$(sed -n 's/^@@STATUS //p' <<<"$out")
  done
  [[ $status == 0 ]]
}

remote_stage "$tmp/stage1.py" stage1 || { echo "COLAB RUN FAILED (stage 1)"; exit 1; }

if (( EVAL )); then
  # Checkpoint before the eval, so a VM reclaimed mid-eval doesn't cost the indexing
  mkdir -p "$(dirname "$CHECKPOINT")"
  colab download -s "$S" /content/embeddings.tar "$CHECKPOINT.part" && mv "$CHECKPOINT.part" "$CHECKPOINT"
  echo "Checkpoint saved: $CHECKPOINT ($(du -h "$CHECKPOINT" | cut -f1))"
  remote_stage "$tmp/stage2.py" stage2 || { echo "COLAB RUN FAILED (stage 2)"; exit 1; }
  cp "$tmp/out.log" data/eval/colab_eval.log; echo "Saved data/eval/colab_eval.log"
fi

echo "Session '$S' still running. Stop it: colab stop -s $S"
grep -q "^ALL PASSED" "$tmp/out.log" || { echo "COLAB RUN FAILED"; exit 1; }
