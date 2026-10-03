#!/usr/bin/env bash
# Run Mira's GPU tests on a Colab T4 against the LOCAL working tree
# (uncommitted changes included), so you can test before pushing.
#
# Usage: scripts/test_colab.sh [session-name]     (default: mira)
#   Reuses the session if it exists, so re-runs skip VM allocation.
#   Stop it when done: colab stop -s mira
#
# Needs: colab CLI authenticated, .env with QDRANT_CLUSTER_ENDPOINT/_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:-mira}

set -a; source .env; set +a
: "${QDRANT_CLUSTER_ENDPOINT:?missing in .env}" "${QDRANT_CLUSTER_API_KEY:?missing in .env}"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# Tracked + untracked-but-not-ignored files, plus the sample PDF (data/ is gitignored)
{ git ls-files -co --exclude-standard; echo data/samples/test.pdf; } | tar czf "$tmp/mira.tgz" -T -

# `colab status` exits 0 even for a missing session, so check its message
if colab status -s "$S" 2>&1 | grep -q "not found"; then
  colab new -s "$S" --gpu T4
fi
colab upload -s "$S" "$tmp/mira.tgz" /content/mira.tgz

# The kernel is long-lived and won't see a fresh `pip install -e`, and subprocess
# output bypasses it, so everything runs as subprocesses with output relayed.
cat > "$tmp/run.py" <<'PY'
import subprocess, sys

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
run(f"{sys.executable} examples/index_and_search.py data/samples/test.pdf")
print("\nALL PASSED")
PY

colab exec -s "$S" -f "$tmp/run.py" --timeout 3600 \
  --env QDRANT_CLUSTER_ENDPOINT="$QDRANT_CLUSTER_ENDPOINT" \
  --env QDRANT_CLUSTER_API_KEY="$QDRANT_CLUSTER_API_KEY" | tee "$tmp/out.log"

echo "Session '$S' still running. Stop it: colab stop -s $S"

# The kernel swallows SystemExit, so colab exec exits 0 even on failure; check the marker
grep -q "^ALL PASSED" "$tmp/out.log" || { echo "COLAB RUN FAILED"; exit 1; }
