#!/usr/bin/env bash
# Run Mira's GPU tests and evals on a Kaggle T4 against the LOCAL working tree
# (uncommitted changes included). Kaggle runs the whole job as one batch kernel (up to 12 h,
# not reclaimed mid-run like free Colab), so there are no stages or checkpoints here.
#
# Usage: scripts/test_kaggle.sh [--eval | --eval-vidore subset[,subset...] | --run 'command']
#                               [--dataset owner/slug]... [--kernel-output owner/kernel]... [--no-tests] [kernel-name]   (default: mira-gpu)
#   Default:       pytest + index/search example on test.pdf
#   --eval:        pytest, index all of data/samples, eval_retrieval.py on all modes
#   --eval-vidore: pytest, then per ViDoRe V3 subset: index, retrieval eval, cropping eval
#                  (Phase 5) and generation on 50 queries (Phase 6)
#   --run:         pytest, then a shell command in the repo, with $OUT (kept as kernel output,
#                  downloaded to .cache/kaggle/<kernel-name>/), $CACHE (scratch) and Qdrant up
#   --dataset:     mount an extra private Kaggle dataset under /kaggle/input
#   --kernel-output: mount another kernel's output (e.g. the demo index from mira-full)
#   Log saved to data/eval/kaggle_<mode>.log; generation answers to data/eval/generation_<subset>.jsonl.
#
# What goes to Kaggle (private):
#   - the code, embedded in the kernel script (git ls-files minus docs and past eval outputs,
#     which pushed it past Kaggle's script size limit; .env is gitignored and also excluded)
#   - data/samples PDFs, as dataset mira-samples, uploaded once. Delete it on Kaggle
#     to re-upload after changing the PDFs.
# Qdrant runs on the Kaggle VM for every mode, so no credentials ever leave this machine.
#
# Needs: kaggle CLI with an API token, a phone-verified account (kernels need internet).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=test SUBSETS="" CMD="" TESTS=1 DATASETS="" KERNELS="" N=mira-gpu  # Kaggle titles need >= 5 chars
while (( $# )); do
  case $1 in
    --eval) MODE=eval ;;
    --eval-vidore) MODE=vidore; SUBSETS=${2:?subsets missing}; shift ;;
    --run) MODE=run; CMD=${2:?command missing}; shift ;;
    --dataset) DATASETS+=", \"${2:?dataset missing}\""; shift ;;
    --kernel-output) KERNELS+="${KERNELS:+, }\"${2:?kernel missing}\""; shift ;;
    --no-tests) TESTS=0 ;;
    *) N=$1 ;;
  esac
  shift
done
USER_NAME=$(kaggle config view | awk '/username/ {print $3}')
: "${USER_NAME:?no Kaggle username in kaggle config}"
KERNEL="$USER_NAME/$N" SAMPLES="$USER_NAME/mira-samples"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# Sample PDFs: a private dataset, created on first use
if ! kaggle datasets status "$SAMPLES" >/dev/null 2>&1; then
  mkdir "$tmp/samples"
  cp -L data/samples/*.pdf "$tmp/samples/"
  printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "mira-samples" "$SAMPLES" \
    > "$tmp/samples/dataset-metadata.json"
  kaggle datasets create -p "$tmp/samples" -q
  echo "Waiting for dataset $SAMPLES..."
  until kaggle datasets status "$SAMPLES" 2>/dev/null | grep -q ready; do sleep 10; done
fi

# Kernel: code tarball (base64) + runner. Results go to /kaggle/working, the kernel's output.
git ls-files -co --exclude-standard | grep -vxE '\.env|docs/.*|data/eval/generation_.*' | tar czf "$tmp/code.tgz" -T -
mkdir "$tmp/kernel"
{
  python3 -c 'import sys; print("MODE, SUBSETS, CMD, TESTS =", repr(sys.argv[1:4])[1:-1] + ",", sys.argv[4] == "1")' \
    "$MODE" "$SUBSETS" "$CMD" "$TESTS"
  echo "CODE = '$(base64 -w0 "$tmp/code.tgz")'"
  cat <<'PY'
import base64, glob, io, os, subprocess, sys, tarfile

SRC, CACHE, OUT = "/tmp/mira", "/tmp/cache", "/kaggle/working"
os.environ["PYTHONUNBUFFERED"] = "1"  # child scripts print progress live, not in buffered chunks
os.makedirs(SRC)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall(SRC)
samples = os.path.dirname(glob.glob("/kaggle/input/**/test.pdf", recursive=True)[0])
os.makedirs(f"{SRC}/data", exist_ok=True)
os.symlink(samples, f"{SRC}/data/samples")

def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    if subprocess.run(cmd, shell=True, cwd=SRC).returncode:
        raise SystemExit(f"FAILED: {cmd}")

run("nvidia-smi --query-gpu=name,memory.total --format=csv")
extras = {"vidore": "dev,generation,vidore", "run": "dev,generation,vidore,demo"}.get(MODE, "dev,generation")
run(f"{sys.executable} -m pip install -q -e '.[{extras}]'")
# Kaggle may ship an old torchao; transformers 5 refuses to load models with torchao < 0.16
run(f"{sys.executable} -m pip uninstall -q -y torchao")
if TESTS:
    run(f"{sys.executable} -m pytest -v -rs tests")

# Qdrant server on the VM (single binary). With no .env, every store defaults to localhost:6333.
run(f"mkdir -p {CACHE}/qdrant-bin && curl -sL https://github.com/qdrant/qdrant/releases/latest/download/"
    f"qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C {CACHE}/qdrant-bin")
subprocess.Popen(f"QDRANT__STORAGE__STORAGE_PATH={CACHE}/qdrant {CACHE}/qdrant-bin/qdrant > {CACHE}/qdrant.log 2>&1",
                 shell=True, start_new_session=True)
run("for i in $(seq 60); do curl -sf localhost:6333/readyz && exit 0; sleep 1; done; exit 1")

py, url = sys.executable, "--qdrant-url http://localhost:6333"
if MODE == "eval":
    paths = f"{url} --bm25-path {CACHE}/bm25"
    run(f"{py} scripts/index_corpus.py {paths} --cache-dir {CACHE}/embeddings")
    run(f"{py} scripts/eval_retrieval.py {paths}")
elif MODE == "vidore":
    for s in SUBSETS.split(","):
        bm25 = f"{CACHE}/bm25_vidore/{s}"
        run(f"{py} scripts/index_vidore.py --subset {s} {url} --bm25-path {bm25} --cache-dir {CACHE}/embeddings_vidore/{s}")
        run(f"{py} scripts/eval_retrieval.py --vidore {s} {url} --bm25-path-vidore {bm25}")
        run(f"{py} scripts/eval_cropping.py --vidore {s} {url}")
        run(f"{py} scripts/eval_generation.py --vidore {s} {url} --bm25-path {bm25} --out {OUT}/generation_{s}.jsonl")
elif MODE == "run":
    os.environ.update(OUT=OUT, CACHE=CACHE)
    run(CMD)
else:
    run(f"{py} examples/index_and_search.py data/samples/test.pdf")
print("\nALL PASSED", flush=True)
PY
} > "$tmp/kernel/run.py"
cat > "$tmp/kernel/kernel-metadata.json" <<JSON
{"id": "$KERNEL", "title": "$N", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$SAMPLES"$DATASETS], "competition_sources": [], "kernel_sources": [$KERNELS]}
JSON
kaggle kernels push -p "$tmp/kernel" --accelerator NvidiaTeslaT4
echo "Running: https://www.kaggle.com/code/$KERNEL (live log on that page)"

# Poll until the new version finishes (a fresh push reports queued/running first)
status=""
for _ in $(seq 800); do  # 800 x 60 s > Kaggle's 12 h limit
  sleep 60
  status=$(kaggle kernels status "$KERNEL" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' || true)
  echo "$(date +%H:%M) ${status:-status unavailable}"
  [[ $status =~ COMPLETE|ERROR|CANCEL ]] && break
done

# Log arrives as a JSON list of stream chunks
mkdir -p data/eval
log=data/eval/kaggle_$MODE.log
kaggle kernels logs "$KERNEL" | python3 -c 'import json,sys; print("".join(c["data"] for c in json.load(sys.stdin)), end="")' > "$log"
echo "Saved $log"
if [[ $MODE == vidore ]]; then
  kaggle kernels output "$KERNEL" -p data/eval --file-pattern '^generation_.*\.jsonl$' -o -q && echo "Saved data/eval/generation_*.jsonl"
elif [[ $MODE == run ]]; then
  kaggle kernels output "$KERNEL" -p ".cache/kaggle/$N" -o -q && echo "Saved outputs to .cache/kaggle/$N/"
fi
tail -n 40 "$log"
[[ $status == *COMPLETE ]] && grep -q "^ALL PASSED" "$log" || { echo "KAGGLE RUN FAILED ($status)"; exit 1; }
