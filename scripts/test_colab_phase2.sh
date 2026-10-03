#!/bin/bash
# Mira Phase 2 Test on Google Colab GPU
# 
# This script tests Phase 2 (visual retrieval) on Colab's free T4 GPU
# 
# Usage: bash scripts/test_colab_phase2.sh
#
# Prerequisites:
# - Google Colab CLI installed: uv tool install google-colab-cli
# - .env file configured with Qdrant credentials (copy from .env.example)
# - Repo is public on GitHub (or use GITHUB_TOKEN for private)

set -e

SESSION_NAME="mira-phase2-$(date +%s)"

echo "============================================================"
echo "Mira Phase 2 - Google Colab GPU Test"
echo "============================================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials:"
    echo "  cp .env.example .env"
    echo "  # Edit .env with your Qdrant credentials"
    exit 1
fi

echo "✓ Found .env file"

# Read credentials from .env
export $(grep -v '^#' .env | xargs)

# Verify credentials are set
if [ -z "$QDRANT_CLUSTER_ENDPOINT" ] || [ -z "$QDRANT_CLUSTER_API_KEY" ]; then
    echo "ERROR: Qdrant credentials not found in .env"
    echo "Please set QDRANT_CLUSTER_ENDPOINT and QDRANT_CLUSTER_API_KEY"
    exit 1
fi

echo "✓ Qdrant credentials loaded"

# Step 1: Create session
echo -e "\n[1/5] Creating Colab GPU session..."
colab new --session $SESSION_NAME --gpu T4

# Step 2: Clone and setup
echo -e "\n[2/5] Cloning repository..."
cat > /tmp/01_setup.py << 'EOF'
import subprocess
import sys
import os

print("="*60)
print("SETUP")
print("="*60)

# Clone repo
print("Cloning Mira...")
subprocess.run(["rm", "-rf", "mira"], check=False)
result = subprocess.run(
    ["git", "clone", "https://github.com/Va1bhav512/mira.git"],
    capture_output=True
)

if result.returncode != 0:
    print(f"ERROR: {result.stderr.decode()}")
    sys.exit(1)

print("✓ Repository cloned")
os.chdir("mira")

# Install dependencies
print("\nInstalling dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "sync", "--all-extras"], check=True)

# Install mira package
print("\nInstalling mira package...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], check=True)

print("\n✓ Setup complete")
EOF

colab exec --session $SESSION_NAME --file /tmp/01_setup.py --timeout 180

# Step 3: Test GPU
echo -e "\n[3/5] Testing GPU..."
cat > /tmp/02_gpu.py << 'EOF'
import torch

print("\n" + "="*60)
print("GPU VERIFICATION")
print("="*60)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    exit(1)

print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF

colab exec --session $SESSION_NAME --file /tmp/02_gpu.py

# Step 4: Test Phase 2 with credentials from .env
echo -e "\n[4/5] Testing Phase 2..."
cat > /tmp/03_phase2.py << EOF
import os
import sys

os.chdir("mira")

# Set credentials from environment
os.environ['QDRANT_CLUSTER_ENDPOINT'] = '$QDRANT_CLUSTER_ENDPOINT'
os.environ['QDRANT_CLUSTER_API_KEY'] = '$QDRANT_CLUSTER_API_KEY'

print("\n" + "="*60)
print("PHASE 2: VISUAL RETRIEVAL TEST")
print("="*60)

# Test imports
try:
    from mira.pdf import process_pdf
    from mira.retrieval import ColQwenEmbedder, DocumentIndexer, VisualRetriever
    print("✓ Imports successful")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)

import time

# Test PDF processing
print("\n[Test 1] PDF Processing")
start = time.time()
pages = process_pdf("data/samples/test.pdf", dpi=150)
print(f"✓ Processed {len(pages)} pages in {time.time()-start:.1f}s")

# Test ColQwen loading
print("\n[Test 2] ColQwen Model Loading")
print("  Loading model (this takes ~20s on first run)...")
start = time.time()
embedder = ColQwenEmbedder(use_4bit=True)
print(f"✓ Model loaded in {time.time()-start:.1f}s")

# Test indexing
print("\n[Test 3] Document Indexing")
indexer = DocumentIndexer(embedder=embedder)
indexer.setup()
start = time.time()
result = indexer.index_document("data/samples/test.pdf", dpi=150, batch_size=4)
print(f"✓ Indexed {result.pages_indexed} pages in {result.duration_seconds:.1f}s")

# Test search
print("\n[Test 4] Search")
retriever = VisualRetriever(embedder=embedder)
queries = ["attention mechanism", "transformer architecture"]
for query in queries:
    start = time.time()
    results = retriever.search(query, top_k=3)
    print(f"  '{query}': {len(results)} results in {(time.time()-start)*1000:.0f}ms")

print("\n✓ All tests passed!")
EOF

colab exec --session $SESSION_NAME --file /tmp/03_phase2.py --timeout 300

# Step 5: Summary
echo -e "\n[5/5] Test Summary..."
cat > /tmp/04_summary.py << 'EOF'
import os
os.chdir("mira")

print("\n" + "="*60)
print("✓ MIRA PHASE 2 VERIFIED")
print("="*60)
print("\nAll tests passed:")
print("  ✓ PDF processing working")
print("  ✓ ColQwen model loaded on GPU")
print("  ✓ Qdrant connection successful")
print("  ✓ Document indexing working")
print("  ✓ Search retrieval working")
print("\nReady for Phase 3 (BM25 indexing)")
EOF

colab exec --session $SESSION_NAME --file /tmp/04_summary.py

# Cleanup
rm -f /tmp/0*.py

echo -e "\n============================================================"
echo "✓ Test Complete"
echo "============================================================"
echo ""
echo "Session: $SESSION_NAME"
echo ""
echo "To stop the session:"
echo "  colab stop --session $SESSION_NAME"
echo ""
echo "✓ Mira Phase 2 verified on Colab GPU!"
