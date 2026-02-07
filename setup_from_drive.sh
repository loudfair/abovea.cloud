#!/bin/bash
set -e

echo "═══════════════════════════════════════════════"
echo "  Epstein Files Search Engine — Quick Setup"
echo "  (syncs pre-built index from Google Drive)"
echo "═══════════════════════════════════════════════"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Config ────────────────────────────────────────────────────────────────
# Remote name configured in rclone (default: "gdrive")
# Folder path on your Drive where colab_build.ipynb saved the index
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_PATH="${DRIVE_PATH:-epstein-search-index}"

# ─── 1. Check rclone ──────────────────────────────────────────────────────

if ! command -v rclone &>/dev/null; then
    echo "  rclone not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install rclone
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl https://rclone.org/install.sh | sudo bash
    else
        echo "  Install rclone: https://rclone.org/install/"
        exit 1
    fi
fi

# Check if remote is configured
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:"; then
    echo ""
    echo "  ⚠  rclone remote '${RCLONE_REMOTE}' not configured."
    echo ""
    echo "  Run:  rclone config"
    echo "  → New remote → name it '${RCLONE_REMOTE}'"
    echo "  → Storage: Google Drive"
    echo "  → Follow the auth flow"
    echo ""
    echo "  Then re-run this script."
    exit 1
fi

echo "  ✓ rclone ready (remote: ${RCLONE_REMOTE}:${DRIVE_PATH})"

# ─── 2. Python environment ────────────────────────────────────────────────

echo ""
echo "[1/3] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
echo "  ✓ Dependencies installed"

# ─── 3. Create directories ────────────────────────────────────────────────

mkdir -p data/{normalized,index}

# ─── 4. Sync index from Google Drive ──────────────────────────────────────

echo ""
echo "[2/3] Syncing index from Google Drive..."
echo "  Source: ${RCLONE_REMOTE}:${DRIVE_PATH}/"
echo ""

# Sync index files
rclone sync "${RCLONE_REMOTE}:${DRIVE_PATH}/" /tmp/epstein-drive-sync \
    --progress \
    --transfers 4 \
    --include "vectors.faiss" \
    --include "metadata.pkl" \
    --include "search_index.pkl" \
    --include "fulltext.jsonl" \
    --include "id_map.json" \
    --include "corpus.jsonl"

# Move to correct locations
for f in vectors.faiss metadata.pkl search_index.pkl fulltext.jsonl id_map.json; do
    if [ -f "/tmp/epstein-drive-sync/$f" ]; then
        mv "/tmp/epstein-drive-sync/$f" data/index/
        echo "  ✓ $f"
    fi
done

if [ -f "/tmp/epstein-drive-sync/corpus.jsonl" ]; then
    mv "/tmp/epstein-drive-sync/corpus.jsonl" data/normalized/
    echo "  ✓ corpus.jsonl"
fi

rm -rf /tmp/epstein-drive-sync

# ─── 5. Verify ────────────────────────────────────────────────────────────

echo ""
echo "[3/3] Verifying index..."

MISSING=0
for f in data/index/vectors.faiss data/index/metadata.pkl data/index/search_index.pkl data/index/fulltext.jsonl; do
    if [ ! -f "$f" ]; then
        echo "  ✗ Missing: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "  ⚠  Some index files are missing."
    echo "  Make sure you've run colab_build.ipynb first to build the index."
    echo "  Check that files exist at: ${RCLONE_REMOTE}:${DRIVE_PATH}/"
    echo "  Run: rclone ls ${RCLONE_REMOTE}:${DRIVE_PATH}/"
    exit 1
fi

echo "  ✓ All index files present"
echo ""
echo "  Index files:"
du -sh data/index/vectors.faiss     2>/dev/null | awk '{print "    " $2 ": " $1}'
du -sh data/index/metadata.pkl      2>/dev/null | awk '{print "    " $2 ": " $1}'
du -sh data/index/search_index.pkl  2>/dev/null | awk '{print "    " $2 ": " $1}'
du -sh data/index/fulltext.jsonl    2>/dev/null | awk '{print "    " $2 ": " $1}'
du -sh data/normalized/corpus.jsonl 2>/dev/null | awk '{print "    " $2 ": " $1}'

echo ""
echo "═══════════════════════════════════════════════"
echo "  Ready!"
echo ""
echo "  source venv/bin/activate"
echo "  python search.py --stats"
echo "  python search.py \"flight logs\""
echo ""
echo "  Web UI:"
echo "  python app.py → http://localhost:5000"
echo "═══════════════════════════════════════════════"
