#!/bin/bash
set -e

echo "═══════════════════════════════════════════════"
echo "  Epstein Files Search Engine — Setup"
echo "═══════════════════════════════════════════════"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── 1. Create virtual environment ───────────────────────────────────────────

echo "[1/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
echo "  ✓ Dependencies installed"

# ─── 2. Create directories ──────────────────────────────────────────────────

mkdir -p downloads/{huggingface,github,archive}
mkdir -p data/{raw,normalized,index}

# ─── 3. Download all data sources in parallel ────────────────────────────────

echo ""
echo "[2/5] Downloading data from all sources (this takes a few minutes)..."
echo ""

# HuggingFace datasets
echo "  Downloading HuggingFace datasets..."
python3 -c "
from datasets import load_dataset
import os

datasets_to_download = [
    ('theelderemo/FULL_EPSTEIN_INDEX', 'full_index'),
    ('to-be/epstein-emails', 'emails'),
    ('svetfm/epstein-files-nov11-25-house-post-ocr-embeddings', 'embeddings'),
    ('svetfm/epstein-fbi-files', 'fbi_files'),
    ('vikash06/EpsteinFiles', 'fbi_ocr'),
    ('567-labs/jmail-house-oversight', 'house_emails'),
]

for name, folder in datasets_to_download:
    outdir = f'downloads/huggingface/{folder}'
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(f'{outdir}/train.parquet'):
        print(f'  ✓ {name} (already downloaded)')
        continue
    print(f'  Downloading {name}...')
    try:
        ds = load_dataset(name)
        for split in ds:
            ds[split].to_parquet(f'{outdir}/{split}.parquet')
        print(f'  ✓ {name} ({len(ds[list(ds.keys())[0]])} rows)')
    except Exception as e:
        print(f'  ✗ {name}: {e}')
" 2>/dev/null

# GitHub repos (parallel)
echo "  Cloning GitHub repositories..."

clone_repo() {
    local url=$1
    local dir=$2
    if [ -d "downloads/github/$dir" ]; then
        echo "  ✓ $dir (already cloned)"
    else
        git clone --depth 1 -q "$url" "downloads/github/$dir" 2>/dev/null && \
            echo "  ✓ $dir" || echo "  ✗ $dir (failed)"
    fi
}

clone_repo "https://github.com/epstein-docs/epstein-docs.github.io.git" "epstein-docs" &
clone_repo "https://github.com/markramm/EpsteinFiles.git" "markramm" &
clone_repo "https://github.com/benbaessler/epfiles.git" "epfiles" &
clone_repo "https://github.com/theelderemo/FULL_EPSTEIN_INDEX.git" "full-index" &
wait

clone_repo "https://github.com/HarleyCoops/TrumpEpsteinFiles.git" "trump-files" &
wait

clone_repo "https://github.com/LMSBAND/epstein-files-db.git" "epstein-files-db" &
clone_repo "https://github.com/promexdotme/epstein-justice-files-text.git" "justice-files-text" &
clone_repo "https://github.com/phelix001/epstein-network.git" "epstein-network" &
clone_repo "https://github.com/maxandrews/Epstein-doc-explorer.git" "doc-explorer" &
wait

clone_repo "https://github.com/yung-megafone/Epstein-Files.git" "magnet-links" &
clone_repo "https://github.com/SvetimFM/epstein-files-visualizations.git" "visualizations" &
clone_repo "https://github.com/paulgp/epstein-document-search.git" "document-search" &
wait

# Archive.org
echo "  Downloading flight logs from Archive.org..."
mkdir -p downloads/archive/flight-logs
if [ ! -f "downloads/archive/flight-logs/epstein-flight-logs.pdf" ]; then
    curl -sL -o "downloads/archive/flight-logs/epstein-flight-logs.pdf" \
        "https://archive.org/download/epstein-flight-logs-unredacted-17/EPSTEIN%20FLIGHT%20LOGS%20UNREDACTED%20%2817%29.pdf" 2>/dev/null && \
        echo "  ✓ Flight logs" || echo "  ✗ Flight logs (failed)"
else
    echo "  ✓ Flight logs (already downloaded)"
fi

echo "  Downloading additional documents from Archive.org..."

mkdir -p downloads/archive/black-book
if [ ! -f "downloads/archive/black-book/black-book.pdf" ]; then
    curl -sL -o "downloads/archive/black-book/black-book.pdf" \
        "https://archive.org/download/jeffrey-epstein-39s-little-black-book-unredacted/Jeffrey%20Epstein%27s%20Little%20Black%20Book%20unredacted.pdf" 2>/dev/null && \
        echo "  ✓ Black book" || echo "  ✗ Black book (failed)"
else
    echo "  ✓ Black book (already downloaded)"
fi &

mkdir -p downloads/archive/epstein-docs-collection
if [ ! -f "downloads/archive/epstein-docs-collection/Epstein-Docs.pdf" ]; then
    curl -sL -o "downloads/archive/epstein-docs-collection/Epstein-Docs.pdf" \
        "https://ia600705.us.archive.org/21/items/epsteindocs/Epstein-Docs.pdf" 2>/dev/null && \
        echo "  ✓ Epstein docs collection" || echo "  ✗ Epstein docs collection (failed)"
else
    echo "  ✓ Epstein docs collection (already downloaded)"
fi &

mkdir -p downloads/archive/depositions
if [ ! -f "downloads/archive/depositions/Edwards-vs-Epstein-depositions.pdf" ]; then
    curl -sL -o "downloads/archive/depositions/Edwards-vs-Epstein-depositions.pdf" \
        "https://ia600705.us.archive.org/21/items/epsteindocs/12%23%20Epstein%20deposition%27s%20-%20Edwards%20vs%20Epstein%20%2B%20attachments.pdf" 2>/dev/null && \
        echo "  ✓ Depositions" || echo "  ✗ Depositions (failed)"
else
    echo "  ✓ Depositions (already downloaded)"
fi &

wait

echo ""
echo "  Downloads complete."

# ─── 4. Normalize all data ───────────────────────────────────────────────────

echo ""
echo "[3/5] Normalizing data from all sources..."
python3 normalize.py
echo "  ✓ Corpus built"

# ─── 5. Build search index ───────────────────────────────────────────────────

echo ""
echo "[4/5] Building search index..."
python3 build_index.py
echo "  ✓ Index built"

# ─── 6. Done ─────────────────────────────────────────────────────────────────

echo ""
echo "[5/5] Setup complete!"
echo ""
echo "═══════════════════════════════════════════════"
echo "  Ready! Try these commands:"
echo ""
echo "  source venv/bin/activate"
echo "  python search.py --stats"
echo "  python search.py \"flight logs\""
echo "  python search.py --name \"Ghislaine Maxwell\""
echo "  python search.py --email --from \"epstein\""
echo "  python search.py --people \"clinton\""
echo ""
echo "  For AI-powered answers:"
echo "  export OPENAI_API_KEY='sk-...'"
echo "  python search.py --ask \"Who visited the island?\""
echo ""
echo "  Web UI:"
echo "  python app.py"
echo "  Open http://localhost:5000"
echo "═══════════════════════════════════════════════"
