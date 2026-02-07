# Epstein Files — Unified Search Engine

> **STATUS: FULLY OPERATIONAL** — 63,494 documents indexed, search working locally. Pushed to GitHub.

---

## SESSION HANDOFF

### What This Is

A locally-running search engine that aggregates **ALL publicly available extracted text** from the Jeffrey Epstein case files (DOJ releases, House Oversight Committee, FBI disclosures) into a single deduplicated, searchable database with vector embeddings and AI-powered query capabilities.

### What Was Done This Session

| Step | Status | Details |
|------|--------|---------|
| Research available data sources | DONE | Found 7 open-source archives with pre-extracted text (no OCR needed) |
| Security audit of all downloads | DONE | All clean — no malicious code, no credential leaks, no executables |
| Download all data in parallel | DONE | 4 HuggingFace datasets + 5 GitHub repos + Archive.org flight logs |
| Normalize 7 different formats | DONE | JSON, JSONL, CSV, Parquet, TXT all parsed into unified schema |
| Deduplicate across sources | DONE | 77,529 raw records → 63,494 unique (14,035 duplicates removed) |
| Build FAISS vector index | DONE | 51,751 docs with 768-dim pre-computed embeddings |
| Build text/name search index | DONE | 226,650 words + 12,761 people indexed |
| Build CLI search tool | DONE | Text search, name search, email search, AI synthesis |
| Push to GitHub | DONE | https://github.com/loudfair/abovea.cloud |
| Clean repo structure | DONE | DIAGRAM.mmd, OPEN-ME.html, comprehensive .gitignore |

### What's on Disk (local machine only — not in git)

```
/Users/m3/epstein-search/
├── downloads/          1.0 GB  ← raw data from all sources (gitignored)
├── data/               686 MB  ← normalized corpus + FAISS index (gitignored)
├── venv/               636 MB  ← Python virtual environment (gitignored)
├── app.py                      ← Flask web UI server
├── templates/                  ← HTML templates for web UI
├── static/                     ← CSS/JS assets for web UI
├── search.py                   ← CLI search tool
├── normalize.py                ← data normalization pipeline
├── build_index.py              ← index builder
├── setup.sh                    ← one-command rebuild
├── README.md                   ← this file
├── DIAGRAM.mmd                 ← architecture diagram
├── OPEN-ME.html                ← visual overview
├── requirements.txt            ← dependencies
└── .gitignore                  ← comprehensive exclusions
```

### What's on GitHub

**Repo:** https://github.com/loudfair/abovea.cloud

Only code — no data. Anyone clones it, runs `./setup.sh`, and gets the full 63K-document search engine built locally in ~10 minutes.

### What's NOT Done / Future Work

| Item | Priority | Notes |
|------|----------|-------|
| Web UI frontend | DONE | Flask web interface at http://localhost:5000 |
| DOJ full 3.5M pages | LOW | Only ~63K docs indexed (what's been OCR'd by community). The DOJ released 3.5M pages total — most not yet OCR'd by anyone |
| Semantic search via API | MEDIUM | FAISS index is built but search.py currently uses text search only (no OpenAI embedding calls for queries). Could add `--semantic` flag that embeds the query via OpenAI and does FAISS similarity search |
| Network graph of connections | LOW | The entity data (12K people, 5K orgs) is there to build person-to-person co-occurrence graphs |
| Kaggle datasets | LOW | Kaggle has additional datasets we didn't download (requires auth) |
| Remove chromadb dependency | LOW | Installed but unused (Python 3.14 incompatible). Could remove from requirements.txt to speed up install |

### How to Resume This Work

```bash
# Everything is at:
cd /Users/m3/epstein-search
source venv/bin/activate

# Search is fully working:
python search.py "flight logs"
python search.py --name "Ghislaine Maxwell"
python search.py --email --from "epstein"
python search.py --people "clinton"
python search.py --stats

# For AI answers (needs API key):
export OPENAI_API_KEY='sk-...'
python search.py --ask "Who visited the island?"

# To rebuild from scratch:
./setup.sh

# To push changes:
git add -A && git commit -m "description" && git push
```

### Key Decisions Made

1. **No OCR needed** — all text was already extracted by community projects. We just downloaded their outputs.
2. **Pre-computed embeddings** — HuggingFace dataset had 768-dim vectors already generated. Cost: $0.
3. **FAISS over ChromaDB** — ChromaDB broke on Python 3.14. FAISS works perfectly, faster, lighter.
4. **Pickle for index storage** — fast serialization but inherently unsafe if loaded from untrusted sources. Only load locally-generated pickles.
5. **Git LFS not needed** — all data is excluded from git and rebuilt by `setup.sh`. Keeps repo tiny (~60KB).

---

## Quick Setup

```bash
git clone https://github.com/loudfair/abovea.cloud.git
cd abovea.cloud
chmod +x setup.sh
./setup.sh
```

## Web UI

```bash
source venv/bin/activate
python app.py
# Open http://localhost:5000
```

## CLI Usage

```bash
source venv/bin/activate

# Full-text search
python search.py "flight logs"
python search.py "palm beach police investigation"

# Search by person name
python search.py --name "Ghislaine Maxwell"
python search.py --name "Alan Dershowitz"

# List all indexed people matching a name
python search.py --people "clinton"
python search.py --people "trump"

# Search emails
python search.py --email --from "epstein"
python search.py --email --from "ghislaine" --results 20

# Show full document text
python search.py "flight logs" --full --results 3

# Database statistics
python search.py --stats

# AI-powered answers (requires OpenAI API key)
export OPENAI_API_KEY='sk-...'
python search.py --ask "Who appears most frequently in flight logs?"
```

## Database Stats

| Metric | Value |
|--------|-------|
| Total documents | 63,494 |
| With vector embeddings | 51,751 |
| Unique people indexed | 12,761 |
| Unique words indexed | 226,650 |
| Emails | 4,833 |
| Court filings | 1,467 |
| Flight logs | 176 |
| Reports | 870 |
| Transcripts | 336 |
| Financial records | 593 |
| Medical records | 380 |

## Data Sources

| Source | Documents | Format | Description |
|--------|-----------|--------|-------------|
| [epfiles](https://github.com/benbaessler/epfiles) | 23,103 | JSONL | Pre-chunked House Oversight docs |
| [HuggingFace embeddings](https://huggingface.co/datasets/svetfm/epstein-files-nov11-25-house-post-ocr-embeddings) | 22,617 | Parquet | OCR'd text + 768-dim vectors |
| [epstein-docs](https://github.com/epstein-docs/epstein-docs.github.io) | 7,508 | JSON | 29K pages with entities + AI summaries |
| [HuggingFace full index](https://huggingface.co/datasets/theelderemo/FULL_EPSTEIN_INDEX) | 3,981 | CSV | Grand jury and DOJ documents |
| [HuggingFace emails](https://huggingface.co/datasets/to-be/epstein-emails) | 3,401 | Parquet | Structured emails (from/to/subject/body) |
| [trump-files](https://github.com/HarleyCoops/TrumpEpsteinFiles) | 2,884 | TXT+JSON | Gemini-processed extractions |
| [Archive.org](https://archive.org/details/epstein-flight-logs-unredacted-17) | — | PDF | Unredacted flight logs |

## Architecture

```
setup.sh          → Downloads all 7 sources in parallel (~1GB, ~5 min)
normalize.py      → Parses JSON/JSONL/CSV/Parquet/TXT → corpus.jsonl (63,494 docs)
build_index.py    → corpus.jsonl → FAISS vectors + inverted text/name index
search.py         → Query interface: text, name, email, AI synthesis
```

**Stack:** Python 3.14 / Flask / FAISS / OpenAI (optional) / Rich CLI

## Security Audit

All downloaded data was audited across 52,555 files:
- Zero executables, zero malicious scripts, zero credential leaks
- 21 benign OCR artifacts (Twitter embeds from scanned web page printouts) — text only, not executable
- All 93 Python packages are legitimate well-known libraries
- Pickle files generated locally — never load from untrusted sources
- If building a web UI: sanitize corpus text before HTML rendering

## Cost

| What | Cost |
|------|------|
| Setup (all data + embeddings) | $0 |
| Text/name/email search | $0 (fully local) |
| AI-powered answers | ~$0.001/query (OpenAI API) |

## License

Code provided as-is for research purposes. All documents are from public U.S. government releases.
