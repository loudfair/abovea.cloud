# Epstein Files — Unified Search Engine

Local search engine across **63,494 documents** from the Jeffrey Epstein case files. Aggregates and deduplicates data from 7 open-source archives into a single searchable corpus with vector embeddings and optional AI-powered answers.

## Quick Setup

```bash
git clone https://github.com/loudfair/abovea.cloud.git
cd abovea.cloud
chmod +x setup.sh
./setup.sh
```

The setup script automatically:
1. Creates a Python virtual environment and installs dependencies
2. Downloads all data sources in parallel (~1GB total)
3. Normalizes and deduplicates into a unified corpus
4. Builds the FAISS vector index + text search indexes

Takes about 5-10 minutes depending on network speed.

## Usage

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
python search.py --ask "What properties are mentioned in the documents?"
```

## What's in the Database

| Metric | Value |
|--------|-------|
| Total documents | 63,494 |
| With vector embeddings | 51,751 |
| Unique people indexed | 12,761 |
| Unique words indexed | 226,650 |
| Flight logs | 176 |
| Emails | 4,833 |
| Court filings | 1,467 |

## Data Sources

| Source | Documents | Description |
|--------|-----------|-------------|
| [epfiles](https://github.com/benbaessler/epfiles) | 23,103 | Pre-chunked JSONL from House Oversight docs |
| [HuggingFace embeddings](https://huggingface.co/datasets/svetfm/epstein-files-nov11-25-house-post-ocr-embeddings) | 22,617 | OCR'd text with 768-dim vector embeddings |
| [epstein-docs](https://github.com/epstein-docs/epstein-docs.github.io) | 7,508 | OCR'd JSON with extracted entities + AI summaries |
| [HuggingFace full index](https://huggingface.co/datasets/theelderemo/FULL_EPSTEIN_INDEX) | 3,981 | Grand jury and DOJ documents |
| [HuggingFace emails](https://huggingface.co/datasets/to-be/epstein-emails) | 3,401 | Structured email metadata (from/to/subject/body) |
| [trump-files](https://github.com/HarleyCoops/TrumpEpsteinFiles) | 2,884 | Gemini-processed document extractions |
| [Archive.org](https://archive.org/details/epstein-flight-logs-unredacted-17) | — | Unredacted flight logs PDF |

All documents sourced from public DOJ releases, House Oversight Committee, and FBI disclosures.

## Architecture

```
search.py         — CLI search interface (text, name, email, AI)
normalize.py      — Parses all 7 source formats → unified corpus.jsonl
build_index.py    — Builds FAISS vector index + inverted text/name indexes
setup.sh          — One-command setup: download → normalize → index
```

- **FAISS** for vector similarity search (768-dim embeddings)
- **Inverted index** for keyword/name search
- **OpenAI GPT-4o-mini** for optional AI answer synthesis

## Rebuilding

If you want to rebuild from scratch:

```bash
source venv/bin/activate
python normalize.py      # Re-normalize from downloaded data
python build_index.py    # Rebuild indexes
```

## Security

All downloaded data has been audited:
- No executables, no malicious scripts, no credential leaks
- 21 benign OCR artifacts (Twitter embeds from scanned web page printouts) in corpus text — not executable
- All Python dependencies are well-known legitimate packages
- Pickle files are generated locally — never load pickle from untrusted sources

## Cost

- Setup: **$0** (all data is public, embeddings are pre-computed)
- Text search queries: **$0** (runs entirely locally)
- AI-powered answers: **~$0.001/query** (requires OpenAI API key)

## License

Code is provided as-is for research purposes. All documents are from public government releases.
