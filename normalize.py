"""
Phase 2: Normalize all downloaded data into a unified JSONL corpus.
Reads from all sources, outputs unified records to data/normalized/corpus.jsonl
"""

import json
import os
import glob
import hashlib
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()
BASE = Path(__file__).resolve().parent
DOWNLOADS = BASE / "downloads"
OUTPUT = BASE / "data" / "normalized"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load dedupe maps from epstein-docs
DEDUPE_PEOPLE = {}
DEDUPE_ORGS = {}
DEDUPE_LOCS = {}
DEDUPE_TYPES = {}

dedupe_path = DOWNLOADS / "github" / "epstein-docs" / "dedupe.json"
if dedupe_path.exists():
    with open(dedupe_path) as f:
        dedupe = json.load(f)
        DEDUPE_PEOPLE = dedupe.get("people", {})
        DEDUPE_ORGS = dedupe.get("organizations", {})
        DEDUPE_LOCS = dedupe.get("locations", {})

dedupe_types_path = DOWNLOADS / "github" / "epstein-docs" / "dedupe_types.json"
if dedupe_types_path.exists():
    with open(dedupe_types_path) as f:
        dt = json.load(f)
        DEDUPE_TYPES = dt.get("mappings", {})


def text_hash(text: str) -> str:
    """Create a stable hash of text content for dedup."""
    cleaned = " ".join(text.lower().split())
    return hashlib.md5(cleaned.encode()).hexdigest()


def normalize_people(names: list) -> list:
    """Normalize people names using dedupe map."""
    seen = set()
    result = []
    for name in names:
        canonical = DEDUPE_PEOPLE.get(name, name)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def normalize_orgs(orgs: list) -> list:
    seen = set()
    result = []
    for org in orgs:
        canonical = DEDUPE_ORGS.get(org, org)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def normalize_doc_type(dtype: str) -> str:
    return DEDUPE_TYPES.get(dtype, DEDUPE_TYPES.get(dtype.lower(), dtype))


def make_record(text, source, doc_id=None, filename=None, people=None,
                orgs=None, doc_type=None, date=None, summary=None,
                extra_meta=None):
    """Create a normalized record."""
    if not text or len(text.strip()) < 10:
        return None
    
    record = {
        "id": text_hash(text),
        "source": source,
        "text": text.strip(),
        "metadata": {
            "doc_id": doc_id or "",
            "filename": filename or "",
            "people": normalize_people(people or []),
            "organizations": normalize_orgs(orgs or []),
            "doc_type": normalize_doc_type(doc_type) if doc_type else "",
            "date": date or "",
            "summary": summary or "",
        }
    }
    if extra_meta:
        record["metadata"].update(extra_meta)
    return record


# ─── Source Parsers ───────────────────────────────────────────────────────────


def parse_epstein_docs():
    """Parse epstein-docs GitHub repo: ~29K JSON files + analyses."""
    console.print("[bold blue]Parsing epstein-docs (29K JSON files)...[/]")
    results_dir = DOWNLOADS / "github" / "epstein-docs" / "results"
    
    # Load analyses for summaries
    analyses_map = {}
    analyses_path = DOWNLOADS / "github" / "epstein-docs" / "analyses.json"
    if analyses_path.exists():
        with open(analyses_path) as f:
            data = json.load(f)
            for a in data.get("analyses", []):
                doc_num = a.get("document_number", "")
                if doc_num:
                    analyses_map[doc_num] = a.get("analysis", {})
    
    records = []
    json_files = list(results_dir.glob("**/*.json"))
    
    # Group by document_number to reconstruct multi-page docs
    doc_pages = {}
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            meta = data.get("document_metadata", {})
            doc_num = meta.get("document_number", jf.stem)
            if doc_num not in doc_pages:
                doc_pages[doc_num] = {
                    "pages": [],
                    "people": set(),
                    "orgs": set(),
                    "locs": set(),
                    "doc_type": meta.get("document_type", ""),
                    "date": meta.get("date", ""),
                    "folder": jf.parent.name,
                }
            doc_pages[doc_num]["pages"].append({
                "page": meta.get("page_number", "0"),
                "text": data.get("full_text", ""),
            })
            entities = data.get("entities", {})
            doc_pages[doc_num]["people"].update(entities.get("people", []))
            doc_pages[doc_num]["orgs"].update(entities.get("organizations", []))
        except (json.JSONDecodeError, KeyError):
            continue
    
    for doc_num, doc in doc_pages.items():
        # Sort pages and combine text
        pages_sorted = sorted(doc["pages"], key=lambda p: str(p["page"]))
        full_text = "\n\n".join(p["text"] for p in pages_sorted if p["text"])
        
        analysis = analyses_map.get(doc_num, {})
        summary = analysis.get("summary", "")
        
        record = make_record(
            text=full_text,
            source="epstein-docs",
            doc_id=doc_num,
            filename=f"{doc['folder']}/{doc_num}",
            people=list(doc["people"]),
            orgs=list(doc["orgs"]),
            doc_type=doc["doc_type"],
            date=doc["date"],
            summary=summary,
        )
        if record:
            records.append(record)
    
    console.print(f"  [green]epstein-docs: {len(records)} documents[/]")
    return records


def parse_epfiles_chunks():
    """Parse epfiles JSONL chunks."""
    console.print("[bold blue]Parsing epfiles chunks (23K JSONL files)...[/]")
    chunks_dir = DOWNLOADS / "github" / "epfiles" / "packages" / "backend" / "data" / "chunks"
    
    if not chunks_dir.exists():
        console.print("  [yellow]epfiles chunks dir not found, skipping[/]")
        return []
    
    records = []
    # Group chunks by doc_id to reconstruct full documents
    doc_chunks = {}
    
    for jf in chunks_dir.glob("*.jsonl"):
        try:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    doc_id = chunk.get("doc_id", jf.stem)
                    if doc_id not in doc_chunks:
                        doc_chunks[doc_id] = []
                    doc_chunks[doc_id].append(chunk)
        except (json.JSONDecodeError, KeyError):
            continue
    
    for doc_id, chunks in doc_chunks.items():
        chunks_sorted = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        full_text = "\n\n".join(c.get("text", "") for c in chunks_sorted)
        
        record = make_record(
            text=full_text,
            source="epfiles",
            doc_id=doc_id,
            filename=chunks_sorted[0].get("source_filename", ""),
        )
        if record:
            records.append(record)
    
    console.print(f"  [green]epfiles: {len(records)} documents[/]")
    return records


def parse_hf_embeddings():
    """Parse HuggingFace embeddings dataset (text only, embeddings handled separately)."""
    console.print("[bold blue]Parsing HF embeddings dataset (69K chunks)...[/]")
    pq_path = DOWNLOADS / "huggingface" / "embeddings" / "train.parquet"
    
    if not pq_path.exists():
        console.print("  [yellow]Embeddings parquet not found, skipping[/]")
        return []
    
    df = pd.read_parquet(pq_path)
    
    # Group chunks by source_file to reconstruct documents
    doc_chunks = {}
    for _, row in df.iterrows():
        src = row["source_file"]
        if src not in doc_chunks:
            doc_chunks[src] = []
        doc_chunks[src].append({
            "index": row["chunk_index"],
            "text": row["text"],
        })
    
    records = []
    for src, chunks in doc_chunks.items():
        chunks_sorted = sorted(chunks, key=lambda c: c["index"])
        full_text = " ".join(c["text"] for c in chunks_sorted)
        
        record = make_record(
            text=full_text,
            source="hf-embeddings",
            doc_id=src.replace(".txt", ""),
            filename=src,
        )
        if record:
            records.append(record)
    
    console.print(f"  [green]HF embeddings: {len(records)} documents[/]")
    return records


def parse_hf_emails():
    """Parse HuggingFace emails dataset."""
    console.print("[bold blue]Parsing HF emails dataset...[/]")
    pq_path = DOWNLOADS / "huggingface" / "emails" / "train.parquet"
    
    if not pq_path.exists():
        console.print("  [yellow]Emails parquet not found, skipping[/]")
        return []
    
    df = pd.read_parquet(pq_path)
    records = []
    
    for _, row in df.iterrows():
        text = row.get("message_html", "") or row.get("subject", "")
        if not text:
            continue
        
        people = []
        if row.get("from_address"):
            people.append(str(row["from_address"]))
        if row.get("to_address"):
            people.append(str(row["to_address"]))
        
        record = make_record(
            text=text,
            source="hf-emails",
            doc_id=str(row.get("document_id", "")),
            filename=str(row.get("source_filename", "")),
            people=people,
            doc_type="Email",
            date=str(row.get("timestamp_raw", "")),
            extra_meta={
                "subject": str(row.get("subject", "")),
                "from": str(row.get("from_address", "")),
                "to": str(row.get("to_address", "")),
            }
        )
        if record:
            records.append(record)
    
    console.print(f"  [green]HF emails: {len(records)} emails[/]")
    return records


def parse_hf_full_index():
    """Parse HuggingFace full index dataset."""
    console.print("[bold blue]Parsing HF full index dataset...[/]")
    pq_path = DOWNLOADS / "huggingface" / "full_index" / "train.parquet"
    
    if not pq_path.exists():
        console.print("  [yellow]Full index parquet not found, skipping[/]")
        return []
    
    df = pd.read_parquet(pq_path)
    records = []
    
    for _, row in df.iterrows():
        record = make_record(
            text=str(row.get("text", "")),
            source="hf-full-index",
            doc_id=str(row.get("id", "")),
            filename=str(row.get("id", "")),
        )
        if record:
            records.append(record)
    
    console.print(f"  [green]HF full index: {len(records)} documents[/]")
    return records


def parse_markramm():
    """Parse markramm TXT files."""
    console.print("[bold blue]Parsing markramm TXT files...[/]")
    txt_dir = DOWNLOADS / "github" / "markramm" / "documents" / "house_oversight_sep_2025"
    
    if not txt_dir.exists():
        console.print("  [yellow]markramm txt dir not found, skipping[/]")
        return []
    
    records = []
    for txt_file in txt_dir.glob("**/*.txt"):
        if "requirements" in txt_file.name.lower():
            continue
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            doc_id = txt_file.stem
            record = make_record(
                text=text,
                source="markramm",
                doc_id=doc_id,
                filename=txt_file.name,
            )
            if record:
                records.append(record)
        except Exception:
            continue
    
    console.print(f"  [green]markramm: {len(records)} documents[/]")
    return records


def parse_trump_files():
    """Parse trump-files repo."""
    console.print("[bold blue]Parsing trump-files...[/]")
    text_dir = DOWNLOADS / "github" / "trump-files" / "PIPELINE" / "TEXT"
    
    if not text_dir.exists():
        console.print("  [yellow]trump-files TEXT dir not found, skipping[/]")
        return []
    
    records = []
    
    # Parse extraction JSONs first (richer data)
    json_files = list(text_dir.glob("**/*_extraction.json"))
    parsed_ids = set()
    
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            content = data.get("content", {})
            text = content.get("full_text", "")
            doc_id = str(data.get("house_oversight_id", jf.stem.replace("_extraction", "")))
            parsed_ids.add(doc_id)
            
            record = make_record(
                text=text,
                source="trump-files",
                doc_id=doc_id,
                filename=data.get("file_name", ""),
            )
            if record:
                records.append(record)
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Fall back to TXT files for anything not in JSON
    for txt_file in text_dir.glob("**/*.txt"):
        doc_id = txt_file.stem
        if doc_id in parsed_ids:
            continue
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            record = make_record(
                text=text,
                source="trump-files",
                doc_id=doc_id,
                filename=txt_file.name,
            )
            if record:
                records.append(record)
        except Exception:
            continue
    
    console.print(f"  [green]trump-files: {len(records)} documents[/]")
    return records


def parse_hf_fbi_files():
    """Parse HuggingFace FBI files dataset (svetfm/epstein-fbi-files)."""
    console.print("[bold blue]Parsing HF FBI files dataset (236K chunks)...[/]")
    pq_path = DOWNLOADS / "huggingface" / "fbi_files" / "train.parquet"

    if not pq_path.exists():
        console.print("  [yellow]FBI files parquet not found, skipping[/]")
        return []

    df = pd.read_parquet(pq_path)

    # Drop embedding column if present to save memory
    embed_cols = [c for c in df.columns if "embed" in c.lower()]
    if embed_cols:
        df = df.drop(columns=embed_cols)

    # Group by source_file to reconstruct full documents
    doc_chunks = {}
    for _, row in df.iterrows():
        src = str(row.get("source_file", ""))
        if not src:
            continue
        if src not in doc_chunks:
            doc_chunks[src] = []
        doc_chunks[src].append({
            "index": int(row.get("chunk_index", 0)),
            "text": str(row.get("text", "")),
        })

    records = []
    for src, chunks in doc_chunks.items():
        chunks_sorted = sorted(chunks, key=lambda c: c["index"])
        full_text = " ".join(c["text"] for c in chunks_sorted if c["text"])

        record = make_record(
            text=full_text,
            source="hf-fbi-files",
            doc_id=src.replace(".txt", "").replace(".pdf", ""),
            filename=src,
            doc_type="FBI File",
        )
        if record:
            records.append(record)

    console.print(f"  [green]HF FBI files: {len(records)} documents[/]")
    return records


def parse_hf_fbi_ocr():
    """Parse HuggingFace FBI OCR dataset (vikash06/EpsteinFiles)."""
    console.print("[bold blue]Parsing HF FBI OCR dataset...[/]")
    pq_path = DOWNLOADS / "huggingface" / "fbi_ocr" / "train.parquet"

    if not pq_path.exists():
        console.print("  [yellow]FBI OCR parquet not found, skipping[/]")
        return []

    df = pd.read_parquet(pq_path)

    # Be flexible about column names for text content
    text_col = None
    for candidate in ["text", "content", "ocr_text", "body", "page_text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Fall back to first string-like column that isn't an ID
        for col in df.columns:
            if col.lower() not in ("id", "filename", "file", "source", "index"):
                sample = df[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], str) and len(sample.iloc[0]) > 50:
                    text_col = col
                    break
    if text_col is None:
        console.print(f"  [yellow]Could not identify text column in FBI OCR (cols: {list(df.columns)}), skipping[/]")
        return []

    # Identify an ID column
    id_col = None
    for candidate in ["filename", "id", "file", "doc_id", "name", "source_file"]:
        if candidate in df.columns:
            id_col = candidate
            break

    records = []
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ""))
        doc_id = str(row.get(id_col, idx)) if id_col else str(idx)

        record = make_record(
            text=text,
            source="hf-fbi-ocr",
            doc_id=doc_id,
            filename=doc_id,
            doc_type="FBI File",
        )
        if record:
            records.append(record)

    console.print(f"  [green]HF FBI OCR: {len(records)} documents[/]")
    return records


def parse_hf_house_emails():
    """Parse HuggingFace House Oversight emails (567-labs/jmail-house-oversight)."""
    console.print("[bold blue]Parsing HF House Oversight emails (3,680 records)...[/]")
    pq_path = DOWNLOADS / "huggingface" / "house_emails" / "train.parquet"

    if not pq_path.exists():
        console.print("  [yellow]House emails parquet not found, skipping[/]")
        return []

    df = pd.read_parquet(pq_path)

    # Identify text column
    text_col = None
    for candidate in ["text", "body", "content", "message", "email_body"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Fall back to longest-string column
        for col in df.columns:
            sample = df[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], str) and len(sample.iloc[0]) > 50:
                text_col = col
                break
    if text_col is None:
        console.print(f"  [yellow]Could not identify text column in house emails (cols: {list(df.columns)}), skipping[/]")
        return []

    records = []
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ""))
        if not text.strip():
            continue

        people = []
        from_addr = str(row.get("from", row.get("from_address", row.get("sender", "")))) if any(
            c in df.columns for c in ("from", "from_address", "sender")
        ) else ""
        to_addr = str(row.get("to", row.get("to_address", row.get("recipient", "")))) if any(
            c in df.columns for c in ("to", "to_address", "recipient")
        ) else ""
        subject = str(row.get("subject", row.get("subject_line", ""))) if any(
            c in df.columns for c in ("subject", "subject_line")
        ) else ""

        if from_addr and from_addr != "nan":
            people.append(from_addr)
        if to_addr and to_addr != "nan":
            people.append(to_addr)

        record = make_record(
            text=text,
            source="hf-house-emails",
            doc_id=str(row.get("id", row.get("message_id", idx))),
            people=people,
            doc_type="Email",
            extra_meta={
                "from": from_addr if from_addr != "nan" else "",
                "to": to_addr if to_addr != "nan" else "",
                "subject": subject if subject != "nan" else "",
            },
        )
        if record:
            records.append(record)

    console.print(f"  [green]HF house emails: {len(records)} emails[/]")
    return records


def parse_epstein_files_db():
    """Parse LMSBAND/epstein-files-db SQLite database."""
    console.print("[bold blue]Parsing epstein-files-db (SQLite)...[/]")
    db_dir = DOWNLOADS / "github" / "epstein-files-db"

    if not db_dir.exists():
        console.print("  [yellow]epstein-files-db dir not found, skipping[/]")
        return []

    # Find SQLite database files
    db_files = (
        list(db_dir.rglob("*.db"))
        + list(db_dir.rglob("*.sqlite"))
        + list(db_dir.rglob("*.sqlite3"))
    )
    if not db_files:
        console.print("  [yellow]No SQLite database files found in epstein-files-db, skipping[/]")
        return []

    records = []
    for db_path in db_files:
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # List all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            console.print(f"  Tables in {db_path.name}: {tables}")

            # Look for tables with text content
            text_tables = [
                t for t in tables
                if any(kw in t.lower() for kw in ("document", "text", "content", "file", "page", "email"))
            ]
            if not text_tables:
                text_tables = tables  # Try all tables

            for table in text_tables:
                try:
                    cursor.execute(f"PRAGMA table_info('{table}')")
                    columns = [row[1] for row in cursor.fetchall()]

                    # Find text column
                    text_col = None
                    for candidate in ["text", "content", "body", "ocr_text", "full_text", "page_text"]:
                        if candidate in columns:
                            text_col = candidate
                            break
                    if text_col is None:
                        continue

                    # Find ID column
                    id_col = None
                    for candidate in ["id", "doc_id", "document_id", "filename", "name"]:
                        if candidate in columns:
                            id_col = candidate
                            break

                    cursor.execute(f"SELECT * FROM '{table}'")
                    col_names = [desc[0] for desc in cursor.description]
                    text_idx = col_names.index(text_col)
                    id_idx = col_names.index(id_col) if id_col else None

                    for row in cursor.fetchall():
                        text = str(row[text_idx]) if row[text_idx] else ""
                        doc_id = str(row[id_idx]) if id_idx is not None and row[id_idx] else ""

                        record = make_record(
                            text=text,
                            source="epstein-files-db",
                            doc_id=doc_id or f"{db_path.stem}-{table}",
                            filename=db_path.name,
                        )
                        if record:
                            records.append(record)
                except Exception as e:
                    console.print(f"  [yellow]Error reading table {table}: {e}[/]")
                    continue

            conn.close()
        except Exception as e:
            console.print(f"  [yellow]Error opening {db_path.name}: {e}[/]")
            continue

    console.print(f"  [green]epstein-files-db: {len(records)} records[/]")
    return records


def parse_justice_files_text():
    """Parse promexdotme/epstein-justice-files-text plain text files."""
    console.print("[bold blue]Parsing justice-files-text (TXT files)...[/]")
    txt_dir = DOWNLOADS / "github" / "justice-files-text"

    if not txt_dir.exists():
        console.print("  [yellow]justice-files-text dir not found, skipping[/]")
        return []

    records = []
    for txt_file in txt_dir.rglob("*.txt"):
        # Skip README, LICENSE, and other non-content files
        if txt_file.name.upper().startswith(("README", "LICENSE", "CHANGELOG", "CONTRIBUTING")):
            continue
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 50:
                continue

            doc_id = txt_file.stem
            record = make_record(
                text=text,
                source="justice-files-text",
                doc_id=doc_id,
                filename=txt_file.name,
            )
            if record:
                records.append(record)
        except Exception:
            continue

    console.print(f"  [green]justice-files-text: {len(records)} documents[/]")
    return records


def parse_epstein_network():
    """Parse phelix001/epstein-network CSV data."""
    console.print("[bold blue]Parsing epstein-network (CSV data)...[/]")
    csv_dir = DOWNLOADS / "github" / "epstein-network"

    if not csv_dir.exists():
        console.print("  [yellow]epstein-network dir not found, skipping[/]")
        return []

    csv_files = list(csv_dir.rglob("*.csv"))
    if not csv_files:
        console.print("  [yellow]No CSV files found in epstein-network, skipping[/]")
        return []

    records = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
            cols_lower = {c.lower(): c for c in df.columns}

            # Check if this is a people/contacts CSV
            if "name" in cols_lower:
                name_col = cols_lower["name"]
                for idx, row in df.iterrows():
                    name = str(row.get(name_col, ""))
                    if not name or name == "nan":
                        continue

                    # Build a text representation of the row
                    parts = [f"Name: {name}"]
                    for col in df.columns:
                        if col == name_col:
                            continue
                        val = str(row.get(col, ""))
                        if val and val != "nan":
                            parts.append(f"{col}: {val}")
                    text = "\n".join(parts)

                    record = make_record(
                        text=text,
                        source="epstein-network",
                        doc_id=f"{csv_path.stem}-{idx}",
                        filename=csv_path.name,
                        people=[name],
                        extra_meta={"data_type": "network_record"},
                    )
                    if record:
                        records.append(record)
            else:
                # Generic CSV — look for text-heavy columns
                text_col = None
                for candidate in ["text", "content", "body", "description", "notes"]:
                    if candidate in cols_lower:
                        text_col = cols_lower[candidate]
                        break
                if text_col is None:
                    # Try to find the column with the longest average string length
                    best_col, best_len = None, 0
                    for col in df.columns:
                        sample = df[col].dropna().astype(str).head(20)
                        avg_len = sample.str.len().mean() if len(sample) > 0 else 0
                        if avg_len > best_len and avg_len > 30:
                            best_col, best_len = col, avg_len
                    text_col = best_col

                if text_col:
                    for idx, row in df.iterrows():
                        text = str(row.get(text_col, ""))
                        record = make_record(
                            text=text,
                            source="epstein-network",
                            doc_id=f"{csv_path.stem}-{idx}",
                            filename=csv_path.name,
                        )
                        if record:
                            records.append(record)
        except Exception as e:
            console.print(f"  [yellow]Error reading {csv_path.name}: {e}[/]")
            continue

    console.print(f"  [green]epstein-network: {len(records)} records[/]")
    return records


def parse_doc_explorer():
    """Parse maxandrews/Epstein-doc-explorer SQLite database."""
    console.print("[bold blue]Parsing doc-explorer (SQLite)...[/]")
    db_dir = DOWNLOADS / "github" / "doc-explorer"

    if not db_dir.exists():
        console.print("  [yellow]doc-explorer dir not found, skipping[/]")
        return []

    db_files = list(db_dir.rglob("*.db")) + list(db_dir.rglob("*.sqlite"))
    if not db_files:
        console.print("  [yellow]No SQLite database files found in doc-explorer, skipping[/]")
        return []

    records = []
    for db_path in db_files:
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            console.print(f"  Tables in {db_path.name}: {tables}")

            for table in tables:
                try:
                    cursor.execute(f"PRAGMA table_info('{table}')")
                    columns = [row[1] for row in cursor.fetchall()]
                    cols_lower = {c.lower(): c for c in columns}

                    # Find text column
                    text_col = None
                    for candidate in ["text", "content", "body", "full_text", "message", "ocr_text"]:
                        if candidate in cols_lower:
                            text_col = cols_lower[candidate]
                            break
                    if text_col is None:
                        continue

                    # Check for email-like fields
                    from_col = cols_lower.get("from", cols_lower.get("from_address", cols_lower.get("sender")))
                    to_col = cols_lower.get("to", cols_lower.get("to_address", cols_lower.get("recipient")))
                    id_col = cols_lower.get("id", cols_lower.get("doc_id", cols_lower.get("document_id")))

                    cursor.execute(f"SELECT * FROM '{table}'")
                    col_names = [desc[0] for desc in cursor.description]

                    for row_data in cursor.fetchall():
                        row_dict = dict(zip(col_names, row_data))
                        text = str(row_dict.get(text_col, ""))
                        if not text.strip():
                            continue

                        from_val = str(row_dict.get(from_col, "")) if from_col else ""
                        to_val = str(row_dict.get(to_col, "")) if to_col else ""
                        doc_id_val = str(row_dict.get(id_col, "")) if id_col else ""

                        has_email_fields = from_col is not None or to_col is not None
                        doc_type = "Email" if has_email_fields else ""

                        people = []
                        if from_val and from_val != "None" and from_val != "nan":
                            people.append(from_val)
                        if to_val and to_val != "None" and to_val != "nan":
                            people.append(to_val)

                        extra = {}
                        if from_val and from_val != "None":
                            extra["from"] = from_val
                        if to_val and to_val != "None":
                            extra["to"] = to_val

                        record = make_record(
                            text=text,
                            source="doc-explorer",
                            doc_id=doc_id_val or f"{db_path.stem}-{table}",
                            filename=db_path.name,
                            people=people,
                            doc_type=doc_type,
                            extra_meta=extra if extra else None,
                        )
                        if record:
                            records.append(record)
                except Exception as e:
                    console.print(f"  [yellow]Error reading table {table}: {e}[/]")
                    continue

            conn.close()
        except Exception as e:
            console.print(f"  [yellow]Error opening {db_path.name}: {e}[/]")
            continue

    console.print(f"  [green]doc-explorer: {len(records)} records[/]")
    return records


def parse_kaggle_franciskarajki():
    """Parse Kaggle franciskarajki/epstein-documents (Giuffre v. Maxwell PDFs)."""
    console.print("[bold blue]Parsing kaggle-franciskarajki (Giuffre v. Maxwell docs)...[/]")
    src_dir = DOWNLOADS / "kaggle" / "franciskarajki"

    if not src_dir.exists():
        console.print("  [yellow]kaggle/franciskarajki dir not found, skipping[/]")
        return []

    records = []

    # 1. Look for .txt files (extracted text from PDFs via pdftotext or similar)
    txt_files = list(src_dir.rglob("*.txt"))

    # 2. Look for metadata files (.csv or .jsonl)
    csv_files = list(src_dir.rglob("*.csv"))
    jsonl_files = list(src_dir.rglob("*.jsonl"))

    # Load metadata from CSV if available
    meta_map = {}
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
            cols_lower = {c.lower(): c for c in df.columns}
            # Try to map filename -> row metadata
            fname_col = None
            for candidate in ("filename", "file", "name", "document", "file_name", "pdf"):
                if candidate in cols_lower:
                    fname_col = cols_lower[candidate]
                    break
            if fname_col:
                for _, row in df.iterrows():
                    key = str(row.get(fname_col, ""))
                    if key and key != "nan":
                        meta_map[key] = {c: str(row.get(c, "")) for c in df.columns}
        except Exception:
            continue

    # Load metadata from JSONL if available
    for jl_path in jsonl_files:
        try:
            with open(jl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = obj.get("filename", obj.get("file", obj.get("name", "")))
                    if key:
                        meta_map[key] = {k: str(v) for k, v in obj.items()}
        except Exception:
            continue

    # 3. If no text files found, check for PDFs and skip with a note
    if not txt_files:
        pdf_files = list(src_dir.rglob("*.pdf"))
        if pdf_files:
            console.print(
                f"  [yellow]Found {len(pdf_files)} PDFs but no extracted .txt files — "
                f"run pdftotext first, skipping[/]"
            )
        else:
            console.print("  [yellow]No .txt or .pdf files found, skipping[/]")
        return []

    for txt_file in txt_files:
        if txt_file.name.upper().startswith(("README", "LICENSE", "CHANGELOG")):
            continue
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 50:
                continue

            doc_id = txt_file.stem
            # Try to match metadata by filename (with or without extension variants)
            file_meta = (
                meta_map.get(txt_file.name)
                or meta_map.get(doc_id)
                or meta_map.get(doc_id + ".pdf")
                or {}
            )

            # 4. Split large files into ~2000 char chunks on paragraph boundaries
            if len(text) <= 2500:
                record = make_record(
                    text=text,
                    source="kaggle-franciskarajki",
                    doc_id=doc_id,
                    filename=txt_file.name,
                    doc_type="court_document",
                    extra_meta={"origin": "giuffre_v_maxwell"} if not file_meta else {
                        "origin": "giuffre_v_maxwell",
                        **{k: v for k, v in file_meta.items() if v and v != "nan"},
                    },
                )
                if record:
                    records.append(record)
            else:
                # Split on paragraph boundaries (double newlines)
                paragraphs = text.split("\n\n")
                chunks = []
                current_chunk = []
                current_len = 0

                for para in paragraphs:
                    para_len = len(para)
                    if current_len + para_len > 2000 and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = [para]
                        current_len = para_len
                    else:
                        current_chunk.append(para)
                        current_len += para_len
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                for i, chunk in enumerate(chunks):
                    record = make_record(
                        text=chunk,
                        source="kaggle-franciskarajki",
                        doc_id=f"{doc_id}_chunk{i}",
                        filename=txt_file.name,
                        doc_type="court_document",
                        extra_meta={
                            "origin": "giuffre_v_maxwell",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            **({k: v for k, v in file_meta.items() if v and v != "nan"} if file_meta else {}),
                        },
                    )
                    if record:
                        records.append(record)

        except Exception:
            continue

    console.print(f"  [green]kaggle-franciskarajki: {len(records)} records[/]")
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    console.print("[bold white]═══ NORMALIZING ALL SOURCES ═══[/]\n")
    
    all_records = []
    
    # Parse all sources
    parsers = [
        parse_epstein_docs,
        parse_epfiles_chunks,
        parse_hf_embeddings,
        parse_hf_emails,
        parse_hf_full_index,
        parse_markramm,
        parse_trump_files,
        parse_hf_fbi_files,
        parse_hf_fbi_ocr,
        parse_hf_house_emails,
        parse_epstein_files_db,
        parse_justice_files_text,
        parse_epstein_network,
        parse_doc_explorer,
        parse_kaggle_franciskarajki,
    ]
    
    for parser in parsers:
        try:
            records = parser()
            all_records.extend(records)
        except Exception as e:
            console.print(f"  [red]Error in {parser.__name__}: {e}[/]")
    
    console.print(f"\n[bold]Total records before dedup: {len(all_records)}[/]")
    
    # Deduplicate by text hash
    console.print("[bold blue]Deduplicating by text hash...[/]")
    seen_hashes = {}
    unique_records = []
    
    # Prefer records with richer metadata (epstein-docs has the most)
    source_priority = {
        "epstein-docs": 0,
        "trump-files": 1,
        "epfiles": 2,
        "hf-full-index": 3,
        "hf-embeddings": 4,
        "markramm": 5,
        "hf-emails": 6,
        "hf-fbi-files": 7,
        "hf-fbi-ocr": 8,
        "hf-house-emails": 9,
        "justice-files-text": 10,
        "epstein-files-db": 11,
        "epstein-network": 12,
        "doc-explorer": 13,
        "kaggle-franciskarajki": 19,
    }
    
    # Sort by source priority so richer records are kept
    all_records.sort(key=lambda r: source_priority.get(r["source"], 99))
    
    for record in all_records:
        h = record["id"]
        if h not in seen_hashes:
            seen_hashes[h] = record
            unique_records.append(record)
        else:
            # Merge metadata from duplicate into existing
            existing = seen_hashes[h]
            # Add any people/orgs not already present
            existing_people = set(existing["metadata"]["people"])
            for p in record["metadata"].get("people", []):
                if p not in existing_people:
                    existing["metadata"]["people"].append(p)
                    existing_people.add(p)
            existing_orgs = set(existing["metadata"]["organizations"])
            for o in record["metadata"].get("organizations", []):
                if o not in existing_orgs:
                    existing["metadata"]["organizations"].append(o)
                    existing_orgs.add(o)
            # Fill in missing fields
            if not existing["metadata"]["summary"] and record["metadata"].get("summary"):
                existing["metadata"]["summary"] = record["metadata"]["summary"]
            if not existing["metadata"]["doc_type"] and record["metadata"].get("doc_type"):
                existing["metadata"]["doc_type"] = record["metadata"]["doc_type"]
            if not existing["metadata"]["date"] and record["metadata"].get("date"):
                existing["metadata"]["date"] = record["metadata"]["date"]
    
    console.print(f"[bold green]Unique records after dedup: {len(unique_records)}[/]")
    dupes = len(all_records) - len(unique_records)
    console.print(f"  Removed {dupes} duplicates")
    
    # Write corpus
    corpus_path = OUTPUT / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for record in unique_records:
            f.write(json.dumps(record) + "\n")
    
    size_mb = corpus_path.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold green]Wrote {len(unique_records)} records to {corpus_path} ({size_mb:.1f} MB)[/]")
    
    # Write a summary
    sources = {}
    for r in unique_records:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    
    console.print("\n[bold]Records by source:[/]")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        console.print(f"  {source}: {count}")
    
    # Count records with rich metadata
    with_people = sum(1 for r in unique_records if r["metadata"]["people"])
    with_summary = sum(1 for r in unique_records if r["metadata"]["summary"])
    console.print(f"\n  With people extracted: {with_people}")
    console.print(f"  With summaries: {with_summary}")


if __name__ == "__main__":
    main()
