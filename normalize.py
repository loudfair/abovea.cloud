"""
Phase 2: Normalize all downloaded data into a unified JSONL corpus.
Reads from all sources, outputs unified records to data/normalized/corpus.jsonl
"""

import json
import os
import glob
import hashlib
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
