"""
Phase 3: Build FAISS vector index + metadata index from normalized corpus.
Uses pre-computed embeddings from HuggingFace where available.
Memory-efficient: streams corpus instead of loading everything into RAM.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()
BASE = Path(__file__).resolve().parent
CORPUS_PATH = BASE / "data" / "normalized" / "corpus.jsonl"
INDEX_DIR = BASE / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

HF_EMBEDDINGS_PATH = BASE / "downloads" / "huggingface" / "embeddings" / "train.parquet"

FAISS_INDEX_PATH = INDEX_DIR / "vectors.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
ID_MAP_PATH = INDEX_DIR / "id_map.json"


def load_precomputed_embeddings():
    """Load pre-computed embeddings from HuggingFace dataset, grouped by source file."""
    console.print("[bold blue]Loading pre-computed embeddings...[/]")
    
    if not HF_EMBEDDINGS_PATH.exists():
        console.print("  [yellow]No pre-computed embeddings found[/]")
        return {}
    
    df = pd.read_parquet(HF_EMBEDDINGS_PATH)
    
    # Build lookup: various key formats -> averaged embedding
    doc_embeddings = {}
    for source_file, group in df.groupby("source_file"):
        embeddings = np.array(group["embedding"].tolist(), dtype=np.float32)
        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize for cosine similarity
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        # Store under multiple key variants for matching
        doc_id_base = source_file.replace(".txt", "")
        doc_embeddings[doc_id_base] = avg_embedding
        
        # Also try without prefix like "IMAGES-005-"
        parts = doc_id_base.split("-", 2)
        if len(parts) >= 3 and parts[0] == "IMAGES":
            short_id = parts[2]
            doc_embeddings[short_id] = avg_embedding
    
    console.print(f"  [green]Loaded embeddings for {len(df.groupby('source_file'))} source files ({len(doc_embeddings)} key variants)[/]")
    
    # Free the dataframe immediately
    del df
    
    return doc_embeddings


def count_lines(path):
    """Fast line count."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def build_index_streaming(precomputed_embeddings):
    """
    Build FAISS index, metadata store, and text search index in a single
    streaming pass over corpus.jsonl — never loads full text into memory.
    """
    EMBEDDING_DIM = 768

    total = count_lines(CORPUS_PATH)
    console.print(f"[bold blue]Streaming {total} records...[/]")

    # ── Pass 1: scan corpus, collect metadata + embeddings ─────────────
    # We keep lightweight dicts (preview + metadata) and embeddings,
    # but NOT the full text.
    indexed_meta = []    # metadata for records with embeddings (FAISS order)
    text_only_meta = []  # metadata for records without embeddings
    embeddings_list = []

    fulltext_path = INDEX_DIR / "fulltext.jsonl"

    with open(CORPUS_PATH, "r") as corpus_in, \
         open(fulltext_path, "w") as ft_out, \
         Progress(
             SpinnerColumn(),
             TextColumn("[progress.description]{task.description}"),
             BarColumn(),
             TextColumn("{task.completed}/{task.total}"),
             TimeElapsedColumn(),
         ) as progress:

        task = progress.add_task("Processing", total=total)

        for line in corpus_in:
            record = json.loads(line)
            text = record["text"]
            meta = record.get("metadata", {})
            doc_id = meta.get("doc_id", "")
            filename = meta.get("filename", "")

            # Build lightweight entry (no full text)
            entry = {
                "id": record["id"],
                "source": record["source"],
                "text_preview": text[:500],
                "metadata": meta,
            }

            # Try to find pre-computed embedding
            embedding = None
            candidates = [
                doc_id,
                filename.replace(".txt", "").replace(".jpg", ""),
                doc_id.replace("HOUSE_OVERSIGHT_", "IMAGES-005-HOUSE_OVERSIGHT_"),
            ]
            for key in candidates:
                if key and key in precomputed_embeddings:
                    embedding = precomputed_embeddings[key]
                    break

            if embedding is not None:
                indexed_meta.append(entry)
                embeddings_list.append(embedding)
            else:
                text_only_meta.append(entry)

            # We DON'T write fulltext yet — we need ordered
            # (indexed first, text_only second). We'll do a 2nd pass.

            progress.update(task, advance=1)

    console.print(f"  Records with embeddings: {len(indexed_meta)}")
    console.print(f"  Records without (text-only): {len(text_only_meta)}")

    # Free embeddings lookup — no longer needed
    precomputed_embeddings.clear()

    # ── Build FAISS index ─────────────────────────────────────────────
    if embeddings_list:
        console.print("[bold blue]Building FAISS vector index...[/]")
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        del embeddings_list

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings_matrix)
        del embeddings_matrix

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        console.print(f"  [green]FAISS index: {index.ntotal} vectors, dim={EMBEDDING_DIM}[/]")
        del index

    # ── Build ID sets for ordering ────────────────────────────────────
    # We need to write fulltext.jsonl in the same order as metadata
    # (indexed first, then text_only). Build lookup sets.
    indexed_ids = {e["id"] for e in indexed_meta}

    # ── Write fulltext.jsonl (2nd pass — streaming, low memory) ──────
    console.print("[bold blue]Writing full-text index...[/]")
    # Two temp files: one for indexed, one for text_only
    ft_indexed_path = INDEX_DIR / "_ft_indexed.jsonl"
    ft_textonly_path = INDEX_DIR / "_ft_textonly.jsonl"

    with open(CORPUS_PATH, "r") as corpus_in, \
         open(ft_indexed_path, "w") as ft_idx, \
         open(ft_textonly_path, "w") as ft_txt:
        for line in corpus_in:
            record = json.loads(line)
            rec_id = record["id"]
            text_json = json.dumps(record["text"])
            if rec_id in indexed_ids:
                ft_idx.write(text_json + "\n")
            else:
                ft_txt.write(text_json + "\n")

    # Concatenate in correct order
    with open(fulltext_path, "w") as out:
        for src in [ft_indexed_path, ft_textonly_path]:
            with open(src, "r") as inp:
                for line in inp:
                    out.write(line)

    # Clean up temp files
    ft_indexed_path.unlink(missing_ok=True)
    ft_textonly_path.unlink(missing_ok=True)
    del indexed_ids

    console.print(f"  Full text: {fulltext_path} ({fulltext_path.stat().st_size / (1024*1024):.1f} MB)")

    # ── Build metadata pickle ─────────────────────────────────────────
    console.print("[bold blue]Building metadata index...[/]")
    metadata_store = {
        "indexed": indexed_meta,
        "text_only": text_only_meta,
    }

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    console.print(f"  Metadata: {METADATA_PATH} ({METADATA_PATH.stat().st_size / (1024*1024):.1f} MB)")

    # ── Build text search index ───────────────────────────────────────
    console.print("[bold blue]Building text search index...[/]")

    all_meta = indexed_meta + text_only_meta
    name_index = {}
    text_index = {}

    for i, entry in enumerate(all_meta):
        # Index people names
        for person in entry.get("metadata", {}).get("people", []):
            name_lower = person.lower()
            if name_lower not in name_index:
                name_index[name_lower] = []
            name_index[name_lower].append(i)

        # Index words from text preview (500 chars, already in memory)
        preview = entry.get("text_preview", "")
        words = set(preview.lower().split())
        for word in words:
            if len(word) < 3:
                continue
            word = word.strip(".,;:!?()[]{}\"'")
            if word and len(word) >= 3:
                if word not in text_index:
                    text_index[word] = []
                if len(text_index[word]) < 1000:
                    text_index[word].append(i)

    search_index = {
        "name_index": name_index,
        "text_index": text_index,
        "total_records": len(all_meta),
        "indexed_count": len(indexed_meta),
        "text_only_count": len(text_only_meta),
    }

    search_index_path = INDEX_DIR / "search_index.pkl"
    with open(search_index_path, "wb") as f:
        pickle.dump(search_index, f)

    # Summary
    console.print(f"\n[bold green]Index build complete![/]")
    console.print(f"  FAISS index: {FAISS_INDEX_PATH} ({FAISS_INDEX_PATH.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Metadata: {METADATA_PATH} ({METADATA_PATH.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Search index: {search_index_path} ({search_index_path.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Full text: {fulltext_path} ({fulltext_path.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Unique people indexed: {len(name_index)}")
    console.print(f"  Unique words indexed: {len(text_index)}")


def main():
    console.print("[bold white]═══ BUILDING VECTOR INDEX ═══[/]\n")
    
    precomputed = load_precomputed_embeddings()
    build_index_streaming(precomputed)


if __name__ == "__main__":
    main()
