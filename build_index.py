"""
Phase 3: Build FAISS vector index + metadata index from normalized corpus.
Uses pre-computed embeddings from HuggingFace where available.
For docs without pre-computed embeddings, uses a lightweight sentence model
or stores them for text-only search.
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
    return doc_embeddings


def load_corpus():
    """Load normalized corpus."""
    console.print("[bold blue]Loading corpus...[/]")
    records = []
    with open(CORPUS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    console.print(f"  [green]Loaded {len(records)} records[/]")
    return records


def build_index(records, precomputed_embeddings):
    """Build FAISS index and metadata store."""
    console.print("[bold blue]Building FAISS index...[/]")
    
    EMBEDDING_DIM = 768  # matching pre-computed embeddings
    
    # Separate records into those with and without embeddings
    indexed_records = []  # records that will go into FAISS
    text_only_records = []  # records for text-only search
    embeddings_list = []
    
    with_emb = 0
    without_emb = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Matching embeddings", total=len(records))
        
        for record in records:
            doc_id = record["metadata"].get("doc_id", "")
            filename = record["metadata"].get("filename", "")
            
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
                indexed_records.append(record)
                embeddings_list.append(embedding)
                with_emb += 1
            else:
                text_only_records.append(record)
                without_emb += 1
            
            progress.update(task, advance=1)
    
    console.print(f"  Records with embeddings: {with_emb}")
    console.print(f"  Records without (text-only): {without_emb}")
    
    # Build FAISS index
    if embeddings_list:
        console.print("[bold blue]Building FAISS vector index...[/]")
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        
        # Use Inner Product (equivalent to cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings_matrix)
        
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        console.print(f"  [green]FAISS index: {index.ntotal} vectors, dim={EMBEDDING_DIM}[/]")
    
    # Build metadata store (previews only — full text stored separately)
    console.print("[bold blue]Building metadata index...[/]")
    
    # For FAISS-indexed records, store metadata in order matching the index
    metadata_store = {
        "indexed": [],  # matches FAISS index order
        "text_only": [],  # records without embeddings
    }
    
    # Write full text to a separate line-indexed file (one JSON per line)
    # This avoids loading all full text into the pickle (which blows RAM)
    fulltext_path = INDEX_DIR / "fulltext.jsonl"
    all_ordered = indexed_records + text_only_records
    
    with open(fulltext_path, "w") as ft:
        for record in all_ordered:
            ft.write(json.dumps(record["text"]) + "\n")
    
    for record in indexed_records:
        metadata_store["indexed"].append({
            "id": record["id"],
            "source": record["source"],
            "text_preview": record["text"][:500],
            "metadata": record["metadata"],
        })
    
    for record in text_only_records:
        metadata_store["text_only"].append({
            "id": record["id"],
            "source": record["source"],
            "text_preview": record["text"][:500],
            "metadata": record["metadata"],
        })
    
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)
    
    console.print(f"  Full text: {fulltext_path} ({fulltext_path.stat().st_size / (1024*1024):.1f} MB)")
    
    # Build text search index (inverted index of words -> record indices)
    console.print("[bold blue]Building text search index...[/]")
    
    # Simple inverted index for name/keyword search
    all_records_ordered = indexed_records + text_only_records
    name_index = {}  # person name -> list of record indices
    text_index = {}  # word -> list of record indices
    
    for i, record in enumerate(all_records_ordered):
        # Index people names
        for person in record.get("metadata", {}).get("people", []):
            name_lower = person.lower()
            if name_lower not in name_index:
                name_index[name_lower] = []
            name_index[name_lower].append(i)
        
        # Index words from text (just first 1000 chars for speed)
        words = set(record["text"][:1000].lower().split())
        for word in words:
            if len(word) < 3:
                continue
            word = word.strip(".,;:!?()[]{}\"'")
            if word and len(word) >= 3:
                if word not in text_index:
                    text_index[word] = []
                if len(text_index[word]) < 1000:  # cap per word
                    text_index[word].append(i)
    
    search_index = {
        "name_index": name_index,
        "text_index": text_index,
        "total_records": len(all_records_ordered),
        "indexed_count": len(indexed_records),
        "text_only_count": len(text_only_records),
    }
    
    search_index_path = INDEX_DIR / "search_index.pkl"
    with open(search_index_path, "wb") as f:
        pickle.dump(search_index, f)
    
    # Summary
    console.print(f"\n[bold green]Index build complete![/]")
    console.print(f"  FAISS index: {FAISS_INDEX_PATH} ({FAISS_INDEX_PATH.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Metadata: {METADATA_PATH} ({METADATA_PATH.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Search index: {search_index_path} ({search_index_path.stat().st_size / (1024*1024):.1f} MB)")
    console.print(f"  Unique people indexed: {len(name_index)}")
    console.print(f"  Unique words indexed: {len(text_index)}")


def main():
    console.print("[bold white]═══ BUILDING VECTOR INDEX ═══[/]\n")
    
    precomputed = load_precomputed_embeddings()
    records = load_corpus()
    build_index(records, precomputed)


if __name__ == "__main__":
    main()
