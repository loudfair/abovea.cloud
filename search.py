#!/usr/bin/env python3
"""
Epstein Files Search — Unified search across 63K+ documents.

Usage:
    python search.py "flight logs passenger list"
    python search.py --name "Bill Clinton"
    python search.py --name "Ghislaine Maxwell" --results 20
    python search.py --ask "Who appears most in flight logs?"
    python search.py --stats
    python search.py --people          # list all indexed people
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional
from collections import Counter

import click
import numpy as np
import faiss
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()
BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "data" / "index"
CORPUS_PATH = BASE / "data" / "normalized" / "corpus.jsonl"

# ─── Lazy-loaded globals ──────────────────────────────────────────────────────

_faiss_index = None
_metadata = None
_search_index = None
_fulltext_offsets = None


def get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(INDEX_DIR / "vectors.faiss"))
    return _faiss_index


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(INDEX_DIR / "metadata.pkl", "rb") as f:
            _metadata = pickle.load(f)
    return _metadata


def get_search_index():
    global _search_index
    if _search_index is None:
        with open(INDEX_DIR / "search_index.pkl", "rb") as f:
            _search_index = pickle.load(f)
    return _search_index


def _build_fulltext_offsets():
    """Build byte-offset index for fulltext.jsonl for O(1) line lookups."""
    global _fulltext_offsets
    if _fulltext_offsets is not None:
        return
    fulltext_path = INDEX_DIR / "fulltext.jsonl"
    if not fulltext_path.exists():
        _fulltext_offsets = []
        return
    offsets = []
    with open(fulltext_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    _fulltext_offsets = offsets


def get_full_text(idx: int) -> str:
    """Get full text for a record by index, reading from fulltext.jsonl."""
    _build_fulltext_offsets()
    if not _fulltext_offsets or idx < 0 or idx >= len(_fulltext_offsets):
        return ""
    fulltext_path = INDEX_DIR / "fulltext.jsonl"
    with open(fulltext_path, "r") as f:
        f.seek(_fulltext_offsets[idx])
        line = f.readline()
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line.strip()


def get_record(idx: int):
    """Get a record by its position in the combined (indexed + text_only) list."""
    meta = get_metadata()
    indexed_count = len(meta["indexed"])
    if idx < indexed_count:
        record = meta["indexed"][idx]
    else:
        text_idx = idx - indexed_count
        if text_idx < len(meta["text_only"]):
            record = meta["text_only"][text_idx]
        else:
            return None
    # Add text_full on demand if not present (new format uses separate file)
    if "text_full" not in record:
        record["text_full"] = record.get("text_preview", "")
    return record


# ─── Search Functions ─────────────────────────────────────────────────────────


def semantic_search(query: str, n_results: int = 10):
    """Search using FAISS vector similarity via OpenAI embeddings."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Set OPENAI_API_KEY for semantic search.[/]")
        console.print("[dim]export OPENAI_API_KEY='sk-...'[/]")
        console.print("[dim]Falling back to text search.[/]\n")
        return text_search(query, n_results=n_results)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        console.print("[dim]Embedding query via OpenAI...[/]")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            dimensions=768,
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        query_vec = query_vec.reshape(1, -1)

        index = get_faiss_index()
        scores, indices = index.search(query_vec, n_results)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = get_record(int(idx))
            if record:
                results.append({
                    "record": record,
                    "score": round(float(score), 4),
                    "index": int(idx),
                })
        return results
    except ImportError:
        console.print("[red]pip install openai[/]")
        return text_search(query, n_results=n_results)
    except Exception as e:
        console.print(f"[red]Semantic search error: {e}[/]")
        return text_search(query, n_results=n_results)


def text_search(query: str, n_results: int = 50, doc_type: str = None):
    """Full-text keyword search using inverted index."""
    search_idx = get_search_index()
    text_index = search_idx["text_index"]
    meta = get_metadata()
    
    query_words = [w.strip(".,;:!?()[]{}\"'").lower() for w in query.split()]
    query_words = [w for w in query_words if len(w) >= 3]
    
    if not query_words:
        return []
    
    # Score documents by how many query words they match
    doc_scores = Counter()
    for word in query_words:
        # Exact match
        if word in text_index:
            for idx in text_index[word]:
                doc_scores[idx] += 2
        # Prefix match
        for indexed_word in text_index:
            if indexed_word.startswith(word) and indexed_word != word:
                for idx in text_index[indexed_word][:100]:
                    doc_scores[idx] += 1
    
    # Also search in people names
    name_index = search_idx["name_index"]
    for word in query_words:
        for name, indices in name_index.items():
            if word in name:
                for idx in indices:
                    doc_scores[idx] += 3  # bonus for name matches
    
    # Get top results
    top_indices = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])
    
    results = []
    for idx in top_indices[:n_results]:
        record = get_record(idx)
        if record is None:
            continue
        
        # Filter by doc_type if specified
        if doc_type:
            rec_type = record.get("metadata", {}).get("doc_type", "").lower()
            if doc_type.lower() not in rec_type:
                continue
        
        results.append({
            "record": record,
            "score": doc_scores[idx],
            "index": idx,
        })
    
    return results


def name_search(name: str, n_results: int = 20):
    """Search for documents mentioning a specific person."""
    search_idx = get_search_index()
    name_index = search_idx["name_index"]
    meta = get_metadata()
    
    name_lower = name.lower()
    
    # Find all matching names (exact and partial)
    matching_indices = set()
    matched_names = []
    
    for indexed_name, indices in name_index.items():
        if name_lower in indexed_name or indexed_name in name_lower:
            matching_indices.update(indices)
            matched_names.append(indexed_name)
    
    if matched_names:
        # Deduplicate matched names for display
        unique_names = list(set(matched_names))[:10]
        console.print(f"[dim]Matched names: {', '.join(n.title() for n in unique_names)}[/dim]\n")
    
    results = []
    for idx in list(matching_indices)[:n_results]:
        record = get_record(idx)
        if record:
            results.append({
                "record": record,
                "score": 10,
                "index": idx,
            })
    
    # Sort by how many matched people appear in the doc
    for r in results:
        people = [p.lower() for p in r["record"].get("metadata", {}).get("people", [])]
        r["score"] = sum(1 for n in matched_names if n in people)
    
    results.sort(key=lambda x: -x["score"])
    return results[:n_results]


def email_search(query: str = None, from_addr: str = None, to_addr: str = None,
                 n_results: int = 20):
    """Search emails by sender, recipient, or content."""
    meta = get_metadata()
    all_records = meta["indexed"] + meta["text_only"]
    
    results = []
    search_terms = []
    if from_addr:
        search_terms.append(("from", from_addr.lower()))
    if to_addr:
        search_terms.append(("to", to_addr.lower()))
    if query:
        search_terms.append(("text", query.lower()))
    
    for i, record in enumerate(all_records):
        rec_meta = record.get("metadata", {})
        
        # Only emails
        if rec_meta.get("doc_type", "").lower() != "email" and "subject" not in rec_meta:
            continue
        
        match = True
        score = 0
        
        for field, term in search_terms:
            if field == "from":
                if term in rec_meta.get("from", "").lower():
                    score += 5
                else:
                    match = False
            elif field == "to":
                if term in rec_meta.get("to", "").lower():
                    score += 5
                else:
                    match = False
            elif field == "text":
                if term in record.get("text_full", "").lower() or term in record.get("text_preview", "").lower():
                    score += 3
                elif term in rec_meta.get("subject", "").lower():
                    score += 4
                else:
                    match = False
        
        if match and score > 0:
            results.append({"record": record, "score": score, "index": i})
    
    results.sort(key=lambda x: -x["score"])
    return results[:n_results]


# ─── Display ──────────────────────────────────────────────────────────────────


def format_result(result, rank: int, full: bool = False):
    """Format and display a single search result."""
    record = result["record"]
    meta = record.get("metadata", {})
    score = result.get("score", 0)
    
    # Title
    doc_id = meta.get("doc_id", record.get("id", "")[:12])
    doc_type = meta.get("doc_type", "")
    date = meta.get("date", "")
    source = record.get("source", "")
    
    title_parts = [f"#{rank}"]
    title_parts.append(f"[bold]{doc_id}[/bold]")
    if doc_type:
        title_parts.append(f"[cyan]{doc_type}[/cyan]")
    if date:
        title_parts.append(f"[dim]{date}[/dim]")
    title_parts.append(f"[yellow]score: {score}[/yellow]")
    
    title = " | ".join(title_parts)
    
    # Body
    body_parts = []
    
    # Summary
    summary = meta.get("summary", "")
    if summary:
        body_parts.append(f"[bold]Summary:[/bold] {summary[:400]}")
    
    # People
    people = meta.get("people", [])
    if people:
        body_parts.append(f"[bold]People:[/bold] {', '.join(people[:15])}")
    
    # Organizations
    orgs = meta.get("organizations", [])
    if orgs:
        body_parts.append(f"[bold]Orgs:[/bold] {', '.join(orgs[:10])}")
    
    # Email fields
    if meta.get("from") or meta.get("to"):
        parts = []
        if meta.get("from"):
            parts.append(f"From: {meta['from']}")
        if meta.get("to"):
            parts.append(f"To: {meta['to']}")
        if meta.get("subject"):
            parts.append(f"Subject: {meta['subject']}")
        body_parts.append("[bold]Email:[/bold] " + " | ".join(parts))
    
    # Text preview (load full text from file on demand)
    if full:
        text = get_full_text(result.get("index", -1)) or record.get("text_full", "") or record.get("text_preview", "")
    else:
        text = record.get("text_preview", "")
    if text:
        preview = text if full else text[:400]
        preview = preview.replace("\n", " ")
        if not full and len(text) > 400:
            preview += "..."
        body_parts.append(f"\n[dim]{preview}[/dim]")
    
    body_parts.append(f"\n[dim italic]source: {source}[/dim italic]")
    
    console.print(Panel("\n".join(body_parts), title=title, border_style="blue"))


def show_stats():
    """Show database statistics."""
    meta = get_metadata()
    search_idx = get_search_index()
    
    all_records = meta["indexed"] + meta["text_only"]
    
    # Count by source
    sources = Counter()
    doc_types = Counter()
    people_count = Counter()
    
    for record in all_records:
        sources[record.get("source", "unknown")] += 1
        dt = record.get("metadata", {}).get("doc_type", "")
        if dt:
            doc_types[dt] += 1
        for person in record.get("metadata", {}).get("people", []):
            people_count[person] += 1
    
    console.print(Panel("[bold]Epstein Files Search Engine[/bold]", border_style="green"))
    
    # Overview
    table = Table(title="Overview")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Total Documents", f"{len(all_records):,}")
    table.add_row("With Vector Embeddings", f"{len(meta['indexed']):,}")
    table.add_row("Text-Only (no vectors)", f"{len(meta['text_only']):,}")
    table.add_row("Unique People Indexed", f"{len(search_idx['name_index']):,}")
    table.add_row("Unique Words Indexed", f"{len(search_idx['text_index']):,}")
    
    index_path = INDEX_DIR / "vectors.faiss"
    meta_path = INDEX_DIR / "metadata.pkl"
    total_size = sum(f.stat().st_size for f in INDEX_DIR.rglob("*") if f.is_file())
    table.add_row("Index Size on Disk", f"{total_size / (1024*1024):.0f} MB")
    console.print(table)
    
    # Sources
    table2 = Table(title="Documents by Source")
    table2.add_column("Source")
    table2.add_column("Count", justify="right")
    for src, count in sources.most_common():
        table2.add_row(src, f"{count:,}")
    console.print(table2)
    
    # Top document types
    table3 = Table(title="Top Document Types")
    table3.add_column("Type")
    table3.add_column("Count", justify="right")
    for dt, count in doc_types.most_common(15):
        table3.add_row(dt, f"{count:,}")
    console.print(table3)
    
    # Most mentioned people
    table4 = Table(title="Most Mentioned People (Top 25)")
    table4.add_column("Person")
    table4.add_column("Documents", justify="right")
    for person, count in people_count.most_common(25):
        table4.add_row(person, f"{count:,}")
    console.print(table4)


def list_people(filter_str: str = None):
    """List all indexed people, optionally filtered."""
    search_idx = get_search_index()
    name_index = search_idx["name_index"]
    
    # Count docs per person
    people = [(name.title(), len(indices)) for name, indices in name_index.items()]
    people.sort(key=lambda x: -x[1])
    
    if filter_str:
        filter_lower = filter_str.lower()
        people = [(n, c) for n, c in people if filter_lower in n.lower()]
    
    table = Table(title=f"Indexed People ({len(people):,} total)")
    table.add_column("Person")
    table.add_column("Documents", justify="right")
    
    for name, count in people[:50]:
        table.add_row(name, f"{count:,}")
    
    if len(people) > 50:
        table.add_row(f"... and {len(people) - 50} more", "")
    
    console.print(table)


def ask_ai(question: str, context_docs: list):
    """Use OpenAI to synthesize an answer from search results."""
    try:
        from openai import OpenAI
    except ImportError:
        console.print("[red]pip install openai[/]")
        return None
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Set OPENAI_API_KEY environment variable.[/]")
        console.print("[dim]export OPENAI_API_KEY='sk-...'[/]")
        return None
    
    client = OpenAI(api_key=api_key)
    
    # Build context
    context_parts = []
    for i, doc in enumerate(context_docs[:10]):
        record = doc["record"]
        meta = record.get("metadata", {})
        text = record.get("text_full", record.get("text_preview", ""))[:2000]
        
        header = f"Doc {i+1}: {meta.get('doc_id', 'unknown')}"
        if meta.get("doc_type"):
            header += f" ({meta['doc_type']})"
        if meta.get("date"):
            header += f" - {meta['date']}"
        
        summary = meta.get("summary", "")
        if summary:
            context_parts.append(f"--- {header} ---\nSummary: {summary}\n\nExcerpt: {text}")
        else:
            context_parts.append(f"--- {header} ---\n{text}")
    
    context = "\n\n".join(context_parts)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant analyzing Jeffrey Epstein case documents. "
                    "Answer based ONLY on the provided document excerpts. "
                    "Cite specific document IDs. Be factual and precise. "
                    "If documents don't contain enough info, say so."
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocuments:\n\n{context}"
            }
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    
    return response.choices[0].message.content


# ─── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.argument("query", required=False)
@click.option("--name", "-n", help="Search by person name")
@click.option("--type", "-t", "doc_type", help="Filter by document type")
@click.option("--email", "-e", is_flag=True, help="Search emails")
@click.option("--from", "-f", "from_addr", help="Email sender filter")
@click.option("--to", "to_addr", help="Email recipient filter")
@click.option("--semantic", is_flag=True, help="Use FAISS semantic search (needs OPENAI_API_KEY)")
@click.option("--ask", "-a", is_flag=True, help="Get AI-synthesized answer")
@click.option("--results", "-r", default=10, help="Number of results")
@click.option("--stats", "-s", is_flag=True, help="Show statistics")
@click.option("--people", "-p", is_flag=True, help="List all indexed people")
@click.option("--full", is_flag=True, help="Show full document text")
def main(query, name, doc_type, email, from_addr, to_addr, semantic, ask, results, stats, people, full):
    """Search the Epstein Files archive (63K+ documents).
    
    \b
    Examples:
        python search.py "flight logs"
        python search.py --name "Ghislaine Maxwell"
        python search.py --email --from "epstein"
        python search.py --ask "What properties are mentioned?"
        python search.py --stats
        python search.py --people
    """
    if stats:
        show_stats()
        return
    
    if people:
        list_people(query)
        return
    
    if not query and not name and not from_addr and not to_addr:
        console.print("[red]Provide a search query, --name, --stats, or --people[/]")
        console.print("[dim]Run: python search.py --help[/]")
        return
    
    search_query = query or name or from_addr or to_addr
    console.print(f"\n[bold]Searching:[/bold] [cyan]{search_query}[/cyan]\n")
    
    # Run search
    if name:
        search_results = name_search(name, n_results=results)
    elif email or from_addr or to_addr:
        search_results = email_search(query=query, from_addr=from_addr, to_addr=to_addr, n_results=results)
    elif semantic:
        search_results = semantic_search(search_query, n_results=results)
    else:
        search_results = text_search(search_query, n_results=results, doc_type=doc_type)
    
    if not search_results:
        console.print("[yellow]No results found.[/]")
        return
    
    console.print(f"[bold green]{len(search_results)} results:[/]\n")
    
    for i, result in enumerate(search_results):
        format_result(result, rank=i + 1, full=full)
    
    # AI answer
    if ask:
        console.print("\n[bold]Generating AI analysis...[/]\n")
        answer = ask_ai(search_query, search_results)
        if answer:
            console.print(Panel(
                Markdown(answer),
                title="[bold green]AI Answer[/bold green]",
                border_style="green",
            ))


if __name__ == "__main__":
    main()
