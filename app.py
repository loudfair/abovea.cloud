#!/usr/bin/env python3
"""
Epstein Files Search — Web UI

Flask web interface for searching 63K+ documents from the Epstein case files.

Usage:
    python app.py                  # runs on http://localhost:5000
    python app.py --port 8080      # custom port
"""

import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
import click

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "data" / "index"
CORPUS_PATH = BASE / "data" / "normalized" / "corpus.jsonl"

app = Flask(__name__, template_folder=str(BASE / "templates"), static_folder=str(BASE / "static"))

# ─── Lazy-loaded globals ──────────────────────────────────────────────────────

_faiss_index = None
_metadata = None
_search_index = None
_fulltext_offsets = None


def get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        path = INDEX_DIR / "vectors.faiss"
        if path.exists():
            _faiss_index = faiss.read_index(str(path))
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
    _build_fulltext_offsets()
    if not _fulltext_offsets or idx < 0 or idx >= len(_fulltext_offsets):
        return ""
    with open(INDEX_DIR / "fulltext.jsonl", "r") as f:
        f.seek(_fulltext_offsets[idx])
        line = f.readline()
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line.strip()


def get_record(idx: int):
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
    if "text_full" not in record:
        record["text_full"] = record.get("text_preview", "")
    return record


# ─── Search Functions ─────────────────────────────────────────────────────────


def semantic_search(query: str, n_results: int = 20):
    """Search using FAISS vector similarity. Requires OpenAI API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [], "Set OPENAI_API_KEY environment variable for semantic search."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

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
        if index is None:
            return [], "FAISS index not found. Run setup.sh first."

        scores, indices = index.search(query_vec, n_results)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            record = get_record(int(idx))
            if record:
                results.append({
                    "record": record,
                    "score": round(float(score), 4),
                    "index": int(idx),
                })

        return results, None
    except ImportError:
        return [], "pip install openai"
    except Exception as e:
        return [], str(e)


def text_search(query: str, n_results: int = 50, doc_type: str = None):
    search_idx = get_search_index()
    text_index = search_idx["text_index"]

    query_words = [w.strip(".,;:!?()[]{}\"'").lower() for w in query.split()]
    query_words = [w for w in query_words if len(w) >= 3]

    if not query_words:
        return []

    doc_scores = Counter()
    for word in query_words:
        if word in text_index:
            for idx in text_index[word]:
                doc_scores[idx] += 2
        for indexed_word in text_index:
            if indexed_word.startswith(word) and indexed_word != word:
                for idx in text_index[indexed_word][:100]:
                    doc_scores[idx] += 1

    name_index = search_idx["name_index"]
    for word in query_words:
        for name, indices in name_index.items():
            if word in name:
                for idx in indices:
                    doc_scores[idx] += 3

    top_indices = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])

    results = []
    for idx in top_indices[:n_results]:
        record = get_record(idx)
        if record is None:
            continue
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
    search_idx = get_search_index()
    name_index = search_idx["name_index"]
    name_lower = name.lower()

    matching_indices = set()
    matched_names = []

    for indexed_name, indices in name_index.items():
        if name_lower in indexed_name or indexed_name in name_lower:
            matching_indices.update(indices)
            matched_names.append(indexed_name)

    results = []
    for idx in list(matching_indices)[:n_results * 3]:
        record = get_record(idx)
        if record:
            results.append({
                "record": record,
                "score": 10,
                "index": idx,
            })

    for r in results:
        people = [p.lower() for p in r["record"].get("metadata", {}).get("people", [])]
        r["score"] = sum(1 for n in matched_names if n in people)

    results.sort(key=lambda x: -x["score"])
    return results[:n_results], [n.title() for n in set(matched_names)][:15]


def email_search(query: str = None, from_addr: str = None, to_addr: str = None,
                 n_results: int = 20):
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


def get_stats():
    meta = get_metadata()
    search_idx = get_search_index()
    all_records = meta["indexed"] + meta["text_only"]

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

    total_size = 0
    if INDEX_DIR.exists():
        total_size = sum(f.stat().st_size for f in INDEX_DIR.rglob("*") if f.is_file())

    return {
        "total_documents": len(all_records),
        "with_embeddings": len(meta["indexed"]),
        "text_only": len(meta["text_only"]),
        "unique_people": len(search_idx["name_index"]),
        "unique_words": len(search_idx["text_index"]),
        "index_size_mb": round(total_size / (1024 * 1024)),
        "sources": dict(sources.most_common()),
        "doc_types": dict(doc_types.most_common(20)),
        "top_people": dict(people_count.most_common(50)),
    }


def list_people(filter_str: str = None, limit: int = 100):
    search_idx = get_search_index()
    name_index = search_idx["name_index"]
    people = [(name.title(), len(indices)) for name, indices in name_index.items()]
    people.sort(key=lambda x: -x[1])

    if filter_str:
        filter_lower = filter_str.lower()
        people = [(n, c) for n, c in people if filter_lower in n.lower()]

    return people[:limit], len(people)


# ─── Serialization helper ────────────────────────────────────────────────────


def serialize_result(r):
    record = r["record"]
    meta = record.get("metadata", {})
    idx = r.get("index", -1)
    full_text = get_full_text(idx) if idx >= 0 else record.get("text_full", record.get("text_preview", ""))
    return {
        "doc_id": meta.get("doc_id", record.get("id", "")[:12]),
        "doc_type": meta.get("doc_type", ""),
        "date": meta.get("date", ""),
        "source": record.get("source", ""),
        "score": r.get("score", 0),
        "summary": meta.get("summary", ""),
        "people": meta.get("people", [])[:20],
        "organizations": meta.get("organizations", [])[:10],
        "text_preview": record.get("text_preview", "")[:500],
        "text_full": full_text,
        "email_from": meta.get("from", ""),
        "email_to": meta.get("to", ""),
        "email_subject": meta.get("subject", ""),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "text")
    from_addr = request.args.get("from", "").strip()
    to_addr = request.args.get("to", "").strip()
    doc_type = request.args.get("type", "").strip()
    limit = min(int(request.args.get("limit", 20)), 100)

    if not query and not from_addr and not to_addr:
        return jsonify({"results": [], "count": 0, "error": "No query provided"})

    matched_names = []

    try:
        error = None
        if mode == "name":
            results, matched_names = name_search(query, n_results=limit)
        elif mode == "email":
            results = email_search(query=query, from_addr=from_addr, to_addr=to_addr, n_results=limit)
        elif mode == "semantic":
            results, error = semantic_search(query, n_results=limit)
        else:
            results = text_search(query, n_results=limit, doc_type=doc_type or None)

        serialized = [serialize_result(r) for r in results]

        resp = {
            "results": serialized,
            "count": len(serialized),
            "query": query,
            "mode": mode,
            "matched_names": matched_names,
        }
        if error:
            resp["error"] = error

        return jsonify(resp)
    except Exception as e:
        return jsonify({"results": [], "count": 0, "error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    try:
        stats = get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/people")
def api_people():
    filter_str = request.args.get("q", "").strip() or None
    limit = min(int(request.args.get("limit", 100)), 500)
    try:
        people, total = list_people(filter_str=filter_str, limit=limit)
        return jsonify({
            "people": [{"name": n, "documents": c} for n, c in people],
            "total": total,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/document/<path:doc_id>")
def api_document(doc_id):
    """Get full document by searching for its doc_id."""
    meta = get_metadata()
    all_records = meta["indexed"] + meta["text_only"]

    for i, record in enumerate(all_records):
        rec_id = record.get("metadata", {}).get("doc_id", record.get("id", ""))
        if rec_id == doc_id:
            return jsonify({
                "doc_id": doc_id,
                "source": record.get("source", ""),
                "text": get_full_text(i) or record.get("text_preview", ""),
                "metadata": record.get("metadata", {}),
            })

    return jsonify({"error": "Document not found"}), 404


# ─── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Epstein Files Search — Web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    print(f"\n  Epstein Files Search Engine")
    print(f"  http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
