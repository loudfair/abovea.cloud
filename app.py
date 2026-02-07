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
import re
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
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


def _parse_gmail_query(query_str: str):
    """Parse a Gmail-style query string into structured operators and free text.

    Supported operators:
        from:value, to:value, subject:value, cc:value, bcc:value,
        after:YYYY-MM-DD (alias: newer:), before:YYYY-MM-DD (alias: older:),
        has:attachment

    Bare words (not part of an operator) become free-text search terms.
    Quoted phrases like "some phrase" are kept together.
    """
    operators = {
        "from": [],
        "to": [],
        "cc": [],
        "bcc": [],
        "subject": [],
        "after": None,
        "before": None,
        "has": [],
    }
    free_text = []

    # Tokenize: split on whitespace but respect quoted values  e.g. from:"John Doe"
    token_re = re.compile(
        r'(\w+):"([^"]*)"'     # operator:"quoted value"
        r'|(\w+):(\S+)'        # operator:value
        r'|"([^"]*)"'          # "quoted free text"
        r"|(\S+)"              # bare word
    )

    for m in token_re.finditer(query_str):
        if m.group(1):  # operator:"quoted"
            op, val = m.group(1).lower(), m.group(2)
        elif m.group(3):  # operator:value
            op, val = m.group(3).lower(), m.group(4)
        elif m.group(5):  # "quoted free text"
            free_text.append(m.group(5))
            continue
        elif m.group(6):  # bare word
            free_text.append(m.group(6))
            continue
        else:
            continue

        if op in ("from", "to", "cc", "bcc", "subject"):
            operators[op].append(val.lower())
        elif op in ("after", "newer"):
            operators["after"] = val
        elif op in ("before", "older"):
            operators["before"] = val
        elif op == "has":
            operators["has"].append(val.lower())

    return operators, free_text


def _parse_date_safe(date_str):
    """Try to parse a date string into a comparable format. Returns None on failure."""
    if not date_str:
        return None
    # Strip time component if present
    date_str = date_str.strip()[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    # Try parsing just the year
    try:
        return datetime.strptime(date_str[:4], "%Y")
    except (ValueError, TypeError):
        return None


def gmail_style_email_search(query_str: str, n_results: int = 50):
    """Gmail-style email search supporting operators and free text.

    Operators: from:, to:, cc:, bcc:, subject:, after:/newer:, before:/older:, has:attachment
    Bare words search across subject and body text.

    Returns (results_list, total_count) where each result is a dict with
    from, to, cc, bcc, subject, date, body_preview, score, doc_id, source.
    """
    operators, free_text = _parse_gmail_query(query_str)

    meta = get_metadata()
    all_records = meta["indexed"] + meta["text_only"]

    # Parse date bounds once
    after_dt = _parse_date_safe(operators["after"]) if operators["after"] else None
    before_dt = _parse_date_safe(operators["before"]) if operators["before"] else None

    results = []

    for i, record in enumerate(all_records):
        rec_meta = record.get("metadata", {})

        # Only consider email-like records
        doc_type = rec_meta.get("doc_type", "").lower()
        has_email_fields = any(rec_meta.get(f) for f in ("from", "to", "subject"))
        if doc_type != "email" and not has_email_fields:
            continue

        score = 0
        matched = True

        # ── Field operator matching ──────────────────────────────────────
        rec_from = rec_meta.get("from", "").lower()
        rec_to = rec_meta.get("to", "").lower()
        rec_cc = rec_meta.get("cc", "").lower()
        rec_bcc = rec_meta.get("bcc", "").lower()
        rec_subject = rec_meta.get("subject", "").lower()
        rec_date_str = rec_meta.get("date", "")

        # from: operator
        for term in operators["from"]:
            if term in rec_from:
                score += 10
            else:
                matched = False
                break

        if not matched:
            continue

        # to: operator
        for term in operators["to"]:
            if term in rec_to:
                score += 10
            else:
                matched = False
                break

        if not matched:
            continue

        # cc: operator
        for term in operators["cc"]:
            if term in rec_cc:
                score += 8
            else:
                matched = False
                break

        if not matched:
            continue

        # bcc: operator
        for term in operators["bcc"]:
            if term in rec_bcc:
                score += 8
            else:
                matched = False
                break

        if not matched:
            continue

        # subject: operator
        for term in operators["subject"]:
            if term in rec_subject:
                score += 8
            else:
                matched = False
                break

        if not matched:
            continue

        # after:/before: date range
        if after_dt or before_dt:
            rec_dt = _parse_date_safe(rec_date_str)
            if rec_dt is None:
                # Can't verify date — skip if date filter is strict
                if after_dt or before_dt:
                    matched = False
            else:
                if after_dt and rec_dt < after_dt:
                    matched = False
                if before_dt and rec_dt > before_dt:
                    matched = False

        if not matched:
            continue

        # has:attachment
        if "attachment" in operators["has"]:
            attachments = rec_meta.get("attachments", rec_meta.get("has_attachment", False))
            if isinstance(attachments, list):
                has_attach = len(attachments) > 0
            elif isinstance(attachments, bool):
                has_attach = attachments
            elif isinstance(attachments, str):
                has_attach = attachments.lower() not in ("", "false", "no", "0", "none")
            else:
                has_attach = bool(attachments)
            if not has_attach:
                matched = False

        if not matched:
            continue

        # ── Free text matching ───────────────────────────────────────────
        body = (record.get("text_full", "") or record.get("text_preview", "")).lower()

        if free_text:
            all_text_matched = True
            for word in free_text:
                word_lower = word.lower()
                if word_lower in rec_subject:
                    score += 6  # subject match is high value
                elif word_lower in rec_from or word_lower in rec_to:
                    score += 5
                elif word_lower in rec_cc or word_lower in rec_bcc:
                    score += 4
                elif word_lower in body:
                    score += 3
                else:
                    all_text_matched = False
                    break

            if not all_text_matched:
                continue

        # If no operators and no free text were given, don't return everything
        if score == 0 and not operators["after"] and not operators["before"]:
            continue

        # Date-range-only queries get a base score
        if score == 0:
            score = 1

        # Build result
        body_raw = record.get("text_full", "") or record.get("text_preview", "")
        results.append({
            "from": rec_meta.get("from", ""),
            "to": rec_meta.get("to", ""),
            "cc": rec_meta.get("cc", ""),
            "bcc": rec_meta.get("bcc", ""),
            "subject": rec_meta.get("subject", ""),
            "date": rec_date_str,
            "body_preview": body_raw[:300] if body_raw else "",
            "score": score,
            "doc_id": rec_meta.get("doc_id", record.get("id", "")[:12]),
            "source": record.get("source", ""),
            "index": i,
        })

    # Sort by score descending, then by date descending (most recent first)
    def sort_key(r):
        dt = _parse_date_safe(r["date"])
        # Use a very old date as fallback so undated items sink
        ts = dt.timestamp() if dt else 0
        return (-r["score"], -ts)

    results.sort(key=sort_key)

    total_count = len(results)
    return results[:n_results], total_count


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
        "email_cc": meta.get("cc", ""),
        "email_bcc": meta.get("bcc", ""),
        "email_subject": meta.get("subject", ""),
    }


# ─── AI Features ──────────────────────────────────────────────────────────────


def ai_ask(question: str, search_results: list, max_context: int = 10):
    """Synthesize an answer from search results using GPT-4o-mini (~$0.001/query)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "Set OPENAI_API_KEY environment variable for AI features."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build context from top results
        context_parts = []
        for i, r in enumerate(search_results[:max_context]):
            record = r.get("record", r)
            meta = record.get("metadata", {})
            idx = r.get("index", -1)
            text = get_full_text(idx) if idx >= 0 else ""
            if not text:
                text = record.get("text_full", record.get("text_preview", ""))
            # Truncate each doc to keep token count manageable
            text = text[:2000] if text else "(no text)"
            doc_id = meta.get("doc_id", record.get("id", f"doc-{i+1}"))
            source = record.get("source", meta.get("source", ""))
            header = f"[Document {i+1}: {doc_id}"
            if source:
                header += f" | {source}"
            if meta.get("from"):
                header += f" | From: {meta['from']}"
            if meta.get("to"):
                header += f" | To: {meta['to']}"
            if meta.get("subject"):
                header += f" | Subject: {meta['subject']}"
            if meta.get("date"):
                header += f" | Date: {meta['date']}"
            header += "]"
            context_parts.append(f"{header}\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant analyzing the Epstein case files. "
                        "Answer the user's question based ONLY on the provided documents. "
                        "Be specific — cite document IDs and quote relevant passages. "
                        "If the documents don't contain enough information to answer, say so. "
                        "Be concise but thorough. Use bullet points for clarity."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\n--- DOCUMENTS ---\n\n{context}",
                },
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        return response.choices[0].message.content, None
    except ImportError:
        return None, "pip install openai"
    except Exception as e:
        return None, str(e)


def ai_expand_query(query: str):
    """Use GPT to expand a vague query into better search terms (~$0.0003/call)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help expand search queries about the Jeffrey Epstein case. "
                        "Given a user query, return a JSON object with:\n"
                        "1. \"expanded\": a better search query with additional relevant terms "
                        "(e.g. full names, aliases, related locations, related people)\n"
                        "2. \"related\": array of 3-5 related follow-up search queries\n"
                        "Be specific to the Epstein case. Only return valid JSON, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        result = json.loads(text)
        return result.get("expanded"), result.get("related", [])
    except Exception:
        return None, None


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
            gmail_results, _total = gmail_style_email_search(query, n_results=limit)
            # Wrap gmail results into the same shape as other modes
            results = [{"record": {"metadata": r, "text_preview": r.get("body_preview", ""),
                                    "source": r.get("source", "")},
                         "score": r["score"], "index": r.get("index", -1)} for r in gmail_results]
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


@app.route("/api/email-search")
def api_email_search():
    """Gmail-style email search endpoint.

    GET /api/email-search?q=from:epstein+to:maxwell+subject:meeting
    """
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 50)), 100)

    if not query:
        return jsonify({"results": [], "total_count": 0, "error": "No query provided"})

    try:
        results, total_count = gmail_style_email_search(query, n_results=limit)
        return jsonify({
            "results": results,
            "total_count": total_count,
            "query": query,
        })
    except Exception as e:
        return jsonify({"results": [], "total_count": 0, "error": str(e)}), 500


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


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """AI-powered question answering over search results.

    POST /api/ask  { "question": "...", "mode": "text" }
    Runs a search, then synthesizes an answer from the top results.
    Cost: ~$0.001 per query using gpt-4o-mini.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "text")
    limit = min(int(data.get("limit", 10)), 20)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # First, expand the query for better search results
        expanded_query, related = ai_expand_query(question)
        search_query = expanded_query or question

        # Run the appropriate search
        error = None
        if mode == "semantic":
            results, error = semantic_search(search_query, n_results=limit)
        elif mode == "email":
            gmail_results, _ = gmail_style_email_search(search_query, n_results=limit)
            results = [{"record": {"metadata": r, "text_preview": r.get("body_preview", ""),
                                    "source": r.get("source", ""), "text_full": r.get("body_preview", "")},
                         "score": r["score"], "index": r.get("index", -1)} for r in gmail_results]
        else:
            results = text_search(search_query, n_results=limit)

        if error:
            return jsonify({"error": error}), 500

        # Synthesize answer
        answer, ai_error = ai_ask(question, results)

        resp = {
            "answer": answer,
            "expanded_query": expanded_query,
            "related_queries": related or [],
            "sources_used": len(results),
        }
        if ai_error:
            resp["error"] = ai_error

        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expand")
def api_expand():
    """Expand a search query into better terms.

    GET /api/expand?q=query
    Cost: ~$0.0003 per call using gpt-4o-mini.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        expanded, related = ai_expand_query(query)
        return jsonify({
            "original": query,
            "expanded": expanded,
            "related": related or [],
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
