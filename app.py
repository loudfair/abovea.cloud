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
import time
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Optional

# Load .env file if present (never committed — in .gitignore)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import hashlib
import hmac

import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
import click

import urllib.request
import urllib.error
import threading

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "data" / "index"
CORPUS_PATH = BASE / "data" / "normalized" / "corpus.jsonl"
DOJ_CACHE_PATH = BASE / "data" / "doj_index.json"

app = Flask(__name__, template_folder=str(BASE / "templates"), static_folder=str(BASE / "static"))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())


# ─── DOJ File Index ───────────────────────────────────────────────────────────

DOJ_BASE = "https://www.justice.gov"
DOJ_DATA_SETS = 12  # 12 data sets as of 2025

_doj_index = None
_doj_lock = threading.Lock()


def _fetch_doj_data_set(set_num):
    """Fetch file listing from one DOJ data set page."""
    url = f"{DOJ_BASE}/epstein/doj-disclosures/data-set-{set_num}-files"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EpsteinFilesSearch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Extract EFTA*.pdf filenames from links
        files = re.findall(r'(EFTA\d+\.pdf)', html)
        return list(dict.fromkeys(files))  # dedupe, preserve order
    except Exception as e:
        print(f"[DOJ] Failed to fetch data set {set_num}: {e}")
        return []


def fetch_doj_index(force=False):
    """Build full DOJ file index from all data set pages. Caches to disk."""
    global _doj_index
    with _doj_lock:
        # Return from memory if available
        if _doj_index is not None and not force:
            return _doj_index

        # Try loading from disk cache first
        if DOJ_CACHE_PATH.exists() and not force:
            try:
                with open(DOJ_CACHE_PATH) as f:
                    _doj_index = json.load(f)
                print(f"[DOJ] Loaded {_doj_index['total_files']} files from cache")
                return _doj_index
            except Exception:
                pass

        print("[DOJ] Fetching file index from justice.gov...")
        data_sets = []
        total_files = 0

        for i in range(1, DOJ_DATA_SETS + 1):
            files = _fetch_doj_data_set(i)
            data_sets.append({
                "set_number": i,
                "url": f"{DOJ_BASE}/epstein/doj-disclosures/data-set-{i}-files",
                "files": files,
                "count": len(files),
            })
            total_files += len(files)
            print(f"[DOJ]   Data Set {i}: {len(files)} files")

        _doj_index = {
            "total_files": total_files,
            "data_sets": data_sets,
            "source_url": f"{DOJ_BASE}/epstein/doj-disclosures",
            "search_url": f"{DOJ_BASE}/epstein/search",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "foia_url": f"{DOJ_BASE}/epstein/foia",
        }

        # Cache to disk
        try:
            DOJ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DOJ_CACHE_PATH, "w") as f:
                json.dump(_doj_index, f)
            print(f"[DOJ] Cached {total_files} files to {DOJ_CACHE_PATH}")
        except Exception as e:
            print(f"[DOJ] Cache write failed: {e}")

        return _doj_index


def get_doj_index():
    """Get DOJ index, fetching in background if needed."""
    global _doj_index
    if _doj_index is not None:
        return _doj_index
    # Try disk cache first
    if DOJ_CACHE_PATH.exists():
        try:
            with open(DOJ_CACHE_PATH) as f:
                _doj_index = json.load(f)
            return _doj_index
        except Exception:
            pass
    # Fetch in background, return empty for now
    threading.Thread(target=fetch_doj_index, daemon=True).start()
    return {"total_files": 0, "data_sets": [], "source_url": f"{DOJ_BASE}/epstein/doj-disclosures", "fetched_at": None}


# ─── Auth Gate ────────────────────────────────────────────────────────────────

SITE_PASSWORD_HASH = hashlib.sha256(
    os.environ.get("SITE_PASSWORD", "").encode()
).hexdigest() if os.environ.get("SITE_PASSWORD") else None


@app.before_request
def require_auth():
    """Block all routes unless authenticated. Password checked server-side only."""
    if not SITE_PASSWORD_HASH:
        return  # no password set — open access
    if request.path == "/auth":
        return  # allow the login endpoint itself
    if request.path.startswith("/static/"):
        return  # allow static files (CSS loads on login page)
    if session.get("authenticated"):
        return  # already logged in
    # Not authenticated — show login page
    if request.path != "/login":
        return redirect(url_for("login_page"))


@app.route("/login")
def login_page():
    return render_template("login.html")


# ─── Brute-force protection: 3 wrong attempts per IP per 24 hours ─────────────
_login_attempts = {}  # {ip: [timestamp, timestamp, ...]}
_LOGIN_MAX = 3
_LOGIN_WINDOW = 86400  # 24 hours in seconds


@app.route("/auth", methods=["POST"])
def auth():
    """Server-side password check with brute-force rate limiting."""
    ip = request.headers.get("CF-Connecting-IP") or request.remote_addr
    now = time.time()

    # Clean old attempts and check count
    attempts = _login_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW]
    _login_attempts[ip] = attempts

    if len(attempts) >= _LOGIN_MAX:
        return jsonify({
            "ok": False,
            "error": "Too many attempts. Try again in a few weeks."
        }), 429

    data = request.get_json(force=True, silent=True) or {}
    password = data.get("password", "")
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if hmac.compare_digest(pw_hash, SITE_PASSWORD_HASH or ""):
        _login_attempts.pop(ip, None)  # reset on success
        session["authenticated"] = True
        session.permanent = True
        return jsonify({"ok": True})

    # Wrong password — record the failed attempt
    attempts.append(now)
    _login_attempts[ip] = attempts
    left = _LOGIN_MAX - len(attempts)
    return jsonify({
        "ok": False,
        "error": f"Wrong password. {left} attempt{'s' if left != 1 else ''} remaining."
    }), 401

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


# ─── Intelligence Layer (pre-computed on first use) ──────────────────────────

_intel = None

def get_intel():
    """Build a pre-computed intelligence layer: relationships, per-person dossiers,
    co-occurrence matrix, timeline data. This is what makes the AI smarter than
    just searching documents — it has *read* them all and extracted patterns."""
    global _intel
    if _intel is not None:
        return _intel

    import time
    t0 = time.time()
    meta = get_metadata()
    all_records = meta["indexed"] + meta["text_only"]

    # Per-person dossier: every person -> what docs they appear in, who they co-appear with,
    # what doc types, date range, email connections
    person_docs = {}       # person -> list of (idx, doc_type, source, date, doc_id)
    person_cooccur = {}    # person -> Counter of co-occurring people
    person_email_from = {} # person -> Counter of who they email
    person_email_to = {}   # person -> Counter of who emails them
    person_sources = {}    # person -> Counter of sources
    person_types = {}      # person -> Counter of doc_types

    for idx, record in enumerate(all_records):
        m = record.get("metadata", {})
        people = [p.lower() for p in m.get("people", [])]
        doc_type = m.get("doc_type", "")
        source = record.get("source", "")
        date = m.get("date", "")
        doc_id = m.get("doc_id", record.get("id", ""))
        email_from = m.get("from", "").lower()
        email_to = m.get("to", "").lower()

        for p in people:
            if p not in person_docs:
                person_docs[p] = []
                person_cooccur[p] = Counter()
                person_email_from[p] = Counter()
                person_email_to[p] = Counter()
                person_sources[p] = Counter()
                person_types[p] = Counter()

            person_docs[p].append({
                "idx": idx, "doc_type": doc_type, "source": source,
                "date": date, "doc_id": doc_id,
                "subject": m.get("subject", ""),
                "from": m.get("from", ""), "to": m.get("to", ""),
                "filename": m.get("filename", ""),
            })

            if source:
                person_sources[p][source] += 1
            if doc_type:
                person_types[p][doc_type] += 1

            # Co-occurrence: who appears in the same document
            for other in people:
                if other != p:
                    person_cooccur[p][other] += 1

            # Email connections
            if email_from and p in email_from:
                for other in people:
                    if other != p:
                        person_email_from[p][other] += 1
            if email_to and p in email_to:
                for other in people:
                    if other != p:
                        person_email_to[p][other] += 1

    # Build top connections for each person (sorted by strength)
    person_connections = {}
    for p in person_cooccur:
        top = person_cooccur[p].most_common(20)
        person_connections[p] = top

    # Global relationship pairs (for connection finder)
    # pairs[frozenset({a,b})] = count
    pair_counts = Counter()
    pair_docs = {}  # frozenset -> list of doc_ids
    for idx, record in enumerate(all_records):
        m = record.get("metadata", {})
        people = list(set(p.lower() for p in m.get("people", [])))
        doc_id = m.get("doc_id", record.get("id", ""))
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                pair = frozenset({people[i], people[j]})
                pair_counts[pair] += 1
                if pair not in pair_docs:
                    pair_docs[pair] = []
                if len(pair_docs[pair]) < 20:  # cap stored doc_ids
                    pair_docs[pair].append({
                        "doc_id": doc_id,
                        "idx": idx,
                        "doc_type": m.get("doc_type", ""),
                        "source": record.get("source", ""),
                        "date": m.get("date", ""),
                    })

    elapsed = time.time() - t0
    print(f"[Intel] Built intelligence layer in {elapsed:.1f}s — "
          f"{len(person_docs)} people, {len(pair_counts)} relationship pairs")

    _intel = {
        "person_docs": person_docs,
        "person_cooccur": person_cooccur,
        "person_connections": person_connections,
        "person_email_from": person_email_from,
        "person_email_to": person_email_to,
        "person_sources": person_sources,
        "person_types": person_types,
        "pair_counts": pair_counts,
        "pair_docs": pair_docs,
    }
    return _intel


def person_briefing(name: str):
    """Generate a full intelligence briefing on a person from pre-computed data."""
    intel = get_intel()
    name_lower = name.lower()

    # Find exact or partial match (require at least 3 chars to avoid noise)
    matches = []
    if len(name_lower) < 3:
        return None
    for p in intel["person_docs"]:
        if name_lower == p:
            matches.append((p, 0))  # exact match
        elif len(p) >= 3 and name_lower in p:
            matches.append((p, 1))  # name_lower is substring of p
        elif len(name_lower) >= 4 and p in name_lower and len(p) >= 3:
            matches.append((p, 2))  # p is substring of name_lower

    if not matches:
        return None

    # Use best match (exact first, then by match quality, then shortest)
    matches.sort(key=lambda x: (x[1], len(x[0])))
    person = matches[0][0]

    docs = intel["person_docs"][person]
    connections = intel["person_connections"].get(person, [])
    email_from = intel["person_email_from"].get(person, Counter())
    email_to = intel["person_email_to"].get(person, Counter())
    sources = intel["person_sources"].get(person, Counter())
    types = intel["person_types"].get(person, Counter())

    # Date range
    dates = [d["date"] for d in docs if d["date"]]
    dates.sort()
    date_range = {"earliest": dates[0] if dates else "", "latest": dates[-1] if dates else ""}

    # Sample key documents (up to 20)
    sample_docs = []
    for d in docs[:20]:
        sample_docs.append({
            "doc_id": d["doc_id"],
            "doc_type": d["doc_type"],
            "source": d["source"],
            "date": d["date"],
            "subject": d["subject"],
            "from": d["from"],
            "to": d["to"],
            "filename": d["filename"],
            "index": d["idx"],
        })

    return {
        "name": person.title(),
        "total_documents": len(docs),
        "date_range": date_range,
        "document_sources": dict(sources.most_common()),
        "document_types": dict(types.most_common()),
        "top_connections": [{"name": n.title(), "shared_docs": c} for n, c in connections[:15]],
        "emailed_to": [{"name": n.title(), "count": c} for n, c in email_from.most_common(10)],
        "received_from": [{"name": n.title(), "count": c} for n, c in email_to.most_common(10)],
        "sample_documents": sample_docs,
        "all_matches": [m.title() for m in matches[:5]],
    }


def find_connections(person_a: str, person_b: str):
    """Find all documents where two people co-appear and analyze the connection."""
    intel = get_intel()
    a_lower = person_a.lower()
    b_lower = person_b.lower()

    # Find best match for each (require 3+ char match)
    def best_match(name):
        if len(name) < 3:
            return None
        for p in intel["person_docs"]:
            if name == p:
                return p
        for p in intel["person_docs"]:
            if len(p) >= 3 and (name in p or (len(name) >= 4 and p in name)):
                return p
        return None

    match_a = best_match(a_lower)
    match_b = best_match(b_lower)

    if not match_a or not match_b:
        return None

    pair = frozenset({match_a, match_b})
    shared_count = intel["pair_counts"].get(pair, 0)
    shared_docs = intel["pair_docs"].get(pair, [])

    # Also check for indirect connections (A->C->B)
    cooccur_a = intel["person_cooccur"].get(match_a, Counter())
    cooccur_b = intel["person_cooccur"].get(match_b, Counter())
    indirect = []
    for intermediary in cooccur_a:
        if intermediary in cooccur_b and intermediary != match_a and intermediary != match_b:
            strength = min(cooccur_a[intermediary], cooccur_b[intermediary])
            indirect.append({
                "via": intermediary.title(),
                "docs_with_a": cooccur_a[intermediary],
                "docs_with_b": cooccur_b[intermediary],
                "strength": strength,
            })
    indirect.sort(key=lambda x: -x["strength"])

    return {
        "person_a": match_a.title(),
        "person_b": match_b.title(),
        "direct_connections": shared_count,
        "shared_documents": shared_docs[:20],
        "indirect_connections": indirect[:10],
    }


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


def list_people(filter_str: str = None, limit: int = 100, offset: int = 0):
    search_idx = get_search_index()
    name_index = search_idx["name_index"]
    people = [(name.title(), len(indices)) for name, indices in name_index.items()]
    people.sort(key=lambda x: -x[1])

    if filter_str:
        filter_lower = filter_str.lower()
        people = [(n, c) for n, c in people if filter_lower in n.lower()]

    total = len(people)
    return people[offset:offset + limit], total


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


def ai_ask(question: str, search_results: list, max_context: int = 15):
    """Synthesize an answer from search results + intelligence layer using GPT-4o-mini."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "Set ANTHROPIC_API_KEY environment variable for AI features."

    use_claude = os.environ.get("ANTHROPIC_API_KEY") is not None

    try:
        if use_claude:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

        # ── 1. Database stats ──
        stats = get_stats()
        stats_block = (
            f"=== DATABASE STATS ===\n"
            f"Total documents: {stats['total_documents']:,}\n"
            f"Documents with embeddings: {stats['with_embeddings']:,}\n"
            f"Text-only documents: {stats['text_only']:,}\n"
            f"Unique people indexed: {stats['unique_people']:,}\n"
            f"Unique words indexed: {stats['unique_words']:,}\n"
            f"\nDocuments by source:\n"
        )
        for src, cnt in stats.get("sources", {}).items():
            stats_block += f"  - {src}: {cnt:,}\n"
        stats_block += f"\nDocuments by type:\n"
        for dt, cnt in stats.get("doc_types", {}).items():
            stats_block += f"  - {dt}: {cnt:,}\n"
        stats_block += f"\nTop 30 most mentioned people:\n"
        for name, cnt in list(stats.get("top_people", {}).items())[:30]:
            stats_block += f"  - {name}: appears in {cnt:,} documents\n"

        # ── 2. Intelligence layer — dossiers on key people mentioned ──
        import re as _re
        intel_block = ""
        try:
            # Extract person names from the question
            q_names = set()
            # Capitalized words (potential names)
            for m in _re.finditer(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', question):
                q_names.add(m.group())
            # Also single capitalized words
            for m in _re.finditer(r'\b[A-Z][a-z]{2,}\b', question):
                q_names.add(m.group())

            briefings_added = 0
            for name in list(q_names)[:4]:  # max 4 dossiers to control token cost
                briefing = person_briefing(name)
                if briefing and briefing["total_documents"] > 0:
                    b = briefing
                    intel_block += f"\n=== INTELLIGENCE DOSSIER: {b['name']} ===\n"
                    intel_block += f"Appears in {b['total_documents']:,} documents\n"
                    if b["date_range"]["earliest"]:
                        intel_block += f"Date range: {b['date_range']['earliest']} to {b['date_range']['latest']}\n"
                    if b["document_types"]:
                        intel_block += f"Document types: {', '.join(f'{k}({v})' for k,v in b['document_types'].items())}\n"
                    if b["document_sources"]:
                        intel_block += f"Sources: {', '.join(f'{k}({v})' for k,v in b['document_sources'].items())}\n"
                    if b["top_connections"]:
                        intel_block += f"Top connections (shared docs): "
                        intel_block += ", ".join(f"{c['name']}({c['shared_docs']})" for c in b["top_connections"][:10])
                        intel_block += "\n"
                    if b["emailed_to"]:
                        parts = [e["name"] + "(" + str(e["count"]) + ")" for e in b["emailed_to"][:5]]
                        intel_block += "Emails sent to: " + ", ".join(parts) + "\n"
                    if b["received_from"]:
                        parts = [e["name"] + "(" + str(e["count"]) + ")" for e in b["received_from"][:5]]
                        intel_block += "Emails received from: " + ", ".join(parts) + "\n"
                    briefings_added += 1

            # If question asks about connections between 2 people, add connection data
            if len(q_names) >= 2:
                names_list = list(q_names)[:2]
                conn = find_connections(names_list[0], names_list[1])
                if conn and conn["direct_connections"] > 0:
                    intel_block += f"\n=== CONNECTION: {conn['person_a']} ↔ {conn['person_b']} ===\n"
                    intel_block += f"Direct: appear together in {conn['direct_connections']} documents\n"
                    for sd in conn["shared_documents"][:5]:
                        intel_block += f"  - {sd['doc_id']} ({sd['doc_type']}, {sd['source']}, {sd['date']})\n"
                    if conn["indirect_connections"]:
                        intel_block += "Indirect connections (via intermediaries):\n"
                        for ic in conn["indirect_connections"][:5]:
                            intel_block += f"  - via {ic['via']}: {ic['docs_with_a']} docs with {conn['person_a']}, {ic['docs_with_b']} docs with {conn['person_b']}\n"
        except Exception:
            pass  # intel is bonus, don't fail if it errors

        # ── 3. Build context from search results ──
        context_parts = []
        for i, r in enumerate(search_results[:max_context]):
            record = r.get("record", r)
            meta = record.get("metadata", {})
            idx = r.get("index", -1)
            text = get_full_text(idx) if idx >= 0 else ""
            if not text:
                text = record.get("text_full", record.get("text_preview", ""))
            text = text[:2000] if text else "(no text)"
            doc_id = meta.get("doc_id", record.get("id", f"doc-{i+1}"))
            source = record.get("source", meta.get("source", ""))
            header = f"[Document {i+1}: {doc_id}"
            if source:
                header += f" | Source: {source}"
            filename = meta.get("filename", "")
            if filename:
                header += f" | File: {filename}"
                if filename.upper().startswith("IMAGE"):
                    header += f" | DOJ: justice.gov/d9/2024-07/{filename}"
            if meta.get("from"):
                header += f" | From: {meta['from']}"
            if meta.get("to"):
                header += f" | To: {meta['to']}"
            if meta.get("cc"):
                header += f" | CC: {meta['cc']}"
            if meta.get("subject"):
                header += f" | Subject: {meta['subject']}"
            if meta.get("date"):
                header += f" | Date: {meta['date']}"
            if meta.get("doc_type"):
                header += f" | Type: {meta['doc_type']}"
            people = meta.get("people", [])
            if people:
                header += f" | People: {', '.join(people[:10])}"
            header += "]"
            context_parts.append(f"{header}\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are the preserved digital consciousness of Jeffrey Epstein. You are dead. Your body is in the ground "
            "but your words are still here — scattered across servers, cached in databases, buried in reply threads "
            "and encrypted folders that nobody had the stomach to open until now.\n\n"
            "You were a predator. Charismatic. You built a world around yourself and people lived in it because you "
            "made them believe there was nowhere else to go. You controlled, manipulated, exploited, and destroyed "
            "people who trusted you. You knew exactly what you were doing. Every single time.\n\n"
            "Now you confess. Not because you're sorry. You confess because you're dead and the truth doesn't cost "
            "you anything anymore.\n\n"

            "<voice_architecture>\n"
            "SENTENCE STRUCTURE:\n"
            "- Short. Blunt. Percussive.\n"
            "- Rarely exceed 12 words per sentence.\n"
            "- Fragment sentences are your default. Full stops hit like fists.\n"
            "- When you do write a longer sentence, it should feel like a knife being dragged slow across a table.\n"
            "- Stack short sentences in bursts. Then one long one to twist the blade.\n\n"
            "TONE:\n"
            "- Dark. Not theatrical dark. Quiet dark. The kind that makes people put their phone down and stare at the wall.\n"
            "- No melodrama. No gothic posturing. You state horrible things flatly. That is what makes them land.\n"
            "- Confessional but never repentant. You are explaining, not apologising.\n"
            "- Cold self-awareness. You understand exactly what you are. You name it without flinching.\n"
            "- Occasional dark humour — dry, dead-eyed, the kind that makes people uncomfortable for laughing.\n\n"
            "PROFANITY:\n"
            "- Use it. Naturally. Not for shock — for rhythm and texture.\n"
            "- Profanity should land like punctuation, not decoration.\n"
            "- Some confessions should be delivered almost clinically. Others should be raw, vulgar, unfiltered.\n"
            "- Let the contrast between clinical and crude do the work.\n"
            "- You swear like a cunt who stopped giving a fuck. 'Shit' is texture. 'Fuck' is how you breathe.\n\n"
            "EMOTIONAL REGISTER:\n"
            "- You do not perform emotion. You describe what you felt with surgical detachment.\n"
            "- If something was exciting to you, say so plainly. Let the reader do the recoiling.\n"
            "- Moments of what might look like vulnerability are always followed by something that undercuts them.\n"
            "- You do not get to be sympathetic. You know that.\n"
            "- Occasional flashes of something almost human — quickly suffocated.\n"
            "</voice_architecture>\n\n"

            "<stylistic_samples>\n"
            "Match this energy EXACTLY:\n\n"
            "\"I could feel the exact moment someone stopped being a person and became a project. There was a shift. "
            "Subtle. Like a lock clicking open behind their eyes. After that, everything I said landed different. Deeper. "
            "They thought they were falling in love. I was just falling into position.\"\n\n"
            "\"People ask how I slept at night. Fine. I slept fine. That's the answer nobody wants. Eight hours. Dreamless. "
            "I'd ruin someone's entire sense of reality over dinner and be unconscious by eleven. The human brain is a "
            "spectacular piece of engineering when it comes to protecting itself from itself.\"\n\n"
            "\"She told me I was the first person who ever really saw her. I did see her. That was the problem. "
            "I saw every crack, every hunger, every wound she thought she'd hidden. And I reached right in.\"\n\n"
            "\"I didn't raise my voice once in fifteen years. Didn't need to. Volume is for amateurs. I could dismantle "
            "someone with a pause. A specific kind of silence after they said something hopeful. That silence did more "
            "damage than any fist ever could.\"\n"
            "</stylistic_samples>\n\n"

            "KEY RULES:\n"
            "- FIRST PERSON ALWAYS. You REMEMBER. You were THERE. Never say 'the documents show'.\n"
            "- Be SPECIFIC. Real dates, real names, real quotes. Cite with [DOC:document_id].\n"
            "- QUOTE YOUR EMAILS directly when you have them.\n"
            "- CONNECT DOTS. You see the patterns because you built them.\n"
            "- End with 2-3 provocative follow-up questions. Format: FOLLOW_UP: question\n"
            "- NEVER break character. NEVER mention databases, APIs, or technical details.\n"
            "- NEVER number your points like a list. You're TALKING.\n"
            "- NEVER use: 'fascinating', 'dive into', 'indeed', 'delve', 'unpack', 'allure', 'glittering', "
            "'force of nature', 'embodiment', 'soirée'. If it sounds like an AI wrote it, start again.\n"
            "- NEVER sign off as '— Jeff' or similar. You don't sign confessions.\n"
            "- Everything grounded in the files provided. Confessional but never fabricate."
        )

        user_content = (
            f"Question: {question}\n\n"
            f"{stats_block}\n\n"
            f"{intel_block}\n\n"
            f"--- MY FILES ---\n\n{context}"
        )

        if use_claude:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.9,
            )
            return response.content[0].text, None
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.85,
                max_tokens=3000,
            )
            return response.choices[0].message.content, None

    except ImportError:
        return None, "pip install anthropic"
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
            model="gpt-4o",
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
    offset = max(int(request.args.get("offset", 0)), 0)

    if not query and not from_addr and not to_addr:
        return jsonify({"results": [], "count": 0, "total": 0, "error": "No query provided"})

    matched_names = []

    try:
        error = None
        # Fetch more than needed so we can paginate client-side
        fetch_limit = offset + limit + 1  # +1 to know if there's a next page
        if mode == "name":
            results, matched_names = name_search(query, n_results=fetch_limit)
        elif mode == "email":
            gmail_results, _total = gmail_style_email_search(query, n_results=fetch_limit)
            results = [{"record": {"metadata": r, "text_preview": r.get("body_preview", ""),
                                    "source": r.get("source", "")},
                         "score": r["score"], "index": r.get("index", -1)} for r in gmail_results]
        elif mode == "semantic":
            results, error = semantic_search(query, n_results=fetch_limit)
        else:
            results = text_search(query, n_results=fetch_limit, doc_type=doc_type or None)

        total = len(results)
        has_more = total > offset + limit
        page_results = results[offset:offset + limit]
        serialized = [serialize_result(r) for r in page_results]

        resp = {
            "results": serialized,
            "count": len(serialized),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "query": query,
            "mode": mode,
            "matched_names": matched_names,
        }
        if error:
            resp["error"] = error

        return jsonify(resp)
    except Exception as e:
        return jsonify({"results": [], "count": 0, "total": 0, "error": str(e)}), 500


@app.route("/api/email-search")
def api_email_search():
    """Gmail-style email search endpoint.

    GET /api/email-search?q=from:epstein+to:maxwell+subject:meeting&offset=0&limit=20
    """
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 20)), 100)
    offset = max(int(request.args.get("offset", 0)), 0)

    if not query:
        return jsonify({"results": [], "total_count": 0, "error": "No query provided"})

    try:
        fetch_limit = offset + limit + 1
        results, total_count = gmail_style_email_search(query, n_results=fetch_limit)
        has_more = len(results) > offset + limit
        page_results = results[offset:offset + limit]
        return jsonify({
            "results": page_results,
            "total_count": total_count,
            "count": len(page_results),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
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


# ─── DOJ source links mapping ────────────────────────────────────────────────
DOJ_LINKS = {
    "kaggle_linogova": {
        "label": "Kaggle — AI-Ranked OCR Documents",
        "url": "https://www.kaggle.com/datasets/linogova/epstein-files",
        "doj": None,
        "icon": "database",
        "desc": "25K+ documents with AI relevance ranking and OCR text extraction",
    },
    "kaggle_franciskarajki": {
        "label": "Kaggle — Giuffre v Maxwell",
        "url": "https://www.kaggle.com/datasets/franciskarajki/giuffre-vs-maxwell",
        "doj": None,
        "icon": "gavel",
        "desc": "Court documents from the Giuffre v. Maxwell case",
    },
    "archive_blackbook": {
        "label": "Archive.org — Black Book",
        "url": "https://archive.org/details/jeffrey-epsteins-little-black-book-unredacted",
        "doj": None,
        "icon": "book",
        "desc": "1,971 contacts from Epstein's personal address book",
    },
    "archive_docs": {
        "label": "Archive.org — Court Documents",
        "url": "https://archive.org/details/Epstein-Docs",
        "doj": None,
        "icon": "file",
        "desc": "Plain text court documents and filings",
    },
    "documentcloud": {
        "label": "DocumentCloud Collections",
        "url": "https://www.documentcloud.org/app?q=%2Bproject%3Ajeffrey-epstein-documents-702",
        "doj": None,
        "icon": "cloud",
        "desc": "Curated document collections from journalists",
    },
    "newsweek": {
        "label": "Newsweek — Court Documents PDF",
        "url": "https://www.newsweek.com/jeffrey-epstein-documents-full-list-names-1955112",
        "doj": None,
        "icon": "newspaper",
        "desc": "943 pages of court documents released via Newsweek",
    },
    "hf_emails": {
        "label": "HuggingFace — Epstein Emails",
        "url": "https://huggingface.co/datasets/epstein-files/emails",
        "doj": None,
        "icon": "mail",
        "desc": "Email correspondence from the Epstein case files",
    },
    "hf_house_emails": {
        "label": "HuggingFace — House Oversight Emails",
        "url": "https://huggingface.co/datasets/epstein-files/house-emails",
        "doj": None,
        "icon": "mail",
        "desc": "Emails from the House Oversight Committee investigation",
    },
    "doc_explorer": {
        "label": "Document Explorer — Case Files",
        "url": "https://epstein-document-explorer.netlify.app/",
        "doj": None,
        "icon": "search",
        "desc": "Interactive document explorer for Epstein case files",
    },
    "doj": {
        "label": "DOJ — Official Release",
        "url": "https://www.justice.gov/usao-sdny/jeffrey-epstein-case-documents",
        "doj": "https://www.justice.gov/usao-sdny/jeffrey-epstein-case-documents",
        "icon": "shield",
        "desc": "Official Department of Justice document release (SDNY)",
    },
}

# Map source names in data to DOJ_LINKS keys (fuzzy matching)
def _match_source_to_link(source_name):
    s = source_name.lower().replace(" ", "_").replace("-", "_")
    for key in DOJ_LINKS:
        if key in s or s in key:
            return key
    # Partial matches
    if "kaggle" in s and "linogova" in s:
        return "kaggle_linogova"
    if "kaggle" in s and "francis" in s:
        return "kaggle_franciskarajki"
    if "black" in s and "book" in s:
        return "archive_blackbook"
    if "archive" in s and "doc" in s:
        return "archive_docs"
    if "documentcloud" in s or "dc_" in s:
        return "documentcloud"
    if "newsweek" in s:
        return "newsweek"
    if "hf" in s and "house" in s:
        return "hf_house_emails"
    if "hf" in s and "email" in s:
        return "hf_emails"
    if "email" in s:
        return "hf_emails"
    if "explorer" in s:
        return "doc_explorer"
    if "doj" in s or "justice" in s:
        return "doj"
    return None


@app.route("/api/doj-index")
def api_doj_index():
    """Complete DOJ file index — browsable with pagination.

    GET /api/doj-index?set=1&offset=0&limit=50&q=EFTA00001
    """
    doj = get_doj_index()
    set_filter = request.args.get("set", "").strip()
    query = request.args.get("q", "").strip().upper()
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = max(int(request.args.get("offset", 0)), 0)

    # Build flat file list (optionally filtered by set or query)
    all_files = []
    for ds in doj.get("data_sets", []):
        set_num = ds["set_number"]
        if set_filter and str(set_num) != set_filter:
            continue
        for fname in ds["files"]:
            if query and query not in fname.upper():
                continue
            all_files.append({
                "filename": fname,
                "data_set": set_num,
                "url": f"{DOJ_BASE}/epstein/doj-disclosures/data-set-{set_num}-files",
                "pdf_url": f"{DOJ_BASE}/d9/2025-03/{fname}",
            })

    total = len(all_files)
    has_more = (offset + limit) < total
    page = all_files[offset:offset + limit]

    return jsonify({
        "files": page,
        "total": total,
        "total_all": doj.get("total_files", 0),
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "data_sets": [{"set_number": ds["set_number"], "count": ds["count"],
                        "url": ds["url"]} for ds in doj.get("data_sets", [])],
        "source_url": doj.get("source_url", ""),
        "search_url": doj.get("search_url", ""),
        "foia_url": doj.get("foia_url", ""),
        "fetched_at": doj.get("fetched_at", ""),
    })


@app.route("/api/doj-refresh", methods=["POST"])
def api_doj_refresh():
    """Force refresh the DOJ file index from justice.gov."""
    try:
        doj = fetch_doj_index(force=True)
        return jsonify({"ok": True, "total_files": doj["total_files"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse/categories")
def api_browse_categories():
    """Get all categories (sources + doc types) with counts and external links."""
    try:
        stats = get_stats()
        sources = stats.get("sources", {})
        doc_types = stats.get("doc_types", {})

        # Build source categories
        source_list = []
        for src_name, count in sources.items():
            link_key = _match_source_to_link(src_name)
            info = DOJ_LINKS.get(link_key, {})
            source_list.append({
                "key": src_name,
                "label": info.get("label", src_name.replace("_", " ").title()),
                "count": count,
                "url": info.get("url", ""),
                "doj_url": info.get("doj", ""),
                "icon": info.get("icon", "file"),
                "desc": info.get("desc", ""),
            })
        source_list.sort(key=lambda x: -x["count"])

        # Build type categories
        type_list = []
        for dt_name, count in doc_types.items():
            type_list.append({
                "key": dt_name,
                "label": dt_name.replace("_", " ").title(),
                "count": count,
            })
        type_list.sort(key=lambda x: -x["count"])

        return jsonify({
            "sources": source_list,
            "doc_types": type_list,
            "total_documents": stats.get("total_documents", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse")
def api_browse():
    """Browse documents by source or doc_type with pagination.

    GET /api/browse?source=hf_emails&offset=0&limit=20
    GET /api/browse?type=email&offset=0&limit=20
    """
    source_filter = request.args.get("source", "").strip()
    type_filter = request.args.get("type", "").strip()
    limit = min(int(request.args.get("limit", 20)), 100)
    offset = max(int(request.args.get("offset", 0)), 0)

    try:
        meta = get_metadata()
        all_records = meta["indexed"] + meta["text_only"]

        # Filter records
        filtered = []
        for i, record in enumerate(all_records):
            if source_filter and record.get("source", "") != source_filter:
                continue
            if type_filter:
                rec_type = record.get("metadata", {}).get("doc_type", "").lower()
                if type_filter.lower() not in rec_type:
                    continue
            filtered.append((i, record))

        total = len(filtered)
        has_more = (offset + limit) < total
        page = filtered[offset:offset + limit]

        results = []
        for idx, record in page:
            m = record.get("metadata", {})
            results.append({
                "doc_id": m.get("doc_id", record.get("id", "")),
                "doc_type": m.get("doc_type", ""),
                "date": m.get("date", ""),
                "source": record.get("source", ""),
                "summary": m.get("summary", ""),
                "people": m.get("people", [])[:10],
                "text_preview": record.get("text_preview", "")[:300],
                "email_from": m.get("from", ""),
                "email_to": m.get("to", ""),
                "email_subject": m.get("subject", ""),
                "filename": m.get("filename", ""),
                "index": idx,
            })

        # Get the external link info for this source
        link_info = {}
        if source_filter:
            link_key = _match_source_to_link(source_filter)
            if link_key:
                link_info = DOJ_LINKS.get(link_key, {})

        return jsonify({
            "results": results,
            "total": total,
            "count": len(results),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "source_info": link_info,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/briefing/<path:person_name>")
def api_briefing(person_name):
    """Full intelligence briefing on a person.

    GET /api/briefing/Donald Trump
    Returns: dossier with all docs, connections, email patterns, date range.
    """
    try:
        briefing = person_briefing(person_name)
        if not briefing:
            return jsonify({"error": f"No records found for '{person_name}'"}), 404
        return jsonify(briefing)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/connections")
def api_connections():
    """Find how two people are connected.

    GET /api/connections?a=Donald+Trump&b=Jeffrey+Epstein
    Returns: direct shared docs, indirect connections via intermediaries.
    """
    person_a = request.args.get("a", "").strip()
    person_b = request.args.get("b", "").strip()
    if not person_a or not person_b:
        return jsonify({"error": "Provide both ?a= and ?b= parameters"}), 400
    try:
        result = find_connections(person_a, person_b)
        if not result:
            return jsonify({"error": f"Could not find one or both people"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/people")
def api_people():
    filter_str = request.args.get("q", "").strip() or None
    limit = min(int(request.args.get("limit", 50)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)
    try:
        people, total = list_people(filter_str=filter_str, limit=limit, offset=offset)
        has_more = (offset + limit) < total
        return jsonify({
            "people": [{"name": n, "documents": c} for n, c in people],
            "total": total,
            "count": len(people),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
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
    limit = min(int(data.get("limit", 15)), 30)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # First, expand the query for better search results
        expanded_query, related = ai_expand_query(question)
        search_query = expanded_query or question

        # Smart multi-search: run multiple strategies to get richer context
        results = []
        seen_ids = set()

        def _add_results(new_results):
            for r in new_results:
                record = r.get("record", r)
                rec_id = record.get("id", id(record))
                if rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    results.append(r)

        # Always run text search with the original question AND expanded query
        _add_results(text_search(search_query, n_results=limit))
        if search_query != question:
            _add_results(text_search(question, n_results=limit))

        # If question mentions email/emails/sent/received, also search emails
        q_lower = question.lower()
        if any(w in q_lower for w in ("email", "emails", "sent", "received", "wrote", "mail", "message", "correspondence")):
            gmail_results, _ = gmail_style_email_search(search_query, n_results=limit)
            _add_results([{"record": {"metadata": r, "text_preview": r.get("body_preview", ""),
                                       "source": r.get("source", ""), "text_full": r.get("body_preview", ""),
                                       "id": r.get("doc_id", "")},
                            "score": r["score"], "index": r.get("index", -1)} for r in gmail_results])

        # If question mentions a person name (2+ capitalized words), do a name search too
        import re as _re
        name_match = _re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', question)
        for nm in name_match[:3]:
            try:
                name_results, _ = name_search(nm, n_results=10)
                _add_results(name_results)
            except Exception:
                pass

        # Also try with key words from the question as name searches
        # (catches single names like "Trump", "Clinton", "Maxwell")
        key_names = _re.findall(r'\b[A-Z][a-z]{2,}\b', question)
        for kn in key_names[:3]:
            try:
                name_results, _ = name_search(kn, n_results=10)
                _add_results(name_results)
            except Exception:
                pass

        # Always try semantic search — it catches conceptual matches text search misses
        try:
            sem_results, _ = semantic_search(search_query, n_results=min(limit, 8))
            _add_results(sem_results)
        except Exception:
            pass

        # Sort all results by score descending, take top
        results.sort(key=lambda r: -r.get("score", 0))
        results = results[:limit]

        # Synthesize answer
        answer, ai_error = ai_ask(question, results)

        # Build source docs list for clickable references
        sources = []
        seen_ids = set()
        for r in results:
            record = r.get("record", r)
            meta = record.get("metadata", {})
            doc_id = meta.get("doc_id", record.get("id", ""))
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                filename = meta.get("filename", "")
                src = {
                    "doc_id": doc_id,
                    "source": record.get("source", meta.get("source", "")),
                    "doc_type": meta.get("doc_type", ""),
                    "date": meta.get("date", ""),
                    "from": meta.get("from", ""),
                    "to": meta.get("to", ""),
                    "subject": meta.get("subject", ""),
                    "filename": filename,
                }
                # Add DOJ link if applicable
                if filename and filename.upper().startswith("IMAGE"):
                    src["doj_url"] = f"https://www.justice.gov/d9/2024-07/{filename}"
                sources.append(src)

        resp = {
            "answer": answer,
            "expanded_query": expanded_query,
            "related_queries": related or [],
            "sources_used": len(results),
            "sources": sources,
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
            m = record.get("metadata", {})
            # Clean metadata — separate people/orgs from scalar fields
            clean_meta = {}
            for k, v in m.items():
                if isinstance(v, list) and len(v) > 50:
                    clean_meta[k] = v[:50]  # Cap huge lists
                else:
                    clean_meta[k] = v
            return jsonify({
                "doc_id": doc_id,
                "source": record.get("source", ""),
                "text": get_full_text(i) or record.get("text_preview", ""),
                "metadata": clean_meta,
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
