#!/usr/bin/env python3
"""
Security audit script — scans all downloaded data for threats before building the index.

Run standalone:  python audit.py

Exit code 0 = clean, 1 = threats found.
"""

import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCAN_DIR = Path(__file__).resolve().parent / "downloads"

# Only first N bytes are read for content inspection (speed).
HEAD_BYTES = 8 * 1024  # 8 KB

# Large-file threshold (flag for manual review).
LARGE_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

# Extensions that are inherently executable / installer packages.
EXECUTABLE_EXTENSIONS = frozenset({
    ".exe", ".bat", ".sh", ".ps1", ".cmd", ".msi",
    ".app", ".dmg", ".deb", ".rpm",
    ".com", ".scr", ".vbs", ".wsf", ".wsh",
})

# Expected binary formats — anything binary that is NOT one of these is suspicious.
EXPECTED_BINARY_EXTENSIONS = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".svg", ".ico",
    ".parquet", ".sqlite", ".db", ".sqlite3",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".zst",
    ".csv", ".tsv", ".json", ".jsonl", ".xml", ".txt", ".md",
})

# Pickle extensions — unsafe deserialization risk.
PICKLE_EXTENSIONS = frozenset({".pkl", ".pickle", ".joblib"})

# Credential / secret patterns (compiled once).
CREDENTIAL_PATTERNS = [
    re.compile(rb"(?:sk-[a-zA-Z0-9]{20,})"),                          # OpenAI / Stripe
    re.compile(rb"(?:ghp_[a-zA-Z0-9]{36,})"),                         # GitHub PAT
    re.compile(rb"(?:gho_[a-zA-Z0-9]{36,})"),                         # GitHub OAuth
    re.compile(rb"(?:AKIA[0-9A-Z]{16})"),                              # AWS access key
    re.compile(rb"(?:password|passwd|pwd)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(rb"(?:api[_-]?key|apikey)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(rb"(?:secret[_-]?key|secretkey)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(rb"(?:access[_-]?token|accesstoken)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(rb"(?:Bearer\s+[a-zA-Z0-9_.~+/=-]{20,})"),             # Bearer tokens
]

# Suspicious code patterns per extension.
SUSPICIOUS_CODE = {
    ".js": [
        re.compile(rb"\beval\s*\(", re.IGNORECASE),
        re.compile(rb"\bexec\s*\(", re.IGNORECASE),
        re.compile(rb"\bFunction\s*\(", re.IGNORECASE),
    ],
    ".py": [
        re.compile(rb"\bos\.system\s*\("),
        re.compile(rb"\bsubprocess\b"),
        re.compile(rb"\bexec\s*\("),
        re.compile(rb"\b__import__\s*\("),
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

console = Console()


def _read_head(path: Path) -> bytes:
    """Read at most HEAD_BYTES from a file. Returns b'' on error."""
    try:
        with open(path, "rb") as f:
            return f.read(HEAD_BYTES)
    except (OSError, PermissionError):
        return b""


def _is_binary(head: bytes) -> bool:
    """Heuristic: file is binary if it contains null bytes in the head."""
    return b"\x00" in head


def _has_exec_permission(path: Path) -> bool:
    """Check if any execute bit is set."""
    try:
        mode = path.stat().st_mode
        return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class AuditResult:
    """Collects scan findings."""

    def __init__(self) -> None:
        self.total_files = 0
        self.extension_counts: Counter = Counter()
        self.threats: list[tuple[str, str]] = []      # (file, reason)
        self.warnings: list[tuple[str, str]] = []     # (file, reason)

    # Convenience ----------------------------------------------------------
    def threat(self, path: str, reason: str) -> None:
        self.threats.append((path, reason))

    def warning(self, path: str, reason: str) -> None:
        self.warnings.append((path, reason))

    @property
    def is_clean(self) -> bool:
        return len(self.threats) == 0


def scan(scan_dir: Path) -> AuditResult:
    """Recursively scan *scan_dir* and return an AuditResult."""
    result = AuditResult()
    scan_dir_resolved = scan_dir.resolve()

    if not scan_dir.is_dir():
        console.print(f"[yellow]Scan directory does not exist: {scan_dir}[/yellow]")
        return result

    for entry in sorted(scan_dir.rglob("*")):
        # Skip .git directories entirely.
        try:
            rel = entry.relative_to(scan_dir)
        except ValueError:
            continue
        if ".git" in rel.parts:
            continue

        # --- Symlink check (before anything else) ---
        if entry.is_symlink():
            target = entry.resolve()
            try:
                target.relative_to(scan_dir_resolved)
            except ValueError:
                result.threat(str(rel), f"Symlink points outside downloads/ → {target}")
            continue  # don't inspect symlink targets further

        if not entry.is_file():
            continue

        result.total_files += 1
        ext = entry.suffix.lower()
        result.extension_counts[ext if ext else "(no ext)"] += 1

        # --- Executable extension ---
        if ext in EXECUTABLE_EXTENSIONS:
            result.threat(str(rel), f"Executable extension: {ext}")

        # --- Execute permission ---
        if _has_exec_permission(entry):
            result.threat(str(rel), "File has execute permission (+x)")

        # --- Pickle files ---
        if ext in PICKLE_EXTENSIONS:
            result.threat(str(rel), f"Pickle file ({ext}) — unsafe deserialization risk")

        # --- Large files ---
        try:
            size = entry.stat().st_size
        except OSError:
            size = 0
        if size > LARGE_FILE_BYTES:
            result.warning(str(rel), f"Large file: {size / (1024**2):.1f} MB — review manually")

        # --- Content-based checks (first 8 KB) ---
        head = _read_head(entry)
        if not head:
            continue

        # Shebang
        if head.startswith(b"#!"):
            result.threat(str(rel), "Contains shebang (#!) — potential script")

        # Credential patterns
        for pat in CREDENTIAL_PATTERNS:
            m = pat.search(head)
            if m:
                # Redact the matched value for safety.
                matched = m.group(0)
                preview = matched[:40] + (b"..." if len(matched) > 40 else b"")
                result.warning(
                    str(rel),
                    f"Possible credential/secret: {preview.decode('utf-8', errors='replace')}",
                )
                break  # one match per file is enough

        # Suspicious code in scripts
        if ext in SUSPICIOUS_CODE:
            for pat in SUSPICIOUS_CODE[ext]:
                m = pat.search(head)
                if m:
                    result.threat(
                        str(rel),
                        f"Suspicious code pattern: {m.group(0).decode('utf-8', errors='replace')}",
                    )
                    break

        # HTML with inline scripts or javascript: URLs
        if ext in (".html", ".htm", ".xhtml"):
            head_lower = head.lower()
            if b"<script" in head_lower:
                result.warning(str(rel), "HTML file contains <script> tag")
            if b"javascript:" in head_lower:
                result.warning(str(rel), "HTML file contains javascript: URL")

        # Unexpected binary
        if _is_binary(head) and ext not in EXPECTED_BINARY_EXTENSIONS and ext not in PICKLE_EXTENSIONS:
            result.warning(str(rel), f"Unexpected binary file (extension: {ext or 'none'})")

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(result: AuditResult) -> None:
    """Print a clean summary using rich."""

    console.print()
    console.rule("[bold]Security Audit Report[/bold]")
    console.print()

    # --- Summary ---
    console.print(f"  Scan directory : [cyan]{SCAN_DIR}[/cyan]")
    console.print(f"  Total files    : [bold]{result.total_files}[/bold]")
    console.print()

    # --- Extension breakdown ---
    if result.extension_counts:
        ext_table = Table(title="Files by Extension", show_lines=False, padding=(0, 2))
        ext_table.add_column("Extension", style="cyan", min_width=12)
        ext_table.add_column("Count", justify="right", style="bold")
        for ext, count in result.extension_counts.most_common():
            ext_table.add_row(ext, str(count))
        console.print(ext_table)
        console.print()

    # --- Threats ---
    if result.threats:
        threat_table = Table(
            title=f"[bold red]Threats ({len(result.threats)})[/bold red]",
            show_lines=True,
            border_style="red",
        )
        threat_table.add_column("File", style="red", ratio=2)
        threat_table.add_column("Reason", style="bold red", ratio=3)
        for filepath, reason in result.threats:
            threat_table.add_row(filepath, reason)
        console.print(threat_table)
        console.print()

    # --- Warnings ---
    if result.warnings:
        warn_table = Table(
            title=f"[bold yellow]Warnings ({len(result.warnings)})[/bold yellow]",
            show_lines=True,
            border_style="yellow",
        )
        warn_table.add_column("File", style="yellow", ratio=2)
        warn_table.add_column("Reason", style="bold yellow", ratio=3)
        for filepath, reason in result.warnings:
            warn_table.add_row(filepath, reason)
        console.print(warn_table)
        console.print()

    # --- Verdict ---
    if result.is_clean:
        verdict = Panel(
            Text("PASS — No threats detected", style="bold green", justify="center"),
            border_style="green",
            padding=(1, 4),
        )
    else:
        verdict = Panel(
            Text(
                f"FAIL — {len(result.threats)} threat(s) found",
                style="bold red",
                justify="center",
            ),
            border_style="red",
            padding=(1, 4),
        )

    console.print(verdict)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    result = scan(SCAN_DIR)
    print_report(result)
    return 0 if result.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
