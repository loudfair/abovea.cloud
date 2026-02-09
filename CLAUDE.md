# CLAUDE.md - AI Agent Instructions

## Security (CRITICAL)
- **NEVER** read, display, or reference: `.env`, `.env.*`, `*.key`, `*.pem`, `credentials.*`, `secrets.*`
- **NEVER** commit secrets, API keys, or credentials
- **NEVER** push to remote - push is disabled in this working copy
- **NEVER** use `--force`, `--hard`, `--no-verify`
- **ALWAYS** fail if required env var is missing (no fallback values)

## Git Protocol
- Local commits only - push is blocked
- Imperative commit messages under 72 chars
- Verify no secrets before every commit
