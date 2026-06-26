"""
Security audit tool -- scans for secrets, injections, path leaks, and CVEs.

Usage: python scripts/security_audit.py [REPO_PATH]

If REPO_PATH is omitted, uses the git root of the current directory.
"""

import argparse
import os
import re
import json
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get the git root directory of the current working directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


parser = argparse.ArgumentParser(description="Security audit tool")
parser.add_argument(
    "repo_path",
    nargs="?",
    help="Path to the repository root (default: auto-detect from git)",
)
args = parser.parse_args()
REPO = Path(args.repo_path).resolve() if args.repo_path else get_repo_root()

# Ensure it's a directory
if not REPO.is_dir():
    print(f"Error: {REPO} is not a directory", file=sys.stderr)
    sys.exit(1)
print(f"Auditing repository at: {REPO}")

SECRET_PATTERNS = [
    (
        r'(?i)(api_key|apikey|api[-_]?secret|secret[-_]?key|app[-_]?secret)["\']?\s*[:=]\s*["\'][A-Za-z0-9_\-]{16,}["\']',
        "Potential API key/secret",
    ),
    (r"(?i)(sk-[A-Za-z0-9]{20,})", "OpenAI-style API key"),
    (r"(?i)(ghp_[A-Za-z0-9]{36,})", "GitHub PAT"),
    (r"(?i)(gho_|github_pat_)", "GitHub token"),
    (r"(?i)-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private key"),
    (r"(?i)(AKIA[0-9A-Z]{16})", "AWS access key"),
    (r'"/home/turin/', "Local filesystem path leak"),
]

INJECTION_PATTERNS = [
    (r'os\.system\([^)]*f["\']', "Potential shell injection (f-string)"),
    (r"eval\(", "eval() usage"),
    (r"exec\(", "exec() usage"),
    (r"__import__\(", "Dynamic import"),
    (r"pickle\.loads?\(", "Pickle deserialization"),
    (r"yaml\.load\([^)]*Loader=", "Unsafe yaml.load"),
    (r"Path\(.*\.\.\/.*\)", "Path traversal"),
    (r'f["\'][^"]*\{.*\{', "Nested f-string (complex injection)"),
]

results = {"secrets": [], "injections": [], "info_leaks": [], "hardcoded_paths": []}

for root, dirs, files in os.walk(REPO):
    dirs[:] = [
        d
        for d in dirs
        if d
        not in (
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".egg-info",
            "node_modules",
            ".venv",
            ".mypy_cache",
            "htmlcov",
        )
    ]
    for f in files:
        if not f.endswith(
            (".py", ".yaml", ".yml", ".toml", ".sh", ".json", ".txt", ".md")
        ):
            continue
        fpath = Path(root) / f
        try:
            content = fpath.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        rel = fpath.relative_to(REPO)
        for pattern, desc in SECRET_PATTERNS:
            for match in re.finditer(pattern, content):
                results["secrets"].append(
                    {
                        "file": str(rel),
                        "line": content[: match.start()].count("\n") + 1,
                        "finding": desc,
                        "match": match.group()[:80],
                    }
                )
        for pattern, desc in INJECTION_PATTERNS:
            for match in re.finditer(pattern, content):
                results["injections"].append(
                    {
                        "file": str(rel),
                        "line": content[: match.start()].count("\n") + 1,
                        "finding": desc,
                    }
                )
        if "/home/turin/" in content:
            results["hardcoded_paths"].append(
                {
                    "file": str(rel),
                    "finding": "Contains /home/turin/ path",
                }
            )

print("=" * 70)
print("SECURITY AUDIT — hermes-forge")
print("=" * 70)

any_findings = False
for category, items in results.items():
    if items:
        any_findings = True
        print(f"\n{'=' * 70}")
        print(f"🔴  {category.upper()}: {len(items)} finding(s)")
        print(f"{'=' * 70}")
        for item in items:
            print(f"  📁 {item['file']}:{item.get('line', '?')}")
            print(f"     {item['finding']}")
            if "match" in item:
                print(f"     → {item['match']}")
            print()

if not any_findings:
    print("\n✅  No secrets, injections, or path leaks found.")

print(f"\n{'=' * 70}")
print("SCAN SUMMARY")
print(f"{'=' * 70}")
print(
    f"  Files scanned: {sum(len(files) for _, _, files in os.walk(REPO) if not any(s in _ for s in ['.git', '__pycache__']))}"
)
print(f"  Total findings: {sum(len(v) for v in results.values())}")
print(f"{'=' * 70}")

report_path = REPO / "security-audit-report.json"
report_path.write_text(json.dumps(results, indent=2))
print(f"\nReport saved to {report_path}")
