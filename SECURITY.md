# Security Policy for hermes-forge

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Active development |

## Reporting a Vulnerability

If you discover a security vulnerability in hermes-forge, please:

1. **Do NOT open a public GitHub issue** — vulnerabilities should be reported privately.
2. Send details to the repository owner via GitHub's private vulnerability reporting: https://github.com/iknowkungfubar/hermes-forge/security/advisories
3. Include:
   - Type of vulnerability
   - Steps to reproduce
   - Affected versions
   - Potential impact
   - Suggested fix (if known)

You should receive a response within 48 hours. We'll keep you informed as the issue is investigated and resolved.

## Security Practices

### Code
- **No secrets in codebase** — API keys and tokens must use environment variables via `.env`
- **Input validation** — all user-facing inputs are validated for type, length, and dangerous patterns
- **No unsanitized `eval()` or `exec()`** — production code uses only safe parsing (json.loads, Pydantic models)
- **No shell injection** — no `os.system()` or `subprocess(shell=True)` in production code
- **SSRF protection** — the proxy validates backend URLs against a blocklist of internal/private IPs
- **Request size limits** — HTTP proxy enforces 10MB max body and 16KB max headers
- **Error sanitization** — internal paths and stack traces are never exposed to clients

### Dependencies
- Only well-maintained packages (httpx, pydantic, mcp)
- No optional dependencies with known critical CVEs
- All transitive dependencies are pinned via pip's resolution

### CI/CD
- GitHub Actions workflows run tests on every push
- GITHUB_TOKEN uses minimum required permissions
- No secrets exposed in CI logs

## Known Security Considerations

### MCP Server Tool Abuse
The forge MCP tools accept input that is validated for:
- Maximum string lengths (100K chars for text, 256 chars for tool names)
- Maximum array sizes (500 items for tool lists, 100 keys for arguments)
- Maximum counts (10K messages, 1M budget tokens)
- All errors return safe messages without internal details

### Proxy Server
The proxy HTTP server runs with these protections:
- Request body limited to 10MB
- Header size limited to 16KB
- URLs validated against SSRF blocklist (private IPs, metadata endpoints)
- Internal errors never leak stack traces or paths
- Backend client timeouts prevent resource exhaustion

### Plugin
The Hermes plugin uses the same validation as the MCP server. It does not:
- Execute arbitrary code from user input
- Open network connections to user-supplied hosts
- Read files outside its scope
