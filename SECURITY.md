# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.5.x (latest) | Yes |
| < 0.5.0 | No |

Only the latest released version receives security fixes. If you are running an older version, please upgrade before reporting.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues by email to **security@bellkis.com** (or **hello@bellkis.com** if the security address is not yet active). Please use the subject line: `[ShrimPK Security] <brief summary>`.

We aim to acknowledge all reports within **72 hours** and will keep you informed as we work toward a resolution.

### What to Include in Your Report

A useful report includes at minimum:

1. **Description** — what the vulnerability is and where it exists (component, file, function if known)
2. **Reproduction steps** — a minimal, step-by-step sequence to trigger the issue
3. **Impact assessment** — what an attacker could accomplish by exploiting it (data exposure, privilege escalation, denial of service, etc.)
4. **Environment** — OS, ShrimPK version, relevant configuration

The more detail you provide, the faster we can triage and respond.

---

## Scope

### In Scope

The following components are in scope for security reports:

- **Echo Memory engine** (`shrimpk-memory`) — retrieval pipeline, Bloom filter, LSH index, cosine scoring, Hebbian/recency weighting, reranker
- **Daemon** (`shrimpk-daemon`) — background process, IPC interface, socket/pipe handling
- **MCP server** (`shrimpk-mcp`) — Model Context Protocol server, tool dispatch, input validation
- **CLI** (`cli`) — command-line interface, argument parsing, privilege requirements
- **Persistence format** — on-disk memory store, serialization/deserialization, file permissions
- **PII handling** — the PII masking system in `shrimpk-core` that detects and redacts sensitive data before storage

### Out of Scope

The following are **not** in scope:

- Third-party dependencies (fastembed, simsimd, etc.) — report these to their respective maintainers
- The Ollama runtime or any local model inference engine — ShrimPK does not control Ollama
- Theoretical vulnerabilities with no practical attack path
- Issues already reported or publicly disclosed

---

## PII and Data Handling

ShrimPK stores conversation memories locally on the user's machine. The `shrimpk-core` crate contains a PII masking subsystem that detects and redacts sensitive data patterns (email addresses, phone numbers, credentials, etc.) before memories are persisted.

If you discover a case where PII is stored in plaintext when it should have been masked, or where the masking system can be bypassed, that is in scope and should be reported immediately.

---

## Disclosure Policy

We follow **coordinated disclosure**:

1. You report the vulnerability privately to security@bellkis.com.
2. We acknowledge receipt within 72 hours.
3. We work with you to understand and reproduce the issue.
4. We develop and test a fix.
5. We release the fix and publish an advisory.
6. The standard disclosure timeline is **90 days** from the date of your initial report. If we need more time, we will communicate that to you and agree on an extension.

We ask that you do not publicly disclose the vulnerability until we have released a fix or the 90-day window has elapsed, whichever comes first.

---

## Credit

We are grateful to security researchers who help keep ShrimPK safe. With your permission, we will credit you by name (or handle) in the security advisory and in the project CHANGELOG. If you prefer to remain anonymous, please say so in your report.

---

## License

ShrimPK is released under the [Apache License 2.0](LICENSE). This security policy is part of that project and is covered by the same license.
