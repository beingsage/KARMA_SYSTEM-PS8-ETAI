# Vulnerability Inventory

Scope: repository code review of the current backend and pipeline surface.

This file lists the concrete security and exposure issues I found in code. It is intentionally conservative: I only included items that are directly supported by the source.

## Summary

| Severity | Issue | Evidence | Impact |
| --- | --- | --- | --- |
| High | Wildcard CORS with credentials | `app/main.py:72-79` | Any origin can interact with credentialed browser requests if cookies/session auth are added later, and it broadens cross-origin abuse today. |
| High | Hardcoded Neo4j defaults and unauthenticated fallback | `app/config.py:65-73`, `app/pipeline/neo4j_store.py:51-89` | Secrets are embedded in code defaults and the graph store will try default/no-auth connections if configured credentials fail. |
| High | Public admin backfill/migration routes | `app/main.py:320-390` | The ontology backfill and migration endpoints are callable without any auth gate. |
| Medium | Public job/result endpoints expose raw analysis payloads | `app/main.py:274-301`, `app/storage.py:60-87` | Any caller who knows or obtains a job id can read the saved job payload and normalized bundle with no access control. |
| Medium | Public advanced analytics / copilot endpoints with no auth or throttling | `app/main.py:393-819` | Expensive model and reasoning endpoints are exposed directly and can be abused for compute exhaustion or data leakage. |
| Medium | Neo4j transport is explicitly unencrypted | `app/pipeline/neo4j_store.py:78-82` | Bolt traffic can be observed or modified on an untrusted network if Neo4j is not already protected elsewhere. |

## Findings

### 1) Wildcard CORS with credentials

Evidence:
- `app/main.py:72-79`

What happens:
- `allow_origins=["*"]`
- `allow_credentials=True`
- `allow_methods=["*"]`
- `allow_headers=["*"]`

Why this matters:
- This is overly permissive for any browser-based deployment.
- If auth cookies, bearer tokens, or other credentialed flows are introduced, any origin can attempt cross-origin interaction.
- Even now, the policy is broader than necessary for a single-page frontend served from the same app.

Fix:
- Restrict origins to the exact frontend hostnames.
- Keep credentials disabled unless they are required.
- Prefer same-origin deployment for the UI and API if possible.

### 2) Hardcoded Neo4j defaults and no-auth fallback

Evidence:
- `app/config.py:65-73`
- `app/pipeline/neo4j_store.py:51-89`

What happens:
- `NEO4J_PASSWORD` defaults to a literal password in code.
- `NEO4J_INITIAL_PASSWORD` also defaults to the same literal password.
- The Neo4j store builds credential candidates using the configured user/password, then a default `neo4j` password, then `None` auth.
- On success with `None`, the code logs that it connected without authentication.

Why this matters:
- Hardcoded defaults are secret leakage by design.
- A failed auth path can degrade into weaker credentials or no auth at all.
- This is especially risky because the graph store persists application data and ontology state.

Fix:
- Remove real password defaults from code.
- Require explicit environment configuration for Neo4j credentials.
- Remove the no-auth fallback in production.
- Fail closed if authentication is not configured.

### 3) Public admin backfill and migration routes

Evidence:
- `app/main.py:320-327`
- `app/main.py:376-390`

What happens:
- `POST /api/v1/ontology/backfill`
- `POST /api/v1/admin/neo4j/migrate-ontology`
- Neither route is protected by any authentication or authorization check in this module.

Why this matters:
- These routes can rewrite graph metadata and run broad ontology migrations.
- In the wrong hands, they can mutate or corrupt the entire graph.
- Even accidental use from a frontend or integration test can have production impact.

Fix:
- Put these routes behind admin-only auth.
- Prefer a separate internal-only deployment or network boundary.
- Require an explicit operator token and log every invocation.
- Keep the current dry-run capability, but do not leave the live path public.

### 4) Public job/result endpoints expose saved analysis payloads

Evidence:
- `app/main.py:274-301`
- `app/storage.py:60-87`

What happens:
- Job payloads are written to `data/jobs/{uuid}.json`.
- `GET /api/v1/jobs/{job_id}` returns the stored payload.
- `GET /api/v1/workflows/{job_id}/bundle` returns a normalized version of the same data, optionally including the raw job.
- There is no access control around job lookup.

Why this matters:
- Job ids are UUIDs, so they are not trivially guessable, but they are still bearer-like identifiers.
- Any leaked id exposes the raw pipeline output, reasoning, ontology proposals, and evidence.
- This becomes a direct data exposure problem if users upload sensitive documents.

Fix:
- Add per-job ownership or an access token.
- Require authentication before returning raw or bundled job data.
- Consider expiring or encrypting job payloads at rest if sensitive documents are processed.

### 5) Public advanced analytics and reasoning endpoints

Evidence:
- `app/main.py:393-819`

What happens:
- The module exposes copilot endpoints, advanced model endpoints, and pipeline introspection endpoints directly.
- Examples include RCA, maintenance, compliance, risk, vector search, graph reasoning, doc query, anomaly detection, RUL prediction, failure prediction, lessons learned, clustering, graph embeddings, and pipeline stage enumeration.

Why this matters:
- These routes are compute-heavy.
- They can be abused for denial of service.
- They can also expose backend reasoning and model capabilities to untrusted callers.

Fix:
- Add auth and authorization boundaries.
- Rate-limit expensive routes.
- Split admin/ops routes from public UI routes.
- Consider a job-scoped token for any route that reads stored analysis data.

### 6) Neo4j transport is explicitly unencrypted

Evidence:
- `app/pipeline/neo4j_store.py:78-82`

What happens:
- The Neo4j driver is instantiated with `encrypted=False`.

Why this matters:
- Bolt traffic may be readable on a shared or untrusted network.
- Credentials and graph content can be exposed if Neo4j is not isolated behind a trusted local boundary.

Fix:
- Enable encryption in production.
- Use TLS certificates or a secured local-only deployment.
- Make the transport mode environment-driven instead of hardcoded.

## Recommended Remediation Order

1. Remove hardcoded Neo4j passwords and no-auth fallback.
2. Add authentication and authorization for all admin and job-read routes.
3. Restrict CORS to the actual frontend origin(s).
4. Enable encrypted Neo4j transport.
5. Add rate limiting to advanced reasoning and analytics endpoints.
6. Introduce job ownership or per-job access tokens before exposing raw bundles.

## Notes

- I did not see a route-level auth dependency or middleware layer in the current `app/main.py` surface.
- This inventory focuses on code-exposed risk, not model quality or ontology correctness.
- If you want, I can turn this file into a tracked remediation checklist and mark the fixes in priority order.
