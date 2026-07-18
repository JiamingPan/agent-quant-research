# Public Release Design

## Goal

Finish the standalone public MVP so a reviewer can clone it, run tests, reproduce the offline
evaluation, start the API locally or in Docker, and use public daily price data without access to
the private `spx-news-intraday` repository.

## Public Price Fallback

Add `yfinance==1.1.0` as a normal dependency. It is the newest release compatible with the
project's local Python 3.9 environment and the Python 3.11 container target. Keep the private cache loader as the first optional
source, but make the public fallback fully installed and documented. Normalize the MultiIndex
columns returned by current single-ticker `yfinance.download` calls into the existing lowercase
OHLCV contract. Reject ambiguous columns rather than silently selecting the wrong field.

No live market-data request belongs in normal tests or CI. Unit tests will use representative
DataFrames.

## Container

Add a single production-style `Dockerfile` based on `python:3.11-slim`. Install dependencies,
copy only runtime files, create a writable Chroma directory, run as a non-root user, expose port
8000, add a standard-library health check, and start Uvicorn on `0.0.0.0:8000`.

Add `.dockerignore` entries for Git state, local environments, credentials, caches, Chroma data,
private/local data, tests, plans, and OS/editor files. Credentials remain runtime environment
variables and never enter the image.

Docker is unavailable on the development Mac. GitHub Actions will be the executable Docker-build
verification; local work will statically inspect the Dockerfile and ignore set.

## Continuous Integration

Add one GitHub Actions workflow on pushes and pull requests. It will:

1. Check out the repository.
2. Install Python 3.11 with pip caching.
3. Install `requirements.txt`.
4. Run the complete pytest suite.
5. Regenerate offline evaluation into `/tmp` and compare it byte-for-byte with
   `eval/results.json`.
6. Build the Docker image.

The workflow receives read-only repository permissions and no API credentials. The optional live
model smoke test remains excluded.

## Documentation

Replace the stale kickoff checklist with an accurate shipped-MVP status and a final verification
checklist. Update the README with local, Docker, CI, environment-variable, public data-source, and
remaining-production-path guidance. Add a short release/operations quiz to `UNDERSTAND.md`.

The final build-status checklist will mark Day 7 complete. It will continue to describe live
trading, private strategies, distributed orchestration, and production infrastructure as out of
scope.

## Non-Goals

- Deploying to a cloud provider
- Adding vLLM, LangGraph, MCP, Redis, Celery, or another tool
- Copying private code, data, credentials, or strategy logic
- Publishing provider-dependent live-eval results
- Adding a software license without an explicit license choice from the repository owner

## Acceptance Criteria

- A MultiIndex yfinance fixture normalizes to lowercase OHLCV columns.
- `yfinance==1.1.0` installs in the project environment.
- All tests pass without network or credentials.
- Offline evaluation is byte-identical across reruns.
- CI syntax parses and the workflow includes tests, eval comparison, and Docker build.
- Docker configuration runs as non-root and excludes credentials/local state from build context.
- No stale document says the event study or agent is still a stub.
- Public changes contain no credentials or private strategy/data files.
- All changes are committed and pushed to `main`.
