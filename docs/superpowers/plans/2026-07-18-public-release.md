# Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public MVP cloneable, testable, containerized, CI-verified, and independent of private price data.

**Architecture:** Preserve the current application and three-tool agent. Harden the public yfinance adapter at the normalization boundary, package the API in a non-root Python 3.11 image, and make GitHub Actions verify tests, deterministic eval output, and the Docker build.

**Tech Stack:** Python 3.11 release target, FastAPI, Pydantic, Chroma, pandas, yfinance 1.1.0, pytest, Docker, GitHub Actions.

## Global Constraints

- Keep exactly three tools and no private strategy/data code.
- Normal tests and CI must not require network market data, model credentials, or private files.
- Do not add vLLM, LangGraph, MCP, Redis, Celery, or cloud deployment.
- Do not add a license without an explicit owner choice.
- Docker must run as non-root and credentials must remain runtime-only.

---

### Task 1: Make Public Price Data Actually Runnable

**Files:**
- Modify: `requirements.txt`
- Modify: `app/tools.py:218-235`
- Test: `tests/test_tools.py`

**Interfaces:**
- Preserves: `_normalize_bars(frame: pd.DataFrame) -> pd.DataFrame`.
- Adds: support for current yfinance single-ticker MultiIndex columns.
- Adds: runtime dependency `yfinance==1.1.0`, the newest release compatible with the project's local Python 3.9 environment.

- [ ] **Step 1: Write the failing MultiIndex normalization test**

```python
def test_normalize_bars_flattens_single_ticker_yfinance_columns():
    frame = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1000]],
        index=pd.to_datetime(["2026-01-02"]),
        columns=pd.MultiIndex.from_tuples(
            [
                ("Open", "SPY"),
                ("High", "SPY"),
                ("Low", "SPY"),
                ("Close", "SPY"),
                ("Volume", "SPY"),
            ]
        ),
    )

    normalized = tools._normalize_bars(frame)

    assert normalized.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert normalized.index.tz is not None
    assert normalized.iloc[0]["close"] == 100.5
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_tools.py::test_normalize_bars_flattens_single_ticker_yfinance_columns -q`

Expected: `ValueError: price frame has no OHLCV columns`.

- [ ] **Step 3: Implement field-aware MultiIndex normalization**

Before the existing lowercase conversion, map each tuple to exactly one known OHLCV field:

```python
known_fields = {"open", "high", "low", "close", "volume"}
if isinstance(frame.columns, pd.MultiIndex):
    flattened: list[str] = []
    for column in frame.columns:
        matches = [str(part).lower() for part in column if str(part).lower() in known_fields]
        if len(matches) != 1:
            raise ValueError(f"ambiguous price column: {column!r}")
        flattened.append(matches[0])
    frame.columns = flattened
else:
    frame.columns = [str(column).lower() for column in frame.columns]
```

Reject duplicated flattened fields with `ValueError("price frame has duplicate OHLCV columns")`.

- [ ] **Step 4: Add and install the public dependency**

Append `yfinance==1.1.0` to `requirements.txt`, then run:

```bash
.venv/bin/python -m pip install yfinance==1.1.0
```

- [ ] **Step 5: Run focused and full tests**

```bash
.venv/bin/python -m pytest tests/test_tools.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass without a live Yahoo request.

- [ ] **Step 6: Commit the public fallback**

```bash
git add requirements.txt app/tools.py tests/test_tools.py
git commit -m "fix: support public yfinance price fallback"
```

---

### Task 2: Add Tested Container and CI Configuration

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_release.py`

**Interfaces:**
- Produces: Docker API listening on `0.0.0.0:8000` with writable `/app/.chroma`.
- Produces: CI checks for pytest, deterministic eval equality, and Docker build.

- [ ] **Step 1: Write failing release-configuration tests**

Create `tests/test_release.py` to read repository files and assert:

```python
def test_dockerfile_is_non_root_and_runs_api():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "USER appuser" in dockerfile
    assert 'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_dockerignore_excludes_secrets_and_local_state():
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for required in [".env", ".env.*", ".git", ".venv", ".chroma", "data", "outputs"]:
        assert required in ignored


def test_ci_runs_tests_eval_comparison_and_docker_build():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python -m pytest -q" in workflow
    assert "python -m app.eval_harness --output /tmp/eval-results.json" in workflow
    assert "diff -u eval/results.json /tmp/eval-results.json" in workflow
    assert "docker build --tag agent-quant-research:ci ." in workflow
```

- [ ] **Step 2: Run the tests and confirm missing-file failures**

Run: `.venv/bin/python -m pytest tests/test_release.py -q`

Expected: three `FileNotFoundError` failures.

- [ ] **Step 3: Add the non-root Dockerfile**

Use this runtime contract:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_DIR=/app/.chroma

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser
COPY app ./app
COPY eval ./eval
COPY sample.txt README.md ./
RUN mkdir -p /app/.chroma && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Add `.dockerignore`**

Include `.env`, `.env.*`, Git state, local virtual environments, Chroma/local data, outputs,
caches, tests, superpowers plans/specs, editor files, and OS files.

- [ ] **Step 5: Add read-only GitHub Actions CI**

Create `.github/workflows/ci.yml` with `permissions: contents: read`, checkout v4, setup-python v5
for Python 3.11 with pip cache, dependency installation, pytest, deterministic eval comparison,
and `docker build --tag agent-quant-research:ci .`.

- [ ] **Step 6: Run release and full tests**

```bash
.venv/bin/python -m pytest tests/test_release.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Parse workflow YAML and inspect Docker configuration**

```bash
.venv/bin/python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())"
git diff --check
```

Docker is unavailable locally; the pushed CI Docker-build step is the executable verification.

- [ ] **Step 8: Commit container and CI**

```bash
git add Dockerfile .dockerignore .github/workflows/ci.yml tests/test_release.py
git commit -m "build: add Docker and CI release checks"
```

---

### Task 3: Finish Public Documentation and Release Verification

**Files:**
- Modify: `KICKOFF.md`
- Modify: `README.md`
- Modify: `UNDERSTAND.md`

**Interfaces:**
- Produces: accurate clone/local/Docker verification instructions and release interview notes.

- [ ] **Step 1: Replace stale kickoff status**

Rewrite `KICKOFF.md` so it lists the completed RAG, three tools, event-study leakage guard,
orchestration traces/eval, and the final local/CI verification commands. Remove every claim that
`/research`, `/event-study`, or tools are stubs.

- [ ] **Step 2: Update README release instructions**

Add Docker build/run examples, document `CHROMA_DIR`, identify yfinance as the installed public
daily fallback, mark Day 7 complete, and state that GitHub Actions tests deterministic contracts
without credentials. Keep live model smoke testing optional.

- [ ] **Step 3: Add a release self-quiz**

Append questions to `UNDERSTAND.md` covering non-root containers, build-time versus runtime
secrets, deterministic CI versus live eval, why yfinance is a fallback rather than a production
market-data SLA, and why a Docker build is not a cloud deployment.

- [ ] **Step 4: Run complete verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m app.eval_harness --output /tmp/eval-results.json
cmp -s eval/results.json /tmp/eval-results.json
git diff --check
```

Expected: all tests pass, eval files match, and the diff check is clean.

- [ ] **Step 5: Scan public changes and commit**

Confirm no secret-shaped token, private dataset, or strategy file appears in `git status` or the
diff, then run:

```bash
git add KICKOFF.md README.md UNDERSTAND.md
git commit -m "docs: finish public MVP release"
```

- [ ] **Step 6: Push and report CI limitation**

```bash
git push origin main
git status --short --branch
```

Expected: `main` matches `origin/main`. Report that local Docker execution was unavailable and
that GitHub Actions is responsible for the actual image build verification.
