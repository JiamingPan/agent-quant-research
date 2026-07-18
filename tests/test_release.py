from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_is_non_root_and_runs_api():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "USER appuser" in dockerfile
    assert (
        'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]'
        in dockerfile
    )
    assert "HEALTHCHECK" in dockerfile


def test_dockerignore_excludes_secrets_and_local_state():
    ignored = (REPO_ROOT / ".dockerignore").read_text(
        encoding="utf-8"
    ).splitlines()

    for required in [
        ".env",
        ".env.*",
        ".git",
        ".venv",
        ".chroma",
        "data",
        "outputs",
    ]:
        assert required in ignored


def test_ci_runs_tests_eval_comparison_and_docker_build():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )

    assert "python -m pytest -q" in workflow
    assert "python -m app.eval_harness --output /tmp/eval-results.json" in workflow
    assert "diff -u eval/results.json /tmp/eval-results.json" in workflow
    assert "docker build --tag agent-quant-research:ci ." in workflow
