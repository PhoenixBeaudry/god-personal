from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_validator_env_dockerfile_installs_with_real_package_roots():
    dockerfile = REPO_ROOT / "ops/docker/validator-env.dockerfile"
    lines = dockerfile.read_text().splitlines()

    install_index = next(i for i, line in enumerate(lines) if line.startswith("RUN pip install") and line.endswith(" ."))
    prior_lines = lines[:install_index]

    assert "RUN mkdir -p src && touch src/__init__.py" not in prior_lines
    for package_root in ["core", "trainer", "validator", "ops"]:
        assert f"COPY {package_root}/ {package_root}/" in prior_lines
