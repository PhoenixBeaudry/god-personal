import ast
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(package: str) -> list[Path]:
    return sorted((ROOT / package).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _assert_no_imports(package: str, forbidden_roots: set[str]) -> None:
    offenders = {}
    for path in _python_files(package):
        forbidden_imports = _imported_roots(path) & forbidden_roots
        if forbidden_imports:
            offenders[path.relative_to(ROOT)] = sorted(forbidden_imports)
    assert offenders == {}


def test_core_stays_service_independent():
    _assert_no_imports("core", {"trainer", "validator"})


def test_services_do_not_import_each_other():
    _assert_no_imports("trainer", {"validator"})
    _assert_no_imports("validator", {"trainer"})
