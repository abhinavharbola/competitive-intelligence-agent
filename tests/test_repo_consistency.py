import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCH = (ROOT / "docs" / "architecture.md").read_text()
README = (ROOT / "README.md").read_text()

CODE_REFS = sorted(set(re.findall(r"`(agent/\w+\.py):(\w+)`", ARCH)))


def test_architecture_doc_has_code_references():
    assert CODE_REFS


@pytest.mark.parametrize("path,name", CODE_REFS)
def test_architecture_code_references_resolve(path, name):
    assert (ROOT / path).is_file()
    module = importlib.import_module(path[:-3].replace("/", "."))
    assert hasattr(module, name)


def test_readme_relative_links_resolve():
    links = re.findall(r"\]\(((?!http|#)[^)]+)\)", README)
    assert links
    for link in links:
        assert (ROOT / link.split("#")[0]).exists(), link


@pytest.mark.parametrize("name", [
    ".gitignore", ".env.example", "requirements.txt", "requirements-dev.txt",
    "pytest.ini", "memory/schema.sql", "eval/results/.gitkeep", "docs/architecture.md",
])
def test_expected_project_files_exist(name):
    assert (ROOT / name).exists()


def test_env_example_lists_every_config_key():
    keys = set(re.findall(r"^([A-Z_]+)=", (ROOT / ".env.example").read_text(), re.M))
    config_src = (ROOT / "config.py").read_text()
    used = set(re.findall(r'(?:_required|getenv)\("([A-Z_]+)"', config_src))
    assert used == keys


def test_every_source_module_imports():
    for path in ["agent.critic", "agent.executor", "agent.graph", "agent.guardrails", "agent.llm",
                 "agent.planner", "agent.schemas", "agent.state", "agent.synthesizer", "agent.tracing",
                 "api.main", "eval.judge", "eval.run_ablation", "tools.calculator", "tools.memory",
                 "tools.results", "tools.search"]:
        importlib.import_module(path)


