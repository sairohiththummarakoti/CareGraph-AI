import pytest
from pathlib import Path
from uuid import uuid4

from caregraph.retrieval import build_index
from tests.helpers import DeterministicEmbedder


def pytest_configure(config):
    """Keep Windows temporary test files in an isolated, disposable workspace."""
    if config.option.basetemp is None:
        workspace = Path(__file__).resolve().parents[1]
        temporary_parent = (workspace / ".pytest-work").resolve()
        if not temporary_parent.is_relative_to(workspace):
            raise RuntimeError("Test temporary files must stay inside the workspace.")
        temporary_parent.mkdir(exist_ok=True)
        config.option.basetemp = str(temporary_parent / ("run-" + uuid4().hex))


@pytest.fixture
def embedder():
    return DeterministicEmbedder()


@pytest.fixture
def reference_index(tmp_path, embedder):
    location = tmp_path / "indexes"
    build_index(location, embedder=embedder)
    return location


@pytest.fixture
def report_bytes():
    return (
        "SYNTHETIC PATHOLOGY REPORT\n"
        "Tumor site: Breast\nHistology: Synthetic carcinoma example\n"
        "Grade: 2\nTumor size: 2 cm\nStage: not provided\nBiomarkers: pending\n"
    ).encode()
