from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import REPO_ROOT, Settings
from app.embeddings import FakeEmbedder
from app.ingest import ingest_laws
from app.llm import FakeLLM, Message
from app.main import create_app
from app.parsing import Law, load_laws
from app.store import InMemoryStore

CLEAN_DIR = REPO_ROOT / "data" / "clean"


@pytest.fixture(scope="session")
def laws() -> dict[str, Law]:
    return load_laws(CLEAN_DIR)


@pytest.fixture(scope="session")
def clean_text() -> Callable[[str], str]:
    def read(name: str) -> str:
        return (CLEAN_DIR / name).read_text(encoding="utf-8")

    return read


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def filled_store(laws: dict[str, Law], embedder: FakeEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    ingest_laws(laws, embedder, store)
    return store


def fake_settings(**overrides: object) -> Settings:
    base = {"embedder": "fake", "llm": "fake", "vector_store": "memory", "min_similarity": 0.1}
    return Settings.model_validate({**base, **overrides})


@pytest.fixture
def make_client(laws: dict[str, Law], embedder: FakeEmbedder, filled_store: InMemoryStore):
    """Build a TestClient around fakes; tests choose the LLM behaviour and settings."""

    def build(
        responder: Callable[[Sequence[Message]], str] | None = None,
        llm: object | None = None,
        **settings: object,
    ) -> tuple[TestClient, FakeLLM | object, InMemoryStore]:
        chosen_llm = llm if llm is not None else FakeLLM(responder)
        app = create_app(fake_settings(**settings), laws=laws, embedder=embedder, store=filled_store, llm=chosen_llm)
        return TestClient(app), chosen_llm, filled_store

    return build


def data_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)
