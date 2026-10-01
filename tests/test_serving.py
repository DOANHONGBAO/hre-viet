import sqlite3

import pandas as pd
from fastapi.testclient import TestClient

from hre_translate.models.dictionary import DictionaryTranslator
from hre_translate.models.retrieval import TranslationMemory
from hre_translate.serving.app import create_app
from hre_translate.serving.engine import ModelUnavailableError, ServingEngine
from hre_translate.serving.feedback import FeedbackStore


class FakeNeural:
    available = True
    loaded = True

    def translate(self, text):
        return {"translation": f"neural:{text}"}


class UnavailableNeural:
    available = False
    loaded = False

    def translate(self, text):
        raise ModelUnavailableError("adapter missing")


def client(tmp_path):
    lexicon = DictionaryTranslator([("A B", "cụm AB")])
    memory = TranslationMemory().fit(
        pd.DataFrame({"hre": ["A B C", "Q R S"], "vi": ["câu ABC", "câu QRS"]})
    )
    engine = ServingEngine(
        lexicon, memory, FakeNeural(),
        medium_threshold=0.5, high_threshold=0.85, top_k=2,
    )
    db = tmp_path / "feedback.sqlite3"
    return TestClient(create_app(engine=engine, feedback_store=FeedbackStore(db))), db


def test_health_and_models(tmp_path):
    test_client, _ = client(tmp_path)
    with test_client as api:
        assert api.get("/health").json() == {
            "status": "ok", "neural_available": True, "neural_loaded": True
        }
        ids = {entry["id"] for entry in api.get("/models").json()}
        assert {"auto", "dictionary", "translation_memory", "nllb"} <= ids


def test_translate_schema_and_router(tmp_path):
    test_client, _ = client(tmp_path)
    with test_client as api:
        result = api.post(
            "/translate", json={"text": "A B C", "source": "hre", "target": "vi", "model": "auto"}
        )
        assert result.status_code == 200
        body = result.json()
        assert body["translation"] == "câu ABC"
        assert body["model"] == "translation_memory"
        assert body["latency_ms"] >= 0
        assert isinstance(body["retrieved_terms"], list)
        assert body["retrieved_examples"][0]["source"] == "A B C"
        dictionary = api.post("/translate", json={"text": "A B", "model": "dictionary"})
        assert dictionary.json()["translation"] == "cụm AB"
        assert dictionary.json()["retrieved_terms"]


def test_batch_and_invalid_input(tmp_path):
    test_client, _ = client(tmp_path)
    with test_client as api:
        response = api.post(
            "/translate/batch",
            json={"texts": ["A B C", "Q R S"], "model": "translation_memory"},
        )
        assert response.status_code == 200
        assert [item["translation"] for item in response.json()["translations"]] == [
            "câu ABC", "câu QRS"
        ]
        assert api.post("/translate", json={"text": "   "}).status_code == 422
        assert api.post("/translate", json={"text": "A", "source": "vi"}).status_code == 422
        assert api.post("/translate", json={"text": "A", "model": "unknown"}).status_code == 422
        assert api.post("/translate/batch", json={"texts": []}).status_code == 422


def test_feedback_persists_to_sqlite(tmp_path):
    test_client, db = client(tmp_path)
    with test_client as api:
        payload = {
            "source": "A B", "prediction": "cụm AB", "correction": "sửa câu",
            "model": "dictionary", "rating": 4,
        }
        response = api.post("/feedback", json=payload)
        assert response.status_code == 201
        assert response.json()["id"] == 1
        assert response.json()["timestamp"]
        assert api.post("/feedback", json={**payload, "rating": 6}).status_code == 422
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT source, prediction, correction, model, rating FROM feedback"
        ).fetchone()
    assert row == ("A B", "cụm AB", "sửa câu", "dictionary", 4)


def test_missing_neural_returns_503_but_tm_still_works(tmp_path):
    lexicon = DictionaryTranslator([("A", "một")])
    memory = TranslationMemory().fit(pd.DataFrame({"hre": ["A B C"], "vi": ["câu ABC"]}))
    engine = ServingEngine(
        lexicon, memory, UnavailableNeural(),
        medium_threshold=0.5, high_threshold=0.85, top_k=1,
    )
    api_app = create_app(engine=engine, feedback_store=FeedbackStore(tmp_path / "feedback.db"))
    with TestClient(api_app) as api:
        assert api.get("/health").json()["neural_available"] is False
        assert api.post("/translate", json={"text": "A B C", "model": "auto"}).status_code == 200
        assert api.post("/translate", json={"text": "A B C", "model": "nllb"}).status_code == 503
