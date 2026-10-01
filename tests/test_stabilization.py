import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hre_translate.models.retrieval import TranslationMemory
from hre_translate.models.rule_based import RuleBasedTranslator
from hre_translate.models.statistical import StatisticalTranslator
from hre_translate.serving.app import create_app
from hre_translate.serving.engine import ServingEngine
from hre_translate.serving.schemas import TranslateRequest
from scripts.benchmark_models import _checked_replay


class FakeNeural:
    available = False

    def translate(self, text):
        raise AssertionError("Default route must not invoke NLLB")


def test_rule_based_longest_phrase_unknown_and_normalization():
    model = RuleBasedTranslator([("A B", "cụm AB"), ("A", "một"), ("B", "hai")])
    result = model.translate("  A   B   Z  ")
    assert result["translation"] == "cụm AB Z"
    assert result["matched_terms"][0]["source"] == "A B"
    assert result["unknown_terms"] == ["Z"]
    assert result["model"] == "rule_based"
    assert result["latency_ms"] >= 0


def test_statistical_save_load_and_probability_normalization(tmp_path):
    frame = pd.DataFrame({"hre": ["mòiq", "baiq", "mòiq baiq"], "vi": ["một", "hai", "một hai"]})
    model = StatisticalTranslator.from_frame(frame, iterations=8)
    before = model.translate("mòiq baiq lạ")
    model.save(tmp_path / "model")
    loaded = StatisticalTranslator.load(tmp_path / "model")
    assert loaded.translate("mòiq baiq lạ")["translation"] == before["translation"]
    assert before["unknown_terms"] == ["lạ"]
    assert all(
        math.isclose(total, 1.0, abs_tol=1e-9)
        for total in loaded.lexical.probability_sums().values()
    )


def test_default_router_uses_selected_model_without_neural():
    frame = pd.DataFrame({"hre": ["mòiq", "baiq"], "vi": ["một", "hai"]})
    statistical = StatisticalTranslator.from_frame(frame, iterations=3)
    engine = ServingEngine(
        RuleBasedTranslator.from_frame(frame).dictionary,
        TranslationMemory().fit(frame),
        FakeNeural(),
        medium_threshold=0.5,
        high_threshold=0.85,
        top_k=1,
        rule_based=RuleBasedTranslator.from_frame(frame),
        ibm1=statistical.lexical,
        statistical=statistical,
        default_model="statistical",
    )
    assert engine.translate(TranslateRequest(text="mòiq", model="auto")).model == "statistical"
    assert engine.translate(TranslateRequest(text="mòiq", model="default")).model == "statistical"
    assert engine.translate(TranslateRequest(text="mòiq", model="rule_based")).model == "rule_based"
    assert {entry["id"] for entry in engine.models()} >= {"default", "statistical", "rule_based"}


def test_project_api_selected_default_and_local_cors():
    with TestClient(create_app()) as api:
        models = {entry["id"]: entry for entry in api.get("/models").json()}
        assert models["statistical"]["available"] is True
        response = api.post("/translate", json={"text": "Lăm kleq kô pa zâu", "model": "default"})
        assert response.status_code == 200
        assert response.json()["model"] == "statistical"
        assert response.json()["translation"]
        preflight = api.options(
            "/translate",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_benchmark_rejects_mismatched_replay(tmp_path):
    test = pd.DataFrame({"hre": ["hre one"], "vi": ["vi one"], "dataset_type": ["sentence"]})
    path = tmp_path / "replay.csv"
    pd.DataFrame(
        {"source": ["hre two"], "reference": ["vi one"], "hypothesis": ["vi two"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="fixed test source"):
        _checked_replay(path, test)
