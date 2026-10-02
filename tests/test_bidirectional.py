import math

import pandas as pd
from fastapi.testclient import TestClient

from hre_translate.models.dictionary import DictionaryTranslator
from hre_translate.models.retrieval import TranslationMemory
from hre_translate.models.statistical import StatisticalTranslator
from hre_translate.serving.app import create_app
from hre_translate.serving.engine import ServingEngine
from hre_translate.serving.feedback import FeedbackStore


def sample_models():
    train = pd.DataFrame({
        "hre": ["mòiq", "baiq", "mòiq baiq"],
        "vi": ["một", "hai", "một hai"],
        "dataset_type": ["sentence"] * 3,
    })
    forward = StatisticalTranslator.from_frame(train, iterations=8)
    reverse = StatisticalTranslator.from_frame(
        train, iterations=8, source_lang="vi", target_lang="hre"
    )
    return train, forward, reverse


def test_independent_directional_models_and_artifacts(tmp_path):
    _, forward, reverse = sample_models()
    assert forward.translate("mòiq")["translation"] == "một"
    assert reverse.translate("một")["translation"] == "mòiq"
    assert reverse.lexical.probabilities != forward.lexical.probabilities
    assert "mòiq" in reverse.language_model.vocabulary
    assert "một" not in reverse.language_model.vocabulary
    for name, model in (("hre_to_vi", forward), ("vi_to_hre", reverse)):
        model.save(tmp_path / name)
        loaded = StatisticalTranslator.load(tmp_path / name)
        assert (loaded.source_lang, loaded.target_lang) == tuple(name.split("_to_"))
        assert all(math.isclose(total, 1.0, abs_tol=1e-9)
                   for total in loaded.lexical.probability_sums().values())


def test_bidirectional_api_and_invalid_pairs(tmp_path):
    train, forward, reverse = sample_models()
    engine = ServingEngine(
        DictionaryTranslator.from_frame(train), TranslationMemory().fit(train), None,
        medium_threshold=0.5, high_threshold=0.85, top_k=1,
        statistical=forward, ibm1=forward.lexical, reverse_statistical=reverse,
        default_model="statistical",
    )
    app = create_app(engine=engine, feedback_store=FeedbackStore(tmp_path / "feedback.db"))
    with TestClient(app) as api:
        for source, target, text, expected in (
            ("hre", "vi", "mòiq", "một"),
            ("vi", "hre", "một", "mòiq"),
        ):
            response = api.post("/translate", json={"text": text, "source": source,
                                                    "target": target})
            assert response.status_code == 200
            assert response.json()["translation"] == expected
            assert response.json()["model"] == "statistical"
            assert response.json()["source"] == source
            assert response.json()["target"] == target
        assert api.post("/translate", json={"text": "một", "source": "vi",
                                            "target": "vi"}).status_code == 422
        assert api.post("/translate", json={"text": "một", "source": "vi",
                                            "target": "hre", "model": "nllb"}).status_code == 422
        batch = api.post("/translate/batch", json={"texts": ["một", "hai"],
                                                    "source": "vi", "target": "hre"})
        assert batch.status_code == 200
        assert [item["translation"] for item in batch.json()["translations"]] == [
            "mòiq", "baiq"
        ]
