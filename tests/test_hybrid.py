import pandas as pd
import pytest

from hre_translate.hybrid import HybridTranslator
from hre_translate.models.dictionary import DictionaryTranslator
from hre_translate.models.retrieval import TranslationMemory


class StubNeural:
    def __init__(self, translation="A B"):
        self.translation = translation
        self.calls = 0

    def translate(self, text):
        self.calls += 1
        return {"translation": self.translation}


def service(translation="A B", **kwargs):
    lexicon = DictionaryTranslator([("A", "một"), ("A B", "cụm dài")])
    memory = TranslationMemory().fit(
        pd.DataFrame(
            {
                "hre": ["A B C", "Q R S", "Alpha Beta Gamma"],
                "vi": ["bản ghi", "khác", "bản ghi dài"],
            }
        )
    )
    neural = StubNeural(translation)
    return HybridTranslator(lexicon, memory, neural, **kwargs), neural


def test_retrieval_longest_phrase_and_output_schema():
    hybrid, _ = service()
    result = hybrid.translate("A B")
    assert result["retrieved_terms"][0]["target"] == "cụm dài"
    assert result["retrieved_examples"]
    assert 0 <= result["similarity"] <= 1
    assert result["latency_ms"] >= 0
    assert set(result) == {
        "translation", "model", "retrieved_terms", "retrieved_examples", "similarity", "latency_ms"
    }


def test_high_similarity_routes_directly_to_tm():
    hybrid, neural = service(high_threshold=0.8)
    result = hybrid.translate("A B C")
    assert result["model"] == "translation_memory"
    assert result["translation"] == "bản ghi"
    assert neural.calls == 0


def test_medium_similarity_uses_retrieval_context():
    hybrid, _ = service(translation="Alpha Beta", medium_threshold=0.1, high_threshold=1.0)
    result = hybrid.translate("Alpha Beta")
    assert result["model"] == "retrieval_augmented_tm_fallback"


def test_dictionary_ablation_replaces_only_copied_longest_phrase():
    hybrid, _ = service(translation="A B nhé", variant="nllb_dictionary")
    result = hybrid.translate("A B")
    assert result["translation"] == "cụm dài nhé"
    assert result["model"] == "retrieval_augmented_dictionary"


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        service(medium_threshold=0.9, high_threshold=0.8)
