import pandas as pd

from hre_translate.models.retrieval import TranslationMemory


def _memory() -> TranslationMemory:
    return TranslationMemory(direct_match_threshold=0.8).fit(
        pd.DataFrame(
            {
                "hre": ["Lăm kleq kô", "Mòiq baiq piq", "Dak kơlei"],
                "vi": ["Đi đâu đó", "Một hai ba", "Nước sông"],
            }
        )
    )


def test_tfidf_retrieves_closest_sentence() -> None:
    result = _memory().retrieve("Lăm kleq kô!", top_k=2)
    assert result[0]["source"] == "Lăm kleq kô"
    assert result[0]["target"] == "Đi đâu đó"
    assert result[0]["similarity"] > result[1]["similarity"]


def test_retrieval_is_deterministic() -> None:
    model = _memory()
    assert model.retrieve("Mòiq baiq", top_k=3) == model.retrieve("Mòiq baiq", top_k=3)
