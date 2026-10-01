import pandas as pd

from hre_translate.models.dictionary import DictionaryTranslator


def test_exact_lookup_and_case_aware_fallback() -> None:
    model = DictionaryTranslator.from_frame(
        pd.DataFrame({"hre": ["Mòiq", "baiq"], "vi": ["một", "hai"]})
    )
    exact = model.translate("Mòiq")
    folded = model.translate("MÒIQ")
    assert exact["translation"] == "một"
    assert exact["matched_terms"][0]["match_type"] == "exact"
    assert folded["translation"] == "một"
    assert folded["matched_terms"][0]["match_type"] == "exact_casefold"


def test_longest_phrase_wins_and_unknown_is_reported() -> None:
    model = DictionaryTranslator([("A", "một"), ("B", "hai"), ("A B", "cụm"), ("C", "ba")])
    result = model.translate("A B không-biết C")
    assert result["translation"] == "cụm không-biết ba"
    assert result["matched_terms"][0]["source"] == "A B"
    assert result["matched_terms"][0]["match_type"] == "longest_phrase"
    assert result["unknown_terms"] == ["không-biết"]
