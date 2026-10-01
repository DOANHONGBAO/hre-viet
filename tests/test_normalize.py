import unicodedata

from hre_translate.data.normalize import normalize_hre, normalize_vietnamese


def test_normalization_preserves_case_and_diacritics() -> None:
    assert normalize_hre("  Hloâi   chrếq  ") == "Hloâi chrếq"
    assert normalize_vietnamese("  Tiếng   Việt  ") == "Tiếng Việt"


def test_normalization_uses_nfc_and_punctuation_spacing() -> None:
    decomposed = unicodedata.normalize("NFD", "tiếng")
    assert normalize_vietnamese(f" {decomposed}  ,  Việt ") == "tiếng, Việt"


def test_empty_values_are_removed() -> None:
    assert normalize_hre(None) is None
    assert normalize_hre(" \t\n ") is None
    assert normalize_vietnamese(float("nan")) is None
