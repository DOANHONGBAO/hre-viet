import pandas as pd

from hre_translate.data.validate import severe_cross_split_leakage, validate_frame


def test_schema_and_content_validation() -> None:
    assert validate_frame(pd.DataFrame({"hre": ["a"], "vi": ["b"]})) == []
    assert "missing required columns" in validate_frame(pd.DataFrame({"hre": ["a"]}))[0]
    errors = validate_frame(pd.DataFrame({"hre": ["", "a"], "vi": ["b", None]}))
    assert any("empty" in error for error in errors)
    assert any("NaN" in error for error in errors)


def test_near_duplicate_guard_detects_cross_split_leakage() -> None:
    splits = {
        "train": pd.DataFrame({"hre": ["Lăm kleq kô pa zâu"], "vi": ["Đang đi đâu đây"]}),
        "validation": pd.DataFrame({"hre": ["Lăm kleq kô pa zâu!"], "vi": ["Đang đi đâu đây!"]}),
        "test": pd.DataFrame({"hre": ["mòiq"], "vi": ["một"]}),
    }
    assert severe_cross_split_leakage(splits, threshold=0.90)
