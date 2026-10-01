import pandas as pd

from hre_translate.data.preprocess import prepare_pairs

SPEC = {
    "sheet": "sample",
    "id_column": "id",
    "hre_column": "source",
    "vi_column": "target",
    "notes_column": "notes",
    "header_row": 1,
}


def test_prepare_pairs_removes_empty_and_exact_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "source": ["A", " A ", None, "B"],
            "target": ["Một", "Một", "Hai", "  "],
            "notes": [None] * 4,
        }
    )
    result, stats = prepare_pairs(frame, SPEC, source_file="sample.xlsx", dataset_type="lexicon")
    assert result[["hre", "vi"]].to_dict("records") == [{"hre": "A", "vi": "Một"}]
    assert stats["exact_duplicates_removed"] == 1
    assert stats["empty_pairs_removed"] == 2
