import pandas as pd

from hre_translate.data.split import grouped_split


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hre": ["mòiq", "mòiq!", "baiq", "piq", "pun", "padam", "tơm", "cơm", "dak", "vok"],
            "vi": ["một", "một!", "hai", "ba", "bốn", "năm", "sáu", "bảy", "nước", "lợn"],
        }
    )


def test_split_is_reproducible_and_disjoint() -> None:
    first, _ = grouped_split(
        _sample(), ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=42, threshold=0.88
    )
    second, _ = grouped_split(
        _sample(), ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=42, threshold=0.88
    )
    for name in first:
        assert first[name][["hre", "vi"]].equals(second[name][["hre", "vi"]])
    pair_sets = {
        name: set(map(tuple, frame[["hre", "vi"]].to_numpy())) for name, frame in first.items()
    }
    assert pair_sets["train"].isdisjoint(pair_sets["validation"])
    assert pair_sets["train"].isdisjoint(pair_sets["test"])
    assert pair_sets["validation"].isdisjoint(pair_sets["test"])


def test_near_duplicates_stay_in_one_split() -> None:
    splits, _ = grouped_split(
        _sample(), ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=7, threshold=0.75
    )
    locations = {}
    for split, frame in splits.items():
        locations.update(dict.fromkeys(frame["hre"], split))
    assert locations["mòiq"] == locations["mòiq!"]
