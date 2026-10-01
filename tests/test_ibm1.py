import math

import pandas as pd

from hre_translate.models.statistical import IBMModel1


def test_ibm1_probabilities_are_normalized() -> None:
    frame = pd.DataFrame(
        {
            "hre": ["mòiq", "baiq", "mòiq baiq"],
            "vi": ["một", "hai", "một hai"],
        }
    )
    model = IBMModel1.from_frame(frame, iterations=8)
    assert model.probability_sums()
    assert all(
        math.isclose(total, 1.0, abs_tol=1e-9) for total in model.probability_sums().values()
    )
    result = model.translate("mòiq baiq lạ")
    assert result["translation"].startswith("một hai")
    assert result["unknown_terms"] == ["lạ"]
