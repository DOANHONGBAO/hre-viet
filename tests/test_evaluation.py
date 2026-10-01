import pandas as pd

from hre_translate.evaluation import evaluate_model


class EchoReferenceModel:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def translate(self, text: str) -> dict[str, object]:
        return {
            "translation": self.mapping[text],
            "matched_terms": [],
            "unknown_terms": [],
        }


def test_evaluation_pipeline_scores_real_predictions() -> None:
    test = pd.DataFrame(
        {
            "hre": ["a", "b"],
            "vi": ["một câu", "hai câu"],
            "dataset_type": ["sentence", "sentence"],
        }
    )
    metrics, predictions = evaluate_model(
        EchoReferenceModel({"a": "một câu", "b": "hai câu"}), test
    )
    assert metrics["bleu"] == 100.0
    assert metrics["chrf_pp"] == 100.0
    assert metrics["examples"] == 2
    assert len(predictions) == 2
