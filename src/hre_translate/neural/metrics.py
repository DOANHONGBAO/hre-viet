from __future__ import annotations

from typing import Any

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


def corpus_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be non-empty and have equal length")
    return {
        "bleu": round(
            float(BLEU(effective_order=True).corpus_score(hypotheses, [references]).score), 6
        ),
        "chrf_pp": round(float(CHRF(word_order=2).corpus_score(hypotheses, [references]).score), 6),
    }


def trainer_compute_metrics(tokenizer: Any):
    def compute(eval_prediction: Any) -> dict[str, float]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(
            eval_prediction.label_ids != -100, eval_prediction.label_ids, tokenizer.pad_token_id
        )
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return corpus_metrics(
            [prediction.strip() for prediction in decoded_predictions],
            [label.strip() for label in decoded_labels],
        )

    return compute
