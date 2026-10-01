import pandas as pd

from hre_translate.neural.data import parallel_records, tokenize_batch


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.src_lang = None
        self.tgt_lang = None

    def __call__(self, text=None, **kwargs):
        self.calls.append((text, kwargs))
        values = kwargs.get("text_target", text)
        return {"input_ids": [[index + 1] for index, _ in enumerate(values)]}


def test_dataset_formatting_uses_parallel_columns() -> None:
    records = parallel_records(pd.DataFrame({"hre": ["mòiq"], "vi": ["một"]}))
    assert records == [{"source": "mòiq", "target": "một"}]


def test_tokenizer_pipeline_sets_languages_and_separate_limits() -> None:
    tokenizer = RecordingTokenizer()
    result = tokenize_batch(
        {"source": ["mòiq"], "target": ["một"]},
        tokenizer,
        source_language="vie_Latn",
        target_language="vie_Latn",
        max_source_length=64,
        max_target_length=32,
    )
    assert tokenizer.src_lang == "vie_Latn"
    assert tokenizer.tgt_lang == "vie_Latn"
    assert tokenizer.calls[0][1]["max_length"] == 64
    assert tokenizer.calls[1][1]["max_length"] == 32
    assert tokenizer.calls[1][1]["text_target"] == ["một"]
    assert result["labels"] == [[1]]
