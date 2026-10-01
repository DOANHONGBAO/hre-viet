import torch

from hre_translate.neural.inference import NLLBTranslator


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self) -> None:
        self.src_lang = None
        self.tgt_lang = None

    def convert_tokens_to_ids(self, token):
        return 42 if token == "vie_Latn" else 0

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def batch_decode(self, generated, skip_special_tokens=True):
        return ["bản dịch"]


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[42, 9]])


def test_inference_wrapper_forces_vietnamese_target_token() -> None:
    model = FakeModel()
    translator = NLLBTranslator(
        model,
        FakeTokenizer(),
        source_language="vie_Latn",
        target_language="vie_Latn",
        device="cpu",
        beam_size=3,
        max_source_length=64,
        max_new_tokens=20,
    )
    result = translator.translate("mòiq")
    assert result["translation"] == "bản dịch"
    assert result["latency_ms"] >= 0
    assert model.generate_kwargs["forced_bos_token_id"] == 42
    assert model.generate_kwargs["num_beams"] == 3
    assert model.generate_kwargs["max_new_tokens"] == 20
