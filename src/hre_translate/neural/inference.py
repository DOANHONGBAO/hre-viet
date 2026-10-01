from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class NLLBTranslator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        source_language: str,
        target_language: str,
        device: str,
        beam_size: int = 4,
        max_source_length: int = 128,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.device = device
        self.beam_size = beam_size
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer.src_lang = source_language
        self.tokenizer.tgt_lang = target_language
        self.target_token_id = self.tokenizer.convert_tokens_to_ids(target_language)
        if self.target_token_id is None or self.target_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Unsupported target language token: {target_language}")
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        model_name: str,
        source_language: str,
        target_language: str,
        device: str,
        beam_size: int,
        max_source_length: int,
        max_new_tokens: int,
    ) -> NLLBTranslator:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            src_lang=source_language,
            tgt_lang=target_language,
        )
        base = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
        model = PeftModel.from_pretrained(base, adapter_path)
        return cls(
            model,
            tokenizer,
            source_language=source_language,
            target_language=target_language,
            device=device,
            beam_size=beam_size,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
        )

    def translate(self, text: str) -> dict[str, Any]:
        import torch

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self.target_token_id,
                num_beams=self.beam_size,
                max_new_tokens=self.max_new_tokens,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        translation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return {"translation": translation.strip(), "latency_ms": latency_ms}
