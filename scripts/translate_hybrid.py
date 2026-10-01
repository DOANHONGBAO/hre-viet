"""Translate one H'rê input with live NLLB + retrieval (requires neural dependencies)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hre_translate.hybrid import HybridTranslator  # noqa: E402
from hre_translate.models.dictionary import DictionaryTranslator  # noqa: E402
from hre_translate.models.retrieval import TranslationMemory  # noqa: E402
from hre_translate.neural.config import load_nllb_config  # noqa: E402
from hre_translate.neural.inference import NLLBTranslator  # noqa: E402
from hre_translate.neural.runtime import detect_runtime  # noqa: E402
from hre_translate.utils.config import load_config, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    parser.add_argument("--nllb-config", default="configs/nllb.yaml")
    parser.add_argument("--adapter", default="artifacts/models/nllb/final_adapter")
    parser.add_argument(
        "--variant", choices=sorted(HybridTranslator.VARIANTS), default="full_hybrid"
    )
    args = parser.parse_args()
    options, root = load_config(args.config)
    nllb, _ = load_nllb_config(args.nllb_config)
    train = pd.read_parquet(resolve(root, options["data"]["train"]))
    lexicon = DictionaryTranslator.from_frame(
        train.loc[train.dataset_type.isin(options["lexicon_types"])]
    )
    retrieval = options["retrieval"]
    memory = TranslationMemory(
        ngram_range=tuple(retrieval["ngram_range"]),
        direct_match_threshold=float(retrieval["high_threshold"]),
    ).fit(train.loc[train.dataset_type.isin(options["memory_types"])])
    neural = NLLBTranslator.from_adapter(
        resolve(root, args.adapter),
        model_name=nllb.model.name,
        source_language=nllb.model.source_language_proxy,
        target_language=nllb.model.target_language,
        device=detect_runtime(nllb.training.fp16).device,
        beam_size=nllb.generation.beam_size,
        max_source_length=nllb.training.max_source_length,
        max_new_tokens=nllb.generation.max_new_tokens,
    )
    service = HybridTranslator(
        lexicon,
        memory,
        neural,
        medium_threshold=float(retrieval["medium_threshold"]),
        high_threshold=float(retrieval["high_threshold"]),
        top_k=int(retrieval["top_k"]),
        variant=args.variant,
    )
    print(json.dumps(service.translate(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
