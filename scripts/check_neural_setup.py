from __future__ import annotations

import json
from pathlib import Path

from hre_translate.neural.config import load_nllb_config
from hre_translate.neural.data import load_parallel_frame
from hre_translate.neural.runtime import runtime_dict


def main() -> None:
    config, root = load_nllb_config()
    split_rows = {}
    for name in ("train", "validation", "test"):
        value = getattr(config.data, name)
        path = Path(value)
        path = path if path.is_absolute() else root / path
        split_rows[name] = len(load_parallel_frame(path))
    print(
        json.dumps(
            {
                "runtime": runtime_dict(config.training.fp16),
                "model": config.model.name,
                "source_language_proxy": config.model.source_language_proxy,
                "target_language": config.model.target_language,
                "split_rows": split_rows,
                "training_output": config.training.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
