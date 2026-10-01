from __future__ import annotations

import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "kaggle" / "hre-translate-stage3-kaggle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    include = [
        root / "src",
        root / "configs" / "nllb.yaml",
        root / "configs" / "tracking.yaml",
        root / "data" / "splits",
        root / "scripts" / "train_nllb.py",
        root / "scripts" / "evaluate_nllb.py",
        root / "scripts" / "check_neural_setup.py",
        root / "artifacts" / "evaluation" / "classical_results.csv",
        root / "kaggle",
        root / "pyproject.toml",
        root / "README.md",
        root / "DECISIONS.md",
    ]
    files = []
    for item in include:
        if item.is_dir():
            files.extend(path for path in item.rglob("*") if path.is_file())
        elif item.exists():
            files.append(item)
    files = sorted(
        {
            path
            for path in files
            if "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and ".ruff_cache" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
            and path.suffix != ".pyc"
        }
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Created {output} with {len(files)} files ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
