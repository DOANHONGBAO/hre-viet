"""Start the local FastAPI development server from the project root."""

from __future__ import annotations

import socket
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            'Install backend dependencies: python -m pip install -e ".[serve]"'
        ) from exc
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise SystemExit(
                "Port 8000 is already in use. Stop the existing API before npm run dev."
            )
    uvicorn.run(
        "hre_translate.serving.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(root / "src")],
    )


if __name__ == "__main__":
    main()
