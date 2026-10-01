from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_gb: float | None
    fp16_enabled: bool


def detect_runtime(request_fp16: bool = True) -> RuntimeInfo:
    import torch

    available = bool(torch.cuda.is_available())
    if not available:
        return RuntimeInfo("cpu", False, None, None, False)
    properties = torch.cuda.get_device_properties(0)
    return RuntimeInfo(
        device="cuda",
        cuda_available=True,
        gpu_name=torch.cuda.get_device_name(0),
        gpu_memory_gb=round(properties.total_memory / (1024**3), 2),
        fp16_enabled=bool(request_fp16),
    )


def runtime_dict(request_fp16: bool = True) -> dict[str, object]:
    return asdict(detect_runtime(request_fp16))
