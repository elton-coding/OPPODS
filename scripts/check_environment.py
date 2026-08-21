from __future__ import annotations

import json
import platform
import sys

import numpy as np
import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1176)
    real = torch.randn((32, 2, 16), generator=generator, device=device, requires_grad=True)
    imag = torch.randn((32, 2, 16), generator=generator, device=device, requires_grad=True)
    channel = torch.complex(real, imag)
    gram = channel @ channel.conj().transpose(-2, -1)
    identity = torch.eye(2, device=device, dtype=torch.complex64)
    solved = torch.linalg.solve(gram + 0.1 * identity, identity.expand_as(gram))
    eigenvalues = torch.linalg.eigvalsh(gram)
    loss = solved.abs().square().mean() + eigenvalues.mean()
    loss.backward()

    if not torch.isfinite(loss):
        raise RuntimeError("non-finite complex linear algebra result")
    if real.grad is None or not torch.isfinite(real.grad).all():
        raise RuntimeError("complex linear algebra backward test failed")

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "complex_linalg_loss": float(loss.detach().cpu()),
        "complex_linalg_grad_norm": float(real.grad.norm().detach().cpu()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
