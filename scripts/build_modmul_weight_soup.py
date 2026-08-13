"""Rebuild the audited alpha=0.75 modmul weight-soup submission.

The two source checkpoints are immutable Git artifacts.  Mixing deliberately
uses float32 arithmetic because that is the artifact described by the original
experiment report; float64 accumulation produces a slightly different model.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATH = Path("experiments/modmul/submission_sota/weights.pt")
OUTPUT_PATH = REPO_ROOT / WEIGHT_PATH
PROVENANCE_PATH = OUTPUT_PATH.with_name("provenance.json")

HARVEST_REF = "c00027c6db90076e58bac25ce6c4c23a46ccfd40"
HARVEST_FILE_SHA256 = (
    "255b2328134b85ac624ea1d45bc2245d04c3d2c1062ad5c7080c037a48f6584d"
)
CHAMPION_REF = "4a6cbbead597cdbafd618b28666a730313857dd1"
CHAMPION_FILE_SHA256 = (
    "8fb5e4cf67d62b14effd85afb81fd96ded64d9da5ed02974171e21fe8707c010"
)
HARVEST_WEIGHT = 0.75
CHAMPION_WEIGHT = 0.25
EXPECTED_TENSOR_SHA256 = (
    "5eb2e582891f59690cf719d8c44e040b6cb33e21356d3b62ff40c26c2ef79961"
)
EXPECTED_FILE_SHA256 = (
    "2a245597f4499f83d0097d87801bd6dce79f014e5d1e3ed9cfa5f7a5e5c2363c"
)


def _git_blob(ref: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{WEIGHT_PATH.as_posix()}"],
        cwd=REPO_ROOT,
    )


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(data: bytes) -> dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    harvest_blob = _git_blob(HARVEST_REF)
    champion_blob = _git_blob(CHAMPION_REF)
    if _file_sha256(harvest_blob) != HARVEST_FILE_SHA256:
        raise ValueError("harvest checkpoint does not match its pinned hash")
    if _file_sha256(champion_blob) != CHAMPION_FILE_SHA256:
        raise ValueError("champion checkpoint does not match its pinned hash")

    harvest = _load(harvest_blob)
    champion = _load(champion_blob)
    if set(harvest) != set(champion):
        raise ValueError("parent checkpoints disagree on tensor names")

    soup: dict[str, torch.Tensor] = {}
    for name, harvest_tensor in harvest.items():
        champion_tensor = champion[name]
        if harvest_tensor.shape != champion_tensor.shape:
            raise ValueError(f"parent checkpoints disagree on shape of {name}")
        if harvest_tensor.dtype != champion_tensor.dtype:
            raise ValueError(f"parent checkpoints disagree on dtype of {name}")
        if not harvest_tensor.dtype.is_floating_point:
            raise ValueError(f"unexpected non-floating tensor {name}")
        # Preserve the exact FP32 operation used by the reported experiment.
        soup[name] = (
            HARVEST_WEIGHT * harvest_tensor
            + CHAMPION_WEIGHT * champion_tensor
        )

    tensor_sha256 = _tensor_sha256(soup)
    if tensor_sha256 != EXPECTED_TENSOR_SHA256:
        raise ValueError(
            f"unexpected soup tensor fingerprint: {tensor_sha256}"
        )

    # PyTorch's ZIP serialization is version-dependent.  Build beside the
    # certified artifact and replace it only if the bytes match the file that
    # completed the remote GPU evaluation.  A serializer mismatch must never
    # silently invalidate the artifact-to-evidence chain.
    with tempfile.TemporaryDirectory(dir=OUTPUT_PATH.parent) as tmp_dir:
        candidate_path = Path(tmp_dir) / OUTPUT_PATH.name
        torch.save(soup, candidate_path)
        output_file_sha256 = _file_sha256(candidate_path.read_bytes())
        if output_file_sha256 != EXPECTED_FILE_SHA256:
            raise RuntimeError(
                "tensor values match, but this PyTorch serializer produced "
                f"file SHA-256 {output_file_sha256}; expected "
                f"{EXPECTED_FILE_SHA256}. The certified weights.pt was left "
                "unchanged. Rebuild with the certification environment."
            )
        candidate_path.replace(OUTPUT_PATH)
    provenance = {
        "schema_version": 1,
        "artifact": "SAIR modmul alpha=0.75 weight soup",
        "method": "elementwise FP32 linear interpolation",
        "parents": [
            {
                "role": "harvest",
                "git_ref": HARVEST_REF,
                "file_sha256": HARVEST_FILE_SHA256,
                "weight": HARVEST_WEIGHT,
            },
            {
                "role": "r15_champion",
                "git_ref": CHAMPION_REF,
                "candidate_id": "17b8eb341153",
                "file_sha256": CHAMPION_FILE_SHA256,
                "weight": CHAMPION_WEIGHT,
            },
        ],
        "output": {
            "file": WEIGHT_PATH.name,
            "file_sha256": output_file_sha256,
            "tensor_sha256": tensor_sha256,
            "tensors": len(soup),
            "parameters": sum(t.numel() for t in soup.values()),
        },
        "certification": {
            "static_check": (
                "clean with the official AST checker on 2026-08-11"
            ),
            "manifest_and_load_smoke": "passed locally on 2026-08-11",
            "public_official_evaluation": (
                "H90=10, overall=1.0000, 1000/1000 scored cases correct, "
                "deterministic, 248.4 inference seconds on an NVIDIA L40S"
            ),
            "independent_seed_evaluations": (
                "5/5 passed at H90=10 and overall=1.0000 on an NVIDIA L40S; "
                "5000/5000 scored cases correct"
            ),
            "metamorphic_evaluation": "140/140 passed across tiers 1-10",
            "weight_randomization": (
                "17 tensors randomized; 0/30 nonzero-answer probes survived"
            ),
            "width_boundary": (
                "2048-bit modulus passed; the declared out-of-scope 2049-bit "
                "modulus is rejected with output zero"
            ),
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance["output"], sort_keys=True))


if __name__ == "__main__":
    main()
