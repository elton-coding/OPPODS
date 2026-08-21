from __future__ import annotations

import struct
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NpyMemberInfo:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool
    data_offset: int


def _stored_npy_member_info(npz_path: str | Path, member: str) -> NpyMemberInfo:
    """Locate an uncompressed .npy member inside an NPZ without extracting it."""
    path = Path(npz_path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member!r} is compressed; direct memmap requires ZIP_STORED")

        with path.open("rb") as handle:
            handle.seek(info.header_offset)
            local_header = handle.read(30)
            if len(local_header) != 30:
                raise ValueError(f"truncated local ZIP header for {member!r}")
            fields = struct.unpack("<IHHHHHIIIHH", local_header)
            signature, name_length, extra_length = fields[0], fields[-2], fields[-1]
            if signature != 0x04034B50:
                raise ValueError(f"invalid local ZIP signature for {member!r}")
            npy_offset = info.header_offset + 30 + name_length + extra_length
            handle.seek(npy_offset)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported NPY format version {version}")
            return NpyMemberInfo(tuple(shape), np.dtype(dtype), fortran_order, handle.tell())


def open_stored_npy_memmap(npz_path: str | Path, member: str) -> np.memmap:
    info = _stored_npy_member_info(npz_path, member)
    return np.memmap(
        npz_path,
        dtype=info.dtype,
        mode="r",
        offset=info.data_offset,
        shape=info.shape,
        order="F" if info.fortran_order else "C",
    )


class ChannelMemmap:
    """Lazy, process-safe reader for the official real/imag NPZ members."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._real: np.memmap | None = None
        self._imag: np.memmap | None = None
        real_info = _stored_npy_member_info(self.path, "real.npy")
        imag_info = _stored_npy_member_info(self.path, "imag.npy")
        if real_info.shape != imag_info.shape:
            raise ValueError("real.npy and imag.npy shapes differ")
        if real_info.dtype != imag_info.dtype:
            raise ValueError("real.npy and imag.npy dtypes differ")
        self.shape = real_info.shape
        self.storage_dtype = real_info.dtype

    def _ensure_open(self) -> None:
        if self._real is None:
            self._real = open_stored_npy_memmap(self.path, "real.npy")
            self._imag = open_stored_npy_memmap(self.path, "imag.npy")

    def __len__(self) -> int:
        return self.shape[0]

    def read(self, indices: int | slice | Sequence[int] | np.ndarray) -> np.ndarray:
        self._ensure_open()
        assert self._real is not None and self._imag is not None
        real = np.asarray(self._real[indices], dtype=np.float32)
        imag = np.asarray(self._imag[indices], dtype=np.float32)
        return real + 1j * imag

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_real"] = None
        state["_imag"] = None
        return state


def deterministic_split_indices(
    num_samples: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 1176,
) -> dict[str, np.ndarray]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive and sum to less than one")
    indices = np.random.default_rng(seed).permutation(num_samples)
    train_end = round(num_samples * train_fraction)
    validation_end = train_end + round(num_samples * validation_fraction)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }
