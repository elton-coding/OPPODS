"""Inspect saved channel IDs without evaluating a model or reading historical scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from oppods.data import deterministic_split_indices


def index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def inspect_archive(path: Path, total: int) -> dict:
    before = path.stat()
    with np.load(path, allow_pickle=False) as archive:
        if "data_index" not in archive.files:
            return {"path": str(path), "status": "no_recorded_channel_ids"}
        raw = archive["data_index"]
        if raw.dtype.kind not in "iu" or raw.ndim != 1:
            raise ValueError(f"invalid channel-index representation: {path}")
        ids = np.unique(raw.astype(np.int64))
        if ids.size and (ids.min() < 0 or ids.max() >= total):
            raise ValueError(f"channel IDs outside declared dataset: {path}")
        metadata = {}
        for name in ("split_seed", "noise_seed", "test_offset"):
            if name in archive.files:
                value = archive[name]
                if value.size != 1:
                    raise ValueError(f"non-scalar {name}: {path}")
                metadata[name] = int(value.item())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"archive changed during inventory: {path}")
    return {"path": str(path), "status": "recorded_ids", "unique_channel_count": int(ids.size),
            "index_sha256": index_digest(ids), "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "metadata": metadata, "ids": ids}


def summarize_windows(total: int, records: list[dict], offsets: list[int], samples: int) -> dict:
    split = deterministic_split_indices(total, seed=1176)
    all_seen = np.zeros(total, dtype=bool)
    explicit_fixed_seen = np.zeros(total, dtype=bool)
    for record in records:
        if record["status"] != "recorded_ids":
            continue
        all_seen[record["ids"]] = True
        if record["metadata"].get("split_seed") == 1176:
            explicit_fixed_seen[record["ids"]] = True
    results = []
    for offset in offsets:
        if offset < 0 or samples <= 0 or offset + samples > len(split["test"]):
            raise ValueError("invalid proposed confirmation window")
        ids = split["test"][offset:offset + samples]
        results.append({
            "test_offset": offset, "samples": samples, "ordered_index_sha256": index_digest(ids),
            "train1176_overlap": int(np.intersect1d(ids, split["train"]).size),
            "validation1176_overlap": int(np.intersect1d(ids, split["validation"]).size),
            "current_offset2000_audit_overlap": int(np.intersect1d(ids, split["test"][2000:4000]).size),
            "all_recorded_evaluation_ids_overlap": int(all_seen[ids].sum()),
            "explicit_split1176_recorded_ids_overlap": int(explicit_fixed_seen[ids].sum()),
        })
    unused = split["test"][~all_seen[split["test"]]]
    return {
        "windows": results,
        "recorded_evaluation_channel_union": int(all_seen.sum()),
        "test1176_channels_absent_from_all_recorded_ids": int(unused.size),
        "unrecorded_test_ids_sha256": index_digest(unused),
        "unresolved_archives_without_ids": sum(r["status"] == "no_recorded_channel_ids" for r in records),
        "unreadable_archives": sum(r["status"] == "unreadable" for r in records),
        "blindness_certified": False,
        "caveat": "Absence from saved IDs is not proof of never evaluated. Archives without IDs, missing logs, "
                  "historical split changes and parent training lineage need separate checks. "
                  "Explicit split1176 is metadata scope, not a claim about model family or experiment date.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[Path("benchmarks"), Path("artifacts")])
    parser.add_argument("--total-channels", type=int, default=100000)
    parser.add_argument("--offsets", nargs="+", type=int, default=[4000, 6000, 8000])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("inventory already exists; use a new immutable snapshot path")
    if any(not root.is_dir() for root in args.roots):
        raise ValueError("all inventory roots must exist")
    paths = sorted({p.resolve() for root in args.roots for p in root.rglob("*.npz")})
    records = []
    for path in paths:
        try:
            record = inspect_archive(path, args.total_channels)
        except (OSError, ValueError, RuntimeError) as error:
            record = {"path": str(path), "status": "unreadable", "error": str(error)}
        records.append(record)
    summary = summarize_windows(args.total_channels, records, args.offsets, args.samples)
    result = {"purpose": "score-blind channel-ID inventory; no model inference and no score values read",
              "dataset": "H_train.npz; total supplied explicitly, not loaded for this inventory",
              "total_channels": args.total_channels, "roots": [str(root.resolve()) for root in args.roots],
              "files_scanned": len(paths), **summary,
              "archives": [{key: value for key, value in record.items() if key != "ids"} for record in records]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"output": str(args.output), "files_scanned": len(paths), **summary}, indent=2))


if __name__ == "__main__":
    main()
