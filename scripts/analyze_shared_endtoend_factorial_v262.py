"""Four-arm paired full-payload analysis; no inference or checkpoint selection."""
from __future__ import annotations

import numpy as np

from oppods.data import deterministic_split_indices

ARMS = ("C", "A", "B", "AB")
SEEDS = (22701, 22702, 22703)
BOOTSTRAP_SEED = 262


def load_groups(paths):
    if set(paths) != set(ARMS) or any(len(paths[arm]) != 3 for arm in ARMS):
        raise ValueError("need exactly C/A/B/AB with three ordered noise replicas")
    arrays, reference, channel_ids = {}, {}, None
    expected_ids = deterministic_split_indices(100000, seed=1176)["test"][2000:4000].repeat(2)
    for arm in ARMS:
        scores = []
        for seed, path in zip(SEEDS, paths[arm], strict=True):
            with np.load(path, allow_pickle=False) as cache:
                for name, expected in (("split_seed", 1176), ("test_offset", 2000), ("noise_seed", seed)):
                    if cache[name].size != 1 or cache[name].dtype.kind not in "iu" or int(cache[name].item()) != expected:
                        raise ValueError("wrong fixed evaluation metadata")
                values = {name: cache[name].copy() for name in ("score", "length", "data_index", "snr")}
            if any(values[name].shape != (4000,) for name in values):
                raise ValueError("need 2000 paired channels / 4000 UE rows")
            if (not np.all(values["length"] == 1152) or not np.isfinite(values["score"]).all()
                    or np.any(values["score"] < 0) or np.any(values["score"] > 100)
                    or not np.isfinite(values["snr"]).all()
                    or np.any(values["snr"] < -20) or np.any(values["snr"] > 20)):
                raise ValueError("invalid score or incomplete 1152-bit payload")
            ids = values["data_index"]
            if not np.array_equal(ids, expected_ids):
                raise ValueError("channel IDs differ from actual split1176 test offset2000 window")
            if (ids.dtype.kind not in "iu" or not np.array_equal(ids[::2], ids[1::2])
                    or np.unique(ids[::2]).size != 2000):
                raise ValueError("UE rows must be paired by distinct channel")
            if channel_ids is None:
                channel_ids = ids.copy()
            elif not np.array_equal(channel_ids, ids):
                raise ValueError("all arms and noise replicas need the same ordered channels")
            if arm == "C":
                reference[seed] = values
            elif any(not np.array_equal(reference[seed][name], values[name]) for name in ("data_index", "snr", "length")):
                raise ValueError("arms have unpaired channel/SNR/payload inputs")
            scores.append(values["score"].astype(np.float64).reshape(2000, 2))
        arrays[arm] = np.stack(scores)
    return arrays


def arm_metrics(values):
    flattened = values.reshape(values.shape[0], values.shape[1], -1)
    efficiency = flattened.mean(axis=-1)
    p10 = np.percentile(flattened, 10, axis=-1)
    return efficiency, p10, .7 * efficiency + .3 * p10


def analyze_groups(groups, repeats=2000):
    if set(groups) != set(ARMS) or repeats < 2:
        raise ValueError("need four arms and at least two bootstrap draws")
    values = np.stack([groups[arm] for arm in ARMS]).astype(np.float64)
    if (values.ndim != 4 or values.shape[1] != 3 or values.shape[2] < 1 or values.shape[3] != 2
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 100):
        raise ValueError("expected finite [arm, three noises, channel, two UEs] scores in [0,100]")
    efficiency, p10, final = arm_metrics(values)
    pairs = ((3, 0), (3, 1), (3, 2))
    bootstrap = np.empty((repeats, 4), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for repeat in range(repeats):
        indices = rng.integers(0, values.shape[2], values.shape[2])
        sampled_final = arm_metrics(values[:, :, indices, :])[2].mean(axis=1)
        bootstrap[repeat, :3] = [sampled_final[a] - sampled_final[b] for a, b in pairs]
        bootstrap[repeat, 3] = sampled_final[3] - sampled_final[1] - sampled_final[2] + sampled_final[0]
    comparisons = {}
    for column, (candidate, control) in enumerate(pairs):
        deltas = final[candidate] - final[control]
        interval = np.percentile(bootstrap[:, column], [2.5, 97.5]).tolist()
        comparisons[f"{ARMS[candidate]}-{ARMS[control]}"] = {
            "mean_delta": float(deltas.mean()), "per_seed_final_deltas": deltas.tolist(),
            "per_seed_efficiency_deltas": (efficiency[candidate]-efficiency[control]).tolist(),
            "per_seed_p10_deltas": (p10[candidate]-p10[control]).tolist(),
            "paired_delta_95_interval": interval,
            "passes_preregistered_gate": bool(np.all(deltas > 0) and interval[0] > 0)}
    interaction = final[3] - final[1] - final[2] + final[0]
    return {"arm_mean_final_from_cache": {arm: float(final[i].mean()) for i, arm in enumerate(ARMS)},
            "arm_per_seed_metrics": {arm: {"efficiency": efficiency[i].tolist(), "p10": p10[i].tolist(),
                                            "final": final[i].tolist()} for i, arm in enumerate(ARMS)},
            "comparisons": comparisons,
            "interaction": {"definition": "(AB-A)-(B-C)", "mean": float(interaction.mean()),
                            "per_seed": interaction.tolist(),
                            "paired_95_interval": np.percentile(bootstrap[:, 3], [2.5, 97.5]).tolist()},
            "bootstrap_repeats": repeats, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "channel, both UEs together; same indices across all arms and noises",
            "eligible_for_new_confirmation": all(v["passes_preregistered_gate"] for v in comparisons.values()),
            "automatic_promotion": False, "online_confirmation": False,
            "caveat": "finite-channel selection uncertainty only; no correction for repeated model selection"}
