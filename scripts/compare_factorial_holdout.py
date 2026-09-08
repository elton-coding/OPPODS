"""Same-input four-arm experiment; never add individually observed gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare_paired_holdout import compare, metrics


def factorial(groups: dict[str, list[Path]], repeats: int = 2000,
              factor_a: str = "factor_A", factor_b: str = "factor_B") -> dict:
    contrasts = {name: compare(groups["C"], groups[name], repeats=repeats) for name in ("A", "B", "AB")}
    ab_vs_a = compare(groups["A"], groups["AB"], repeats=repeats)
    arrays = {}
    for name, paths in groups.items():
        scores = []
        for path in paths:
            with np.load(path) as data:
                scores.append(data["score"].astype(np.float64).reshape(-1, 2))
        arrays[name] = np.stack(scores)
    means = {name: float(np.mean([metrics(row)["final"] for row in values]))
             for name, values in arrays.items()}
    interaction = means["AB"] - means["A"] - means["B"] + means["C"]
    rng = np.random.default_rng(229)
    draws = []
    for _ in range(repeats):
        index = rng.integers(0, arrays["C"].shape[1], arrays["C"].shape[1])
        boot_means = {name: np.mean([metrics(row[index])["final"] for row in values])
                      for name, values in arrays.items()}
        draws.append(boot_means["AB"] - boot_means["A"] - boot_means["B"] + boot_means["C"])
    return {
        "factor_labels": {"A": factor_a, "B": factor_b},
        "group_mean_final": means,
        "contrasts_vs_control": contrasts,
        "factor_B_effect_with_A_AB_minus_A": ab_vs_a,
        "interaction_AB_minus_A_minus_B_plus_C": interaction,
        "interaction_paired_95_interval": np.percentile(draws, [2.5, 97.5]).tolist(),
        "bootstrap_repeats": repeats,
        "caveat": "method-level factors; same-input audit comparison, not online or selection-adjusted inference",
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("C", "A", "B", "AB"):
        parser.add_argument(f"--{name.lower()}", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-a", default="factor_A")
    parser.add_argument("--factor-b", default="factor_B")
    args = parser.parse_args()
    result = factorial({name: getattr(args, name.lower()) for name in ("C", "A", "B", "AB")},
                       factor_a=args.factor_a, factor_b=args.factor_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in (
        "contrasts_vs_control", "factor_B_effect_with_A_AB_minus_A",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
