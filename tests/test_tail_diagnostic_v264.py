import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_tail_bottleneck_v264 as d


def test_boundaries_match_right_bucket():
    np.testing.assert_array_equal(d.bins(np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 7])


def test_contributions_partition_and_partner_swap():
    snr = np.array([-20., 20., -12., -7., 1., 7., 12., 17.])
    reference = np.arange(8.) + 50
    scores = {name: reference.copy() for name in d.LABELS}
    scores["V258"] = reference + np.arange(8.) / 10
    result = d.analyze(snr, scores)
    for rows in result["slices"].values():
        nonempty = [v for v in rows.values() if v["rows"]]
        assert sum(v["rows"] for v in nonempty) == 8
        assert sum(v["reference_bottom10_rows"] for v in nonempty) == 1
        actual = sum(v["candidates"]["V258"]["global_efficiency_contribution"] for v in nonempty)
        assert actual == pytest.approx(.35)
    assert result["slices"]["own_partner_snr"]["0,7"]["rows"] == 1
    assert result["slices"]["own_partner_snr"]["7,0"]["rows"] == 1
    assert result["slices"]["pair_min_expert"]["0"]["rows"] == 2


def test_fixed_reference_tail_not_reselected_per_candidate():
    reference = np.array([50., 51., 52., 53.])
    candidate = reference[::-1]
    value = d.summarize(reference, {"test": candidate}, np.ones(4, dtype=bool),
                        np.array([True, False, False, False]), np.array([True, True, False, False]))
    assert value["candidates"]["test"]["mean_delta_on_fixed_reference_tail"] == 3.
    assert value["candidates"]["test"]["mean_delta_on_fixed_reference_near_p10"] == 2.


def test_completed_output_conserves_exact_metrics():
    result = d.json.loads(d.OUTPUT.read_text(encoding="utf-8"))
    assert len(result["per_seed"]) == 3
    for record in result["per_seed"]:
        for rows in record["slices"].values():
            selected = [v for v in rows.values() if v["rows"]]
            assert sum(v["rows"] for v in selected) == 4000
            assert sum(v["reference_bottom10_rows"] for v in selected) == record["reference_bottom10_rows_including_ties"]
            for name in d.LABELS:
                if name == "V257":
                    continue
                contribution = sum(v["candidates"][name]["global_efficiency_contribution"] for v in selected)
                delta = record["model_metrics"][name]["efficiency"] - record["model_metrics"]["V257"]["efficiency"]
                assert contribution == pytest.approx(delta, abs=1e-12)
