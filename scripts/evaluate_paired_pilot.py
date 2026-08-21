from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oppods.data import ChannelMemmap, deterministic_split_indices
from oppods.denoised_link import DenoisedSparseMUMIMOLink
from oppods.metrics import per_sample_score, summarize_scores
from oppods.paired_pilot_link import PairedPilotMUMIMOLink
from oppods.reserved_pilot_link import ReservedPilotMUMIMOLink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one-Walsh-pilot and blind profiles")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sparse_denoiser_task.pt"))
    parser.add_argument("--data", type=Path, default=Path("ziliao/data_train/H_train.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1176)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pilot-amplitudes", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--separators", nargs="+", default=["pair"])
    parser.add_argument("--steering-shrinkages", type=float, nargs="+", default=[0.0])
    parser.add_argument("--loading-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--central-boosts", type=float, nargs="+", default=[-0.5])
    parser.add_argument("--gain-refinement-iterations", type=int, nargs="+", default=[0])
    parser.add_argument("--gain-refinement-min-snrs", type=float, nargs="+", default=[0.0])
    parser.add_argument("--gain-refinement-rates", type=float, nargs="+", default=[0.5])
    parser.add_argument("--gain-refinement-soft-temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--sample-covariance-blends", type=float, nargs="+", default=[0.0])
    parser.add_argument("--regularization-scales", type=float, nargs="+", default=[1.5])
    parser.add_argument("--fairness-exponents", type=float, nargs="+", default=[-1.0])
    parser.add_argument("--pilot-offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--identity-modes", nargs="+", default=["threshold"])
    parser.add_argument("--identity-margins", type=float, nargs="+", default=[0.0])
    parser.add_argument("--identity-margin-snr-slopes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--identity-margin-bin-slopes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--identity-score-modes", nargs="+", default=["mean"])
    parser.add_argument("--identity-windows", type=float, nargs="+", default=[1.25])
    parser.add_argument("--identity-margins-to-zero", type=float, nargs="+", default=[0.3])
    parser.add_argument("--identity-margins-to-one", type=float, nargs="+", default=[0.3])
    parser.add_argument("--control-levels", type=int, nargs="+", default=[31])
    parser.add_argument("--control-compandings", type=float, nargs="+", default=[1.0])
    parser.add_argument("--tri3-side-weights", type=float, nargs="+", default=[0.25])
    parser.add_argument("--frequency-interpolations", nargs="+", default=["none"])
    parser.add_argument("--gain-interpolation-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gain-interpolation-snr-slopes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-gain-refinement-scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-gain-refinement-radii", type=int, nargs="+", default=[4])
    parser.add_argument("--data-gain-refinement-min-snrs", type=float, nargs="+", default=[-10.0])
    parser.add_argument("--data-gain-kernels", nargs="+", default=["uniform"])
    parser.add_argument("--data-gain-center-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--data-gain-models", nargs="+", default=["constant"])
    parser.add_argument("--data-gain-soft-temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-gain-soft-temperature-decays", type=float, nargs="+", default=[1.0])
    parser.add_argument("--data-gain-residual-modes", nargs="+", default=["initial"])
    parser.add_argument("--data-gain-refinement-iteration-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--data-gain-refinement-snr-slopes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-vector-refinement-scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-vector-refinement-min-snrs", type=float, nargs="+", default=[-5.0])
    parser.add_argument("--data-vector-smoothing-sides", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-vector-refinement-snr-slopes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pre-vector-gain-refinement-iterations", type=int, nargs="+", default=[0])
    parser.add_argument("--data-vector-soft-temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-vector-confidence-floors", type=float, nargs="+", default=[1.0])
    parser.add_argument("--data-vector-reliability-powers", type=float, nargs="+", default=[0.0])
    parser.add_argument("--data-vector-refinement-iteration-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--interference-cancellation-scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--interference-cancellation-temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--interference-cancellation-min-snrs", type=float, nargs="+", default=[0.0])
    parser.add_argument("--interference-cancellation-confidence-floors", type=float, nargs="+", default=[1.0])
    parser.add_argument("--interference-gain-refinement-scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--interference-gain-refinement-iteration-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--interference-cancellation-snr-slopes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--reciprocal-cancellation-scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--reciprocal-cancellation-temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--interference-filter-loading-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--interference-vector-refinement-scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--joint-detection-candidates", type=int, nargs="+", default=[0])
    parser.add_argument("--joint-detection-prior-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--joint-detection-min-snrs", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pilot-bit-min-snrs", type=float, nargs="+", default=[100.0])
    parser.add_argument("--pilot-phase-bits", type=int, nargs="+", default=[1])
    parser.add_argument("--pilot-phase-weight-powers", type=float, nargs="+", default=[1.0])
    parser.add_argument("--pilot-phase-segments", type=int, nargs="+", default=[1])
    parser.add_argument("--pilot-slope-bits", type=int, nargs="+", default=[0])
    parser.add_argument("--pilot-slope-min-snrs", type=float, nargs="+", default=[100.0])
    parser.add_argument("--pilot-slope-steps", type=float, nargs="+", default=[0.1308996939])
    parser.add_argument("--pilot-phase-gate-thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--pilot-phase-schedules",
        nargs="*",
        default=[],
        help="Adaptive phase levels such as '2.5:2,6.25:3' (SNR dB:bits)",
    )
    parser.add_argument("--tail-threshold", type=float, default=-10.5)
    parser.add_argument(
        "--scores-out",
        type=Path,
        help="Optional NPZ path for per-sample SNRs and profile scores.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = ChannelMemmap(args.data)
    indices = deterministic_split_indices(len(data), seed=args.seed)[args.split][: args.samples]
    count = len(indices)
    rng = np.random.default_rng(args.seed)
    bits_np = rng.integers(0, 2, (count, 2, 1152), dtype=np.int8)
    snr_np = rng.uniform(-20.0, 20.0, (count, 2)).astype(np.float32)
    links: dict[str, object] = {
        "blind": DenoisedSparseMUMIMOLink().to(device),
        "reserved": ReservedPilotMUMIMOLink(2.0).to(device),
    }
    include_single_phase_profiles = not args.pilot_phase_schedules or (
        args.pilot_bit_min_snrs != [100.0] or args.pilot_phase_bits != [1]
    )
    phase_profiles = (
        [
            (((pilot_bit_min_snr, pilot_phase_bits),), f"e{pilot_bit_min_snr:g}q{pilot_phase_bits}")
            for pilot_bit_min_snr in args.pilot_bit_min_snrs
            for pilot_phase_bits in args.pilot_phase_bits
        ]
        if include_single_phase_profiles
        else []
    )
    for raw_schedule in args.pilot_phase_schedules:
        schedule = tuple(
            (float(level.split(":", maxsplit=1)[0]), int(level.split(":", maxsplit=1)[1]))
            for level in raw_schedule.split(",")
        )
        schedule_name = "-".join(f"e{threshold:g}q{bits}" for threshold, bits in schedule)
        phase_profiles.append((schedule, schedule_name))
    links.update(
        {
            f"paired_{separator}_{amp:g}_s{shrinkage:g}_l{loading:g}_c{boost:g}_d{iterations}_{minimum:g}_{rate:g}t{gain_refinement_soft_temperature:g}_v{covariance_blend:g}_r{regularization:g}_f{fairness:g}_o{pilot_offset}_i{identity_mode}{identity_margin:g}z{identity_margin_snr_slope:g}b{identity_margin_bin_slope:g}_{identity_score_mode}{identity_window:g}a{identity_margin_to_zero:g}b{identity_margin_to_one:g}_q{control_levels}x{control_companding:g}_w{tri3_side_weight:g}_x{frequency_interpolation}{gain_interpolation_scale:g}z{gain_interpolation_snr_slope:g}_g{data_gain_refinement_scale:g}r{data_gain_refinement_radius}m{data_gain_refinement_min_snr:g}k{data_gain_kernel}c{data_gain_center_weight:g}d{data_gain_model}t{data_gain_soft_temperature:g}a{data_gain_soft_temperature_decay:g}v{data_gain_residual_mode}n{data_gain_refinement_iterations}z{data_gain_refinement_snr_slope:g}_u{data_vector_refinement_scale:g}m{data_vector_refinement_min_snr:g}s{data_vector_smoothing_side:g}z{data_vector_refinement_snr_slope:g}p{pre_vector_gain_refinement_iterations}t{data_vector_soft_temperature:g}c{data_vector_confidence_floor:g}w{data_vector_reliability_power:g}n{data_vector_refinement_iterations}_k{interference_cancellation_scale:g}t{interference_cancellation_temperature:g}m{interference_cancellation_min_snr:g}c{interference_cancellation_confidence_floor:g}g{interference_gain_refinement_scale:g}n{interference_gain_refinement_iterations}z{interference_cancellation_snr_slope:g}_j{reciprocal_cancellation_scale:g}t{reciprocal_cancellation_temperature:g}_l{interference_filter_loading_scale:g}v{interference_vector_refinement_scale:g}_b{joint_detection_candidates}p{joint_detection_prior_scale:g}m{joint_detection_min_snr:g}_h{pilot_phase_weight_power:g}z{pilot_phase_segments}_d{pilot_slope_bits}m{pilot_slope_min_snr:g}s{pilot_slope_step:g}c{pilot_phase_gate_threshold:g}_{pilot_phase_name}": PairedPilotMUMIMOLink(
                amp,
                regularization_scale=regularization,
                separator=separator,
                steering_shrinkage=shrinkage,
                covariance_loading_scale=loading,
                central_boost=boost,
                gain_refinement_iterations=iterations,
                gain_refinement_min_snr=minimum,
                gain_refinement_rate=rate,
                gain_refinement_soft_temperature=gain_refinement_soft_temperature,
                sample_covariance_blend=covariance_blend,
                fairness_exponent=None if fairness < 0.0 else fairness,
                pilot_offset=pilot_offset,
                identity_mode=identity_mode,
                identity_margin=identity_margin,
                identity_margin_snr_slope=identity_margin_snr_slope,
                identity_margin_bin_slope=identity_margin_bin_slope,
                identity_score_mode=identity_score_mode,
                identity_window=identity_window,
                identity_margin_to_zero=identity_margin_to_zero,
                identity_margin_to_one=identity_margin_to_one,
                control_levels=control_levels,
                control_companding=control_companding,
                tri3_side_weight=tri3_side_weight,
                frequency_interpolation=frequency_interpolation,
                gain_interpolation_scale=gain_interpolation_scale,
                gain_interpolation_snr_slope=gain_interpolation_snr_slope,
                data_gain_refinement_scale=data_gain_refinement_scale,
                data_gain_refinement_radius=data_gain_refinement_radius,
                data_gain_refinement_min_snr=data_gain_refinement_min_snr,
                data_gain_kernel=data_gain_kernel,
                data_gain_center_weight=data_gain_center_weight,
                data_gain_model=data_gain_model,
                data_gain_soft_temperature=data_gain_soft_temperature,
                data_gain_soft_temperature_decay=data_gain_soft_temperature_decay,
                data_gain_residual_mode=data_gain_residual_mode,
                data_gain_refinement_iterations=data_gain_refinement_iterations,
                data_gain_refinement_snr_slope=data_gain_refinement_snr_slope,
                data_vector_refinement_scale=data_vector_refinement_scale,
                data_vector_refinement_min_snr=data_vector_refinement_min_snr,
                data_vector_smoothing_side=data_vector_smoothing_side,
                data_vector_refinement_snr_slope=data_vector_refinement_snr_slope,
                pre_vector_gain_refinement_iterations=pre_vector_gain_refinement_iterations,
                data_vector_soft_temperature=data_vector_soft_temperature,
                data_vector_confidence_floor=data_vector_confidence_floor,
                data_vector_reliability_power=data_vector_reliability_power,
                data_vector_refinement_iterations=data_vector_refinement_iterations,
                interference_cancellation_scale=interference_cancellation_scale,
                interference_cancellation_temperature=interference_cancellation_temperature,
                interference_cancellation_min_snr=interference_cancellation_min_snr,
                interference_cancellation_confidence_floor=interference_cancellation_confidence_floor,
                interference_gain_refinement_scale=interference_gain_refinement_scale,
                interference_gain_refinement_iterations=interference_gain_refinement_iterations,
                interference_cancellation_snr_slope=interference_cancellation_snr_slope,
                reciprocal_cancellation_scale=reciprocal_cancellation_scale,
                reciprocal_cancellation_temperature=reciprocal_cancellation_temperature,
                interference_filter_loading_scale=interference_filter_loading_scale,
                interference_vector_refinement_scale=interference_vector_refinement_scale,
                joint_detection_candidates=joint_detection_candidates,
                joint_detection_prior_scale=joint_detection_prior_scale,
                joint_detection_min_snr=joint_detection_min_snr,
                pilot_phase_schedule=pilot_phase_schedule,
                pilot_phase_weight_power=pilot_phase_weight_power,
                pilot_phase_segments=pilot_phase_segments,
                pilot_slope_bits=pilot_slope_bits,
                pilot_slope_min_snr=pilot_slope_min_snr,
                pilot_slope_step=pilot_slope_step,
                pilot_phase_gate_threshold=pilot_phase_gate_threshold,
            ).to(device)
            for separator in args.separators
            for amp in args.pilot_amplitudes
            for shrinkage in args.steering_shrinkages
            for loading in args.loading_scales
            for boost in args.central_boosts
            for iterations in args.gain_refinement_iterations
            for minimum in args.gain_refinement_min_snrs
            for rate in args.gain_refinement_rates
            for gain_refinement_soft_temperature in args.gain_refinement_soft_temperatures
            for covariance_blend in args.sample_covariance_blends
            for regularization in args.regularization_scales
            for fairness in args.fairness_exponents
            for pilot_offset in args.pilot_offsets
            for identity_mode in args.identity_modes
            for identity_margin in args.identity_margins
            for identity_margin_snr_slope in args.identity_margin_snr_slopes
            for identity_margin_bin_slope in args.identity_margin_bin_slopes
            for identity_score_mode in args.identity_score_modes
            for identity_window in args.identity_windows
            for identity_margin_to_zero in args.identity_margins_to_zero
            for identity_margin_to_one in args.identity_margins_to_one
            for control_levels in args.control_levels
            for control_companding in args.control_compandings
            for tri3_side_weight in args.tri3_side_weights
            for frequency_interpolation in args.frequency_interpolations
            for gain_interpolation_scale in args.gain_interpolation_scales
            for gain_interpolation_snr_slope in args.gain_interpolation_snr_slopes
            for data_gain_refinement_scale in args.data_gain_refinement_scales
            for data_gain_refinement_radius in args.data_gain_refinement_radii
            for data_gain_refinement_min_snr in args.data_gain_refinement_min_snrs
            for data_gain_kernel in args.data_gain_kernels
            for data_gain_center_weight in args.data_gain_center_weights
            for data_gain_model in args.data_gain_models
            for data_gain_soft_temperature in args.data_gain_soft_temperatures
            for data_gain_soft_temperature_decay in args.data_gain_soft_temperature_decays
            for data_gain_residual_mode in args.data_gain_residual_modes
            for data_gain_refinement_iterations in args.data_gain_refinement_iteration_counts
            for data_gain_refinement_snr_slope in args.data_gain_refinement_snr_slopes
            for data_vector_refinement_scale in args.data_vector_refinement_scales
            for data_vector_refinement_min_snr in args.data_vector_refinement_min_snrs
            for data_vector_smoothing_side in args.data_vector_smoothing_sides
            for data_vector_refinement_snr_slope in args.data_vector_refinement_snr_slopes
            for pre_vector_gain_refinement_iterations in args.pre_vector_gain_refinement_iterations
            for data_vector_soft_temperature in args.data_vector_soft_temperatures
            for data_vector_confidence_floor in args.data_vector_confidence_floors
            for data_vector_reliability_power in args.data_vector_reliability_powers
            for data_vector_refinement_iterations in args.data_vector_refinement_iteration_counts
            for interference_cancellation_scale in args.interference_cancellation_scales
            for interference_cancellation_temperature in args.interference_cancellation_temperatures
            for interference_cancellation_min_snr in args.interference_cancellation_min_snrs
            for interference_cancellation_confidence_floor in args.interference_cancellation_confidence_floors
            for interference_gain_refinement_scale in args.interference_gain_refinement_scales
            for interference_gain_refinement_iterations in args.interference_gain_refinement_iteration_counts
            for interference_cancellation_snr_slope in args.interference_cancellation_snr_slopes
            for reciprocal_cancellation_scale in args.reciprocal_cancellation_scales
            for reciprocal_cancellation_temperature in args.reciprocal_cancellation_temperatures
            for interference_filter_loading_scale in args.interference_filter_loading_scales
            for interference_vector_refinement_scale in args.interference_vector_refinement_scales
            for joint_detection_candidates in args.joint_detection_candidates
            for joint_detection_prior_scale in args.joint_detection_prior_scales
            for joint_detection_min_snr in args.joint_detection_min_snrs
            for pilot_phase_weight_power in args.pilot_phase_weight_powers
            for pilot_phase_segments in args.pilot_phase_segments
            for pilot_slope_bits in args.pilot_slope_bits
            for pilot_slope_min_snr in args.pilot_slope_min_snrs
            for pilot_slope_step in args.pilot_slope_steps
            for pilot_phase_gate_threshold in args.pilot_phase_gate_thresholds
            for pilot_phase_schedule, pilot_phase_name in phase_profiles
        }
    )
    for link in links.values():
        link.load_denoiser_checkpoint(args.checkpoint)
        link.eval()
    score_chunks: dict[str, list[np.ndarray]] = {name: [] for name in links}
    one_chunks: dict[str, list[np.ndarray]] = {name: [] for name in links}
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        channel = torch.from_numpy(data.read(indices[start:end])).to(device)
        bits = torch.from_numpy(bits_np[start:end]).to(device=device, dtype=torch.float32)
        snr = torch.from_numpy(snr_np[start:end]).to(device)
        for name, link in links.items():
            generator = torch.Generator(device=device).manual_seed(args.seed + start)
            output = link(channel, bits, snr, generator=generator)
            llr = output if name == "blind" else output[0]
            user_scores = []
            for user in range(2):
                full_score = per_sample_score(bits[:, user], llr[:, user])
                if isinstance(link, PairedPilotMUMIMOLink) and link.pilot_bit_min_snr < 90.0:
                    # The official harness invokes the submission with batch size one, so
                    # the receiver omits phase-carried bits below their SNR threshold.
                    # Keep the research evaluator vectorized while matching that dynamic
                    # output length exactly instead of scoring unsent padded bits.
                    maximum_phase_bits = link.pilot_phase_bits * link.pilot_phase_segments
                    maximum_pilot_bits = maximum_phase_bits + link.pilot_slope_bits
                    base_llr = llr[:, user, :-maximum_pilot_bits]
                    full_score = per_sample_score(bits[:, user], base_llr)
                    for phase_threshold, phase_bits in link.pilot_phase_schedule:
                        active_phase_bits = phase_bits * link.pilot_phase_segments
                        phase_score = per_sample_score(
                            bits[:, user], llr[:, user, : base_llr.shape[1] + active_phase_bits]
                        )
                        if (
                            phase_bits == link.pilot_phase_bits
                            and link.pilot_phase_segments == 1
                            and link.pilot_phase_gate_threshold > 0.0
                        ):
                            phase_llr = llr[
                                :, user, base_llr.shape[1] : base_llr.shape[1] + phase_bits
                            ]
                            phase_confidence = torch.abs(phase_llr[:, -1]) / torch.abs(
                                phase_llr
                            ).amax(dim=1).clamp_min(1e-9)
                            fallback_score = per_sample_score(
                                bits[:, user], llr[:, user, : base_llr.shape[1] + phase_bits - 1]
                            )
                            phase_score = torch.where(
                                phase_confidence >= link.pilot_phase_gate_threshold,
                                phase_score,
                                fallback_score,
                            )
                        full_score = torch.where(
                            snr[:, user] >= phase_threshold,
                            phase_score,
                            full_score,
                        )
                    if link.pilot_slope_bits:
                        slope_score = per_sample_score(
                            bits[:, user], llr[:, user, : base_llr.shape[1] + maximum_pilot_bits]
                        )
                        full_score = torch.where(
                            snr[:, user] >= link.pilot_slope_min_snr,
                            slope_score,
                            full_score,
                        )
                user_scores.append(full_score)
            score_chunks[name].append(torch.stack(user_scores, dim=1).cpu().numpy())
            one_chunks[name].append(
                torch.stack([per_sample_score(bits[:, user], llr[:, user, :1]) for user in range(2)], dim=1)
                .cpu()
                .numpy()
            )
    scores = {name: np.concatenate(parts) for name, parts in score_chunks.items()}
    one_scores = {name: np.concatenate(parts) for name, parts in one_chunks.items()}
    if args.scores_out is not None:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {"snr": snr_np}
        for profile_index, name in enumerate(links):
            payload[f"score_{profile_index}"] = scores[name]
            payload[f"one_score_{profile_index}"] = one_scores[name]
        payload["profile_names"] = np.asarray(list(links), dtype=np.str_)
        np.savez_compressed(args.scores_out, **payload)
    low = snr_np < args.tail_threshold
    maximum = snr_np.max(axis=1)
    candidates = []
    blind_dynamic = np.where(low, one_scores["blind"], scores["blind"])
    for name in links:
        if name in {"blind", "reserved"}:
            continue
        paired_dynamic = np.where(low, one_scores[name], scores[name])
        for threshold in np.arange(-10.0, 20.01, 0.5):
            use_paired = maximum < threshold
            selected = np.where(use_paired[:, None], paired_dynamic, blind_dynamic)
            summary = summarize_scores(selected)
            candidates.append(
                {
                    "profile": name,
                    "threshold_db": float(threshold),
                    "fraction": float(use_paired.mean()),
                    "efficiency": summary.efficiency,
                    "fairness": summary.fairness,
                    "final": summary.final,
                }
            )
    reserved_dynamic = np.where(low, one_scores["reserved"], scores["reserved"])
    paired_profiles = [name for name in links if name.startswith("paired_")]
    for name in paired_profiles:
        paired_dynamic = np.where(low, one_scores[name], scores[name])
        for reserved_threshold in np.arange(-10.0, 10.01, 0.5):
            for paired_threshold in np.arange(reserved_threshold, 20.01, 0.5):
                use_reserved = maximum < reserved_threshold
                use_paired = (maximum >= reserved_threshold) & (maximum < paired_threshold)
                selected = np.where(
                    use_reserved[:, None],
                    reserved_dynamic,
                    np.where(use_paired[:, None], paired_dynamic, blind_dynamic),
                )
                summary = summarize_scores(selected)
                candidates.append(
                    {
                        "profile": f"reserved_then_{name}",
                        "reserved_threshold_db": float(reserved_threshold),
                        "threshold_db": float(paired_threshold),
                        "fraction": float((use_reserved | use_paired).mean()),
                        "efficiency": summary.efficiency,
                        "fairness": summary.fairness,
                        "final": summary.final,
                    }
                )
        for tail_threshold in np.arange(-20.0, -4.99, 0.5):
            candidate_low = snr_np < tail_threshold
            candidate_blind = np.where(candidate_low, one_scores["blind"], scores["blind"])
            candidate_paired = np.where(candidate_low, one_scores[name], scores[name])
            for paired_threshold in np.arange(10.0, 20.01, 0.5):
                use_paired = maximum < paired_threshold
                selected = np.where(use_paired[:, None], candidate_paired, candidate_blind)
                summary = summarize_scores(selected)
                candidates.append(
                    {
                        "profile": f"tail_search_{name}",
                        "tail_threshold_db": float(tail_threshold),
                        "threshold_db": float(paired_threshold),
                        "fraction": float(use_paired.mean()),
                        "efficiency": summary.efficiency,
                        "fairness": summary.fairness,
                        "final": summary.final,
                    }
                )
    blind_summary = summarize_scores(blind_dynamic)
    best_direct_by_profile = []
    for name in paired_profiles:
        direct = [item for item in candidates if item["profile"] == name]
        best_direct_by_profile.append(max(direct, key=lambda item: item["final"]))
    print(
        json.dumps(
            {
                "samples": count,
                "split": args.split,
                "blind": blind_summary.__dict__,
                "best": max(candidates, key=lambda item: item["final"]),
                "best_direct_by_profile": sorted(
                    best_direct_by_profile, key=lambda item: item["final"], reverse=True
                ),
                "top": sorted(candidates, key=lambda item: item["final"], reverse=True)[:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
