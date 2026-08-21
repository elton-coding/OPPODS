from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    num_ue: int = 2
    num_uplink_subcarriers: int = 96
    num_downlink_subcarriers: int = 144
    num_downlink_ctrl_bits: int = 5
    num_tx_antennas: int = 16
    num_rx_antennas: int = 2
    max_bits_per_ue: int = 1152
    snr_dl_min_db: float = -20.0
    snr_dl_max_db: float = 20.0
    snr_ul_gap_db: float = 10.0
    fairness_percentile: float = 10.0
    efficiency_weight: float = 0.7


DEFAULT_SYSTEM = SystemConfig()
