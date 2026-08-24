# V121 control data-mode gate sweep

## Objective

V121 assigns one of the 32 control codewords to the high-SNR data-mode
template and the other 31 codewords to the weak-user threshold quantizer. This
experiment exposes the boundary between the two modes as
`DATA_MODE_MIN_SNR_DB` and tests whether moving the V121 boundary away from
10.25 dB improves the local simulator score.

## Protocol

- submission: V121 (`DATA_MODE_CODEWORDS=1`, `THRESHOLD_CODEWORDS=31`)
- validation seed: 1176
- samples: 300 (600 user scores)
- device: CUDA
- unchanged reference at 10.25 dB: efficiency 67.403284, P10 50.685764,
  final 62.388028

## Results

| gate (dB) | efficiency | P10 | final | delta vs V121 |
|---:|---:|---:|---:|---:|
| 5.00 | 67.398582 | 50.607639 | 62.361299 | -0.026729 |
| 6.00 | 67.385634 | 50.685764 | 62.375673 | -0.012355 |
| 7.00 | 67.387442 | 50.607639 | 62.353501 | -0.034527 |
| 8.00 | 67.409505 | 50.607639 | 62.368945 | -0.019083 |
| 9.00 | 67.388383 | 50.607639 | 62.354159 | -0.033869 |
| 10.00 | 67.407046 | 50.607639 | 62.367224 | -0.020804 |
| **10.25** | **67.403284** | **50.685764** | **62.388028** | **0.000000** |
| 11.00 | 67.377749 | 50.598958 | 62.344112 | -0.043916 |
| 12.00 | 67.401765 | 50.598958 | 62.360923 | -0.027105 |
| 13.00 | 67.379919 | 50.520833 | 62.322193 | -0.065835 |
| 15.00 | 67.407841 | 50.520833 | 62.341739 | -0.046289 |
| 17.00 | 67.387659 | 50.520833 | 62.327611 | -0.060417 |
| 20.00 | 67.370226 | 50.512153 | 62.312804 | -0.075224 |

## Decision

Reject the gate change. No candidate beats V121 on the identical validation
set, and the P10 staircase becomes worse above 10.25 dB. Keep the
parameterization on this feature branch because it makes the negative result
reproducible, but do not merge it into `main`.

Raw outputs are stored as
`artifacts/diagnostics/v121_gate_*_300_seed1176.json`.
