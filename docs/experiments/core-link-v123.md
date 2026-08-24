# V123 核心链路参数复扫

日期：2026-08-24  
分支：`feature/core-link-v123`  
对照：V123

seed1176、300 样本单参数扫描：

- `RZF_REGULARIZATION=1.0/1.25/1.75/2.0` 与 V123 逐项相同；比赛 SNR
  范围始终走 reserved profile，该全局分支不生效；
- `PILOT_RZF_REGULARIZATION=0.3/0.4/0.5/0.6` 均低于默认 `0.45`；
- `PILOT_AMPLITUDE=1.25/1.75/2.0` 均显著低于默认 `1.5`；
- `PILOT_COVARIANCE_LOADING_SCALE=3/5/6` 均低于默认 `4`。

所有实际生效参数在当前点附近都呈单峰或单向退化，没有候选进入多 seed。
V123 核心链路参数保持不变。
