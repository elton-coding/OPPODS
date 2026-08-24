# 星座、反馈、预编码与导频位置全局扫描

日期：2026-08-24  
分支：`feature/constellation-precoder-sweep`  
对照：V120

300 样本 exact 扫描结论：

- `CENTRAL_BOOST=-1/-0.75/-0.25/0/0.25` 均显著低于默认 `-0.5`；
- `WIENER_NOISE_SCALE=0.5/0.625/0.875/1.0/1.25` 均低于默认 `0.75`；
- `PILOT_OFFSET=0/3/4/6/7/11` 均低于组中心附近的默认 `5`；
- 普通 `RZF_REGULARIZATION` 在官方 SNR 范围内不改变结果，因为链路始终使用 reserved-pilot 专用 RZF。

本轮没有候选进入多 seed，不合并 main。
