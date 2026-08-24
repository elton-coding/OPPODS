# FATE-MIMO V133：高 SNR Receiver 协方差加载增强

版本：V133，2026-08-25

V133 保留 V132 的双 UE 多区间 RZF 和 Receiver SNR 路由，仅将单 UE SNR `[8.75,20) dB` 的导频协方差加载从 `5.0` 加强到 `7.0`，其他区间维持 `4.0`。固定三 seed final 为 `63.054906/63.291874/63.235400`，均值 **`63.194060`**，相对 V132 提升 **`0.004355`**；冻结后三条盲测全部为正且 P10 不变。详见 [V133 高 SNR Receiver 协方差加载强度搜索](../experiments/receiver-covload-high-v133.md)。

