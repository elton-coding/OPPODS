# FATE-MIMO V132：高 SNR Receiver 协方差专家

版本：V132，2026-08-25

V132 保留 V131 的双 UE 多区间 RZF，并在单 UE SNR `[8.75,20) dB` 将 Receiver 导频协方差加载从 `4.0` 提高到 `5.0`。固定三 seed final 为 `63.049999/63.288031/63.231086`，均值 **`63.189705`**，相对 V131 提升 **`0.006912`**；冻结后三条盲测全部为正且 P10 不变。详见 [V132 高 SNR Receiver 协方差加载专家](../experiments/receiver-covload-v132.md)。

