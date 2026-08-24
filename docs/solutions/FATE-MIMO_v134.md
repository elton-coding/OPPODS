# FATE-MIMO V134：分层高 SNR Receiver 协方差专家

版本：V134，2026-08-25

V134 保留 V133 的整体结构，将 Receiver 协方差加载进一步按单 UE SNR 细分：默认 `4.0`，`[8.75,11.65) dB` 为 `5.0`，`[11.65,20) dB` 为 `7.0`。固定三 seed final 为 `63.055088/63.293545/63.238332`，均值 **`63.195655`**，相对 V133 提升 **`0.001595`**；冻结后三条盲测全部为正且 P10 不变。详见 [V134 分层高 SNR Receiver 协方差加载专家](../experiments/receiver-covload-ultrahigh-v134.md)。

