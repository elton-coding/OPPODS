# V218：最低 SNR 联合 Transmitter/Receiver profiles

## 设计

以 V216 为共享起点，将 `[-20,20] dB` 划分为 16 个 2.5 dB profile。双用户较低 SNR 决定 Transmitter profile；Transmitter 用 5-bit 控制码发送 profile 编号，两个 Receiver 均据此选择同一模型。Encoder 保持 V216 权重冻结。每个 profile 在“最小 SNR 落入本区间、另一用户不低于它”的联合分布上独立微调。

训练器修复了从单专家包初始化多专家库时没有剥离 `experts.0.*` 前缀的问题。修复后的 500 样本起点为 65.091997；错误随机初始化实验作废。

## 结果

- profiles 0–4 均因 P10 回落被第 0 步门禁拒绝。
- profiles 5–15 多数获得局部正增益；其中 profile 7 约 +0.659，profile 8 约 +1.072，profiles 9–12 约 +1.15 至 +1.33。
- 完整官方逐样本 seed1176/2000：efficiency 70.861328、P10 51.996528、final **65.201888**。
- 相对 V216 同 seed 65.125291，增量 **+0.076597**。

联合控制码解决了 V214 的发射/接收专家错配，产生真实正增益，但局部验证增益不能线性叠加到全局。V218 暂不晋升主干，等待长程训练和三 seed 复评。
