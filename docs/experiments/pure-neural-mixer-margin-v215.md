# V215：MLP-Mixer Receiver 与评分对齐损失

## 设计

- 保留 V213 的 k7/B1008 Encoder 与 Transmitter。
- Receiver 从 6 层 Transformer 改为 8 层、宽度 256 的 Post-Norm MLP-Mixer。
- 分三阶段训练：Receiver BCE 25k、Receiver Hinge+BCE 5k、全链路 Hinge+BCE 10k。
- Hinge+BCE 阶段加入 10% 尾部、权重 0.5 的 score-aligned loss。

## 结果

随机初始化 Mixer Receiver 的固定验证分从 48.1060 提升到 62.2045；Hinge+BCE 继续提升到 62.3470；端到端解冻后达到 62.5107。严格 seed1176/2000 最终为 62.407784，低于 V213 的 62.458605。

逐 2.5 dB 配对显示，V215 在 `[-5,2.5)` 比 V213 高约 0.27 至 0.51 分，但在 10 dB 以上低约 0.43 至 0.91 分。Mixer 与 margin 在中 SNR 有效，但新 Receiver 被迫适配旧 Transmitter 的表示，高 SNR 上限反而下降。

## 结论

V215 不晋升主干。下一版不再只替换 Receiver，而按公开视频中更完整的高分路线，联合重构逐子载波 Transmitter/Receiver，并从头端到端训练。
