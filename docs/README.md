# OPPODS 文档索引

文档按职责分类，避免在仓库根目录继续生成带“最终版”“最新版”含义不清的文件。

## 赛题资料

- [DataFountain 1176 完整原文](competition/DataFountain_1176_完整原文.md)

## 方案与提交

- [FATE-MIMO 初始技术方案](solutions/FATE-MIMO_v1.md)
- [FATE-MIMO V115 冻结方案](solutions/FATE-MIMO_v115.md)
- [FATE-MIMO V117 冠军增量方案](solutions/FATE-MIMO_v117.md)
- [FATE-MIMO V118 SNR 分段 Receiver 方案](solutions/FATE-MIMO_v118.md)
- [FATE-MIMO V119 SNR 分段导频插值方案](solutions/FATE-MIMO_v119.md)
- [FATE-MIMO V120 弱用户输出策略](solutions/FATE-MIMO_v120.md)
- [FATE-MIMO V121 控制码字预算方案](solutions/FATE-MIMO_v121.md)
- [FATE-MIMO V122 SNR 分段匹配前缀方案](solutions/FATE-MIMO_v122.md)
- [FATE-MIMO V123 Walsh 尾部码本方案](solutions/FATE-MIMO_v123.md)
- [FATE-MIMO V124 置信度前缀专家方案](solutions/FATE-MIMO_v124.md)
- [FATE-MIMO V125 SNR 分箱截断置信度方案](solutions/FATE-MIMO_v125.md)
- [FATE-MIMO V126 裸前缀语义扩展方案](solutions/FATE-MIMO_v126.md)
- [FATE-MIMO V127 Walsh 尾部置信度再校准方案](solutions/FATE-MIMO_v127.md)
- [V115 提交说明](submission/V115_提交说明.md)
- [V117 提交说明](submission/V117_提交说明.md)
- [V118 提交说明](submission/V118_提交说明.md)
- [V119 提交说明](submission/V119_提交说明.md)
- [V120 提交说明](submission/V120_提交说明.md)
- [V121 提交说明](submission/V121_提交说明.md)
- [V122 提交说明](submission/V122_提交说明.md)
- [V123 提交说明](submission/V123_提交说明.md)
- [V124 提交说明](submission/V124_提交说明.md)
- [V125 提交说明](submission/V125_提交说明.md)
- [V126 提交说明](submission/V126_提交说明.md)
- [V127 提交说明](submission/V127_提交说明.md)

## 实验记录

- [2026-08-24 持续优化总结](experiments/2026-08-24-continuous-optimization-summary.md)
- [离散 SNR 去噪专家银行](experiments/snr-expert-bank.md)
- [SNR 分段 Receiver 物理参数专家](experiments/snr-receiver-physical-profiles.md)
- [中低 SNR 输出策略扫描](experiments/snr-output-policy-sweep.md)
- [控制码字预算扫描](experiments/control-codeword-allocation.md)
- [V121 中档前缀边界复扫](experiments/v121-middle-prefix-resweep.md)
- [弱用户 Walsh 尾部码本扫描](experiments/threshold-codebook-walsh.md)
- [中档 SNR 置信度前缀扫描](experiments/confidence-prefix-v123.md)
- [V124 中档 SNR 扩展置信度诊断](experiments/confidence-diagnostics-v124.md)
- [V125 SNR 分箱截断置信度门控](experiments/snr-binned-clipped-confidence-v125.md)
- [V126 裸 924 位回退的高 SNR 扩展](experiments/bare-prefix-extension-v126.md)
- [V127 Walsh 尾部置信度再校准](experiments/walsh-tail-confidence-v127.md)

## 外部方案研究

- [2025 获奖方案视频精读](research/2025获奖方案视频精读.md)
- [2025 获奖方案对 V115 的行动项](research/2025方案对V115的行动项.md)
- `research/samsung-2025/`：三星电子冠军方案的关键截图。

## 项目治理

- [版本与实验管理](governance/版本与实验管理.md)

生成的日志、临时候选、模型汤和压缩包统一保存在 Git 忽略的 `artifacts/`、`checkpoints/`、`runs/` 中。需要长期保留的结论必须提炼为本文档树中的 Markdown，成绩必须登记到 `benchmarks/leaderboard.json`。
