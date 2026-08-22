# OPPODS 文档索引

文档按职责分类，避免在仓库根目录继续生成带“最终版”“最新版”含义不清的文件。

## 赛题资料

- [DataFountain 1176 完整原文](competition/DataFountain_1176_完整原文.md)

## 方案与提交

- [FATE-MIMO 初始技术方案](solutions/FATE-MIMO_v1.md)
- [FATE-MIMO V115 冻结方案](solutions/FATE-MIMO_v115.md)
- [FATE-MIMO V117 冠军增量方案](solutions/FATE-MIMO_v117.md)
- [V115 提交说明](submission/V115_提交说明.md)
- [V117 提交说明](submission/V117_提交说明.md)
- [Score-aligned CVaR 与弱用户 SNR 定向训练实验](experiments/score-aligned-cvar-training.md)

## 外部方案研究

- [2025 获奖方案视频精读](research/2025获奖方案视频精读.md)
- [2025 获奖方案对 V115 的行动项](research/2025方案对V115的行动项.md)
- `research/samsung-2025/`：三星电子冠军方案的关键截图。

## 项目治理

- [版本与实验管理](governance/版本与实验管理.md)

生成的日志、临时候选、模型汤和压缩包统一保存在 Git 忽略的 `artifacts/`、`checkpoints/`、`runs/` 中。需要长期保留的结论必须提炼为本文档树中的 Markdown，成绩必须登记到 `benchmarks/leaderboard.json`。
