# OPPODS / DataFountain 1176

本仓库用于开发 6G/B6G 内生 AI 多用户 MIMO 端到端传输方案。官方原始样例保留在 `ziliao/`，研发代码位于 `src/`、`scripts/` 和 `tests/`，当前最高分可部署版本位于 `modelSubmit/`。

## 环境

当前开发环境：

```text
D:\Tools\Anaconda\envs\oppods-df1176
Python 3.11
PyTorch 2.10.0 + CUDA 12.8
NumPy 1.26.4
```

PowerShell 中运行：

```powershell
$python = 'D:\Tools\Anaconda\envs\oppods-df1176\python.exe'
& $python -m pip install -e 'D:\Source\OPPODS[dev]'
& $python -m pytest
```

## 当前冠军方案（V117）

最终采用任务导向稀疏反馈、解析 Wiener 零初始化残差 Transformer、鲁棒 RZF、分层 Gray 256-QAM、正交保留导频，以及 5-bit 控制信道上的“用户排序 + 弱用户约束尾部向量量化”。控制阈值在两用户 SNR 之间通常存在多个等价选择；发送端利用这些等价码字压缩瞬时弱用户未发送的 bit 尾部，接收端在不增加接口的情况下恢复固定模板。由于弱用户尾部 VQ 已能保护极低 SNR，V117 在赛题范围内取消了原 V115 的 1-bit 截断。

冻结版严格本地官方同口径评测（2000 样本、4000 UE 分数、seed 1176）：效率 `68.242752`，公平 P10 `50.607639`，总分 **`62.952218`**。三个固定 seed 的平均总分为 **`63.077458`**，相对 V115 提升 `0.250021`。这些是本地验证结果，不等同于线上排行榜成绩。

完整设计见 [FATE-MIMO V117](docs/solutions/FATE-MIMO_v117.md)，冻结参数见 [final.yaml](configs/final.yaml)，可追溯成绩见 [冠军基准表](benchmarks/leaderboard.json)。

## Git 与版本纪律

- `main` 只保存经过固定协议复核的最高总分方案。
- 新算法、新模块和可能影响得分的尝试必须从 `main` 创建 `feature/<topic>` 分支，并先推送远端。
- 实验原始产物留在被忽略的 `artifacts/`、`checkpoints/` 和 `runs/`；只有摘要指标、复现配置和晋级模型进入 Git。
- 只有候选方案通过测试、多 seed 评测且总分高于当前冠军后，才允许合并到 `main`。

详细规则见 [版本与实验管理](docs/governance/版本与实验管理.md)，文档入口见 [docs/README.md](docs/README.md)。

## 常用命令

完美 CSI 上界：

```powershell
& $python scripts/evaluate_oracle.py --samples 2000 --batch-size 128
```

测试、构建和复刻官方逐样本评测：

```powershell
& $python -m pytest
& $python scripts/build_submission.py --package-only
& $python scripts/evaluate_submission.py --samples 2000
```

最终提交包为 `artifacts/FATE_MIMO_submission.zip`。本地固定测试结果写入 `artifacts/final_results.json`；这些结果不等同于线上排行榜成绩。
