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

## 当前冠军方案（V227）

V227 是纯神经黑盒链路：共享 Encoder 压缩为96个复反馈符号；按双用户最低SNR=-10dB划分两组联合Transmitter/Receiver专家，均使用宽512、10块逐子载波MLP。低模式从V224联合训练，高模式和Encoder冻结。每RE8bit、每UE完整1152bit，使用 `0.7×效率+0.3×P10` 的软评分代理及少量BCE。

固定留出集（split1176、offset2000、noise22701/22702/22703）本地总分为 `67.950836/68.211780/68.301918`，均值 **`68.154844`**；同口径V224为68.069933。相比同预算普通续训，分区方法净增0.035048。P10有小幅回落，并非所有区间都改善；尚未达到69，线上尚未确认。历史评测交叉问题见[划分审计](docs/experiments/evaluation-partition-audit-v227.md)。

完整设计见[V227实验报告](docs/experiments/pure-neural-joint-snr-v227.md)，四组消融见[V229组合报告](docs/experiments/pure-neural-joint-hard-rank-v229.md)，冻结参数见[final.yaml](configs/final.yaml)，成绩见[冠军基准表](benchmarks/leaderboard.json)。

## Git 与版本纪律

- `main` 只保存经过固定协议复核的最高总分方案。
- 新算法、新模块和可能影响得分的尝试必须从 `main` 创建 `codex/<topic>-v<number>` 分支，并先推送远端。
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
& $python scripts/evaluate_submission.py --samples 2000 --split-seed 1176 --test-offset 2000 --seed 22701
```

最终提交包为 `artifacts/FATE_MIMO_submission.zip`。当前成绩以 `benchmarks/leaderboard.json` 为准；`artifacts/final_results.json` 是旧版本历史产物，不能作为当前冠军依据。本地结果不等同于线上排行榜成绩。
