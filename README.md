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

## 当前冠军方案（V224）

V224 是纯神经黑盒链路：Encoder 把分组三频带信道压缩为 96 个复反馈符号，Transmitter 与 Receiver 使用宽度 512、10 个残差块的逐子载波共享 MLP。每个资源单元传输 8 bit，每 UE 固定输出完整 1152 bit；训练后期直接优化可导的官方 `0.7×效率+0.3×P10` 评分代理。

固定 seed1176/1177/1178 的官方兼容本地总分为 `67.890964/68.239655/68.064460`，均值 **`68.065026`**，相对 V223 提升 `0.822413`。以上均为本地验证结果，不等同于线上排行榜成绩。

完整设计与消融见 [V224 实验报告](docs/experiments/pure-neural-soft-score-loss-v224.md)，冻结参数见 [final.yaml](configs/final.yaml)，可追溯成绩见 [冠军基准表](benchmarks/leaderboard.json)。

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
& $python scripts/evaluate_submission.py --samples 2000
```

最终提交包为 `artifacts/FATE_MIMO_submission.zip`。本地固定测试结果写入 `artifacts/final_results.json`；这些结果不等同于线上排行榜成绩。
