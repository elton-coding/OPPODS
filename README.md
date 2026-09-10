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

## 当前通过确认的本地冠军方案（V257）

纯神经八组5dB联合最低SNR专家，宽512/10块MLP；同一父组共享前8块，后2块及输出保持独立。Encoder冻结，73,547,840个独立参数。从V227初始化、训练72000步，验证选择70000步。B1152/k8、96复反馈、5控制位；RMS软评分损失及0.05原始BCE保持不变。

固定offset2000三次本地均分 **68.650598**，相对V250 +0.044570，95%区间[0.007856,0.065347]。预登记offset8000确认均分 **68.462279**，同窗口相对V250 +0.041591，区间[0.007333,0.067913]，三次总分均提高。不同窗口绝对分数不可直接比较。

约273MB包低于1GB。确认窗口有1812/2000信道历史评测复用和祖先谱系缺口，不能称完全盲测；未达69、线上未知。V258选型均分略高，但未通过可靠优于V257的双参考门槛，因此不直接替换。唯一用户确认的线上成绩仍为V223的67.01164987745。

见[提交说明](docs/submission/V257_提交说明.md)、[确认记录](docs/experiments/frozen-confirmation-v257.md)、[配置](configs/final.yaml)及[冠军记录](benchmarks/leaderboard.json)。V250固定包、原始权重和标签保留。

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
