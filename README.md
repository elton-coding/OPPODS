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

## 当前通过条件性确认的本地冠军方案（V273）

八组5dB联合最低SNR纯神经专家，512宽10块MLP，共享前8块；冻结Encoder，73,547,840独立参数。V227初始化、全新Adam/144000步，验证选择139000步。相对V269只将预算72k翻倍144k，不是同计算预算提升；恒定学习率3e-5、B1152/k8、RMS评分代理+BCE0.05不变。

固定offset2000本地均分 **68.932765**，相对V269 +0.154174，95%[0.113431,0.182284]。冻结offset8000新噪声确认均分 **68.757872**，同窗口相对V269 +0.130223，95%[0.092956,0.161649]，各三组综合分均正。不同窗口绝对分数不能直接比较。

约273MB提交包，完整CPU/实际ZIP接口校验通过。确认2000/2000信道历史复用、祖先谱系有缺口，不是盲测；未达69、线上未知。唯一用户确认线上成绩仍为V223的67.01164987745。

见[提交说明](docs/submission/V273_提交说明.md)、[确认结果](docs/experiments/frozen-confirmation-v273-results.md)、[配置](configs/final.yaml)、[冠军记录](benchmarks/leaderboard.json)。V269及更早不可变包保留。

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
