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

## 当前本地冠军方案（V250）

V250 是纯神经黑盒链路：共享Encoder压缩为96个复反馈符号；按双用户最低SNR将[-20,20]dB分为八个5dB区间，选择对应联合Transmitter/Receiver专家，均使用宽512、10块逐子载波MLP。从V227复制初始化后训练全部Tx/Rx共72000步，最佳验证检查点为71000步，Encoder冻结。每RE8bit、每UE完整1152bit；损失先按UE对logits作可导RMS归一化，再计算 `0.7×软效率+0.3×软P10`，另外保留原始logits上的0.05 BCE，温度0.5。RMS只用于训练损失，不改变推理接口。

固定选型audit（split1176、offset2000、noise22701/22702/22703）本地总分为 `68.414028/68.623123/68.780935`，均值 **`68.606029`**；相比原冠军V242E提高0.074403，配对95%区间[0.041448,0.103418]。唯一改动是训练预算36k→72k，前36k全部37个验证点四项指标差0。三组总分/效率均升，一组P10下降0.173611。冻结后的offset6000确认均值68.426862，同窗口对照68.338613，净增0.088250、区间[0.053781,0.118330]，三组总分均升、P10两升一平；不同窗口绝对均分不可直接比较。

约707MB提交包低于1GB；尚未达到69，线上未确认。确认窗口未用于本轮候选选型，但与更早历史评测有1792/2000通道交叉，祖先训练谱系记录也未完全补齐，不能称全项目盲测或已证明全谱系无泄漏。详见[冻结确认及限制](docs/experiments/frozen-confirmation-v250.md)和[划分审计](docs/experiments/evaluation-partition-audit-v227.md)。

当前结果见[V250预算消融](docs/experiments/pure-neural-long-budget-v250.md)与[提交说明](docs/submission/V250_提交说明.md)，旧V242E/V240C固定包和原始权重保留，消融见[总表](docs/experiments/ablation-registry.md)，冻结参数见[final.yaml](configs/final.yaml)，成绩见[冠军基准表](benchmarks/leaderboard.json)。

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
