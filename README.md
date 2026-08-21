# OPPODS / DataFountain 1176

本目录用于开发 6G/B6G 内生 AI 多用户 MIMO 端到端传输方案。官方原始样例保留在 `ziliao/`，新代码位于 `src/`、`scripts/` 和 `tests/`。

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

## 最终方案

最终采用任务导向稀疏反馈、解析 Wiener 零初始化残差 Transformer、鲁棒 RZF、分层 Gray 256-QAM、正交保留导频，以及 5-bit 控制信道上的“用户排序 + 约束尾部向量量化”。控制阈值在两用户 SNR 之间通常存在多个等价选择；发送端利用这些等价码字压缩高角色未发送的 bit 尾部，接收端在不增加接口的情况下恢复固定模板。SNR 小于 -15.5 dB 的 1-bit 输出保持不变，保护全局 P10。

冻结版严格本地官方同口径评测（2000 样本、4000 UE 分数、seed 1176）：效率 `68.133659`，公平 P10 `49.956597`，总分 **`62.680540`**。相对上一稳定版 `62.454635` 提升 `0.225905`。这些是本地验证结果，不等同于线上排行榜成绩。

完整设计见 `获奖技术方案_v2_落地版.md`，冻结参数见 `configs/final.yaml`。

## 常用命令

完美 CSI 上界：

```powershell
& $python scripts/evaluate_oracle.py --samples 2000 --batch-size 128
```

测试、构建和复刻官方逐样本评测：

```powershell
& $python -m pytest
& $python scripts/build_submission.py
& $python scripts/evaluate_submission.py --samples 2000
```

最终提交包为 `artifacts/FATE_MIMO_submission.zip`。本地固定测试结果写入 `artifacts/final_results.json`；这些结果不等同于线上排行榜成绩。
