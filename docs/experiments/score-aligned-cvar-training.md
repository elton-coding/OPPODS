# Score-aligned CVaR 与弱用户 SNR 定向训练实验

日期：2026-08-22  
分支：`feature/score-aligned-cvar-loss`  
主干冠军：V117（seed 1176 / 2000 样本 final `62.952218`）

## 目标

参考 2025 年获奖方案中反复出现的 SNR 分段、评分代理损失与尾部样本优化，在 V117 去噪器检查点
`sparse_denoiser_task_reg000439453125_rec08_ft1e5.pt` 上尝试两条路线：

1. 把原先写死的全用户 `log(1+SINR)` CVaR 改成可配置的层级公平目标；
2. 让完整生产版 Transmitter 和 Receiver 进入训练图，直接用官方 bit-score 的平滑代理反传。

所有检查点与候选提交包保存在本地忽略目录 `checkpoints/` 和 `artifacts/candidates/`。代码、测试和本报告纳入 Git；失败候选不污染 `modelSubmit/`，也不晋级 `main`。

## 实现

### 层级公平与 SNR 定向采样

`scripts/train_precoder_aware_denoiser.py` 新增：

- 可配置 `tail_alpha` 和 `tail_target`；
- 每样本 weakest-user 平均效用；
- 指定一个弱用户落入 `[-15.5,-9.5] dB`，并可把另一用户限制在更高 SNR；
- 训练分布可定向采样，验证仍使用赛题官方 `[-20,20] dB` 均匀分布。

### 直接 soft bit-score

`scripts/train_score_aligned_denoiser.py` 使用 V117 原始生产链生成反馈、下行信号和 LLR。每个样本的训练分数为：

`50 + 100 / 1152 * sum(sigmoid(target_sign * LLR / temperature) - 0.5)`

总目标按官方结构混合 mean score 与 lower-CVaR，并只更新反馈去噪器。1-step 诊断得到有限的 `grad_norm=0.388`，确认完整链路可反传。

### Checkpoint interpolation

`scripts/interpolate_denoiser_checkpoints.py` 对基准与候选 state dict 做兼容性检查和线性插值，用于判断训练差分能否作为局部修正叠加到 V117。

## 固定 seed 1176 短评结果

评测命令统一使用 `scripts/evaluate_submission.py --samples 500 --seed 1176`。同口径 V117 为：

- efficiency `67.920399`
- P10 `50.520833`
- final `62.700530`

| 候选 | 核心设置 | efficiency | P10 | final | 对 V117 |
|---|---|---:|---:|---:|---:|
| A | weakest CVaR 10%，定向采样 50%，500 step | 67.895486 | 50.598958 | 62.706528 | +0.005998 |
| B | weakest mean 40%，显式弱/强用户，750 step | 67.904688 | 50.598958 | 62.712969 | +0.012439 |
| C | B 继续训练至累计 2000 step | 67.893229 | 50.694444 | 62.733594 | +0.033064 |
| Soup 25% | V117 75% + C 25% | 67.906597 | 50.607639 | 62.716910 | +0.016380 |
| Soup 50% | V117 50% + C 50% | 67.892014 | 50.607639 | 62.706701 | +0.006172 |
| Soup 75% | V117 25% + C 75% | 67.896962 | 50.598958 | 62.707561 | +0.007031 |
| D | 直接 soft score，累计 100 step | 67.897743 | 50.607639 | 62.710712 | +0.010182 |
| E | D 继续训练至累计 500 step | 67.897917 | 50.520833 | 62.684792 | -0.015738 |

## 长评否决

短评最好的 C 进入 seed 1176 / 2000 样本复评：

| 模型 | efficiency | P10 | final | 差值 |
|---|---:|---:|---:|---:|
| V117 | 68.242752 | 50.607639 | 62.952218 | — |
| C | 68.205208 | 50.607639 | 62.925938 | -0.026280 |

C 的短评 P10 增益没有扩展到 2000 样本，反而因 efficiency 下降被正式口径否决。因此没有继续运行另外两个 seed，也没有生成 V118 正式版本。

## 结论

1. V117 的反馈去噪器已接近这一路线的局部最优。重新分配梯度能移动少量边缘样本，但不足以稳定改变正式 P10 排序位置。
2. 极低 SNR weakest-user 的 `log(1+SINR)` CVaR 约为 `0.003`，物理上几乎没有有效梯度；必须隔离目标 SNR 区间，不能直接让全范围最差样本支配训练。
3. 直接 soft bit-score 可以通过完整收发链反传，但 batch 仅 4 时 tail score 长期约为 50，方差较大；100-step 的小幅正增益在 500 step 后消失。
4. checkpoint interpolation 未保留 C 的短评 P10 跳变，说明差分不是可线性叠加的稳定局部修正。

本实验分支保留训练能力和复现实证，当前冠军继续为 V117，`main`、`modelSubmit/` 和正式提交包均不变。下一次高成本训练应转向独立的分 SNR Receiver 专家或更大 batch 的离线 score replay，而不是继续微调同一个反馈去噪器。
