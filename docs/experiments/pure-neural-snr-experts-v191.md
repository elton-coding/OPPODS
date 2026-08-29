# V191：赛题 baseline 纯神经 SNR 专家主线

- 日期：2026-08-29
- 分支：`codex/pure-neural-snr-experts-v191`
- 状态：首轮完成，未晋级 `main`
- 对照：赛题方纯神经 baseline；V190 仅作为最终冠军门槛

## 1. 假设

赛题方 baseline 使用统一 Encoder、反馈 Decoder、神经调制、Transformer Precoder 和 Transformer Receiver 覆盖整个 `[-20,20] dB`。2025 年多支获奖队伍表明不同 SNR 的容量、反馈可靠性、干扰结构和最优网络早停点不同。本实验检验：在完全不加入 Wiener、RZF、QAM、导频、显式均衡或干扰抵消等物理模块的条件下，独立 SNR 网络能否提高官方 exact final。

## 2. 架构

以赛题方 baseline 为唯一网络原型，复制为八个 5 dB 专家：

```text
[-20,-15)、[-15,-10)、[-10,-5)、[-5,0)、
[0,5)、[5,10)、[10,15)、[15,20]
```

- Encoder：按本 UE SNR 选择专家；
- Transmitter：按两 UE 最小 SNR 选择一套完整的 Decoder、两用户神经调制器和 Transformer Precoder；
- Receiver：按本 UE SNR 选择专家；
- 控制字段：第一阶段保持赛题 baseline 的固定全 1，不引入人工通信语义；
- 输出：固定 1152 logits，不使用手工输出门控。

两 UE 落入不同区间时，较弱用户决定共享 Transmitter 专家。这是保证共享发射策略唯一性的路由规则，不改变专家内部的黑盒神经结构。独立预训练后增加全范围混合 SNR 联合校准，使高 SNR Encoder/Receiver 能适应由较低 SNR Transmitter 管理的非对称样本。

## 3. 训练阶段

### 阶段 A：零增量复制

把赛题方三个权重文件逐套复制到八个专家。官方 batch-size-one 接口下，复制银行必须与原 baseline 的 Encoder、Transmitter、Receiver 逐元素一致。这一阶段用于证明路由本身不改变基线。

### 阶段 B：逐区间独立预训练

每次只解冻一个区间对应的 Encoder、Transmitter 和 Receiver，两个 UE 的 SNR 独立采样于同一 5 dB 区间。初始计划每段 750 step，Adam、学习率 `1e-4`，以固定 channel-held-out validation 的 exact final 早停并保存最优点。

### 阶段 C：混合 SNR 联合校准

两个 UE 独立均匀采样 `[-20,20] dB`，解冻全专家银行，重点修复跨区间组合时的 Encoder–Transmitter–Receiver 接口失配。该阶段必须相对阶段 B 做单变量消融。

## 4. 预注册对比

| 组别 | 权重 | 路由 | 目的 |
| --- | --- | --- | --- |
| C0 | 赛题方单模型 | 无 | 原始参考 |
| C1 | 八份完全相同的 baseline 权重 | 5 dB 路由 | 验证路由零增量 |
| A | 八个独立区间微调权重 | 5 dB 路由 | 检验分区训练 |
| B | A 后混合 SNR 联合校准 | 5 dB 路由 | 检验跨专家兼容性 |

阶段 A/B 不与 V190 混入任何因子。只有纯神经方案本身在固定三 seed 平均分超过 V190，才允许复制到 `modelSubmit/` 并晋级 `main`。

## 5. 评测与门禁

1. 训练使用 deterministic train split，早停使用 validation split；
2. 参数冻结后才运行 test split 的 discovery seeds；
3. 先逐区间报告 efficiency、P10、final，再报告全范围组合；
4. 全范围至少评测 seed1176/1177/1178，并检查每个 5 dB 区间和双 UE 非对称组合；
5. 训练选模 seed 与冻结盲测 seed 分离；
6. 纯神经路线未超过 V190 前只保存在本特性分支和 `artifacts/pure_neural_v191/`。

## 6. 首轮结果

### 6.1 同区间独立预训练

八个区间相对各自复制 baseline 的固定 validation final 全部提升，但最佳早停点差异很大：

| 区间 | 初始 final | 最佳 final | 增量 | 最佳 step |
| --- | ---: | ---: | ---: | ---: |
| `[-20,-15)` | 52.189841 | 52.801310 | +0.611469 | 100 |
| `[-15,-10)` | 53.659077 | 54.342548 | +0.683471 | 100 |
| `[-10,-5)` | 55.657891 | 56.394883 | +0.736991 | 700 |
| `[-5,0)` | 58.210348 | 58.978238 | +0.767890 | 200 |
| `[0,5)` | 60.878777 | 61.969117 | +1.090341 | 750 |
| `[5,10)` | 63.032341 | 64.331023 | +1.298682 | 750 |
| `[10,15)` | 64.789346 | 65.709210 | +0.919864 | 500 |
| `[15,20]` | 65.859832 | 66.215397 | +0.355565 | 600 |

最低 BCE 权重并不等于最高 exact final，例如第一段在 step100 达到最高 final，之后 BCE 继续下降而 final 回退。因此每段独立 early stopping 是必要的。

### 6.2 跨区间失配和联合校准

只组合八个独立预训练专家时，seed1176/2000 全范围 final 仅 `60.390805`。两个 UE 落在不同区间时，高 SNR Encoder 输出会进入弱用户选择的另一套 Transmitter，独立预训练没有见过该接口组合。

使用两个 UE 独立均匀 SNR 做 1000-step 联合校准后，mixed validation 从 `60.520908` 提升到 `62.223060`，最佳点为 step800。seed1176 exact 提升到 `62.265143`。

同 seed、同随机流下，赛题方 baseline 为 `61.997695`，所以基础校准专家净增益为 `+0.267448`。

### 6.3 区间回退 C/A/B/AB

seed1176 逐 UE 诊断中 `[0,5)` 和 `[15,20]` 均值局部为负，因此分别把这两段回退到独立预训练权重：

| 组合 | final | 相对完整校准 |
| --- | ---: | ---: |
| C：完整联合校准 | 62.265143 | — |
| A：回退 `[0,5)` | 62.168832 | -0.096311 |
| B：回退 `[15,20]` | 62.062739 | -0.202405 |
| AB：两段同时回退 | 61.867975 | -0.397168 |

所有回退均下降，说明单 UE 局部分数不能脱离双 UE 交互直接拼装专家。

### 6.4 最差 10% 链路加权

在基础校准权重上继续使用 `tail_weight=0.5`、`tail_fraction=0.1`、学习率 `1e-5` 训练，以每个 batch 中 BCE 最大的 10% UE 链路构造 CVaR 风格附加项。固定三 seed 结果：

| seed | 基础校准 | tail-weight | 增量 |
| ---: | ---: | ---: | ---: |
| 1176 | 62.265143 | 62.274709 | +0.009566 |
| 1177 | 62.344329 | 62.403027 | +0.058698 |
| 1178 | 62.221332 | 62.268409 | +0.047077 |
| 均值 | **62.276935** | **62.315382** | **+0.038447** |

tail-weight 在三个 seed 均为正，成为 V191 纯神经路线内部最佳，但仍低于 V190 的 `63.217950`，差距约 `0.902568`，因此不晋级主干。

## 7. 工程状态

- 专家总参数量：17,711,712；
- 初始化权重约 72 MB，低于赛题 1 GB 上传限制；
- 三个模块的 batch-size-one baseline 等价回归通过；
- 八个边界路由和跨专家梯度回归通过；
- 2-step 端到端训练烟测通过，loss 可下降；
- 纯神经最佳权重：`artifacts/pure_neural_v191/tail05_modelSubmit/`；
- 下一项单变量实验：把 Transmitter 的反馈 Decoder 和神经调制器改为按本 UE SNR 路由，仅共享 Precoder 使用双 UE 最小 SNR 路由，减少跨专家接口失配。

复现入口：

```powershell
$python = 'D:\Tools\Anaconda\envs\oppods-df1176\python.exe'

& $python scripts/train_pure_neural_snr_experts.py --stage initialize

0..7 | ForEach-Object {
    & $python scripts/train_pure_neural_snr_experts.py `
        --stage pretrain --expert-index $_ --steps 750 --batch-size 32 `
        --validate-every 100 --validation-samples 256 --seed (1191 + $_)
}

& $python scripts/train_pure_neural_snr_experts.py `
    --stage calibrate --steps 1000 --batch-size 32 `
    --validate-every 100 --validation-samples 512 --seed 2191
```

训练权重与详细 JSON 保存在 Git 忽略的 `artifacts/pure_neural_v191/`，实现模板位于 `research/pure_neural_v191/modelDesign.py`。
