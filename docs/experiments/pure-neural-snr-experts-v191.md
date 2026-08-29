# V191：赛题 baseline 纯神经 SNR 专家主线

- 日期：2026-08-29
- 分支：`codex/pure-neural-snr-experts-v191`
- 状态：实现完成，训练进行中
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

## 6. 当前工程状态

- 专家总参数量：17,711,712；
- 初始化权重约 72 MB，低于赛题 1 GB 上传限制；
- 三个模块的 batch-size-one baseline 等价回归通过；
- 八个边界路由和跨专家梯度回归通过；
- 2-step 端到端训练烟测通过，loss 可下降；
- 首轮正式逐区间训练尚在进行。

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
