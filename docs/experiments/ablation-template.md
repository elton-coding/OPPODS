# VXXX 消融实验记录模板

日期：YYYY-MM-DD  
分支：`codex/<topic>-vXXX`  
对照版本/提交：VXXX / `<commit>`  
实验提交：`<commit>`  
状态：进行中 / A / D / B / N

## 1. 假设

用一句可证伪的话说明：为什么这个改动应在哪类信道上改善 efficiency 或 P10。

## 2. 因子定义

- 因子族：`CTRL / OUT / TX-RZF / TX-WIENER / RX-COV / RX-IC / RX-DG / RX-DV / RX-PG / RX-PILOT / RX-RES`
- 唯一自变量：
- 对照值：
- 候选值：
- 作用对象：单 UE / 任一 UE / 全部 UE / 整个信道对
- SNR 区间：`[low, high)`
- 保持不变的控制变量：
- 依赖的已晋级特性：
- 可能冲突或交互的特性：

如果同时改变多个参数，本实验不是单变量消融，必须拆分为 C/A/B/AB 因子设计。

## 3. 预注册评测协议

- smoke seed/样本数：
- discovery seeds/样本数：
- 每 seed 分块数：
- blind seeds：冻结时保持未知
- 主指标：`0.7 × efficiency + 0.3 × P10`
- 晋级门禁：每 discovery seed、每块、P10 非负；冻结后每 blind seed、P10 非负
- 冻结提交：
- 明确禁止：根据 blind 结果修改区间、阈值或候选值

## 4. 结果

| 阶段 | seed | baseline efficiency | candidate efficiency | baseline P10 | candidate P10 | final 增量 | 最差块增量 | 变化 UE 正/负 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | | | | | | | | |
| discovery | | | | | | | | |
| blind | | | | | | | | |

补充记录 SNR 分箱结果，并注明任何 final 大变化是否来自 P10 离散跳档。

## 5. 组合消融（如适用）

| 组 | A | B | efficiency | P10 | final | 相对 C | 交互项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 0 | 0 | | | | 0 | — |
| A | 1 | 0 | | | | | — |
| B | 0 | 1 | | | | | — |
| AB | 1 | 1 | | | | | `AB-A-B+C` |

## 6. 决策

- 结论：晋级 / 发现集拒绝 / 盲测拒绝 / 无效
- 违反或满足的门禁：
- 是否进入组合候选池：
- 可安全复用的结论：
- 不应重复的搜索：

## 7. 可复现性

- 评测命令：
- 数据与 split：
- 环境：`D:\Tools\Anaconda\envs\oppods-df1176`
- 分数归档：`artifacts/diagnostics/<experiment>/`
- 模型/配置哈希：
- 提交 ZIP 与 SHA256（仅晋级版本）：
- 测试与 Ruff 结果：
