# V171 924→1056 位扩展门控六 seed 重搜（拒绝）

日期：2026-08-25  
分支：`codex/extension-six-seed-v171`  
对照：V156

## 方法

复用 seed1176/1177/1178/2098/2099 的全量 extension diagnostics，并为 seed2120 新生成 V156 exact reference、924 位 fallback、1056 位 extension 反事实及置信度特征。搜索集合共 6 个 seed、24,000 个 UE 分数，所有 seed 再切前后半块。

候选包括：

- `mean_abs/median_abs/q25_abs/q75_abs/std_abs`；
- 三种 clipped mean 及多个比值/差值派生特征；
- 全局置信度阈值；
- 7 个 2.5 dB 中间 SNR 分箱阈值；
- 65 分位阈值网格；
- 相对当前 V156 exact reference 的整 seed与半块 final 非负门禁。

V156 当前 7 个基础分箱阈值 `0.246/0.246/0.246/0.124/0.101/0.260/0.296287667` 作为搜索起点；reference 还包含 V156 的两个细区间门控。

## 结果

```text
top_global = []
top_binned = []
```

没有任何全局统计/阈值组合，也没有任何按 2.5 dB 分箱的坐标下降策略，能够同时在 6 个 discovery seed 和 12 个半块上不低于 V156。旧 V149 fallback 在不同 seed 相对 V156 有正有负，不能通过简单重置门控获得稳定收益。

结论：V171 无可晋级候选，不进入真正盲测、不修改模型、不构建包、不合并 `main`；V152/V153/V156 的 924→1056 位门控保持不变，冠军继续为 V156。
