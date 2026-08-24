# V123 Walsh 列偏移扫描

日期：2026-08-24  
分支：`feature/walsh-column-offset-v123`  
对照：V123（column offset 0）

增加 `THRESHOLD_WALSH_COLUMN_OFFSET`，在保持 start65 连续行组不变的
情况下，将 228 个尾部 bit 映射到循环平移后的 Walsh 列。默认 offset0
与 V123 精确一致。

seed1176、300 样本扫描 offset
`1/2/3/4/5/7/8/11/16/31/32/64/128`：

- V123 offset0：`62.420218`；
- 非等价候选范围：`62.301874–62.396202`，全部退化；
- offset128 与 offset0 完全一致，因为 start65–80 行不包含 Walsh 第 8
  位，增加 128 不改变这些行的奇偶内积。

没有候选进入多 seed。列偏移接口保留在特性分支，不合入 `main`。
