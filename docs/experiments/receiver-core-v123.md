# V123 Receiver 核心参数复扫

日期：2026-08-24  
分支：`feature/receiver-core-v123`  
对照：V123

seed1176、300 样本单参数扫描：

- `PILOT_STEERING_SHRINKAGE=0/0.025/0.05/0.075/0.1`：0.075 在
  seed1176 微升，但 seed1177/1178 大幅掉 P10，淘汰；
- `PILOT_GAIN_REFINEMENT_RATE=0.25/0.375/0.4375/0.5`：全部低于默认
  0.3125；
- `PILOT_GAIN_REFINEMENT_ITERATIONS=3/5/6`：全部低于默认 4；
- `DATA_VECTOR_REFINEMENT_SCALE=0.1/0.15/0.25/0.3`：全部低于默认
  0.2；
- `DATA_GAIN_REFINEMENT_SCALE=0.1/0.15/0.25/0.3`：全部低于默认 0.2；
- `DATA_GAIN_REFINEMENT_RADIUS=1/3/4`：全部低于默认 2；
- `DATA_GAIN_REFINEMENT_ITERATIONS=2/3/5/6`：全部低于默认 4。

本轮没有稳定候选。V123 的导频估计与数据增益/向量更新参数均处于清晰
局部最优，保持不变。
