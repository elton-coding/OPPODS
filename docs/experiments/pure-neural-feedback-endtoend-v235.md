# V235：共享Encoder解冻单因素（预登记）

## 假设

V227只训练低模式时冻结共享Encoder，有利于保持其他模式不变。V230C已经同时训练两组Tx/Rx，但Encoder仍被冻结。本实验检验允许反馈编码与两组收发模型重新共同适配是否有净收益；不增加物理模块、不增加模型容量、不改变SNR路由。

## 严格对照

对照为已完成的V230C，初始化仍使用同一个V227父模型，而不是直接从新冠军多训练后归因于解冻。唯一变化是train-components由transmitter receiver变为encoder transmitter receiver；其他配置与V230C完全一致：

- design：research/pure_neural_v227/modelDesign.py；父模型：artifacts/pure_neural_v227/joint_low，映射[0,1]；
- 全SNR均匀独立采样，seed15240，batch100，Adam1e-5，12000步；
- 固定validation 2000通道，每1000步验证，patience12；
- soft_score，T0.5、P10权重0.3、tail_fraction0.1、带宽0.025、BCE0.05；
- CUDA分配器上限0.4，FP32；不与八专家、温度或门控改动组合。

可训练参数由47387896增至48191288，推理参数数量、接口和ZIP规模不变。输出目录artifacts/pure_neural_v235/end_to_end。

## 执行状态与判定

代码使用现有经过测试的训练组件选择功能；另以端到端反向测试确认解冻后Encoder确实得到梯度，而不仅是将名字加入优化器。正式训练将在GPU训练槽释放后进行，不抢占或重启当前两个正常运行的任务。

训练完成后固定validation选点，使用scripts/audit_pure_neural_candidate.py指定baseline-label v230_control（当前冠军）进行三噪声配对audit。若超过69，再冻结模型做尚未用于选型的确认窗口，线上成绩单独记录。

状态：预登记，尚未训练；不能声称有效。
