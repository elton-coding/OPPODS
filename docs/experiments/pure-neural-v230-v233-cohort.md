# V230—V233：冲击69分的单因素实验组

## 目标与证据边界

当前已复核的V227为本地68.154844，而用户已反馈的线上结果是V223的67.01164987745。二者不能直接相减解释提升。目标仍是超过69；训练验证峰值、既有audit与未参与选型的确认集分开记录，线上分数必须由实际提交确认。

参考本仓库[2025获奖视频精读](../research/2025获奖方案视频精读.md)中的分区训练、评分代理、条件注入及控制语义。本轮保持纯神经黑盒，不采用该旧文中对物理主线的历史建议。

## 实验矩阵

所有组从V227严格继承，固定共享Encoder，仅训练Tx/Rx；相同seed15240、batch100、Adam1e-5、12000步、每1000步验证2000通道。双用户SNR独立均匀[-20,20]。不使用训练集之外的数据更新权重。

| 组 | 唯一方法变化 | 输出目录 | 状态 |
|---|---|---|---|
| V230-control | 无：两专家同预算续训 | artifacts/pure_neural_v230/two_control_clean | 12000步完成，audit中 |
| V230-eight | 2→8个联合Tx/Rx专家，每段5dB，按最低SNR选组 | artifacts/pure_neural_v230/eight | 训练中 |
| V231 | 软判决温度0.5→0.1 | artifacts/pure_neural_v231/temperature01 | 训练中 |
| V232 | H与SNR驱动的通用神经特征缩放/偏移 | artifacts/pure_neural_v232/context | 排队 |
| V233 | 5bit内传模式及双用户SNR之和，Rx加入伙伴SNR嵌入 | artifacts/pure_neural_v233/partner | 排队 |
| V234（后续） | 反馈驱动的Tx比特特征神经门控 | artifacts/pure_neural_v234/feedback | 代码就绪，尚未训练 |

八专家映射父模型[0,0,1,1,1,1,1,1]；其他组映射[0,1]。新增门控/嵌入的输出初始化为零，非新增参数全部严格加载。各组起始数值经过检查；分组矩阵乘法允许浮点级差异，不允许随机重置核心权重。

## 资源处理与公平性

首次两专家对照由于并行显存争用被中断，保存至two_control，只作诊断记录，不参与最终消融。完整对照从原父模型重跑至two_control_clean，不拼接中断检查点。

两步探针通过后，完整对照及其后三项设置CUDA分配器上限0.4（约13GB），不改变批量、精度、数据、优化器或损失。重跑对照的0/1000/2000/3000步验证值与中断前完全一致。八专家原进程继续，未修改其训练状态。最多同时两项GPU训练，不再启动第三项训练争抢显存。

## 评测与晋级

1. 固定validation选择每组checkpoint；训练结束的training_report.json完整保留。
2. 运行scripts/audit_pure_neural_candidate.py：固定split1176、test offset2000、noise22701/22702/22703，官方兼容逐样本评测。
3. 记录模型源文件及三份权重的SHA256；每次评测后核对未变化；拒绝覆盖既有audit标签。先评control，再评候选并指定control-label。
4. 同时计算相对V227与同预算control的配对差值和信道聚类bootstrap区间。主指标为总分，同时披露P10、效率和SNR分箱，不挑单个噪声种子的最高值。
5. 候选接近/超过69后，冻结模型，在尚未用于选型的test后续窗口和新噪声种子复核；不据确认集继续调参后仍称其“盲测”。
6. 晋级前检查模型接口、控制bit数、ZIP小于1GB及推理耗时。只有验证更优且符合约束的权重进入main。

目前无新候选通过上述完整流程，不能声称已经超过69。

两专家完整对照耗时1571.5秒，最佳checkpoint为11000步，固定validation为68.174618（起点68.023285）。训练曲线并不单调。候选ZIP为artifacts/FATE_MIMO_submission_pure_neural_v230_control_official.zip，178999997字节，SHA256=6C59BCB2A1A77C0443325D64EDFBB9A1DBCD23420A671DF7F495DB84AE6009B3。尚未替换canonical提交包。
