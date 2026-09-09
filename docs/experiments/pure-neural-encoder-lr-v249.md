# V249：只提高共享Encoder学习率

预登记：2026-09-09 12:40。当前尚无GPU正式效果分数；分支codex/pure-neural-encoder-lr-v249继承6d599be，不改在训V242/V247或排队V248的绑定依赖。

## 假设与单因素对照

V235解冻共享Encoder并与两组Tx/Rx共同训练，同12k均分68.311847，相对冻结Encoder的C仅-0.002052，区间跨0；未证明解冻有益，也没有证明编码器无法进一步优化。V249检验编码器在已有收发网络旁是否需要不同的适配步长，不把解冻默认当成正增益。

对照是V235，而不是只与冻结C比较。两者均从原V227重新初始化，Encoder和全部Tx/Rx可训练，总48191288参数；唯一新超参数是Encoder Adam学习率1e-5→5e-5，Tx/Rx仍1e-5。模型、两专家mapping[0,1]、k8/B1152、seed15240、batch100、12k/120万信道抽样、FP32、soft_score温度0.5/P10权重0.3/带宽0.025/BCE0.05、全SNR、validation2000每1000步、patience12均同V235。

仍是同一个Adam，只将参数分成Encoder803392个、Tx/Rx47387896个两组。动量系数、eps、weight_decay、每步更新频率及统一全参数梯度裁剪不变，不交替优化，不更改噪声或batch。由于训练会相互影响，后续Tx/Rx梯度轨迹可以变化；不能声称训练后Tx/Rx权重与V235保持相同。本实验不是高学习率与大批次/RMS/温度的组合，也不从V247或V242权重接着训练。

## 实现与检查

隔离入口train_pure_neural_encoder_lr_v249.py仅在自身进程捕获实际Encoder参数并构建两个Adam参数组；退出或异常恢复原函数，不修改共享训练器和推理模型文件。报告中learning_rate保留收发网络的1e-5，并显式记录encoder_learning_rate、transceiver_learning_rate及两组参数量；不会把所有参数误标成同一个学习率。

单测覆盖命令与V235一致（仅训练入口不同）、参数分组无重复无遗漏、相同合成梯度下Tx/Rx首次Adam更新精确一致而Encoder更新为5倍、异常恢复、完整12k与真实参数组元数据门禁。CPU真实V227父模型batch2、两步完整训练通过，输出artifacts/resource_probe/v249/cpu2，证据benchmarks/v249_cpu_runtime_probe.json；它只验证运行与元数据，不以4通道分数证明效果，也不代替GPU完整batch100资源检查。CPU检查没有占第三个GPU训练槽。

## 顺序队列和判定

运行器run_pure_neural_encoder_lr_v249.py等待V247完整3000步三噪声audit释放其训练槽，再从原V227做GPU batch100两步探针，然后独立正式12k。等待不占GPU，和接续V242的V248各占一个训练槽，保持最多两项GPU训练。若前驱本地>=69则暂缓该新实验，先冻结确认；等待失败不删目录重跑。

正式输出artifacts/pure_neural_v249/encoder_lr5，探针artifacts/resource_probe/v249/encoder_lr5，audit标签v249_encoder_lr5。独占锁、已存在输出拒绝续训、源文件与原父权重哈希绑定，启动时验证与CPU探针相同源码；GPU报告须有准确两组参数数目和学习率，正式起点四项validation须匹配V235。

完整训练后自动官方兼容batch1，固定split1176/offset2000/noise22701、22702、22703每噪声2000通道复评，对V235计算同预算单因子效应，同时与发布冠军V240C比较。如果冠军变化，晋级前补最新冠军对照。收益必须经完整审计，不以较高训练验证点代替，不直接加到RMS/专家数/温度上。

保持V243祖先与历史评测交叉的口径限制：共享同父模型的消融不等于全谱系独立留出已获证；当前audit经过反复选型，不能叫全项目未见盲测。若接近69先冻结再登记确认规则，最终线上69需要实际提交结果。
