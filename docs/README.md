# OPPODS 文档索引

文档按职责分类，避免在仓库根目录继续生成带“最终版”“最新版”含义不清的文件。

## 综合报告

- [V250 八专家RMS长预算](experiments/pure-neural-long-budget-v250.md)：当前本地冠军，72k固定audit均分68.606029；[冻结确认](experiments/frozen-confirmation-v250.md)对V242E净增0.088250，仍未达69、线上未确认；[提交说明](submission/V250_提交说明.md)。
- [V242 RMS×长预算](experiments/pure-neural-rms-budget-v242.md)：历史冠军V242E，八专家36k均分68.531625，完整损失×预算消融保留。
- [V251 公平损失权重](experiments/pure-neural-fairness-v251.md)：68.518753，净收益未获支持，不采纳。
- [V252 近等频SNR分段](experiments/pure-neural-balanced-routing-v252.md)：68.483786，相对V242E配对区间为负，不采纳。
- [V253 联合min/max SNR分段](experiments/pure-neural-pair-routing-v253.md)：完整36k固定均分68.562225；优于同概率V252，但对同预算V242E净增区间跨0，低于现冠军V250，不晋级。
- [V254 十六个2.5dB联合专家](experiments/pure-neural-sixteen-experts-v254.md)：原父模型完整72k重训中，另预登记专家数×存储精度四组消融；八专家half存储CPU转换完成但尚未评分，不以超限完整精度包或源模型分数替代转换后的验证。
- [V255 后半程余弦学习率](experiments/pure-neural-cosine-v255.md)：仅将固定学习率改为前36k保持、后36k衰减；CPU完整72k更新证明及GPU初始检查已通过，正式独立72k训练中，不同时改变Adam/损失/专家架构，尚无新成绩。
- [V256 RMS温度减半](experiments/pure-neural-rms-temperature-v256.md)：相对V250只改归一化评分代理温度0.5→0.25，恒定学习率和72k预算不变；9项专项、完整224通过10跳过及Ruff通过，待排队，尚未训练或评分。
- [V240C 长训练预算](experiments/pure-neural-long-budget-v240.md)：历史冠军，两专家36k固定audit均分68.439628；[提交说明](submission/V240C_提交说明.md)。
- [V241 专家数×RMS损失](experiments/pure-neural-eight-rms-v241.md)：完整组合audit68.395108，历史短预算对照。
- [V244 跨子载波残差](experiments/pure-neural-token-context-v244.md)：完整audit68.299965，未证明净收益，不采纳。
- [V230E 八段SNR专家](experiments/pure-neural-eight-snr-v230.md)：历史冠军，固定留出集均分68.360464。
- [V230C 同预算续训](experiments/pure-neural-continuation-v230c.md)：同预算对照与历史冠军，固定留出集均分68.313899。
- [V230—V234实验组](experiments/pure-neural-v230-v233-cohort.md)：八专家、温度、接收端条件、伙伴SNR、发射端门控的独立尝试。
- [V227 联合 SNR 专家](experiments/pure-neural-joint-snr-v227.md)：历史冠军与本轮初始化，固定留出集均分68.154844。
- [V229 四组组合消融](experiments/pure-neural-joint-hard-rank-v229.md)：组合不超过单独联合专家；不得相加单项收益。
- [V224 评分代理损失](experiments/pure-neural-soft-score-loss-v224.md)：历史初始化版本，修订后固定留出集均分68.069933。
- [V228 硬排名损失消融](experiments/pure-neural-hard-rank-loss-v228.md)：同初始化、同样本和预算的损失对照。
- [评测划分审计](experiments/evaluation-partition-audit-v227.md)：旧种子复评的训练通道交叉问题及修订协议。
- [V243确认窗口交叉审计](experiments/confirmation-overlap-audit-v243.md)：1590个归档的通道ID清查；后续窗口与当前训练不交叉，但大部分曾被早期评测，不能称全项目未见盲测。
- [FATE-MIMO V223 纯神经 k8/1152 技术与消融报告](experiments/pure-neural-adaptive-k8-v223.md)：历史版本，用户回报线上67.01164987745；历史本地三种子均分67.242613不属于固定留出协议。
- [FATE-MIMO V190 技术方案与消融实验总报告](reports/FATE-MIMO_V190_技术方案与消融实验总报告.md)：历史混合方案、SNR 专家与消融记录。

## 赛题资料

- [DataFountain 1176 完整原文](competition/DataFountain_1176_完整原文.md)

## 方案与提交

- [FATE-MIMO 初始技术方案](solutions/FATE-MIMO_v1.md)
- [FATE-MIMO V115 冻结方案](solutions/FATE-MIMO_v115.md)
- [FATE-MIMO V117 冠军增量方案](solutions/FATE-MIMO_v117.md)
- [FATE-MIMO V118 SNR 分段 Receiver 方案](solutions/FATE-MIMO_v118.md)
- [FATE-MIMO V119 SNR 分段导频插值方案](solutions/FATE-MIMO_v119.md)
- [FATE-MIMO V120 弱用户输出策略](solutions/FATE-MIMO_v120.md)
- [FATE-MIMO V121 控制码字预算方案](solutions/FATE-MIMO_v121.md)
- [FATE-MIMO V122 SNR 分段匹配前缀方案](solutions/FATE-MIMO_v122.md)
- [FATE-MIMO V123 Walsh 尾部码本方案](solutions/FATE-MIMO_v123.md)
- [FATE-MIMO V124 置信度前缀专家方案](solutions/FATE-MIMO_v124.md)
- [FATE-MIMO V125 SNR 分箱截断置信度方案](solutions/FATE-MIMO_v125.md)
- [FATE-MIMO V126 裸前缀语义扩展方案](solutions/FATE-MIMO_v126.md)
- [FATE-MIMO V127 Walsh 尾部置信度再校准方案](solutions/FATE-MIMO_v127.md)
- [FATE-MIMO V128 双 UE SNR 分段 RZF 专家方案](solutions/FATE-MIMO_v128.md)
- [FATE-MIMO V129 双 UE 多区间 RZF 专家方案](solutions/FATE-MIMO_v129.md)
- [FATE-MIMO V130 极高 SNR RZF 子专家方案](solutions/FATE-MIMO_v130.md)
- [FATE-MIMO V131 超高 SNR RZF 强正则专家方案](solutions/FATE-MIMO_v131.md)
- [FATE-MIMO V132 高 SNR Receiver 协方差专家方案](solutions/FATE-MIMO_v132.md)
- [FATE-MIMO V133 高 SNR Receiver 协方差加载增强方案](solutions/FATE-MIMO_v133.md)
- [FATE-MIMO V134 分层高 SNR Receiver 协方差专家方案](solutions/FATE-MIMO_v134.md)
- [FATE-MIMO V135 中档 Receiver 协方差专家方案](solutions/FATE-MIMO_v135.md)
- [FATE-MIMO V137 峰值 SNR Receiver 协方差专家方案](solutions/FATE-MIMO_v137.md)
- [FATE-MIMO V138 极高双用户 SNR RZF 专家方案](solutions/FATE-MIMO_v138.md)
- [FATE-MIMO V139 超峰值双用户 SNR RZF 专家方案](solutions/FATE-MIMO_v139.md)
- [FATE-MIMO V143 低中 SNR 配对 RZF 重校准方案](solutions/FATE-MIMO_v143.md)
- [FATE-MIMO V147 SNR 分段软干扰抵消专家方案](solutions/FATE-MIMO_v147.md)
- [FATE-MIMO V149 数据增益温度专家方案](solutions/FATE-MIMO_v149.md)
- [FATE-MIMO V152 高中间 SNR 扩展门控方案](solutions/FATE-MIMO_v152.md)
- [FATE-MIMO V153 超高中间 SNR 子区间门控方案](solutions/FATE-MIMO_v153.md)
- [FATE-MIMO V156 内部上中间 SNR 门控方案](solutions/FATE-MIMO_v156.md)
- [FATE-MIMO V189 弱用户保护 Wiener 专家方案](solutions/FATE-MIMO_v189.md)
- [FATE-MIMO V190 安全 Wiener 与置信度 IC 组合方案](solutions/FATE-MIMO_v190.md)
- [V115 提交说明](submission/V115_提交说明.md)
- [V117 提交说明](submission/V117_提交说明.md)
- [V118 提交说明](submission/V118_提交说明.md)
- [V119 提交说明](submission/V119_提交说明.md)
- [V120 提交说明](submission/V120_提交说明.md)
- [V121 提交说明](submission/V121_提交说明.md)
- [V122 提交说明](submission/V122_提交说明.md)
- [V123 提交说明](submission/V123_提交说明.md)
- [V124 提交说明](submission/V124_提交说明.md)
- [V125 提交说明](submission/V125_提交说明.md)
- [V126 提交说明](submission/V126_提交说明.md)
- [V127 提交说明](submission/V127_提交说明.md)
- [V128 提交说明](submission/V128_提交说明.md)
- [V129 提交说明](submission/V129_提交说明.md)
- [V130 提交说明](submission/V130_提交说明.md)
- [V131 提交说明](submission/V131_提交说明.md)
- [V132 提交说明](submission/V132_提交说明.md)
- [V133 提交说明](submission/V133_提交说明.md)
- [V134 提交说明](submission/V134_提交说明.md)
- [V135 提交说明](submission/V135_提交说明.md)
- [V137 提交说明](submission/V137_提交说明.md)
- [V138 提交说明](submission/V138_提交说明.md)
- [V139 提交说明](submission/V139_提交说明.md)
- [V143 提交说明](submission/V143_提交说明.md)
- [V147 提交说明](submission/V147_提交说明.md)
- [V149 提交说明](submission/V149_提交说明.md)
- [V152 提交说明](submission/V152_提交说明.md)
- [V153 提交说明](submission/V153_提交说明.md)
- [V156 提交说明](submission/V156_提交说明.md)
- [V189 提交说明](submission/V189_提交说明.md)
- [V190 提交说明](submission/V190_提交说明.md)
- [V223 提交说明](submission/V223_提交说明.md)
- [V224 提交说明](submission/V224_提交说明.md)

## 实验记录

- [消融实验总表与组合路线](experiments/ablation-registry.md)
- [V191 赛题 baseline 纯神经 SNR 专家主线](experiments/pure-neural-snr-experts-v191.md)
- [V192 纯神经专家组件级 SNR 路由](experiments/pure-neural-component-routing-v192.md)
- [V193 纯神经 SNR 专家输出前缀策略](experiments/pure-neural-prefix-policy-v193.md)
- [消融实验记录模板](experiments/ablation-template.md)
- [2026-08-24 持续优化总结](experiments/2026-08-24-continuous-optimization-summary.md)
- [V212 官方纯神经 baseline 的 B 与 NUM_BITS_PER_RE 从头训练消融](experiments/official-baseline-b-bpr-v212.md)
- [离散 SNR 去噪专家银行](experiments/snr-expert-bank.md)
- [SNR 分段 Receiver 物理参数专家](experiments/snr-receiver-physical-profiles.md)
- [中低 SNR 输出策略扫描](experiments/snr-output-policy-sweep.md)
- [控制码字预算扫描](experiments/control-codeword-allocation.md)
- [V121 中档前缀边界复扫](experiments/v121-middle-prefix-resweep.md)
- [弱用户 Walsh 尾部码本扫描](experiments/threshold-codebook-walsh.md)
- [中档 SNR 置信度前缀扫描](experiments/confidence-prefix-v123.md)
- [V124 中档 SNR 扩展置信度诊断](experiments/confidence-diagnostics-v124.md)
- [V125 SNR 分箱截断置信度门控](experiments/snr-binned-clipped-confidence-v125.md)
- [V126 裸 924 位回退的高 SNR 扩展](experiments/bare-prefix-extension-v126.md)
- [V127 Walsh 尾部置信度再校准](experiments/walsh-tail-confidence-v127.md)
- [V128 双 UE SNR 分段 RZF 专家实验](experiments/snr-pair-rzf-v128.md)
- [V129 双 UE 多区间 RZF 专家实验](experiments/paired-rzf-multiband-v129.md)
- [V130 极高 SNR RZF 子专家实验](experiments/paired-rzf-high-subband-v130.md)
- [V131 超高 SNR RZF 强正则实验](experiments/paired-rzf-ultrahigh-v131.md)
- [V132 高 SNR Receiver 协方差加载专家](experiments/receiver-covload-v132.md)
- [V133 高 SNR Receiver 协方差加载强度搜索](experiments/receiver-covload-high-v133.md)
- [V134 分层高 SNR Receiver 协方差加载专家](experiments/receiver-covload-ultrahigh-v134.md)
- [V135 中档 Receiver 协方差加载专家](experiments/receiver-covload-mid-v135.md)
- [V137 峰值 SNR Receiver 协方差加载专家](experiments/receiver-covload-peak-v137.md)
- [V138 极高双用户 SNR RZF 专家](experiments/paired-rzf-veryhigh-v138.md)
- [V139 超峰值双用户 SNR RZF 专家](experiments/paired-rzf-ultrapeak-v139.md)
- [V143 低中 SNR 配对 RZF 重校准](experiments/paired-rzf-low-v143.md)
- [V147 Receiver 干扰抵消强度 SNR 专家](experiments/receiver-ic-scale-v147.md)
- [V149 Receiver 数据增益温度 SNR 专家](experiments/receiver-data-gain-temperature-v149.md)
- [V152 高中间 SNR 输出扩展稳健重校准](experiments/extension-highbin-threshold-v152.md)
- [V153 超高中间 SNR 输出扩展子区间](experiments/extension-ultrahigh-middle-v153.md)
- [V156 内部上中间 SNR 输出扩展子区间](experiments/extension-inner-upper-v156.md)
- [V188 Wiener × 置信度 IC 因子组合（拒绝）](experiments/wiener-confidence-factorial-v188-rejected.md)
- [V189 弱用户保护 Wiener 专家](experiments/wiener-weak-user-guard-v189.md)
- [V190 安全 Wiener 与置信度 IC 组合](experiments/wiener-confidence-safe-combo-v190.md)
- [V220 24.5M 宽残差 MLP](experiments/pure-neural-wide-resmlp-v220.md)
- [V221 低 SNR 聚焦回放（拒绝）](experiments/pure-neural-low-snr-replay-v221.md)
- [V222 配对 SNR 控制信令（拒绝）](experiments/pure-neural-control-snr-v222.md)
- [V223 纯神经 k8/1152 自适应满载方案](experiments/pure-neural-adaptive-k8-v223.md)
- [V224 可导官方评分代理损失](experiments/pure-neural-soft-score-loss-v224.md)

- [V245 BCE辅助权重单因素消融](experiments/pure-neural-bce-weight-v245.md)
- [V246 大批次全局P10、同样本预算消融](experiments/pure-neural-global-batch-v246.md)
- [V247 批次×学习率完整因子实验](experiments/pure-neural-batch-lr-v247.md)

## 外部方案研究

- [2025 获奖方案视频精读](research/2025获奖方案视频精读.md)
- [2025 获奖方案对 V115 的行动项](research/2025方案对V115的行动项.md)
- `research/samsung-2025/`：三星电子冠军方案的关键截图。

## 项目治理

- [版本与实验管理](governance/版本与实验管理.md)

生成的日志、临时候选、模型汤和压缩包统一保存在 Git 忽略的 `artifacts/`、`checkpoints/`、`runs/` 中。需要长期保留的结论必须提炼为本文档树中的 Markdown，成绩必须登记到 `benchmarks/leaderboard.json`。
