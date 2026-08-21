<!--
抓取来源 1: https://www.datafountain.cn/competitions/1176
抓取来源 2: https://www.datafountain.cn/competitions/1176/datasets
抓取时间: 2026-08-19 (Asia/Shanghai)
说明: 正文保持网页原始中英文内容与顺序；仅将网页结构转换为 Markdown，数学公式使用等价 LaTeX 表示。
-->

<!-- ==================== 赛制规则页原文 ==================== -->

![](https://competition-oss.datafountain.cn/dfadminwebsite-production/uploads/images/2026-08-16/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260816163130_22_138-328945.jpg)

大奖赛

## 6G/B6G内生AI：多用户MIMO端到端传输系统设计

IMT-2030（6G）推进组

智能算法 人工智能 大数据

**队伍 / 人数**

**46 / 46**

**奖金**

￥300,000

开赛

初赛

08.18 ~ 10.30

结束

---

- [大赛背景 Competition Background](https://www.datafountain.cn/competitions/1176#___Competition_Background_7)
- [赛题介绍 Competition Topic Introduction](https://www.datafountain.cn/competitions/1176#___Competition_Topic_Introduction_23)
- [大赛赛程 Competition Schedule](https://www.datafountain.cn/competitions/1176#___Competition_Schedule_208)
- [大赛规则 Competition Rules](https://www.datafountain.cn/competitions/1176#___Competition_Rules_237)

点击查看[《参赛指南中文版》](https://competition-main-data.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1176/%E5%8F%82%E8%B5%9B%E6%8C%87%E5%8D%97%E4%B8%AD%E6%96%87%E7%89%88.docx)[《Competition task guide》](https://competition-main-data.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1176/%E5%8F%82%E8%B5%9B%E6%8C%87%E5%8D%97%E8%8B%B1%E6%96%87%E7%89%88_2026_v3.docx)

## 大赛背景 Competition Background

通信与人工智能技术的深度融合已成为无线通信系统发展的最重要方向之一，面向 6G 以及其后续演进版本 B6G，通信与 AI 融合的角度和深度将进一步扩展，迎接"无限"可能。

The deep integration of communication and artificial intelligence (AI) has become one of the most crucial directions for the development of wireless communication systems. Looking toward 6G and its subsequent evolution (B6G), the perspective and depth of this convergence will expand even further, ushering in “infinite” possibilities.

大赛旨在向社会各界推广 6G/B6G 愿景、先进技术和概念，广泛吸引全社会的优秀人才，系统性、多角度地分析和研究 AI 对未来无线通信系统的影响以及对关键问题的解决方案，以期全面推动智慧 6G/B6G 的技术突破，构建产业先发优势。

The competition aims to promote the 6G/B6G vision, advanced technologies, and concepts across all sectors of society, while broadly attracting top talent. It seeks to systematically and multi-dimensionally analyze and research the impact of AI on future wireless communication systems as well as solutions to key challenges, ultimately driving technological breakthroughs in intelligent 6G/B6G and building a first-mover advantage for the industry.

大赛主旨秉承公益，坚持公平、公正、公开的原则，广泛吸引全球无线通信技术研究的爱好者、企事业单位研究人员、高校与研究机构在校学生、老师等组队参与比赛，吸引优秀人才助力 6G/B6G 研究。

The competition is committed to serving the public interest and upholding the principles of fairness, justice, and openness. It seeks to bring together wireless communication technology enthusiasts, researchers from enterprises and institutions, and students and faculty from universities and research institutes worldwide to form teams and participate, thereby attracting exceptional talent to support 6G/B6G research.

---

## 赛题介绍 Competition Topic Introduction

**1. 赛题背景 Background**

在现有无线通信系统中，物理层的数据传输链路通常由多个独立设计的信号处理模块串联而成，各模块各司其职：信道编码、调制、预编码等构成发射端，信道估计、均衡、解调、信道解码等构成接收端。控制机制则根据实时传输需求或信道条件，动态调整链路中各个模块的参数，比如编码码率、调制阶数、传输层数、预编码向量等。数据传输链路与控制机制相互配合，共同支撑多样化的通信业务。

In existing wireless communication systems, the physical layer data transmission link is typically composed of multiple independently designed signal processing modules connected in series, each performing its own function: channel coding, modulation, precoding, etc. constitute the transmitter side, while channel estimation, equalization, demodulation, channel decoding, etc. constitute the receiver side. The control mechanism dynamically adjusts the parameters of each module in the link, such as coding rate, modulation order, number of transmission layers, and precoding vectors, based on real-time transmission demands or channel conditions. The data transmission link and the control mechanism work in concert to jointly support diverse communication services.

这种模块化划分让每个模块可以独立优化到极致，但模块间的隔离性也意味着各模块独自最优并不等于整条链路全局最优。随着 6G/B6G 研究的推进，AI 被视为打破这一壁垒的关键技术。近年的研究表明，用一体化 AI 模型替代传统链路中的多个模块，能够突破模块间的性能边界，获得可观的端到端增益。但引入 AI 模型后，原本为传统模块化链路配套的控制机制难以直接套用。如何从系统的视角，将 AI 驱动的数据传输链路和 AI 驱动的控制机制统一设计，是需要回答的问题。而这一挑战在多用户 MIMO（MU-MIMO）场景下被进一步放大。

This modular partitioning allows each module to be independently optimized to its limit, but the isolation between modules also means that individual optimality does not equate to global optimality for the entire link. As 6G/B6G research advances, AI is regarded as a key technology to break through this barrier. Recent studies have shown that replacing multiple modules in the traditional link with an integrated AI model can transcend the performance boundaries between modules and yield substantial end-to-end gains. However, after introducing AI models, the control mechanisms originally designed for modular links are difficult to apply directly. How to jointly design the AI-driven data transmission link and the AI-driven control mechanism from a system perspective is a question that needs to be answered. And this challenge is further amplified in the multi-user MIMO (MU-MIMO) scenario.

MU-MIMO 是提升系统频谱效率的关键手段。在同一时频资源上，基站通过空分复用同时服务多个用户，系统容量可以成倍增长。但实现这一收益是有代价的。基站需要在空域上协调多个用户之间的资源分配，包括哪些用户适合配对传输、功率如何分配、如何设计预编码以抑制用户间干扰、是否在某些信道条件下退回到单用户传输更划算。这些决策本身的复杂度已经远高于单用户场景，当传输链路本身由 AI 驱动时，控制机制面临的挑战则更为严峻。发射机需要基于各用户反馈回来的信道信息来做调度决策，而这些反馈信息本身经过了有损压缩，调度策略的选择又必须与 AI 接收机的解码能力相匹配。上行反馈压缩到什么程度、反馈什么信息、预编码以什么方式设计、如何基于信道状况选择传输方案、整个系统是端到端一体化还是保留模块化结构，这些设计维度彼此耦合，共同决定 MU-MIMO 系统的最终性能。这正是本赛题希望选手们深入探索的方向。

MU-MIMO is a key means of improving system spectral efficiency. On the same time-frequency resources, the base station serves multiple users simultaneously through spatial multiplexing, enabling system capacity to grow multiplicatively. But achieving this gain comes at a cost. The base station must coordinate resource allocation among multiple users in the spatial domain, including which users are suitable for paired transmission, how to allocate power, how to design precoding to suppress inter-user interference, and whether it is more advantageous to fall back to single-user transmission under certain channel conditions. The complexity of these decisions is already far higher than in the single-user scenario, and when the transmission link itself is AI-driven, the challenges faced by the control mechanism become even more severe. The transmitter needs to make scheduling decisions based on the channel information fed back by each user, but this feedback information has itself undergone lossy compression, and the choice of scheduling strategy must match the decoding capability of the AI receiver. To what extent uplink feedback should be compressed, what information to feedback, how to design precoding, how to select transmission schemes based on channel conditions, and whether the entire system should be end-to-end integrated or retain a modular structure. These design dimensions are mutually coupled and jointly determine the ultimate performance of the MU-MIMO system. This is precisely the direction that the competition invites participants to explore in depth.

本大赛以 “6G/B6G 内生 AI：多用户 MIMO 端到端传输系统设计” 为题，邀请选手设计一套面向 MU-MIMO 场景的、基于 AI 的数据传输链路与控制机制，在复杂多变的信道条件下实现高效且公平的多用户传输。

Under the theme “6G/B6G Native AI: Multi-User MIMO End-to-End Transmission System Design”, this competition invites participants to design an AI-based data transmission link and control mechanism tailored for MU-MIMO scenarios, achieving efficient and fair multi-user transmission under complex and variable channel conditions.

**2. 赛题任务 Task**

本赛题要求选手设计一套完整的 MU-MIMO 端到端无线传输系统，涵盖多用户场景下的信号发送、信号接收以及信道信息反馈，在复杂信道条件下实现高效且公平的多用户数据传输。下面将从**系统框架、数学描述和方案设计**三个层面展开说明。

This competition topic requires participants to design a complete MU-MIMO end-to-end wireless transmission system, covering signal transmission, signal reception, and channel information feedback in multi-user scenarios, achieving efficient and fair multi-user data transmission under complex channel conditions. The following sections elaborate from three perspectives: system framework, mathematical description, and solution design.

**2.1 系统框架 System Framework**

考虑一个基站同时服务多个终端用户的下行传输场景。整个系统包含三条链路：

Consider a downlink transmission scenario in which a single base station simultaneously serves multiple user terminals. The entire system consists of three links:

- **下行数据链路 **：基站将发往不同用户的数据，在相同的时频资源上同时发送出去，各用户终端各自接收并恢复自己的数据。传输过程中，信号会经历信道衰落和噪声的叠加，从而产生畸变。
- **Downlink Data Link**: The base station simultaneously transmits data destined for different users on the same time-frequency resources, and each user terminal independently receives and recovers its own data. During transmission, the signals undergo channel fading and noise superposition, resulting in distortion.
- **上行反馈链路 **：各用户终端根据估计到的下行信道信息，提取并压缩关键特征，通过上行链路反馈给基站。传输过程中，反馈信号同样会叠加噪声，基站侧利用收到的反馈信息来指导发射机的决策。
- **Uplink Feedback Link**: Each user terminal extracts and compresses key features from the estimated downlink channel information and feeds them back to the base station via the uplink. During transmission, the feedback signal is also subject to noise superposition, and the base station uses the received feedback information to guide transmitter decisions.
- **下行控制链路 **：基站生成下行控制比特，告知各终端当前采用的传输策略，使收发双方对齐。为简化问题，该链路假设为无损传输。
- **Downlink Control Link**: The base station generates downlink control bits to inform each terminal of the current transmission strategy, aligning the transmitter and receiver. To simplify the problem, this link is assumed to be lossless.

而从选手的角度出发，如图 1 所示，本赛题考察选手设计部署在基站侧的发射机 $f_{tx}(\cdot)$，以及部署在每个用户终端侧的接收机 $f_{rx}(\cdot)$ 和编码器 $f_{enc}(\cdot)$ 以实现前文所述的三条链路。三者的分工如下：

From the participant’s perspective, as illustrated in Figure 1, this competition topic examines the design of the transmitter $f_{tx}(\cdot)$ deployed at the base station, and the receiver $f_{rx}(\cdot)$ and encoder $f_{enc}(\cdot)$ deployed at each user terminal, to realize the three links described above. Their respective roles are as follows:

- **编码器$f_{enc}(\cdot)$**：部署于各终端，负责将本地的下行信道信息与信噪比信息压缩为上行反馈信号，通过有损上行反馈链路发给基站。
- **Encoder $f_{enc}(\cdot)$**: deployed at each terminal and responsible for compressing the local downlink channel information and SNR information into an uplink feedback signal, which is sent to the base station via the lossy uplink feedback link.
- **发射机$f_{tx}(\cdot)$**：部署于基站，汇集各用户的反馈信息与待传输数据，将待传输比特处理为下行发送信号，同时生成下行控制信令。
- **Transmitter $f_{tx}(\cdot)$**: deployed at the base station, aggregating the feedback information from all users and the data to be transmitted, processing the bits to be transmitted into a downlink transmit signal, and simultaneously generating downlink control signaling.
- **接收机$f_{rx}(\cdot)$**：部署于各终端，根据本地的接收信号、信道信息、下行控制信令和信噪比，恢复出数据比特。
- **Receiver $f_{rx}(\cdot)$**: deployed at each terminal and recovers the data bits based on the local received signal, channel information, downlink control signaling, and SNR.

![系统整体架构示意图](https://competition-oss.datafountain.cn/dfadminwebsite-production/uploads/admin/editor/2026-08-17/ScreenShot_2026-08-16_165536_476-917640.png)

*图 1 系统整体架构 / Figure 1. System Overall Architecture*

> **理想化假设 / Idealized Assumptions**：为简化流程，赛题在若干环节做了如下假设——下行信道信息在终端侧是理想已知的；下行信噪比在基站和终端侧均为理想已知；一次上行反馈与下行数据传输周期内信道保持不变。
> 
> To simplify the process, the competition topic makes idealized assumptions in several aspects: downlink channel information is assumed to be perfectly known at the terminal side; the downlink SNR is assumed to be perfectly known at both the base station and terminal sides; the channel remains unchanged within one uplink feedback and downlink data transmission cycle.

**2.2 数学描述 Mathematical Description**

定量来说，本指南进一步用公式化的方式精确定义赛题中各模块的接口。考虑有 $N_{ue}$ 个用户终端，基站配置 $N_{tx}$ 根发送天线，每个终端配置 $N_{rx}$ 根接收天线。特别地，针对下行数据链路，考虑进行频域 $S_{dl}$ 子载波、$N_{tx}$ 发送天线、$N_{rx}$ 接收天线配置下的多天线有损传输；针对下行控制链路，考虑进行 $K$ 比特的无损传输；针对上行反馈链路，考虑进行频域 $S_{ul}$ 子载波、单发送天线、单接收天线配置下的单天线有损传输。

Quantitatively, this guide further defines the interfaces of each module in the competition topic using formulaic descriptions. Consider $N_{ue}$ user terminals, with the base station equipped with $N_{tx}$ transmit antennas and each terminal equipped with $N_{rx}$ receive antennas. Specifically, for the downlink data link, consider lossy multi-antenna transmission with $S_{dl}$ subcarriers in the frequency domain, $N_{tx}$ transmit antennas, and $N_{rx}$ receive antennas; for the downlink control link, consider lossless transmission of $K$ bits; for the uplink feedback link, consider lossy single-antenna transmission with $S_{ul}$ subcarriers in the frequency domain, one transmit antenna, and one receive antenna.

**① 编码器 $f_{enc}(\cdot)$ / Encoder $f_{enc}(\cdot)$**

在终端侧设计编码器 $f_{enc}(\cdot)$，以终端 $i$ 为例，其输入及输出可以表示为：

At the terminal side, design the encoder $f_{enc}(\cdot)$. Taking terminal $i$ as an example, its input and output can be expressed as:

$$
U_i = f_{enc}(H_i, snr_{dl,i})
$$

其中 $H_i \in \mathbb{C}^{N_{rx} \times N_{tx} \times S_{dl}}$ 为终端 $i$ 的下行信道矩阵，$snr_{dl,i}$ 为终端 $i$ 的下行信噪比（dB），$U_i \in \mathbb{C}^{S_{ul}}$ 为终端 $i$ 生成的上行反馈发送信号，$\mathbb{C}$ 表示复数集合。该信号经过上行加性噪声信道传输至基站：

where $H_i \in \mathbb{C}^{N_{rx} \times N_{tx} \times S_{dl}}$ is the downlink channel matrix for terminal $i$, $snr_{dl,i}$ is the downlink SNR (in dB) for terminal $i$, and $U_i \in \mathbb{C}^{S_{ul}}$ is the uplink feedback transmit signal generated by terminal $i$, where $\mathbb{C}$ denotes the set of complex numbers. This signal is transmitted to the base station via an uplink additive noise channel:

$$
I_i = U_i + N_{ul,i}
$$

其中 $N_{ul,i} \in \mathbb{C}^{S_{ul}}$ 为上行加性高斯白噪声，其功率由上行信噪比 $snr_{ul,i}$ 决定，$I_i \in \mathbb{C}^{S_{ul}}$ 表示基站侧接收到的上行反馈信息。上行信噪比与下行信噪比之间保持固定差值关系。

where $N_{ul,i} \in \mathbb{C}^{S_{ul}}$ is the uplink additive white Gaussian noise, whose power is determined by the uplink SNR $snr_{ul,i}$, and $I_i \in \mathbb{C}^{S_{ul}}$ denotes the uplink feedback information received at the base station. There is a fixed offset relationship between the uplink SNR and the downlink SNR.

**② 发射机 $f_{tx}(\cdot)$ / Transmitter $f_{tx}(\cdot)$**

在基站侧设计发射机 $f_{tx}(\cdot)$，其输入及输出可以表示为：

At the base station side, design the transmitter $f_{tx}(\cdot)$. Its input and output can be expressed as:

$$
b_{ctrl}, X = f_{tx}(b_{data}, I, snr_{dl})
$$

其中 $b_{data}$ 为各终端待传输比特的集合，而单个终端的待传输比特为 $b_{data,i} \in \{0,1\}^{B_{max} \times 1}$，$B_{max}$ 表示最大传输比特数，$I$ 为各终端上行反馈信息的集合，$snr_{dl}$ 为各终端对应的下行信噪比的集合，$X \in \mathbb{C}^{N_{tx} \times S_{dl}}$ 表示经过发射机 $f_{tx}(\cdot)$ 处理后的下行数据发送信号，$b_{ctrl} \in \{0,1\}^{K \times 1}$ 表示下行控制比特。下行发送信号 $X$ 经过各自的乘性信道到达各终端。以终端 $i$ 为例，其接收信号为：

where $b_{data}$ is the set of bits to be transmitted for each terminal, with the bits for a single terminal being $b_{data,i} \in \{0,1\}^{B_{max} \times 1}$, $B_{max}$ denotes the maximum number of transmission bits, $I$ is the set of uplink feedback information from all terminals, $snr_{dl}$ is the set of downlink SNRs for all terminals, $X \in \mathbb{C}^{N_{tx} \times S_{dl}}$ denotes the downlink data transmit signal after processing by the transmitter $f_{tx}(\cdot)$, and $b_{ctrl} \in \{0,1\}^{K \times 1}$ denotes the downlink control bits. The downlink transmit signal $X$ reaches each terminal through its respective multiplicative channel. Taking terminal $i$ as an example, its received signal is:

$$
Y_{s,i} = H_{s,i}X_s + N_{dl,s,i}
$$

其中 $Y_{s,i} \in \mathbb{C}^{N_{rx}}$ 表示第 $s$ 个子载波上的终端 $i$ 接收信号，$1 \le s \le S_{dl}$ 表示下行子载波索引，$H_{s,i} \in \mathbb{C}^{N_{rx} \times N_{tx}}$ 表示第 $s$ 个子载波的对应于终端 $i$ 的下行信道，$N_{dl,s,i} \in \mathbb{C}^{N_{rx}}$ 表示下行第 $s$ 个子载波上根据 $snr_{dl,i}$ 计算的对应于终端 $i$ 加性高斯白噪声。最后，将全部 $S_{dl}$ 个下行时频资源的接收信号 $Y_{s,i}$ 进行拼接，获得最终的接收信号 $Y_i \in \mathbb{C}^{N_{rx} \times S_{dl}}$。

where $Y_{s,i} \in \mathbb{C}^{N_{rx}}$ denotes the received signal of terminal $i$ on the $s$-th subcarrier, $1 \le s \le S_{dl}$ denotes the downlink subcarrier index, $H_{s,i} \in \mathbb{C}^{N_{rx} \times N_{tx}}$ denotes the downlink channel for terminal $i$ on the $s$-th subcarrier, and $N_{dl,s,i} \in \mathbb{C}^{N_{rx}}$ denotes the additive white Gaussian noise for terminal $i$ on the $s$-th downlink subcarrier computed according to $snr_{dl,i}$. Finally, the received signals $Y_{s,i}$ across all $S_{dl}$ downlink time-frequency resources are concatenated to obtain the final received signal $Y_i \in \mathbb{C}^{N_{rx} \times S_{dl}}$.

**③ 接收机 $f_{rx}(\cdot)$ / Receiver $f_{rx}(\cdot)$**

在终端侧设计接收机 $f_{rx}(\cdot)$，以终端 $i$ 为例，其输入及输出可以表示为：

At the terminal side, design the receiver $f_{rx}(\cdot)$. Taking terminal $i$ as an example, its input and output can be expressed as:

$$
c_{data,i} = f_{rx}(Y_i, H_i, b_{ctrl}, snr_{dl,i})
$$

其中 $c_{data,i} \in \mathbb{R}^{B \times 1}$ 表示终端 $i$ 接收数据比特对应的对数似然比，$\mathbb{R}$ 表示实数集合，$B \le B_{max}$ 表示实际传输比特数，$Y_i \in \mathbb{C}^{N_{rx} \times S_{dl}}$ 表示终端 $i$ 的下行接收信号，$H_i \in \mathbb{C}^{N_{rx} \times N_{tx} \times S_{dl}}$ 表示终端 $i$ 对应的下行信道，$b_{ctrl} \in \{0,1\}^{K \times 1}$ 表示下行控制比特，$snr_{dl,i}$ 表示终端 $i$ 对应的下行数据信噪比。

where $c_{data,i} \in \mathbb{R}^{B \times 1}$ denotes the log-likelihood ratios corresponding to the received data bits of terminal $i$, $\mathbb{R}$ denotes the set of real numbers, $B \le B_{max}$ denotes the actual number of transmitted bits, $Y_i \in \mathbb{C}^{N_{rx} \times S_{dl}}$ denotes the downlink received signal of terminal $i$, $H_i \in \mathbb{C}^{N_{rx} \times N_{tx} \times S_{dl}}$ denotes the downlink channel for terminal $i$, $b_{ctrl} \in \{0,1\}^{K \times 1}$ denotes the downlink control bits, and $snr_{dl,i}$ denotes the downlink data SNR for terminal $i$.

**④ 优化目标 / Optimization Objective**

赛题在上述系统框架下，考察选手的发射机 $f_{tx}(\cdot)$、接收机 $f_{rx}(\cdot)$ 以及编码器 $f_{enc}(\cdot)$ 的联合设计方案，以实现数据比特流的高精度传输，即最优化

Under the above system framework, the competition examines the joint design of the participant’s transmitter $f_{tx}(\cdot)$, receiver $f_{rx}(\cdot)$, and encoder $f_{enc}(\cdot)$ to achieve high-precision transmission of the data bit stream, i.e., optimizing:

$$
\max g(b_{data}, p(c_{data}))
$$

其中 $p(\cdot)$ 表示硬判决过程（当 $c_{data} < 0$ 时 $p(c_{data}) = 0$，当 $c_{data} \ge 0$ 时 $p(c_{data}) = 1$），$g(\cdot)$ 通过计算 $p(c_{data})$ 与 $b_{data}$ 中的前 $B \le B_{max}$ 位比特相同的位数输出正确传输的比特数 $B_c$。

where $p(\cdot)$ denotes the hard decision process ($p(c_{data}) = 0$ when $c_{data} < 0$, and $p(c_{data}) = 1$ when $c_{data} \ge 0$), and $g(\cdot)$ outputs the number of correctly transmitted bits $B_c$ by counting the number of bits for which $p(c_{data})$ matches the first $B \le B_{max}$ bits in $b_{data}$.

**⑤ 系统配置 / System Configuration**

相应系统配置在下表中给出。

The corresponding system configuration is given in the table below.

| 参数 (Parameter) | 数值 (Value) |
| --- | --- |
| 下行数据频域子载波数 𝑆 𝑑 𝑙 S dl ​ / Number of downlink data subcarriers in frequency domain, 𝑆 𝑑 𝑙 S dl ​ | 144 |
| 用户终端数 𝑁 𝑢 𝑒 N ue ​ / Number of user terminals, 𝑁 𝑢 𝑒 N ue ​ | 2 |
| 下行数据发送天线数 𝑁 𝑡 𝑥 N tx ​ / Number of downlink transmit antennas, 𝑁 𝑡 𝑥 N tx ​ | 16 |
| 单个终端下行数据接收天线数 𝑁 𝑟 𝑥 N rx ​ / Number of downlink receive antennas per terminal, 𝑁 𝑟 𝑥 N rx ​ | 2 |
| 下行控制比特数 𝐾 K / Number of downlink control bits, 𝐾 K | 5 |
| 多用户下行最大传输总比特数 𝐵 𝑚 𝑎 𝑥 B max ​ / Maximum total number of downlink transmission bits for multi-user, 𝐵 𝑚 𝑎 𝑥 B max ​ | 2304 |
| 下行信噪比 𝑠 𝑛 𝑟 𝑑 𝑙 snr dl ​ / Downlink SNR, 𝑠 𝑛 𝑟 𝑑 𝑙 snr dl ​ | -20~20 dB |
| 上行反馈频域子载波数 𝑆 𝑢 𝑙 S ul ​ / Number of uplink feedback subcarriers in frequency domain, 𝑆 𝑢 𝑙 S ul ​ | 96 |
| 上行信噪比 𝑠 𝑛 𝑟 𝑢 𝑙 snr ul ​ / Uplink SNR, 𝑠 𝑛 𝑟 𝑢 𝑙 snr ul ​ | 𝑠 𝑛 𝑟 𝑑 𝑙 snr dl ​ - 10 dB |

**3. 方案设计 Solution Design**

以上给出了赛题的框架和接口描述，但从方案设计的角度来看，这个框架留下了大量的发挥空间。下面从几个维度展开讨论，帮助选手理解可以从哪些方向入手。

The above provides the framework and interface description of the competition topic, but from a solution design perspective, this framework leaves substantial room for creativity. The following discussion explores several dimensions to help participants understand possible directions to pursue.

**3.1 架构选择：一体化还是模块化 Architecture Selection: Integrated vs. Modular**

一体化思路是将发射机和接收机各自设计为一体化的 AI 模型，发射端的所有信号处理步骤（例如调制、预编码等）由一个神经网络完成，接收端的所有步骤（例如信道均衡、解调等）同样由一个神经网络完成，控制信息可以作为模型的附加输入直接参与推理。这种方案设计自由度大、性能上限高，但端到端训练的收敛难度也更高。另一种思路是保留一定的模块化结构，但让 AI 模型分别实现各模块的功能，并负责模块的选择和切换。比如，发射机内部预置多套方案（每套方案是一组可以配合使用的编码器、预编码器和调制器），控制机制根据反馈信息实时选择最合适的方案。模块化方案可以借助经典通信设计的先验知识降低学习难度，同时选手也可以在模块之间引入跨模块联合优化，在保留结构的前提下追求更高的性能。

The integrated approach designs the transmitter and receiver each as a unified AI model, where all signal processing steps at the transmitter (e.g., modulation, precoding, etc.) are performed by a single neural network, and all steps at the receiver (e.g., channel equalization, demodulation, etc.) are likewise performed by a single neural network. Control information can directly participate in inference as additional model inputs. This approach offers high design freedom and a high performance ceiling, but end-to-end training convergence is also more challenging. An alternative approach is to retain a certain modular structure while having AI models realize the functions of individual modules and handle module selection and switching. For example, the transmitter may internally pre-configure multiple schemes (each scheme being a set of compatible encoder, precoder, and modulator), and the control mechanism selects the most suitable scheme in real time based on feedback information. Modular approaches can leverage the prior knowledge of classical communication design to reduce learning difficulty, while participants can also introduce cross-module joint optimization between modules to pursue higher performance while preserving the structure.

**3.2 上行反馈：从信道到控制信息 / Uplink Feedback: From Channel to Control Information**

编码器 $f_{enc}$ 的输入是完整的高维信道矩阵 $H_i$，输出是低维的上行反馈信号 $U_i$。本质上，这是一个有损压缩问题，将几百个复数压缩到几十个复数，同时保留对发射机决策最有用的信息。输出不再是比特流，而是直接在时频资源上传输的归一化复数信号。选手可以自由选择压缩的方式和反馈的内容。在压缩方式上，可以对信道做奇异值分解并反馈特征向量和奇异值，也可以对信道做傅里叶变换后在变换域压缩，还可以直接用神经网络做端到端的特征提取，不对中间表示做任何人为约束。在反馈内容上，是反馈完整的信道矩阵让基站做预编码计算，还是终端侧自己算好推荐的预编码向量再反馈，抑或是反馈某种"信道质量指标"让基站做调度决策，不同的选择会导向完全不同的系统设计。

The input to the encoder $f_{enc}$ is the complete high-dimensional channel matrix $H_i$, and the output is the low-dimensional uplink feedback signal $U_i$. In essence, this is a lossy compression problem: compressing hundreds of complex numbers into tens of complex numbers while preserving the information most useful for transmitter decision-making. The output is no longer a bit stream but a normalized complex signal directly transmitted on time-frequency resources. Participants may freely choose the compression method and the content of feedback. In terms of compression methods, one may perform singular value decomposition on the channel and feedback eigenvectors and singular values, compress in the transform domain after applying a Fourier transform to the channel, or directly use a neural network for end-to-end feature extraction without imposing any artificial constraints on intermediate representations. In terms of feedback content, whether to feed back the complete channel matrix for the base station to compute precoding, or to have the terminal compute recommended precoding vectors and then feed them back, or to feed back some “channel quality indicator” for the base station to make scheduling decisions; different choices lead to entirely different system designs.

**3.3 多用户传输策略：MU 还是 SU Multi-User Transmission Strategy: MU vs. SU**

基站同时持有多个用户的反馈信息，需要决定如何在有限的时频资源上调度这些用户。在 MU-MIMO 中，基站可以通过预编码在空间上分离不同用户的数据流，实现多用户空分复用传输。但这样做的前提是用户之间的信道相关性足够低，预编码能够有效抑制用户间干扰。当两个用户的信道高度相关时，强行配对反而会让干扰淹没信号，此时退回到单用户传输（SU），把全部功率和时频资源分配给一个用户，可能反而是更优的选择。这就引出了一系列需要选手考虑的问题：如何从反馈信息中判断用户信道是否适合配对？预编码矩阵如何设计以平衡有用信号增强和干扰抑制？功率在各用户之间如何分配？是否需要在不同子载波上做不同的配对决策？

The base station simultaneously holds the feedback information of multiple users and needs to decide how to schedule these users on the limited time-frequency resources. In MU-MIMO, the base station can spatially separate the data streams of different users through precoding, achieving multi-user spatial multiplexing transmission. However, the premise is that the channel correlation between users is sufficiently low so that precoding can effectively suppress inter-user interference. When the channels of two users are highly correlated, forcing pairing may instead cause interference to drown out the signal, in which case falling back to single-user (SU) transmission, allocating all power and time-frequency resources to a single user, may be the better choice. This raises a series of questions for participants to consider: How to determine from feedback information whether user channels are suitable for pairing? How to design the precoding matrix to balance desired signal enhancement and interference suppression? How to allocate power among users? Should different pairing decisions be made on different subcarriers?

**3.4 传输方案自适应 Adaptive Transmission Scheme**

选手可以为不同用户、不同子载波、不同信道条件设计自适应的传输方案。一个自然的思路是，在信道条件好的情况下用更激进的传输方式传输更多比特，在信道条件差的情况下则保守一些以保证传输可靠性。同一个系统中，不同用户可能面临截然不同的信道条件，不同子载波上的信道质量也可能参差不齐，传输方案可以考虑在不同用户、不同信道质量下动态调整，而非一刀切。

Participants may design adaptive transmission schemes for different users, different subcarriers, and different channel conditions. A natural approach is to use more aggressive transmission modes to transmit more bits under good channel conditions, and be more conservative under poor channel conditions to ensure transmission reliability. Within the same system, different users may face markedly different channel conditions, and the channel quality across different subcarriers may also vary considerably. Transmission schemes may be dynamically adjusted across different users and under different channel qualities, rather than adopting a one-size-fits-all approach.

**3.5 控制信令 Control Signaling**

下行控制比特 $b_{ctrl}$ 为基站向终端传递控制信息提供了接口。选手可以利用这 $K$ 个比特告知终端当前的传输策略，比如采用了 MU 还是 SU 传输、不同频率资源是否采用了不同的传输方式，等等。终端侧的接收机读到这些信息后，可以对应地切换信号接收策略。参考代码中这 $K$ 个比特目前是占位实现（固定为全 1），选手自行设计完整的控制信令表示方式。

The downlink control bits $b_{ctrl}$ provide an interface for the base station to convey control information to the terminals. Participants may use these $K$ bits to inform the terminals of the current transmission strategy, such as whether MU or SU transmission is adopted, whether different transmission modes are used on different frequency resources, and so on. Upon reading this information, the receiver at the terminal side can correspondingly switch its signal reception strategy. In the reference code, these $K$ bits are currently a placeholder implementation (fixed to all ones); participants are to design the complete control signaling representation on their own.

> **写在最后 / Final Words**：上面关于发散思路的内容仅是用于抛砖引玉，期待富有创意的选手们在后续的赛程中打造出优秀作品，绽放异彩！
> 
> The above discussion of divergent ideas is merely intended to spark inspiration. We look forward to creative participants producing outstanding works and shining brilliantly in the subsequent stages of the competition!

---

## 大赛赛程 Competition Schedule

> ※ 大赛具体赛程请及时关注大赛平台和交流群通知。
> 
> ※ Please keep an eye on the competition platform and the communication group for the specific schedule of the competition.

| 时间 (Date) | 日程 (Schedule) |
| --- | --- |
| 2026年8月18日 / August 18, 2026 | 大赛启动。参赛选手登录官网报名 / Competition launch. Participants register on the official website. |
| 2026年9月1日 - 2026年10月30日 / September 1, 2026 – October 30, 2026 | 大赛评测。选手登录官网下载示例代码和数据集，并将结果提交到竞赛平台 / Competition evaluation. Participants log in to the official website to download sample code and datasets, and submit results to the competition platform. |
| 2026年10月30日 - 2026年11月20日 / October 30, 2026 – November 20, 2026 | 作品审核 / Submission review |
| 2026年11月20日 - 2026年11月25日 / November 20, 2026 – November 25, 2026 | 获奖名单公示 / Announcement of winners |
| 2026年12月（待定）/ December 2026 (TBD) | 成果展示，颁奖 / Presentation of winning solutions & Awards ceremony |

---

## 大赛规则 Competition Rules

- **参赛人群**：大赛面向全球社会各界开放，不限年龄、国籍，各高等院校、科研单位、企事业单位、个人等均可登录官网报名参赛。相关主办、承办单位及其关联单位所属员工可参赛但不可获奖，不占用获奖名额；（榜单锁定后，如果该团队进入榜单获奖排名范围内，不可获奖，不占用获奖名额，可参加分享活动，获奖团队顺延）
- **Participants**: The competition is open to all people worldwide, regardless of age and nationality. Universities, research institutes, enterprises, institutions, and individuals may all register on the official website. Employees of the relevant hosts, organizers, and their affiliated entities may participate but are ineligible to win awards and do not occupy award slots. (i.e., if the team falls within the award-winning ranking range after the leaderboard is locked, the team shall not receive an award and shall not occupy an award slot. The team may participate in sharing activities, and the award will be extended to the next-ranked team.)
- **报名要求 **：每位参赛人在每道赛题下，可以个人身份或仅以一支参赛团队队员身份参赛，每支参赛团队人数不超过 5 人。报名时所有成员需提供个人基本信息，并通过实名认证；需在组队截止日期前完成组队，一旦组队不得退队或者换人；参赛账号仅限参赛者本人使用，不得将其以任何形式向第三方转让、出租、出借、出售、披露、泄露等。参赛者有义务保证账号信息的真实性和有效性，因信息无效或错误造成的后果由参赛者自行承担。参赛者对参赛账号的使用自行承担责任，遭遇安全问题（如账号遗失、被盗、被未授权使用等）时应立即通知组委会。
- **Registration Requirements**: Each participant may compete in each competition track either as an individual or as a member of exactly one team, with each team having no more than 5 members. A participant may sign up for multiple competition tracks. All members must provide their basic personal information and pass real-name authentication during registration. Team formation must be completed before the deadline, and once a team is formed, no withdrawal or replacement is allowed. The participant’s account is for their personal use only and may not be transferred, rented, lent, sold, disclosed, or divulged to any third party in any form. Participants are obliged to ensure the authenticity and validity of their account information and shall bear the consequences caused by invalid or incorrect information. Participants are responsible for the use of their own accounts and should notify the organizing committee immediately upon encountering security issues (such as account loss, theft, or unauthorized use).
- **提交规则**：为保证每支队伍享有相对平等的提交机会，每支队伍每天最多提交作品 3 次；
- **Submission Rules**: To ensure that all teams have relatively equal submission opportunities, each team may submit up to 3 entries per day.
- **公平竞技**：参赛者禁止在指定考核技术能力的范围外，利用规则漏洞或技术漏洞等不良途径提高成绩排名，禁止在比赛中抄袭他人作品、交换答案、使用多个小号提交作品、不同队伍使用雷同作品恶意提交，经发现将取消比赛成绩并严肃处理，也请参赛者保持竞赛公平秩序。
- **Fair Competition**: Participants are prohibited from exploiting rule loopholes or technical vulnerabilities to improve their ranking outside the designated scope of technical capability assessment. Participants are prohibited from plagiarizing others’ work, exchanging answers, using multiple accounts to submit entries, or using identical entries across different teams for malicious submission. Participants are requested to maintain fair competition order and will be disqualified if the above-mentioned improper behaviors are discovered.
- **组织声明 **：组委会保留对比赛规则进行调整修改的权利、比赛作弊行为的判定权利和处置权利、收回或拒绝授予影响组织及公平性的参赛团队奖项的权利；
- **Organizational Statement**: The organizing committee reserves the right to adjust and modify the competition rules, the right to determine and handle cheating in the competition, and the right to withdraw or refuse to grant awards to teams that compromise the organization or fairness of the competition.
- **竞赛数据 **：组委会授权参赛人员使用提供的数据进行指定比赛的模型训练工作，参赛人员不得将数据用于任何商业用途，不得将数据转交或分享给非参赛人员。若做科研使用，请注明数据来源于相关数据提供单位；
- **Competition Data**: The organizing committee authorizes participants to use the provided data for model training in the designated competition. Participants are not allowed to use the data for any commercial purposes, or to share the data with non-participants. If used for scientific research, please cite the data source as the relevant data provider.
- **作品知识产权**：参赛作品（包含但不限于算法、模型等）知识产权归参赛者所有，组委会经选手同意后可以将参赛作品、作品相关、参赛团队信息用于宣传品、相关出版物、指定及授权媒体发布、官方网站浏览及下载、展览（含巡展）等活动项目，大赛相关组织单位享有优先合作权利。参赛作品必须保证原创性，不违反任何中华人民共和国的有关法律，不侵犯任何第三方知识产权或者其他权利。比赛期间，数据、代码、模型严禁开源。
- **Intellectual Property of Submissions**: The intellectual property rights of the submissions (including but not limited to algorithms, models, etc.) belong to the participants. The organizing committee may, with the consent of the participants, use the submissions, related materials, and team information for promotional materials, relevant publications, designated and authorized media releases, official website browsing and downloading, exhibitions (including touring exhibitions), and other activities. The competition organizer has the priority right of cooperation. Submissions must be original, must not violate any relevant laws of the People’s Republic of China, and must not infringe upon any third-party intellectual property rights or other rights. During the competition period, data, code, and models are strictly prohibited from being open-sourced.
- **特别回避 **：相关主办、承办单位中涉及题目编写、数据接触的人员禁止参赛，禁止委托他人参赛或指导参赛团队；
- **Special Recusal**: Personnel involved in problem development and data processing from the relevant hosts and organizers are prohibited from participating in the competition, and from entrusting others to participate or coaching participating teams.
- **通知沟通 **：参赛团队预留联系方式将作为大赛组委会与参赛团队联系的唯一接口，大赛各项信息包括但不限于比赛活动安排、培训和分享、获奖通知、颁奖典礼等。若参赛团队在上述相关通知发出后 3 日内未答复则视为自动放弃相应机会，主办方有权顺位递补其他参赛团队。
- **Communication**: The contact information provided by participating teams shall serve as the sole interface for communication between the organizing committee and the teams. Competition information includes but is not limited to event schedules, training and sharing sessions, award notifications, and award ceremonies. If a team does not reply within 3 days after the above-mentioned notification is issued, it shall be deemed to have automatically forfeited the corresponding opportunity, and the organizer has the right to fill the vacancy with the next-ranked team in sequence.

<!-- ==================== 数据与评测页原文 ==================== -->

![](https://competition-oss.datafountain.cn/dfadminwebsite-production/uploads/images/2026-08-16/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260816163130_22_138-328945.jpg)

大奖赛

## 6G/B6G内生AI：多用户MIMO端到端传输系统设计

IMT-2030（6G）推进组

智能算法 人工智能 大数据

**队伍 / 人数**

**46 / 46**

**奖金**

￥300,000

开赛

初赛

08.18 ~ 10.30

结束

---

- [赛题数据 Dataset Introduction](https://www.datafountain.cn/competitions/1176/datasets#___Dataset_Introduction_0)
- [赛题提交要求 Submission Requirements](https://www.datafountain.cn/competitions/1176/datasets#___Submission_Requirements_36)
- [提交示例 Submission Example](https://www.datafountain.cn/competitions/1176/datasets#___Submission_Example_58)
- [赛题打分规则 Scoring Rules](https://www.datafountain.cn/competitions/1176/datasets#___Scoring_Rules_94)

**2026/08/18 12:18:24** **赛题数据**

训练数据集 - MD5: EAC0FEF9BA0DA420CC7F9DEC1A05FBA8

**2026/08/18 12:20:01** **示例程序**

模型定义 - MD5: 736335902833ae7b8b53be6167de7c34

**2026/08/18 12:21:33** **示例程序**

模型训练 - MD5: 4c8e71201ba19fe0426641952a6fb304

**2026/08/18 12:22:24** **示例程序**

评测示例 - MD5: 39c07ede8d7c782c63e2f8f139924372

## 赛题数据 Dataset Introduction

本赛题给出下行信道样本作为训练数据，具体来说，赛方给出 **100000** 样本，每个样本的维度含义为 **[用户数，接收天线数，发送天线数，子载波数]**。

This competition topic provides downlink channel samples as training data. Specifically, the organizer provides **100,000** samples, with each sample having the dimensional meaning of **[number of users, number of receive antennas, number of transmit antennas, number of subcarriers]**.

进一步地，赛方将提供如下材料：

Furthermore, the organizer will provide the following materials:

```
pt_template.zip：PyTorch版本训练模板示例文件夹
├── modelTrain.py：训练示例
├── modelDesign.py：模型定义示例
└── modelEval.py：本地评测示例

data.zip：数据文件夹
└── H_train：下行信道训练数据

submit_pt.zip：PyTorch版本提交示例（结构见下文"提交示例"章节）
```

```
pt_template.zip: PyTorch training template example folder
├── modelTrain.py: Training example
├── modelDesign.py: Model definition example
└── modelEval.py: Local evaluation example

data.zip: Data folder
└── H_train: Downlink channel training data

submit_pt.zip: PyTorch submission example (see the "Submission Example" section below for structure)
```

---

## 赛题提交要求 Submission Requirements

各参赛选手请按以下要求完成方案设计，并将结果的压缩包文件上传至竞赛平台：

All participants are requested to complete their solution design according to the following requirements and upload the compressed archive file to the competition platform:

1. 编程语言版本建议：**Python 3.9**；
2. 调用宏包版本建议：**PyTorch > 2.8.0；NumPy 1.26.4**；
3. 上传文件大小限制：上传文件大小不得超过 **1G**；
4. 评测推理时间限制：推理时长不得超过 **1000s**。
5. Recommended programming language version: **Python 3.9**;
6. Recommended package versions: **PyTorch > 2.8.0; NumPy 1.26.4**;
7. Upload file size limit: The uploaded file size must not exceed **1 GB**;
8. Evaluation inference time limit: Inference duration must not exceed **1000 s**.

本赛题支持PyTorch版本结果的提交，详细提交内容见后文提交示例。

This competition topic supports PyTorch-based result submission. See the submission example below for detailed submission content.

---

## 提交示例 Submission Example

需提交方案设计文件 **modelDesign.py**、发射机 $f_{tx}(\cdot)$ 权重文件 **transmitter.pth**、接收机 $f_{rx}(\cdot)$ 权重文件 **receiver.pth**、编码器 $f_{enc}(\cdot)$ 权重文件 **encoder.pth**，其中请将所需函数、模型架构、信号流程设计等所有依赖通过 modelDesign.py 文件定义。

The following files must be submitted: the solution design file **modelDesign.py**, the transmitter $f_{tx}(\cdot)$ weight file **transmitter.pth**, the receiver $f_{rx}(\cdot)$ weight file **receiver.pth**, and the encoder $f_{enc}(\cdot)$ weight file **encoder.pth**. All required functions, model architecture, signal flow design, and other dependencies must be defined in the modelDesign.py file.

请将文件以如下结构进行压缩打包并上传，例如：

Please compress and package the files according to the following structure and upload, for example:

```
submit_pt.zip
└── submit_pt（文件夹）
    ├── modelDesign.py（方案设计文件）
    └── modelSubmit（文件夹）
        ├── transmitter.pth（发射机ftx(·)权重文件）
        ├── receiver.pth（接收机frx(·)权重文件）
        └── encoder.pth（编码器fenc(·)权重文件）
```

```
submit_pt.zip
└── submit_pt (folder)
    ├── modelDesign.py (solution design file)
    └── modelSubmit (folder)
        ├── transmitter.pth (transmitter ftx(·) weight file)
        ├── receiver.pth (receiver frx(·) weight file)
        └── encoder.pth (encoder fenc(·) weight file)
```

> **注意**：请各位选手保证上传内容完整，不要遗漏 modelDesign.py、transmitter.pth、receiver.pth、encoder.pth 等文件。
> 
> **Note**: Please ensure that the uploaded content is complete and do not omit files such as modelDesign.py, transmitter.pth, receiver.pth, encoder.pth, etc.

---

## 赛题打分规则 Scoring Rules

本次大赛采用"双维双榜"综合评审机制，旨在兼顾算法硬实力与方案软价值。

This competition adopts a “Dual-Dimension, Dual-Leaderboard” comprehensive evaluation mechanism, designed to balance algorithmic hard power with the soft value of the solution.

- **A榜（客观量化榜）**：依据提交结果的客观性能指标进行严格排序，是方案可行性与鲁棒性的"试金石"，反映方案在当前赛题下的实证效果。
- **Leaderboard A (Objective Quantitative Leaderboard)**: Strictly ranked based on objective performance metrics of the submitted results. It serves as the “touchstone” of solution feasibility and robustness, reflecting the empirical effectiveness of the solution under the current competition topic.
- **B榜（创新潜力榜）**：聚焦方案的颠覆性潜力与学术价值。参赛团队需通过Workshop进行方案陈述，由专家评审团从创新性、方法论前瞻性及对下一代通信标准的启发性三个维度综合排序。
- **Leaderboard B (Innovation Potential Leaderboard)**: Focuses on the disruptive potential and academic value of the solution. Participating teams are required to present their solutions through a Workshop, and an expert review panel will comprehensively rank them across three dimensions: innovativeness, methodological foresight, and inspiration for next-generation communication standards.

**终极排名公式**：总成绩 = 0.8×A榜名次 + 0.2×B榜名次。同分时，A榜硬实力高者胜，彰显"优中选优，性能为本"的公平原则。

**Final Ranking Formula**: Total Score = 0.8 × Leaderboard A Rank + 0.2 × Leaderboard B Rank. In case of a tie, the team with higher hard power on Leaderboard A prevails, reflecting the fairness principle of “selecting the best of the best, with performance as the foundation.”

对于A榜对应的选手方案客观得分，本赛题同时考察传输的效率和用户间的公平性。具体而言，对单个用户的一次传输，其分数为：

For the objective score of a participant’s solution on Leaderboard A, this competition topic simultaneously evaluates transmission efficiency and fairness among users. Specifically, for a single transmission to a single user, the score is:

$$
\mu = 100 \times \frac{B_c + (B_{max} - B) \times 0.5}{B_{max}}
$$

其中 $B_{max}$ 表示最大传输比特数，$B \le B_{max}$ 表示实际传输的比特数，$B_c \le B$ 表示正确传输的比特数。在评测时，系统会在大批量信道、待传输数据样本上测试，每个样本产生多个用户的分数。所有用户分数汇集后计算两个指标：

where $B_{max}$ denotes the maximum number of transmission bits, $B \le B_{max}$ denotes the actual number of transmitted bits, and $B_c \le B$ denotes the number of correctly transmitted bits. During evaluation, the system tests on a large batch of channel and data samples, with each sample producing scores for multiple users. After aggregating all user scores, two metrics are computed:

- **效率分**：所有用户分数的均值，衡量系统的平均传输性能。
- **Efficiency Score**: The mean of all user scores, measuring the average transmission performance of the system.
- **公平分**：所有用户分数按升序排列后的第10百分位值，衡量系统对"弱势用户"的服务质量。
- **Fairness Score**: The 10th percentile value of all user scores sorted in ascending order, measuring the system’s quality of service for “disadvantaged users.”

**A榜最终客观得分 = 0.7 × 效率分 + 0.3 × 公平分。**

**Leaderboard A Final Objective Score = 0.7 × Efficiency Score + 0.3 × Fairness Score.**

在比赛过程中，选手的最高A榜得分将在排行榜上展示，排名以提交的最高得分为依据。选手可在提交记录中查询每次提交的分数。

During the competition, the participant’s highest Leaderboard A score will be displayed on the leaderboard, and ranking will be based on the highest submitted score. Participants may check the score of each submission in their submission history.

竞赛数据下载协议

同意并下载数据
