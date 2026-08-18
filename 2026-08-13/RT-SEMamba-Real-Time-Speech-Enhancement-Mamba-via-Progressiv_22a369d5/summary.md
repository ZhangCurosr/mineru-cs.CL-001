---
title: "RT-SEMamba-Real-Time-Speech-Enhancement-Mamba-via-Progressiv"
source: https://arxiv.org/pdf/2608.12099v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:58:19"
field: "实时语音增强"
keywords: ["speech enhancement", "real-time processing", "Mamba", "knowledge distillation", "state-space models", "streaming audio"]
innovations: ["全因果时频Mamba流式架构配合渐进式知识蒸馏", "从8层教师到1层学生的深度压缩实现2.75x加速且PESQ仅降0.14", "流式推理三缓冲状态管理（帧缓冲/卷积状态/SSM隐状态）"]
benchmarks: ["VCTK-DEMAND"]
---

# 论文速读：RT-SEMamba-Real-Time-Speech-Enhancement-Mamba-via-Progressiv

## 一句话总结
论文提出 RT-SEMamba，一种基于因果时频 Mamba 块的全因果语音增强模型，通过渐进式知识蒸馏将 8 层深度教师网络压缩至 1 层轻量学生网络，在 25 ms 算法延迟约束下实现 3.18 PESQ（1 层蒸馏后）与 0.11 RTF，显著优于朴素 1 层基线（3.06 PESQ），且比 8 层教师快约 2.75×。

## 研究问题与动机
1. **实时语音增强的低延迟需求**：助听器、AR/VR、视频会议等场景要求前端满足严格算法延迟（如 25 ms）和 RTF 约束，同时保持对非平稳噪声和混响的鲁棒性。
2. **Transformer 类模型在流式部署中的瓶颈**：基于 Transformer 的 SE 模型依赖不断增长的 KV cache，导致内存和带宽开销随序列长度线性增长，不适合边缘设备流式推理。
3. **已有 Mamba 模型的因果性缺失**：现有基于 Mamba 的语音增强方法（如 SEMamba）主要在非因果离线设置下评估，未显式支持流式部署；需要重新设计为完全因果架构。
4. **深度与延迟的质量权衡困境**：加深 cTF-Mamba 层数可提升增强质量，但 RTF 几乎线性增长（1 层 RTF=0.11 → 8 层 RTF=0.29），而浅层模型质量明显不足，需通过蒸馏弥合差距而不增加延迟。

## 核心贡献（创新点）
1. **全因果 RT-SEMamba 架构**：将 SEMamba 改造为完全因果的流式语音增强模型，采用因果 STFT/iSTFT（hop=100, W=400, center=False）、不对称因果卷积填充、时间轴单向 Mamba，频率轴保持双向建模，实现 25 ms 固定算法延迟。
2. **渐进式知识蒸馏策略**：提出从 8 层教师到 1 层学生的深度压缩方案，联合蒸馏复数频谱输出（幅度、相位、复数谱）和中间特征表示（逐层归一化后聚合），通过渐进式 ramp-up（前 10% 训练步骤线性引入蒸馏信号）加速收敛。
3. **流式推理状态管理设计**：设计了帧缓冲（temporal frame buffer）、深度卷积状态缓冲（conv state buffer）和 SSM 隐状态（ssm state）三类状态传播机制，实现 1-frame-in/1-frame-out 在线模式，避免重复计算历史帧，内存和每帧计算量与序列长度无关。
4. **质量-延迟帕累托前沿优化**：实验证明蒸馏后 1 层学生（PESQ=3.18）相比朴素 1 层基线（PESQ=3.06）显著改进，同时保持相同 RTF（0.11），且恢复教师-学生间 46.2% 的 PESQ 差距，验证了 SSM 配合蒸馏在流式 SE 中的竞争力。

## 方法详解
**输入表示**：在复数 STFT 域操作，对噪声波形 $x(t)$ 计算 $\mathbf{X}(f, \tau) = \text{STFT}(x(t)) = \mathbf{M}(f, \tau) e^{j\mathbf{P}(f, \tau)}$，网络输入 $(\mathbf{M}, \mathbf{P})$，预测 $(\hat{\mathbf{M}}, \hat{\mathbf{P}}, \hat{\mathbf{C}})$，其中 $\hat{\mathbf{C}} = \hat{\mathbf{M}} e^{j\hat{\mathbf{P}}}$ 为复数重建谱，增强波形由逆 STFT 得到。

**因果架构设计**：
- 前端使用 16 kHz 采样率的因果 STFT/iSTFT，window size W=400，hop H=100，center=False，确保算法延迟 ≤25 ms。
- 编码器与解码器中所有时序卷积采用不对称因果填充：kernel size K 时左侧填充 $(K-1)$ 个零，右侧不填充； utterance 开头缺失的历史用零填充。
- 将 InstanceNorm2d 替换为沿时间轴带因果填充的 channel-wise LayerNorm。
- 每个 cTF-Mamba 块后插入额外 MLP 增强单帧建模能力。
- 时间 Mamba 严格单向（$t=1,\dots,T$ 顺序处理，无前瞻），频率轴仍双向。

**流式推理状态传播**：
- 时间帧缓冲：保存前 $(K-1)$ 个特征帧供因果卷积使用。
- 卷积状态缓冲：保存 depthwise 1D 卷积内部缓冲区最后 $(d_{\text{conv}}-1)$ 个时间步。
- SSM 隐状态：保存选择性状态空间模型的隐状态 $h_t$。
- 每步消费一帧输入，更新状态，输出一帧增强结果，丢弃过期值。

**知识蒸馏设计**：
- 教师 $\mathcal{T}$：8 层 cTF-Mamba；学生 $\mathcal{S}$：1 层 cTF-Mamba（初始化权重来自预训练教师）。
- 输出级蒸馏损失：
$$\mathcal{L}_{\text{KD}}^{\text{out}} = w_{\text{mag}}\|\hat{\mathbf{M}}^s - \hat{\mathbf{M}}^t\|_2^2 + w_{\text{pha}}\|\hat{\mathbf{P}}^s - \hat{\mathbf{P}}^t\|_2^2 + w_{\text{com}}\|\hat{\mathbf{C}}^s - \hat{\mathbf{C}}^t\|_2^2$$
权重设为 $w_{\text{mag}}=1.0, w_{\text{pha}}=0.3, w_{\text{com}}=0.5$。
- 中间特征蒸馏：取教师第 $i$ 层输出 $\mathbf{H}_i^t = \Phi_i^t(\mathbf{Z}_{i-1}^t)$，学生仅有一层 $\mathbf{H}^s$，教师特征聚合为 $\mathbf{H}_{\text{agg}}^t = \frac{1}{8}\sum_{i=1}^8 \mathbf{H}_i^t$；按样本逐通道/时间/频率归一化后计算 MSE：
$$\mathcal{L}_{\text{KD}}^{\text{feat}} = \|\text{Norm}(\mathbf{H}^s) - \text{Norm}(\mathbf{H}_{\text{agg}}^t)\|_2^2$$
- 渐进 ramp-up：$\gamma(k) = \min\left(\frac{k}{K_{\text{ramp}}}, 1\right)$，$K_{\text{ramp}}$ 为总步骤的 10%；总损失：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \gamma(k)\left(\lambda_{\text{out}}\mathcal{L}_{\text{KD}}^{\text{out}} + \lambda_{\text{feat}}\mathcal{L}_{\text{KD}}^{\text{feat}}\right)$$
其中 $\lambda_{\text{out}}=0.5, \lambda_{\text{feat}}=0.1$，$\mathcal{L}_{\text{task}}$ 包含幅度、相位、复数谱、时域和一致性项。

## 实验与结果
**数据集**：VCTK-DEMAND（16 kHz），训练集 11,572 对（28 说话人 × 10 噪声类型 × 4 SNR 水平），测试集 824  utterances（2 未见说话人 × 5 未见噪声 × 4 SNR 水平）。

**模型深度与质量-延迟权衡**（Table 2）：
| 模型 | PESQ | CSIG | CBAK | COVL | STOI |
|------|------|------|------|------|------|
| 1-layer（朴素） | 3.06 | 4.38 | 3.61 | 3.79 | 0.94 |
| 2-layer | 3.19 | 4.51 | 3.66 | 3.93 | 0.95 |
| 4-layer | 3.29 | 4.59 | 3.76 | 4.03 | 0.95 |
| 8-layer | 3.32 | 4.64 | 3.72 | 4.08 | 0.95 |
| 8→1 蒸馏 | 3.18 | 4.43 | 3.68 | 3.89 | 0.95 |
| 8→2 蒸馏 | 3.22 | 4.55 | 3.69 | 3.97 | 0.95 |

- 1→2 层带来最大 PESQ 提升（+0.13），4 层后边际收益递减，5 层不再优于 4 层。
- 8 层教师达到最优 PESQ=3.32，但 RTF=0.29；1 层朴素基线 RTF=0.11 但 PESQ 仅 3.06。

**蒸馏效果**（8 层 → 1 层）：
- PESQ 从 3.06 提升至 3.18（+0.12），恢复教师-学生差距的 46.2%。
- CSIG 恢复 19.2%，CBAK 恢复 63.6%，COVL 恢复 34.5%。
- RTF 保持 0.11（与朴素 1 层相同），比 8 层教师快约 2.75×（0.29→0.11）。

**与基线对比**（Table 3，VCTK-DEMAND 25 ms 延迟）：
- RT-SEMamba（8→1）：PESQ=3.18，COVL=3.89，STOI=0.95，参数 1.05M，延迟 25 ms。
- 相比 DeepFilterNet2（PESQ=3.08，延迟 40 ms）、FRCRN（PESQ=3.21，延迟 30 ms）、aTENNuate（PESQ=3.27，延迟 46.5 ms），在更严格延迟预算下取得竞争力结果。

**混合 Mamba-Transformer 教师探索**（Table 4）：
- 5 层中替换第 4 块为 cTF-Transformer 可将 PESQ 从 3.27 提升至 3.32，与 8 层全 Mamba 持平，但需额外管理 0.5 s KV cache，故最终选择纯 Mamba 教师。

## 相关工作脉络
1. **SEMamba [34]**：本文前身，非因果离线语音增强 Mamba 模型；本文将其改造为完全因果流式版本，并加入蒸馏压缩。
2. **Mamba/SSM 语音处理**：SPMamba [35]、Mamba-SEUNet [36]、SepMamba [37]、Hybrid Mamba-SE [38]、Universal Mamba-SE [39] 等探索 SSM 在语音分离/增强中的应用，但均非针对流式部署设计；本文明确面向 streaming 场景。
3. **实时语音增强基线**：DeepFilterNet2/3 [47,49]、PercepNet [43]、DCCRN+ [44]、FullSubNet+ [45]、FRCRN [48] 等因果/低延迟模型；本文在同等或更严延迟（25 ms vs 30-40 ms）下取得竞争力性能。
4. **知识蒸馏在 SE 中的应用**：深度压缩方案 [40,41] 已有先例；本文创新在于联合复数频谱输出蒸馏与归一化中间特征蒸馏，并配合渐进 ramp-up。
5. **Jamba 混合架构 [51]**：LLM 中交替 Mamba 与 Transformer 块的设计启发本文尝试 hybrid teacher；发现单 Transformer 块可显著提升中等深度模型，但引入额外缓存管理开销。

## 局限性与未来方向
1. **蒸馏仅在 VCTK-DEMAND 上验证**：未在其他基准（如 DNS Challenge）或真实场景数据上评估泛化性；教师可能也在较小数据集上训练。
2. **固定 25 ms 延迟约束**：对于更低延迟（如 10 ms）或更高音质需求场景，未做探索。
3. **纯 Mamba 教师选择**：虽然 hybrid 5 层模型可与 8 层 Mamba 持平，但作者因 KV cache 复杂度放弃；未来可研究如何高效管理 streaming 下的 KV cache。
4. **未讨论极端低资源场景**：1 层学生已非常轻量，但更小参数量（如 <1M）或 CPU/嵌入式部署未涉及。
5. **蒸馏仅针对深度压缩**：未探索宽度压缩、量化、剪枝等其他模型压缩技术与蒸馏的组合。

## 研究启发与可借鉴点
1. **渐进式蒸馏 ramp-up 策略**：前 10% 训练步骤线性引入蒸馏损失，可有效防止学生早期过拟合教师噪声输出，适用于其他 teacher-student 蒸馏任务。
2. **复数谱联合蒸馏**：同时蒸馏幅度、相位、复数谱三个级别输出，比仅蒸馏复数谱或幅度谱更全面，可迁移至其他复数域语音处理任务。
3. **流式状态管理三缓冲设计**：temporal buffer + conv state buffer + ssm state 的分离管理清晰且高效，可作为流式 SSM 模型部署的参考模板。
4. **Mamba 替代 Transformer 的 streaming 优势**：固定大小隐状态 vs KV cache 增长，在长序列和边缘设备上优势明显，值得在 ASR、TTS 等任务中验证。
5. **质量-延迟帕累托分析视角**：通过 RTF 与 PESQ 联合分析展示蒸馏如何"上移"帕累托前沿，为模型选择提供直观决策依据。

## 关键术语表
**RT-SEMamba**：本文提出的实时语音增强模型，基于因果时频 Mamba 块，支持流式推理。
**cTF-Mamba**：causal Time-Frequency Mamba block，时间轴单向、频率轴双向的选择性状态空间模型块。
**Progressive Knowledge Distillation**：渐进式知识蒸馏，通过 ramp-up 函数逐步引入蒸馏损失以加速收敛。
**Algorithmic Latency**：算法延迟，由模型结构决定的最小时延，本文固定为 25 ms（一帧窗口）。
**RTF (Real-Time Factor)**：实时因子，处理每秒音频所需计算时间，RTF<1 表示可实时运行。
**Complex STFT**：复数短时傅里叶变换，同时保留幅度谱 M 和相位谱 P 的时频表示。
**Selective State-Space Model**：选择性状态空间模型（Mamba 核心），通过输入相关的 $\mathbf{B}, \mathbf{C}$ 参数实现内容感知的状态更新。

## 可复现要素
- **数据集**：VCTK-DEMAND，公开可用。
- **代码**：论文声明将开源，GitHub 地址 https://github.com/RoyChao19477/RT-SEMamba。
- **权重**：教师与学生模型权重未明确说明是否公开，需查阅代码仓库。
- **关键超参**：STFT hop=100, W=400, center=False；采样率 16 kHz；蒸馏权重 $w_{\text{mag}}=1.0, w_{\text{pha}}=0.3, w_{\text{com}}=0.5$；$\lambda_{\text{out}}=0.5, \lambda_{\text{feat}}=0.1$；ramp-up 步数 $K_{\text{ramp}}$ 为总步数 10%。
- **硬件**：NVIDIA RTX 5090 GPU 评估 RTF。
