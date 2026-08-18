---
title: "Confucius4-TTS-Transcript-Free-Cross-Lingual-Zero-Shot-TTS-w"
source: https://arxiv.org/pdf/2608.11650v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:43:55"
---

# 论文速读：Confucius4-TTS-Transcript-Free-Cross-Lingual-Zero-Shot-TTS-w

## 一句话总结
本文提出 Confucius4-TTS，一个支持 14 种语言的端到端跨语言零样本 TTS 系统，无需参考音频的转录文本即可实现高质量音色克隆；模型采用 T2S + 条件流匹配 S2A 两阶段架构，默认支持无转录参考克隆，并可在有转录时无缝切换为续写克隆模式，在多项公开与内部评测中达到顶尖水平。

## 研究问题与动机
- 现有零样本 TTS 普遍依赖“文本–音频配对提示”，推理时必须提供参考音频的准确转录，而野外采集的语音（尤其低资源语言与方言）往往无可用转录。
- 跨语言声音克隆需在目标语言生成清晰自然的语音的同时保持参考说话人身份，转录依赖大幅限制了真实场景的落地。
- 直接借用预训练说话人验证编码器（如 ECAPA-TDNN、CAM++）提取音色，其特征空间与生成任务目标不一致，难以充分适配 TTS 的韵律与音质需求。
- 现有无转录方案多依赖强制对齐或合成提示对（如 Cross-Lingual F5-TTS、X-Voice），引入额外模块或数据开销，缺乏统一且高效的端到端双模式推理框架。

## 核心贡献（创新点）
- **无转录跨语言零样本 TTS 系统**：Confucius4-TTS 默认仅凭参考音频与目标文本即可在 14 种语言间完成音色克隆，摆脱对参考转录的依赖；与 VALL-E、CosyVoice 等需配对提示的方法本质不同，后者推理时必须提供转录文本。
- **联合训练的 SSL 说话人编码器**：采用 ECAPA-TDNN 对 w2v-BERT 2.0 自监督表征进行池化，并与 T2S 模块联合优化，使提取的音色嵌入更契合语音合成任务而非纯验证任务；区别于单独使用 Cam++ 或冻结预训练 WavLM 作为全局说话人条件的方法。
- **双推理配方统一架构**：同一模型支持参考克隆（无转录）与续写克隆（有转录）两种布局，前者保留更大风格自由度，后者在拥有转录时进一步提升相似度；与仅采用单一 prefix/suffix 条件模式（如 VoxCPM2、MOSS-TTS）的系统相比更具工程适应性。
- **T2S 隐状态桥接 S2A 的条件设计**：S2A 不仅接收离散语义 token，还接入 T2S 在语义位置处的连续隐状态，有效缓解离散量化带来的信息瓶颈；区别于仅依赖 token embedding 或单纯 prompt mel 的声学解码方案。

## 方法详解
- **整体架构**：分为自回归 Text-to-Semantic (T2S) 模块、基于条件流匹配（CFM）的 Semantic-to-Acoustic (S2A) 模块，以及 BigVGAN 声码器。T2S 生成语义 token 序列，S2A 将其转化为 mel-spectrogram，声码器重建波形。
- **T2S 模块与输入构造**：采用 24 层 Decoder-only Transformer（隐藏维度 1280）。训练输入序列为 `[e^r, H^txt, ⟨BOS⟩, E_sem^T2S(y_1), …, E_sem^T2S(y_N)]`，其中 `e^r` 为参考音频池化得到的说话人嵌入，`H^txt` 为冻结 LLM 词表经轻量 MLP 投影后的目标文本嵌入，`y` 为 MaskGCT 语义 codec 提取的离散 token 序列。
- **联合说话人编码器**：使用 w2v-BERT 2.0 作为 SSL 编码器提取帧级特征 `H^ssl ∈ R^{T_ref × D_ssl}`，经 ECAPA-TDNN 注意力统计池化输出固定维度嵌入 `e^r ∈ R^d`。输出维度与 T2S 隐藏维度一致，无需额外投影；编码器与 T2S 骨干、文本投影器、语义 token 嵌入表联合优化，预训练文本词表保持冻结。
- **双模式推理布局**：参考克隆模式下 T2S 输入为 `[e^r; H^txt; ⟨BOS⟩] → y`；续写克隆模式下追加参考文本嵌入与参考语义 token，即 `[e^r; H^txt,r; H^txt; ⟨BOS⟩; y^r] → y`。续写模式能进一步锁住音色但会略微牺牲流畅度。
- **S2A 条件设计与长度控制**：语义条件序列 `c^sem = [E_sem^S2A(y); H^T2S]` 经长度调节器上采样至 mel 帧率。冻结的 CAM++ 说话人验证模型提取全局嵌入 `g`，重复拼接后形成帧级条件 `c`。同时输入参考 mel-spectrogram prompt `m^p` 作为声学上下文，提示帧对应的语义条件替换为可学习占位符以避免泄露显式语义。
- **流匹配训练与 CFG 推理**：S2A 沿最优传输路径训练，向量场目标为 `ω_t = m_1 − m_0`，DiT 骨干通过 L1 损失拟合该场。训练时对语义条件、全局说话人嵌入、prompt mel 联合随机 dropout，使推理时可启用 classifier-free guidance (CFG)：`ṽ = (1+α)·v_cond − α·v_uncond`，其中 α=0.7。推理时以 Euler ODE 求解器积分 25 步生成 mel，再由 BigVGAN 转波形。

## 实验与结果
- **数据集与配置**：约 500k 小时 14 语言混合语音（中/英/日/韩/德/法/西/印尼/意/泰/葡/俄/马来/越南），含真实与合成数据；经分离去噪、VAD 分割、多系统 ASR 过滤（CER/WER < 2.5%）、说话人聚类后训练。T2S 与 S2A 分两阶段在 32× NVIDIA A40 上训练，AdamW + 余弦学习率调度。
- **评测基准与指标**：CV3-Eval（6 向跨语言）、X-Voice（7 向源→中）、Seed-TTS-eval（中英内语言）、MiniMax-MLS-Test（11 语言跨语言）。使用 Whisper large-v3 / Paraformer 计算 WER/CER，WavLM-large 计算 SIM（余弦相似度）。
- **主要结果**：
  - CV3-Eval 跨语言平均 WER 达 **3.73%**，6 个方向中 4 项最佳（如 ja→zh 仅 4.87%，对比 CosyVoice 2 的 48.10% 优势显著）。
  - X-Voice 跨语言 CEF 在 7 个
