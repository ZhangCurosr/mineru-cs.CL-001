---
title: "Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence"
source: https://arxiv.org/pdf/2608.10720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:42:13"
field: "多模态对话与生成"
keywords: ["全模态对话", "视觉规划", "流式视频生成", "多模态蒸馏", "语音-视频同步"]
innovations: ["结构化 Visual Thought Plan 视觉规划与响应解耦", "原生多码本语音接口桥接异构模态训练路径", "Prefix Streaming 机制缓解流式视频累积退化"]
benchmarks: ["CommonEval", "OmniCharacter", "VoiceBench"]
---

# 论文速读：Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

## 一句话总结
论文针对全模态对话系统视觉"无实体感"问题，提出 Ex-Omni-2D 框架，通过结构化 Visual Thought Plan (VTP) 与原生多码本语音接口，实现文本、个性化语音与参考条件视频的协同生成，并以全序列 Teacher 蒸馏为几步流式 Student，在质量与效率间提供互补操作点。

## 研究问题与动机
- **视觉缺失**：现有全模态对话模型能理解语音/文本并生成自然语音回复，但响应缺乏视觉化身（visually disembodied）。
- **视觉意图难以从对话状态推导**：现有说话人动画依赖已完成的外部波形和人工提示，无法直接从对话状态推导响应特定的视觉意图。
- **缺乏大规模成对数据**：查询-文本-语音-视频完全配对的全模态对话数据难以获取，需要一种无需完全成对监督的训练方案。
- **全序列模型无法满足交互式部署**：全序列视频扩散推理缓慢，需要高效的流式增量生成方案。

## 核心贡献（创新点）
- **结构化 VTP 视觉规划**：引入 Visual Thought Plan 将对话上下文转化为显式的场景、情感、运动风格等视觉指导，区别于开放式思维链或手动提示工程。
- **原生多码本语音接口桥接**：提出基于 Qwen3-TTS 的 16 码本语音单元作为共享声学-时序接口，语音与视频路径可分别利用异构数据训练，而不在推理时依赖完整的波形后处理。
- **Teacher-Student 蒸馏与流式方案**：将 50 步全序列 Teacher 蒸馏为 2/4/8 步块因果流式 Student，引入 Prefix Streaming 机制以减少累积晚期块退化。
- **多阶段分步训练协议**：设计了语音接口对齐→全模态响应适配→化身视频实现→流式学生蒸馏的四阶段训练流程，解耦了不同模态的监督信号。

## 方法详解
- **多模态输入编码**：文本通过语言模型嵌入，语音经编码器投影至同一隐空间，形成 $\mathbf{E}_x$；参考图像通过 Qwen3-VL-2B 视觉塔获取对话模型条件 $\mathbf{z}_I^{\text{llm}}$，并通过 Wan 3D VAE 获取外观参考潜变量 $\mathbf{R}^{\text{ref}}$。
- **结构化 VTP 协议**：LLM（Qwen3-8B）遵循 `<thinking>`-$\mathbf{p}$-`</thinking>`-`<response>`-$\mathbf{y}$-`</response>` 协议，VTP 包含五个原子字段：首帧场景、整体场景、情感、运动风格、详细动作。
- **多码本语音接口**：语音生成器（Qwen3-TTS-0.6B）基于响应隐状态生成 $N \times 16$ 的多码本单元序列 $\mathbf{U}$；每帧 12.5 Hz，视频 25 FPS，每个声学特征重复两次以对齐视频帧：$\mathbf{A}_{2n-1} = \mathbf{A}_{2n} = \widetilde{\mathbf{A}}_n$。
- **全序列 Teacher**：基于 Wan2.1-T2V-1.3B，使用双向时间上下文，接收参考潜变量、帧对齐声学条件、VTP 文本编码的三元条件，采用 flow-matching 损失。
- **Prefix Streaming Student**：采用 AnyFlow 的流映射和 on-policy 蒸馏；每窗口保持 4 个潜变量槽位，首个窗口为 $[\mathbf{R}^{\text{ref}}, \mathbf{Z}_{1:3}^{(0)}]$，后续窗口为 $[\text{sg}(\widehat{\mathbf{Z}}_3^{(m-1)}), \mathbf{Z}_{1:3}^{(m)}]$，通过停止梯度防止前缀漂移。
- **四阶段训练**：Stage 1（ASR/TTS 数据对齐语音接口）、Stage 2（Omni-modal 响应适配，引入 VTP 协议）、Stage 3（140K SpeakerVid 片段训练视频生成器）、Stage 4（Teacher 蒸馏为流式 Student，Phase I 流映射学习、Phase II on-policy 分布匹配）。

## 实验与结果
- **数据集**：CommonEval（VoiceBench，200 个语音查询）、OmniCharacter（400 轮对话）、SpeakerVid（过滤后约 140K 片段）。
- **音视频质量**（CommonEval，Teacher vs 最强下游基线）：
  - SC: Teacher 94.62 vs echomimic 97.87；IQ: 67.31 vs UniAVGen 68.02；DD: 72.00 vs echomimic 74.00；Sync-C: 4.95 vs OmniAvatar 5.64。
  - SIM（Seed-TTS）: 0.417，体现个性化语音相似度。
- **多轮对话质量**（OmniCharacter）：Fluency 3.812、Coherency 4.100、Consistency 3.902，三项平均 3.938，高于 Qwen3-8B 基线的 3.537；十二维均值 3.283 vs 3.264。
- **语音 QA**（VoiceBench）：AlpacaEval 4.28、CommonEval 3.71、BBH 58.70，仅次于 Qwen2.5-Omni（4.49/3.93/60.80）。
- **效率指标**（四 GPU，400×720/720×400）：
  - Teacher（50 步）：FPS 1.409，E2E RTF 26.917。
  - Student 4 步：FPS 26.512，E2E RTF 1.293，首段可播放视频延迟 3.142 秒。
  - Student 2 步：FPS 39.546，E2E RTF 1.201。
- **Prefix Streaming 有效性**：较无前缀版本，SC 提升 +0.80（92.85→93.65），IQ 提升 +2.22（55.18→57.40），Sync-C 提升 +0.21（3.69→3.90）；尾部一致性误差斜率降低 23.5%（0.00783→0.00599/chunk）。

## 相关工作脉络
- **Omni-modal LLMs**（Qwen2.5-Omni、Moshi、VITA-1.5）：侧重语音-文本交互，缺乏原生视频生成能力，定位为纯音频输出对话助手。
- **音频驱动头像动画**（echomimic、StableAvatar、OmniAvatar、FantasyTalking）：以已完成波形为驱动条件，不直接从对话状态生成视觉意图，属后处理渲染管道。
- **联合音视频生成**（MM-Diffusion、SyncFlow、UniForm、UniVerse-1、UniAVGen）：从提示/参考信号生成配对音视频，未对接多模态查询的响应式对话场景。
- **Speech Language Models**（SpeechT5、GLM-4-Voice、SpeechGPT-Gen）：消除 ASR-TTS 级联，但同样聚焦纯语音输出，无视觉化身路径。
- **流式视频生成**（StreamAvatar）：探索实时交互式头像生成，但未与对话规划模块深度耦合。

## 局限性与未来方向
- **参考说话人相似度**仍有提升空间，SIM 0.417 表明语音个性化可进一步加强。
- **VTP 控制信号不够独立**：最终视频由 VTP 与声学条件联合决定，固定 Text/Audio CFG 比例无法在所有响应中保持一致的自然度。
- **长序列累积退化**：虽经 Prefix Streaming 缓解，但 Student 在 IQ、DD、Sync-C 上仍显著低于 Teacher。
- **实时性不足**：4 步推理 E2E RTF 1.293 > 1，首段视频延迟 3.142 秒，尚未达到真正意义上的端到端实时交互。
- **VTP 生成引入语言能力权衡**：Table 7 显示添加 VTP 监督后 CommonEval 下降 0.11、BBH 下降 2.40，未来需探索规划器隔离或参数解耦。
- **自适应跨模态引导缺失**：当前固定引导尺度未根据响应内容动态调整。

## 研究启发与可借鉴点
- **分阶段异构数据训练**：利用语音/对话数据训练语言和语音路径，利用有 VTP 标注的视频片段训练视频路径，避免了对大规模完全配对数据的需求。
- **结构化中间表示的解耦设计**：VTP 作为显式、可监督、可检查的语义规划层，为多模态生成提供了可解释的中间接口。
- **共享声学-时序接口桥接独立模态管道**：多码本语音单元同时服务于语音合成与视频条件，使语音和视觉路径可在不同数据源上分别训练，再通过推理时的接口重新连接。
- **Prefix Streaming 机制应对累积退化**：通过剥离前一块终态干净潜变量作为后续窗口的固定前缀，并结合停止梯度和重叠去重缓存更新，有效缓解了自回归流式生成的晚期质量下降问题。
- **流映射+on-policy 蒸馏的轻量部署策略**：结合 AnyFlow 的双时间流映射和分布匹配，以几步推理逼近全序列质量，为视频生成模型的实用化部署提供了参考范式。

## 关键术语表
- **Visual Thought Plan (VTP)**：结构化五字段视觉规划（首帧场景、整体场景、情感、运动风格、动作描述），将对话上下文转化为显式的视频生成指导。
- **多码本语音单元**：Qwen3-TTS 生成的 $N \times 16$ 离散声学表示，作为语音合成与视频帧对齐条件的共享接口。
- **Prefix Streaming**：流式视频中每块复用前块末尾干净潜变量作为前缀，通过停止梯度防止漂移，减少累积晚期退化。
- **Flow-matching**：视频扩散模型采用的流匹配目标，预测噪声到干净的流场 $\epsilon - \mathbf{x}_0$。
- **On-policy distillation**：流式学生蒸馏的第二阶段，在学生的自回归 rollout 上进行分布匹配，同时保留流映射损失。
- **RTF (Real-Time Factor)**：推理时间与生成媒体时长之比，RTF < 1 表示快于实时。

## 可复现要素
- **数据集**：SpeakerVid-5M（论文过滤后约 140K 片段，用于 Stage 3 视频训练）；InstructS2S-200K、OmniCharacter（Stage 2）；LibriSpeech、Emilia（Stage 1 ASR/TTS）。论文未明确声明公开状态，但提到 Project Page 及 GitHub 仓库。
- **代码/权重**：Project Page 为 https://logo-cuhksz.github.io/Ex-Omni-2D；论文提及 arXiv 源码可能开源，但需确认具体发布状态。
- **关键超参**：Teacher 50 步 FlowMatch 去噪，CFG Text=3.5/Audio=8.5；Student 4 步，CFG=1.0/1.0；流式窗口 121 帧，每块 3 新潜变量；VTP 生成采用温度 0、top-p 1 确定性解码。
