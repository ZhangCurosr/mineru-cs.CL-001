---
title: "Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence"
source: https://arxiv.org/pdf/2608.10720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:41:45"
field: "全模态对话与视频生成"
keywords: ["omni-modal dialogue", "visual thought plan", "multi-codebook speech units", "prefix streaming", "video diffusion distillation", "audio-video synchronization", "heterogeneous data training"]
innovations: ["结构化VTP将对话上下文转化为显式五字段视觉指导", "原生16码书语音单元作为共享声学-时间接口连接语音合成与视频生成", "前缀流式蒸馏机制通过停梯度clean latent前缀抑制累积尾部退化"]
benchmarks: ["VoiceBench CommonEval", "OmniCharacter", "SpeakerVid"]
---

# 论文速读：Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

## 一句话总结
Ex-Omni-2D 提出了一种**原生视觉在场的全模态对话框架**，在接收多模态查询、参考图像与参考音频后，通过结构化“视觉思维计划（VTP）”与**原生多码书语音单元接口**，联合生成响应文本、个性化语音与同步视频。该方法解耦了语音生成与视频生成的监督路径，并将全序列视频教师蒸馏为**前缀流式（Prefix Streaming）块因果学生**，实现了增量高效生成与质量‑效率可权衡的部署点。

## 研究问题与动机
1. **视觉缺席**：现有全模态对话系统能理解多模态输入并合成语音回复，但响应始终缺乏“视觉化身（avatar）”呈现。
2. **视频生成依赖外部波形**：已有口型驱动/Talking-avatar 方法通常以已完成或外部提供的音频波形作为驱动信号，无法从对话状态中直接派生响应特定的视觉意图。
3. **联合音频-视频生成缺乏对话条件**：Joint audio-video 生成模型多由 prompt 或参考信号驱动，而非响应多模态用户查询，缺少“对话状态→视觉行为”的映射机制。
4. **大规模对话‑语音‑视频配对数据稀缺**：直接构建 query–text–speech–video 全配对数据成本高，需要一种可通过异构数据分阶段训练、并在推理时通过紧凑接口重新连接的方案。

## 核心贡献（创新点）
1. **结构化视觉思维计划（VTP）**：将对话上下文转化为包含首帧场景、整体场景、情绪、运动风格与细节动作五个字段的显式视觉指导，区别于以往依赖人工 prompt 或跨模型隐藏状态直接映射的做法。
2. **原生多码书语音单元共享接口**：以 16 码书、12.5 Hz 的离散语音单元同时作为语音合成与视频帧对齐条件，避免完成级波形重编码，并支持从异构语音、对话与头像视频数据中分 pathway 学习。
3. **全序列教师 → 前缀流式学生的蒸馏**：在 AnyFlow 的 flow-map 与 on-policy 分布匹配基础上，引入**前缀流式（Prefix Streaming）机制**——每个后续窗口的首个 clean latent 作为停梯度前缀，减少累积性尾部主体退化，同时保持块因果增量生成。
4. **质量‑效率双Operating Point**：50 步全序列教师提供高质量主渲染，2/4/8 步流式学生在吞吐量与质量之间提供可控权衡；四步推理在四 GPU  pipeline 下达到 E2E RTF 1.293（400×720/720×400）。

## 方法详解
- **输入**：多模态查询 $\boldsymbol{x}=(x_t, x_s)$（文本/语音）、参考图像 $I^{\mathrm{ref}}$、参考音频 $a^{\mathrm{ref}}$。
- **对话骨干**：Qwen3‑8B，遵循 `<thinking>`（内含 VTP $\mathbf{p}$）–`<response>`（响应文本 $\mathbf{y}$）结构化协议。
- **VTP**：$\mathbf{p}=(p_{\mathrm{first}}, p_{\mathrm{scene}}, p_{\mathrm{emotion}}, p_{\mathrm{style}}, p_{\mathrm{motion}})$，经文本编码器得 $\mathbf{L}^{\mathrm{vtp}}$，作为视频生成的高层语义条件。
- **共享声学接口**：语音生成器（Qwen3‑TTS‑0.6B）以响应隐藏状态 $\mathbf{H}_{\mathbf{y}}^{\ell}$、文本 $\mathbf{y}$ 与参考声纹 $\mathbf{s}^{\mathrm{ref}}$ 预测多码书单元 $\mathbf{U}\in\mathbb{N}^{N\times 16}$；单位 12.5 Hz，每 acoustic frame 重复对齐至 25 FPS 视频时间线（$\mathbf{A}_{2n-1}=\mathbf{A}_{2n}=\widetilde{\mathbf{A}}_n$）。
- **全序列教师（Video Generator）**：基于 Wan2.1‑T2V‑1.3B + OmniAvatar‑1.3B LoRA，接收参考 latent $\mathbf{R}^{\mathrm{ref}}$、帧对齐声学条件 $\mathbf{A}$、VTP 条件 $\mathbf{L}^{\mathrm{vtp}}$，使用双向时序注意力与 50 步 flow‑matching 去噪。
- **前缀流式学生**：采用块因果注意力（窗口内双向、窗口间因果）；初始窗口 $[ \mathbf{R}^{\mathrm{ref}}, \mathbf{Z}_{1:3}^{(0)} ]$，后续窗口 $[ \mathrm{sg}(\widehat{\mathbf{Z}}_3^{(m-1)}), \mathbf{Z}_{1:3}^{(m)} ]$，每 6 个 acoustic frame（0.48 s）生成 12 视频帧；蒸馏分两阶段：Phase I 学习 source‑destination flow map，Phase II 进行 on‑policy 块级 rollout 与分布匹配。
- **训练阶段**：
  1. Stage 1（语音接口对齐）：800K ASR + 1M TTS，冻结 LLM。
  2. Stage 2（全模态响应适配）：InstructS2S‑200K + OmniCharacter，更新 LLM/投影/语音生成器，引入 VTP–response 协议。
  3. Stage 3（头像视频实现）：140K SpeakerVid 过滤片段，训练全序列教师。
  4. Stage 4（流式学生蒸馏）：冻结教师，两阶段流式蒸馏。

## 实验与结果
- **数据集/基准**：VoiceBench CommonEval（200 条语音 QA）、OmniCharacter（400 轮多轮对话）、SpeakerVid（训练过滤数据）。
- **评估指标**：音频 PQ/CU/SIM；视频 SC/IQ/DD；A‑V Sync Sync‑C；对话 Fluency/Coherency/Consistency 等 12 维。
- **关键数字**：
  - 对话能力：Ex‑Omni‑2D 在 OmniCharacter 上 Fluency=3.812、Coherency=4.100、Consistency=3.902，平均 3.283，高于 Qwen3‑8B 的 3.264。
  - 语音 QA（VoiceBench）：AlpacaEval 4.28、CommonEval 3.71、BBH 58.70（仅次于 Qwen2.5‑Omni）。
  - 视频质量（Teacher，表 1）：SC 94.62、IQ 67.31、DD 72.00、Sync‑C 4.95、SIM 0.417。
  - 流式学生四步：SC 93.65、IQ 57.40、DD 32.00、Sync‑C 4.00；**E2E RTF=1.293**（四 GPU，400×720/720×400），首音频 2.308 s、首视频 3.142 s。
  - 前缀流式消融：200 样本上 SC 92.85→93.65、IQ 55.18→57.40、Sync‑C 3.69→3.90；尾部一致性误差降低 21.4%。
  - VTP 消融：移除响应特定 VTP 后 SC 94.62→93.58、Sync‑C 4.95→4.65；移除个性化参考语音后 SIM 0.417→0.015。
- **结论**：教师提供高质量主渲染，学生提供效率权衡；VTP 与多码书接口有效连接异构监督路径，无需大规模全配对数据。

## 相关工作脉络
1. **EchoMimic、StableAvatar、OmniAvatar、FantasyTalking**：音频驱动头像/视频生成，依赖已完成波形与参考图像，缺乏从对话状态派生视觉意图的机制。
2. **Universe‑1、UniAVGen**：联合音频‑视频生成模型，但以 prompt/参考信号为条件，未面向多模态用户查询的响应式生成。
3. **Qwen2.5‑Omni、Llama‑Omni、Mini‑Omni2、VITA‑1.5、SLAM‑Omni**：全模态对话大模型，能理解多模态输入并合成语音，但均缺少原生视频响应能力。
4. **StreamAvatar**：流式扩散模型用于实时头像，本文借鉴其块因果设计并引入前缀流式与 flow‑map 蒸馏。
5. **AnyFlow**：any‑step 视频扩散模型的 flow‑map 与 on‑policy 蒸馏目标，本文将其适配至头像流式场景并添加前缀机制。
6. **SpeakerVid**：大规模高质量语音‑视频双交互式数据集，本文以其 140K 过滤片段训练视频生成 pathway。

## 局限性与未来方向
1. **参考说话人相似度仍有提升空间**：SIM 0.417 表明个性化语音还原尚未达到顶尖 TTS 水平。
2. **VTP 为高层语义指导而非独立控制信号**：最终视频由 VTP 与声学条件联合决定，固定 Text CFG/Audio CFG 尺度难以在所有响应中自适应平衡。
3. **VTP 生成带来语言‑能力权衡**：引入 VTP 后 CommonEval 下降 0.11、BBH 下降 2.40 分，共享自回归通道存在干扰。
4. **长时程分析有限**：仅评估了主体稳定性与面部动力学规律，未覆盖更复杂的行为连贯性。
5. **全序列教师计算昂贵**：50 步双向去噪不适合交互部署；流式学生虽提升吞吐，但单请求 E2E RTF 仍 >1（四 GPU），尚未达到端到端实时交互。
6. **未来方向**：隔离规划器、自适应跨模态引导、扩展因果视频 backbone、改进 few‑step 蒸馏以缩小质量‑效率 gap。

## 研究启发与可借鉴点
1. **结构化中间表示（VTP）解耦规划与渲染**：将视觉意图显式化为受限字段，既便于自回归监督，又为下游生成器提供可解释条件，可迁移至其他多模态生成任务。
2. **离散语音单元作为跨模态共享接口**：16 码书 12.5 Hz 单元同时服务 TTS 解码与视频帧对齐，避免了波形后处理重编码，为语音‑视频联合生成提供轻量时间契约。
3. **前缀流式蒸馏机制**：通过停梯度 clean latent 前缀在块边界提供显式状态锚定，有效抑制自回归累积漂移，可推广至其他视频/序列生成流式场景。
4. **分阶段异构数据训练**：Stage 1‑3 分别利用 ASR/TTS、对话、头像视频数据，仅在推理时通过 VTP 与语音单元重新连接 pathway，缓解全配对数据稀缺问题。
5. **质量‑效率可权衡 Operating Point**：固定 checkpoint 下仅改变去噪步数（2/4/8）即可连续调节 SC/IQ/DD/Sync‑C 与 FPS/RTF，便于实际部署按需选择。

## 关键术语表
- **Ex‑Omni‑2D**：本文提出的全模态对话框架，支持原生视觉在场的响应生成。
- **Visual Thought Plan (VTP)**：结构化五字段视觉思维计划，描述首帧场景、整体场景、情绪、运动风格与动作细节。
- **Native Multi‑codebook Speech Units**：16 码书、12.5 Hz 的离散语音单元，作为语音合成与视频条件共享接口。
- **Prefix Streaming**：流式学生中每个后续窗口prepend前一段最终 clean latent（停梯度）作为前缀，减少累积尾部退化。
- **Flow‑matching**：视频生成采用的扩散目标参数化，预测 $\epsilon-\mathbf{x}_0$ 流场。
- **On‑policy Distribution Matching**：蒸馏阶段在学生自采样轨迹上进行分布匹配，保持流式 rollout 一致性。
- **CommonEval / VoiceBench**：语音问答与多模态对话评估基准，包含 AlpacaEval、CommonEval、BBH 等子指标。
- **OmniCharacter**：400 轮多轮角色对话评测集，评估十二维度对话质量。

## 可复现要素
- **数据集**：SpeakerVid（训练过滤，约 140K 片段）、InstructS2S‑200K、OmniCharacter、LibriSpeech（800K ASR）、Emilia（1M TTS 合成源）；论文未声明公开，但基准多为已公开数据集。
- **代码/权重**：项目页面 https://logo‑cuhksz.github.io/Ex‑Omni‑2D；模型基于 Qwen3‑8B、Qwen3‑TTS‑0.6B、Wan2.1‑T2V‑1.3B、OmniAvatar‑1.3B LoRA，具体开源状态论文未明确声明。
- **关键超参**：去噪步数 Teacher=50、Student=2/4/8；学习率 Stage1 Proj. 1e‑3/Gen. 1e‑4、Stage2 LLM 1e‑6/Proj.Gen. 2e‑5、Stage3 2e‑5、Stage4 Phase I 5e‑5/Phase II 2e‑6；batch 1‑16/GAcc 1；flow‑map 权重 0.5/0.25；Text CFG=3.5、Audio CFG=8.5（Teacher）；12.5 Hz 语音单元对齐 25 FPS 视频。
