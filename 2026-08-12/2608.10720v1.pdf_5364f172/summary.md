---
title: "Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence"
source: https://arxiv.org/pdf/2608.10720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:42:22"
field: "多模态对话与视听生成"
keywords: ["omni-modal dialogue", "visual thought plan", "multi-codebook speech units", "prefix streaming", "flow-matching video generation", "teacher-student distillation"]
innovations: ["提出 VTP 结构化视觉规划与多 codebook 语音单元作为对话驱动的视频生成接口", "将全序列 Teacher 蒸馏为带 Prefix Streaming 的块因果 Student 以降低累积漂移", "分阶段异质数据训练策略绕过全配对语料需求"]
benchmarks: ["CommonEval", "OmniCharacter", "VoiceBench", "AlpacaEval", "BBH"]
---

# 论文速读：Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

## 一句话总结
Ex-Omni-2D 提出了一种支持视觉存在的 omnimodal 对话框架，通过结构化 Visual Thought Plan (VTP) 和原生多 codebook 语音单元接口，在无需大规模图文音视四模态配对数据的前提下，实现对话上下文驱动的个性化语音与参考图像条件视频响应；论文进一步将全序列 Teacher 蒸馏为带 Prefix Streaming 机制的流式 Student，在质量与效率之间提供可调节的部署选项。

## 研究问题与动机
- **现有 omni-modal 对话模型缺乏视觉存在感**：当前系统能理解多模态输入并合成口语回复，但无法同步生成与对话状态对齐的说话人视频。
- **独立 A2V 方法难以从对话状态派生视觉意图**：既有说话人生成器通常依赖外部已完成的波形或人工编写提示，无法直接从对话上下文推断视觉行为。
- **联合音视频生成未面向对话式响应设定**：现有 joint audio-video 系统多基于提示或参考信号生成内容，而非对多模态用户查询进行响应式生成。
- **全序列视频扩散推理太慢，难以交互式部署**：需要高效的流式实现来支持增量式语音/视频输出，同时避免累积式后期画面退化。

## 核心贡献（创新点）
- **提出 VTP 结构化视觉规划接口**：在对话流程中显式生成 scene/emotion/motion 等五字段计划，使视觉行为与对话状态直接关联；与已有工作相比，其本质差异在于视觉意图来源于对话上下文而非人工提示或后处理模块。
- **引入多 codebook 语音单元作为共享声学-时间接口**：将语音合成与视频帧对齐解耦为同一组单位表示，支持分别来自 ASR/TTS 与 avatar-video 的异质数据监督；区别于以往仅依赖完整波形再编码的做法，该接口避免了高维 LLM 特征到视频的昂贵直接映射。
- **设计分阶段训练策略以绕过全配对数据依赖**：分别用 800K ASR、1M TTS、InstructS2S-200K/OmniCharacter 与 140K SpeakerVid 片段训练语音与视频子路径；与全模态配对监督相比，该方法通过 VTP 与语音单元在推理时重连独立训练的路径。
- **将全序列 Teacher 蒸馏为带 Prefix Streaming 的块因果 Student**：通过在每个 chunk 边界注入前一段干净 latent 作为前缀，降低后期 chunk 累积性主体退化；与通用流式蒸馏相比，该设计显式约束了 latent 空间的边界连续性。
- **提供质量-效率可调节的部署点**：Teacher（50-step）提供高质量离线渲染，Student（2/4/8-step）支持增量生成；四卡 400×720/720×400 下 4-step 达到 E2E RTF=1.293，首段可播放视频chunk 在 3.142s 后可用。

## 方法详解
- **多模态查询与角色条件接入**：文本 $x_t$ 与语音 $x_s$ 经各自编码器投影到统一隐空间；参考图像经 Qwen3-VL-2B 视觉塔得到 $\mathbf{z}_I^{\text{llm}}$ 进入对话模型，同时经 Wan 3D VAE 得到外观参考 latent $\mathbf{R}^{\text{ref}}$ 供视频生成器独立使用；参考音频经 speaker encoder 提取 $\mathbf{s}^{\text{ref}}$。
- **Visual Thought Plan (VTP) 结构化生成**：LLM（Qwen3-8B）在 `<thinking>` 块内输出五字段 plan $\mathbf{p}=(p_{\text{first}}, p_{\text{scene}}, p_{\text{emotion}}, p_{\text{style}}, p_{\text{motion}})$，再通过文本编码器得到 $\mathbf{L}^{\text{vtp}}$；随后在 `<response>` 块内生成面向用户的回复文本 $\mathbf{y}$，并保留该段最后一层隐藏状态 $\mathbf{H}_{\mathbf{y}}^{\ell}$。
- **多 codebook 语音接口**：Speech Generator（Qwen3-TTS-0.6B）以 $\mathbf{H}_{\mathbf{y}}^{\ell}$、$\mathbf{y}$ 和 $\mathbf{s}^{\text{ref}}$ 为条件自回归预测 $\mathbf{U} \in \mathbb{N}^{N \times C}$（$C=16$），首码本逐帧自回归，其余 15 个码本细化声学内容；声学特征以 12.5Hz 生成，每帧重复两次对齐 25FPS 视频。
- **全序列 Teacher（Video Generator）**：基于 Wan2.1-T2V-1.3B + OmniAvatar-1.3B LoRA，条件包括 $\mathbf{R}^{\text{ref}}$、帧对齐声学条件 $\mathbf{A}$ 与 $\mathbf{L}^{\text{vtp}}$；采用 flow-matching 损失 $\mathcal{L}_{\text{vid}} = \| D_\theta(\mathbf{x}_\sigma, \sigma; \mathbf{R}^{\text{ref}}, \mathbf{A}, \mathbf{L}^{\text{vtp}}) - (\epsilon - \mathbf{x}_0) \|_2^2$；投影声学特征注入 DiT 第 2–15 层。
- **Prefix Streaming Student**：采用 block-causal 注意力（窗口内双向、窗口间因果）；每一后续窗口的第一个 slot 为上一 chunk 末尾干净 latent 的 stop-gradient 副本，后接三个新 latent，避免累积漂移；语音块与视频块按 6 acoustic frames → 12 video frames 对齐。
- **四阶段训练**：Stage 1 冻结 LLM，使用 ~800K ASR + 1M TTS 训练 Speech Projector 与 Speech Generator；Stage 2 使用 InstructS2S-200K 与 OmniCharacter 联合更新 LLM/Projector/Generator 并引入 VTP-response 协议；Stage 3 使用 140K SpeakerVid 片段训练 Teacher Video Generator；Stage 4 分 Phase I（flow-map）与 Phase II（on-policy distribution matching）蒸馏 Student。
- **损失函数组合**：$\mathcal{L} = \mathcal{L}_{\text{lm}} + \mathcal{L}_{\text{sp}} + \mathcal{L}_{\text{vid}}$，每项仅在对应路径存在监督时启用，不假设全配对样本。

## 实验与结果
- **数据集与评测协议**：音频视频评测使用 VoiceBench CommonEval 200 条语音问答；多轮对话使用 OmniCharacter 400 对话；所有方法共享相同参考图像/语音；基线以上游 Qwen2.5-Omni-7B 为统一响应控制器。
- **音视频生成指标（Table 1）**：
  - Teacher（50-step）：SC=94.62，IQ=67.31，DD=72.00，Sync-C=4.95，SIM=0.417。
  - Streaming Student（4-step）：SC=93.65，IQ=57.40，DD=32.00，Sync-C=3.90。
  - Streaming Student（8-step）：SC=93.91，IQ=61.15，DD=48.00，Sync-C=4.00。
  - 随 step 增加，SC/IQ/DD/Sync-C 单调上升，体现可调节质量-效率折中。
- **多轮对话质量（Table 2）**：Ex-Omni-2D 在 Fluency=3.812、Coherency=4.100、Consistency=3.902 上领先对比方法，三指标均值 3.938（Qwen3-8B 基座 3.537）；十二维度均值 3.283，略高于 Qwen3-8B 的 3.264。
- **语音问答（Table 3）**：AlpacaEval=4.28、CommonEval=3.71、BBH=58.70，仅次于 Qwen2.5-Omni（4.49/3.93/60.80）。
- **质量-效率（Table 4）**：四卡 400×720/720×400 下，2-step 达 39.5 FPS、E2E RTF=1.201；4-step 26.5 FPS、E2E RTF=1.293；8-step 15.6 FPS、E2E RTF=1.932。Teacher 50-step 为 1.4 FPS、E2E RTF=26.917。首段可听语音 2.308s，首段可播放视频 3.142s。
- **消融（VTP / 个性化语音 / 无前缀流式）**：移除特定 VTP 导致 SC 下降（94.62→93.58）、Sync-C 下降（4.95→4.65），DD 上升（72→81.5，反映大幅运动比例增加而非质量提升）；Prefix Streaming 在全集 200 样本上将 SC 从 92.85 提升至 93.65，最后 chunk 一致性误差降低 21.4%（0.1005→0.0790）。
- **语音条件接口对比（Table 6）**：Waveform+wav2vec Sync-C=5.83（latency 0.051s）；单 codebook Sync-C=2.07（0.003s）；16-codebook Sync-C=4.95（0.011s）。

## 相关工作脉络
- **Omni-modal Dialogue（Qwen2.5-Omni、Moshi、VITA-1.5、Mini-Omni2、SLAM-Omni 等）**：论文将其定位为具备语音理解与口语合成能力但缺少视觉响应的系统；本文定位差异在于引入参考图像/语音条件与对话驱动的视频生成。
- **Talking-avatar / A2V（echomimic、StableAvatar、OmniAvatar、FantasyTalking 等）**：这些方法提供强视觉合成但依赖外部已完成的波形与人工提示；本文通过 VTP 与语音单元接口将视觉生成与对话状态绑定。
- **Joint Audio-Video Generation（SyncFlow、UniForm、Universe-1、UniAVGen 等）**：最接近的比较组；它们生成配对的 audio-video 内容，但通常由 prompt 或参考信号驱动而非多模态查询响应；本文强调对话条件响应与多路径异质数据训练范式。
- **Streaming Avatar（StreamAvatar 等）**：论文借鉴了其窗口内双向/窗口间因果的流式架构，并在此基础上引入 Prefix Streaming 以缓解长程漂移。
- **语音语言模型（SpeechGPT、GLM-4-Voice 等）**：减少 ASR-LLM-TTS 级联的路线；本文沿用了类似思路但将其与视频生成器在共享语音单元接口上重新连接。

## 局限性与未来方向
- **说话人相似度仍有提升空间**：SIM 为 0.417，与参考 speaker 的一致性尚不充分。
- **VTP 是高层语义引导而非独立可控信号**：最终视频由 VTP 与声学条件共同决定，固定 Text CFG/Audio CFG 比例在不同回复上不稳定，需探索自适应跨模态平衡。
- **VTP 生成引入语言-视觉能力权衡**：加入 VTP 监督后 CommonEval 下降 0.11、BBH 下降 2.40，说明共享 autoregressive 通道存在干扰，未来可考虑 planner 隔离。
- **长时程分析仅限于主体稳定性与面部动力学**：未评估更高层次的语义连贯性或交互质量。
- **Teacher 计算开销大**：50-step 双向扩散在单请求延迟与吞吐上不适合交互部署。
- **Student E2E RTF 仍大于 1**：当前 4-step 下 E2E RTF=1.293，尚未实现端到端实时交互；Streaming 在此指增量输出而非 full pipeline 实时。
- **未来方向**：扩展因果视频骨干、降低响应规划延迟、改进 few-step 蒸馏、探索 planner 参数隔离与自适应跨模态引导。

## 研究启发与可借鉴点
- **分阶段训练 + 中间接口重连策略**：通过 VTP 与多 codebook 语音单元将异构数据（ASR/TTS/对话/视频）解耦训练，推理时通过共享接口重连，可有效绕过全配对数据稀缺问题；该方法可迁移至其他多模态生成任务（如视频 dubbing、avatar 角色交互）。
- **Prefix Streaming 降低累积漂移**：将前一 chunk 末尾干净 latent 以 stop-gradient 方式作为当前窗口前缀，显式约束边界连续性；该思想可推广到其他因果视频/序列生成任务的流式部署。
- **结构化内部规划（VTP）显式化视觉意图**：在 LLM 的 `<thinking>` 块中输出结构化语义字段，既便于监督也便于调试；该设计可借鉴至需要视觉/动作规划的 agent 或 embodied AI 工作。
- **多码本语音单元作为跨模态时间对齐中介**：12.5Hz 的固定速率天然对齐 25FPS 视频，避免 waveform re-encoding 的高延迟；这种离散声学表示可作为 speech-video 联合生成中的通用时间契约。
- **质量-效率可调节的 Teacher-Student 部署对**：通过不同 denoising steps（2/4/8）提供可选 operating point，便于产品侧按延迟/质量需求选择；该思路可用于其他视频生成模型的工程化落地。

## 关键术语表
- **Ex-Omni-2D**：本文提出的 expressive omni-modal 对话框架，支持参考图像/语音条件下的文本、个性化语音与同步视频生成。
- **Visual Thought Plan (VTP)**：由对话模型生成的五字段结构化视觉规划（首帧场景、整体场景、情绪、运动风格、动作细节），作为视频生成器的语义条件。
- **Multi-codebook speech units**：Qwen3-TTS 输出的 16 码本离散声学表示（12.5Hz），同时用于语音解码与视频帧对齐条件。
- **Full-sequence Teacher**：基于 Wan2.1-T2V-1.3B 与 OmniAvatar LoRA 的全序列双向视频扩散模型，作为主质量基线与蒸馏源。
- **Prefix Streaming Student**：蒸馏后的块因果流式视频生成器，通过在前一个干净 latent 上做 stop-gradient 前缀注入缓解后期 chunk 漂移。
- **Flow-matching**：视频生成器采用的扩散目标参数化方式，预测 $\epsilon - \mathbf{x}_0$ 的 flow 目标。
- **On-policy distribution matching (DMD)**：Stage 4 Phase II 中用于蒸馏 Student 的对策分布匹配目标，配合 flow-map 共同优化。
- **Sync-C (SyncNet confidence)**：基于 SyncNet 的唇音同步置信度指标，值越高表示口型与语音对齐越可靠。
- **RTF (Real-time Factor)**：生成时长与计算耗时之比，RTF<1 表示快于实时。
- **CommonEval / AlpacaEval / BBH**：VoiceBench 套件中用于评估语音问答能力的三个子基准。
- **OmniCharacter**：400 条多轮角色对话基准，用于评估多轮 fluency/coherency/consistency 等十二维度指标。

## 可复现要素
- **数据集**：
  - SpeakerVid test split（参考条件采样）：论文未声明公开。
  - InstructS2S-200K、OmniCharacter、VoiceBench/CommonEval：公开资源。
  - ASR/TTS 训练数据（LibriSpeech、Emilia 子集）：部分公开。
- **代码/权重**：项目主页 https://logo-cuhksz.github.io/Ex-Omni-2D；论文未明确声明 GitHub 仓库与模型权重开源状态，需以项目页为准。
- **关键超参**：
  - 音频采样率：16kHz（输入/视频条件）、24kHz（参考音频至 Speech Generator）。
  - 帧率：视频 25 FPS，声学单位 12.5 Hz。
  - Denoising steps：Teacher=50，Student=2/4/8。
  - Text CFG / Audio CFG：Teacher=3.5/8.5，Student=1.0/1.0。
  - 训练 horizon / sigma shift：1000 / 5.0。
  - 块大小：6 acoustic frames → 12 video frames（0.48s）。
  - 分辨率桶：400×720、720×400、720×720。
  - 训练硬件：Stage 1/2 用 8 GPU，Stage 3 用 24 GPU，Stage 4 用 40 GPU。
