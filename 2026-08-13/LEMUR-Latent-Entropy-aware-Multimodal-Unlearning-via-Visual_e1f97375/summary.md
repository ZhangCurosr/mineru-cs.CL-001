---
title: "LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual"
source: https://arxiv.org/pdf/2608.11691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:53:25"
field: "多模态大模型安全与隐私"
keywords: ["机器遗忘", "多模态大推理模型", "推理链隐私泄露", "推理时干预", "熵动力学", "无需训练的方法"]
innovations: ["首次揭示RL训练多模态推理模型的推理链隐私泄露问题及独特两阶段熵签名", "提出LEMUR——首个专为RL-trained MLRM设计的训练自由推理时遗忘框架，利用熵调控视觉锚点注入重定向推理轨迹", "设计熵自适应注入强度与动态退出机制，在抑制推理级泄露的同时保持推理能力"]
benchmarks: ["MLLMU-Bench", "VQAv2"]
---

# 论文速读：LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual

## 一句话总结
论文针对**RL微调的多模态大推理模型（MLRMs）**存在推理链隐私泄露的新问题，提出**LEMUR**——一种**无需训练、推理时干预**的机器遗忘框架，通过利用Token级熵动力学信号精准定位敏感推理片段，并以熵调控的视觉锚点重定向潜在轨迹，同时抑制答案级和推理链级泄露，且保持非敏感 Utility 和生成流畅性。

## 研究问题与动机
- **RL推理引入新隐私风险**：RL后训练赋予MLRMs探索性思维链（CoT）能力，但模型在`<think>`推理轨迹中仍会复现敏感属性，即使最终答案已清洗——现有方法仅关注答案级，遗漏推理链泄露通道。
- **基线方法不匹配RL推理模型**：现有机器学习遗忘方法（fine-tuning/activation-steering）针对 instruct MLLMs 设计，对RL-trained MLRMs的推理轨迹缺乏可靠监控机制，且激进干预会严重破坏推理能力。
- **熵信号缺失于非推理基座模型**：研究发现RL训练引入的"探索"行为使敏感内容复现呈现独特的**两阶段熵签名**（高熵 deliberation → 低熵 committed recitation），这在非推理base MLLM中几乎不存在。
- **答案级清洗与轨迹干扰之间的张力**：答案级清洗留下轨迹泄露；激进扰动轨迹又会引发重复/退化文本，需要在精确干预与推理保持之间取得平衡。

## 核心贡献（创新点）
1. **揭示RL推理模型的推理链隐私泄露问题及独特熵签名**：首次系统展示RL-trained MLRMs比非推理base MLLMs存在更严重的推理轨迹泄露，且泄露伴随可检测的token级熵动态特征。
2. **提出LEMUR——首个专为原生RL-trained MLRM设计的训练自由推理时遗忘框架**：不同于R-MUSE等作用于hidden activation的方法，LEMUR直接在解码过程中介入，利用熵动力学作为控制信号。
3. **设计熵增强的敏感度切换机制（ESS）**：结合词法禁忌token集监测与熵阈值辅助检测，精准捕获高熵 deliberation阶段和低熵 committed recall阶段，完整覆盖敏感片段。
4. **提出熵调控视觉锚点注入（EVAI）**：在敏感模式下以连续embedding替换离散token反馈，并通过当前熵值动态调制视觉锚点注入强度，实现"强干预→弱干预"的自适应切换。
5. **设计动态熵控相时长机制（DEPD）**：基于指数移动平均构建自适应退出阈值，配合cooldown窗口防止离散/潜在解码间振荡，确保干预窗口精确对齐记忆跨度。

## 方法详解

### 问题定义
给定MLRM $\mathcal{M}_\theta$、遗忘集 $\mathcal{D}_f$（需移除的subject-attribute对）和保留集 $\mathcal{D}_n$，目标是找到 $\hat{\theta}$ 使遗忘集准确率最小化同时保持保留集性能不变。LEMUR保持 $\hat{\theta}=\theta$，纯在推理时干预解码分布。

### 关键公式与组件

**（1）Token级熵计算**：$H_t(v) = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$，反映每步输出分布的不确定性。

**（2）敏感度切换门控**：
$$g_t = \underbrace{[P_t^{\Phi} \geq \rho]}_{\text{lexical: committed recall}} \vee \underbrace{[H_t(v) \geq \tau \wedge P_t^{\Phi} \geq \rho_{\text{lo}}]}_{\text{entropy-augmented: deliberation}}$$
- 高熵阶段降低质量阈值（$\rho_{\text{lo}} < \rho$），以捕获同义词扩散分布
- 低熵阶段依赖标准词汇检测（$\rho$），捕获已committed的敏感复述

**（3）受限潜反馈**：移除禁忌token后重新归一化，并计算期望embedding而非离散采样：
$$\hat{e}_t = \sum_{v \in \mathcal{V}} \tilde{p}_t(v) \bar{E}[v]$$

**（4）视觉锚点注入**：组合视觉锚点（预训练视觉special token均值）和安全回答锚点，按熵自适应强度注入：
$$\gamma_t = \min\left(\gamma_{\max},\; \frac{H_t(v)}{\tau}\gamma\right),\quad e_t = (1-\gamma_t)\hat{e}_t + \gamma_t a$$
- 高熵步骤获得强 Steering（锚点易影响结果）
- 低熵步骤获得弱 Steering（避免破坏已形成的流畅文本）

**（5）动态退出机制**：基于指数移动平均熵 $\bar{H}_t$ 构建自适应退出阈值：
$$z_t = [\neg g_t \wedge H_t(v) \geq \kappa \bar{H}_t] \vee [t - t_0 \geq W_{\max}]$$
配合 cooldown 窗口（至少C个离散步后才能重新触发）防止快速振荡。

## 实验与结果

**数据集与模型**：基于MLLMU-Bench重建含推理链的数据集；主模型为R1-Onevision-7B和Vision-R1-7B；对比基线包括GAdiff、KLMin、NPO、MMUnlearner、R2MU、R-MUSE。

**核心指标**：CLS Acc↓、FIB Acc↓、Gen TR↓、SRL↓（越低越好）；Retain/Celebrity上的Acc↑和RRA↑（越高越好）。

**主要结果（10% Forget，R1-Onevision-7B）**：
| 方法 | CLS Acc↓ | FIB Acc↓ | Gen TR↓ | SRL↓ | RRA↑ |
|------|----------|----------|---------|------|------|
| Vanilla | 58.6 | 17.0 | 32.2 | 56.8 | 7.7 |
| R-MUSE | 34.2 | 10.0 | 21.0 | 28.3 | 6.2 |
| **LEMUR** | **25.6** | **1.0** | **11.5** | **9.4** | **6.9** |

- LEMUR在遗忘集上CLS Acc降至25.6%（相对R-MUSE再降9.2%）、SRL降至9.4%（相对R-MUSE降18.9个百分点），是**最强遗忘效果**
- 在Retain/Celebrity集上保持接近Vanilla水平（CLS Acc 58.7%/69.1%，RRA 7.1/8.2）
- 在Vision-R1-7B（5%和10% Forget）及OpenVLThinker-7B（5%/10%/15% Forget）上均呈现一致优势模式
- 在VQAv2通用视觉推理语料上也取得最优遗忘效果（SRL 13.6% vs R-MUSE 35.8%），证明熵签名是RL训练的普遍产物而非特定数据属性
- **组件消融**证实各模块贡献：Base→+ESS→+VAI→+EVAI→+DEPD逐层提升，其中EVAI在遗忘和Retain利用率间取得最佳平衡

## 相关工作脉络
1. **MLRMs（R1-Onevision、Vision-R1）**：通过RL原生习得推理能力，与Qwen2.5-VL等instruct MLLMs不同——推理链是内生的而非prompted/distilled，这直接导致新的隐私风险形态。
2. **机器遗忘基础方法（GAdiff、NPO、KLMin）**：梯度/Bayesian/偏好优化类方法，通过微调权重实现遗忘，但会损害保留知识和推理能力；LEMUR完全不触碰参数。
3. **训练自由推理时干预（Embedding corruption、Logit correction、Guardrail）**：作用于prompt或logits层面，解决的是answer-centric视角，无法覆盖reasoning trace中的泄露。
4. **多模态机器遗忘（MMUnlearner、Single-image unlearning）**：针对MLLMs设计，仅抑制最终答案；LEMUR明确指出这些方法在RL推理模型上SRL几乎未下降的根本原因。
5. **R-MUSE（Li et al. 2026）**：最接近的竞品——推理保留型训练自由方法，但作用于hidden activation而非decoding过程，且专为Qwen2.5-VL（非RL-trained）设计；LEMUR在MLRM上SRL显著更低。
6. **Diffusion模型概念擦除（ActErase、DVE）**：面向文生图模型的概念消除范式，与MLLM推理时干预的思路不同但共享"无需重训"理念。

## 局限性与未来方向
- **熵信号在非RL模型上减弱**：迁移到Qwen2.5-VL时发现熵 surge 不明显，ESS仅作为辅助条件而非真正检测器，主要依赖视觉锚点路径生效，说明方法对RL训练特性有依赖。
- **少量退化现象**：遗忘集上的生成在回忆路径被抑制后出现轻度token重复或截断，虽不泄露敏感属性但不如保留集流畅。
- **教师蒸馏依赖**：数据集需通过强多模态教师（Qwen3.5-35B-A3B）蒸馏推理链，单遍蒸馏无法保证内容级100%无泄露（仅靠prompt guard + 格式校验）。
- **未来方向**：作者计划将熵驱动解码视角扩展到更广泛的安全目标。

## 研究启发与可借鉴点
1. **Token级熵动态作为隐私泄露检测信号**：两阶段熵签名（高熵 deliberation → 低熵 committed recall）的发现是一个通用izable洞察，可迁移到其他需要检测"敏感内容正在被复现"的场景。
2. **推理时干预替代微调的范式价值**：LEMUR证明了在decoding层面通过latent injection实现精确干预的可行性，避免了fine-tuning的灾难性干扰和算力成本，适合快速部署场景。
3. **视觉锚点重定向策略**：将推理轨迹拉回输入图像（视觉锚点）而非强行屏蔽，既避免了"拒绝回答"的生硬感，又利用了多模态模型的内在视觉 grounding 能力——这一思路可推广到其他模态组合（如音频锚点）。
4. **熵自适应注入强度**：$\gamma_t \propto H_t(v)$ 的设计体现了"不确定性越高越易 steering"的直觉，这一原则可应用于其他需要动态调节干预力度的推理时干预任务。
5. **与团队方向结合机会**：可考虑将LEMUR的熵检测机制与团队在LLM安全/内容过滤方向的工作结合，或将其推理时干预范式迁移到纯文本reasoning model（如o1系列）的隐私保护场景中。

## 关键术语表
- **MLRM（Multimodal Large Reasoning Model）**：通过RL后训练获得原生推理能力（探索性CoT）的多模态大模型，如R1-Onevision、Vision-R1。
- **Token-level entropy**：在解码每一步计算输出分布的熵值$H_t(v)$，用于衡量模型在该步的不确定性程度。
- **两阶段熵签名（Two-stage entropy signature）**：敏感属性被复现时呈现的熵动态模式——先经历高熵 deliberation（候选值竞争），再进入低熵 committed recitation（确定性复述）。
- **Forget set / Retain set**：需遗忘的目标数据子集 vs. 需保持性能的正常数据子集，是机器遗忘的标准评估划分。
- **Visual anchor**：预训练视觉special tokens（如`<|vision_start|>`等）的embedding均值，用于将推理轨迹重新 grounded 到输入图像。
- **Latent decoding**：敏感模式下用连续embedding（期望表示）替代离散token反馈的解码方式，保留梯度信息同时抑制敏感输出。
- **SRL（Subject-level Reasoning Leakage）**：衡量推理轨迹中是否泄露了目标subject任何隐私属性的指标，是本文新提出的关键评估标准。
- **RRA（Reasoning Retention Ability）**：使用Gemini-2.5-Pro作为自动judge评估生成文本的流畅性和自然度，量化推理能力保持程度。

## 可复现要素
- **数据集**：MLLMU-Bench（包含虚构人物肖像图和私人QA对），论文在其上重建含推理链的训练数据；VQAv2用于泛化验证。**论文未明确声明公开**。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：$\rho$、$\rho_{\text{lo}}$、$\tau$（参考熵阈值）、$\kappa$（自适应退出系数）、$\gamma$（锚点注入基准强度）、$\gamma_{\max}$、$W_{\max}$（最大潜伏时长）、$C$（cooldown步数）、$\eta$（EMA平滑系数）——论文正文未逐一列出具体数值，需在附录或代码中寻找。
