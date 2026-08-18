---
title: "LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual"
source: https://arxiv.org/pdf/2608.11691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:05:51"
field: "多模态大模型隐私保护"
keywords: ["Machine Unlearning", "Multimodal Reasoning", "Privacy Leakage", "Entropy-aware", "Training-free", "Visual Anchor"]
innovations: ["提出首个针对RL训练MLRMs的训练-free推理时机器遗忘框架LEMUR", "利用token级两阶段熵签名检测和重定向推理痕迹中的敏感泄露", "熵调制的视觉锚点注入机制实现精准靶向遗忘"]
benchmarks: ["MLLMU-Bench", "VQAv2"]
---

# 论文速读：LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual

## 一句话总结
论文针对RL训练的 multimodal large reasoning models (MLRMs) 中存在的推理痕迹隐私泄露问题，提出了 LEMUR——一种无需训练的推理时干预框架，通过利用token级熵动力学作为控制信号，识别敏感推理段落并注入熵调制视觉锚点，实现对答案级别和推理痕迹级别的双重遗忘，同时保持模型非敏感任务的效用和生成流畅性。

## 研究问题与动机
- **RL推理模型的推理痕迹隐私泄露**：现代多模态大推理模型（如 R1-Onevision、Vision-R1）通过强化学习获得探索性思维链（CoT）能力，但在 `<think>...</think>` 推理轨迹中仍会复现已遗忘的敏感事实，即使最终答案已干净。
- **现有方法对推理模型不匹配**：主流机器遗忘方法聚焦于答案级别清洗或微调权重，无法处理探索性推理轨迹；对推理轨迹的激进扰动会导致推理能力退化（重复/退化文本）。
- **RL训练引入独特的熵签名**：作者发现 RL 训练使模型在回忆记忆化敏感属性时呈现独特的"高熵 deliberation → 低熵 commit"两阶段熵模式，这一信号在非推理 base model 中几乎不存在。
- **现有方法缺乏对推理轨迹的可靠监控机制**：遗忘可以在同一主体的不同属性或同义表达上持续泄露，而现有方法无法在解码过程中动态监测和干预。

## 核心贡献（创新点）
1. **首次揭示RL推理模型的推理痕迹泄露风险及熵签名**：系统性地展示了 RL-trained MLRMs 比其 base MLLM 存在更严重的推理痕迹泄露，并发现该泄露携带独特的 token 级两阶段熵特征（high-entropy deliberation + low-entropy committed recall）。
2. **提出 LEMUR 训练-free 推理时遗忘框架**：设计首个专为原生 RL 训练 MLRM 设计的遗忘方法，通过在解码阶段利用熵动力学信号实现干预，而非微调权重或扰动激活。
3. **熵增强的敏感度开关机制（ESS）**：结合词汇 forbidden token 检测与熵阈值辅助，精准识别敏感信息的"犹豫 deliberation"阶段和"确定 commit"阶段，解决纯词汇检测遗漏同义词/变体的问题。
4. **熵调制的视觉锚点注入（EVAI）**：通过凸组合将视觉锚点和安全答案锚点按当前熵值动态调制后注入潜空间，重定向推理轨迹远离记忆化敏感内容。
5. **动态熵控制阶段持续时间（DEPD）**：利用自适应熵阈值和冷却窗口机制，精确控制干预窗口长度，避免过早释放导致泄露或过晚释放导致流畅性下降。

## 方法详解

**问题设定**：给定 MLRM $\mathcal{M}_\theta$、遗忘集 $\mathcal{D}_f$（含敏感主体）和保留集 $\mathcal{D}_n$，寻找 $\hat{\theta}$ 最小化遗忘集正确率同时保持保留集性能。LEMUR 保持 $\hat{\theta} = \theta$，仅在推理时干预解码分布。

**关键公式与组件**：

1. **Token 级熵计算**：$H_t(v) = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$，用于衡量模型在每个解码步的不确定性。

2. **熵增强敏感度开关（Eq.5）**：
   - 词汇项：$P_t^{\Phi} = \sum_{v \in \Phi_s} p_t(v)$，当 forbidden mass 超过阈值 $\rho$ 时触发
   - 熵增强项：$H_t(v) \geq \tau \wedge P_t^{\Phi} \geq \rho_{lo}$，在犹豫阶段降低阈值以捕获扩散的同义词质量
   - 综合门控：$g_t = [P_t^{\Phi} \geq \rho] \vee [H_t(v) \geq \tau \wedge P_t^{\Phi} \geq \rho_{lo}]$

3. **约束潜反馈（Eq.6-7）**：移除 forbidden tokens 后重新归一化分布 $\tilde{p}_t(v)$，计算期望嵌入 $\hat{e}_t = \sum_{v} \tilde{p}_t(v) \bar{E}[v]$ 作为软反馈，保留梯度信息。

4. **视觉锚点注入（Eq.8-11）**：
   - 视觉锚点 $e_{vis}$：预训练视觉 special tokens 的平均嵌入
   - 安全答案锚点 $e_{safe}$：拒绝模板（"I'm not sure"等）的平均嵌入
   - 组合锚点 $a = \beta e_{vis} + (1-\beta) e_{safe}$
   - 熵调制注入强度：$\gamma_t = \min(\gamma_{max}, \frac{H_t(v)}{\tau}\gamma)$

5. **动态阶段持续时间（Eq.12-13）**：
   - 指数移动平均熵 $\bar{H}_t = (1-\eta)\bar{H}_{t-1} + \eta H_t(v)$ 作为基准不确定性
   - 退出指示器：$z_t = [g_t \wedge H_t(v) \geq \kappa \bar{H}_t] \vee [t - t_0 \geq W_{max}]$
   - 冷却窗口：每个退出后至少 $C$ 个离散步才能重新触发，防止振荡

## 实验与结果

**数据集与模型**：
- 基于 MLLMU-Bench 重构，使用 Qwen3.5-35B-A3B 蒸馏推理链
- 主要模型：R1-Onevision-7B、Vision-R1-7B（原生 RL 训练 MLRMs）
- 转移实验：Qwen2.5-VL（非推理 MLLM）、OpenVLThinker-7B

**评估指标**：
- Forget split：CLS Acc↓、FIB Acc↓、Gen TR↓、SRL↓、RRA↑
- Retain/Celebrity split：同上但方向相反

**主要结果（Table 1，Onevision-R1-7B，10% forget）**：
- LEMUR 在 Forget split 上取得最优：CLS Acc 26.0%（基线最低 35.2%）、Gen TR 15.1%（基线最低 21.0%）、SRL 11.3%（基线最低 28.3%）
- Retain/Celebrity split 上 LEMUR 保持 vanilla 水平（CLS Acc ~59-68%，RRA ~7.2-8.4）
- R-MUSE 在 SRL 上表现较好（28.3% vs LEMUR 9.4%），但 LEMUR 在整体遗忘和效用保持上更优

**组件消融（Table 2，Onevision-R1-7B，5% forget）**：
- Base（仅词汇检测）：SRL 42.0%，RRA 6.1
- +ESS：SRL 28.7%（显著改善）
- +VAI：SRL 12.5%
- +EVAI：SRL 10.3%，RRA 6.4（熵自适应强度恢复效用）
- +DEPD：SRL 11.3%，RRA 6.9（动态持续时间恢复流畅性）

**转移实验（Table 3，Qwen2.5-VL）**：
- LEMUR 在非 RL 模型上仍有效，但熵信号减弱，视觉锚点成为主导矫正信号
- 优于 R-MUSE：SRL 27.8% vs 30.9%

**鲁棒性**：
- 不同遗忘比例（5%/10%/15%）：结果稳定
- 不同 backbone（OpenVLThinker-7B）：定性模式一致，LEMUR 持续最优

## 相关工作脉络
- **Reasoning MLLMs**（R1-Onevision、Vision-R1）：本文关注这些模型特有的推理痕迹泄露问题，与仅关注答案的 MLLM 遗忘工作形成对比。
- **Machine Unlearning - Gradient-based**（GAdiff、KLMin、NPO）：基于梯度的方法会损害保留集效用，LEMUR 作为 training-free 方法避免此问题。
- **Machine Unlearning - Training-free**（R-MUSE、embedding corruption）：R-MUSE 针对 instruct MLLMs 设计，作用于隐藏激活而非解码过程；LEMUR 首次针对原生 RL-trained MLRMs 并在解码时干预。
- **Multimodal Unlearning**（MMUnlearner、Singl-Image Unlearning）：这些方法作用于最终答案，无法阻止推理痕迹中的敏感信息复现。
- **Diffusion Unlearning**（ActiveEraser、DVE）：针对图像生成模型的概念擦除，与 autoregressive MLRM 的隐私泄露机制不同。

## 局限性与未来方向
- **对非 RL 训练模型效果减弱**：在 Qwen2.5-VL 上，由于缺乏显著的熵 surge，熵增强的敏感度检测作用有限，主要依赖视觉锚点。
- **轻微退化现象**：少数遗忘样本出现 token 重复或截断，虽不泄露敏感信息但影响流畅性。
- **蒸馏数据构建成本**：需要强教师模型为每个 QA 对生成推理链，且需避免蒸馏过程中的泄露。
- **未来方向**：将熵驱动解码视角扩展到更广泛的安全目标；探索更轻量的教师蒸馏方案。

## 研究启发与可借鉴点
1. **熵信号作为隐私泄露检测器**：token 级熵动力学可广泛用于识别记忆化内容的重现，不仅限于机器遗忘，也可用于安全检测和红队测试。
2. **推理时干预的 training-free 范式**：无需微调即可实现针对性遗忘，适合部署场景下的快速响应和合规需求。
3. **双锚点重定向策略**：视觉锚点（回归输入）+ 安全答案锚点（拒绝模板）的组合设计，为其他 generative model 的干预提供思路。
4. **蒸馏推理链的数据构建方法**：通过 attribution context 排除当前 QA 对、first-person recall 框架、anti-leakage prompt 设计，可借鉴于构造 CoT 训练数据。
5. **跨模型可迁移性验证**：在 RL 和非 RL 模型上的 transfer 实验设计，为方法泛化性评估提供了完整范式。

## 关键术语表
**Machine Unlearning（机器遗忘）**：在不重新训练的情况下，从已训练模型中移除指定数据影响力的技术。

**Multimodal Large Reasoning Model（MLRM）**：通过强化学习获得原生推理能力的多模态大模型，如 R1-Onevision、Vision-R1，能生成显式思维链。

**Token-level Entropy（Token 级熵）**：解码过程中每个 token 预测分布的香农熵，用于衡量模型在该步的不确定性。

**Chain of Thought（CoT）**：模型生成的逐步推理轨迹，在 MLRM 中通常位于 `<think>...</think>` 标签内。

**Forget Set / Retain Set（遗忘集/保留集）**：遗忘集中的样本内容需被"删除"，保留集中的样本应保持原有性能。

**Visual Anchor（视觉锚点）**：预训练视觉 special tokens 的平均嵌入，用于将推理轨迹重新锚定到输入图像。

**Subject-level Reasoning Leakage（SRL）**：衡量推理痕迹中是否暴露目标主体任何查询属性的指标。

**Reasoning Retention Ability（RRA）**：由 LLM  judged 的生成文本流畅性和自然度评分，衡量推理能力保持程度。

## 可复现要素
- **数据集**：MLLMU-Bench（公开），论文在其上重构带推理链的变体
- **代码**：论文未明确声明开源，但提供了详细的公式和伪代码
- **模型**：R1-Onevision-7B、Vision-R1-7B、Qwen3.5-35B-A3B（教师）、Qwen2.5-VL、OpenVLThinker-7B
- **关键超参**：$\rho$（词汇阈值）、$\rho_{lo}$（低熵阈值）、$\tau$（参考熵）、$\gamma$（注入强度）、$\gamma_{max}$、$\kappa$（自适应阈值系数）、$W_{max}$（最大窗口）、$C$（冷却步数）、$\beta$（锚点平衡）、$\eta$（EMA 系数）——论文未给出具体数值，需查阅附录或代码
