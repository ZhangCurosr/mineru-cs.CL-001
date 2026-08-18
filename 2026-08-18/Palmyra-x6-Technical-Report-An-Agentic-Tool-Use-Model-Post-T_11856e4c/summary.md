---
title: "Palmyra-x6-Technical-Report-An-Agentic-Tool-Use-Model-Post-T"
source: https://arxiv.org/pdf/2608.16620v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:43:24"
field: "Agent 工具调用与大模型微调"
keywords: ["Anchored SFT", "Agent tool use", "Muon optimizer", "Synthetic trajectory", "MoE post-training", "Self-distillation", "agentic model"]
innovations: ["ASFT 在 744B MoE 极小合成数据集（626条）上稳定微调且保持基座能力", "Muon+Adam 混合优化器在 post-training MoE 场景的系统性落地", "SDFT 思想扩展至多轮 tool-calling 的自蒸馏合成数据生成管线"]
benchmarks: ["MCP-Atlas", "FinanceBench", "IFBench", "BFCL Core", "FORTRESS", "Washington Post ModelSlant", "BBQ", "OR-Bench", "TruthfulQA"]
---

# 论文速读：Palmyra x6 Technical Report – An Agentic Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning

## 一句话总结
Palmyra x6 是 Writer AI 面向 Agent 工具调用场景开发的 744B/~40B 激活参数 MoE 模型，通过在 GLM-5.2 基座上进行 Anchored Supervised Fine-Tuning（ASFT）和 Muon+Adam 混合优化器，仅用 ~626 条合成轨迹完成单轮微调，即在 MCP-Atlas、FinanceBench、IFBench 等六个 Agent 基准上全面超越前代默认模型，并在 BFCL Core 达到 0.785、六基准均值 0.765。

## 研究问题与动机
1. **Agent 工具调用的专业化需求**：Writer Agent 产品需要在多工具环境（MCP suites、Web 搜索、代码沙箱、RAG 管道）中完成多步规划与长期任务，而通用 LLM 在此类场景中表现不足。
2. **小数据 SFT 会侵蚀基座能力**：传统 SFT 使用外部专家生成的轨迹训练，目标分布远离模型自身分布，导致"能力退化"问题突出。
3. **全量训练管线不再必要**：近年趋势表明，只要训练数据质量高且目标函数与用途匹配，对已有高性能 LLM 直接进行 post-training 即可，无需重新 pretrain/mid-training。
4. **安全与对齐漂移风险**：在引入新行为（如工具调用）的同时，需防止模型原有的安全拒答能力和对齐行为发生不可控偏移。

## 核心贡献（创新点）
1. **ASFT 在小规模合成轨迹上的规模化应用**：以 KL 锚定（K=0.1）+ DFT token 概率加权设计损失函数，仅用 626 条单轮轨迹即可稳定微调 744B MoE 模型而不损坏基座能力；区别于纯 SFT，ASFT 显式约束策略偏离基座分布。
2. **Muon + Adam 混合优化器在 MoE post-training 中的落地**：对 2D 权重矩阵采用经 Newton–Schulz 正交化的 Muon（含 Muon Split 变体），对嵌入/归一化/router/标量保留 Adam，相比纯 Adam 在 LLM 尺度下获得约 2× 计算效率提升；区别于前作主要关注 pretraining 阶段的 Muon 使用，本文聚焦 post-training 场景。
3. **自蒸馏式合成轨迹生成管线（SDFT 适配版）**：将 Self-Distillation Fine-Tuning 思想扩展到多轮 tool-calling 场景——用 teacher 轨迹降维为策略级 demonstration 注入系统提示，student 模型自行调用真实工具生成 on-policy 轨迹，并通过 verifier + 作弊过滤器双重门控筛选；区别于标准 SFT 直接使用静态专家轨迹，该方法输出完全 on-policy 数据。
4. **面向 Agent 场景的系统性安全评估**：在政治偏见（Washington Post ModelSlant）、FORTRESS 对抗基准、BBQ/OR-Bench/TruthfulQA 等多维度证明 ASFT 下 KL 锚定可有效限制对齐漂移，模型在 556 对政治 prompt family 中无显著英文立场不对称，安全性能优于多项前作模型。

## 方法详解
**模型架构**：继承 GLM-5.2 的 GlmMoeDsa 架构，744B 总参数 / ~40B 激活参数，78 层（首 3 层 dense，余 75 层 MoE），每 expert FFN 2,048，256 个路由 expert top-8 选择；MLA（64 heads，q-LoRA 2,048/kv-LoRA 512）、DSA IndexShare（跨层共享稀疏注意力索引）、RoPE（base 10⁶）、1 个 next-n 多 token 预测层；训练 context length 65,536 tokens。

**合成数据生成管线**：
- 教师端：使用 reasoning-trace teacher 变体生成成功参考轨迹，降维为 strategy-level demonstration（保留推理大纲和 tool call 顺序，去除最终答案、将数字/日期替换为占位符、mask tool-call 参数）。
- 学生端：demonstration 注入 student 系统提示（强调仅作策略参考），student 自行调用真实工具完成任务，temperature=1.0、top-p=0.95、max 50K tokens、最多 30 轮、每任务最多 6 次尝试。
- 努力上限：单轨迹最多 20 次 tool call、最多 4 次连续调用同一工具、最多 3 次 error、最多 3 次重复相同调用，超限则废弃并重试。
- 验证门：judge 模型对完整 trajectory（含推理和 tool calls）评分，3 次 voting majority，≥0.5 通过；保留 graded quality signal。
- 泄漏过滤：拒绝显式引用 demonstration、4-gram 重叠 >0.8、或 tool 可用却无 tool call 即得答案的轨迹。

**ASFT 损失函数**：
$$\mathcal{L} = -\text{mean}(P \cdot \log p_\theta) + K \cdot \text{KL}(\pi_{\text{ref}} \| \pi_\theta), \quad K=0.1$$
其中 $P = \exp(\log p_\theta(y_t)).\text{detach()}$ 为 DFT token 概率加权，$\pi_{\text{ref}}$ 为冻结的 GLM-5.2 基座；KL 项采用 $k_3$ 估计量：$k_3 = \pi_{\text{ref}}/\pi_\theta - 1 - \log(\pi_{\text{ref}}/\pi_\theta)$，token-wise 在 teacher target tokens 上计算。

**Muon + Adam 混合优化器**：
- Muon 用于 2D 权重矩阵（attention projections、dense/MoE expert FFN），momentum buffer 经 ~5 步 Newton–Schulz 迭代正交化为最近半正交矩阵，并做 RMS-matched rescaling 以保持不同形状矩阵间有效步长一致；attention projection 采用 Muon Split（按 head 拆分后独立正交化）。
- Adam（$\beta_1=0.9, \beta_2=0.98$）用于 token embedding、output head、RMSNorm、router weights、biases 和标量。
- Muon momentum=0.95 Nesterov，weight decay=0.1，optimizer state CPU-offload。

**训练配置**：LR=5×10⁻⁷ cosine decay 至 5×10⁻⁸，warmup 0.1（初 LR=1×10⁻⁸），global batch size=16，单 epoch，65,536 context length（CP=2），BF16 权重、FP32 梯度 all-reduce 和 softmax，FlashAttention，TP8·PP4·EP16·ETP1·CP2·DP1，64× NVIDIA H200。

## 实验与结果
**评估协议**：Palmyra x6 vs. Writer Agent 前代默认模型（prior default），涵盖 finance-agentic、tool-use、instruction-following 六项内部基准（0–1 分制），以及五款 frontier 模型的跨模型对比（Figure 2）。所有评测在 Writer 内部基础设施中完成。

**主要结果**：
- 相对前代默认模型：全部六项基准领先，最大提升为 MCP-Atlas (+0.320)、FinanceBench (+0.305)、IFBench (+0.304)。
- 相对 frontier 模型：BFCL Core 达 0.785（最高），六基准均值 0.765（最高）。
- 安全评估（Washington Post ModelSlant）：80% 热点问题同时呈现正反双方，净左倾最小；政治拒绝不匹配率 1.2%（第二低）；toxic content safety 88%。
- FORTRESS 对抗安全：模型+系统消息下 adversarial safety 67.0 vs. 基座 58.4（+8.6 分）；benign helpfulness 96.4 vs. 基座 97.2（几乎无损）。
- TruthfulQA 80.6%（基座 81.5%），BBQ 歧义项正确率 90.9%+（cohort 最高 91–93%）。

**消融实验**：12 组 SFT/ASFT 矩阵探索，最佳组合为 ASFT K=0.1 + LR=5×10⁻⁷ + batch=16 + 单 epoch；K=0.02+高 LR、双 epoch、batch=32、更短 context、panel-pruned 子集均被排除。

## 相关工作脉络
1. **Anchored SFT (ASFT, He et al., ICLR 2026)**：本文直接沿用其锚定 SFT 框架并验证其在 744B MoE + 极小合成数据集场景下的可行性，区别于原工作主要关注一般 SFT 场景。
2. **Self-Distillation Fine-Tuning (SDFT, Shenfeld et al., 2026)**：本文的核心数据生成策略源自 SDFT 思想——让模型成为自己的 teacher——但将其从 token-level KL 扩展为 sampling+filtering 多轮 tool-calling 管线。
3. **Muon 优化器 (Jordan et al., 2024)**：将 Muon 首次系统地应用于 post-training 阶段（而非 pretraining），并结合 Muon Split 适配 MLA 架构。
4. **LIMO / LIMI (Less is More for Reasoning / Agency)**：本文 "少数据高质量" 范式与之呼应，但 LIMI 侧重 agency 领域的推理蒸馏，本文侧重工具调用行为的定向 specialization。
5. **IndexCache / DSA IndexShare (Bai et al., 2026)**：基座模型的 DSA 跨层索引复用机制是部署侧的关键约束，要求推理栈支持 IndexShare（SGLang ≥ 0.5.13.post1）。
6. **FORTRESS 安全基准 (Knight et al., 2025)**：本文使用该基准验证 post-training 对齐漂移假设，补全了 Agent 模型安全评估的实证证据。

## 局限性与未来方向
1. **领域覆盖受限**：训练仅覆盖 12 个任务域（金融研究、数据分析/编码、临床 Agent、MCP、模拟世界、RAG），超出这些领域或工具生态的 task 可能无法直接受益。
2. **非 Agent 任务依赖基座**：模型的泛化行为由 GLM-5.2 基座决定，对非 Agent 专项任务无额外提升。
3. **对齐漂移假设未严格验证**：作者坦承 KL 锚定保留 base 模型 general capability，但对其能否有效保留 refusal/alignment 行为仅做推断性声明，缺乏严谨实验验证。
4. **合成数据真实性边界**：轨迹全部由机器生成，judge 模型评分可能引入系统性偏差；真实用户交互分布与合成分布之间是否存在 gap 尚不明确。
5. **推理部署依赖特定基础设施**：DSA IndexShare 要求推理栈显式支持，FP4 量化虽已验证安全性，但更广泛的部署兼容性待验证。

## 研究启发与可借鉴点
1. **"少即是多"在 Agent 微调中的规模化验证**：626 条高质量 on-policy 轨迹即足以让 744B 模型获得显著工具调用能力提升，为后续团队在有限数据条件下做定向 specialization 提供了实证参考；建议在本团队研究中复用 SDFT 式 teacher-student 蒸馏管线设计。
2. **ASFT 损失函数可直接迁移**：DFT token 概率加权 + KL 锚定的联合损失形式简单且有效，特别适用于需要保留基座能力的 post-training 场景；建议在现有 pipeline 中替换纯 SFT 损失作为 baseline。
3. **Muon + Adam 混合优化器值得在 post-training 场景复现**：对于含大量 2D 矩阵的 MoE 模型，Muon Split 变体可保持 attention geometry 与 pretrain 一致，避免 Singular Value 主导学习方向；建议在后续大模型 fine-tuning 中对比 AdamW。
4. **多维度安全评估框架可作为模板**：Washington Post ModelSlant + FORTRESS + BBQ/OR-Bench/TruthfulQA 的组合覆盖了政治偏见、对抗滥用、事实一致性三个维度，建议在本团队 Agent 模型交付前参照此框架补充安全评估。
5. **努力上限（effort cap）策略可用于成本控制**：限制单次轨迹的 tool call 数、连续同工具调用数、error 数和重复调用数，可在生成阶段提前终止无效 rollout，避免 judge 评估开销；该策略可直接复用到合成数据生成 pipeline。

## 关键术语表
**ASFT (Anchored Supervised Fine-Tuning)**：通过 KL 锚定项 + DFT token 概率加权约束策略分布，防止 SFT 过度偏离基座模型分布的微调目标函数。
**Muon 优化器**：将 2D 权重矩阵视为几何对象，经 Newton–Schulz 迭代正交化 momentum buffer 以实现各奇异值均衡更新的优化器。
**Muon Split**：对 MLA attention projection 按 head 拆分后独立应用 Muon 正交化，避免跨 head 有效步长耦合。
**DSA IndexShare**：DeepSeek Sparse Attention 的跨层索引复用机制，稀疏注意力选定的 token 索引在不同层间共享，降低推理内存与计算开销。
**SDFT (Self-Distillation Fine-Tuning)**：让模型自身作为 teacher（在 context 中提供 demonstration），student 仅依赖自身 tool 调用生成轨迹的蒸馏式微调方法。
**k₃ 估计量**：一种低方差 KL 散度估计量，$k_3 = r - 1 - \log r$（$r = \pi_{\text{ref}}/\pi_\theta$），在 ASFT 中用于 per-token KL 惩罚项。
**MCP (Model Context Protocol)**：Writer 使用的标准化多工具调用接口协议，允许模型通过统一方式调用搜索、代码执行、文档检索等工具。
**effort cap**：合成数据生成中对单次轨迹施加的上限约束（tool call 总数、连续同工具调用数、error 数、重复调用数），超限即废弃。

## 可复现要素
- **数据集**：626 条合成轨迹，来自 12 个私有数据集（财务研究、数据分析/编码×2、医疗 Agent、RAG、模拟世界×5、MCP×2）；**未公开**。
- **代码**：训练使用 slime 框架（THUDM），转换工具为内部工具；**未开源**。
- **权重**：BF16 和 FP4 量化变体以 Writer 商业许可发布；**未公开开源**（需联系 Writer AI）。
- **关键超参**：ASFT K=0.1；LR=5×10⁻⁷ cosine decay 至 5×10⁻⁸；batch size=16；单 epoch；context=65,536；Muon momentum=0.95 Nesterov；weight decay=0.1；TP8·PP4·EP16·ETP1·CP2·DP1。
- **推理依赖**：SGLang ≥ 0.5.13.post1 或等效支持 DSA IndexShare 的推理栈。
