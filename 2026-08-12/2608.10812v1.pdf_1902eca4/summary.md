---
title: "Reference-Free Post-Training of Open Large Language Models for Multilingual Machine Translation"
source: https://arxiv.org/pdf/2608.10812v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 07:48:05"
---

# 论文速读：Reference-Free Post-Training of Open Large Language Models for Multilingual Machine Translation

## 一句话总结
本文提出针对开源大语言模型的**无参考后训练**方法，利用 Group Relative Policy Optimization (GRPO) 与 XCOMET/COMETKiwi 无参考质量评估作为奖励信号，仅凭源侧文本对 MiLMMT-46 系列（1B/4B/12B）进行强化学习微调，在 46 种语言、WMT24++ 与 FLORES+ 基准上显著提升多语言机器翻译质量，部分开源小模型规模即可超越 Gemini 2.5/3 Pro、GPT-5 及 NLLB-54.5B 等商业/重型基线。

## 研究问题与动机
1. **高质量平行数据稀缺**：多语言 MT 尤其低资源语言与非英中心方向严重依赖人工平行语料，数据获取成本高昂且覆盖有限。
2. **传统奖励信号依赖参考译文**：现有 RL/SFT 后训练多依赖 BLEU、有参考 COMET 等需要目标参考的指标，在参考不可得或参考质量参差时失效。
3. **无参考奖励易引发 Reward Hacking**：直接使用 XCOMET/COMETKiwi 等 QE 模型打分作为奖励，模型可能通过语言漂移或表面形式优化“作弊”，导致目标语言错误。
4. **开源 LLM 的 MT 后训练潜力未被充分挖掘**：Gemma、Qwen 等开源基座已具备多语言能力，但缺乏针对无参考奖励信号的系统化后训练验证与工程落地。

## 核心贡献（创新点）
1. **无参考 RL 后训练范式**：首次将 GRPO 与 XCOMET/COMETKiwi 无参考奖励结合用于开源 LLM 的多语言 MT 后训练，本质区别在于完全剥离人工参考译文，实现仅凭源侧文本驱动的端到端质量优化。
2. **语言识别门控防 Hack 机制**：引入 OpenLID-v3 对候选译文进行语言预测，若 $\hat{\ell}(y) \neq \ell$ 则奖励强制置零，区别于纯 QE 打分方法，从根本上抑制无参考信号下的语言漂移问题。
3. **SFT-RL 检查点插值策略**：提出 $\theta_\alpha = \alpha \theta_{\text{SFT}} + (1-\alpha)\theta_{\text{RL}}$（$\alpha=0.5$）融合两个检查点，区别于纯 RL 微调常伴随的 spBLEU 下降，在保留语义质量增益的同时恢复词汇重叠指标。
4. **开源中等规模模型对标商业巨头的可行性验证**：MiLMMT-46-12B-v1.0 在 WMT24++ 与 FLORES+ 多方向组全面超越 Gemini 2.5/3 Pro、GPT-5 及 NLLB-54.5B，证明无需百亿参数堆叠即可触及商业系统水平。
5. **On-Policy Distillation 增益验证**：采用 PG-OPD 将 12B 教师模型的 RL 增益蒸馏至 1B/4B 学生，虽未超越插值 SOTA，但为算力受限场景提供了稳健的知识迁移路径。

## 方法详解
- **基座模型**：MiLMMT-46-v0.1（Shang et al., 2026），SFT 阶段已支持 46 种语言；底层为 Gemma/Qwen 系列开源 LLM。
- **RL 算法**：Group Relative Policy Optimization (GRPO)（Shao et al., 2024），组大小 $G=8$。
- **奖励函数**：对源句 $x$ 与候选译文 $y$，取 **XCOMET** 与 **COMETKiwi** 两个无参考 QE 模型分数的平均值：$R(y) = \frac{1}{2}(\text{XCOMET}(x,y) + \text{COMETKiwi}(x,y))$。
- **语言门控**：通过 **OpenLID-v3**（Fedorova et al., 2026）识别 $y$ 的语言 $\hat{\ell}(y)$，若 $\hat{\ell}(y) \neq \ell$（目标语言），则 $R(y)=0$，防止模型输出非目标语言以刷高 QE 分数。
- **检查点插值**：最终模型 $\theta_\alpha = \alpha \theta_{\text{SFT}} + (1-\alpha)\theta_{\text{RL}}$，取 $\alpha=0.5$ 得到 **MiLMMT-46-v1.0**，平衡无参考语义质量与有参考词汇精度。
- **训练数据构建**：从 MiLMMT SFT 数据中移除参考译文，保留 263,982 条（192 个翻译方向）；按组内均值 $0.30 < \mu_x < 0.95$、标准差 $\sigma_x \ge 0.05$ 筛选后保留 **31,572 条**（30,572 训练 / 1,000 验证）。
- **超参数（跨 1B/4B/12B 统一）**：学习率 $1\times10^{-6}$、batch size 128、PPO mini-batch 128、训练 3 epochs、KL 系数 $\beta=1\times10^{-3}$、组大小 $G=8$、温度 0.8、top-p 0.95、prompt/response 最大长度 4096 tokens。
- **工程实现**：使用 **vLLM** 推理后端与 **verl** 训练框架，生成策略为 greedy decoding。

## 实验与结果
- **数据集与基线**：WMT24++（去低质量句）与 FLORES+（覆盖 46 种语言）；基线包括 TranslateGemma-4B/12B/27B、GemmaX2-28-2B/9B、Hunyuan-MT、Seed-X-Instruct/PPO-7B、Tower-Plus-2B/9B/72B、Google Translate、NLLB-54.5B、Gemini 2.5/3 Pro、GPT-5。
- **评估指标**：无参考 XCOMET、无参考 COMETKiwi；FLORES+ 另报告 spBLEU 与有参考 XCOMET。
- **相对 SFT 基线（WMT24++ 平均三尺度）**：XCOMET **+2.75**，COMETKiwi **+2.44**；FLORES+ 有参考 XCOMET +1.17、无参考 XCOMET +1.41、COMETKiwi +1.17，spBLEU **-1.2
