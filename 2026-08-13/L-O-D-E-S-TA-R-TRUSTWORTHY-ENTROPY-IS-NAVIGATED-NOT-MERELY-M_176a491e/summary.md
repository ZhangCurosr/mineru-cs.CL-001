---
title: "L-O-D-E-S-TA-R-TRUSTWORTHY-ENTROPY-IS-NAVIGATED-NOT-MERELY-M"
source: https://arxiv.org/pdf/2608.11922v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:59:32"
---

# 论文速读：LODESTA*R*: TRUSTWORTHY ENTROPY IS NAVIGATED, NOT MERELY MEASURED

## 一句话总结
针对检索增强问答（RAG）中“最小熵选择规则”在误导性片段上反而置信度最高（confidently wrong）的失效问题，LODESTAR 提出通过强化学习离线训练一个短自然语言极化器字符串，插入提示词以主动引导冻结 LLM 的熵分布方向，在无金答案干预的推理阶段即可显著提升答案选择的可靠性与准确率。

## 研究问题与动机
- **核心问题**：开放域 RAG 通常依赖冻结响应模型读取检索候选片段生成答案，缺乏金答案时主流做法是选取响应模型生成熵最低的候选答案；但论文验证发现，误导性片段会诱导冻结模型以极低熵（高置信度）输出错误答案，导致最小熵选择规则被反向误导。
- **现有方法不足**：更强检索器/重排器反而推高误导性片段比例（Table 1 显示占比 20.3%~35.0%）；既有工作多将熵视为跨问题统计量或仅记录失效现象，未提供在单问题内部区分误导与支持片段的有效手段。
- **干预动机**：冻结模型权重不可改，唯一自由变量是输入上下文；因此论文转向“改变提示输入以重塑熵响应”而非“被动测量原始熵”，提出定向熵（directed entropy）范式。

## 核心贡献（创新点）
1. **揭示最小熵选择的系统性失效并证明单字符串可修复**：在五个 QA 基准上，纯最小熵规则平均以 30.3% 的概率选中误导性片段（高于随机抽取的 28.9%），引入极化器后降至 26.0%，且在所有数据集上稳定改善。
2. **构建公平的统一选择器基准并确立新 SOTA**：将 14 种已发表方法重制为同等条件下的段落选择器，LODESTAR 在相同冻结响应模型下取得最高平均 $F_1$（0.5339）、最高 Exact Match（0.4136）与最高 GPT-4o Judge 分（0.6435），且对所有基线均达到配对统计显著。
3. **系统绘制极化器跨冻结模型的迁移图谱**：覆盖 Llama-3.1-8B、Qwen2.5-7B 与 Qwen3.5-9B 的 3×3 交叉配置，发现最优熵信号形式随模型族变化（Llama 适用首 token 熵，Qwen 适用全 token 熵），但极化器干预效应本身具有跨架构一致性。

## 方法详解
- **提示结构**：冻结响应模型 $R$ 的输入由 $[p; q]$ 扩展为 $[p; \psi^\star; q]$，其中 $\psi^\star$ 为固定自然语言极化器，插入在候选段落与问题之间。
- **信号定义**：令 $\bar{H}_L(q, p)$ 为 $R$ 生成答案 $L$ 个 token 的平均归一化熵；定义问题内熵分离度 $\Delta_{\mathrm{Mis-Sup}} \bar{H}_L(q) = \mathrm{mean}_{p\in\mathrm{Mis}(q)}\bar{H}_L(q,p) - \mathrm{mean}_{p\in\mathrm{Sup}(q)}\bar{H}_L(q,p)$。极化器的目标是使 $\Delta$ 变为正值。
- **推理选择规则**：$\hat{a} = \arg\min_{a(q,p;\psi^\star),\, p\in P(q)} \bar{H}_L(q, p, \psi^\star)$，即定向熵最小化；仅依赖冻结模型的一次前向计算，零额外模型、零采样、零金答案。
- **训练目标**：采用 GRPO 离线训练极化器策略 $\pi_\theta$（基座为 Qwen3-4B-Instruct）。奖励函数为分离度的基线修正增益：
  $S(\psi) = \widehat{\mathbb{E}}_{q\in B}\!\left[\Delta_{\mathrm{Mis-Sup}}\bar{H}_L(q;\psi) - \Delta_{\mathrm{Mis-Sup}}\bar{H}_L(q)\right]$
  奖励从不读取任何答案，仅依赖片段分类标签；组内标准化优势 $\hat{A}_i = \frac{S(\psi_i)-\mathrm{mean}\{S(\psi_j)\}}{\mathrm{std}\{S(\psi_j)\}}$，配合 clip range $(0.2, 0.28)$ 与 KL 正则 $\beta=10^{-3}$。
- **标签构建**：使用 gpt-oss-12
