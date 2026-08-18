---
title: "CAPO-Constraint-Aware-Prompt-Optimization-for-LLM-Agents"
source: https://arxiv.org/pdf/2608.16068v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 17:27:01"
---

# 论文速读：CAPO-Constraint-Aware-Prompt-Optimization-for-LLM-Agents

## 一句话总结
CAPO 提出了一种基于原始‑对偶优化的系统提示词自动生成框架，将离散 LLM 重写嵌入带投影的对偶更新循环中，使代理在行为、安全、隐私与资源等多维硬约束下协同优化准确率与合规性。

## 研究问题与动机
- 多约束 LLM 部署（Agent/Chatbot/代码修复/隐私保护）中，系统提示词需在目标准确率与多项相互冲突的约束（HAR、ToolEx、AdvBench 拒答率、字符长度、PII 违规等）之间取得平衡。
- 现有提示优化多依赖固定模板或单目标 RL，采用固定标量奖励与单位权重，难以动态感知约束违反程度并自适应调整优化重心。
- 离散文本重写无法直接求梯，连续可微放松方法（如 Gumbel-Softmax）在真实 LLM rewrite 场景中缺乏理论支撑与泛化保证。
- 需要一种兼具理论收敛分析、可处理高维约束、且能在真实 LLM 离散空间高效搜索的自动化提示词优化方法。

## 核心贡献（创新点）
- **原始‑对偶形式化与不精确预言机设计**：将基于 pool 的提示词搜索嵌入带投影的对偶更新框架，证明离散 LLM rewrite 在期望意义上与 surrogate gradient 对齐，残差 $\eta_t=\epsilon_t+\frac{L}{2}\sigma_d^2$ 可控。
- **严格的收敛性上界**：在 A1–A7 假设下推导 primal‑dual gap 上界，证明常数步长收敛至有限邻域，递减步长下 gap 收敛至 0。
- **动态乘子反馈机制**：通过持续性约束违反自动推高对偶乘子强化惩罚，约束满足后乘子稳定/下降以专注 Objective 提升，实现约束‑目标的自适应权衡。
- **跨域统一评测与消融协议**：在 TAU2‑BENCH、GSM8K、PUPA‑IFBench、SWE‑agent 四类任务上建立 AllSat 可行性标准，并系统验证噪声敏感度、阈值收紧与迁移边界。

## 方法详解
- **优化目标与对偶更新**：将提示词优化建模为 $\min_z J(z,\lambda)$ s.t. $g_i(z)\le 0$，采用对偶上升法 $\lambda_{t+1}=[\lambda_t+\beta\cdot g(z_t)]_+$，$\beta$ 为对偶学习率。
- **不精确原始预言机**：基于 prompt pool 的离散 LLM 重写充当 primal oracle。理论桥梁 B1/B2 保证 $\mathbb{E}[\langle \nabla_z J, d_t\rangle]\ge\kappa\|\nabla_z J\|^2-\epsilon_t$ 且 $\mathbb{E}[\|d_t\|^2]\le\nu\|\nabla_z J\|^2+\sigma_d^2$，使离散搜索方向在期望上对齐连续梯度。
- **Pool 递归与精英裁剪**：候选池按 $J$ 值排序，精英裁剪以概率 $\pi_t$ 丢弃非最优父代（精英裁剪时 $\pi_t=0$），pool‑gap 递归式 $\mathbb{E}[\Delta_{t+1}]\le(1-\alpha\mu p_t)\Delta_t+p_t\frac{L\alpha^2}{2}\sigma_g^2+\pi_t B_{\max}+2\beta_t G_g^2$，离散情形以收缩因子 $\rho=2\mu(\kappa-L\nu/2)$ 替换 $\alpha\mu$。
- **训练动态机制**：Figure 12 显示持续违反会增大 multiplier 强化对应约束；所有约束满足后稳定/下降的 multiplier 允许在保持可行性的同时提升 accuracy。
- **关键超参**：对偶学习率 $\beta=4$ 为唯一同时在 Airline/Telecom 可行且各域最优的值；TT sampling $k\in\{4,8,16\}$；PLen 阈值 5.0（等价相对超额成本 0.25）。

## 实验与结果
- **数据集**：TAU2‑BENCH（Airline/Retail/Telecom）、GSM8K、AdvBench、PUPA‑IFBench、SWE‑BENCH Lite（10 实例优化/30 实例评估）。
- **SWE‑agent**：CAPO 解决全部 5 个问题，Patch/Tool/Files 成本最低（1.055/0.371/1.300），Resolve 率 0.167，唯一满足 AllSat 阈值（Patch≤1.2、Tool≤0.4、Files≤1.7）的方法。
- **TAU2‑BENCH 多域**（GPT‑5‑mini）：Airline Acc=**0.5000**、HAR=**0.0500**；Retail Acc=**0.5500**；Telecom Acc=**0.5000**、ToolEx=0.3395，均为各域最高可行结果。Ministral‑8B 上亦取得最高可行 Acc。
- **RL 基线对比**：StablePrompt 与 Agent‑GRPO 在 Airline/Telecom 均不可行；DCAPO 是唯一在所有三个域同时满足 AllSat 的对比方法。
- **
