---
title: "TOWARDS-UNDERSTANDING-ON-POLICY-DISTILLA-TION-THROUGH-THE-LE"
source: https://arxiv.org/pdf/2608.11829v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:31:24"
---

# 论文速读：TOWARDS-UNDERSTANDING-ON-POLICY-DISTILLA-TION-THROUGH-THE-LE

## 一句话总结
本文通过测试时缩放（Test-Time Scaling）视角系统分析了对策蒸馏（OPD），发现OPD主要提升了小采样预算下的采样效率，并未持续扩展学生的推理能力边界，甚至会导致部分原本可解问题变得不可解，呈现出“幻觉式蒸馏（illusory distillation）”效应。

## 研究问题与动机
- OPD作为LLM推理增强的热门后训练技术，被广泛假设能通过强教师指导为学生注入新知识，从而扩展推理能力边界。
- 然而近期实证研究也报告了OPD训练后模型可能无法稳定超越预训练基座、甚至表现下滑的矛盾现象。
- 核心科学问题：OPD究竟是在真正转移教师的新推理能力，还是仅像RL训练一样重塑分布、提升学生对已有能力的访问效率？
- 现有工作多依赖单一K值或离线准确率评估，缺乏从测试时计算分配角度对“采样效率”与“能力边界”的定量分离与分析。

## 核心贡献（创新点）
1. **构建基于测试时缩放的OPD系统性诊断框架**：通过变化采样预算K并同步分析pass@K与avg@K，首次清晰解耦OPD对采样效率与能力边界的差异化影响。
2. **提出“幻觉式蒸馏（illusory distillation）”概念**：论证OPD的表观增益主要来源于提升学生访问已有正确推理路径的概率，而非真正从教师处获取新的推理能力。
3. **揭示OPD“遗忘多于学习”的能力边界收缩规律**：以pass@1024为可解性判据，发现OPD训练后变为不可解的题目比例高于新变得可解的比例。
4. **验证改进型OPD变体的共性权衡行为**：ExOPD、Direct-OPD、EOPD及纯前向KL等变体均复现了“小K增益、大K衰减”的趋势，表明这是OPD范式的内在属性。
5. **建立与离策略蒸馏的对照基准**：证明离策略蒸馏能同时提升小K与大K性能、真正扩展能力边界，反衬出OPD与经典知识转移机制的本质差异。

## 方法详解
- **测试时缩放评估协议**：对同一问题独立采样K条推理轨迹（K∈{1,2,…,512,1024}），计算两类互补指标：
  - `pass@K`：K次采样中至少有一条正确的期望概率，小K反映采样效率，大K反映能力边界可达性。
  - `avg@K`：K次采样正确率的期望，直接衡量推理分布的整体质量与效率。
- **问题级可解性分类分析**：以pass@1024为阈值将题目划分为保留（两模型均可解）、学习（仅OPD后可解）、遗忘（仅Base可解）三类，量化能力边界的净迁移。
- **训练动力学追踪**：记录OPD各checkpoint（Step 20/80/140/200/260）的pass@1与pass@1024，观察小K提升与大K退化的出现时机与波动特征。
- **多设置与多基线实验**：覆盖3组学生-教师配置（Qwen3、Skywork、JustRL），对比标准OPD与5种改进变体（EOPD, ExOPD, Direct-OPD, 纯forward-KL），并引入DeepSeek-R1-Distill-Qwen-1.5B/7B作为离策略蒸馏对照。
- **轨迹困惑度分析**：计算Base、OPD、Teacher三者在Base与Teacher分布下生成轨迹的perplexity，验证OPD是否引入了教师独有的新推理模式。
- **核心优化目标**：标准OPD采用反向KL散度 $\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x,y\sim\pi_\theta}\left[\sum_t D_{\mathrm{KL}}(\pi_\theta(\cdot|x,y_{<t}) \| \pi_T(\cdot|x,y_{<t}))\right]$，实践中使用top-k token KL近似；离策略蒸馏则直接在教师生成的轨迹上拟合。

## 实验与结果
- **数据集与基准**：训练集DAPO-Math-17k；测试集AMC2023、AIME2024、AIME2025、AIME2026。评估温度0.7、top-p 0.95、seed 0，上下文32,768 tokens，prompt上限1,024，输出上限31,744。
- **主要定量结果**：
  - **小K显著增益**：Qwen3-AMC23 pass@1从32.3%→45.4%；JustRL-AMC23从72.2%→87.6%（+15.4pp）；Skywork-AIME24从29.9%→36.4%。
  - **大K优势逆转或持平**：Qwen3-AIME24 pass@1024为Base 70.0% > OPD 53.3%；Skywork-AIME25为Base 76.7% > OPD 70.0%；JustRL-AIME26为Base 86.7% > OPD 83.3%。
  - **avg@K持续领先**：所有设置、所有K下OPD均稳定高于Base，印证效率提升的一致性。
  - **遗忘>学习**：大K判据下forgotten集合显著大于learned集合，净变化为负；保留类问题占比随K增大而上升。
  - **离策略蒸馏对照**：DeepSeek-R1-Distill-Qwen-1.5B/7B在小K与大K下均实现pass@K与avg@K双提升，真正扩展能力边界。
  - **变体一致性**：EOPD/ExOPD/Direct-OPD等与标准OPD趋势一致；仅纯forward-KL在AIME2026 pass@1024略超Base。
- **最强结果与提升幅度**：OPD在K=1时提升最大（JustRL-AMC23 +15.4pp）；但大K场景下Pre-OPD Base模型始终构成能力边界的强基线。

## 相关工作脉络
- **On-Policy Distillation (GKD/MiniLLM)**：Agarwal et al. (2024)、Gu et al. (2024) 提出用学生自采样轨迹+教师反馈缩小train-test mismatch；本文在此基础上指出OPD的本质是分布重塑而非能力转移。
- **Test-Time Scaling & Pass@K**：Brown et al. (2024) 开创通过重复采样分配推理计算；Yue et al. (2025) 曾用大K曲线分析RLVR；本文将该诊断框架系统化并首次应用于OPD。
- **Distillation & Capability Transfer**：Hinton et al. (2015) 与Hsieh et al. (2023)/Guo et al. (2025) 的经典蒸馏假设能实现知识/思维链迁移；本文证明OPD偏离此范式，更像“访问优化”。
- **改进型OPD变体 (EOPD/ExOPD/Direct-OPD)**：J
