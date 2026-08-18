---
title: "VisEditBench: Can Vision-Language Models Edit Visualization Code from Multimodal Feedback?"
source: https://arxiv.org/pdf/2608.10408v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:08:54"
---

# 论文速读：VisEditBench: Can Vision-Language Models Edit Visualization Code from Multimodal Feedback?

## 一句话总结
本文提出 VisEditBench，首个面向可视化代码多模态迭代编辑的基准测试，涵盖反馈引导修复与参考引导重样式两类真实工作流；实验揭示当前 VLM 代码可执行率虽高，但在视觉忠实度与风格适配上仍存在显著差距，并在此基础上提出 VisEditAgent 渲染反馈驱动的迭代编辑框架，将 GPT-4o 整体通过率从 55.75% 提升至 67.99%。

## 研究问题与动机
- **核心问题**：现有 VLM 评测多集中于“从零生成图表”或“图表反推代码”，但实际数据可视化工作流是高度迭代的：用户已拥有可执行代码与渲染图，需基于多模态反馈（标注缺陷图/参考风格图+文本指令）进行局部修改，且必须保持底层数据语义不变。
- **现有基准不足**：Text2Vis/VisEval 等仅评估单次文本到可视化的生成；ChartMimic/Plot2Code 仅评估图表到代码的重建；SWE-bench Multimodal 等通用代码基准未覆盖可视化特有的误导性编码、布局拥挤、级联样式变化等挑战。
- **编辑难度本质**：可视化编辑要求模型联合推理代码结构、渲染视觉反馈、用户意图与图表美学，小改动易引发级联布局/样式变化，需要比单次代码生成更强的多模态 grounded reasoning。
- **现有模型瓶颈**：强模型代码可执行率已较高（GPT-4o 达 93.31%，Claude-4.6-Sonnet 达 96.19%），但视觉相似度与风格匹配得分仍偏低，表明“能跑通”不等于“改对/改像”。

## 核心贡献（创新点）
1. **提出 VisEditBench 基准**：构建 1,395 条人工标注的可视化代码编辑任务，覆盖反馈修复与参考重样式两种设定。
   - *本质区别*：突破现有基准单次生成/单向重建的局限，首次系统评测基于多模态反馈的可视化代码迭代修改。
2. **构建八类编辑意图分类体系**：将编辑任务细分为正确性修复、质量改进、鲁棒性提升、风格适配、约束满足、一致性调和、结构转换与风格感知修复。
   - *本质区别*：超越传统的二分类 bug/feature 标注，提供可定量分析的意图级难度分布与失败模式归因。
3. **提出 VisEditAgent 渲染 grounded 编辑框架**：通过编辑规划→多候选生成→执行渲染→视觉校验→反馈精修的闭环 pipeline 实现迭代优化。
   - *本质区别*：打破单次 prompt 生成范式，将实时渲染图像作为显式校验信号，显著提升视觉忠实度与布局对齐能力。
4. **设计多维度严格评测协议**：引入代码可执行性、任务准确率、可读性/清晰度、视觉质量、视觉相似度五项指标，并定义严格的 final pass rate 阈值与自动化 VLM 评估器。
   - *本质区别*：从纯代码正确性扩展至视觉对齐与语义保持，配套 rubric 具备意图感知特性，且与人工评分 Pearson 相关达 83%~87%。

## 方法详解
- **任务形式化**：每个样本定义为 $x_i = (c_i, I_i, u_i, c_i')$，其中 $c_i$ 为原始可视化代码，$I_i$ 为渲染图/参考图/标注图，$u_i$ 为自然语言编辑指令，目标为生成可执行代码 $c_i'$ 使新渲染图满足指令且保持数据语义。两种 setting：(1) Feedback-guided repair：输入缺陷或人工标注图+文字反馈进行修复；(2) Reference-guided restyling：输入目标风格参考图，在不变动数据的前提下匹配视觉风格。
- **VisEditAgent 四阶段流程**：
  1. **Edit Planning**：结合输入代码、图表图像、指令与编辑意图，定位视觉问题所在代码区域并制定修改策略。
  2. **Candidate Generation**：针对同一修改目标生成多个候选代码，分别执行并渲染为图表图像，捕获可视化编辑中“多解并存”的特性。
  3. **Visual Validation & Selection**：基于任务指令、输入图与参考图对候选进行联合评估，筛选在程序正确性与视觉对齐上最优者。
  4. **Feedback-Based Refinement**：利用执行诊断、渲染失败日志与视觉校验反馈对选中候选进行迭代精修，纠正一次性生成的典型错误。
- **评估协议**：采用基于固定 rubric 的 VLM 自动评估器（默认 GPT-4o，评估自身输出时替换为 Gemini 2.5 Pro 避免自评估偏差）。最终通过率阈值严格要求：可执行=1、任务准确率≥4.5、可读性/视觉质量≥4.0、视觉相似度≥90。

## 实验与结果
- **数据集与规模**：VisEditBench 共 1,395 条任务，来源为 Stack Overflow/Matplotlib-Vega-Lite GitHub issue 与 Text2Vis 基准上的模型失败案例；横跨 2 种可视化库（Matplotlib 1,156 / Vega-Lite 239）、多种图表类型与难度（Easy 37.0% / Medium 35.7% / Hard 27.3%），57.1% 为多因故障任务。
- **评测基线**：20 个 SOTA VLM（GPT-4o/5, Claude-4.5/4.6-Sonnet, Gemini-3.0-Flash, Qwen3-VL 系列, InternVL-3.5, Gemma-3, Pixtral 等）。
- **主要结果**：
  - **整体 Pass Rate**：Claude-4.6-Sonnet 以 74.46% 领先，GPT-5 为 65.68%，GPT-4o 为 55.75%；多数开源模型低于 50%，Qwen3-VL-32B 最高为 51.72%。
  - **分项差距**：正确性修复（70.34%）与质量改进（79.73%）表现较好；但视觉接地风格适配极弱，Claude-4.6-Sonnet 仅 55.71%，GPT-4o 仅 10.00%。
  - **可执行≠视觉正确**：Claude-4.6-Sonnet 与 GPT-4o 可执行率高达 96.19% / 93.31%，但视觉相似度仅 89.42 / 78.13
