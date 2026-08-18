---
title: "CT-Bench-A-Benchmark-for-Longitudinal-3D-Medical-Imaging-Dif"
source: https://arxiv.org/pdf/2608.11534v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:40:47"
---

# 论文速读：CT-Bench-A-Benchmark-for-Longitudinal-3D-Medical-Imaging-Dif
## CT-∆Bench: A Benchmark for Longitudinal 3D Medical Imaging Difference Reporting with Vision-Language Models

## 一句话总结
本文针对临床随访中对比前后CT影像的核心需求，首次提出了纵向CT差异报告生成基准 **CT-∆Bench** 与 **Change-aware** 事件级评估体系，并设计了显式建模时序变化的基线模型 **DeltaMed**，填补了现有医学视觉-语言模型在成对CT时序推理与差异报告生成任务上的空白。

## 研究问题与动机
- **临床需求与现有方法的错位**：CT随访的价值核心在于跨时间点对比以判断病变演化（新发、进展、消退、稳定等），但现有医学报告生成模型与3D医学VLM仍几乎全部聚焦于单次检查的静态描述，缺乏对成对时序影像的直接推理能力。
- **成对CT推理的固有难点**：跨期CT计算开销大、解剖对应关系常不完全对齐、关键变化往往细微且局灶化，导致模型在生成自然语言差异总结时极易出现信息遗漏与幻觉。
- **评估指标的临床脱节**：传统文本相似度指标（ROUGE、BERTScore等）难以衡量模型是否真正捕捉到了临床有意义的时序变化，缺乏面向“差异报告”的专用评测协议。
- **纵向研究的任务边界局限**：已有纵向医学影像工作多将历史检查作为辅助上下文改善当前期报告草稿，任务形式仍为“当前期生成”而非“直接输出聚焦区间变化的差异报告”，且主要集中在2D胸部X光而非3D CT体数据。

## 核心贡献（创新点）
- **提出 CT-∆Bench 基准**：基于 CT-RATE 构建患者级划分的纵向成对CT数据集（2,638 训练 / 169 验证），首次系统化定义“差异报告生成”任务，从数据划分机制上杜绝跨子集信息泄露。
- **设计 Change-aware 评估指标**：突破表面文本相似度局限，通过 LLM 抽取原子变化事件（NEW/RESOLVED/INCREASED/DECREASED/STABLE）并计算 Change-F1、漏报率、幻觉率与变化类型准确率，实现更贴合临床真实性的度量。
- **提供直接推理与间接两阶段的全方位基线对比**：系统评测 5 个主流医学 VLM 在零样本、间接两阶段文本差分及 1%/10%/100% 微调数据下的表现，量化揭示现有模型在时序推理上的根本性不足。
- **提出 DeltaMed 基线模型**：采用共享视觉编码器 + 显式差分支（$z_{t_2} - z_{t_1}$）+ 轻量时序融合模块 + 冻结 LLM（LoRA 微调）的架构，在低数据 regime 下显著优于直接双路径微调的 MedGemma 基线。

## 方法详解
- **基准数据构建**：从 CT-RATE 筛选同患者两次不同时间点的 CT 扫描及对应报告，截取 Prior/Follow-up 报告的 Findings 与 Impression 部分输入 **Gemini-2.5-Flash**，按标准化 Prompt 生成结构化的 Difference Findings 与 Difference Impression 作为参考目标；再用 **Qwen2.5-14B-Instruct** 将参考报告转化为原子变化事件 JSON，格式为 `(type, text)`。
- **DeltaMed 网络架构**：先后两次 CT 分别经共享权重的 **MedSigLIP** 视觉编码器提取特征 $z_{t_1}$ 与 $z_{t_2}$；构造显式差分支 $z_{t_2} - z_{t_1}$ 编码方向性时序变化；将 $[z_{t_1}, z_{t_2}, z_{t_2} - z_{t_1}]$ 拼接后经过线性投影与归一化完成时序融合，输出融合表示 $H$。
- **解码与训练策略**：融合特征经原有多模态投影器接入冻结的 **Gemma 3 4B** 语言模型进行自回归解码；采用参数高效微调，仅更新时序融合模块与 LLM 中的 LoRA 适配器，视觉编码器与基础 LLM 权重保持冻结；损失函数为条件负对数似然：
  $$\mathcal{L}_{\text{gen}} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, H)$$
- **Change-aware 事件匹配协议**：采用模糊事件匹配流程（文本规范化 → 临床约束硬过滤 → 软相似度打分 $\tau=0.5$ → 一对一最大匹配），在此基础上按标准 P/R 框架计算 TP/FP/FN，并导出四项核心指标：
  - Change-F1：$\frac{2\text{TP}}{2\text{TP}+\text{FP}+\text{FN}}$
  - Missing Rate：$\frac{\text{FN}}{\text{TP}+\text{FN}}$
  - Hallucination Rate：$\frac{\text{FP}}{\text{TP}+\text{FP}}$
  - Change Type Accuracy：匹配事件中预测类型与参考类型一致的比例。

## 实验与结果
- **数据集与设置**：CT-∆Bench（患者级划分）；评测基线涵盖 MedGemma-1.5-4B、M3D-LaMed-Phi-3-4B、RadFM-13B、Med3DVLM-Qwen2.5-7B、Merlin-RadLLaMA-7B；实验分为零样本、两阶段间接管道、1%/10%/100% 微调三种设定，硬件为 2×80GB NVIDIA A100。
- **零样本表现**：所有模型 Change-F1 极低（0 ~ 0.0175），RadFM-13B 完全失效（Change-F1=0, Missing/Hallucination Rate 均为 1）；文本指标最优的 Med3DVLM-Qwen2.5-7B（ROUGE-L 0.098, BLEURT 0.3822）Change-F1 仍仅 0.0138，充分揭示文本相似性与临床正确性的严重脱节。
- **两阶段间接管道**：对 RadFM-13B 与 Med3DVLM-Qwen2.5-7B 有一定正面作用，但存在误差传播风险；MedGemma 略降，Merlin 性能断崖式下跌（Change-F1 降至 0），说明中间单期报告的质量直接决定最终差分上限。
- **微调对比（DeltaMed vs Direct Paired MedGemma）**：1% 数据下 DeltaMed Change-F1 达 0.0909（MedGemma 仅 0.0010）；10% 下为 0.1313 vs 0.0649；100% 下为 0.1980 vs 0.1577。DeltaMed 在低/中数据 regime 的事件检测与漏报控制上优势显著，验证了显式差分支归纳偏置的有效性；文本指标方面 DeltaMed ROUGE-L 全面领先，但 MedGemma 在 BERTScore/BLEURT 上略高，再次印证单靠文本指标会高估时序推理能力。
- **临床验证**：50 例独立医生双盲评审，合成参考报告可接受度/正确性/完整性均分 ≥4.82/5，99/100 判定为临床可接受，零严重幻觉/遗漏；事件抽取正确率 4.83/5，医师间一致率达 97.5%，证实基准合成管线与评估协议具备可靠信度。

## 相关工作脉络
- **单期医学影像报告生成**（如 M3D-LaMed、MedGemma、CT-RATE 相关工作）：聚焦单次影像到文本的静态描述，
