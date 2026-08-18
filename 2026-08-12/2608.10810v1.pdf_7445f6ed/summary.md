---
title: "Surfacing the Unsaid: CUE-Bench for Affective Stance in Chinese Discourse"
source: https://arxiv.org/pdf/2608.10810v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:43:19"
---

# 论文速读：Surfacing the Unsaid: CUE-Bench for Affective Stance in Chinese Discourse

## 一句话总结
本文提出 CUE-Bench，一个面向中文话语中“言外之意”的情感理解基准。通过显式-隐式立场矩阵构建 9 类结构化情感立场，并设计矩阵引导的链式推理提示协议，在立场识别、语用意图与细粒度情感三个关联任务上显著提升了 LLM 的深层情感推理能力。

## 研究问题与动机
- 现有情感基准多聚焦表层极性分类或单一任务，缺乏对“显式表达”与“隐式情感倾向”交互关系的结构化建模，难以评估模型在礼貌、压制、反讽、委婉等中式话语场景下的深层推理能力。
- 传统评测将语用意图与细粒度情感视为独立预测目标，忽略了它们作为显式-隐式立场冲突中介变量的作用，导致对“表面中立实则不满”等错位表达的诊断灵敏度不足。
- 中文话语高度依赖语境与交际策略，单一维度的情感标注无法刻画说话人的交际定向（Affective Stance），现有数据集在多源场景覆盖与多任务联合评估上存在明显短板。

## 核心贡献（创新点）
- **提出 CUE-Bench 中文情感基准**：构建 51,823 条样本，提供显式/隐式情感层、9 类情感立场、8 类语用意图与 25 类细粒度情感的四级结构化标注，支持多任务联合诊断。
- **设计显式-隐式立场矩阵（Explicit-Implicit Stance Matrix）**：将情感立场定义为显式信号与隐式倾向的有序对映射（φ: O×O→S），以 3×3 矩阵形式生成 9 种可解释立场类别，实现从“所言”到“所隐”的结构化关联。
- **提出矩阵引导链式推理（Matrix-Guided CoT）**：强制 LLM 按“显式信号→隐式倾向→情感立场→语用意图→细粒度情感”固定顺序输出中间字段，相比自由 CoT 更可追踪错误传播路径。
- **构建混合标注与质量控制流水线**：双模型候选生成 + 人工裁决 + LLM 正反向一致性校验，有效降低位置偏差并量化残余噪声（约 3.1%）。

## 方法详解
- **样本与标注定义**：每个实例为 x_i = (C_i, u_i)，其中 C_i 为上下文，u_i 为目标话语。标注目标 y_i = (y_exp, y_imp, y_stance, y_intent, y_emotion)，分别对应显式情感层、隐式情感倾向、情感立场、语用意图与细粒度情感。
- **立场矩阵构建**：定义方向投影算子 π(·) 将任意情感层映射到三值取向空间 O = {+, 0, −}。显式信号 e_i = π(y_exp)，隐式倾向 h_i = π(y_imp)。情感立场 s_i = φ(e_i, h_i)，φ 将 9 种 (e_i, h_i) 组合映射至 9 类立场（如 POSITIVE、FORMULAIC POSITIVE、SARCASTIC NEGATIVE、VEILED NEGATIVE、NEUTRAL 等）。矩阵满足结构依赖：改变 e_i 或 h_i 必然改变 s_i，保证可解释性。
- **矩阵引导 CoT 推理链**：形式化为确定性依赖路径：
  (ê_i, ĥ_i) = F_sig(x_i)
  ŝ_i = φ(ê_i, ĥ_i)
  ŷ_intent = F_prag(x_i, ê_i, ĥ_i, ŝ_i)
  ŷ_emotion = F_emo(x_i, ê_i, ĥ_i, ŝ_i, ŷ_intent)
  提示工程中要求模型按顺序输出中间字段；Oracle 消融仅注入人工金标中间字段以验证各层贡献，不改变目标字段。
- **标注流水线**：使用 GPT-4o-mini 与 DeepSeek-V4 独立生成候选；一致样本直接保留（20,000 条）；不一致样本抽取 10,000 条由人类专家裁决作为 Gold 集；剩余 30,000 条交由 LLM 裁决，并通过正反候选顺序交换（forward-reverse）校验，仅保留两次决策一致的样本（21,823 条），最终汇聚为 51,823 条基准。

## 实验与结果
- **设置**：CUE-Bench（51,823 样本，按来源分层划分）。评测模型覆盖闭源（DeepSeek-V4-Flash, GPT-4o-mini）与开源（LLaMA-4-Maverick, LLaMA-3.1-8B, Qwen-3-8B）。基线为 Direct、Few-shot、Free-form CoT，本文方法为 Matrix-Guided CoT。指标为 Acc、macro-F1、weighted-F1。
- **主结果**：矩阵引导方法在所有模型上取得最高平均得分，较最优基线提升 +0.027 ~ +0.096。语用意图增益最稳定，DeepSeek-V4-Flash 准确率提升 +0.120（0
