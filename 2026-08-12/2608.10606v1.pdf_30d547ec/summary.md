---
title: "ASR-ROUNDTRIP EVALUATION CAN MASK CONTEXT- AND CONVENTION-DEPENDENT READING ERRORS IN CHINESE NEWS TTS"
source: https://arxiv.org/pdf/2608.10606v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:59:53"
---

# 论文速读：ASR-ROUNDTRIP EVALUATION CAN MASK CONTEXT- AND CONVENTION-DEPENDENT READING ERRORS IN CHINESE NEWS TTS

## 一句话总结
本文揭示了 ASR roundtrip 评测在中文新闻 TTS 场景下存在系统性“假阴性”盲区：当文本发音依赖上下文或领域惯例时，TTS 可能读错但 ASR 仍会将其转录为表面正确的文本，从而掩盖真实听感错误。研究通过带完整分母的定向人工审计、跨度隔离诊断与跨系统对照，验证了该机制的存在性与评估器依赖性，并明确建议将 ASR roundtrip 仅视为筛选工具而非独立真值。

## 研究问题与动机
- **核心问题**：ASR roundtrip（TTS合成音频→ASR转写→比对参考）作为可扩展的 TTS 清晰度代理指标，在中文新闻短形式/领域惯例文本中会漏报听众可感知的读音错误。
- **现有方法不足1**：传统评测过度依赖自动转写匹配，忽视了 ASR 解码器与后处理中的语境恢复及书面规范归一化能力会“修正”错误发音，导致假阴性泛滥。
- **现有方法不足2**：现有中文文本规整与多音字消歧工作聚焦于 G2P 前端优化，缺乏对下游 TTS 输出在 ASR roundtrip 下掩盖效应的机制性审计与定量刻画。
- **动机**：亟需一套音频优先、带完整分母的人工跨度审计协议，量化并可视化此类盲区，防止业界将 ASR roundtrip 误用为免人工评测的金标准。

## 核心贡献（创新点）
1. **提出 CDRD 风险分类体系并构建高覆盖冻结基准**。本研究首次按实体/多音/邻近惯例三分风险跨度，并基于 10.8 万生产脚本挖掘出 200 例冻结基准，区别于仅依赖通用 WER 或 MOS 的粗糙评测范式。
2. **设计音频优先的跨度级人工审计协议**。该协议强制标注员先听音频再查 ASR 转录，杜绝自动指标反哺主观判断，填补了现有 TTS 评测中自动代理指标与人类听感脱节的流程空白。
3. **验证 ASR roundtrip 假阴性机制并给出交叉控制实验**。本文通过定向审计与跨度隔离诊断证实掩盖现象，与以往仅报告单一系统指标的工作相比，明确分离了 TTS 发声错误与 ASR 语境恢复的贡献。
4.
