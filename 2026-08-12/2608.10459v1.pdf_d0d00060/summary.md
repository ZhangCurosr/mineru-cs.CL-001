---
title: "MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection"
source: https://arxiv.org/pdf/2608.10459v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:55:38"
---

# 论文速读：MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection

## 一句话总结
本文提出 MD-ProTector，一种仅依赖输入文本的轻量级编码器检测器，通过在嵌入空间中为人类文本与 LLM 生成文本各维护一组可训练原型，并引入 Prototype Positioning loss 显式分离类共享方向与类内残差变异，从而在跨域、跨生成器、对抗扰动及多语言等五类严苛设置下显著提升 AI 生成文本检测的平衡召回率与鲁棒性。

## 研究问题与动机
- **核心问题
