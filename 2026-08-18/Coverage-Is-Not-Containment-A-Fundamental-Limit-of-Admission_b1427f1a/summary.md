---
title: "Coverage-Is-Not-Containment-A-Fundamental-Limit-of-Admission"
source: https://arxiv.org/pdf/2608.16044v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:27:33"
---

# 论文速读：Coverage-Is-Not-Containment-A-Fundamental-Limit-of-Admission

## 一句话总结
本文证明了对抗向量检索协同投毒的“准入时防御”存在根本性几何极限：攻击者仅需注入少量各自独立可被准入的文档，即可共同包围目标查询并占据其 top-k；任何仅基于文档与哨兵点的准入时统计均无法在可控误报率下将其与合法垂直领域批量上传区分开，而引入检索时需求观察可突破该极限。

## 研究问题与动机
- **核心问题**：在开放信道持续写入的 RAG 系统中，协同投毒攻击能否绕过所有准入时（admission-time）防御机制？
- **现有方法不足**：现有 hub 检测防御（如
