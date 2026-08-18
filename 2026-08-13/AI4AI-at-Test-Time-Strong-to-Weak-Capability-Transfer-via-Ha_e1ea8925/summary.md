---
title: "AI4AI-at-Test-Time-Strong-to-Weak-Capability-Transfer-via-Ha"
source: https://arxiv.org/pdf/2608.12307v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:00:50"
---

# 论文速读：AI4AI-at-Test-Time-Strong-to-Weak-Capability-Transfer-via-Ha

## 一句话总结
本文提出并系统验证了“测试时强模型搭建（strong-to-weak scaffolding）”范式：一个更强的builder模型仅利用5%验证集数据迭代设计推理时外部harness，即可在不更新目标模型参数的前提下，将较弱target模型在Theory-of-Mind基准上的平均准确率从0.49大幅提升至0.91，证明推理环境工程可成为训练时蒸馏的有效互补路径。

## 研究问题与动机
- 现有模型蒸馏与能力迁移方法均依赖更新弱模型参数（知识蒸馏、on-policy蒸馏、指令微调等），属于训练时转移；本文
