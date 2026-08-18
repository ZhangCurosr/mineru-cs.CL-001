---
title: "ReRef-3D-A-Benchmark-for-Spatial-Referring-Expression-Guided"
source: https://arxiv.org/pdf/2608.16011v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:54:44"
---

# 论文速读：ReRef-3D-A-Benchmark-for-Spatial-Referring-Expression-Guided

## 一句话总结
本文提出 ReRef-3D，一个面向 3D 视觉语言模型的**语言引导场景重排基准**，要求模型将自包含的空间指代表达式解析为连续放置坐标并满足多重几何与物理约束。评测发现当前最强模型 LLaVA-3D 仅能正确完成约 68.3% 的指令，且物理碰撞规避而非语义理解是主要瓶颈，句式变化对性能影响极小。

## 研究问题与动机
1. **静态理解与动态重排的鸿沟**：现有 3D 视觉语言基准（grounding、QA、captioning、文本规划）主要评估对固定场景的被动理解，缺乏对“将空间语言指令转化为可验证的连续场景状态改变”这一中间能力的系统评测。
2. **单点距离评估的误导性**：空间指令通常定义的是合法放置区域而非单一坐标，仅用预测点与标注点的欧氏距离评估会严重低估模型性能，甚至错误排序模型优劣。
3. **目标与约束未耦合**：最接近的基线 PlaceIt3D 将待放置物体作为独立资产提供，而真实交互中物体与其目的地约束往往融合在同一条自包含指令
