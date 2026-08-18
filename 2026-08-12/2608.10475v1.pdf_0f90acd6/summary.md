---
title: "E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l L<sub>a</sub>n<sub>guage</sub>"
source: https://arxiv.org/pdf/2608.10475v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 02:17:08"
---

# 论文速读：E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>atu</sub>ra<sub>l</sub> L<sub>an</sub>gu<sub>ag</sub>e

## 一句话总结
本文在 Concordia+ContractSim 仿真平台中，系统评估了 Claude Opus 5、Gemini 3.6 Flash、GPT-5.6-Sol 三类前沿 LLM agent 通过自然语言协商生成结构化合同的能力，重点量化其在效率、互惠利益与条件执行（contingent compliance）下的合规性、违约响应与后悔损失。

## 研究问题与动机
- LLM agent 能否在长时间、附带条件的自然语言合同协商中，可靠地实现经济效率与相互受益？
- 现有评测多聚焦“是否达成协议”，缺乏对**不完全合同、条件执行概率与违约动态响应**的系统性量化指标。
- 自然语言提案到形式化约束（价格/排程/条款）的语义漂移与算术一致性如何影响最终协议质量？
- 不同推理能力等级的 LLM 在面对 grim trigger、payment deduction、rollover、substitution 等条件条款时，履约与报复策略呈现何种差异？

## 核心贡献（创新点）
1. **端到端合同协商-执行评测框架**：构建基于 Concordia+ContractSim 的仿真管道，支持从自然语言提案到结构化约束 $C^\omega$ 的离线翻译、质量验证与绩效测试，填补自然语言合同形式化对齐的评测空白。
2. **双轨量化指标体系**：同时覆盖协商阶段（提案可行性、合同完整性、相互受益、采纳率后悔）与执行阶段（宏观/微观平均合规率、ex-post regret、单侧/互惠违约率、TFT 遵循度），实现对协议质量的细粒度归因。
3. **自动化合成合同基准**：以 grim-trigger 为基底，通过多起点坐标上升搜索生成 180 份覆盖 5 个 $P_{sat}$ 水平（0.5–0.9）与 6 种条件变体的标准化合同，支持 Catering/Hotel Cleaning/AI Model Hosting 多领域迁移。
4. **Pareto 前沿对齐分析**：揭示 LLM 谈判结果在理论边界 $F_1$（非条件合同）与 $F_2$（支持条件条款且 $\bar{P}_{sat} \geq 0.95$）上的分布规律，确立“贴近理论最优”优于“单纯追求高总效用”的评测范式。

## 方法详解
- **环境设定**：6 个 Catering 环境（按随机性与价值/资本比划分，含高随机 Env 5/6），启用 high reasoning 模式。
- **合同结构化翻译**：独立离线翻译器（Gemini 3.6 Flash）将自然语言 $\omega$ 映射为 $C^\
