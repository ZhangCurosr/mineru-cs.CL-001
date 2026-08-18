---
title: "Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models"
source: https://arxiv.org/pdf/2608.10484v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:57:06"
---

# 论文速读：Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models

## 一句话总结
本文指出当前 VLA 的动作分词器仅以 Euclidean 空间重建为目标，会系统性丢失动词所承载的运动动力学语义；为此提出 **SALT**（Semantically ALigned action Tokenizer），在 VQ-VAE 分词器中引入冻结 VLM 还原 episode 指令的生成式辅助损失，使动作离散表征与语言抽象对齐。实验表明该方法在 SimplerEnv 上将任务成功率从 42.7% 提升至 71.9%，且动作词汇集自然涌现出动词特化编码。

## 研究问题与动机
1. **核心问题**：现有 VLA（如 RT-1/2、OpenVLA 等）的动作表征仅在 Euclidean 控制空间下以 L1/L2 重建损失优化，未显式对齐语言抽象（尤其是动词描述的“如何执行”），导致离散化动作 token 丢失对语言条件控制至关重要的语义结构。
2. **现有方法不足**：主流 discrete action tokenization（Bin、VQ-VAE、FAST）在语言介入前即固定动作词汇表；重建误差小不代表语义等价，连续空间中的微小扰动可能对动词分类产生重大影响，形成语言→控制的表征瓶颈。
3. **动词语义的双重性**：动词同时编码 **action goal**（首尾帧视觉状态变化）与 **motion dynamics**（7-DoF 轨迹形态、接触模式、夹爪时序）。仅靠视觉观测无法捕获后者，必须依赖动作表征传递。
4. **诊断证据**：在 BridgeV2 上的互信息分析表明，随压缩率提高，Bin/VQ-VAE/FAST 等纯重建分词器保留的 `I(verb; tokens)` 单调下降，下游策略训练无法完全弥补该损失。

## 核心贡献（创新点）
1. **提出 SALT 语义对齐动作分词器**：在 VQ-VAE 训练中插入冻结 VLM 的指令生成辅助目标，使量化隐变量直接驱动语言还原；与已有工作的本质区别在于将语义对齐压力施加于分词器训练阶段，而非修改下游 VLA 架构或引入对比学习模块。
2. **建立动作-语言对齐的诊断框架**：利用交叉熵探针估计 `I(Y; X)` 量化动词与运动动力学/动作目标的互补信息量，并首次系统性证明离散化重建分词会随压缩率升高而侵蚀 verb grounding 信号。
3. **揭示动词特化 codebook 的自然涌现**：证明语义对齐无需预定义动词词表或对比负样本，仅凭生成式语言监督即可促使离散动作词汇表按动作语义（如 flip、turn）重新组织，而非均匀混合于高频通用类。
4. **验证语义对齐对下游控制的显著增益**：在 SimplerEnv 上实现 71.9% 平均成功率，较纯重建 VQ-VAE（42.7%）提升 29.2 pp，同时保持同等 L1 重建保真度，证实“动作表征应同时保留可执行轨迹与语言抽象”的设计原则。

## 方法详解
1. **基础架构**：采用残差 VQ-VAE 动作分词器，将 8-timestep、7-DoF 动作 chunk 编码为 $K=7$ 个 codebook 索引，量化隐变量为 $\mathbf{q}_i = \sum_{k=1}^{K} \mathbf{e}_{z_{i,k}}^{(k)}$。
2. **损失函数设计**：总损失 $\mathcal{L} = \mathcal{L}_{\mathrm{recon}} + \mathcal{L}_{\mathrm{VQ}} + \lambda \mathcal{L}_{\mathrm{align}}$。其中 $\mathcal{L}_{\mathrm{recon}}$ 为 L1 重建损失，$\mathcal{L}_{\mathrm{VQ}}$ 包含 codebook 更新与 commitment loss，$\mathcal{L}_{\mathrm{align}}$ 为语言对齐损失。
3. **对齐机制**：将 episode
