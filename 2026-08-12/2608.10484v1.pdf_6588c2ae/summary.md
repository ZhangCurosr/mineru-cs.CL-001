---
title: "Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models"
source: https://arxiv.org/pdf/2608.10484v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:56:30"
---

# 论文速读：Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models

## 一句话总结
本文指出当前视觉-语言-动作模型（VLA）中仅以欧氏空间重建为目标的动作分词器会系统性丢失动词接地信息，提出SALT（Semantically ALigned action Tokenizer）通过在VQ-VAE训练中引入冻结VLM的指令生成辅助损失，使离散动作码本保留语义结构，在SimplerEnv基准上将策略平均成功率从42.7%提升至71.9%。

## 研究问题与动机
- VLA的动作表征通常依赖L1/L2重建损失训练离散分词器（如VQ-VAE、Bin、FAST），隐含假设“低重建误差即保留语言接地信号”，但该假设在压缩离散化过程中缺乏验证。
- 动词语义同时编码动作目标（视觉状态变化）与运动动力学（轨迹形态、夹爪时序、接触模式），后者无法仅从首尾帧视觉推断，必须通过动作表征传递。
- 实证表明，纯重建导向的离散分词会随压缩率升高系统性稀释动作-动词互信息，形成语言条件控制的信息瓶颈。
- 现有VLA研究多聚焦视觉-语言对齐（对象/空间引用），动作接口的语义结构设计长期被忽视。

## 核心贡献（创新点）
- **诊断框架**：在BridgeV2上建立互信息探测协议，首次量化证明动作轨迹携带独立于视觉结果的动词接地信息，且主流离散分词会系统性侵蚀该信息。
- **SALT对齐分词器**：在VQ-VAE训练中引入冻结预训练VLM的指令生成辅助损失，使量化潜变量可直接预测原始自然语言指令，无需额外文本编码器或对比负样本。
- **性能跃升与保真度兼顾**：SALT在SimplerEnv达到71.9%平均成功率，较纯重建VQ-VAE提升29.2个百分点，同时在匹配压缩率下保持相近的L1重建误差（0.088 vs 0.080）。
- **码本自组织现象**：语义对齐促使动作词汇表涌现高度动词特化单元（如flip专属码98%、turn 74%），而重建分词器的码元多呈现语义混杂分布。
- **即插即用兼容性**：仅修改分词器训练目标，下游VLA架构、推理流程与token预测接口完全不变，具备工程落地便利性。

## 方法详解
- **基础分词架构**：采用残差VQ-VAE将连续动作块（8步，7-DoF）编码为K=7个残差组的码本索引，量化潜变量 $\mathbf{q}_i = \sum_{k=1}^K \mathbf{e}_{z_{i,k}}^{(k)}$ 通过直通过量化器（straight-through estimator）传递梯度。
- **总损失函数**：$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{VQ}} + \lambda \mathcal{L}_{\text{align}}$，其中 $\mathcal{L}_{\text{recon}}$ 为动作重建损失，$\mathcal{L}_{\text{VQ}}$ 包含码本EMA更新与commitment loss，$\mathcal{L}_{\text{align}}$ 为语言对齐损失，$\lambda$ 为平衡权重。
- **对齐机制**：将episode内M个动作块的量化潜变量转换为软prefix embedding $\mathbf{p}_i = g\mathbf{q}_i + \mathrm{PE}(i)$（$g$为可学习标量，$\mathrm{PE}$为位置编码），拼接至冻结的Qwen2.5-0.5B语言模型，辅以简短文本prompt，令LM自回归生成该episode的自然语言指令 $w_{1:L}$。
- **对齐损失**：$\mathcal{L}_{\text{align}} = -\frac{1}{L}\sum_{t=1}^{L} \log p_{\text{LM}}(w_t | w_{<t}, \mathbf{P}, s)$，梯度经直通过量化器反传至编码器与码本，实现动作潜空间向语言语义空间的生成式对齐。
- **部署流程**：分词器训练完成后冻结，下游VLA训练阶段策略仅预测离散码本索引，由分词器解码为连续动作；语言模型在策略训练前丢弃，不增加推理开销。

## 实验与结果
- **数据集与基线**：BridgeV2（27,271条带自然语言指令的真实WidowX操作轨迹，17个动词类别）；对比分词器包括Bin（逐维均匀离散）、FAST（频域变换+BPE，V=1024）、VQ-VAE（纯重建）与SALT；策略基于miniVLA（Qwen2.5-0.5B backbone）。
- **闭环任务结果**（SimplerEnv WidowX套件，4任务×24 rollout）：SALT平均成功率71.9%，VQ-VAE为42.7%，FAST为31.2%；最大提升见于叠方块（70.8% vs 33.3%）与茄子放入篮子（79.2% vs 33.3%）。
- **语义可解码性**：SALT在离散token ID上的动词宏观F1达39.1%，在策略学到的动作token嵌入上达43.7%，显著优于VQ-VAE（37.3%/38.3%）与FAST（30.3%/36.3%），逼近连续动作参考值（53.0%）。
- **重建保真度**：匹配压缩率（约7 token/8步，7.0–8.6 bits/timestep）下，SALT的L1重建误差（0.088）与VQ-VAE（0.080）相近；63项可解释运动特征的秩相关系数≥0.92，证明语义对齐未牺牲执行精度。
- **词汇表组织分析**：SALT码本出现清晰动词特化行（flip、turn、pour、topple等），而VQ-VAE与FAST的同类动作被分散至语义混杂码元；同一SALT码元可统一不同措辞的等价指令（如含/不含“turn”字样的杠杆操作）。

## 相关工作脉络
- **RT-1/RT-2、OpenVLA、VQ-VLA、BeT、QueST、FAST、OAT**：主流离散动作表示方案，分词器仅优化重建或自监督预测，未显式维持语言可
