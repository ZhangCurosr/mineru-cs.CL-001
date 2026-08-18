---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:07:18"
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
论文提出UniF-MoE框架，将Mixture-of-Experts (MoE)中的共享专家建模、细粒度通道选择与动态路由统一为一个有序分配过程，遵循"先共享可复用计算、再路由剩余需求"的原则，在Vision和Language任务上同时提升预测性能并降低计算开销。

## 研究问题与动机
- **现有方法的割裂性**：Shared expert、modular/nested experts、dynamic router等机制通常独立发展，忽视了它们之间的内在依赖关系——提取可复用计算会改变剩余内容，也会影响最佳专家选择与容量需求。
- **固定专家作为计算单元的冗余**：传统top-k路由将整个expert视为不可分割的单位，当多个选中expert存在重复响应时造成冗余计算；同时所有token被分配相同数量的expert，无法适应不同token的语义复杂度差异。
- **通道级可复用性未 exploitation**：稀疏upcycled MoE中expert源自同一预训练FFN，其hidden channel positions可直接比较，但现有工作未系统利用这一结构来分离共享与残差计算。

## 核心贡献（创新点）
1. **揭示路由条件的依赖关系**：发现共激活expert在部分value positions上对齐，移除这些位置会显著改变expert偏好；共享覆盖率越高，残差expert需求量越低，将共享建模、细粒度计算与动态路由统一为"先共享、后路由剩余"的有序原则。
2. **UniF-MoE统一框架**：每个expert被划分为对齐blocks，通过共享需求分数α(x)决定共享block数量与权重，key prototypes选择共享内容，累积路由质量决定残差expert数量，单一router按功能顺序协调三个决策。
3. **Gram正则化促进路由几何**：初始化expert嵌入为正交并施加Gram约束，使路由方向多样化、减少expert重叠，同时保持简单的归一化路由几何。
4. **精度-效率trade-off优化**：在DomainBed和GLUE基准上，相比代表性static/dynamic MoE获得更强预测性能，同时将激活参数减少9.1%、FLOPs减少16.1%，推理时间与内存分别降低45.2%和52.7%。

## 方法详解
- **Blockwise Shared-Residual Partitioning**：每层包含1个shared expert $E_{shr}$ 和K个residual experts $\{E_1,...,E_K\}$，每个expert中间宽度H被分为B个aligned blocks，block b包含位置$\mathcal{H}_b$。所有expert从同一dense FFN初始化，保证block边界对齐。
- **Token-Adaptive Shared Modeling**：扩展router嵌入矩阵$\mathbf{W}_g^\star = [\mathbf{W}_{shr}, \mathbf{W}_g]$，共享需求分数$\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{xW}_{shr})$，其中$\tau=(B-1)/B^2$保证$\alpha \in (\tau, 1-\tau)$，shared block数量$b(\mathbf{x})=\text{round}(B\alpha(\mathbf{x}))$。
- **Shared Block Selection**：用block up-projection key均值$\mu_b$作为prototype，priority$u_b(\mathbf{x})=\mathbf{x}\mu_b$，TopK选择$b(\mathbf{x})$个blocks作为共享路径。
- **Cumulative Residual-Expert Routing**：残差需求$\beta(\mathbf{x})=1-\alpha(\mathbf{x})$，expert按affinity排序$p_i(\mathbf{x})$，激活最小前缀满足$\sum_{i=1}^{k} p_i(\mathbf{x}) \geq \beta(\mathbf{x})$，即$k(\mathbf{x})$个residual experts。
- **Output Merging**：$\mathbf{y}=\alpha(\mathbf{x})E_{shr}^S(\mathbf{x})+\sum_{i=1}^{k(\mathbf{x})}p_i(\mathbf{x})E_{q_i(\mathbf{x})}^\mathcal{R}(\mathbf{x})$，总系数质量受控在$[1, 1+p_{k(\mathbf{x})}(\mathbf{x}))$。
- **Training Objective**：$\mathcal{L}=\mathcal{L}_{task}+\lambda_{div}\mathcal{L}_{div}$，其中$\mathcal{L}_{div}=\|(\mathbf{W}_g^\star)^\top\mathbf{W}_g^\star-\mathbf{I}_{K+1}\|_F$，强制embedding正交以多样化routing方向。
- **Compute Cost**：$C_B(\mathbf{x})=b(\mathbf{x})+k(\mathbf{x})[B-b(\mathbf{x})]$ blocks，reusable work仅执行一次，重复cost限于residual pathway。

## 实验与结果
- **Vision**：基于ImageNet预训练DeiT-S/16，在DomainBed五个数据集上测试。UniF-MoE平均准确率69.5%，超越GMoE(67.9%)、LFME(68.5%)、DynMoE(67.9%)、MASS；在PACS(89.6%)、VLCS(81.7%)、DomainNet(49.4%)上最优，OfficeHome与GMoE并列(74.2%)，TerraIncognita次优(52.6% vs LFME 53.4%)。
- **Language**：基于BERT-large，K=16, B=16。UniF-MoE在GLUE全部五个任务上最优：CoLA(66.83%)、MRPC(91.57%)、QNLI(93.10%)、MNLI(86.84%)、RTE(75.47%)，平均82.76%，超越所有fixed top-k变体、DynMoE(81.64%)、MASS(82.19%)。
- **Cost Comparison**：在VLCS上，相对top-2 GMoE，激活参数减少9.1%、FLOPs减少16.1%、推理时间减少45.2%、内存减少52.7%；推理时间0.17s vs GMoE 0.31s、DynMoE 1.06s。
- **Ablation**：移除任一自适应决策均降低准确率并增加计算；固定α影响最大；Gram正则化$\lambda_{div}=0.01$时表现最佳，减少62.9% co-activation。

## 相关工作脉络
- **Shared Expert**：DeepSeekMoE细分expert并添加dedicated shared experts；Union-of-Experts从routing neuron输出构建virtual shared expert。本文区别：通过channel alignment显式识别可复用响应，而非独立设计shared expert模块。
- **Expert Specialization**：Orthogonality/variance objectives减少expert overlap；MP-MoE用inter-expert covariance选择diverse expert sets。本文区别：Gram正则作用于router embeddings而非expert outputs，直接塑造routing geometry。
- **Dynamic Routing**：DynMoE auto-tuning路由；MASS结合cumulative mass与expert expansion；Alloc-MoE预算感知分配。本文区别：expert数量由shared-residual budget决定，而非独立预测。
- **Fine-grained Computation**：Emergent MoE用key centroids暴露modular structure；nested/slimmable experts自适应width。本文区别：将fine-grained selection置于shared modeling之后，形成有序依赖。
- **Sparse Upcycling**：从dense checkpoint训练sparse MoE（Komatsuzaki et al. 2023）。本文以此为基础，利用channel alignment性质进行unified allocation。

## 局限性与未来方向
- **TerraIncognita表现受限**：该数据集受location-dependent backgrounds主导，LFME的explicit domain specialization更强，UniF-MoE在此场景未达最优(52.6% vs 53.4%)。
- **Block granularity敏感**：B=4过粗(quarter FFN变动)，B=32过细(protype覆盖不足)，B=8为稳定平衡点，但未探索更大规模下的最佳B。
- **Router embedding正交约束**：强Gram正则可能抑制有用expert合作，需精细调节$\lambda_{div}$；极端正则可能压制task loss。
- **仅验证小规模模型**：实验基于DeiT-S/16和BERT-large，未扩展至LLaMA等大规模语言模型验证泛化性。
- **共享与残差边界**：固定block划分可能非最优，channel-level精细化分配或进一步提升效率。

## 研究启发与可借鉴点
- **有序分配原则**：将多项MoE自适应决策（共享、细粒度、动态）按因果依赖顺序编排，而非并行独立控制，可避免重复计算与容量错配。
- **Channel-level可复用性分析**：利用sparse upcycling下expert初始同源性，通过value channel alignment识别共享响应，为expert decomposition提供量化依据。
- **Cumulative routing mass**：用排序affinity的prefix sum匹配需求阈值，替代固定top-k或独立预测expert数量，实现budget-consistent动态路由。
- **Gram regularization for router**：正交约束塑造简洁routing geometry，减少expert overlap同时保留合作空间，可迁移至其他routing-based架构。
- **Patch/token-level resource allocation**：共享block跨element复用，residual expert选择性激活，为视觉多尺度处理提供新视角。

## 关键术语表
**Mixture-of-Experts (MoE)**：通过router将token分发到少量expert FFN，实现参数扩容与稀疏计算。
**Sparse Upcycling**：从预训练dense FFN初始化多个expert，通过稀疏化训练扩展模型容量。
**Key-Value Channels**：FFN中间层每个hidden position由up-projection key（激活条件）和down-projection value（输出方向）定义。
**Shared-Residual Decomposition**：将expert输出拆分为可复用shared部分与token-specific residual部分。
**Cumulative Routing Mass**：按affinity排序后prefix sum，用于确定满足残差需求的expert数量。
**Gram Regularization**：对router embedding矩阵施加正交约束，促进routing方向多样性。
**Token-Adaptive Computation**：根据每个token的语义需求动态调整shared width与expert count。
**Blockwise Partitioning**：将expert中间宽度划分为B个aligned blocks，支持细粒度shared/residual分配。

## 可复现要素
- **数据集**：DomainBed（PACS, VLCS, OfficeHome, TerraIncognita, DomainNet）、GLUE（CoLA, MRPC, QNLI, MNLI, RTE）——均为公开数据集。
- **代码**：已开源，https://github.com/existence0420/UniF-MoE。
- **关键超参**：Vision: K=6, B=8, d=384, H=1536, Transformer layers 8/10转换；Language: K=16, B=16, 转换layers 20/22；$\lambda_{div}=0.01$；dropout=0.1, stoch-depth=0.1；学习率$\{2,3,5\}\times10^{-5}$；batch size=32。
- **硬件环境**：NVIDIA GeForce RTX 3090 GPU, AMD EPYC 75F3 CPU, 503 GiB RAM。
- **实现细节**：PyTorch 2.4.1, CUDA 12.1, cuDNN 9.1, FP16 mixed precision, Hugging Face Transformers 4.46.3。
