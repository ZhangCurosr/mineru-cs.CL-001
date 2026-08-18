---
title: "RadFusion: Towards Threshold-Controllable Radiology Report Generation"
source: https://arxiv.org/pdf/2608.10505v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:58:50"
field: "医学影像报告生成"
keywords: ["radiology report generation", "threshold controllability", "ROC analysis", "medical VQA", "confidence calibration", "controllable text generation", "clinical AI"]
innovations: ["首个实现阈值可控放射学报告生成的框架，使报告ROC曲线紧密跟踪分类器ROC", "分类器-生成器融合架构，在匹配特异性时敏感性提升6.9%、匹配敏感性时特异性提升20.7%", "可扩展至三维阈值控制，支持阴性/阳性/半阳性三区间及时间敏感性分级"]
benchmarks: ["MIMIC-CXR", "CheXpert 13-class AUC-ROC"]
---

# 论文速读：RadFusion: Towards Threshold-Controllable Radiology Report Generation

## 一句话总结
RadFusion 是首个实现**阈值可控放射学报告生成**的框架，通过融合医学图像分类器的置信度评分、VQA 驱动的报告生成器和 LLM 重写器，使生成的报告能够随阈值调节敏感性-特异性权衡，既保持丰富临床描述，又支持基于 ROC 曲线的量化监管验证。

## 研究问题与动机
1. **现有报告生成模型缺乏诊断行为可控性**：当前 VLM 报告生成模型仅输出固定报告，无法像感知模型（如分类器）那样通过阈值调节敏感性与特异性的权衡。
2. **不同临床场景需要不同诊断策略**：急诊分诊需高敏感性以减少漏诊，确认性评估需高特异性以减少不必要的干预，单一固定报告无法满足多样化需求。
3. **监管审批需要 ROC-based 量化验证**：FDA 等监管机构期望通过 ROC 曲线评估诊断性能，而现有生成模型无法提供可追溯的数值置信度以支持此类验证。
4. **生成模型与感知模型能力互补但未融合**：分类器能给出精确置信度但缺乏临床细节描述，报告生成器能生成丰富文本但缺乏可解释的置信度机制，两者尚未有效结合。

## 核心贡献（创新点）
1. **首个阈值可控报告生成框架**：将报告诊断内容与分类器 ROC 特性对齐，通过调节阈值 τ 实现敏感性-特异性权衡的可控调节，使报告可适配不同临床场景。
2. **分类器-生成器融合提升诊断准确性**：结合两类模型优势，在匹配特异性时敏感性提升 6.9%，在匹配敏感性时特异性提升 20.7%，显著优于无控制的生成结果。
3. **ROC 合规性验证方法**：提出的闭环评估协议表明，阈值调节报告的 ROC 曲线紧密跟踪底层分类器的 ROC 曲线，AUC 高度一致，使生成报告具备可量化的诊断性能验证能力。
4. **可扩展至三维阈值控制**：将阈值空间扩展为三维 $(T_a, T_b, T_c)$，支持阴性/阳性/半阳性三区间划分及时间敏感性分级，仅需修改重写指令无需重新训练模型。
5. **模块化可移植设计**：框架兼容多种分类器实现（MI2 微调、QRad 线性探测、单token概率）和多种 LLM 重写器，为实际部署提供灵活配置。

## 方法详解

### 整体架构（Figure 1）
RadFusion 由三个核心组件构成：
1. **感知模型（Perception Model）**：输入胸片图像，输出 K 个预定义疾病类别的置信度分数 $\{P(C_i=1)\}_{i=1}^K$，给定阈值 τ 后得到二值预测 $\hat{y}_i = \mathbf{1}[P(C_i=1) > \tau]$。
2. **报告生成模型（Report Generation Model）**：基于 QRad 的 Auto-VQA 架构，先生成问题序列 $Q = f_Q(I)$，再逐问题生成答案构成报告 $Y = f_A(I, Q)$，支持对省略发现的追问。
3. **LLM 重写器（LLM Rewriter）**：接收分类器的二值预测和报告证据池 $\mathcal{E}$（初始报告 + 可选追问答案），重写报告使其诊断内容与阈值化分类一致，同时保留原始临床细节。

### 感知模型三种实现（Section 3.2）
- **Approach 1: MI2 微调分类器（默认）**：基于 MedImageInsight (MI2) 的 Image-Text-Class Hybrid Contrastive Loss：
  $$\mathcal{L} = \mathcal{L}_{\text{image-text}} + \lambda \mathcal{L}_{\text{image-class}}$$
  其中 $\mathcal{L}_{\text{image-text}}$ 基于 softmax cross-entropy，$\mathcal{L}_{\text{image-class}}$ 基于 sigmoid cross-entropy。使用模板 prompt（如 "a chest X-ray showing [CLASS NAME]"）构建类别嵌入，推理时预计算文本编码器权重作为线性分类头。置信度通过 temperature scaling 校准。

- **Approach 2: QRad 编码器线性探测**：在 QRad 冻结图像编码器上训练线性分类头，分类器与报告生成器共享编码器，降低计算成本，在难检测类别（如 Pneumothorax）上表现更优。

- **Approach 3: QRad 单 token 概率**：利用 QRad 的 VQA 能力，提取二元分类问题（"Is this image classified as [CLASS]? (yes/no)"）的 [Yes]/[No] logits softmax 概率作为置信度，但多数类别表现不及独立分类器。

### 报告生成模型：QRad Auto-VQA（Section 3.3）
将报告生成重构为自导向视觉问答过程：
$$Q = f_Q(I), \quad Y = f_A(I, Q)$$
Question Generator $f_Q$ 预测临床相关问题序列，Answer Generator $f_A$ 逐问题生成报告句子。当阈值降低导致某类别翻转为阳性时，系统可主动追问以获取被初始报告省略的发现细节，形成完整的证据池 $\mathcal{E}$。

### LLM 重写机制（Section 3.4）
重写指令包含三部分：任务描述、每类别示例文本模板字典、重写规则。核心规则：
- **阳性类别** $(\hat{y}_i = 1)$：若报告已提及则保留；若缺失或矛盾，从证据池补充或修正。
- **阴性类别** $(\hat{y}_i = 0)$：若报告声称存在则删除或否定；若已否定则无需修改。
- **非类别内容**（影像质量、未涵盖的辅助设备）：保持原样。

示例模板字典将类别名称与多样化自然语言表达关联（如 "Pleural Effusion" 可对应 "small bilateral pleural effusions"、"fluid in the costophrenic angles"、"blunting of the costophrenic recesses"）。

### 三维阈值控制扩展（Section 3.5）
将单阈值扩展为 $(T_a, T_b, T_c)$ 三维空间：
- $T_a \leq T_b$ 将置信度分为三区间：
  - $P < T_a$：阴性区，报告否认该发现
  - $P > T_b$：阳性区，报告肯定该发现
  - $T_a \leq P \leq T_b$：半阳性区，报告描述发现并明确标注不确定性，建议进一步评估
- $T_c$ 阈值化时间紧迫性分数，高于此阈值的发现被标记为时间敏感且可操作。

## 实验与结果

### 数据集与设置
- **MIMIC-CXR**：227,835 项研究，377,110 张胸片，使用官方测试集（2,347 项研究）
- 以 Findings 部分作为生成目标
- 13 个 CheXpert 疾病类别（排除 "No Finding"）

### 评估方法
**闭环评估协议**：分类器阈值化决策 → LLM 重写报告 → 使用 GPT-5 将重写报告转换回二值标签 → 与真实标签比较计算 TPR/FPR。
阈值 τ 从 0.0 到 1.0 以 0.1 为步长扫描，得到 11 个操作点，连接成 ROC 曲线。

### 主要结果

**分类器实现对比（Table 1）**：
| 分类器 | AVG AUC-ROC | 强项类别 | 弱项类别 |
|--------|-------------|----------|----------|
| MI2 FT | 0.90 | Support Devices (.95), Cardiomegaly (.93) | Lung Lesion (.85) |
| QRad Linear Prob | 0.91 | Pneumothorax (.95), Lung Lesion (.89) | Support Devices (.87) |
| QRad Token Logit | 0.73 | — | 多数类别显著落后 |

**LLM 重写器对比（Table 2）**：
- GPT-5 默认配置达到 AVG AUC-ROC 0.90
- 移除类别示例文本导致 Fracture 从 0.89 降至 0.73，AVG 从 0.90 降至 0.88
- GPT-5.4 与 GPT-5 表现相当（0.90 vs 0.90），新模型不一定优于旧模型
- GPT-5.4-mini 默认配置仅 0.76，提升至 medium reasoning 后达 0.89

**跨模型提供商对比（Table 3，使用 QRad Linear Prob 分类器）**：
- GPT-5 default: 0.90
- GPT-5.4 medium: 0.89
- DS-V4-Pro default: 0.87

**核心性能提升（Section 4.3）**：
- **敏感性与特异性改善**：在匹配特异性时敏感性提升 **6.9%**，在匹配敏感性时特异性提升 **20.7%**
- **ROC 一致性**：阈值控制报告的 ROC 曲线紧密跟踪底层分类器 ROC，AUC 高度吻合，验证了阈值的可控性
- **非控制报告性能**：单次固定报告（绿色十字）在所有类别上均低于 ROC 曲线

## 相关工作脉络

1. **放射学报告生成**：从 R2Gen、R2GenCMN 等编码器-解码器架构到 LLaVA-Rad、MAIRA、CheXagent 等大视觉语言模型，本研究延续这一趋势但首次引入阈值可控性；QRad 作为本文生成器基础，统一了报告生成与医学 VQA 任务。

2. **医学视觉问答（Medical VQA）**：Med-Flamingo、LLaVA-Med 等多模态模型扩展至医学领域；Rad-ReStruct、RaDialog 探索结构化报告与多轮对话；本文利用 Auto-VQA 能力补充被初始报告省略的发现，区别于传统一次性生成范式。

3. **可控文本生成**：PPLM、FUDGE、Classifier-free Guidance 等方法在 token 级别引导生成属性，但无法实现基于疾病类别的阈值化诊断控制，本文在控制粒度上更为精细和临床导向。

4. **置信度估计与校准**：P(True) logit 方法（Kadavath et al., 2022）、verbalized confidence（Xiong et al., 2024）、self-consistency（Wang et al., 2023）等集中在 LLM 自身置信度表达；temperature scaling 是标准校准手段；本文结合外部分类器置信度与生成文本，形成"数值+文本"双重置信表达。

5. **诊断测试监管验证**：FDA 审批依赖 ROC 分析（Food, 2007）；本文使生成报告首次可通过标准 ROC 协议评估，弥合了生成式 AI 与监管要求之间的鸿沟。

## 局限性与未来方向

1. **组件误差传播**：分类器错误会传递至重写报告；报告生成器可能遗漏发现；LLM 重写器可能引入细微的语言 artifacts，三者质量共同决定最终性能。
2. **指令模板依赖**：重写指令为模板化设计且针对胸片 findings 调优，在不同 LLM、成像模态或临床域上需重新调优（消融显示效果因模型而异）。
3. **预定义类别限制**：阈值控制仅作用于 14 个 CheXpert 预定义疾病类别，类别外的发现不受阈值调节影响。
4. **自动标签提取误差**：评估依赖 GPT-5 从报告中提取诊断标签，该步骤本身可能引入误差。
5. **辅助而非替代**：作者强调系统旨在辅助而非取代放射科医生审核，临床部署仍需人机协作验证。

未来方向包括：扩展三维阈值控制的临床 UI 验证（Figure 3 已展示原型）、探索更多分类器-生成器组合、将框架迁移至其他医学影像模态或文本生成领域（如自主系统、内容审核、法律合规）。

## 研究启发与可借鉴点

1. **感知-生成融合范式**：将分类器的数值置信度与生成器的描述能力结合，既保留可解释的 ROC 验证能力，又提供丰富的临床细节，可作为其它生成任务的参考架构。

2. **模板字典增强指令跟随**：为 LLM 提供每类别的正负示例文本模板，显著改善重写忠实度（移除示例导致 Fracture AUC 下降 0.16），验证了结构化先验知识对指令遵循的重要性。

3. **闭环评估协议设计**：从阈值化决策→重写报告→标签提取→ROC 绘制的闭环验证方法，为生成模型的诊断性能评估提供了可复用的标准化框架。

4. **Auto-VQA 补充机制**：利用生成模型的追问能力补全被初始报告省略的发现，解决了生成式报告常见遗漏问题，此机制可推广至其它需要完整性保证的文本生成任务。

5. **多维阈值可扩展性**：三维阈值控制设计仅需修改指令无需重新训练，展示了"指令驱动行为调节"的灵活性，为后续开发更细粒度的临床决策支持界面提供设计思路。

## 关键术语表
**Threshold-controllable**：通过调节置信度阈值来控制系统输出的敏感性-特异性权衡，使生成内容可适配不同风险偏好的临床场景。
**ROC curve（Receiver Operating Characteristic）**：绘制真阳性率（敏感性）与假阳性率（1-特异性）关系的曲线，AUC 反映模型整体判别能力，是医疗器械监管审批的标准评估工具。
**Sensitivity-specificity trade-off**：敏感性与特异性之间的此消彼长关系，低阈值提高敏感性但降低特异性，高阈值反之。
**Auto-VQA（Auto-directed Visual Question Answering）**：模型自主生成临床相关问题并逐一回答以构建完整报告的方法，支持对初始遗漏发现的主动追问。
**Temperature scaling**：后处理校准方法，通过学习单一标量温度参数最小化 ECE（Expected Calibration Error），不改变模型判别性能仅改善置信度校准。
**CheXpert**：大规模胸部 X 光数据集及标注体系，包含 14 个疾病类别的二元标签，本文以此定义疾病类别集合。
**Linear probing**：在预训练模型冻结编码器上训练轻量线性分类头的评估/适配方法，可检验特征表示质量并降低计算成本。
**LLM Rewriting**：利用大语言模型按指令修改生成文本，使其与外部约束（如分类器决策）一致同时保持内容 grounded 的技术。

## 可复现要素
- **数据集**：MIMIC-CXR（公开可用，https://mimic-cxr.lille.inria.fr/）
- **代码/权重**：论文未明确声明开源，QRad 和 MedImageInsight 有相关公开资源引用
- **关键超参**：阈值 τ 步长 0.1（0.0~1.0），共 11 个操作点；温度参数 T 通过验证集 ECE 学习；对比损失权重 λ 未明确说明
