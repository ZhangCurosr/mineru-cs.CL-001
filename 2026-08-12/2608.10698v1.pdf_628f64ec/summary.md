---
title: "EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection"
source: https://arxiv.org/pdf/2608.10698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:38:42"
field: "AI 生成内容检测"
keywords: ["LLM 生成文本检测", "中文三分类", "编辑程度回归", "冲突感知融合", "零样本检测", "混合集成学习"]
innovations: ["提出编辑程度连续轴建模替代硬分类以缓解 OOD 偏移", "设计冲突感知融合策略差异化整合异构检测信号", "构建九元 LGT 支持投票面板配合高精度规则后处理"]
benchmarks: ["NLPCC 2026 Shared Task 6", "CUDRT 数据集", "DetectRL-X 中文 split"]
---

# 论文速读：EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection

## 一句话总结
本文提出 EVIL-Detect，一个针对中文场景三分类任务（HWT/LGT/HLT）的多信号冲突感知融合检测系统，通过编辑程度回归、零样本似然对比、词汇统计与保守规则协同，在 NLPCC 2026 Shared Task 6 上取得 macro-F1 0.8888 的冠军成绩。

## 研究问题与动机
- **现实需求**：大语言模型广泛使用带来虚假信息、学术不端等风险，可靠检测 LLM 生成文本对负责任的 NLP 至关重要，尤其在中文场景中直接迁移英语方法不可行（分词、词汇粒度、书写惯例差异）。
- **三分类挑战**：NLPCC 2026 Shared Task 6 将任务扩展为人类写作文本（HWT）、LLM 生成文本（LGT）与 LLM 润色文本（HLT）三分类，反映真实 AI 辅助写作场景（用户常利用 LLM 改写/润色人类文本），但现有工作多聚焦英语二分类，对 OOD 分布偏移鲁棒性不足。
- **单一信号局限**：监督分类器（如 QLoRA 微调分类器）在匹配设置下有效，但跨域/跨生成器偏移时性能骤降（如 Generative SFT 在 testp1 仅 macro-F1 0.1690）；无训练方法（如 DetectGPT、Binoculars）对模型选择与阈值敏感，Binoculars 单独作为三分类预测器仅 0.3928 macro-F1。
- **标签结构重新思考**：HLT 是 HWT 与 LGT 之间的中间状态（LLM 润色保留人类内容同时引入机器痕迹），不同检测方法对类别偏好不同，需组合互补信号而非均匀平均。

## 核心贡献（创新点）
1. **提出 EVIL-Detect 多信号冲突感知融合框架**：集成 EditLens 编辑程度回归、Soft-EditLens 语义短语匹配、EchoPrompt 零样本似然对比投票、词汇统计与保守文本规则，通过校准决策边界与冲突解析实现稳健三分类。
2. **设计编辑程度连续轴建模方案**：将 HWT-HLT-LGT 视为连续编辑强度轴（HWT=0, LGT=1, HLT=d(h,t)），训练 LoRA 适配的 decoder-only LLM 进行回归预测，替代硬标签分类以更好捕捉边界案例。
3. **构建九元二元 LGT 支持投票面板**：融合零样本似然对比、编辑程度、词汇倾向等异构信号的 LGT-support 投票，为冲突感知融合提供辅助证据而非独立预测。
4. **实现冲突感知集成与高精度规则修正**：定义基于 Soft-EditLens 回归/桶预测与 LGT 投票数的条件解析规则，并在融合后应用保守 HTML/XML 标记、脚本残迹与改写痕迹检测规则作为最终安全网。
5. **在 NLPCC 2026 Shared Task 6 上取得冠军**：在 testp1 (macro-F1 0.8913) 与 testp2 (macro-F1 0.8888) 均排名第一，较 EditLens 基线提升 4.19 个百分点，显著减少 LGT↔HLT 误判。

## 方法详解
- **整体架构**：EVIL-Detect（Edit-aware View-Integrated Learning for Detection）由监督编辑程度模块、零样本检测模块、词汇频率统计模块与融合决策模块组成，如图 1 所示。
- **监督训练模块（EditLens & Soft-EditLens）**：
  - **目标定义**：对每个训练三元组 $(h, g, t)$（HWT, LGT, HLT），设定连续编辑程度目标 $r(h)=0, r(g)=1, r(t)=d(h,t)$，其中 $d(h,t) \in [0,1]$ 度量 HLT 偏离对应 HWT 源文的程度。
  - **EditLens**：使用表面级软字符 n-gram 距离构建 $d(h,t)$，公式 (2) 计算 $d_{ng}(h,t)=1-\frac{\sum_{n\in N}w_n\cos(C_n(h),C_n(t))}{\sum_{n\in N}w_n}$，通过验证集扫描选择最优 n-gram 阶数与权重配置（rank044：1-7 阶线性递增权重），用 LoRA（rank 16, α 32, dropout 0.05）适配 Qwen3.5-4B-Base 进行回归训练。
  - **Soft-EditLens**：替换为语义短语级软匹配，公式 (3) 计算 $d_{ph}(h,t)=1-\frac{\sum_{q\in P(t)}c_t(q)\max_{p\in P(h)}\cos(e(q),e(p))}{\sum_{q\in P(t)}c_t(q)}$，训练回归模型与有序桶预测器双重输出，捕捉语义保留表达而非仅词汇替换。
- **零样本检测模块（EchoPrompt）**：
  - 基于完全 LLM 生成文本更符合助手风格生成分布、人类文本较少遵循该分布的直觉，使用指令微调模型与基础模型对的提示似然对比分数，通过多个固定 prompt/模型分支（Qwen3.5-4B, Qwen2.5-1.5B）产生 LGT 支持投票而非独立预测。
- **词汇频率统计模块**：
  - 从训练集构建标签条件频率词表，移除跨类相似模式后，计算平滑条件概率 $p(g|y)$，通过平均对数几率聚合偏好，公式 (4)：$s_{a/b}(x)=\frac{1}{|G(x)|}\sum_{g\in G(x)}\log\frac{p(g|a)}{p(g|b)}$，衍生统计量包括 LGT-vs-HWT 倾向、AI-vs-HWT 倾向、机器导向词汇占比等，作为辅助投票输入融合模块。
- **融合与决策模块**：
  - **基础标签生成**：使用两组校准边界 $(\tau_1^{(k)}, \tau_2^{(k)})$，$k\in\{1,2\}$，将 EditLens 分数 $s_E(x)$ 离散化为基础标签 $y^{(k)}(x)$（公式 5）：$s_E\leq\tau_1^{(k)}$ → HWT，$\tau_1^{(k)}<s_E<\tau_2^{(k)}$ → HLT，$s_E\geq\tau_2^{(k)}$ → LGT。
  - **冲突感知集成**：若 $y^{(1)}=y^{(2)}$ 则直接采纳；若不一致则通过冲突解析器 $R(y^{(1)},y^{(2)},z_{soft},V_{LGT})$ 解决（公式 6），利用 Soft-EditLens 回归/桶分数 $z_r,z_b$ 与九元 LGT 支持投票总数 $v(x)=\sum_{i=1}^9v_i(x)$ 的条件阈值（$\gamma_H,\gamma_{HL},\beta_T,\gamma_{TL}$）进行解析。
  - **高精度文本规则**：融合后应用保守规则作为最终修正：①HTML/XML 结构标记（`<!DOCTYPE html>`, `<html>`, `<div>`, `<a href=...>`）强证据指向 LGT；②脚本/渲染残迹（`document.write("<br/>")`、未完成标签序列）触发 LGT 修正；③明确改写/润色痕迹（如 "answer wrappers" 指示重写版本）在非 LGT 标签下支持 HLT。

## 实验与结果
- **数据集**：NLPCC 2026 Shared Task 6 官方数据，训练集 57,589 样本（HWT 19,634 / LGT 18,368 / HLT 19,587），源自 CUDRT 数据集并覆盖 GPT-4、Qwen-family、ChatGLM、Baichuan 四类生成器及新闻/学术写作两领域；testp1（3,600 样本）与 testp2（1,152 样本）均来自 DetectRL-X 中文 split 的 OOD 数据（可能含未见领域、生成器与生成方案）。
- **评估指标**：官方 macro-F1（三类平均 F1），另报告准确率与每类 F1。
- **主要结果**：
  - testp1：macro-F1 0.8913，准确率 0.8911，HWT-F1 0.9083，LGT-F1 0.9267，HLT-F1 0.8391。
  - testp2：macro-F1 0.8888，准确率 0.8880，HWT-F1 0.9039，LGT-F1 0.9219，HLT-F1 0.8407。
  - 两阶段微小差距表明多信号设计在 OOD 分布下稳定。
  - LGT 检测最容易（F1≈0.92），HLT 最具挑战（F1≈0.84），符合其作为中间状态的特性。
- **组件分析**：EditLens（rank044）最强单组件（testp1 macro-F1 0.8494），比初始软 n-gram 配置提升 0.0205；零样本最佳策略 0.6325，词汇统计最佳分类器 0.5535，均作为辅助投票；Full fusion 较 EditLens 提升 4.19 个百分点。
- **消融实验**：EditLens-only 0.8411 → 加冲突感知+规则 0.8816 → 全系统 0.8888；错误减少主要体现在 LGT→HLT 从 59 降至 30，HLT→LGT 从 58 降至 28。
- **替代设计对比**：Generative SFT（GLM-4-9B/Qwen3.5-9B）分别仅 0.1896/0.1690 macro-F1，Binoculars 仅 0.3928，验证编辑程度建模作为基础信号的必要性。

## 相关工作脉络
- **训练型监督检测**：GPT-style detectors [4]、RADAR [5]、DeTeCtive [6] 通过对抗/对比学习提升鲁棒性，但在域/生成器/提示风格/文本长度偏移下脆弱；本文定位差异：采用连续编辑程度回归而非硬分类，减轻 OOD 偏移影响。
- **无训练/统计/表征检测**：DetectGPT [7]、Fast-DetectGPT [8]、DNA-GPT [9]、Binoculars [10] 依赖似然曲率、n-gram 发散、交叉模型 perplexity 或隐式提示恢复；EchoPrompt [25] 作为零样本辅助证据而非独立预测器；RepreGuard [16] 关注隐藏表征模式。本文利用 EchoPrompt 产生投票信号，避免单一似然方法阈值敏感问题。
- **混合集成与中文共享任务**：EnsemJudge [17]（NLPCC 2025 Task 1 冠军）结合多模型投票；CUDRT [12]、DetectRL [18]、DetectRL-X [13] 强调真实 OOD 与 AI 编辑场景。本文相对定位：首次针对 HWT/LGT/HLT 三分类设计冲突感知融合，将异构信号按类别偏好差异化整合。
- **编辑程度建模前作**：EditLens [15] 提出 AI 编辑程度连续估计框架；本文扩展至 Soft-EditLens 语义短语匹配，并双输出（回归+桶）供融合模块使用。
- **任务基准**：NLPCC 2025 Task 1 [11] 推动中文二分类；NLPCC 2026 Task 6 [14] 扩展为三分类；本文直接对标并夺冠该任务。

## 局限性与未来方向
- **规则依赖人工设计**：高精度文本规则（HTML/XML 标记、脚本残迹、改写痕迹）需人工归纳，可能无法覆盖所有伪造或变异模式。
- **零样本模块计算开销**：EchoPrompt 需运行多个 prompt/模型分支产生投票，推理成本高于单一分类器。
- **词汇统计模块特征工程依赖**：n-gram 词表构建与阈值校准需充分训练数据，低资源场景可能受限。
- **三分类边界模糊性**：HLT 作为中间状态 inherently 难以精确界定，即使融合后 HLT F1 仍显著低于 LGT/F1≈0.84 vs 0.92。
- **未来方向**：可扩展至多语言场景、探索动态阈值学习替代人工校准、结合表征级信号（如 RepreGuard）进一步丰富异构证据源、研究端到端融合架构替代规则后处理。

## 研究启发与可借鉴点
- **编辑程度连续轴建模**：将 HWT-LGT 视为连续谱、HLT 为软中间状态的思想可迁移至其他 AI 辅助内容检测任务（如代码生成检测、图像生成检测）。
- **冲突感知融合策略**：针对不同类别对偏好设计差异化解析规则（如 HWT/HLT 用 Soft-EditLens 桶、HLT/LGT 用 LGT 投票数），比均匀加权更贴合信号语义，可复用至多源证据集成。
- **异构信号分工设计**：监督模块提供基础连续性标签、零样本模块提供 LGT 倾向投票、规则模块作高精度安全网，这种分层职责划分避免信号冗余与冲突，值得在检测系统中推广。
- **验证集扫描优化目标构造**：EditLens 的 rank044 配置通过分离度扫描选择 n-gram 阶数与权重，该方法论可用于其他回归目标的超参优化。
- **保守规则后处理**：HTML/XML 结构标记、脚本残迹等表面线索作为高精度修正手段，在文本清洗与内容安全检测中同样适用。

## 关键术语表
- **HWT (Human-Written Text)**：人类写作文本，三分类任务中最基础的参考类别。
- **LGT (LLM-Generated Text)**：完全由大语言模型生成的文本，编辑程度最高的类别。
- **HLT (LLM-Refined Text)**：LLM 润色/改写后的文本，介于 HWT 与 LGT 之间的中间状态。
- **EditLens**：基于连续编辑程度回归的 AI 编辑检测框架，使用字符 n-gram 距离构造软标签。
- **Soft-EditLens**：EditLens 的语义增强变体，采用短语级软匹配替代表面 n-gram 重叠，更好捕捉语义保留表达。
- **EchoPrompt**：零样本提示检测模块，通过指令微调模型与基础模型的似然对比产生 LGT 支持投票。
- **Conflict-aware Fusion**：冲突感知融合策略，根据基础标签对的一致性选择不同的辅助信号解析冲突。
- **Macro-F1**：三类 F1 分数的算术平均，NLPCC 2026 Task 6 的官方评估指标。

## 可复现要素
- **数据集**：NLPCC 2026 Shared Task 6 官方数据（训练集 57,589 样本，testp1 3,600，testp2 1,152），源自 CUDRT [12] 与 DetectRL-X [13] 中文 split；训练集已清理空/损坏样本与任务无关伪影，测试集为官方黑盒。
- **代码**：已开源，GitHub 链接 https://github.com/bbbhrrrr/evildetect（论文声明）。
- **权重**：使用 Qwen3.5-4B-Base、Qwen3.5-9B-Base、Qwen2.5-1.5B，通过 LoRA 适配；具体权重未单独开源。
- **关键超参**：LoRA rank=16, α=32, dropout=0.05；EditLens n-gram 配置 rank044（阶数 1-7，线性递增权重）；融合边界阈值 $(\tau_1^{(k)},\tau_2^{(k)})$、$\gamma_H,\gamma_{HL},\beta_T,\gamma_{TL}$ 经验证集校准；九元投票面板配置见表 3。
- **训练环境**：NVIDIA V100 GPU；推理设备未明确说明。
