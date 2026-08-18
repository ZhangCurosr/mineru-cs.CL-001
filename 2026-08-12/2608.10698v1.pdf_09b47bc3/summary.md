---
title: "EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection"
source: https://arxiv.org/pdf/2608.10698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:37:52"
field: "AI生成文本检测"
keywords: ["LLM生成文本检测", "三分类", "编辑程度回归", "冲突感知融合", "中文文本检测", "NLPCC 2026"]
innovations: ["将HLT建模为HWT-LGT轴上的软编辑连续态并辅以双阈值基类与冲突感知融合", "结合EditLens编辑程度、EchoPrompt零样本、词汇统计与高精度规则的多信号集成框架", "在NLPCC 2026 Shared Task 6中文三分类OOD设置下获macro-F1 0.8888官方第一"]
benchmarks: ["NLPCC 2026 Shared Task 6", "CUDRT", "DetectRL-X"]
---

# 论文速读：EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection

## 一句话总结
本文提出 EVIL-Detect，一个面向 NLPCC 2026 Shared Task 6（中文 HWT/LGT/HLT 三分类）的冲突感知多信号集成检测框架，融合编辑程度回归、零样本似然对比、词汇统计与保守规则，以 macro-F1 0.8888 获得官方评测第一。

## 研究问题与动机
- **现有方法以英文二元检测为主**：训练型监督检测器（如 GPT-style detector、RADAR、DeTeCtive）和免训练统计型方法（DetectGPT、Fast-DetectGPT、Binoculars 等）主要面向英文 HWT vs LGT 二分类，在跨域/跨生成器 OOD 设置下鲁棒性显著下降。
- **中文不能直接移植英文方案**：中文在分词粒度、词汇分布和书写规范上与英文差异显著，英文方法难以直接迁移到中文三分类场景。
- **真实 AI 辅助写作催生三分类需求**：NLPCC 2026 Shared Task 6 要求区分 HWT（人工写作）、LGT（LLM 生成）与 HLT（LLM 润色），更贴近“用户用 LLM 改写/润色原文”的真实场景。
- **直接端到端三分类模型 OOD 失效**：直接用 QLoRA 微调生成式分类器在 testp1 仅获 macro-F1 0.1690，说明需要重新审视标签结构并引入多信号融合。

## 核心贡献（创新点）
- **系统性基准评测**：分析了 EditLens、EchoPrompt、Binoculars、Generative SFT 等代表方法在中文三分类 OOD 设置下的表现，确认单一信号不足以稳健区分 HWT/LGT/HLT。
- **编辑程度连续回归作为主信号**：受 EditLens 启发，将 HLT 建模为 HWT–LGT 轴上的实例依赖软编辑状态而非独立硬类，通过字符 n-gram（EditLens）和短语级语义匹配（Soft-EditLens）两种回归目标刻画编辑强度。
- **冲突感知融合策略**：以双阈值 EditLens 基类标签为基础，结合 Soft-EditLens 回归/分桶信号与 9 项 LGT-support 投票，针对 HWT/LGT、HWT/HLT、HLT/LGT 三类冲突分别设计校准阈值解析规则，避免简单平均。
- **高精度文本规则兜底**：在融合后追加 HTML/XML 结构标记、脚本残留等强 LGT 启发式规则以及明确的“润色后文本”标记规则，作为高精确率修正层。
- **NLPCC 2026 Task 6 官方第一**：在 testp1 macro-F1 0.8913、testp2 macro-F1 0.8888，较单模型 EditLens 提升约 4 个百分点。

## 方法详解
- **标签结构重定义**：在训练三元组 (h, g, t) 上定义连续编辑程度 r∈[0,1]，r(h)=0、r(g)=1、r(t)=d(h,t)，其中 d(h,t) 度量 HLT 相对 HWT 的偏离程度，HLT 被视作软中间态。
- **EditLens（字符 n-gram 版）**：计算 HWT 与 HLT 的字符 n-gram 余弦相似度加权平均，得到 d_ng(h,t)，经验证集分离度扫参选出 rank044（n=1–7、线性递增权重）作为训练目标；基于 Qwen3.5-4B-Base + LoRA（rank 16, α 32, dropout 0.05）做回归预测，输出连续编辑分数 s_E(x)。
- **Soft-EditLens（短语级语义版）**：将 HWT/HLT 切分为短 token 短语，对每个 HLT 短语 q 与其最相似 HWT 短语 p 计算 max cos(e(q),e(p)) 加权距离 d_ph(h,t)；同时训练回归头与有序分桶头，输出 z_r(x) 和 z_b(x)，二者作为 HWT/HLT 边界修正信号。
- **EchoPrompt 零样本模块**：利用指令微调模型与对应 base 模型的 likelihood contrast 衡量文本与助手风格生成的兼容度；采用 Qwen3.5-4B 和 Qwen2.5-1.5B 等多 prompt/model 分支，输出 9 项二元 LGT-support 投票 v_i(x)。
- **词汇频率统计模块**：在训练集构建标签条件 n-gram lexicon，计算 log-odds 倾向 s_{a/b}(x)，派生 LGT-vs-HWT、AI-vs-HWT 等统计量作为辅助证据。
- **冲突感知融合**：用两组校准边界 (τ_1^(k), τ_2^(k)) 将 s_E(x) 离散为基标签 y^(k)，一致则直接采纳；不一致时按冲突类型调用 R(·)：HWT/LGT 冲突看 z_r 与 v(x) 阈值，HWT/HLT 冲突看 z_b≥β_T，HLT/LGT 冲突看 v(x)≥γ_TL。
- **高精度规则层**：融合后若命中 `<!DOCTYPE html>`/`<html>`/`<div>`/`<a href=...>` 等结构标记或 `document.write("<br/>")` 等渲染残留则强制 LGT；若出现明确"润色/改写后文本"包装语且非 LGT 则修正为 HLT。

## 实验与结果
- **数据集**：训练集来自 CUDRT，涵盖 GPT-4、Qwen 系列、ChatGLM、Baichuan 四个生成器，新闻与学术写作两个域，共 57,589 条（HWT 19,634 / LGT 18,368 / HLT 19,587）；testp1 与 testp2 来自 DetectRL-X 中文 OOD 切分，分别为 3,600 与 1,152 条。
- **官方指标**：三类 macro-F1。
- **主要结果**：testp1 macro-F1 0.8913（HWT 0.9083 / LGT 0.9267 / HLT 0.8391），testp2 macro-F1 0.8888（HWT 0.9039 / LGT 0.9219 / HLT 0.8407）；两 phase 差距小，表明多信号设计跨分布稳定。LGT 最高、HLT 最难，符合“中间类更难”直觉。
- **最强结果与提升**：EVIL-Detect 较 EditLens rank044 单模块（testp1 0.8494）提升 +4.19 points；较 QLoRA 生成式 SFT（0.1690）提升超过 0.72。Ablation 显示 testp2 上由双阈值融合与规则层贡献约 +0.0477（0.8411→0.8888），且 HLT 混淆 error 显著下降（LGT→HLT 59→30、HLT→LGT 58→28）。
- **替代方案失效**：Binoculars 单模块 0.3928；GLM-4-9B/Qwen3.5-9B 生成式 SFT 仅 0.1896/0.1690，验证单信号/端到端三分类路径在 OOD 下的脆弱性。

## 相关工作脉络
- **训练型监督检测器（GPT-style detector、RADAR、DeTeCtive）**：在匹配分布上表现强，但对域/生成器/长度/提示风格的 OOD 转移敏感，本文将其作为背景对照而非直接采用。
- **免训练概率/扰动型检测器（DetectGPT、Fast-DetectGPT、DNA-GPT、Binoculars、EchoPrompt）**：依赖模型选择与阈值，本文复用 EchoPrompt 作为 LGT 倾向辅助投票而非主判据，避免单模型阈值漂移风险。
- **编辑程度建模（EditLens、RepreGuard）**：EditLens 以连续编辑强度为核心信号是本文主干；RepreGuard 从隐层表示模式检测，本文未直接采用但同为非纯分类思路的代表。
- **中文已有共享任务（NLPCC 2025 Task 1、CUDRT、DetectRL-X）**：Task 1 聚焦中文二元检测；CUDRT/DetectRL-X 强调现实/OOD 场景；本文承接 NLPCC 2026 Task 6 的三分类设定并扩展至融合框架。
- **集成检测（EnsemJudge）**：2025 任务冠军方案采用多模型投票；本文定位差异在于引入“冲突感知+双阈值基类+规则兜底”的分层融合，而非统一 voting。
- **AI 润色/改写场景检测（DetectRL、DetectRL-X）**：强调半自动改写痕迹；本文将 HLT 显式建模为软编辑连续态，并在融合层专门处理 HWT/HLT 与 HLT/LGT 两类边界冲突。

## 局限性与未来方向
- **规则层依赖人工经验与训练集先验**：HTML/XML 残留与"润色包装语"规则在更泛化的 OOD 或新型生成 pipeline 下可能失效，泛化性受限。
- **多模块推理成本较高**：同时运行双 LoRA 微调模型（4B/9B）、多分支 EchoPrompt 与词汇统计，离线部署与低延迟场景受限。
- **Lexical 与 EchoPrompt 作为辅助投票时信息利用率不充分**：当前仅以离散 vote 或简单统计汇入融合，未做连续校准或置信度加权。
- **跨语言外推未验证**：方法在中文训练/测试分布上有效，但向其他非英文语言迁移（如日文、韩文、多语种混合）的效果未作评估。
- **未深入探索表征层信号**：与 RepreGuard 等隐层方法未作联合实验，未来可把表示信号并入融合层进一步提升 HLT 边界精度。

## 研究启发与可借鉴点
- **连续编辑程度回归优于硬三分类**：将 HLT 视为 HWT–LGT 轴上的软中间态，比直接训练三分类头更能缓解边界模糊问题，可迁移到任意“人机混合写作”检测任务。
- **冲突感知融合的设计范式**：以强基信号定基类、再以辅助信号按冲突类型差异化解析，避免等权投票带来的噪声扩散；这一分层融合逻辑可复用到其他多源异构检测任务。
- **预部署高精度规则兜底有效且成本低**：在融合决策后追加少量高真阳性规则，可在几乎不牺牲召回的前提下提升精确率，适合作为线上系统的最后防线。
- **验证集分离度扫参选目标配置**：编辑程度目标的构建（n-gram 阶数、权重、短语匹配策略）通过验证集分离度排序选取，这一数据驱动目标构造流程可推广至其他回归型检测信号。
- **团队创新机会**：可将 Soft-EditLens 的短语语义匹配推广到句级依存/语义角色一致性度量，或将 Lexical log-odds 改为连续置信度后与 EchoPrompt 联合校准，形成多模态（文本+统计+概率）融合的新基线。

## 关键术语表
- **HWT / LGT / HLT**：Human-Written Text（人工写作）、LLM-Generated Text（LLM 生成）、LLM-Refined Text（LLM 润色），本文三类检测目标。
- **EditLens**：基于字符 n-gram 距离估计 AI 编辑程度的连续回归检测框架，本文取其思想并扩展。
- **Soft-EditLens**：以短语级语义匹配替代表面 n-gram 重合度、并引入有序分桶预测的编辑程度变体。
- **EchoPrompt**：通过指令微调模型与 base 模型的似然对比评估文本与助手风格生成的兼容性的零样本检测模块。
- **macro-F1**：三类 F1 的算术平均，本文官方主指标，对少数类（如 HLT）给予等权。
- **OOD（Out-of-Distribution）**：指测试集在生成器/域/提示风格等分布上与训练集存在差异，本文 task 的核心难点来源。
- **QLoRA / LoRA**：低秩微调技术；本文使用 LoRA 对 Qwen3.5-Base 做编辑程度回归微调。
- **冲突感知融合（Conflict-aware Fusion）**：当双阈值基类标签不一致时，按具体冲突类型调用不同辅助信号进行解析的决策机制。

## 可复现要素
- **数据集**：训练集来自 CUDRT（已清洗）；testp1/testp2 来自 NLPCC 2026 Shared Task 6 官方，源自 DetectRL-X 中文切分（官方提供）。
- **代码**：已开源，GitHub https://github.com/bbbhrrrr/evildetect。
- **模型与配置**：EditLens 基于 Qwen3.5-4B-Base + LoRA（rank 16, α 32, dropout 0.05）；Soft-EditLens 基于 Qwen3.5-9B-Base 同配置；EchoPrompt 使用 Qwen3.5-4B 与 Qwen2.5-1.5B instruct/base 多分支。
- **关键超参**：EditLens 目标采用 rank044（字符 n-gram n=1–7、线性递增权重）；融合使用两组校准阈值 (τ_1^k, τ_2^k)；Lexical log-odds 平滑与分桶边界均经验证集校准（论文未给出具体数值）。
- **训练环境**：NVIDIA V100 GPU；论文未提及详细 batch size、学习率、epoch 等训练细节。
