---
title: "DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?"
source: https://arxiv.org/pdf/2608.10366v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:05:46"
field: "多模态智能体与计算机交互"
keywords: ["Data Science Agent", "GUI Agent", "Benchmark", "End-to-End Workflow", "OS-Level Evaluation", "Vision-Language Model"]
innovations: ["首个在真实OS中评估端到端数据科学工作流的基准DSAgentBench，覆盖275个多阶段任务", "扩展OSWorld集成Jupyter/VS Code/Kaggle/OpenML/SQLite，支持跨工具协同与确定性产出验证", "提出Screenshot与Screenshot+A11y Tree双模态评测协议并揭示接地/编排/推理三重瓶颈"]
benchmarks: ["DSAgentBench", "OSWorld", "DA-CODE", "DS-1000", "DABStep", "MLAgentBench"]
---

# 论文速读：DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?

## 一句话总结
本文提出 **DSAgentBench**，首个在真实操作系统环境中评估智能体能否自动化端到端数据科学工作流的基准测试，包含275个覆盖数据科学全生命周期的真实任务；实验表明，即使最强闭源智能体 Claude-4.6-Sonnet 仅达 56.70% 成功率，开源智能体均低于 1%，揭示出现有 Agent 系统在 OS 接地、工具编排与长程分析推理上的重大能力缺口。

## 研究问题与动机
- **核心问题**：当前大模型智能体能否在真实计算机环境中自主完成从数据获取到模型验证的端到端数据科学工作流？
- **现有基准不足一**：DS-1000、DABStep、MLAgentBench、DSBench、DSEval 等仅评估孤立代码生成或静态执行，不要求启动应用、导航文件系统或跨工具协调。
- **现有基准不足二**：DA-CODE 虽接近真实工作流，但仍局限于沙箱 Notebook 环境，缺乏 OS 级访问与跨应用协作能力。
- **OSWorld 类基准局限**：OSWorld、WebArena、VisualWebArena 等评估通用桌面/Web 交互，未针对数据科学中多阶段推理、分析正确性与可视化的端到端评估需求设计。

## 核心贡献（创新点）
1. **DSAgentBench 基准提出**：首次构建在真实操作系统内运行、覆盖完整数据科学生命周期（数据获取→EDA→特征工程→建模→评估→可视化）的端到端工作流评测平台，与仅评估代码片段的主流基准形成本质差异。
2. **基于 OSWorld 的数据科学环境扩展**：在 OSWorld 基础上集成 Jupyter Notebook、VS Code、Chrome、Kaggle API、OpenML 与 SQLite，使智能体能在真实多工具生态中执行分析任务，而非静态代码沙箱。
3. **确定性+语义化评估体系**：每个任务配备可执行 Python 评估器，验证输出文件、数值精度（如 Pearson 相关系数容差 ε=0.01）、可视化语义（轴/标题/图例）与模型性能阈值（如 accuracy/F1≥0.7），并辅以 Gemini-2.5-Pro/GPT-4o 视觉评审，超越“代码是否跑通”的表层判断。
4. **大规模人工 authored 任务集与双标注验证**：275 个由两名资深数据科学家独立设计、LLM 仅用于措辞润色与边界case挖掘的任务，经双标注员独立执行验证，初始一致率达 86%，全部任务最终实现可复现执行。

## 方法详解
- **问题形式化**：任务集 $\mathcal{T} = \{(\mathcal{C}_i, \mathcal{T}_i, \mathcal{V}_i)\}_{i=1}^N$，其中 $\mathcal{C}_i$ 为系统初始配置（数据集、文件系统、已安装库与应用），$\mathcal{T}_i$ 为自然语言指令，$\mathcal{V}_i$ 为确定性 Python 评估器；成功当且仅当 $\mathcal{V}_i$ 输出 ≥0.95。
- **环境架构**：基于 OSWorld 扩展，运行 Ubuntu 22.04 LTS，分辨率 1920×1080；预装 VS Code、Jupyter Notebook、Chrome、Python 及常用科学计算库；支持 Kaggle API、OpenML、直接 URL、SQLite 等多种数据获取途径。
- **观察空间**：① Screenshot-only：像素级屏幕截图；② Screenshot + A11y Tree：截图叠加 AT-SPI 提取的结构化 UI 元数据（元素 role、accessible name、bounding box、交互状态）。
- **动作空间**：统一基于 PyAutoGUI，包含鼠标（CLICK、DRAG、SCROLL 等）、键盘（TYPING、HOTKEY、PRESS 等）与控制动作（WAIT、DONE、FAIL）；所有坐标归一化至 1920×1080。
- **执行流程**：任务初始化后进入感知-动作循环，每步最大交互预算 15 步（消融中测试 30/50 步），每步后等待 2s 保证窗口渲染稳定，总任务超时 1800s；轨迹全程日志化。
- **评估设计**：数值型任务采用确定性校验（加载产出文件、容差匹配）；可视化任务先通过规则检查文件存在性、标签/图例/轴命名与数据映射，再调用 LLM 视觉评委评估语义对齐与图表清晰度；分类任务验证模型性能是否达标。

## 实验与结果
- **评测模型**：15 个闭源/开源 VLM Agent，包括 GPT-4o、GPT-5-mini、GPT-5、O4-mini、Claude-4/4.5/4.6-Sonnet、Gemini-2.5-Pro、OpenAI CUA，以及 Jedi-3B/7B、UI-TARS-2B/7B、GUI-OWL-7B、OpenCUA-72B。
- **人类基线**：3 名具备应用科学/硕士背景的研究者达成 85.09% 整体成功率。
- **核心结果（Screenshot + A11y Tree）**：Claude-4.6-Sonnet 最佳，总体 56.70%；GPT-5 为 29.81%；其余闭源模型约 15–25%；所有开源模型在 Screenshot-only 下均 <1%。
- **分阶段表现**：EDA 任务相对容易（Claude-4.6-Sonnet 达 64.88%），数据获取（DA）与评估（Eval）最难（最高仅 47.82%/66.67%）；Jupyter 任务显著优于 VS Code 任务（55.77% vs. 56.92% 单项对比需参考表 4）。
- **消融结论**：① 步数从 15 增至 50 仅提升 24.54%→25.81%，说明瓶颈非步数预算而是接地/规划/推理/工具编排综合能力；② Terminal-first 提示对 GPT-4o 仅带来 19.34%→20.73% 微增；③ A11y Tree 整体有正向收益但模型利用不均。
- **错误根因**：开源模型 97–98% 失败源于 UI 接地错误；强闭源模型错误分布更广（接地 32–57%、终端/代码 13–43%、逻辑 1–14%）；CUA 与 GUI-OWL-7B 多为晚期失败（>93% 在 13 步后），反映无效探索；Gemini-2.5-Pro 成功步数最短（均值 6.76 步），体现效率与早期终止的权衡。

## 相关工作脉络
- **DS-1000 / DABStep / MLAgentBench / DSBench / DSEval**：聚焦代码生成或孤立分析步骤，静态执行评估，无真实 OS/跨工具交互；DSAgentBench 定位填补"真实环境端到端工作流”空白（Table 1 对比维度最全）。
- **DA-CODE**：引入多文件分析与任务规划，但仍局限于沙箱 Notebook，无 OS 级文件导航与多应用协同；本文扩展至完整桌面环境。
- **OSWorld / WebArena / VisualWebArena**：评估通用桌面/Web 操控能力，侧重界面导航与应用使用，未涉及数据分析推理、统计验证与模型训练闭环；本文将其引入数据科学领域并设计专用评估器。
- **ChartQA / Text2Vis / VisEval**：仅评估图表生成或视觉问答，不涵盖数据获取、清洗、特征工程与建模；本文可视化任务嵌入完整分析链条。
- **KRAMABench / ARCADE**：部分支持多阶段 pipeline 但在静态/受限环境中运行；本文强调真实运行时依赖管理、错误调试与中间结果迭代利用。
- **GUI Agent 基础模型（CogAgent / ShowUI / Ferret-UI / UI-TARS / GUI-G2 / GUI-R1）**：提供底层 UI 接地与动作能力，但本文显示即便结合最强 grounding，开源模型在数据科学复杂长程任务上仍几乎为零，凸显领域推理与工具编排仍是独立瓶颈。

## 局限性与未来方向
- **开源模型缺乏 A11y 支持**：当前开源 Agent 无法使用 Accessibility Tree，评测仅在 Screenshot-only 下进行，可能低估其潜力；未来需推动开源模型接入 A11y 输入。
- **错误分析样本有限**：人工标注仅覆盖 604 条闭源轨迹与 150 条开源轨迹，罕见失败模式未被充分捕获；需更大规模自动化错误分类与根因挖掘。
- **可视化评估仍以最终产物为主**：未深入测量分析过程中的迭代调试、交互式探查与可视化修正能力，后续可引入过程级评估指标。
- **任务难度分布偏难**：Hard 任务占 47.6%，Medium 46.9%，仅 5.5% Easy，对初步能力诊断可能过于严苛；可补充阶梯式难度集以细粒度定位模型能力边界。
- **环境可扩展性待验证**：当前仅基于 Ubuntu/OSWorld，未来需验证在 Windows/macOS 及 R/RStudio、PyCharm、云环境中的移植稳定性与评估一致性。

## 研究启发与可借鉴点
- **确定性评估器设计范式**：将"代码可执行"升级为"产出物可验证"（数值容差+文件结构+语义标签+模型阈值），可直接迁移至代码生成、Agent 工具调用、自动化实验等需要客观结果校验的场景。
- **Screenshot + A11y 双模态评测协议**：并行列两种观察设置以量化结构化 UI 元数据对接地与推理的帮助，为后续 GUI Agent 评测提供标准对照实验设计。
- **Human-in-the-loop 任务构建流程**：专家主导任务逻辑与评估器、LLM 仅辅助措辞与边界 case 发现、双标注员独立执行验证，该 pipeline 可复用于构建高质量 Agent 评测基准。
- **步数预算消融揭示真实瓶颈**：证明增加交互长度并不能显著提升成功率，提示研究重点应从"更长 horizon"转向"更强单步规划与错误恢复"，对 Agent 系统设计有明确指导意义。
- **Terminal-first 提示消融价值**：验证单纯引导优先使用 CLI 不能大幅缓解困难，提示多模态 GUI 控制与工具链协同仍是必要能力，避免未来研究陷入单一交互范式的过度优化。

## 关键术语表
- **DSAgentBench**：首个在真实操作系统中评估智能体自动化端到端数据科学工作流能力的基准测试，包含 275 个多阶段任务与确定性评估器。
- **A11y Tree（Accessibility Tree）**：通过 AT-SPI 从桌面 UI 提取的结构化元数据，包含元素角色、名称、边界框与状态，用于增强视觉接地精度。
- **Deterministic Evaluator**：不依赖 LLM 的 Python 验证脚本，用于检查输出文件存在性、数值容差、可视化语义标签与模型性能阈值。
- **Long-horizon Workflow**：跨越数据获取、清洗、EDA、特征工程、建模、评估与可视化的多步骤、多工具协作分析链条，通常需 4–5 步以上。
- **Grounding Error**：智能体在 GUI 环境中未能正确识别/定位界面元素或误解当前屏幕状态导致的操作失误，是开源模型主要失败原因。
- **Tool Orchestration**：在多应用（终端、IDE、浏览器、数据库）间协调切换、依赖管理与跨进程数据传递的能力，是端到端数据科学的核心挑战。

## 可复现要素
- **数据集**：275 个任务配置与数据集，来自 GitHub、Kaggle、OpenML、SQLite 等公开源，均已预下载并将随基准开源（https://github.com/vis-nlp/DSAgentBench）。
- **代码/权重**：完整实验仓库将开源，包含提示模板、环境配置、评估脚本与推理脚本；模型本身为闭源 API 或开源权重（UI-TARS、GUI-OWL、OpenCUA、Jedi 等）。
- **关键超参**：分辨率 1920×1080；温度 0.1；top-p 0.9；最大输出 tokens 2000；每步延迟 2s；任务超时 1800s；默认最大步数 15（消融 30/50）；评分阈值 ≥0.95 判定为成功。
- **开源模型评测环境**：GCP n1-standard-4 + Docker + vLLM；闭源模型通过 API 在 VMware Ubuntu VM 上评测。
