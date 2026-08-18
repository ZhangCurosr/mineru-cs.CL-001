---
title: "DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?"
source: https://arxiv.org/pdf/2608.10366v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:07:05"
field: "数据科学Agent评测"
keywords: ["Data Science Agent", "GUI Agent", "Benchmark", "OS Interaction", "Multi-tool Workflow", "Vision-Language Model"]
innovations: ["首个在真实操作系统中评估端到端数据科学工作流自动化的基准（275任务/六阶段生命周期）", "确定性执行评测+LLM视觉裁判的分层评估机制，超越纯代码正确性检查"]
benchmarks: ["DSAgentBench"]
---

# 论文速读：DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?

## 一句话总结
论文提出 **DSAgentBench**，首个在真实操作系统环境中评估 AI Agent 能否自主完成端到端数据科学工作流的基准测试，包含 275 个覆盖数据采集→探索分析→建模→可视化的长周期任务；实测显示最强模型 Claude-4.6-Sonnet 仅达 56.70% 成功率，开源模型均低于 1%，暴露了当前 Agent 在 OS 接地、工具编排和长程推理方面的显著能力缺口。

## 研究问题与动机
- **现有基准的局限**：DS-1000、DABStep、MLAgentBench 等数据科学基准仅评估孤立代码执行的正确性，不要求 Agent 真正操作系统环境（文件导航、依赖管理、跨工具协调）。
- **缺乏端到端工作流评测**：OSWorld、WebArena 等 GUI Agent 基准关注通用桌面操作，而非数据分析推理，无法检验 Agent 作为"数据科学家"的能力。
- **工业报告印证差距**：OpenAI（2024）报告指出当前 AI 系统在多步骤跨工具分析流程中仍不可靠，尽管在单一任务上表现优异。
- **核心问题**：AI Agent 能否在真实计算机环境中通过长周期推理，自主执行涵盖数据采集、清理、建模、可视化的完整数据科学工作流？

## 核心贡献（创新点）
1. **提出 DSAgentBench 基准**：首个在真实操作系统（Ubuntu）内评估端到端数据科学工作流自动化的基准，覆盖完整数据科学生命周期，弥补了现有基准仅评测孤立代码或通用 GUI 交互的空白。
2. **扩展 OSWorld 环境**：在 OSWorld 基础上集成 Jupyter Notebook、VS Code、Chrome 浏览器、Kaggle API、OpenML 和 SQLite 数据库等数据科学核心工具，支持多工具协同的任务配置与自动化数据获取。
3. **设计确定性执行式评测机制**：评测器不仅验证代码是否运行，还通过确定性脚本检查输出文件的数值正确性、可视化语义对齐性和模型性能阈值；同时使用 LLM 视觉裁判辅助评估可视化质量，避免纯代码执行的表面化评估。
4. **系统性实验与误差分析**：评测 15 个闭源和开源 Agent，发现最强模型 Claude-4.6-Sonnet 仅 56.70%，开源模型均低于 1%，并深入剖析了接地失败、工具编排失效和长程推理错误等根因模式。

## 方法详解
- **任务形式化**：每个任务表示为三元组 $(\mathcal{C}_i, \mathcal{T}_i, \mathcal{V}_i)$，其中 $\mathcal{C}_i$ 定义初始系统配置（数据集、文件系统、已安装库），$\mathcal{T}_i$ 为自然语言指令，$\mathcal{V}_i$ 为确定性 Python 评测器。任务成功当且仅当 Agent 输出满足 $\mathcal{V}_i$。
- **环境架构**：基于 OSWorld，运行 Ubuntu 虚拟机（分辨率 1920×1080），预装 Python 及常用科学计算库；观测空间支持两种模态：纯 Screenshot 和 Screenshot + A11y Tree（通过 AT-SPI 获取结构化 UI 元数据）；动作空间为统一的 pyautogui 操作（鼠标点击/拖拽、键盘输入、WAIT/DONE/FAIL 元操作）。
- **任务构成**：275 个任务分六类——数据采集（23）、EDA（119）、特征工程（37）、建模（41）、评估与部署（12）、可视化与报告（33）；难度分布：Hard 47.6%、Medium 46.9%、Easy 5.5%；56.7% 为多阶段任务，平均 4–5 步分析。
- **数据收集流程**：从 Kaggle、OpenML、SQLite、GitHub 等平台获取多样化数据集（95.3% 表格数据）；4 名专家标注者历时 3 个月（约 400 工时）协同设计任务，LLM 仅用于措辞润色；双标注者验证机制确保 86% 初始一致率，全部 275 个任务最终通过互审。
- **评测标准**：主指标为任务成功率（得分 ≥ 0.95 的比例），次指标为平均评分（[0,1] 连续值）；数值型任务在容忍度 ϵ=0.01 内校验；可视化任务结合确定性检查（轴标签、图例、数据映射）与 Gemini-2.5-Pro/GPT-4o 作为视觉裁判。
- **执行约束**：最大步数 15 步（消融实验对比 30/50 步），任务超时 1800 秒，temperature=0.1，top-p=0.9，每次操作后延迟 2 秒等待渲染稳定。

## 实验与结果
- **模型范围**：15 个 Agent，含闭源（GPT-4o、GPT-5-mini、GPT-5、Claude 4/4.5/4.6 Sonnet、Gemini-2.5-Pro、OpenAI CUA）、混合（Jedi-3B/7B w/ GPT-4o）和开源（UI-TARS 2B/7B、GUI-OWL-7B、OpenCUA-72B）。
- **最佳结果**：Claude-4.6-Sonnet（Screenshot + A11y Tree）总体准确率 **56.70%**，各阶段得分：DA 47.82%、EDA 64.88%、FE 56.75%、Model 46.34%、Vis 42.42%、Eval 66.67%。
- **次要结果**：GPT-5 以 29.81% 位居第二；GPT-4o 24.54%，Gemini-2.5-Pro 20.81%，GPT-5-mini 19.03%；其他闭源模型均低于 10%。
- **开源模型**：全部在 Screenshot-only 设置下低于 **1%**，Jedi-7B w/GPT-4o 最高 0.73%，OpenCUA-72B 0.73%。
- **人类基准**：3 名人类受试者（2 名应用科学家+1 名硕士生）达到 **85.09%** 成功率。
- **关键消融**：增加步数预算（15→50）仅使 GPT-4o 从 24.54% 微增至 25.81%，表明瓶颈不在行动步数而在接地、规划和推理能力；Jupyter 任务（平均 55.77%）显著优于 VS Code 任务（54.01%）；多阶段任务成功率远低于单阶段任务。
- **效率对比**：Gemini-2.5-Pro 成功任务平均仅需 6.76 步，效率最高；CUA 需 15 步（全部因预算耗尽失败）。

## 相关工作脉络
1. **DS-1000 / DABStep / MLAgentBench / DSBench / DSEval**：数据科学代码生成基准，仅评测静态执行正确性，不涉及真实操作系统交互和多工具协调；本文在此基础上引入真实 OS 环境和端到端工作流评测。
2. **DA-CODE**：较前作更接近真实流程（支持多文件分析和任务规划），但仍局限于沙盒 notebook 环境，无 OS 级访问；本文进一步开放完整桌面操作能力。
3. **OSWorld**：通用桌面 Agent 基准，支持全 OS 交互但无数据科学任务；本文在其之上扩展数据科学专用工具和评测协议。
4. **WebArena / VisualWebArena / ScreenSpot-Pro**：网页和 GUI 接地基准，侧重界面导航而非分析推理；本文聚焦数据科学这一需要深度领域推理的场景。
5. **ChartQA / Text2Vis / VisEval**：纯可视化基准，仅评估图表生成和视觉问答，不涉及数据获取、清洗和建模；本文的可视化任务嵌入完整分析流水线中。

## 局限性与未来方向
- **开源模型不支持 A11y Tree**：所有开源模型仅在 Screenshot-only 下评测，无法公平对比 A11y 增强的增益，限制了开源生态的评估深度。
- **误差分析样本有限**：手动检查的轨迹仅覆盖 604 条闭源模型和 150 条开源模型运行，稀有失败模式可能未被充分捕捉。
- **可视化评测依赖 LLM 裁判**：虽然仅 ~10% 任务使用 LLM 视觉裁判且经过确定性门控，仍存在裁判偏差风险。
- **环境局限**：仅基于 Ubuntu 测试，未验证 macOS/Windows 下的可移植性（尽管框架设计支持）。
- **未来方向**：改进 Agent 的 OS 接地鲁棒性、开发更强的多工具编排策略、提升长周期状态管理和错误恢复能力。

## 研究启发与可借鉴点
1. **双标注者验证机制**：任务创建者+独立验证者的双重审核流程（86% 初始一致率，迭代修订后 100% 通过）可作为高质量基准构建的方法论范本，值得复用到其他 Agent 评测场景。
2. **确定性评测 + LLM 裁判的分层设计**：先通过确定性脚本校验数值/文件存在性，仅在视觉质量等难以规则化的维度引入 LLM 裁判（且交叉使用避免自评分），兼顾稳定性和语义深度。
3. **A11y Tree 与 Screenshot 的对比实验设计**：同时评估两种观测模态，量化结构化 UI 元数据对接地精度的增益，为多模态 Agent 的设计提供实证依据。
4. **任务难度与步数预算的解耦分析**：通过消融证明增加步数上限无法显著提升性能，说明瓶颈在推理而非资源，启示后续研究应聚焦规划质量和错误恢复机制。
5. **跨工具真实环境仿真的评估范式**：将 Agent 置于包含终端、IDE、浏览器、数据库的完整工作环境中，比沙盒环境更能揭示真实部署缺陷，为 OS Agent 评测提供了可迁移的架构思路。

## 关键术语表
- **DSAgentBench**：首个在真实操作系统中评估 AI Agent 能否自主完成端到端数据科学工作流的基准测试，含 275 个多阶段任务。
- **A11y Tree（无障碍树）**：通过 AT-SPI 从桌面界面提取的结构化 UI 元数据，包含元素角色、可访问名称、边界框和交互状态，增强 Agent 的接地精度。
- **Deterministic Evaluator**：针对每个任务编写的确定性 Python 评测脚本，校验输出文件的数值正确性、文件存在性和可视化语义，而非仅检查代码能否运行。
- **Multi-stage Task**：需要 Agent 在多步骤间迭代执行代码、检查中间输出并调整后续操作的复杂任务（占基准 56.7%）。
- **OSWorld**：评估多模态 Agent 在真实操作系统中执行开放式任务的基准框架，DSAgentBench 在此基础上扩展数据科学专用工具。
- **GUI Grounding**：Agent 将视觉观察（截图）映射到具体 UI 操作坐标的能力，是桌面 Agent 成功执行任务的关键前提。
- **Terminal-first Strategy**：引导 Agent 优先使用终端/命令行完成子任务以绕过 GUI 交互风险的策略；实验表明此策略对整体性能提升有限。

## 可复现要素
- **数据集**：275 个任务及所有数据集（来自 Kaggle、OpenML、SQLite、GitHub）已预下载并将随基准一同发布；链接：https://github.com/vis-nlp/DSAgentBench
- **代码**：完整的系统提示、评测函数和模型推理脚本将开源
- **环境**：基于 OSWorld 的 Ubuntu 虚拟机，pyautogui 动作空间，1920×1080 分辨率
- **关键超参**：temperature=0.1，top-p=0.9，max output tokens=2000，task timeout=1800s，max steps=15（主实验），操作后延迟 2s
- **开源模型推理**：GCP n1-standard-4 实例 + vLLM 引擎；闭源模型通过 API 调用
