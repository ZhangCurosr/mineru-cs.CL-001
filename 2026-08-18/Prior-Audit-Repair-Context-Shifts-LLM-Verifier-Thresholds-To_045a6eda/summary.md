---
title: "Prior-Audit-Repair-Context-Shifts-LLM-Verifier-Thresholds-To"
source: https://arxiv.org/pdf/2608.16003v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:09"
field: "LLM verification & evaluation"
keywords: ["LLM verifier", "false alarm rate", "signal detection theory", "in-context learning", "pipeline architecture", "process verification"]
innovations: ["Prior audit-repair episode lowers false alarm rate by 2.8-11.5 pp across all 15 model×wording combinations", "Effect is a criterion shift (c) not discrimination gain (d') per signal detection analysis", "Five-control decomposition (AF/AV/AN/AX/AXN) isolates content vs format effects, contradicting polarity drift theory"]
benchmarks: ["ProcessBench"]
---

# 论文速读：Prior-Audit-Repair-Context-Shifts-LLM-Verifier-Thresholds-To

## 一句话总结
本文发现当LLM验证器上下文中包含一个已完成的"审计→修复"episode时，会显著降低其对正确解的错误报警率（FAR降低2.8–11.5 pp），且该变化源于判断标准（criterion）的偏移而非判别力改善；这一现象与既有极性漂移理论预测相反，提示pipeline架构设计会隐性改变验证器行为。

## 研究问题与动机
- **核心问题**：在自动化检查pipeline中，将LLM配置为checker与fixer时，这种管道结构是否会影响checker的判断输出？现有研究多关注"如何让checker更准"，却忽视了pipeline配置本身会改变checker行为。
- **现有方法不足**：
  1. Jin & Chen (2026)发现要求解释和修复的prompt格式会增加误判，但修复请求与审计在同一响应中，导致**输出格式效应**与内容效应混淆。
  2. Khullar et al. (2026)发现monitor对"自己的作品"更宽容，但未分离"归属声明"与"内容位置"两个因素。
  3. Temkit (2026)发现对话历史会使判断向极性漂移，但本任务是负向历史（报告错误），按该理论应**增加**误报而非减少。
- **研究缺口**：现有工作均未隔离"已完成审计→修复episode"这一纯上下文效应，也未在byte-identical的任务条件下量化其对验证器行为的影响。

## 核心贡献（创新点）
1. **首次量化prior-audit-repair episode对FAR的因果效应**：在15种model×wording组合中均观察到FAR下降，与极性漂移理论预测方向相反。
2. **设计分离"内容效应"与"格式效应"**：通过控制五组语义等价但措辞不同的指令（F1–F5），证明效应来自episode内容而非prompt格式。
3. **引入信号检测理论解构变化机制**：通过$d'$与criterion $c$的联合估计，证明变化发生在决策阈值而非判别力层面。
4. **构建多对照分解框架**：AF（长度匹配填充物）、AV（仅审计）、AN（惰性续接）、AX/AXN（错误判决episode）五重对照，分离出repair content与audit verdict在不同模型上的互补作用。
5. **人工审计验证效应无害性**：对被消除的50个false alarms进行手工审查，82%确为错误判断，表明该criterion shift在当前operating point上是有益的。

## 方法详解
**实验设计**：
- **任务**：ProcessBench数学推理trace审计，模型输出约束JSON：verdict、first-error step、confidence、error type、≤40词evidence。
- **arms**：clean arm（929条人类标注正确的trace，用于测量FAR）+ detection arm（929条标注错误的trace，用于测量检测率）；两arm disjoint且source-matched（GSM8K/MATH/Olympiad-Bench/OmniMath）。
- **prior-context manipulation**：在target请求前插入一个已完成的audit→repair episode（来自50个独立item，每个episode被复用约9.3次target）。

**五重对照设计**（2×2 attribution交叉 + controls）：
| Contrast | 分离变量 |
|----------|----------|
| episode − AF | audit内容 vs 纯context |
| AV − AF | 仅audit（无续接）vs non-audit context |
| AN − AF | audit + 惰性续接 vs non-audit |
| AS − AN | repair请求存在性 vs 惰性续接 |
| AX − AS | 错误判决episode vs 正确判决episode |

**统计约定**：
- 20,000次bootstrap cluster bootstrap（按episode聚类，有效N≈50而非465）
- Holm–Bonferroni step-down within family（k=24 claim family）
- $d'$与c按Macmillan & Creelman (2004)计算，两arm独立resample后合并

**关键发现路径**：
- AS − AF主效应：Qwen3.6-27B -4.00 pp, Qwen3.6-35B-A3B -3.59 pp, Ministral-3-14B -8.83 pp（F1 wording）
- 全5种wording全部15组合显著
- $\Delta c$在15/15组合中向"更宽容"移动，13/15通过多重校正；$\Delta d'$ 0/15显著

## 实验与结果
**数据集**：ProcessBench (Zheng et al., 2024)，从1,101条eligible clean traces中分配三arm：929 clean targets + 50 warmup items + 122 reserve；detection arm另取929条labelled-incorrect traces。

**模型**：Qwen3.6-27B、Qwen3.6-35B-A3B、Ministral-3-14B（经instrument-sensitivity screen预筛选通过）；Gemma-4-26B/31B因frame responsiveness不足被screened out。

**主要结果**：
- **主效应（Table 1, F1）**：episode − AF = -4.00, -3.59, -8.83 pp；AF − R0 filler本身对Qwen-27B为正（+1.58 pp），证明效应来自audit内容而非context presence。
- **全wording鲁棒性（Table 4）**：episode − AF在5/5 wordings上全部显著（k=24）。
- **AX vs AS极性测试（Table 2）**：Ministral上AX − AS = -5.62 pp（所有5 wordings显著），与polarity drift预测（应升高FAR）完全相反；Qwen-27B因37/50 repair仍输出verdict="correct"导致wedge退化。
- **组件分解**：Repair content（AX − AXN）主导Qwen模型（4/5 wordings），error-verdict episode（AXN − AN）主导Ministral（5/5），repair request（AS − AN）三模型均有survivor。
- **信号检测（§6）**：$\Delta c$ 15/15组合同向移动（median |Δc|/|Δd'| = 1.85），$\Delta d'$ 0/15显著；balanced accuracy 15/15提升，F1下+1.22 pp（27B）、+2.62 pp（Ministral）。
- **人工审计**：50个R0 false alarms中82%确为错误（41/50），16%为可辩护严格解读，2%疑似gold label error；逻辑错误类型占defensible flags的7/20。
- **Reasoning enabled**：R0 FAR从0.185→0.063（27B）和0.232→0.059（35B），AS − AF相对效应-19.7%/-17.5%与off-reasoning的-21.6%/-14.3%相近，threshold读法依然成立。

## 相关工作脉络
1. **Temkit (2026)**：AME（Accumulated Message Effects）发现judgment向prior conversation极性漂移，negativity asymmetry = 1.52×；本文AX cell证伪该理论在此task上的预测（负向episode反而降低FAR）。
2. **Zhou et al. (2026)**：直接在ProcessBench上steer verifier strictness；本文差异在于无人工干预时pipeline配置自动移动criterion。
3. **Jin & Chen (2026)**：prompt format影响misjudgment，但fix请求与audit同响应，无法区分format效应与content效应；本文通过byte-identical任务+前置独立episode隔离content。
4. **Khullar et al. (2026)**：self-attribution bias；本文发现explicit attribution为null，placement效应在Ministral显著但与self-leniency方向相反。
5. **CriticBench/CriticEval (Lin et al., 2024; Lan et al., 2024)**：benchmark critique-correct pipeline质量；本文问的是上游问题——completed repair已在context时对critique本身的影响。
6. **LLM-as-judge literature (Zheng et al., 2023; Gu et al., 2024)**：关注judge sensitivity to self-preference、sycophancy、paraphrase等；本文引入新lever："judge被告知接下来要做什么"。
7. **Signal detection in LLMs (Cacioli, 2026)**：将temperature与criterion类比；本文扩展应用至verification setting并证明$\Delta d'$检验power仅为$\Delta c$一半（因两arm独立resample）。

## 局限性与未来方向
- **模型泛化**：仅测试三个open-weight模型，Gemma被screened out（frame responsiveness不足），未见frontier/closed judge结果。
- **One benchmark/task family**：仅在ProcessBench数学推理上验证，未扩展至code review或其他verification task。
- **Wedge不识别**：AX vs AS仅Ministral干净地separate polarity effect；Qwen-27B因37/50 repair仍输出correct而退化，Qwen-35B仅F1显著。
- **pool confound**：AXN与AN来自不同pool（错误vs正确trace），error verdict token与trace内容无法完全分离。
- **手工审计单一annotator**：50个样本由一位作者审查，无inter-annotator agreement。
- **Reasoning traces dropout**：开启reasoning时truncation和schema-invalid output是outcome-associated而非random，影响$\Delta d'$估计。
- **Prospective responsibility未见replication**：Appendix B的"stated future obligation" ladder与experienced episode效应分裂，未claim。

## 研究启发与可借鉴点
1. **Pipeline architecture as experimental variable**：checker-fixer wiring不仅是工程选择，更是影响checker输出的因果变量；建议将"管道配置"纳入LLM verification study的控制变量。
2. **五重对照分解框架**：AF/AV/AN/AX/AXN的对照设计可复用于分离任何"in-context episode"效应（如自我修正、多轮对话、role-playing）。
3. **仪器敏感性预筛选**：Screen模型前先测其FAR对explicit framing的响应（PC/PCL/PCH），避免dependent variable不移动导致null interpretation失效。
4. **信号检测理论在verification benchmark中的标准应用**：FAR alone不足以解释效应，必须配对detection rate计算$d'$与c；本文提供了完整操作模板。
5. **人工审计验证"缓解无害"**：当发现模型变得更lenient时，应抽样手工审查被"放过"的样本，确认是否为真正的fabricated false alarm（本文82%是）；这是区分"有益的criterion shift"与"有害的discrimination loss"的关键步骤。

## 关键术语表
**False Alarm Rate (FAR)**：模型在人类标注为正确的trace上报告错误的发生率；本文核心dependent variable。
**Criterion (c)**：信号检测理论中的决策阈值，c向positive方向移动表示更reluctant to flag（leniency）。
**Discrimination ($d'$)**：模型区分正确与错误trace的能力，d'变化反映真正的判断质量改善。
**Prior-context episode**：插入在target请求前的已完成audit→repair exchange，本实验的核心manipulation。
**Polarity drift**：Temkit (2026)提出的理论，prediction为judgment向prior conversation的polarity方向漂移。
**Instrument-sensitivity screen**：预实验筛选步骤，要求模型FAR对explicit leniency/strictness prime有显著响应才纳入主实验。
**Cluster bootstrap**：按episode聚类的bootstrap，有效sample size接近episode pool大小（~50）而非target数量（465）。
**Balanced accuracy**：(detection rate + specificity) / 2，本文用于综合评估FAR下降与detection rate变化的net effect。

## 可复现要素
- **数据集**：ProcessBench (Zheng et al., 2024)；clean arm 929条 + detection arm 929条 + warmup 50条 + reserve 122条
- **代码/权重**：所有模型open-weight（Qwen3.6-27B/35B-A3B、Ministral-3-14B、Gemma-4-26B/31B）；分析代码、frozen episode pools、per-item outputs、hand-audit verdicts随submission发布
- **关键超参**：T=0.7、8 samples/item、16384-token context、reasoning disabled（main study）、constrained JSON decoding
- **随机种子**：allocation seed = 20260805；bootstrap seed per (model, contrast)
- **模型版本**：Qwen3.6-27B (6a9e13bd)、Qwen3.6-35B-A3B (995ad96e)、Ministral-3-14B-Instruct-2512 (29439f81)
