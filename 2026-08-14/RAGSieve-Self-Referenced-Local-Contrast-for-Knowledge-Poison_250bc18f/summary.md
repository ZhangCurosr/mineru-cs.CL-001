---
title: "RAGSieve-Self-Referenced-Local-Contrast-for-Knowledge-Poison"
source: https://arxiv.org/pdf/2608.13010v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:47:43"
---

# 论文速读：RAGSieve-Self-Referenced-Local-Contrast-for-Knowledge-Poison

## 一句话总结
论文提出 RAGSieve，一种基于“自引用局部对比”的 RAG 知识投毒检测框架，通过在线模块 RSQ（对比当前查询的生成候选与检索尾部）和离线模块 RSG（对比文档与其局部语料图），在无需外部干净语料或攻击标签的情况下，有效识别并过滤检索注入的虚假证据。

## 研究问题与动机
- RAG 系统的外部语料可被攻击者少量注入后修改，导致目标查询将恶意构建的错误答案排名置顶，现有防御依赖可信参考源、特定攻击痕迹或全局阈值。
- 现有在线检测器（RA-Guard、GMTP、EcoSafeRAG）依赖 fluency/perplexity 变化、retriever 梯度或攻击诱饵，对分散式、协同优化或高流畅载体的隐蔽投毒覆盖不足。
- 现有离线方法（CleanBase、AHD）依赖全局固定边阈值检测绝对语义 clique，无法适应不同主题区域的自然密度差异，跨语料泛化性差。
- 核心难点在于缺乏“可信参考”：语料本身就是被攻击对象，且正常检索结果本身也具备语义相关性和词汇多样性，需设计一种与 inspected event 同源的对比基准。

## 核心贡献（创新点）
- **提出自引用局部对比范式**：将检测参考系从外部干净语料转变为系统自身生成的对照集，避免全局阈值对异质主题区域的敏感性。与 CleanBase 等依赖单一全局长尾阈值的方法本质不同，该方法以被检测事件的内生环境作为 matched control。
- **设计 RSQ 在线查询时检测模块**：利用检索尾部（ranks 6–20）作为同源对照，融合答案锚点集中度、字符级脚本完整性、多尺度语言模型困惑度跃迁及查询对齐断层四项证据进行联合打分。与 GMTP 等依赖单一梯度或困惑度信号的方法本质不同，RSQ 从词汇分布、字符 artifact、局部 fluency 跃迁与语义对齐断层四个正交维度捕获 promotion 痕迹。
- **设计 RSG 离线语料图检测模块**：在语料 ingestion/审计阶段，仅保留语义相近但词汇 distinct 的邻居构建局部图，测量最强邻居相似度超出文档自身 kth-neighbor baseline 的超额密度。与 AHD 等通用异常分数或绝对 clique 检测方法本质不同，RSG 的基线随文档所在拓扑区域自适应，能识别 deliberate dispersed 的协同注入。
- **验证双端点联合部署的安全-可用性权衡**：证明离线图检测与在线尾部对比可互补，联合部署将攻击成功率从 67.4% 降至 14.0%，同时保留 41.3% 的未中毒 F1。与单一时间点防御相比，本文构建了覆盖 corpus ingestion 与 active retrieval 两个信任边界的纵深防护框架。

## 方法详解
- **核心思想**：将可疑证据与同源局部环境对比（local contrast），位置/密度差异而非绝对异常值作为检测信号。查询局部对比与语料局部对比使用不同对照集，但共享同一比较原则。
- **RSQ（在线检测）**：给定查询 $q$ 及排序结果 $D_q$，生成候选 $C_q=\{d_1..d_5\}$，检索尾部参考 $R_q=\{d_6..d_{20}\}$。对每个候选计算四项归一化证据并求和 $S_{\text{RSQ}}$：
  1. *Answer-Anchor*：统计候选中答案承载 token 在尾部中的出现频率，在 rank-exchangeability 假设下服从超几何分布，计算集中度 p 值并用 Simes 程序合并，转换为 $E_a \in [0,1]$。
  2. *Script Integrity*：将字母字符映射到粗粒度书写系统，计算文档非主脚本比例 $v(d)$，与尾部 $m$ 个文档做有限样本中秩上尾概率检验，得 $E_i$。
  3. *Surprisal*：用 Qwen3-0.6B-Base 计算窗口 NLL，检测高于文档典型窗口的突发（burst）与左右 change point，通过信息门控 $\tau_w = \tau_0\sqrt{w_0/w}$ 过滤微小波动，得 $E_p$。
  4. *Query Alignment*：将文档切分为重叠窗口，计算每窗口与 $q$ 的 BERTScore F1 轨迹，检测最大跳变 $J(d,q)$，同样与尾部做概率检验得 $E_s$。
  超过阈值 $\eta_Q=1$ 则标记该候选，并从原始排序中按序回填。
- **RSG（离线检测）**：对每文档 $d_i$，找 exact $k_G$ 近邻，仅保留满足余弦相似度 $\geq \tau_e=0.85$ 且 Jaccard $\leq \tau_l=0.60$ 的邻居（语义近但词汇不同，防重复膨胀）。计算最强 $h_G$ 邻居均值相似度 $\mu_i$ 与第 $k_G$ 邻居基线 $b_i$ 的相对超额，结合支持度缩放 $|N_i|/c_G$ 得图密度证据 $D_i$。转为实证上尾概率 $p_{\text{RSG}}(i)$ 后，结合字符级完整性谓词 $I_i$ 分配告警预算 $\alpha_G=0.05$，最终得分 $S_{\text{RSG}} = \max(T_i, I_i)$，超阈 $\eta_G=0.5$ 则隔离。

## 实验与结果
- **设置**：NQ、HotpotQA、MS MARCO 三个 QA 数据集；BGE-M3、E5-large-v2、all-MiniLM-L6-v2 三种稠密检索器；PR-B/W、CEM-C/D、CPA-RAG、CamoDocs 六种投毒构造（覆盖黑/灰/白盒）。
- **RSQ 文档级检测**：宏平均 AUROC 达 **95.2%**，在 ≤5% 干净文档移除预算下检出 **82.2%** 投毒文档（GMTP 为 81.1%/52.5%）；QA 操作点上移除 73.9% 投毒与 2.2% 干净文档。
- **RSG 文档级检测**：宏平均 AUROC **93.3%**，检出 **79.8%**（CleanBase 为 79.4%/37.6%）；跨 54 种组合中 37/38 项优于 CleanBase，在 CamoDocs 分散攻击上反超最达
