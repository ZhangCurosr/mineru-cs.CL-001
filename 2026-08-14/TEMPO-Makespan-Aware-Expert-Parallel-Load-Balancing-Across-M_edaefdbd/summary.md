---
title: "TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M"
source: https://arxiv.org/pdf/2608.13057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:36:34"
field: "大语言模型推理系统优化"
keywords: ["Mixture-of-Experts", "Expert Parallelism", "Load Balancing", "LLM Serving", "Makespan Optimization", "MoE Dispatch", "Inference System"]
innovations: ["两段式专家成本模型（激活地板+M-tile填充）与黑盒校准闭环", "将per-batch EP调度形式化为固定费用makespan问题并给出加法近似保证", "毫秒级启发式求解器+零关键路径SGLang集成，相图预测部署收益"]
benchmarks: ["Testbed A 8-GPU 壁钟微基准", "Qwen3-235B-FP8 EP8 端到端吞吐与p99 TPOT", "DeepSeek-V3 EP8 端到端验证", "Testbed B 2-node EP16 RoCE 多节点基准"]
---

# 论文速读：TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M

## 一句话总结
TEMPO 提出了一种感知 makespan 的专家并行（EP）调度器，通过黑盒校准获得包含"内存瓶颈（激活地板）"和"计算瓶颈（分组 GEMM 分块填充）"两段式专家成本模型，将每批次调度形式化为固定费用 makespan 问题并设计毫秒级求解器，在 Qwen3-235B 上实现 4–7% 吞吐提升、p99 token 间延迟降低约 15.6%，其相图方法能提前预测部署收益或无收益。

## 研究问题与动机
- **现有代理目标不成立**：生产系统中 EPLB/LPLB/UltraEP 最小化最大 token 数（隐含 time ∝ tokens），METRO 最小化最大激活专家数（隐含 time ∝ expert count），均假设专家耗时与单一代理呈线性关系。
- **硬件实测揭示两 regime 结构**：在 DeepGEMM fp8 grouped GEMM 下，每专家 FFN 耗时在约 156–168 token/专家（n\*）以下时呈平坦（HBM 权重流主导，属内存瓶颈 regime）；超过 n\* 后呈线性（分组 GEMM 计算主导，属计算瓶颈 regime）。
- **两个隐藏硬件效应**：① 激活地板 b——多激活一个专家副本需从 HBM 加载其权重，与 token 数无关；② M-tile 填充——分组 GEMM 以 128 token 为单位向上对齐，将一个专家拆到多个副本会产生额外的 padded tiles。
- **混合 regime 是常态**：真实 decode 批次中 92–100% 同时包含 flat/linear 两类专家（热专家深在线性区、冷专家在平坦区），单一代理无法同时兼顾。

## 核心贡献（创新点）
1. **两段式（+Tile-aware）专家成本模型**：$t_g = \max(a + bG_g,\; c + \beta N_g)$ 及其带参数 $b_2$ 的扩展，同时刻画权重流激活地板与 128-token M-tile 阶梯效应；现有工作仅对 token 数或激活数建立线性关系。
2. **将 per-batch 调度形式化为固定费用 makespan 问题并证明其 NP-hardness（2 GPU 已 NP-hard）**，同时给出退化极限（b→0 或 β→0）下的多项式归约，以及全复制下的加法近似保证 $M(A_3) \leq \text{OPT} + \max(b, \beta n_{\max})$。
3. **time-fast 启发式求解器**：四阶段流水线（成本感知贪心种子 → 增广链激活重平衡 → 瓶颈局部搜索含部分迁移 → 集成切换容差），在 2ms 内逼近 10s MILP（均值差距 ≤1.005×）。
4. **零关键路径开销的 SGLang 集成**：概率 dispatch 与 count collection 融合为单个 CUDA Graph 常驻 kernel，求解器以 out-of-process 运行，比 SGLang  shipped token-LP dispatcher 吞吐高 1.4–1.7×。
5. **相图方法论**：沿 batch size × 偏斜度 × 复制率 × 专家形状扫表，预测每格最优固定策略及 TEMPO 增益，并以 Qwen3-235B（赢区内部）和 DeepSeek-V3（赢区外部）两个旗舰模型做 bracketing 验证。

## 方法详解
- **成本模型（核心公式）**：
  $$t_g = \max(a + bG_g,\; c + \beta N_g)$$
  其中 $G_g$ 为 GPU g 上被激活的 (expert, replica) 对数，$N_g$ 为 token 总数。$b$ 为激活地板（μs/expert），$\beta$ 为计算斜率（μs/token）。
- **Tile-aware 扩展**：引入 $b_2 \approx b/3$ 描述 M-tile 阶梯后续台阶：
  $$t = \max\!\big(a + bG + b_2(T - G),\; c + \beta N\big),\quad T = \sum_e \lceil n_e / 128 \rceil$$
  解码时 $n_e \leq 128$ 所以 tile-blind 与 tile-aware 解重合；prefill 阶段启用 $b_2$。
- **通信项扩展**（多节点）：加入第三段 $c_2 + \gamma N_g$：
  $$t_g = \max(a + bG_g,\; c + \beta N_g,\; c_2 + \gamma N_g)$$
- **拓扑感知两阶段 dispatch**：先求解 per-GPU 份额 $x_{e,g}$，再以 same-node-first 运输规则配对源节点与副本，实现跨节点流量最小化而不改变 per-GPU 负载。
- **求解器四阶段**：
  1. 降序 token 贪心放置整专家；2. 1/2 步增广链重平衡激活；3. 瓶颈局部搜索 + 三元搜索部分迁移；4. 集成（token-LP + round-robin 证书 $A_3$ + 局部搜索输出），以 1% 容差切换。
- **理论保证**：Theorem 2 给出全复制下 $M(A^*) \leq (\text{OPT} + \max(b, \beta n_{\max}))/(1-\tau)$（$\tau=1\%$）。

## 实验与结果
- **黑盒校准**：10 分钟对 (G,N) 网格跑部署 MoE 流水线，fp8 grouped GEMM 拟合误差 4–8%（Table 6：DSv3 $b=14.78\mu s$, $\beta=0.0945\mu s/token$）。
- **相图（calibrated simulation）**：DSv3 shape，EP8–64，TEMPO 在所有格子内 ≤1% 偏离最佳固定策略，在混合区赢至多 8.5%（DSv3）/10.2%（Qwen3）/11.8%（DSv2）；EP32–64 外推赢至多 15.5%。
- **8-GPU Testbed A 壁钟微基准**：B=128（内存 regime）TEMPO 比 EPLB-even 快 11–14%；B=2048（计算 regime）与 token-LP 持平；全程 within 5% of per-B best。
- **端到端 Qwen3-235B-FP8（Testbed B，EP8，94 MoE 层）**：GovReport 吞吐 +5.0%（521→549 tok/s）；Poisson 8 req/s p99 TPOT −15.6%（191→226 ms），TTFT −12.5%。
- **DeepSeek-V3（Testbed B，EP8，61 层）**：所有 workload 均 −2 至 −3%，与 noop 控制无差异，符合相图预测（32 expert/GPU 统计平均抹平 imbalance）。
- **SGLang 原生 token-LP**：在两模型上全面崩溃（−10% 至 −56%），主要因 per-layer in-graph 求解 + EP collective 在 94 层叠加死锁/失败。
- **多节点 EP16（Testbed B，RoCE 8×400G）**：drift-16 场景下 flat table −3.5%，topology-aware hier split +4.1%（decode-heavy）/ +7.2%（Qwen3-235B end-to-end）。
- **Staleness 分析**：stale-1（前一窗口计数）保留 exact solve 22 pp 增益的 75–85%，end-to-end 仅损失约 2 pp。

## 相关工作脉络
- **Token-_proxy 调度（EPLB [5], LPLB [6], UltraEP [17], FlexMoE [15], SmartMoE [21]）**：假设 time ∝ tokens；本文指出此假设在 flat regime 失效，且 LPLB 文档已将非线性专家成本列为开放问题。
- **Activation- proxy 调度（METRO [13]）**：假设 time ∝ expert count；本文证明其在 compute regime 下系统性地过估冷专家代价。
- **Weight-moving 调度（MoonEP [1]）**：在线预取冗余专家实现完美 token 平衡；本文证实在 decode regime 其权重预取开销（283–502 μs）已超过整个 MoE 层预算（95–107 μs），二者划分相图而非竞争。
- **训练侧负载均衡（LLEP [14], TAOT [22]）**：利用梯度步间 weight migration window；本文指出 dispatch 与 placement 分层互补，$R=16/K\geq16$ 塌陷行正是 placement 层的修复任务。
- **NP-hardness 序列**：Huang et al. [11] 证明 GPU-quota 粒度下 MoE serving NP-hard；本文 Theorem 1 在固定 placement + per-batch dispatch 粒度下再证 NP-hard，二者互不包含。
- **Semi-matching 理论**：Lemma 2 将 β→0 极限归约为二分图最优半匹配 [10]，其增广路算法构成 tempo fast 的第二阶段基础。

## 局限性与未来方向
- **L1–L2**： headline 数字来自校准模拟器（mid-range transfer error 2–6 pp）；58 层完整 DSv3 规模（EP32+）仍为外推；更深 hierarchy（4+ 节点）需更细粒度通信项。
- **L3**：赢区高度条件依赖——小专家 bf16 shape 无收益、placement 新鲜 + 充裕复制率下所有策略持平；赢区三坐标：中等 expert/GPU、充分 skew、expert compute 占 step 足够份额。
- **L4**：kernel/driver 更新需重新做 10 分钟校准；自动收敛但非零维护成本。
- **L5**：$(G,N)$ 模型存在底限——DeepGEMM 在 uniform per-slot load 下更快，这一第三成本维度目前只能通过 ensemble 的 1% 容差缓解。
- **L6**：tempo fast 是有界启发式而非理论最优；受限副本集（restricted replica sets）下 Theorem 2 的加法保证尚未证明，为开放问题。

## 研究启发与可借鉴点
1. **两段式 black-box 校准范式**：先在关键 kernel 跑 (G,N) 网格测 wall-clock，再做全管道 refit 闭合 loop（pairwise ranking agreement 93%），可用于其他 kernel-heavy 系统的代价建模。
2. **相图（phase diagram）作为部署前预测工具**：沿 batch size × skew × replication 扫表，输出"赢/输/边界"而非单一数字，直接指导生产部署决策（本文 bracketing 两种旗舰模型两次验证）。
3. **加法近似保证 + 工程截断**：Theorem 2 的全复制保证可作为集成解的安全底线（1% 容差下 $M(A^*)\leq(\text{OPT}+\max(b,\beta n_{\max}))/(1-\tau)$），启发其他 NP-hard 调度问题"理论下界 + 启发式"双轨设计。
4. **拓扑感知两阶段分离**：先解 per-GPU 份额、再以 same-node-first 运输配对，在不改 solver 核心的前提下将跨节点流量降 ~11 pp，可迁移至其他 multi-node sparse model 调度。
5. **out-of-process solver + fused in-graph kernel**：消除 GIL 争用（in-thread 损失 8–10%）且保持零额外 collectives，为 SGLang/TGI 等推理框架集成自定义 dispatcher 提供架构参考。

## 关键术语表
- **Expert Parallelism (EP)**：将 MoE 各专家分片到多 GPU，每层经历 dispatch all-to-all → per-expert grouped GEMM → combine all-to-all，以 makespan（最慢 GPU 时间）为层开销度量。
- **Activation floor $b$**：从 HBM 流式加载单个专家权重的最低开销（μs/expert），与 token 数无关，决定 flat regime 的代价下限。
- **Inflection $n^*$**：per-expert 耗时从 flat 转为 linear 的临界 token 数（本实验 ≈156–168 tokens），由 HBM 带宽与计算屋顶线交叉决定。
- **Fixed-charge makespan 调度**：目标函数形如 $\min_g \max(a+bG_g,\; c+\beta N_g)$ 的 min-max 问题，同时含固定启动费与线性变量费，在 2 GPU 下已 NP-hard。
- **Round-robin certificate $A_3$**：按 token 降序 cyclic 分配专家到副本的 greedy 策略，提供 additive approximation guarantee 的解下界。
- **Stale dispatch**：使用前一窗口（≤200 ms 前）的计数做决策，牺牲少量精确度换取 solver 不阻塞 critical path；实测保留 75–85% 精确解增益。
- **Same-node-first transportation**：拓扑感知 dispatch 的第二阶段规则——优先将同一 node 的 token 源配对到同 node 的副本，使 intra-node 流量最大化。
- **Drift**：placement 按分钟级 horizon 更新，drift K 表示 top-K 热专家的统计被随机冷专家统计替换，模拟 placement 滞后导致的 imbalance。

## 可复现要素
- **数据集**：公开 routing traces（Wikitext、GSM8K、OASST1、GovReport、ShareGPT、bilingual Q&A corpus），合成 Zipf/D以直分布用于相图扫描；论文未提供原始 trace 下载链接（仅 artifact 链接匿名化）。
- **代码/权重**：SGLang 集成补丁位于 anonymized artifact 链接；DeepGEMM fp8、DeepEP、EPLB/LPLB 均为开源仓库（见参考文献 [1][3][4][5][6]）。
- **关键超参**：$n^*\approx156$–168（校准得）、$b$（1.74–14.78 μs 随 shape）、$\beta$（0.0108–0.0945 μs/token）、$b_2\approx b/3$、ensemble 容差 $\tau=1\%$、refresh period 200 ms（默认）、stale sweep {8,16,32}。
- **硬件**：Testbed A（8 GPU，具体型号文中以 footnote 标注）、Testbed B（2×8 GPU，RoCE 8×400G/node，CUDA 13.2）。
- **校准流程**：(G,N) 网格 $G\in[1,40]\times$ tokens/expert $\in[1,4096]$（log ladder），30-replay CUDA graph median，expert weight copy rotation 防 L2 reuse，alternating assignment 两段 max-affine 拟合（50 次迭代）。
