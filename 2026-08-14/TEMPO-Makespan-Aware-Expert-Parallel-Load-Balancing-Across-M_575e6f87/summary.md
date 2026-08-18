---
title: "TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M"
source: https://arxiv.org/pdf/2608.13057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:22:05"
---

# 论文速读：TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M

## 一句话总结
论文提出 TEMPO，一种感知 makespan 的专家并行（EP）MoE 服务负载均衡调度器，通过黑盒校准的双相成本模型（区分显存绑定的 activation floor 与计算绑定的 token 线性项）精确定价 expert 耗时，在毫秒级四阶段启发式求解器的支持下，在真实部署环境中实现自适应调度，在混合相区场景下比现有 token/activation 代理策略最高提升 15.5% 吞吐，并通过相图明确预测赢区与输区。

## 研究问题与动机
- **核心问题**：MoE 模型专家并行服务中，每个 MoE 层的耗时由 EP group 中最慢 GPU 决定（makespan），生产系统通过 placement 层（长期平均负载复制 hot expert）和 dispatch 层（每批 token 分配）两级应对；dispatch 层是剩余 batch-to-batch 波动的关键，但现有 dispatcher 优化的是 token 数或激活 expert 数等代理指标，而非真实耗时。
- **代理假设失效**：Token balancing（LPLB、UltraEP）假设 time ∝ tokens；activation balancing（METRO）假设 time ∝ expert count；但实测显示 per-expert FFN 时间在低于拐点 n*≈156–168 tokens 时 flat（HBM 权重流式传输主导，cost 附着于激活的 replica 而非 token），高于时 linear（grouped GEMM 以 128-token M-tile 分块，分割 expert 会制造 padded compute），两种代理在对方主场相区系统性错误 30–70%。
- **混合相区是常态**：在真实 decode 路由 trace 中，92–100% 的批次同时包含 ≥20% 的 flat-regime expert（冷 expert，约占 47% 激活）和 ≥20% 的 linear-regime token（热 expert，承载约 91% token），单一固定 proxy 无法兼顾。
- **每批调度的计算难度**：形式化为固定费用 makespan 问题，在 2 GPU 全复制条件下已 NP-hard，但在纯 flat（β→0）或纯 linear（b→0）退化极限下可多项式求解，硬度来源于两相区交互。

## 核心贡献（创新点）
- **双相成本模型与相图**：提出 max-affine 时间模型 t=max(a+bG, c+βN) 及扩展的 tile-aware 三参数形式，通过 10 分钟黑盒 benchmark 校准，捕获权重流式传输 floor（b）和 M-tile padding 阶梯（b₂）两项硬件效应；基于校准模型构建相图，揭示无固定 proxy 可通吃所有场景，最优策略沿解析可预测边界翻转。
- **tempo fast 四阶段求解器**：代价感知贪心初始化（避免切割冷 expert 制造额外 floor）+ 增广链激活重平衡（flat 相区的 semi-matching 截断）+ 瓶颈局部搜索部分迁移（linear 相区的三分搜索）+ token-LP 与轮询证书 A₃ 的集成选择（>1% 切换容差），在 2 ms 内求解且保证不超过任何经典代理 1% 以上，全复制下有加法近似保证 OPT+max(b,βn_max)。
- **零开销 SGLang 集成架构**：in-graph 仅一个 fused kernel 完成 probabilistic dispatch 与 count collection，solver 运行在分离进程并通过 race-safe pinned table 发布，消除 critical path 上的额外 kernel 与 collective，端到端延迟可忽略（±2–3% noise 内）。
- **拓扑感知多节点分发**：两阶段方案先求解 per-GPU token 份额，再以 same-node-first 运输规则将 source 拆分分配到 replica，生成每节点一张分发表，在不改变 solver 核心的前提下将跨节点流量降至最低，在 2-node EP16 场景额外贡献 ~8 pp 吞吐 swing。
- **可证伪的部署诊断规则**：通过 Qwen3-235B（赢区内）和 DeepSeek-V3（赢区外）两个旗舰模型 bracket 相图预测区域，实测分别获得 +4–6% 吞吐 / −15.6% p99 尾延迟和 −2–3% 机制成本，将相图从模拟 artifact 转化为可 falsifiable 的部署规则。

## 方法详解
- **成本模型校准**：在 (G, tokens-per-expert) 网格上黑盒测量 DeepGEMM fp8 masked grouped GEMM（CUDA-graph 时序、权重 copy rotation 防 L2 复用、固定 expected m 冻结 JIT tuner），拟合两段 max-affine：flat 区由 b（activation floor，HBM 带宽 80–88%）主导，linear 区由 β（per-token compute cost）主导；Testbed B 进一步解析 tile staircase，引入 b₂≈b/3 参数捕获 128-token 边界的阶梯跳跃，decode 区两模型解重合。
- **问题形式化**：给定 router 输出的 n_e、placement 给出的 R(e)、token 分配 x_{e,g}≥0（Σ_g x_{e,g}=n_e，支撑于 R(e)）和 activation 指示 z_{e,g}=1[x_{e,g}>0]，优化 min_g max(a+bΣ_e z_{e,g}, c+βΣ_e x_{e,g})；扩展加入通信项 max(c₂+γN_g) 形成三参数模型 (4)。
- **四阶段求解器**：
  1. Cost-aware greedy seeding：按 token 降序将 whole expert 放置到当前边际代价最低的 replica，避免切割冷 expert。
  2. Augmenting chain activation rebalancing：flat 相区瓶颈是 max_g G_g，直接使用单步移动会导致 destination 上升，使用 1/2 步截断的半匹配增广链移动激活。
  3. Bottleneck local search with partial migrations：linear 相区从瓶颈 GPU 迁移 token 质量，用三分搜索找最优 partial split（目标函数沿 split size 为 piecewise-linear unimodal），迭代 ≤300 次。
  4. Ensemble with switching tolerance τ=1%：token-LP 输出与轮询证书 A₃（Theorem 2 的加法保证）在模型下打分，仅当 >1% 更好时切换，保护 near-tie 下硬件对 whole-expert 结构的偏好；在 full replication 下继承加法保证至部署求解器。
- **拓扑感知分发**：求解 per-GPU 份额后，对每个 expert 按 same-node-first 规则拆分 source 到 replica，生成每 source node 一张分发表；per-GPU 负载（compute makespan）与 flat 解严格相同，仅 source-to-replica pairing 变化。
- **集成细节**：fused kernel 原子递增 persistent cumulative per-expert counter（masked 对 CUDA-graph padding rows），background thread 在 side stream 快照计数、跨 rank 聚合 window diff、派发 numpy-only worker 进程求解；idle expert 行不修改、invalid replica 列为零、kernel 在 use 时 renormalize 每行，race 读到旧/新混合行仍为合法分布，最坏 tear 回退到 uniform over replicas。

## 实验与结果
- **数据集与硬件**：Testbed A（8-GPU）和 Testbed B（2×8 GPU，RoCE 8×400G per node），使用 Qwen3-235B-FP8（94 MoE layers，16 logical experts/GPU）与 DeepSeek-V3-0324（61 layers，32 experts/GPU）两个旗舰模型；routing traces 来自 real workload（ShareGPT、OASST1、GSM8K、GovReport、bilingual Q&A）。
- **评估基线**：LPLB（token-LP）、METRO（activation balancing）、EPLB-even（uniform replica split）、static placement、SGLang shipped dispatcher、MoonEP（weight-moving baseline）。
- **主要结果**：
  - Testbed A EP8 微基准：B=32（memory-bound）TEMPO 超越 EPLB-even 11–14%、token-LP 7%；B=2048（compute-bound）与 token-LP 持平，METRO 落后 7–11%；全范围 within 5% of per-B best fixed policy，固定 proxy 均有 ≥7% 失败区。
  - Testbed B Qwen3-235B（赢区内）：GovReport 长文档 summarization 吞吐 +5.0%（非重叠范围 521–524 vs. 549–554 tok/s），Poisson 8 req/s 下 p99 TPOT −15.6%（191 vs. 226 ms），TTFT −12.5%；ShareGPT parity（−0.2% throughput，−8.9% p99）。
  - Testbed B DeepSeek-V3（赢区外）：所有 workload 位于 −2 至 −3%，与 noop control 不可区分，验证相图预测。
  - 与 SGLang shipped LPLB 对比：+38–70% request throughput，但 like-for-like port 分析表明大部分差距来自架构（in-graph per-layer collective vs. zero-collective）而非目标函数。
  - 2-node EP16 拓扑感知：decode-heavy 场景 +7.2%（hier vs. flat +4.1%）、GovReport +6.1%；flat table 在 all-to-all-bound 点反而 −3.5%，拓扑项贡献 ~8 pp swing。
  - 相图验证：min gain ≥−0.3%，max win 8.5%（DSv3）/10.2%（Qwen3）/11.8%（DSv2 1.5× replication）；EP32–64 外推最大 win 15.5%。
- **最强结果**：DSv3 形状 EP32–64 外推场景 +15.5%；Qwen3-235B GovReport 长 prefll +5.0% 吞吐、p99 TPOT −15.6%。

## 相关工作脉络
- **LPLB [6]**：per-batch token LP 基于 replica 图，文档明确标记非线性 expert 成本为 open problem；本文完成该问题陈述，提出可处理双相成本的时间模型，并证明 like-for-like port 后差距主要来自架构。
- **METRO [13]**：balance activated experts 应对 memory-bound decode；本文指出其在 compute-bound regime 系统性错误，两者可组合使用（placement 层 vs. dispatch 层正交）。
- **EPLB [5]**：placement 层使用长期平均负载复制 hot expert；本文定位在 dispatch 层处理 batch-to-batch 波动，与 placement 正交可组合，且证明更快刷新 placement 在 floor regime 反而有害（fragment more experts）。
- **MoonEP [1]**：weight-moving balancer 在线 prefetch redundant experts 实现完美 token balance；本文证明在 decode scale 其权重预取开销（283–502 µs）已超过整个 MoE 层预算（95–107 µs），在 prefill high-skew（MaxVio≳20）corner 才有效，与 token-moving 形成相区互补。
- **UltraEP [17] / TAOT [22]**：训练侧负载均衡器；本文在相同 replica 约束下公平比较其 kernel，指出 dispatch 与 placement 分工，在 R=16/K≥16 时所有 kernel 坍缩到 1.000（修复属于 placement 层）。
- **ViBE [9]**：heterogeneous hardware latency-aware balancing；本文认为其线性率假设忽略了本文明确建模的 regime nonlinearity，两者优化轴正交。

## 局限性与未来方向
- **L1**：headline 数字来自模型空间，虽经 wall-clock 验证但 mid-range transfer error 为 2–6 pp。
- **L2**：完整 DSv3 规模（58 MoE layers, EP32+）仍为外推；更深层次（rail-optimized fabrics, 4+ nodes）可能需要更细粒度的 traffic term。
- **L3**：win region 条件性，小 expert bf16 形状（b≈1.7 µs）无 adaptive gain；fresh placement + 充足 replication 下所有 policy 平局。
- **L4**：静态校准需 kernel/driver 更新后重跑 10 分钟校准循环（循环自动化且在一次迭代内收敛）。
- **L5**：(G,N) 模型存在 floor，未捕获 DeepGEMM 在 uniform per-slot load 下的 kernel 偏好
