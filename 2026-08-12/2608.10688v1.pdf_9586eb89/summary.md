---
title: "Leveraging Human Reading Behavior for Keyphrase Extraction: A Webcam-based Eye-tracking Corpus"
source: https://arxiv.org/pdf/2608.10688v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:34:30"
field: "关键短语抽取与认知计算"
keywords: ["Keyphrase Extraction", "Eye-tracking", "Webcam-based", "Chinese Academic Text", "Reading Behavior", "CLIS-ET", "Cognitive NLP"]
innovations: ["低成本网络摄像头眼动采集平台与中文 LIS 学术语料库 CLIS-ET", "系统验证 FFD/FN/TFD 三特征在多架构 KPE 模型上的提升效果", "揭示数据规模与眼动特征类型的交互效应（小样本偏好 FFD，大数据偏好 FN/TFD）"]
benchmarks: ["Abstract-320", "Abstract-5190"]
---

# 论文速读：Leveraging Human Reading Behavior for Keyphrase Extraction: A Webcam-based Eye-tracking Corpus

## 一句话总结
本文开发了一种低成本的网络摄像头眼动数据采集平台，构建了面向中文图情领域学术摘要的字符级眼动语料库 CLIS-ET（含 FFD、FN、TFD 三个特征），实验证明将阅读行为信号融入 KPE 模型可稳定提升抽取性能，Att-BiLSTM+CRF 模型搭配 FN+TFD 组合获得最佳效果。

## 研究问题与动机
1. **已有 KPE 方法忽视认知注意力**：现有研究主要依赖文本内部线索（词频、位置、句法、上下文语义），未利用读者阅读过程中的注意力分配模式，而眼动研究表明注视模式与认知加工密切相关。
2. **中文学术领域眼动语料稀缺**：已有开源眼动语料（ZuCo、GECO 等）主要来自微博、新闻和小说，缺乏针对学术文献的数据；中文场景下尤其匮乏。
3. **低成本采集技术尚未充分应用于 KPE**：传统眼动设备（如 EyeLink 1000 Plus）价格昂贵且依赖实验室环境，限制了大规模多被试数据采集；基于消费级网络摄像头的方案可行性有待在 KPE 任务中验证。
4. **前作局限于英文社交媒体数据**：Zhang & Zhang (2021) 和 Yan et al. (2024) 使用 ZuCo 语料分别验证了眼动特征对微博 KPE 的提升，但无法覆盖中文学术领域词汇的认知特征。

## 核心贡献（创新点）
1. **轻量级网络摄像头眼动采集平台**：基于 Flask + 开源 SearchGazer 库开发，支持同时采集多被试的实时注视坐标（最高 60Hz），大幅降低硬件门槛与部署成本；与 ZuCo/GECO 等依赖专业设备的工作本质不同，可在普通笔记本电脑上实现。
2. **首个中文图情（LIS）学术眼动语料库 CLIS-ET**：包含 1,215 个句子、64,969 个字符的字符级眼动标注，提取 FFD、FN、TFD 三个核心特征，填补了中文学术阅读眼动数据的空白；与现有语料均以小说/新闻为主源不同，专为学术摘要设计。
3. **系统验证眼动特征对多种 KPE 架构的提升效果**：在 BiLSTM 系与 BERT 系共 8 种模型上分别测试单个/组合眼动特征，发现 FFD 在数据有限时增益显著，FN+TFD 在较大规模数据集上更稳定；与已有仅关注单一模型或单一特征的研究相比，本文提供了跨架构的一致性证据。

## 方法详解
- **数据采集平台**：基于 Flask 构建 Web 界面，前端嵌入 SearchGazer（JavaScript 库）调用笔记本内置摄像头，以 60Hz 采样率采集注视坐标（约 16.67ms 间隔）；每段摘要前执行九点校准，阅读过程中鼠标跟随注视点辅助校准。
- **眼动特征定义**（Table 4）：
  - **FFD（First Fixation Duration）**：首次注视目标字符的持续时间，反映初始阅读认知加工深度；
  - **FN（Fixation Number）**：注视目标字符的总次数，体现重读频率；
  - **TFD（Total Fixation Duration）**：对目标字符的全部注视时间之和，反映累积加工量。
  - 有效注视阈值设为 16.7ms–1500ms（对应 2–91 个采样点），某字符超半数被试无效则特征丢弃。
- **跨被试聚合**：对同一字符在各被试中的眼动指标取均值，形成字符级特征向量。
- **KPE 模型设计**：
  - **RNN 系**（BiLSTM / BiLSTM+CRF / Att-BiLSTM / Att-BiLSTM+CRF）：眼动特征作为外部特征加入输入层；在注意力模型中额外用作 attention indicator 引导序列标注。
  - **预训练语言模型系**（BERT-base-chinese / RoBERTa-wwm-ext / MacBERT / Randeng-T5-784M）：在词/字符嵌入层接入眼动特征，与语言模型表示拼接后进入下游序列标注或生成头。
- **评估方案**：以 F1@K（K=3/5/10）为指标；Abstract-320（小规模）采用五折交叉验证，Abstract-5190（大规模）采用 4:1 单次划分；显著性检验采用双尾配对 t 检验（p < 0.05 标记 *，p < 0.01 标记 **）。

## 实验与结果
- **数据集**：CLIS-ET 含 320 篇摘要（1,215 句/64,969 字，眼动覆盖率 94.69%）；Abstract-5190 含 5,190 篇摘要（15,999 句/804,210 字，眼动覆盖率 54.65%）。
- **基线模型**：BiLSTM、BiLSTM+CRF、Att-BiLSTM、Att-BiLSTM+CRF、BERT、MacBERT、RoBERTa、Randeng-T5（共 8 种架构），均与无眼动特征版本对比。
- **单特征最佳结果**：在 Abstract-320 上，**FFD 对 BERT 提升最大**（F1@5 从 37.04% → 39.87%，+2.83%）；在 Abstract-5190 上，**RoBERTa+FN 取得最高 F1@5 = 44.51%**。
- **组合特征最佳结果**：**FN+TFD 在 Att-BiLSTM+CRF 上达到最优**：Abstract-320 F1@5 = 25.62%（无特征 23.55%，+2.07pp）；Abstract-5190 F1@5 = 36.79%（无特征 33.80%，+2.99pp）。
- **显著性结论**（Table 9）：FFD、FN、TFD 在 BERT/MacBERT/BiLSTM 系中多数设置下 p < 0.05；RoBERTa 显著性较弱，仅 FFD+TFD 等少数组合达显著；Randeng-T5 中 FFD 与 TFD 类特征高度显著。FN+TFD 组合在多个模型中 consistently 显著。
- **核心结论**：眼动特征对 KPE 有稳定正向贡献；小规模数据下早期加工特征（FFD）增益更大，大数据下累积加工特征（TFD/FN）支持更深语义建模；特征组合效果因模型架构和数据规模而异，不存在万能组合。

## 相关工作脉络
1. **ZuCo 语料（Hollenstein et al., 2018）**：英文 Wikipedia 句子的眼动+EEG 多模态语料；本文定位差异在于聚焦中文学术摘要且基于网络摄像头低成本采集。
2. **Zhang & Zhang (2021)**：将 ZuCo 的 TFD 作为 KPE 模型的 ground-truth attention 信号；本文将其扩展至多特征（FFD/FN/TFD）和多架构，并换用中文学术领域。
3. **Yan et al. (2024)**：结合 ZuCo 的眼动与 EEG 特征做微博 KPE；本文差异为去除 EEG、专注纯眼动、针对中文 LIS 学术文本。
4. **GECO / GECO-CN（Cop et al., 2017; Sui et al., 2023）**：小说文本的双语眼动语料；本文弥补了学术文献场景的空白。
5. **AKEGIS（Scholz et al., 2019）**：利用用户搜索日志提取关键词；本文以真实阅读行为替代间接搜索行为，提供更直接的认知信号。
6. **Webcam-based eye-tracking（Kaduk et al., 2024 验证）**：本文在前人验证基础上首次将网络摄像头眼动系统系统化用于 KPE 任务，并开源采集平台。

## 局限性与未来方向
- **网络摄像头精度限制**：60Hz 采样率远低于专业设备（500–2000Hz），难以捕捉快速扫视和微小注视细节，无法替代高精度 oculomotor 研究。
- **被试数量较少**：仅 10 名 LIS 研究生/高年级本科生，统计推断的外部效度受限。
- **领域单一**：语料仅覆盖图情领域，方法推广至其他学科需重新采集验证。
- **字符级聚合丢失个体差异**：跨被试平均处理可能抹平重要个体阅读策略差异。
- **三特征组合未带来单调提升**：FFD+FN+TFD 全组合有时不如双特征组合，特征融合策略有待优化。
- 作者指出未来方向：扩展至更多学术领域、改进追踪精度与验证流程、探索更深层次的心理学洞察与计算模型的融合策略。

## 研究启发与可借鉴点
1. **低成本眼动数据采集可作为认知 NLP 任务的通用范式**：Flask + SearchGazer 的 Web 端方案无需专业设备，适合科研团队在有限预算下快速构建领域眼动语料；可迁移到摘要生成、句法解析等任务。
2. **FFD 在小样本下增益最显著**：实验揭示数据规模与特征类型的交互效应——数据有限时应优先使用 FFD（早期加工特征），数据充足时可侧重 FN/TFD（累积加工特征）；为后续研究中的特征选择提供经验依据。
3. **眼动特征作为注意力机制的引导信号**：在 Att-BiLSTM 中将 FFD/FN/TFD 作为 attention indicator 的设计思路，可为其他序列标注任务（命名实体识别、关系抽取）中引入外部认知信号提供参考。
4. **字符级标注规避分词偏差**：针对中文特点采用字符粒度眼动标注，避免了分词错误对特征对齐的影响；该方法论可直接复用于其他中文 NLP 任务。
5. **跨架构一致性验证的评估框架**：同时在 RNN 系和预训练模型系上测试同一眼动特征，这种"多架构对照"实验设计可有效排除模型特定偏差，建议后续类似工作借鉴。

## 关键术语表
- **Keyphrase Extraction (KPE)**：从文本中自动识别出最能代表其核心主题的短语或关键词序列的 NLP 任务。
- **First Fixation Duration (FFD)**：读者首次注视某个字符的停留时长，反映初始词汇加工难度和认知负荷。
- **Fixation Number (FN)**：对目标字符的总注视次数，次数越多通常表示该字符越关键或理解难度越大。
- **Total Fixation Duration (TFD)**：所有注视目标字符的时间之和，刻画累积加工深度。
- **SearchGazer**：开源 JavaScript 眼动追踪库，通过浏览器调用网络摄像头实时估计注视坐标。
- **CLIS-ET**：Chinese Library and Information Science Eye-Tracking Corpus，本文构建的中文图情学术摘要眼动语料库。
- **BiLSTM+CRF**：双向 LSTM 捕获上下文序列表示，CRF 层建模标签转移约束，广泛用于中文序列标注任务。
- **五折交叉验证（5-fold CV）**：将数据随机分为 5 份，轮流以 4 份训练、1 份测试，取平均性能，适用于小样本数据。

## 可复现要素
- **数据集**：CLIS-ET（320 篇摘要眼动数据）和 Abstract-5190（5,190 篇摘要 KPE 数据）；论文声明开源，访问地址：https://github.com/yan-xinyi/Reading_ET_System（眼动采集平台代码）；数据集具体链接论文未明确给出，需联系作者或查看 GitHub 仓库。
- **代码/权重**：眼动数据采集平台代码已开源（GitHub: yan-xinyi/Reading_ET_System）；KPE 模型代码论文未提供单独链接，需在 GitHub 仓库中查找。
- **关键超参**：
  - max_length = 512；
  - Abstract-320：BiLSTM 系 epochs=30, lr=0.01；注意力系 epochs=65, lr=0.005；预训练模型 epochs=10, lr=5e-5；
  - Abstract-5190：BiLSTM 系 epochs=30, lr=0.003；预训练模型 epochs=8, lr=5e-5；
  - Randeng-T5：epochs=20, early stopping patience=5, lr=5e-5, optimizer=AdamW。
- **硬件环境**：NVIDIA A100 GPU（主要实验）；NVIDIA RTX 4090（Randeng-T5 补充实验）。
