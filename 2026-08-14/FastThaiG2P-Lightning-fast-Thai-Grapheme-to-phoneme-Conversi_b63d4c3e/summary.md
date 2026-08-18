---
title: "FastThaiG2P-Lightning-fast-Thai-Grapheme-to-phoneme-Conversi"
source: https://arxiv.org/pdf/2608.12814v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:42:45"
field: "低资源语音合成"
keywords: ["Thai G2P", "grapheme-to-phoneme", "text-to-speech", "low-latency", "CPU inference", "Kokoro-TTS", "text normalization"]
innovations: ["62,112词IPA词典结合LLM生成与人工校验", "正则无界缓存优化实现15倍延迟降低（0.15ms/句）", "Kokoro-82M音素映射扩展支持泰语5调系统"]
benchmarks: ["27,242句合成泰语语料", "Som-TTS 20小时"]
---

# 论文速读：FastThaiG2P: Lightning-fast Thai Grapheme-to-phoneme Conversion for Voice Agent Pipelines

## 一句话总结
本文提出了 FastThaiG2P，一个面向实时语音助手流水线的超低延迟泰语 G2P 工具，结合 62,112 词 IPA 词典、文本归一化规则和基于 TLTK 的 OOV 回退，实现了平均 0.15 ms/句的推理延迟；并以此训练了 Thai Kokoro-82M TTS 模型，在 CPU 上达到 4× 实时率。

## 研究问题与动机
- 泰语正字法复杂：无显式词边界、声调系统（5 个词调）依赖音节结构、元音为不连续字符、存在大量非透明映射规则，字符级规则无法独立完成 G2P。
- 现有工具各有缺陷：TLTK 每调用重编译正则，延迟高达 2 ms/句；Epitran 丢弃声调标记且需预分词；thai-g2p 和 CharsiuG2P 均需外部分词或缺少文本归一化，且神经模型推理延迟远高于词典查询。
- 端到端图形化 TTS 模型（MMS-TTS、ThonburianTTS、VoxCPM 等）参数量大（336M–2B），需要 GPU 推理，不适用于边缘设备或成本敏感的低延迟场景。
- 语音助手流水线对 G2P 阶段有严苛的延迟预算（整条 TTS 管线仅 500–1,000 ms），需要同时具备高覆盖率、综合文本归一化和亚毫秒延迟的前端。

## 核心贡献（创新点）
1. **62,112 词 IPA 词典**：整合 Wiktionary、LLM（Claude Opus 4.6）生成转录和人工覆盖三源数据，按优先级合并，fatal error rate 仅 1.8%；与既有工作相比，首次为泰语构建了覆盖量级最大且经过语音学约束校验的 IPA 词典。
2. **15 类文本归一化管道**：覆盖数字、缩写、单位、符号、时间、电话号码、电子邮件等不可发音文本，确保 LLM 失败时仍能为 TTS 提供合法输入；区别于已有工具（Epitran/thai-g2p 均无此模块），本文归一化是确定性流水线且顺序设计避免歧义。
3. **15 倍延迟优化**：通过缓存未限制的正则表达式（解决 Python 512 项 LRU 缓存溢出问题）和导入时预加载音节规则，将 G2P 延迟从 ~2 ms 降至 0.15 ms/句；本质区别在于针对 TLTK 原始实现的运行时瓶颈做了结构性工程改造而非算法替换。
4. **端到端 CPU TTS 演示**：将 FastThaiG2P 输出的 IPA 映射至 Kokoro-82M 音素集（复用未使用的 token ID 170 表示高调），在 Som-TTS（20 小时）上微调后实现 0.25 RTF 的 CPU 推理；与参数庞大的端到端泰语 TTS 相比，本方案仅 330 MB、无需 GPU。

## 方法详解
**四阶段流水线架构**：文本归一化 → 分词 → 词典查询 → OOV 回退 G2P。

- **文本归一化**：按固定顺序处理 15 类不可发音文本（展开 maiyamok 重复标记→泰语数字转阿拉伯数字→读取邮箱→英语缩写/品牌名音译→单位→符号→时间→电话号码→字母数字标识符→千分位数字→小数→泰语缩写→残留拉丁字符），短数字（≤6 位）按泰语位值读取，长数字逐位读取。
- **分词**：使用 PyThaiNLP `newmm`（最大匹配）引擎，搭配自定义词典（`data/dict.txt` 和 `data/ipa.json`），词典同时作为分词词表和 IPA 查找键，确保每个 token 均有对应 IPA（除 OOV）。
- **词典构建**：三源合并，优先级：人工覆盖 > Wiktionary（约 13,000 条）> LLM 生成（约 49,000 条，Claude Opus 4.6，Amazon Bedrock，每批 100 词）；LLM 输出经 38 codepoint 音素白名单验证，含非法字符或缺少斜杠分隔符的条目被拒绝；prompt 在 500 条 Wiktionary 验证集上达到 78.8% 精确匹配率、1.8% fatal error rate。
- **OOV 回退**：借用 TLTK 规则引擎，trigram 统计分音节→按声母类/元音模式/声调规则映射为 romanization→转为 IPA（Chao 调号、送气、不除阻塞音）；返回 phonologically valid IPA 但可能不适用于不规则词。
- **正则缓存优化**：在 fallback 的三个热点循环（音节解析、音节枚举、规则加载）引入无界显式正则缓存，结合 import-time 预加载，消除 Python 512 项 LRU 缓存溢出导致的重复编译。
- **IPA-to-Kokoro 映射**：将 5 调 Chao 调号映射至 Kokoro 音素集（中调→`↓`，低调→`↓`，降调→`V`，高调→复用未用 token ID 170 `↑`，升调→`7`）；去除仅 IPA 存在的附加符号， affricates `tʃ`/`tʃh` 替换为预组合形式，ASCII `g` 规范化为 IPA `ɡ`。

## 实验与结果
- **数据集**：27,242 句合成泰语句子（涵盖对话文本、数字、缩写、领域词汇），OOV rate 仅 0.5%；TTS 训练使用 Som-TTS（约 20 小时单说话人录音）。
- **G2P 延迟结果**：平均 0.15 ms/句，吞吐量 6,600 句/秒；对比 TLTK 基线（~2 ms/句）提升约 15 倍。
- **延迟分解**：分词 30%、归一化 12%、OOV 回退 58%（仅 0.5% OOV）、词典查询 <1%。
- **TTS 结果**：Thai Kokoro-82M（基于 StyleTTS 2 架构，82M 参数，kikiri-tts 两阶段训练 recipe）在 ONNX/CPU 上实现 RTF 0.25（4× 实时），总内存足迹 330 MB，24 kHz 采样率，最大输入 510 音素；音频质量适合原型开发和测试。
- **最强结果**：0.15 ms/句 G2P 延迟（当前最低之一）；4× RTF 的 CPU TTS 推理。

## 相关工作脉络
- **TLTK**：trigram 音节分段+规则映射，无预建词典，每调用重编译正则导致高延迟；本文在其基础上引入无界正则缓存实现 15× 加速。
- **Epitran**：61 语言的字符级 IPA 映射，但丢弃泰语声调、需预分词、无文本归一化；本文同时提供声调推断和完整归一化管线。
- **thai-g2p**：基于 MarianMT 的 seq2seq 模型，需外部分词，无归一化；本文无需神经网络推理，延迟更低。
- **CharsiuG2P**：ByT5 多语言神经 G2P，PER 26.9%/WER 59%，需预分词且推理延迟高于词典查询；本文以词典为主、规则为辅，延迟低两个数量级。
- **Kokoro-82M**：原为英文预训练的 82M 参数 TTS 模型；本文通过音素映射扩展支持泰语，实现低资源语言适配。
- **StyleTTS 2 / Piper / MMS-TTS**：端到端图形化 TTS 范式；本文定位为低延迟 CPU 场景的音素驱动替代方案，参数和算力需求远低于这些模型。

## 局限性与未来方向
- 词典覆盖 gaps：区域词汇、新借词、俚语、泰英混合代码转换文本覆盖不足。
- OOV 回退质量：基于规则的回退对不规则词（巴利语/梵语复合词静默字母、非标准借词发音）可能出错；可考虑规则+小型神经模型的混合方案。
- 声调准确性：标准中部泰语规则外存在声调标记省略、方言差异等例外，当前未建模区域变体。
- TTS 质量评估：未进行正式 MOS 主观评测，缺少与商业泰语 TTS 系统的对比。
- 未来方向：作为更强大 CPU-only 音素 TTS 系统的使能组件，支撑混合型语音助手（复杂任务走云端大模型，ASR/TTS 下放到本地设备）。

## 研究启发与可借鉴点
1. **正则缓存优化策略**：针对 Python `re` 模块 512 项 LRU 缓存限制，引入无界显式缓存是低资源场景下大幅提升规则引擎吞吐的通用技巧，可迁移至其他语言的 G2P/归一化系统。
2. **LLM + 规则校验的词典构建 pipeline**：用 LLM 批量生成音素转录、再用音素白名单和 prompt 验证过滤、人工覆盖兜底，是一种可扩展的低资源语言语音资源构建范式。
3. **音素集的跨语言复用与扩展**：将已有英文 TTS 模型（Kokoro）的音素表通过 token 复用（如复用未用 ID 170 表示高调入声调对比）扩展至新语言，比从头训练更节省数据和算力。
4. **归一化管线的确定性设计**：按固定顺序处理多类不可发音文本以避免歧义，适用于任何需要预处理非规范文本的语音/文本流水线。
5. **延迟剖析驱动的工程优化**：通过 cProfile 发现 OOV 回退虽只触发 0.5% 但占 58% 时间的"长尾热点"，针对性优化可带来整体显著收益，值得在其他预处理模块中复用。

## 关键术语表
**G2P（Grapheme-to-Phoneme）**：将书面字符序列转换为音素序列的前端处理模块，是音素驱动 TTS 的关键步骤。
**Chao tone letters**：赵元任发明的声调标号系统，用数字上标（如 ˥˧）表示五度制调值，本文用于标注泰语五个词调。
**OOV（Out-of-Vocabulary）**：词典中未收录的词汇，本文通过基于 TLTK 的规则回退系统为其生成 IPA 转录。
**RTF（Real-Time Factor）**：推理耗时与音频时长的比值，RTF 0.25 表示推理速度为实时率的 4 倍。
**IPA（International Phonetic Alphabet）**：国际音标，本文采用符合 English Wiktionary 泰语转写规范的 IPA 约定。
**newmm**：PyThaiNLP 的最大匹配分词引擎（maximum matching with new algorithm），本文用作泰语分词器。
**Kokoro-82M**：hexgrad 发布的 82M 参数开源 TTS 模型，基于 StyleTTS 2 架构，本文将其扩展至泰语。
**Som-TTS**：PyThaiNLP 发布的开源泰语 TTS 数据集，约 20 小时单说话人录音及对齐文本。

## 可复现要素
- **数据集**：Som-TTS（公开，约 20 小时）；27,242 句合成评测语料（`data/synthetic/utterances.jsonl`）。
- **代码**：开源，Apache-2.0 协议，GitHub: github.com/aws/FastThaiG2P。
- **权重**：Thai-finetuned Kokoro-82M checkpoint 可通过 kikiri-tts recipe 复现；Kokoro-82M 原始权重来自 HuggingFace（hexgrad/Kokoro-82M）。
- **关键超参**：词典规模 62,112 词；LLM 生成批次大小 100 词；音素白名单 38 codepoints；TTS 采样率 24 kHz；最大输入 510 音素；ONNX 导出用于 CPU 推理。
- **环境**：Python 3.11，单线程 CPU 评测。
