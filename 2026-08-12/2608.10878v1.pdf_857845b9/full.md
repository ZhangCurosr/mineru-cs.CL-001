# X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction

Kaiqi Fu<sup>1</sup>, Rime Wen<sup>1</sup>, Altman Lin<sup>1</sup>, Shawn Qin<sup>1</sup>, Roy Gan<sup>1</sup>, Hao Wang<sup>1</sup>, Qian Wang<sup>1</sup>

<sup>1</sup>X Square Robot

kaiqifu,wanghao@x2robot.com

## Abstract

Accurate and responsive turn-taking is essential for spoken dialogue systems, which must distinguish in real time between user interruptions, backchannels that should be ignored, and the completion of an utterance. Prior modular approaches typically optimize turn state prediction at the utterance or fixed-chunk level, creating a mismatch with the continuous turn state estimate, and often depend on an auxiliary ASR model, which limits responsiveness and increases overall system complexity. Therefore, we present X2-Turn, a frame-synchronous turn state prediction method via delayed-stream modeling. Specifically, building on the pretrained Voxtral Realtime model, we introduce a frame-synchronous turn state head that operates in parallel with the ASR head on shared streaming representations, jointly predicting ASR tokens and fine-grained turn states at the frame level. We evaluate our method on the bilingual Chinese-English Easy-Turn test sets, and the results demonstrate its effectiveness in achieving accurate turn-taking detection while maintaining low latency.

Index Terms: Turn-taking, Spoken Dialogue Systems, Delayed Streams Modeling, Streaming Automatic Speech Recognition

## 1. Introduction

Achieving natural spoken dialogue requires systems to seamlessly handle continuous speech, backchannels, and user interruptions while maintaining low latency [1]. To manage these complex conversational dynamics, a responsive system must continuously estimate fine-grained turn states. These states serve as the foundation for real-time dialogue control, determining when to interrupt text-to-speech (TTS) playback, take the conversational floor, or ignore a user backchannel.

Existing approaches broadly follow either end-to-end or cascaded paradigms. End-to-end full-duplex models [2–6] jointly learn speech understanding, interaction timing, and response generation. For example, Moshi employs a dual-stream architecture that jointly generates text and audio tokens for both speakers, using an inner-monologue mechanism to align semantic and acoustic generation [3]. Although these systems directly model synchronous interactions, jointly optimizing all components on limited full-duplex data may constrain the scale and general capabilities of the dialogue backbone.

Cascaded approaches instead decouple interaction control from response generation [7–12]. A representative example is the VAD–ASR–turn-detection pipeline [13], in which a frontend voice activity detection (VAD) module first segments the continuous user speech stream, an ASR model transcribes each resulting segment, and a semantic turn-detection model determines the turn state from the transcript. Because each stage depends on the output of the preceding stage, this pipeline introduces sequential latency and error propagation. To reduce the dependence on a separate upstream ASR module, subsequent studies have explored tighter integration between ASR and turn detection [8–11]. EasyTurn jointly predicts transcriptions and four turn states from VAD-segmented utterances, replacing the separate downstream turn-detection model with a unified model that reasons over ASR-derived transcripts [8]. JAL-Turn combines frozen SenseVoice and CPC representations to classify hold and shift states at candidate boundaries, whereas FastTurn integrates partial CTC hypotheses with acoustic cues to make low-latency decisions as the transcript is incrementally updated, without waiting for utterance completion [9,10]. SoulX-Duplug further interleaves chunk-level ASR and state tokens within a single autoregressive stream, achieving strong turn-taking performance. However, it still relies on an external ASR model to guide state prediction during inference [11]. Despite these advances, these methods generally operate at the utterance or chunk level rather than continuously estimating the turn state at every frame. This mismatch in temporal granularity limits their responsiveness in real-time interactions.

More recently, Voxtral Realtime introduced a natively streaming ASR architecture based on delayed-stream modeling [3, 14]. It emits transcription tokens synchronously with the input audio at a fixed frame rate of 80 ms [15]. Inspired by this architecture, we introduce a turn state prediction head parallel to the ASR head, with both heads jointly optimized over shared causal decoder representations to predict ASR tokens and turn states simultaneously. To temporally align the two tasks, we propose ASR-anchored supervision, which projects word-level turn annotations onto the frame-level positions of the corresponding ASR tokens. Experiments on the Chinese and English EasyTurn test sets demonstrate that the proposed method achieves an effective trade-off between turn state accuracy and decision latency.

## Our main contributions are summarized as follows:

1. We propose X2-Turn, which extends a pretrained delayedstream ASR model with a parallel turn state head, enabling joint frame-synchronous ASR and turn state prediction within a single streaming forward pass.

2. We design a unified turn state label set that supports interruption, turn completion, and backchannel detection. We further introduce an ASR-anchored supervision method that projects word-level turn annotations onto the frame-level ASR token timeline.

3. We conduct bilingual experiments on the EasyTurn Chinese and English test sets, validating the effectiveness of the proposed method for streaming turn state prediction under different latency settings controlled by τ.

![](images/3db373248171deed02ef615bc832b3d311831bd3e56026fc7b432a63a33fe6f3.jpg)  
Figure 1: Overview ofX2-Turn, the proposedframe-synchronous dual-head architecture, with a target delay of τ = 80 ms. Once the onset of a word has been observed and the target delay has elapsed, the ASR head emits a word-boundary token [W], while the turn state head simultaneously predicts the corresponding state. Subsequent subword tokens are then emittedframe byframe.

## 2. Method

In this section, we first introduce the streaming ASR backbone and dual-head architecture, and then describe the turn state token design and the construction of ASR-anchored turn state labels. An overview of the framework is shown in Fig. 1.

## 2.1. Dual-Head Modeling

Our goal is to estimate turn states synchronously with ASR transcription as speech unfolds, while keeping interaction control independent of the downstream dialogue model. To this end, we build our system on Voxtral Realtime [15], a natively streaming ASR model based on delayed-stream modeling (DSM). Voxtral Realtime maps an input waveform to a delayed token stream using a causal audio encoder and a language decoder. We retain the backbone architecture and introduce a parallel turn state prediction head. This design decouples turn state estimation from response generation, allowing the resulting turn-taking module to be integrated into different dialogue systems.

The architecture of the proposed method is illustrated in Fig. 1. We first briefly review the Voxtral Realtime backbone before presenting our dual-head extension. Voxtral Realtime consists of three main components: a causal audio encoder that maps 16-kHz waveforms to frame-level audio features, a temporal adapter that downsamples the encoder features to 12.5 Hz, and a decoder-only language model that emits one token at each 80-ms step. At step i, the decoder takes as input the sum of the current audio embedding and the text embedding of the previously emitted ASR token $y _ { i - 1 } ^ { \mathrm { a s r } }$ . It then produces a hidden state $h _ { i } .$ , from which the ASR head predicts the next token. The ASR head is trained using token-level cross-entropy over the ASR vocabulary:

$$
\mathcal { L } _ { \mathrm { a s r } } = - \sum _ { i = 1 } ^ { T } \log P _ { \mathrm { a s r } } \left( y _ { i } ^ { \mathrm { a s r } } \mid x _ { \le i } , y _ { < i } ^ { \mathrm { a s r } } \right) ,\tag{1}
$$

where $x _ { \leq i }$ denotes the audio observed up to step i, $y _ { < i } ^ { \mathrm { a s r } }$ denotes the previously emitted ASR tokens, and T is the sequence length.

In addition to ordinary subword tokens, the ASR vocabulary contains two special symbols: a padding token [P] and a word-boundary token [W]. The output token stream is delayed relative to the input audio by a configurable target delay τ. This delay is conditioned into the decoder through AdaRMSNorm, allowing a single model to operate at delays corresponding to different multiples of 80 ms.

We preserve the original ASR prediction head and add a parallel turn state prediction head. Both heads operate on the shared hidden state $h _ { i } ,$ enabling ASR and turn state prediction within a single forward pass. The two heads are jointly optimized using the following objective:

$$
\mathcal { L } _ { \mathrm { t u r n } } = - \sum _ { i = 1 } ^ { T } \log P _ { \mathrm { t u r n } } \left( y _ { i } ^ { \mathrm { t u r n } } \mid x _ { \le i } , y _ { < i } ^ { \mathrm { a s r } } \right) ,\tag{2}
$$

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { a s r } } + \lambda \mathcal { L } _ { \mathrm { t u r n } } , } \end{array}\tag{3}
$$

where $y _ { i } ^ { \mathrm { t u r n } }$ denotes the ground-truth turn state at step i, and λ controls the contribution of the turn state loss. For notational simplicity, the target delay τ is omitted from the conditional distributions above $( \tau = 0 )$

At inference time, autoregressive decoding is driven solely by the ASR head. At each step, the turn head independently predicts the turn state from the shared hidden state $h _ { i } ,$ and its prediction is not fed back into the decoding loop. Therefore, turn state prediction errors do not affect subsequent ASR decoding. Both transcription and turn states are produced within the same forward pass, without introducing an additional sequential inference stage and latency.

## 2.2. Turn-Taking State Token Design

We define five turn state tokens to represent the evolving state of a user turn:

• <|idle|> represents user silence, corresponding to nonspeech segments derived from forced alignment.

• <|noidle|> indicates active speech without semantic content (e.g., the initial syllables of an utterance).

• <|incomplete|> denotes active speech containing partial semantic content.

• <|complete|> signifies active speech with complete semantic content.

• <|backchannel|> captures user backchannel signals or filler words $( \mathrm { e . g . , \tilde { \Omega } u m ^ { 3 } , \tilde { \Omega } ^ { \ast } a h ^ { 3 } } )$ .

Unlike the definition adopted in prior work [11], our $< | \neg \neg \mathrm { i d } 1 \in | >$ state refers specifically to early active speech for which sufficient semantic content has not yet been observed. It does not represent non-speech or background noise. This distinction allows the model to separate true silence from the early portion of an ongoing user utterance.

## 2.3. ASR-Anchored Turn State Supervision

We next describe the construction of paired ASR and turn state targets. We first construct the delayed-stream ASR target sequence. Let the i-th word span the time interval $[ s _ { i } , e _ { i } ] .$ Each word is tokenized into one or more subword tokens, for example, multiple BPE tokens for an English word. Regardless of the number of subword tokens, each word is represented by a single word-boundary token [W] followed by its corresponding subword tokens.

Given the onset time $s _ { i } ,$ the target position $p _ { i }$ of the wordboundary token is computed as

$$
p _ { i } = \mathrm { r o u n d } \left( \frac { s _ { i } } { \Delta } \right) + n _ { \tau } ,\tag{4}
$$

where $\Delta = 8 0$ ms is the frame duration and $n _ { \tau } = \tau / \Delta$ is the target delay measured in frames.

Unlike the original Voxtral formulation, which places [W] according to the word offset, we anchor [W] to the word onset. The word’s subword tokens are placed immediately after [W], while all remaining positions are filled with the padding token [P]. This onset-based placement makes the ASR and turn state predictions available earlier relative to the spoken word. Because each word occupies one [W] position followed by at least one subword position, its representation requires at least two 80-ms steps.

Turn state targets are constructed using the same placement procedure. We first prompt a powerful language model [16] to assign a turn state label to each word. The resulting wordlevel label is then assigned to all target positions occupied by the corresponding [W] token and subword tokens. All unoccupied positions are labeled as <|idle|>, including positions corresponding to pre-speech silence, inter-word pauses, midutterance pauses, and trailing silence. In this way, the ASR and turn state targets are aligned on the same 80-ms discrete timeline.

This ASR-anchored supervision scheme has three main properties. First, each word-level turn state label is placed at the same positions as its corresponding ASR tokens. The turn head therefore predicts the state from the same decoder representations used by the ASR head to emit the associated transcription. Second, silence is explicitly supervised using the $< | \mathrm { ~ i d } 1 \mathrm { e } \mathrm { ~ } | >$ state, allowing a downstream interaction policy to perform endpointing by counting consecutive idle frames following a <|complete|> prediction. Third, because the ASR and turn state targets are aligned position by position rather than at the utterance or chunk level, both tasks are optimized over the same frame-synchronous discrete timeline.

## 3. Experimental setup

## 3.1. Data Preparation

The corpora used in this work consist of two parts: Chinese-English ASR data and turn-taking data. For the ASR data, we use AISHELL 1∼4 [17–20], AliMeeting [21],

WenetSpeech [22], KeSpeech [23], LibriSpeech [24], GigaSpeech [25], TED-LIUM [26], and VoxPopuli [27], totaling approximately 26k hours (14k hours in Chinese and 12k hours in English). This portion of data is used in Stage 1 to strengthen the model’s ASR capability in both languages. For the turntaking data, we select a subset of the EasyTurn training set for Chinese (approximately 126 hours) and a subset of Fisher [28] telephone conversations for English (approximately 249 hours). This portion of data is used in Stage 2 for joint ASR and turntaking modeling.

For both parts of the data, Qwen3-ForceAligner [29] was employed to obtain word-level timestamps. In addition, for the turn-taking data, we use Qwen3.5-Plus as an LLM annotator to perform word-level semantic turn state labeling, following the annotation criteria defined in Section 2.3. Word-level labels are then projected onto the 80 ms frame delayed-stream positions of their ASR word markers [W] and subword tokens.

## 3.2. Implementation Details

We use Voxtral-Mini-4B-Realtime <sup>1</sup> as the pretrained streaming backbone. All experiments fully fine-tune both the causal audio encoder and the language decoder. Training proceeds in two stages, with the ASR streaming delay τ sampled per batch between 1 and 30 frames (80–2400 ms) in both stages, so that a single model covers a range of latency configurations.

During Stage 1, streaming ASR adaptation adapts the backbone to the delayed-stream ASR protocol on a large-scale Chinese–English corpus with frame-level ASR labels. Stage 2 then performs joint ASR and turn state fine-tuning. A turn state head is added, initialized as a copy of the ASR head, and the full model is fine-tuned on the Chinese–English turn-taking training set with paired ASR and frame-level turn state labels. The joint objective uses $\lambda = 0 . 1$

## 3.3. Latency Metric

We measure latency relative to the end of each word. For a word spanning $[ s _ { i } , e _ { i } ]$ , the corresponding turn state becomes available at time s<sub>i</sub>+τ, i.e., after the configured streaming delay τ from the word onset. The resulting latency relative to the end of the word is

$$
\begin{array} { r } { L _ { i } { = } \tau - ( e _ { i } - s _ { i } ) , } \end{array}\tag{5}
$$

which measures how long after the user finishes speaking word i the corresponding turn decision becomes available. If the word duration exceeds τ , $L _ { i }$ can be negative, meaning the state is available before the word ends. We report the average $L _ { i }$ over all words in the EasyTurn test set. For cascaded baselines, following prior work [11], we report inference time plus the frontend VAD delay latency $\tau _ { \mathrm { v a d } }$ . While not strictly identical to $L _ { i }$ this reflects their end-to-end decision delay.

## 4. Results and analysis

We first compare the proposed method with cascaded baselines on the EasyTurn Chinese and English test sets. We then analyze the effect of the streaming delay τ on turn-taking accuracy and latency, and finally compare ASR performance against chunkbased streaming baselines under different training stages.

## 4.1. Main results

Table 1 compares the proposed method (with $\tau \ = \ 4 8 0 m s )$ against baselines on bilingual test sets in terms of turn state classification accuracy and latency. $\mathrm { A C C _ { c o m p / i n c o m p / b c } }$ denotes end-of-utterance accuracy. We compare the last non-idle predicted state against the ground-truth utterance.

Table 1: Turn state classification accuracy and latency on EasyTurn-zh and EasyTurn-en. $\mathrm { A C C } _ { \mathrm { c o m p } } , \mathrm { A C C } _ { \mathrm { i n c o m p } } ,$ and $\mathrm { A C C _ { b c } }$ denote utterance-level accuracyfor the complete, incomplete, and backchannel categories, respectively $( \mathrm { A C C } _ { \mathrm { b c } }$ reportedfor Chinese only, as English has no backchannel split). $^ { 6 \cdot \mathrm { ~ \it ~  ~ } }$ denotes an unsupported or unavailable state, and $\mathrm { l a t e n c y } _ { \mathrm { v a d } }$ denotes thefront-end VAD delay incurred by cascaded methods.
<table><tr><td></td><td>Method</td><td>Streaming</td><td> $\mathbf { A C C } _ { \mathrm { c o m p } } \left( \% , \uparrow \right)$ </td><td> $\mathbf { A C C } _ { \mathrm { i n c o m p } } \ ( \% , \uparrow )$ </td><td> $\mathbf { A C C } _ { \mathrm { b c } } \left( \mathcal { T } _ { O } , \uparrow \right)$ </td><td>Latency (ms, ↓)</td></tr><tr><td rowspan="5">ZH</td><td>Paraformer + TEN Turn</td><td>X</td><td>86.67</td><td>89.30</td><td></td><td> $\mathrm { l a t e n c y } _ { \mathrm { v a d } } + 2 0 4$ </td></tr><tr><td>Smart Turn V3</td><td>X</td><td>91.33</td><td>60.00</td><td></td><td> $\mathrm { \ l a t e n c y { v a d } } + 2 4$ </td></tr><tr><td>EasyTurn</td><td>X</td><td>96.33</td><td>97.67</td><td>91.00</td><td>latencyvad + 263</td></tr><tr><td>SoulX-Duplug</td><td>√</td><td>77.67</td><td>88.96</td><td></td><td>295</td></tr><tr><td>X2-Turn (Ours)</td><td>√</td><td>91.00</td><td>93.00</td><td>96.00</td><td>288</td></tr><tr><td rowspan="4">EN</td><td>SenseVoice En + TEN Turn</td><td>X</td><td>95.60</td><td>76.59</td><td></td><td>latencyvad + 57</td></tr><tr><td>Smart Turn V3</td><td>X</td><td>78.93</td><td>72.24</td><td></td><td> $\mathrm { l a t e n c y } _ { \mathrm { v a d } } + 2 1$ </td></tr><tr><td>SoulX-Duplug</td><td>√</td><td>89.33</td><td>79.33</td><td></td><td>205</td></tr><tr><td>X2-Turn (Ours)</td><td>√</td><td>92.10</td><td>84.60</td><td></td><td>225</td></tr></table>

Among fully streaming baseline systems, e.g., SoulX-Duplug, the proposed method consistently outperforms the SoulX-Duplug baseline while remaining highly competitive in latency. Moreover, our approach requires no auxiliary, separately optimized ASR model at inference time. A single model jointly decodes reliable ASR transcripts and turn states in real time. This confirms that jointly modeling streaming ASR and turn state prediction improves turn state prediction without sacrificing real-time responsiveness.

For cascaded, VAD-dependent baselines, real-world deployment typically requires segmenting audio with an external VAD module before inference can begin. This pre-inference segmentation step adds non-trivial latency [7]. Consequently, the apparent high accuracy of cascaded systems does not translate into a real-world responsiveness advantage; their end-toend latency remains both higher and less predictable. The proposed method, in contrast, matches the accuracy of the strongest cascaded system while operating fully streaming, yielding a markedly better accuracy–latency trade-off across both languages.

## 4.2. Effect of streaming delay τ for Turn-taking

We report the effect of different values of the streaming delay τ on both turn-taking and ASR performance. We first examine the effect of τ on turn-taking performance, considering three configurations (320, 400 and 480 ms), as turn-taking inherently favors low-latency settings.

Table 2: Ablation on the delay τ on the EasyTurn testsets.
<table><tr><td></td><td>Delay τ (ms)</td><td> $\mathrm { A C C } _ { \mathrm { c o m p } } \left( \% , \uparrow \right)$ </td><td> $\mathrm { A C C } _ { \mathrm { i n c o m p } } \ : ( \% , \uparrow )$ </td><td>Avg. (%, ↑)</td><td>Latency (ms, ↓)</td></tr><tr><td rowspan="3">ZH</td><td>480</td><td>91.00</td><td>93.00</td><td>92.00</td><td>288</td></tr><tr><td>400</td><td>88.70</td><td>94.30</td><td>91.50</td><td>208</td></tr><tr><td>320</td><td>87.33</td><td>94.00</td><td>90.67</td><td>120</td></tr><tr><td rowspan="3">EN</td><td>480</td><td>92.10</td><td>84.60</td><td>88.49</td><td>225</td></tr><tr><td>400</td><td>85.20</td><td>85.30</td><td>85.25</td><td>145</td></tr><tr><td>320</td><td>82.70</td><td>87.60</td><td>85.09</td><td>65</td></tr></table>

Table 2 shows a consistent latency–accuracy trade-off across both languages. As τ decreases from 480 ms to 320 ms, latency drops substantially, while turn state accuracy degrades only mildly. This indicates that our model is robust under tight latency constraints, allowing a suitable operating point to be chosen for different latency requirements with little loss in turntaking accuracy.

Table 3: ASR performance comparison between chunk-based streaming baselines and our frame-synchronous model across τ configurations and training stages. Uni-ASR uses a 320 ms chunk with beam search; Freeze-Omni uses a chunk size of4.
<table><tr><td rowspan="2">Test Set</td><td rowspan="2">Uni-ASR [30]</td><td rowspan="2">Freeze-Omni [4]</td><td colspan="2">Stage1-ASR</td><td rowspan="2">Stage2-Turn 480ms</td></tr><tr><td>480 ms</td><td>2400 ms</td></tr><tr><td>AISHELL-1</td><td>2.90</td><td>2.79</td><td>2.57</td><td>1.48</td><td>3.94</td></tr><tr><td>test-meeting</td><td></td><td>14.2</td><td>9.25</td><td>7.68</td><td>12.18</td></tr><tr><td>test-net</td><td></td><td>12.6</td><td>9.50</td><td>8.40</td><td>10.39</td></tr><tr><td>GigaSpeech</td><td></td><td></td><td>12.23</td><td>10.87</td><td>12.55</td></tr><tr><td>LS-clean</td><td>3.21</td><td>4.05</td><td>2.40</td><td>1.54</td><td>3.30</td></tr><tr><td>LS-other</td><td>7.71</td><td>10.48</td><td>5.87</td><td>3.77</td><td>8.53</td></tr></table>

## 4.3. Streaming ASR Performance Comparison

In this subsection, we compare our frame-synchronous approach against representative chunk-based streaming ASR systems. Since SoulX-Duplug does not report ASR results, we instead select two strong chunk-based streaming baselines, Uni-ASR and Freeze-Omni, for comparison. Table 3 summarizes the results. Under comparable or lower delay settings, Stage1- ASR at $\tau { = } 4 8 0$ ms already outperforms both baselines across nearly all evaluation sets. This indicates that frame-wise prediction with a short lookahead is competitive with, and often superior to fixed-chunk streaming even before accounting for its lower latency. Even after Stage2-Turn joint training, where turn-state supervision introduces a set-dependent degradation relative to Stage1-ASR at the same delay, our model still matches or exceeds Freeze-Omni. This suggests that the framesynchronous backbone retains most of its recognition advantage over chunk-based streaming even under multi-task optimization.

## 5. Conclusion

This paper presents X2-Turn, a frame-synchronous dual-head extension of a pretrained delayed-stream ASR model for joint streaming ASR and turn state prediction. A parallel turn state head shares causal decoder representations with the ASR head, with ASR-anchored supervision projecting word-level turn labels onto the native 80 ms token timeline. The streaming delay τ offers a controllable trade-off between turn-taking accuracy, response latency, and ASR quality. Experiments on the bilingual EasyTurn test sets that X2-Turn achieves accurate turntaking detection while maintaining low latency. Future work will further balance the ASR and turn state objectives and improve robustness in more challenging conversational settings.

## 6. References

[1] G. Skantze, “Turn-taking in conversational systems and humanrobot interaction: a review,” Computer Speech & Language, vol. 67, p. 101178, 2021.

[2] T. A. Nguyen, E. Kharitonov, J. Copet, Y. Adi, W.-N. Hsu, A. Elkahky, P. Tomasello, R. Algayres, B. Sagot, A. Mohamed et al., “Generative spoken dialogue language modeling,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 250–266, 2023.

[3] A. Defossez, L. Mazar´ e, M. Orsini, A. Royer, P. P´ erez, H. J´ egou,´ E. Grave, and N. Zeghidour, “Moshi: a speech-text foundation model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024.

[4] X. Wang, Y. Li, C. Fu, Y. Zhang, Y. Shen, L. Xie, K. Li, X. Sun, and L. MA, “Freeze-omni: A smart and low latency speechto-speech dialogue model with frozen LLM,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=s1EImzs5Id

[5] Q. Zhang, L. Cheng, C. Deng, Q. Chen, W. Wang, S. Zheng, J. Liu, H. Yu, C.-H. Tan, Z. Du et al., “Omniflatten: An end-toend gpt model for seamless voice conversation,” in Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2025, pp. 14 570–14 580.

[6] R. Roy, J. Raiman, S.-g. Lee, T.-D. Ene, R. Kirby, S. Kim, J. Kim, and B. Catanzaro, “Personaplex: Voice and role control for full duplex conversational speech models,” in ICASSP 2026- 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 16 137–16 141.

[7] B. Liao, Y. Xu, J. Ou, K. Yang, W. Jian, P. Wan, and D. Zhang, “Flexduo: A pluggable system for enabling fullduplex capabilities in speech dialogue systems,” arXiv preprint arXiv:2502.13472, 2025.

[8] G. Li, C. Wang, H. Xue, S. Wang, D. Gao, Z. Zhang, Y. Lin, W. Li, L. Xiao, Z. Fu et al., “Easy turn: Integrating acoustic and linguistic modalities for robust turn-taking in full-duplex spoken dialogue systems,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 16 957–16 961.

[9] G. Yang, Y. Pan, S. Qiu, and N. Bai, “Jal-turn: Joint acousticlinguistic modeling for real-time and robust turn-taking detection in full-duplex spoken dialogue systems,” arXiv preprint arXiv:2603.26515, 2026.

[10] C. Wang, H. Xue, C. He, J. Hu, S. Wang, B. Wu, Y. Ji, J. Zheng, R. Chen, Z. Zhu et al., “Fastturn: Unifying acoustic and streaming semantic cues for low-latency and robust turn detection,” arXiv preprint arXiv:2604.01897, 2026.

[11] R. Yan, W. Chen, Z. Liu, Z. Ma, H. Lin, H. Wen, H. Xie, J. Wu, Y. Liang, Y. Zhao et al., “Soulx-duplug: Plug-and-play streaming state prediction module for realtime full-duplex speech conversation,” arXiv preprint arXiv:2603.14877, 2026.

[12] JD.com, “JoyAI-Talker: Full-duplex speech interactive large model built for empathetic voice agents,” arXiv preprint arXiv:2608.01119, 2026.

[13] T. Team, “Ten vad: A low-latency, lightweight and highperformance streaming voice activity detector (vad),” https://github.com/TEN-framework/ten-vad.git, 2025.

[14] N. Zeghidour, E. Kharitonov, M. Orsini, V. Volhejn, G. de Marmiesse, E. Grave, P. Perez, L. Mazar´ e, and´ A. Defossez, “Streaming sequence-to-sequence learning with de-´ layed streams modeling,” Tech. Rep., 2025. [Online]. Available: https://arxiv.org/abs/2509.08753

[15] A. H. Liu, A. Ehrenberg, A. Lo, C.-Y. Sun, G. Lample, J.-M. Delignon, K. R. Chandu, P. von Platen, P. R. Muddireddy, R. Arora et al., “Voxtral realtime,” arXiv preprint arXiv:2602.11298, 2026.

[16] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[17] H. Bu, J. Du, X. Na, B. Wu, and H. Zheng, “Aishell-1: An open source mandarin speech corpus and a speech recognition baseline,” in 2017 20th conference ofthe oriental chapter ofthe international coordinating committee on speech databases and speech I/O systems and assessment (O-COCOSDA). IEEE, 2017, pp. 1–5.

[18] J. Du, X. Na, X. Liu, and H. Bu, “Aishell-2: Transforming mandarin asr research into industrial scale,” arXiv preprint arXiv:1808.10583, 2018.

[19] Y. Shi, H. Bu, X. Xu, S. Zhang, and M. Li, “Aishell-3: A multispeaker mandarin tts corpus and the baselines,” arXiv preprint arXiv:2010.11567, 2020.

[20] Y. Fu, L. Cheng, S. Lv, Y. Jv, Y. Kong, Z. Chen, Y. Hu, L. Xie, J. Wu, H. Bu et al., “Aishell-4: An open source dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario,” arXiv preprint arXiv:2104.03603, 2021.

[21] F. Yu, S. Zhang, Y. Fu, L. Xie, S. Zheng, Z. Du, W. Huang, P. Guo, Z. Yan, B. Ma et al., “M2met: The icassp 2022 multichannel multi-party meeting transcription challenge,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 6167–6171.

[22] B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng et al., “Wenetspeech: A 10000+ hours multidomain mandarin corpus for speech recognition,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 6182–6186.

[23] Z. Tang, D. Wang, Y. Xu, J. Sun, X. Lei, S. Zhao, C. Wen, X. Tan, C. Xie, S. Zhou et al., “Kespeech: An open source speech dataset of mandarin and its eight subdialects,” in Thirty-fifth conference on neural information processing systems datasets and benchmarks track (Round 2), 2021.

[24] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an asr corpus based on public domain audio books,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015, pp. 5206–5210.

[25] G. Chen, S. Chai, G. Wang, J. Du, W.-Q. Zhang, C. Weng, D. Su, D. Povey, J. Trmal, J. Zhang et al., “Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio,” arXiv preprint arXiv:2106.06909, 2021.

[26] F. Hernandez, V. Nguyen, S. Ghannay, N. Tomashenko, and Y. Esteve, “Ted-lium 3: Twice as much data and corpus repartition for experiments on speaker adaptation,” in International conference on speech and computer. Springer, 2018, pp. 198–208.

[27] C. Wang, M. Riviere, A. Lee, A. Wu, C. Talnikar, D. Haziza, M. Williamson, J. Pino, and E. Dupoux, “Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 993–1003.

[28] C. Cieri, D. Miller, and K. Walker, “The fisher corpus: A resource for the next generations of speech-to-text.” in LREC, vol. 4, 2004, pp. 69–71.

[29] X. Shi, X. Wang, Z. Guo, Y. Wang, P. Zhang, X. Zhang, Z. Guo, H. Hao, Y. Xi, B. Yang et al., “Qwen3-asr technical report,” arXiv preprint arXiv:2601.21337, 2026.

[30] Y. Xia, J. Tang, J. Hou, G. Xu, and H. Yao, “Uni-asr: Unified llm-based architecture for non-streaming and streaming automatic speech recognition,” arXiv preprint arXiv:2603.11123, 2026.