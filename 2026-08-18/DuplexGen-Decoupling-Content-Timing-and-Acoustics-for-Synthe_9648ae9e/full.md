# DuplexGen: Decoupling Content, Timing, and Acoustics for Synthetic Dialogue Speech

Pengcheng Wang<sup>∗</sup>, Sheng Li<sup>∗</sup>, Jiyi Li<sup>†</sup>, and Takahiro Shinozaki<sup>∗</sup>

Department of Information and Communications Engineering,

Institute of Science Tokyo, Yokohama, Japan

<sup>†</sup> Hokkaido University, Hokkaido, Japan

Abstract—Synthetic conversational speech has become an important resource for developing and evaluating conversational speech systems. However, existing dialogue synthesis pipelines typically generate dialogue content first and then insert interruptions, overlap, and backchannels using handcrafted markers or timing rules, making conversational timing prescribed rather than interaction-driven. We present DuplexGen, a dialogue synthesis framework that explicitly decouples content, timing, and acoustics. An LLM first generates the dialogue script, and then two full-duplex conversational models perform the script while listening to each other in real time. This allows conversational timing to emerge naturally while preserving the scripted content. Finally, a high-fidelity text-to-speech model re-renders the interaction without altering its timing. As a demonstration of the proposed framework, we construct a patient–clinician conversational speech corpus with construction-time annotations, including word timestamps, speaker activity, overlap regions, and interaction events. Experimental results show that the proposed framework produces conversational dynamics closer to the real-dialogue reference distribution than the stitching baselines evaluated in this study.

Index Terms—conversational speech synthesis, full-duplex dialogue, turn-taking, overlapping speech, data augmentation, conversational ASR

## I. INTRODUCTION

Synthetic conversational speech has become an increasingly important resource for developing conversational speech recognition and speaker diarization systems. Recent progress in large language models (LLMs) and neural text-to-speech (TTS) has made it possible to generate large-scale dialogue corpora automatically, significantly reducing the cost of collecting human conversations. However, many existing pipelines still treat dialogue generation as a single process: the dialogue and timing are generated together, while overlap, interruption, and backchannel behaviors are subsequently introduced through markers or handcrafted timing rules [1], [2]. As a result, conversational dynamics are largely prescribed by the generation pipeline instead of emerging from interaction.

Recent full-duplex conversational models naturally generate conversational interaction through continuous listening while speaking. However, these models generate conversations freely and cannot reliably follow a predefined dialogue script. Existing approaches often emphasize either controllable content or emergent interaction, leaving their combination relatively underexplored.

Our hypothesis is that dialogue generation should be decomposed into three independent processes: semantic generation, interaction generation, and acoustic realization. We address this limitation by explicitly decoupling dialogue generation into three stages: content, timing, and acoustics. An LLM determines only the dialogue content and speaker order. Two fullduplex conversational models then perform the script while listening to each other in real time, allowing conversational dynamics to emerge naturally without altering the scripted dialogue. Finally, a Conditional TTS system re-renders the generated interaction while preserving the timing. This separation ensures that each stage is responsible for exactly one aspect of the conversation.

We demonstrate this framework as MedDialSpeech for patient–clinician dialogue generation. Compared with the evaluated stitching baselines, the proposed approach produces conversational dynamics closer to real clinical conversations while maintaining exact script fidelity and high-quality speech synthesis. Besides, the generation process naturally produces aligned transcripts, speaker activity and interaction events, which makes the resulting corpus directly applicable to conversational ASR and speaker diarization research.

The main contributions of this work are as follows:

• We propose a dialogue synthesis framework that explicitly decouples dialogue content, conversational dynamics, and acoustic rendering.

• We introduce a script-constrained duplex decoding strategy that enforces exact lexical constraints during duplex generation while allowing overlap, interruptions, and backchannels to emerge naturally through real-time interaction.

• We construct an interaction-aware patient-clinician conversational speech corpus with construction-time annotations, and demonstrate that the proposed framework closer to the reference distribution than the evaluated stitching baselines.

## II. METHOD

Fig. 1 shows the full pipeline. Its organizing principle is that content, timing, and acoustics are produced by three separate stages that never share a decision: the script fixes content, the duplex performance fixes timing, and re-rendering fixes acoustics.

![](images/fa38ab73d819464500b8a08ef4418b6d15f61b4c1ebfc455609d132086e20e6c.jpg)  
Fig. 1. Overview of the proposed dialogue synthesis framework. An LLM first generates the dialogue script (content). Two full-duplex conversational models then perform the script through real-time interaction, allowing conversational dynamics to emerge naturally. Finally, a high-fidelity TTS system re-renders the interaction while preserving the generated dynamics.

## A. Overview and Notation

The proposed framework separates dialogue generation into three independent stages, each responsible for one aspect of the conversation. An LLM generates the dialogue script (content), two full-duplex conversational models perform the script through real-time interaction (timing), and a high-fidelity TTS system re-renders the interaction (acoustics). The three stages are strictly decoupled: the script determines only what is spoken, duplex interaction determines only when it is spoken, and re-rendering determines only how it sounds.

Let the dialogue script consist of an ordered sequence of turns assigned to two speakers, $i \in \{ A , B \}$ . Concatenating all turns belonging to speaker i gives a target token sequence

$$
s ^ { i } = ( s _ { 1 } ^ { i } , \ldots , s _ { K _ { i } } ^ { i } ) ,
$$

with a cursor $p _ { i }$ indicating the next token to be spoken. During duplex generation, both speakers continuously receive each other’s audio and independently decide whether to emit the next script token or remain silent. The transcript is therefore fixed by the script, whereas the sequence of emit-or-wait decisions determines the conversational timing.

## B. Script-Constrained Duplex Generation

Dialogue content is fixed by constraining the lexical output of the full-duplex model. Following the inner-monologue architecture of Moshi [3], the model predicts a text token before generating the corresponding audio frame. At each frame, we intercept this prediction and restrict the candidate vocabulary to

$$
\begin{array} { r } { \mathcal C _ { t } ^ { i } = \left\{ \begin{array} { l l } { \{ \mathrm { P A D } \} \cup \{ s _ { p _ { i } } ^ { i } \} , } & { i \mathrm { ~ i s ~ f l o o r \mathrm { - } e l i g i b l e } , } \\ { \{ \mathrm { P A D } \} \cup \mathcal B , } & { i \mathrm { ~ i s ~ a ~ l i s t e n e r } , } \end{array} \right. } \end{array}\tag{1}
$$

where $s _ { p { i } } ^ { i }$ is the next script token and B is a fixed whitelist of listener backchannels (e.g., mm-hmm, yeah, right).

At each frame, the model selects

$$
w _ { t } ^ { i } = \arg \operatorname* { m a x } _ { w \in \mathcal { C } _ { t } ^ { i } } P ( w \mid h _ { t } ^ { i } ) .
$$

If $w _ { t } ^ { i } = s _ { p _ { i } } ^ { i }$ , the corresponding audio is emitted and the cursor is advanced,

$$
p _ { i } \gets p _ { i } + 1 .
$$

If $w _ { t } ^ { i } = \mathbb { P } \mathbb { A } \mathbb { D }$ , the speaker remains silent for the current frame. If $w _ { t } ^ { i } \in \mathbb { B } ,$ , a listener backchannel is produced subject to the refractory constraint introduced in Section II-C.

The constraint removes lexical uncertainty from duplex generation. Whenever a speaker decides to speak, the emitted word is uniquely determined by the script. Consequently, the duplex model no longer decides what to say; it decides only whether to emit the next script token at the current frame or to wait. Dialogue content is therefore fixed by construction, while conversational timing remains completely determined by the interaction between the two duplex instances.

## C. Emergent Timing

Conversational timing is generated through the interaction between two cross-listening full-duplex models. At every frame, each instance receives the other speaker’s audio and independently decides whether to emit the next script token or remain silent. No global schedule specifies when a turn begins or ends. Instead, the interaction is regulated by three control parameters.

The first parameter is the handoff window $W _ { h }$ , which defines the interval around the projected end of the current turn during which the listener is allowed to take the floor:

$$
\mathrm { h a n d o f f \ a l l o w e d \iff } t \in \left[ \tau _ { \mathrm { e n d } } - \frac { W _ { h } } { 2 } , \tau _ { \mathrm { e n d } } + \frac { W _ { h } } { 2 } \right] ,\tag{2}
$$

where $\tau _ { \mathrm { e n d } }$ is estimated from the remaining script tokens of the current speaker.

The second parameter, max\_padding $( P _ { \operatorname* { m a x } } )$ , limits the maximum number of consecutive PAD frames before a flooreligible speaker is forced to begin its queued turn, preventing unrealistically long silent gaps.

The third parameter, bc\_refractory $( R _ { b c } )$ , specifies the minimum interval between two listener backchannels, preventing excessive acknowledgements.

These parameters regulate only the frequency of interaction events. The precise onset of a floor transition, the duration of an overlap, and the attachment point of a backchannel are determined online by the two duplex models in response to each other’s audio. Conversational timing therefore emerges from interaction rather than from an explicit schedule.

## D. Fidelity Re-rendering

The duplex models generate a symbolic interaction score rather than the final waveform. For each conversational event, the score records the active speaker, turn boundaries, overlap regions, and backchannel positions. This symbolic representation is subsequently rendered into high-fidelity speech using CosyVoice [4].

Since the duplex model and the TTS system generally speak at different rates, absolute timestamps cannot be copied directly. Instead, timing is transferred using relative positions.

For a silent transition, the silence duration

$$
g = \frac { a _ { u + 1 } - b _ { u } } { f }
$$

is preserved directly so that the rendered turn begins $g$ seconds after the previous rendered turn ends.

For an overlapped transition, the relative entry position is computed as

$$
e = \frac { a _ { u + 1 } - a _ { u } } { b _ { u } - a _ { u } } ,\tag{3}
$$

where e denotes the fraction of the current turn that has elapsed when the next speaker begins. The rendered overlap is reproduced by placing the onset of turn $u + 1$ at $e D _ { u } ,$ where $D _ { u }$ is the rendered duration of turn u. Backchannels are mapped identically according to their relative position within the hosting turn. Consequently, overlap structure is preserved regardless of differences in speaking rate between the duplex model and the renderer.

Three implementation details further improve rendering quality. A sliding prompt maintains speaker consistency throughout long conversations. Interrupted speech is physically truncated with a short crossfade to produce acoustically real interruptions. Backchannels are synthesized independently and mixed into the main conversation at their mapped positions to reproduce concurrent speech.

Because the symbolic interaction score completely specifies the generated conversation, annotations are obtained automatically during synthesis rather than through post-processing. Word timestamps are produced directly from CosyVoice alignment, speaker activity is exported as RTTM segments, and overlap, interruption, gap, and backchannel labels are read directly from the symbolic score.

## III. RELATED WORK

## A. Scripted conversational speech synthesis

Recent advances in LLMs and neural TTS have enabled large-scale conversational speech synthesis. Existing systems typically generate a dialogue script with an LLM and synthesize each utterance using a neural speech generator. Behavior-SD [1] incorporates behavior prompts to control conversational events, while PersonaPlex [2] further models speaker personas and dialogue roles. Other dialogue synthesis systems similarly introduce overlap, interruption, or backchannel events through dialogue markers, behavior tags, or handcrafted timing rules [5]–[8]. These approaches provide fine-grained control over dialogue semantics, making them effective for scalable corpus construction and controllable dialogue generation.

## B. Full-duplex conversational modeling

Full-duplex conversational models directly model bidirectional spoken interaction. dGSLM [9] first demonstrated dualstream spoken dialogue generation, while Moshi [3] introduced an inner-monologue architecture that enables realtime duplex conversation. More recent systems, including SyncLLM [10] and SALMONN-Omni [11], further improve synchronous conversational modeling and naturally produce interruptions, and other conversation dynamics through continuous listening while speaking. These models demonstrate that realistic interaction can emerge directly from duplex communication rather than explicit scheduling.

These two research directions address complementary aspects of conversational speech generation: one emphasizes controllable dialogue generation, while the other emphasizes interaction modeling. Our work combines the strengths of both by explicitly separating content generation, interaction generation, and acoustic rendering into three independent stages.

## IV. EXPERIMENTS

Unless noted, real-dialogue reference statistics are drawn from PriMock57 [12] together with published clinical-dialogue statistics; the reference voice for rendering is drawn from the same source but is used only for acoustic consistency, orthogonal to the timing statistics under test. Dialogue scripts are generated with DeepSeek-V4 [13].

![](images/16f8345553f998108f2a0e8b4a483bd9969d75092b695de1d62ed896114fc4af.jpg)  
Fig. 2. Floor-transfer-offset distributions. Emergent output overlaps the real distribution including its overlapped-transition mode and long silent tail; stitching collapses to a single near-zero-gap mode.

## A. Experiment 1: Distributional Realism

We first evaluate whether the proposed duplex interaction produces conversational dynamics that matches real dialogue statistics. Specifically, we compare the generated conversations against two stitching-based baselines using four complementary measures: floor-transfer-offset (FTO) Wasserstein distance, Kolmogorov–Smirnov (KS) statistic, overlappedtransition ratio, and long-tail gap ratio. The reference distribution is computed from 5,611 floor transfers extracted from PriMock57, and is consistent with published turn-taking statistics [14]–[16]. Emergent statistics in this experiment are computed on general-domain scripts, so that the mechanism is evaluated independently of register effects; the medicaldomain instantiation of the same configuration is reported in Table III and Table IV.

As summarized in Table I and Fig. 2, the proposed method produces conversational dynamics closer to the real-dialogue reference distribution than the stitching baselines evaluated in this study. The FTO Wasserstein distance is reduced from 0.695 to 0.366 (47% relative improvement), while the KS statistic decreases from 0.432 to 0.197. More importantly, unlike stitching-based synthesis, the proposed framework naturally reproduces both overlapped transitions and the long-tail silence distribution.

The observed improvement remains stable under multiple robustness checks. Varying the utterance-end estimation constant over {0.3, 0.4, 0.5} s yields consistent rankings $( W _ { \mathrm { f t o } } =$ 0.435/0.357/0.292), which outperform both baselines. Bootstrap evaluation across six random seeds produces a 95% confidence interval of [0.280, 0.545], which remains well separated from the baselines. Re-estimating utterance boundaries using an energy-based VAD further improves the agreement with real conversations, reducing $W _ { \mathrm { f t o } }$ to 0.240 and bringing the emergent FTO median to 0.08 s, close to the real value of 0.11 s.

The proposed interaction mechanism is most effective within the overlap range observed in natural dialogue. Excessively increasing overlap leads to less realistic conversational timing, indicating that the method is designed to reproduce natural interaction rather than arbitrary overlap patterns.

![](images/2e129dc141d719fcfb47c5bd67d7716c77ae270ccbcd26f314a3372bfc62ade1.jpg)

![](images/bf3ab498c87d4882d6b349bbddea28f61e2541d89c8c0be118008cb3319712a2.jpg)  
Fig. 3. Realism as a function of the overlap-control setting. The mechanism’s advantage holds inside the real-dialogue envelope and disappears once the overlap knob is driven past it—honestly bounding the claim.

## B. Experiment 2: Fidelity of Decoupled Generation

The proposed framework explicitly separates content generation, conversational timing, and acoustic rendering. This experiment verifies that information does not unintentionally leak across stages and each stage preserves the decision established by the previous one. Specifically, we evaluate content fidelity, timing preservation, and speaker consistency after the complete re-rendering pipeline.

Table II summarizes the fidelity of each stage in the proposed pipeline. Since lexical content is fixed before duplex interaction, the generated transcripts remain identical to the input script, resulting in 0% forced WER. Likewise, overlap events are transferred exactly from the symbolic interaction score to the rendered speech without deviation. Acoustic rerendering introduces at most 1.7% WER while preserving speaker identity, achieving a CAM++ [17] similarity of 0.82.

The last two rows of Table II confirm acoustic-level timing consistency: delivered overlap closely matches the annotations, with the remaining discrepancies explained by low-level (−6,dB) backchannel overlays that are annotated separately from floor overlap.

## C. Experiment 3: Emergence and Controllability

To examine whether the proposed interaction remains both adaptive and controllable, we compare conversations generated under different dialogue styles and timing configurations. Since dialogue content is fixed throughout generation, differences in conversational behavior are solely determined by the duplex interaction process. Under identical scripts and knob settings, different random seeds yield distinct performances (zero to three barge-ins per dialogue), confirming that overlap placement is stochastic and interaction-driven rather than deterministic.

Table III shows that the generated interaction adapts naturally to different dialogue styles while remaining controllable. Under identical timing parameters, medical conversations consistently exhibit less overlap than casual conversations (7.2% vs. 10.5%), closely matching the restrained interaction style observed in real clinical dialogue (6.3%). Meanwhile, backchannels emerge at prosodically appropriate positions in 95.5% of cases, substantially exceeding the random baseline (71.7%). Finally, increasing the handoff window consistently increases overlap frequency, whereas increasing max\_padding produces more long silent gaps, allowing conversational dynamics to be adjusted smoothly from polite to adversarial interaction.

TABLE I  
FTO DISTRIBUTIONAL REALISM VERSUS REAL CLINICAL DIALOGUE (PRIMOCK57, 5,611 TRANSITIONS). LOWER $W _ { \mathrm { f t o } }$ AND KS INDICATE BETTER AGREEMENT WITH THE REAL DISTRIBUTION. BEST SYNTHETIC RESULTS ARE SHOWN IN BOLD.
<table><tr><td>Method</td><td> $W _ { \mathrm { f t o } } \downarrow$ </td><td>KS↓</td><td>Overlapped transitions (%)</td><td>Long-tail gaps (%)</td></tr><tr><td>Real (PriMock57)</td><td></td><td></td><td>42.8</td><td>20.5</td></tr><tr><td>Stitching (fixed gap)</td><td>0.783</td><td>0.557</td><td>0.0</td><td>0.0</td></tr><tr><td>Stitching (random gap)</td><td>0.695</td><td>0.432</td><td>0.0</td><td>0.0</td></tr><tr><td>Ours</td><td>0.366</td><td>0.197</td><td>38.5</td><td>30.8</td></tr></table>

TABLE II  
PIPELINE FIDELITY. CONTENT LOCK, TIMING TRANSFER, AND RE-RENDERING EACH PRESERVE THEIR INVARIANT.
<table><tr><td>Check</td><td>Result</td></tr><tr><td>Forced WER (content lock)</td><td>0%</td></tr><tr><td>Overlap-event mapping deviation</td><td>0 (exact)</td></tr><tr><td>Re-rendering WER</td><td> $\leq 1 . 7 \%$ </td></tr><tr><td>Speaker consistency (CAM++)</td><td>0.82</td></tr><tr><td>Truncation validity (per segment)</td><td>0.01 pp</td></tr><tr><td>Overlap audit, dominant tier</td><td>0.5pp</td></tr></table>

TABLE III

REGISTER-ADAPTIVE OVERLAP UNDER IDENTICAL KNOB SETTINGS, AND BACKCHANNEL PLACEMENT.
<table><tr><td colspan="2">Condition</td><td>Value</td></tr><tr><td>Overlap % — medical script</td><td></td><td>7.2</td></tr><tr><td>Overlap % — casual script</td><td></td><td>10.5</td></tr><tr><td>Overlap % — real clinical</td><td></td><td>6.3</td></tr><tr><td colspan="2">Backchannel at prosodic boundary baseline</td><td>95.5% 71.7%</td></tr></table>

## D. Experiment 4: Downstream Stress Test

We evaluate whether the generated conversations provide a controllable benchmark for overlapping speech recognition. The released benchmark contains 180 segments (175 minutes) spanning three interaction difficulty levels, constructed by varying the timing parameters described in Section II-C. As summarized in Table IV, the three tiers exhibit progressively denser interaction patterns, with the natural tier closely matching the overlap statistics of real clinical dialogue.

We evaluate Whisper large-v2 [18] and wav2vec2 [19] on the three difficulty tiers, reporting overlap-region WER separately from the overall WER. As shown in Table V, overlapregion errors are consistently 3–7× higher than aggregate WER and increase monotonically with interaction difficulty for both systems, whereas clean-region performance remains nearly unchanged. These results indicate that the proposed timing controls effectively regulate ASR difficulty through conversational overlap rather than overall speech quality.

TABLE IV  
THE THREE TIERS SEPARATE MONOTONICALLY ON INTERACTION DENSITY; THE NATURAL TIER CLOSELY MATCHES THE TIME-OVERLAP RATE OF REAL CLINICAL DIALOGUE. TIME OVERLAP AND LONG-TAIL FROM CONSTRUCTION-TIME GROUND TRUTH.
<table><tr><td>Tier</td><td>Overlap %</td><td>Overlap. trans. %</td><td>Long-tail %</td><td>BC/min</td></tr><tr><td>Polite</td><td>2.0</td><td>11.6</td><td>34.4</td><td>3.9</td></tr><tr><td>Natural</td><td>6.6</td><td>20.6</td><td>28.8</td><td>3.7</td></tr><tr><td>Adversarial</td><td>12.2</td><td>27.2</td><td>21.0</td><td>2.6</td></tr><tr><td>Real</td><td>6.3</td><td>42.8</td><td>20.5</td><td>1.5</td></tr></table>

TABLE V

ASR WER (%) ON CORRECTED AUDIO. OVERLAP-REGION ERROR FAR EXCEEDS AGGREGATE AND RISES MONOTONICALLY WITH DIFFICULTY ACROSS BOTH SYSTEMS, WHILE CLEAN REGIONS STAY  
FLAT—DEGRADATION IS OVERLAP-DRIVEN AND CONTROLLABLE.
<table><tr><td>System</td><td>Region</td><td>Polite</td><td>Natural</td><td>Advers.</td></tr><tr><td rowspan="3">Whisper</td><td>aggregate</td><td>4.1</td><td>6.8</td><td>12.7</td></tr><tr><td>overlap-region</td><td>28.3</td><td>48.1</td><td>56.7</td></tr><tr><td>clean</td><td>3.1</td><td>1.8</td><td>2.2</td></tr><tr><td rowspan="3">wav2vec2</td><td>aggregate</td><td>16.5</td><td>20.2</td><td>24.5</td></tr><tr><td>overlap-region</td><td>50.7</td><td>64.8</td><td>71.9</td></tr><tr><td>clean</td><td>15.2</td><td>14.7</td><td>13.3</td></tr></table>

To distinguish the effects of conversational timing and acoustic rendering, we further conduct a controlled $2 \times 2$ ablation with matched overlap rates (Table VI). Note that the absolute WER values in Table VI are not directly comparable to the by-tier numbers in Table V: the ablation cells hold the overlap rate and renderer fixed and exclude backchannel overlays, so their overlap regions consist only of turn-transition overlaps, a different and acoustically simpler composition than the full benchmark tiers. The comparison of interest is therefore within the 2×2, not across tables. Holding rendering fixed, emergent and stitched timing produce comparable overlapregion WER, suggesting that conversational timing itself has only a limited influence on downstream ASR performance. In contrast, replacing truncate-and-fade rendering with equalvolume overlap substantially increases recognition difficulty, indicating that rendering intensity dominates overlap-region ASR performance.

Two effects separate cleanly. Timing has no detectable effect: with rendering held fixed, emergent and stitched confidence intervals overlap almost entirely (18.8 vs. 23.2; 44.1 vs. 43.3), so at n=30 any effect of emergent versus authored timing on ASR error is small relative to rendering. Rendering dominates: with timing held fixed, replacing truncate-andfade with equal-volume overlay roughly doubles the error (disjoint CIs). Counter-intuitively, the acoustically cleaner truncate-and-fade rendering makes overlap easier: fading the interrupted speaker lets the recognizer lock onto the surviving barge-in, moving error away from the real value. The harder, equal-volume condition is closer to real but still leaves a large gap—every synthetic cell sits well below the real overlapregion WER of 76.5%.

TABLE VI  
2 × 2 ABLATION OF OVERLAP-REGION WER (%, 95% CI). RENDERING, NOT TIMING, DRIVES DOWNSTREAM DIFFICULTY; ALL SYNTHETIC CELLS FALL WELL BELOW REAL.
<table><tr><td>Timing</td><td>Rendering</td><td>Overlap-region WER</td></tr><tr><td>Emergent</td><td>truncate+fade</td><td>18.8 (13.1–24.6)</td></tr><tr><td>Stitched</td><td>truncate+fade</td><td>23.2 (15.5–31.8)</td></tr><tr><td>Emergent</td><td>equal-volume</td><td>44.1 (37.5–49.6)</td></tr><tr><td>Stitched</td><td>equal-volume</td><td>43.3 (36.9–50.0)</td></tr><tr><td>Real</td><td></td><td>76.5 (56.4–99.6)</td></tr></table>

Robustness. The observed trend generalizes across different speaker pairs and dialogue domains. Re-rendering with 12 LibriTTS [20] speaker pairs preserves the overlap-region degradation, and similar behavior is observed on out-ofdomain casual dialogue. In contrast, speaker diarization error (pyannote-3.1 [21]) remains largely unchanged across difficulty levels, indicating that the benchmark primarily stresses speech recognition rather than speaker attribution.

## V. LIMITATIONS

The proposed framework has several limitations. First, conversational interaction is only partially emergent. Although turn timing is generated online through duplex interaction, global interaction statistics such as overlap frequency and backchannel density are still regulated by explicit control parameters. Likewise, overlap is currently confined to turntransition regions and does not model deep mid-utterance interruptions.

Second, the strict decoupling between content and timing inevitably limits conversational adaptability. Since dialogue content is fixed before duplex generation, speakers cannot revise or abandon planned utterances after being interrupted, preventing richer interaction phenomena such as repair, selfcorrection, or content negotiation.

Finally, the current benchmark remains a simplified approximation of real clinical conversations. It is constructed from a single reference corpus and synthetic rendering, and downstream ASR difficulty is influenced more by acoustic rendering than by emergent timing itself. Although the proposed benchmark provides controllable conversational dynamics, further improvements in overlap rendering and multispeaker interaction are needed to better match real-world conversations.

## VI. CONCLUSION

We presented DuplexGen, a zero-training pipeline that decouples content, timing, and acoustics so that overlapping conversational speech can be scripted for content yet emergent in timing and high-fidelity in sound. Emergent turn-taking is distributionally closer to real clinical dialogue than stitching and reproduces overlapped transitions and long gaps that are absent from the evaluated stitching baselines, while three physical knobs give controllable densities with a clean control/emergence boundary. The resulting three-difficulty medical benchmark ships with construction-time ground truth for stress-testing ASR and diarization.

## ETHICS AND DATA STATEMENT

The reference corpus (PriMock57) is used under its license. All released audio is synthetic; we label it as such and restrict its use to research on speech recognition and dialogue robustness. The synthetic data carries no clinical validity claim and must not be used for medical decision-making.

## ACKNOWLEDGMENT AND AI DISCLOSURE

This anonymous draft follows the IEEE submission requirements. Claude and Grammarly were used to polish drafts and assist with table organization in this manuscript. The experimental design and results, claims, citations were run and written by the authors.

## REFERENCES

[1] S. Lee, K.-w. Kim, and G. Kim, “Behavior-SD: Behaviorally aware spoken dialogue generation with large language models,” in Proc. NAACL-HLT, Albuquerque, NM, USA, 2025, pp. 9574–9593.

[2] R. Roy, J. Raiman, S.-g. Lee, T.-D. Ene, R. Kirby, S. Kim, J. Kim, and B. Catanzaro, “PersonaPlex: Voice and role control for full-duplex conversational speech models,” arXiv preprint arXiv:2602.06053, 2026.

[3] A. Defossez, L. Mazar ´ e, M. Orsini, A. Royer, P. P ´ erez, H. J ´ egou,´ E. Grave, and N. Zeghidour, “Moshi: A speech-text foundation model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024.

[4] Z. Du, C. Gao, Y. Wang, F. Yu, T. Zhao, H. Wang, X. Lv, H. Wang, C. Ni, X. Shi et al., “CosyVoice 3: Towards in-the-wild speech generation via scaling-up and post-training,” arXiv preprint arXiv:2505.17589, 2025.

[5] Z. Zhou, Q. Zhang, L. Luo, J. Liu, and R. Zhou, “Open-source full-duplex conversational datasets for natural and interactive speech synthesis,” arXiv preprint arXiv:2509.04093, 2025.

[6] S. Pan, S. Banerjee, D. Hebbar, S. Patel, A. Gupta, K. J. Cheng, H. Kim, Z. A. Li, M. Q. Ma, T. Li, G. Anumanchipalli, and J. Lian, “Enabling conversational behavior reasoning capabilities in full-duplex speech,” arXiv preprint arXiv:2512.21706, 2025.

[7] G.-T. Lin, J. Lian, T. Li, Q. Wang, G. Anumanchipalli, A. H. Liu, and H.-y. Lee, “Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities,” arXiv preprint arXiv:2503.04721, 2025.

[8] Y. Peng, Y.-W. Chao, D. Ng, Y. Ma, C. Ni, B. Ma, and E. S. Chng, “FD-Bench: A full-duplex benchmarking pipeline designed for full duplex spoken dialogue systems,” arXiv preprint arXiv:2507.19040, 2025.

[9] T. A. Nguyen, E. Kharitonov, J. Copet, Y. Adi, W.-N. Hsu, A. Elkahky, P. Tomasello, R. Algayres, B. Sagot, A. Mohamed, and E. Dupoux, “Generative spoken dialogue language modeling,” Trans. Assoc. Comput. Linguistics, vol. 11, pp. 250–266, 2023.

[10] B. Veluri, B. N. Peloquin, B. Yu, H. Gong, and S. Gollakota, “Beyond turn-based interfaces: Synchronous LLMs as full-duplex dialogue agents,” in Proc. EMNLP, 2024, pp. 21 390–21 402.

[11] W. Yu, S. Wang, X. Yang, X. Chen, X. Tian, J. Zhang, G. Sun, L. Lu, Y. Wang, and C. Zhang, “SALMONN-omni: A codec-free LLM for full-duplex speech understanding and generation,” arXiv preprint arXiv:2411.18138, 2024.

[12] A. Papadopoulos Korfiatis, F. Moramarco, R. Sarac, and A. Savkov, “PriMock57: A dataset of primary care mock consultations,” in Proc. 60th Annu. Meeting Assoc. Comput. Linguistics (Short Papers), 2022, pp. 588–598.

[13] DeepSeek-AI, “DeepSeek-V4: Towards highly efficient million-token context intelligence,” 2026. [Online]. Available: https://arxiv.org/abs/ 2606.19348

[14] M. Heldner and J. Edlund, “Pauses, gaps and overlaps in conversations,” Journal of Phonetics, vol. 38, no. 4, pp. 555–568, 2010.

[15] T. Stivers, N. J. Enfield, P. Brown, C. Englert, M. Hayashi, T. Heinemann, G. Hoymann, F. Rossano, J. P. de Ruiter, K.-E. Yoon, and S. C. Levinson, “Universals and cultural variation in turn-taking in conversation,” Proc. Natl. Acad. Sci. USA, vol. 106, no. 26, pp. 10 587– 10 592, 2009.

[16] S. C. Levinson and F. Torreira, “Timing in turn-taking and its implications for processing models of language,” Frontiers in Psychology, vol. 6, p. 731, 2015.

[17] H. Wang, S. Zheng, Y. Chen, L. Cheng, and Q. Chen, “CAM++: A fast and efficient network for speaker verification using context-aware masking,” in Proc. Interspeech, 2023, pp. 5301–5305.

[18] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023, pp. 28 492–28 518.

[19] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proc. NeurIPS, 2020, pp. 12 449–12 460.

[20] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “LibriTTS: A corpus derived from LibriSpeech for text-tospeech,” in Proc. Interspeech, 2019, pp. 1526–1530.

[21] H. Bredin, “pyannote.audio 2.1 speaker diarization pipeline: Principle, benchmark, and recipe,” in Proc. Interspeech, 2023, pp. 1983–1987.