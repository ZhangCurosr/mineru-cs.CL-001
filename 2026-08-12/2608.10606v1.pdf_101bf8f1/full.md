# ASR-ROUNDTRIP EVALUATION CAN MASK CONTEXT- AND CONVENTION-DEPENDENT READING ERRORS IN CHINESE NEWS TTS

Shijun Luo∗ and Lizhi Wan

NetEase Cloud Music, Hangzhou, China

## ABSTRACT

ASR-roundtrip evaluation is widely used as a scalable proxy for text-to-speech (TTS) intelligibility, but it can produce false negatives for reading errors perceived by listeners. We study Chinese news TTS spans whose correct reading depends on context or domain conventions, such as sports scores, aircraft models, technical units, and membership names. In these cases, Raw TTS can choose a plausible but wrong reading while ASR transcribes the audio as the intended or surface-correct text. A targeted audit over 110 high-risk MiMo TTS cases, reported with a complete denominator, confirms 46 masked false negatives, 9 exposed TTS errors, and 55 cases with no Raw TTS error. A span-isolation diagnostic re-exposes 18/46 previously masked errors. A Rawonly CosyVoice audit on the same targeted pool confirms 51 masked cases. Across the 97 TTS-specific audio files labeled confirmed masked across the two audits, Qwen3-ASR surface-recovers 40 cases, whereas Paraformer does so in only 2. The results suggest that ASR-roundtrip is useful for screening but insuficient as standalone ground truth for Chinese news reading-risk evaluation.

Index Terms— text-to-speech evaluation, ASR-roundtrip, Chinese news TTS, reading errors, speech evaluation

## 1. INTRODUCTION

Automatic speech recognition (ASR) roundtrip evaluation is attractive for TTS: synthesize speech, transcribe it, and compare the transcript to a reference. It scales cheaply and often tracks intelligibility, but its reliability depends on the task and protocol [1, 2]. This paper shows a specific failure mode for Chinese news TTS: the audio can be wrong to a listener, while ASR transcribes it as the intended or surface-correct text.

The problem appears in short, high-prior written forms. In a snooker report, 13-11 should be read as a score, i.e., 十三 比十一, rather than a range-like reading such as 十三至十一. In a military report, 伊尔-76 is an aircraft model, not a negative number. In technology and automotive news, 640kW, 350Wh/kg, or 88VIP require domain conventions. These errors are often fluent: a listener hears the wrong reading, but the transcript can still look correct.

We frame the core family as Context-Dependent Reading Decisions (CDRD): written spans whose correct reading depends on context beyond the local character sequence. We also include CDRD-adjacent risks such as unit strings, membership names, abbreviations, and foreign names. These adjacent cases are not always CDRD under the strict definition, but they expose the same evaluation failure: ASR can normalize or recover a surface-correct transcript despite the wrong spoken reading.

We study an evaluation failure rather than propose a new TTS frontend. We provide: (i) a targeted audit, reported with a complete denominator, for TTS-wrong / ASR-surfacecorrect false negatives; (ii) a span-isolation diagnostic showing that removing sentence context re-exposes many errors; (iii) cross-TTS and cross-ASR controls showing both recurrence and evaluator dependence; and (iv) a human-audited evaluation protocol.

## 2. READING-RISK BENCHMARK

## 2.1. Risk definition

A CDRD span is a written span x whose correct reading depends on context c. The same local pattern can require different readings across domains: a hyphen may mark a score, range, model, or subtraction-like form; a mixed digit-letter span may be a unit, product name, or membership tier. CDRD overlaps with Chinese text normalization and G2P/polyphone disambiguation [3, 4], but our target is the downstream evaluation failure when audio with a wrong reading is recovered as surface-correct text. We use three case labels:

• CDRD-entity: scores, military/aircraft models, financial quarters, generation labels, and entity-dependent hyphenated forms.

• CDRD-polyphone: Chinese polyphone ambiguity, tracked separately because it is closer to G2P.

• CDRD-adjacent: units, percentages, mixed-script strings, memberships, abbreviations, and foreign names.

Each case is annotated at the risk-span level with surface form, expected reading, known negative readings when available, context evidence, and whether transcript-based scoring is allowed.

## 2.2. Data construction

We start from 108,124 company-produced Chinese news scripts used in a production TTS workflow. Taxonomy-driven mining follows the text-normalization tradition of explicitly modeling non-standard written forms [5, 6]: hyphens, scores, mixed digit/Latin strings, technical units, financial quarters, generation labels, aircraft/military models, and known polyphone-prone spans. For real news, a scripted candidateconstruction pipeline cleans titles and summaries, filters missing or unusable titles and rows without rule-detected auditable risk spans, scores candidates by the number and diversity of risk spans, and applies type/domain caps to select a 500-case real-news candidate pool. These construction filters use only text and risk-span metadata before TTS/ASR generation, so they are risk-coverage and data-availability filters rather than outcome filters.

We also build a 5K synthetic hard-case pool and freeze a 200-case benchmark for both human listening audits and automatic TTS/ASR experiments. The frozen set contains 155 real-news cases and 45 synthetic hard cases. It covers all three labels and seven news domains: 85 CDRD-entity, 35 CDRD-polyphone, and 80 CDRD-adjacent/non-CDRD cases; domains include auto (44), general (43), finance (33), tech (31), sports (27), international (12), and military (10). Human listening audits are conducted after the benchmark and targeted audit pools are frozen; human judgments of TTS or ASR outcomes are not used to select the candidate pool.

## 3. EVALUATION PROTOCOL

## 3.1. Systems

The primary system pairs MiMo-V2.5-TTS API synthesis with the audio-capable MiMo mimo-v2.5 API [7] prompted for verbatim transcription; mimo-v2-omni is used as a fallback and protocol-ablation route. The strict ASR protocol mainly targets spoken numeral preservation and discourages transcript normalization; it reduces but cannot fully suppress unit- or entity-level canonicalization in the decoder or postprocessing. We use Whisper-small [8] as a non-MiMo ASR control and CosyVoice-300M-SFT [9] as a Raw-only second-TTS validation. We additionally transcribe the 220 TTSspecific Raw files from the two 110-case targeted audits and the 46 aligned MiMo clips with Paraformer-zh v2.0.4 [10] and Qwen3-ASR-1.7B [11]. Paraformer uses FSMN-VAD without punctuation, hotwords, or an external language model; Qwen uses its local Transformers backend with empty context and automatic language detection. Neither system receives source text or target readings.

We evaluate two text conditions. In the Raw condition, the TTS input is the original news title and summary, without rewriting CDRD/CDRD-adjacent risk spans into expected spoken forms; model settings are matched across conditions, and no target-span expected readings are supplied in Raw. In the Structured condition, the same source text is converted into a protocol-level spoken-text form by instantiating precomputed risk-span annotations: expected readings replace written spans, and negative readings and context evidence are retained for audit. Because the readings come from predefined rules and mappings that also define the benchmark targets, Structured is an oracle-style diagnostic condition. It should be interpreted as an upper bound, not as a deployable frontend or a fair end-to-end system baseline.

## 3.2. Human span audit

One primary annotator listened to Raw and Structured audio for all 200 frozen benchmark cases and counted correct risk spans. This is a targeted span-audit task, not a blind MOS or blind preference test. The review page shows the original text, Structured TTS text, key risk spans with expected and known negative readings, Raw and Structured audio players, and ASR transcripts.

The annotation protocol is audio-first. For the primary pass and IAA passes, annotators judge risk-span correctness from the audio against expected and known negative readings before using ASR transcripts as audit context. ASR transcripts are not treated as evidence that audio is correct.

The primary pass covers all 200 cases. A 30-case IAA subset is independently reviewed by two additional annotators, stratified as 10 CDRD-entity, 10 CDRD-polyphone, and 10 CDRD-adjacent/non-CDRD cases. Over N cases, the metric is case-macro risk-span audio accuracy for condition s:

$$
\operatorname { A c c } _ { s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { n _ { i , s } ^ { \mathrm { c o r r e c t } } } { n _ { i } ^ { \mathrm { r i s k } } } , \quad s \in \{ \mathrm { R a w } , \mathrm { S t r u c t u r e d } \} .
$$

If an entered count exceeds the number of spans, the oficial score clamps it to the total and preserves the original value for audit. We report correct-count IAA rather than better-pipeline agreement because the latter is more subjective.

## 3.3. Targeted masked-error audit

We conduct a 110-row targeted audit over high-risk frozenbenchmark candidates. A row enters the audit pool as a candidate, not as a confirmed masked case. Candidates are mined using high-risk span rules and suspected masking signals, including CDRD/CDRD-adjacent forms, Raw/Structured disagreements, and ASR transcripts that appear surface-correct despite high-risk audio.

The masked-error audit is also audio-first. The primary reviewer first listens to Raw TTS and judges whether the target span is spoken incorrectly. Only after this audio judgment does the reviewer consult ASR transcripts to categorize the row as:

• confirmed masked: Raw TTS is wrong and at least one ASR route writes the expected/surface-correct form;

• exposed TTS error: Raw TTS is wrong and ASR exposes or preserves the wrong reading;

• no Raw TTS error: Raw TTS is correct;

• uncertain/not judgeable: insuficient evidence.

The final 110 audit categories are then fully checked by a second reviewer before the counts in Table 1 are reported. As an additional robustness check, we build a 30-row label-blind relabel subset stratified from the targeted audit pool, including 15 confirmed-masked cases that are not re-exposed by span isolation, 7 exposed-error cases, and 8 no-error cases. Blind relabeling reaches 23/30 exact agreement over full audit labels (κ = 0.634), and 27/30 agreement for confirmed-masked versus other outcomes (κ = 0.800). This second pass and blind subset cover audit categories, while the 30-case IAA above covers correct-count labels for the 200-case Raw/Structured span audit. This targeted audit demonstrates the mechanism, not its production prevalence.

## 4. RESULTS

## 4.1. ASR false negatives in targeted audits

Table 1 reports the targeted audit, with a complete denominator, that supports the ASR false-negative claim. In the MiMo audit, human listening confirms 46 cases where Raw TTS is wrong and ASR transcribes the audio as the expected or surface-correct text. The same audit explicitly accounts for 9 exposed TTS errors and 55 candidates with no Raw TTS error, avoiding positive-only reporting. The CosyVoice column uses the same targeted pool and is interpreted in Section 4.5 as cross-TTS mechanism validation, not as a second prevalence estimate.

Table 1. Primary targeted-audit evidence. Counts are targeted audit yields, not production prevalence.
<table><tr><td>Outcome</td><td>MiMo Raw</td><td>CosyVoice Raw</td></tr><tr><td>confirmed masked</td><td>46</td><td>51</td></tr><tr><td>exposed TTS error</td><td>9</td><td>27</td></tr><tr><td>no Raw TTS error</td><td>55</td><td>30</td></tr><tr><td>uncertain / not judgeable</td><td>0</td><td>2</td></tr><tr><td>total targeted pool</td><td>110</td><td>110</td></tr></table>

This evidence can be read as an ASR false-negative contingency table: wrong Raw audio plus surface-correct ASR is a masked false negative (46); wrong Raw audio plus exposed ASR is an exposed TTS error (9); correct Raw audio is no Raw TTS error (55). Masking is protocol-sensitive: a conservative separator-normalized match against either the written target span or the expected reading identifies surface-correct recovery in 34/46 cases under default ASR and 27/46 under the MiMo-V2-Omni strict route; normalization removes only whitespace, hyphens, colons, full-width colons, and middle dots. Thus the audit label is not defined by one ASR prompt alone; it requires a human-confirmed wrong Raw reading and at least one surface-correct ASR route. By benchmark label, the MiMo outcomes are 33/6/42 for CDRD-entity, 3/1/7 for CDRD-polyphone, and 10/2/6 for CDRD-adjacent/non-CDRD, respectively ordered as confirmed masked, exposed error, and no Raw TTS error. The denominator is targeted and high-risk, so it supports existence and mechanism, not incidence.

## 4.2. Representative failure modes

Table 2 makes the masking mechanism concrete: the Raw audio contains the wrong reading, while ASR recovers a surface-correct or conventional written form. The route is not uniform: scores and model names mainly reflect contextual recovery, whereas units and membership names often reflect convention- or normalization-driven canonicalization. The 46 MiMo confirmed masked cases are not random acoustic glitches. They cover sports scores (20), kW/kWh unit strings (10), military models (9), hyphen-range or torque forms (3), and smaller financial-quarter, compute-unit, membership, and voltage categories (one each).

Table 2. Representative confirmed masked cases.
<table><tr><td>Surface</td><td>Intended</td><td>Raw heard</td><td>ASR recovery / route</td></tr><tr><td>13-11</td><td>十三比十一</td><td>十三至十一</td><td>score form (context)</td></tr><tr><td>F-35</td><td>F三五</td><td>F杠三五</td><td>F-35 (model)</td></tr><tr><td>伊尔-76</td><td>伊尔七六</td><td>伊尔负七六</td><td>aircraft form (entity)</td></tr><tr><td>88VIP</td><td>八十八VIP</td><td>八十八伏IP</td><td>88VIP (membership)</td></tr><tr><td>350Wh/kg</td><td>三百五十瓦时每 公斤</td><td>partial unit-letter reading</td><td>normalized unit</td></tr></table>

## 4.3. Context isolation

To test context dependence, we isolate target-span clips for the 46 confirmed masked MiMo cases. Full-context masking is established by a case-specific ASR route; all clips use main strict ASR, so the MiMo comparison is cross-route. Table 3 shows that isolation re-exposes 18/46 errors. Section 4.5 provides a same-decoder Qwen control.

Table 3. Context isolation on 46 confirmed cases. Full sentences use the original audit route; clips use main strict ASR.
<table><tr><td>Condition</td><td>Exp.</td><td>Mask</td><td>No out.</td><td>Other</td></tr><tr><td>Original full sentence</td><td>0</td><td>46</td><td>0</td><td>0</td></tr><tr><td>Rough 6s clip</td><td>16</td><td>11</td><td>17</td><td>2</td></tr><tr><td>Aligned clip</td><td>18</td><td>12</td><td>13</td><td>3</td></tr></table>

A rough 6-second extraction exposes 16 cases. Whispersmall timestamp alignment exposes 18: 16 strong erroneousreading and 2 partial unit-letter recoveries. Representative isolated transcripts include 二十一至十七 for 21-17, F 杠三五 for F-35, and 伊尔负七十六 for 伊尔-76; a full-context route had recovered each to a surface-correct form.

This is not proposed as a replacement metric. It is mechanism evidence: ASR can recover the local acoustic evidence in many confirmed masked cases. Rather, sentence context can push the transcript back toward the intended or conventional written form.

## 4.4. Human and automatic evaluation

The 200-case Raw/Structured comparison is a secondary oracle-style diagnostic, interpreted as an upper-bound sanity check rather than a deployable-system result. It asks whether making reading decisions explicit reduces risk-span errors perceived by listeners. Overall case-macro accuracy is 0.8889 for Raw and 0.9503 for Structured, a +0.0614 gain (95% CI: [+0.0352, +0.0891]). The largest case-macro gain is on CDRD-entity cases: 0.7887 to 0.9146, +0.1259 (95% CI: [+0.0728, +0.1807]). This supports the interpretation that many failures are reading-decision failures rather than generic acoustic defects, but it is not presented as a fair frontend comparison. On the 30-case IAA subset, pairwise exact agreement for correct counts ranges from 0.900 to 0.933 on Raw and from 0.833 to 0.900 on Structured; Pearson correlations range from 0.848 to 0.948.

ASR-roundtrip scores are useful but protocol-sensitive and should not be treated as ground truth. Across ASR protocols, Raw/Structured automatic scores range from 0.495/0.528 under default ASR to 0.728/0.805 under MiMo-V2-Omni strict ASR, while the Raw-to-Structured direction remains stable. In the MiMo strict-ASR 200-case matrix, Raw scores 0.6121 and Structured scores 0.7826. On the aligned 196-case subset, human labels exceed strict-ASR scores by +0.2745 for Raw (95% CI: [+0.2184, +0.3320]) and +0.1667 for Structured (95% CI: [+0.1146, +0.2220]), showing a calibration gap rather than a direct masking rate.

## 4.5. Cross-system validation

CosyVoice is evaluated with Raw input on the same 110 targeted pool, so it tests cross-TTS recurrence rather than prevalence. Human review follows the same audio-first protocol. CosyVoice yields 51 confirmed masked cases; MiMo strict ASR masks 37, Whisper-small masks 36, and both mask 22. Its 27 exposed TTS errors, versus 9 for MiMo, also show that error realizations are TTS-dependent.

Occurrence-aware review of the 97 files previously labeled confirmed masked gives the open-source ASR comparison in Table 4.

Table 4. Occurrence-aware surface recovery on confirmedmasked files.
<table><tr><td>ASR</td><td>MiMo</td><td>CosyVoice</td><td>Total</td></tr><tr><td>Paraformer-zh</td><td>0/46</td><td>2/51</td><td>2/97</td></tr><tr><td>Qwen3-ASR-1.7B</td><td>19/46</td><td>21/51</td><td>40/97</td></tr></table>

Paraformer preserves a wrong/noncanonical form in 88/97 cases. Its FunASR use\_itn toggle changes 0/220 full-file and 0/46 aligned-clip transcripts, so this is a no-efect check rather than an ITN ablation. Qwen preserves a wrong/noncanonical form in 56/97. Among its 19 full-context MiMo recoveries, aligned-span transcription re-exposes 12 and leaves 7 surface-correct. Thus an open-source ASR-LLM reproduces both masking and the isolation efect, while Paraformer shows strong decoder and protocol dependence.

## 5. DISCUSSION

The central claim is deliberately narrow: ASR-roundtrip can miss reading errors perceived by listeners in CDRD and CDRD-adjacent spans. This does not imply that ASR always masks such errors, that 46/110 is a natural production rate, or that Structured is a deployable frontend. ASR-roundtrip remains useful for scalable screening and ablation, but for Chinese news reading-risk benchmarks it should not be treated as standalone ground truth. Human listening references, and span-level diagnostics when mechanism claims are made, remain necessary.

This has practical implications for speech and language processing evaluation. Short written forms carry languagemodel, normalization, and domain-convention priors for both TTS reading decisions and ASR transcript recovery. The masking route therefore difers across cases: score and entity examples mainly support contextual recovery, while unit and membership examples often involve written-form canonicalization. Evaluating only the transcript can still hide precisely the fluent wrong readings that matter to listeners. Future work should test more open-source ASR/TTS systems and broaden risk-span discovery beyond the current rules; this paper focuses on documenting the ASR-roundtrip blind spot.

The main limitations follow the same boundary: the 110- row audit is targeted rather than random; CosyVoice reuses that pool and is Raw-only; and isolated clips can produce nooutput or other transcripts. These experiments establish mechanism and evaluator dependence, not production incidence or a deployable isolation metric.

Data availability. Released prompts, settings, transcripts, labels, metadata, audio, summaries, and scoring code are available on GitHub and archived on Zenodo. Code is MIT; the 500 company-authorized scripts, 5K synthetic cases, and other released data are CC BY 4.0. The 108,124-item source export is excluded.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study uses news text, synthesized speech, ASR transcripts, and human listening judgments for TTS evaluation. Human annotators performed span-level listening audits of synthetic audio and authorized the use of their anonymized judgments for this study. No sensitive personal data, medical data, biometric identification task, speaker identification task, or clinical intervention is involved. The paper reports aggregate evaluation results and does not release annotator identities. The study follows the principles of the Declaration of Helsinki. It was reviewed under the authors’ institutional legal and compliance process and was determined not to require formal ethics-board review.

Funding and competing interests. No external funding was received for this study. The authors are employees of NetEase Cloud Music. The authors declare no other relevant financial or non-financial competing interests.

## 7. REFERENCES

[1] J. Taylor and K. Richmond, “Confidence intervals for ASR-based TTS evaluation,” Proc. Interspeech, 2021.

[2] B. Favre et al., “Automatic human utility evaluation of ASR systems: Does WER really predict performance?” Proc. Interspeech, 2013.

[3] W. Dai et al., “An end-to-end Chinese text normalization model based on rule-guided flat-lattice Transformer,” Proc. ICASSP, 2022.

[4] E. Choi, H.-Y. Kim, J.-H. Kim, and J.-M. Kim, “Label embedding for Chinese grapheme-to-phoneme conversion,” Proc. Interspeech, 2021.

[5] R. Sproat et al., “Normalization of non-standard words,” Computer Speech & Language, 2001.

[6] P. Ebden and R. Sproat, “The Kestrel TTS text normalization system,” Natural Language Engineering, 2015.

[7] Xiaomi MiMo, “Xiaomi MiMo Audio Understanding API Documentation,” 2026. [Online]. Available: https://mimo. mi.com/docs/en-US/quick-start/ usage-guide/multimodal-understanding/ audio-understanding. Accessed: Jul. 17, 2026.

[8] A. Radford et al., “Robust speech recognition via largescale weak supervision,” Proc. ICML, 2023.

[9] Z. Du et al., “CosyVoice: A scalable multilingual zeroshot text-to-speech synthesizer based on supervised semantic tokens,” arXiv:2407.05407, 2024.

[10] Z. Gao, S. Zhang, I. McLoughlin, and Z. Yan, “Paraformer: Fast and accurate parallel Transformer for non-autoregressive end-to-end speech recognition,” Proc. Interspeech, pp. 2063–2067, 2022.

[11] X. Shi et al., “Qwen3-ASR technical report,” arXiv:2601.21337, 2026.