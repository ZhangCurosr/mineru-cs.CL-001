# PolyDebate: A Game-Orchestrated Multimodal System for Debate Skills Practice and Evaluation

Jianing Yin<sup>1</sup>, Weng Pan Kuan<sup>2</sup>, Xiaoyun Liu<sup>1</sup>, Zhiyuan Wen<sup>1,3(B)</sup>, Yuxuan Li<sup>1</sup>, Milos Stojmenovic<sup>4</sup>, and Jiannong Cao<sup>1,3</sup>

<sup>1</sup> Department of Computing, The Hong Kong Polytechnic University, Hong Kong SAR, China

{jianing-laetitia.yin, xiaoyun.liu, yuuxuann.li}@connect.polyu.hk {zhiyuan.wen, jiannong.cao}@polyu.edu.hk

Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University, Hong Kong SAR, China

3 Institute for Higher Education Research and Development, The Hong Kong Polytechnic University, Hong Kong SAR, China

Department of Computer Science and Electrical Engineering, Singidunum University, Belgrade, Serbia mstojmenovic@singidunum.ac.rs

Abstract. Debate is a structured form of persuasive communication that trains argument construction, rebuttal, oral delivery, and audience awareness. These skills are valued in education, language learning, and professional communication. Recent AI debate systems and LLM-based judges have advanced argument generation and debate evaluation, but most remain text-centered and rarely support learners through a complete multimodal practice experience. We introduce PolyDebate, a gameorchestrated multimodal system for English debate practice and evaluation. PolyDebate guides learners through staged one-on-one (1v1) debates with an AI opponent, while skill cards, props, and coins make persuasive strategies explicit and turn practice into a game-like interaction. During each session, the system captures learner speech and visual delivery evidence, generates context-aware opponent responses, and produces rubric-informed stage-level and overall feedback. PolyDebate is available as both an immersive Unity 3D game version and a web platform version that share the same workflow and evaluation services. Four studies covering AI opponent quality, evaluation coverage, AI judge feedback, and user perception show that PolyDebate brings debate interaction, gamified scafolding, multimodal assessment, and structured feedback together in a practical workflow for debate skills practice. The demonstration video is available at https://youtu.be/mHwBG1\_8Ebk.

Keywords: Debate practice · Multimodal learning system · AI debater · Automated evaluation · Game-based learning

## 1 Introduction

Debate is a useful form of persuasive communication that asks learners to construct arguments, use evidence, respond to opposing views, and deliver speeches to an audience. It is widely used in education and language learning because it trains critical thinking, spoken communication, and audience awareness [1–3]. However, high-quality debate practice is hard to provide at scale: learners need a responsive opponent, a clear debate procedure, and formative feedback on both argument quality and oral delivery.

Large language models (LLMs) open new opportunities for AI-supported debate practice: AI debaters can retrieve evidence, generate arguments, and sustain multi-turn interactions [4–6], and LLM-based judges can support scoring, critical-thinking assessment, and guided debate activities [7, 1, 3]. Yet two limitations remain. First, most systems focus on generating strong arguments or scoring transcripts rather than supporting a complete learner-facing practice process, lacking staged guidance, strategy scafolding, and immediate feedback. Second, debate is inherently multimodal: persuasiveness depends not only on argument content but also on speech delivery, facial expression, gesture, and posture, so text-only practice and evaluation lose important evidence about delivery.

To fill these gaps, we introduce PolyDebate, a game-orchestrated multimodal system for English debate practice and evaluation, built from three modules. The Game-Orchestrated Practice Module runs a complete 1v1 debate round with side assignment, stage control, speaking turns, skill cards, props, and coins. The Stage-Aware AI Debater Module generates opponent responses conditioned on the motion, debate side, current stage, interaction history, and assigned skill card, so the AI acts as both an opponent and a pedagogical example. The Rubric-Aligned Multimodal Evaluation Module analyzes transcript, speech/audio, and visual evidence to produce stage-level and overall feedback. PolyDebate is implemented as both an immersive Unity 3D game version and a web platform version that share the same workflow and evaluation services.

In summary, this paper contributes (i) PolyDebate, an integrated 1v1 debatepractice system combining staged oral debate, an AI opponent, two interface versions, and learner-facing feedback; (ii) a game-orchestrated practice mechanism with skill-card guidance, props, and coins for in-game support and immediate feedback; and (iii) a rubric-aligned multimodal evaluation workflow linking transcript, speech/audio, and visual evidence to actionable learning feedback.

## 2 Related Work

AI debaters and LLM-based debate systems. Early autonomous debating systems showed that AI can retrieve evidence, construct arguments, and deliver rebuttals in competitive settings [4]. Agent4Debate uses multi-agent collaboration for argument preparation, analysis, writing, and review [5]; DebateBrawl combines LLMs with genetic algorithms and adversarial search for adaptive arguments [6]; and R-Debater improves multi-turn generation through retrievalaugmented argumentative memory [8]. These systems target AI debating performance rather than staged practice guidance, skill-card guidance, or multimodal learner feedback; PolyDebate instead uses the AI opponent within a learnerfacing 1v1 practice workflow.

Automated debate evaluation. Debatrix proposes a multi-dimensional LLM judge that analyzes performance across dimensions and chronological stages [7]. Debatable Intelligence shows that evaluating debate speeches requires complex judgments about argument strength, relevance, organization, style, and tone [9]. InspireDebate introduces InspireScore, combining subjective dimensions (e.g., emotional appeal, argument clarity) with objective ones (e.g., factual authenticity, logical validity) [10]. Multi-agent debate chatbots have also assessed learners’ critical thinking from their argumentation [3]. These methods, however, evaluate debate mainly through text and do not jointly cover argument content, oral delivery, visual behavior, and learner-facing feedback.

AI-supported debate learning. A ChatGPT-based debate game shows how prompt engineering can support topic selection, debate flow, dificulty control, and simple scoring [1]. In EFL training, generative pedagogical agents and teacher-guided scafolds have been used for role-driven debate preparation [2], and structured chatbots can guide learners through evidence, warrant, and rebuttal stages, mainly for critical-thinking assessment [3]. Such systems usually cover only part of the practice experience and rarely combine a complete staged round, speech interaction, an AI opponent, skill cards, game mechanics, and multimodal feedback in one playable environment. PolyDebate addresses this gap through a single 1v1 workflow.

## 3 PolyDebate

## 3.1 Overview and Workflow

PolyDebate combines staged oral debate, AI opponent interaction, rubric-aligned evaluation, and lightweight game mechanics in a single practice round, summarized in Fig. 1, and runs as a Unity 3D game version for immersive practice and a web platform version for browser access. Centered on a complete 1v1 round rather than an isolated chat, a session begins with a debate motion, assigns the learner and AI to opposing sides, and then follows a fixed four-stage sequence: constructive speech, cross-examination, rebuttal, and closing speech. Throughout, the system maintains a shared debate state (motion, sides, stage type, assigned skill cards, recent speeches, timing constraints, and game records) that the AI debater, the evaluation module, and the gamification layer all read from, keeping generation, assessment, and game feedback synchronized.

![](images/4d70dfb52ca24a71281d9f6f3a74e15b1112de3bbdfd45cd4ea26f2336b26e0a.jpg)  
Fig. 1. PolyDebate overview for gamified multimodal debate practice and rubricaligned feedback.

Each stage follows a repeated turn structure. The system assigns skill cards to the learner and the AI debater, shows stage-specific guidance, and lets the learner buy props before speaking. The learner’s speech is captured and transcribed, with audio and video evidence prepared for analysis. The AI debater then produces an opposing turn appropriate to its side, stage, the learner’s previous speech, and its assigned technique, and the evaluation module scores the turn under the four rubric categories, converting the score into coins and game feedback. After all stages, the turn records are aggregated into an overall evaluation with a final result, rating, strengths, weaknesses, and recommendations.

## 3.2 Gamified Practice Design

The gamification layer turns abstract debate strategies into explicit in-game guidance cues, choices, and outcomes. At the start of each stage, both the learner and the AI debater are assigned skill cards that specify concrete techniques (e.g., Data-Driven, Chain of Reasoning, Address Opponent, Emotional Appeal), so both sides of the exchange are shaped by explicit techniques rather than free-form conversation. Figure 1 lists the full skill-card deck and props shop.

Coins and props provide immediate feedback and learner agency. After each evaluated turn the score is converted into coins, and cumulative coins decide the final outcome. Before the next stage, learners can spend coins in the props shop on limited support or strategic efects (the full set is listed in Fig. 1). These mechanics do not replace the rubric evaluation; they translate assessment results into visible progress and make repeated practice more engaging. Together, stage control, skill cards, AI opponent behavior, rubric-based scoring, and coin outcomes turn a debate round into a complete practice loop.

## 3.3 AI Debater

The AI debater is a stage-aware opponent rather than a general question-answering agent. For each AI turn, the generation module receives a structured context: the motion, both sides, the current stage, the learner’s latest speech, the debate history, and the AI’s skill card. This keeps the response side-consistent and stage-appropriate, for example, by establishing a position in the constructive stage, probing the opponent in cross-examination, attacking weak points in rebuttal, and summarizing the strongest clash points in closing. Because the same skill-card representation used by the learner is also applied to AI generation, the response is not only an opposing argument but also a visible example of a debate technique in context.

After text generation, the response is synthesized with Edge-TTS (a Microsoft neural English voice) and passed to the LiveTalking digital-human module, where a Wav2Lip-based lip-synchronization model [11] drives the avatar’s mouth movements. The audio-visual stream is delivered over WebRTC, so the opponent appears as a speaking debate partner rather than a text-only agent, and each response is stored in the debate history for later turns and the final evaluation.

## 3.4 AI Evaluation of Debate Performance

The evaluation module is organized around a rubric adapted from ELC2012, an undergraduate English course on persuasive communication at The Hong Kong Polytechnic University. This course rubric provides the pedagogical basis for four categories: Analysis, Persuasiveness, Clarity, and Appropriacy. It anchors the AI evaluation in an existing teaching context and makes feedback more interpretable.

Following this rubric, PolyDebate maps multimodal evidence onto the same four categories, as detailed in the rubric board of Fig. 1. Analysis draws on the transcript for argument/logos and opponent targeting; Persuasiveness adds video evidence for non-verbal delivery; Clarity adds audio evidence for pronunciation and fluency; and Appropriacy adds audio and video evidence for audience appeal and delivery appropriateness. The categories are weighted 30%, 30%, 25%, and 15%, respectively.

The turn-level evaluator takes the stage, motion, learner side, transcript, opponent context, skill card, and audio/video descriptors, and returns category scores and item-specific feedback, which is converted into coins. A final evaluator then aggregates the debate history, multimodal evidence, and turn-level results into an overall rating, strengths, weaknesses, and recommendations, so feedback reflects the full trajectory rather than a single utterance.

![](images/1900217f1648a46b88282f70fb20a66013346905dca830a2f6a5a3466521ffca.jpg)  
Fig. 2. Demonstration walkthrough of a complete 1v1 PolyDebate session in the Unity 3D game version.

## 4 Demonstrations

Figure 2 walks through a complete 1v1 session in the immersive Unity 3D arena, with each panel showing one step of the practice-evaluation loop: from session start and side assignment, through the constructive, cross-examination, rebuttal, and closing stages, to the final coin outcome and overall feedback. Figure 3 shows representative screens from the browser-based web platform version, which shares the same workflow, and the accompanying video presents both versions at https://youtu.be/mHwBG1\_8Ebk.

## 5 Experiments and Evaluation

We evaluated PolyDebate through four analyses: AI opponent next-turn response quality, evaluation coverage against representative frameworks, AI judge feedback quality with ablations, and a user-perception study.

## 5.1 AI Opponent Next-Turn Response Quality

This experiment evaluates the AI debater as a practice opponent: given a learner’s previous turn and the current context, each system produces one opposing response. We sampled 100 next-turn cases from the Intelligence Squared Debates Corpus [12] distributed with ConvoKit [13]. The cases cover eight public debates and were mapped to the four PolyDebate stages. Each case contains a human debate utterance treated as the learner-side turn, the opposite AI side, up to two previous turns as context, and an assigned skill card. All variants used the same GPT-5.4 mini model for generation. An independent LLM then scored the three anonymized responses per case on a 1–5 scale: six criteria adapted from InspireScore [10] cover subjective and objective quality, and three skill-usage criteria check whether the target technique is visibly demonstrated as a pedagogical example. The overall score averages the subjective, objective, and skill scores.

![](images/4557656e3d2ac43d0d3d33e70bc15a6b9943f6f68a95a9fc9d34b3d824810924.jpg)  
Fig. 3. Representative key interfaces of the PolyDebate web platform version.

Table 1 reports the results. PolyDebate achieves the highest overall score (4.0) against the generic LLM opponent (3.1) and the stage-only opponent (3.6). The largest gain is in skill usage (2.1 to 3.9): PolyDebate visibly demonstrates target techniques within the opponent’s response, showing that skill cards are more than interface guidance cues and also shape the AI into a stage-aware pedagogical example.

Table 1. AI opponent next-turn response quality. Scores use a 1–5 scale.
<table><tr><td></td><td colspan="5">Subjective quality</td><td colspan="3">Objective quality</td><td colspan="4">Skill usage</td><td></td></tr><tr><td>Variant</td><td></td><td></td><td></td><td></td><td>EA AC AA TR Avg.</td><td></td><td>FA LV</td><td>Avg.</td><td>Pres. Corr. Vis. Avg. Overall</td><td></td><td></td><td></td><td></td></tr><tr><td>Generic LLM opponent</td><td>3.2</td><td>4.0</td><td>3.2</td><td>3.8</td><td>3.6</td><td></td><td>3.6 3.9</td><td>3.8</td><td>2.0</td><td>1.8</td><td>2.6</td><td>2.1</td><td>3.1</td></tr><tr><td>Stage-only opponent</td><td>3.7</td><td>4.0</td><td>3.7</td><td>4.5</td><td></td><td>4.0</td><td>3.7 3.9</td><td>3.8</td><td>2.8</td><td>2.6</td><td>3.2</td><td>2.9</td><td>3.6</td></tr><tr><td>PolyDebate opponent</td><td>3.8</td><td>4.0</td><td>4.3</td><td>4.6</td><td></td><td>4.2</td><td>3.8 4.0</td><td>3.9</td><td>4.2</td><td>3.5</td><td>4.1</td><td>3.9</td><td>4.0</td></tr></table>

EA=emotional appeal; AC=argument clarity; AA=audience adaptation; TR=topic relevance; FA=factual authenticity; LV=logical validity; Pres.=skill presence; Corr.=skill correctness; Vis.=pedagogical visibility.

## 5.2 Debate Evaluation Coverage

We compare PolyDebate’s evaluation coverage with representative frameworks: Debatrix [7], InspireScore [10], Debatable Intelligence [9], and a multi-agent debate chatbot for critical-thinking assessment [3]. As shown in Table 2, these methods evaluate debate mainly from text and cover aspects such as argument quality, relevance, organization, language use, and persuasive strength, but none jointly covers argument content, oral delivery, visual behavior, and learner-facing feedback together. PolyDebate adapts the ELC2012 rubric into a multimodal framework spanning text, audio, and video across analysis, persuasiveness, clarity, and appropriacy, and converts the results into structured feedback with strengths, weaknesses, and recommendations. This provides broader coverage and is better suited to formative learning.

Table 2. Comparison of representative debate and debate-context assessment frameworks in terms of modality, assessed aspects, and learner-facing feedback output.
<table><tr><td>Framework</td><td colspan="2">Modality</td><td colspan="2">Analysis</td><td colspan="2">Persuasiveness</td><td colspan="2">Clarity</td><td colspan="2">Appropriacy</td><td colspan="3">Feedback output</td></tr><tr><td></td><td>TA</td><td>V</td><td>Arg.</td><td> Opp. Pathos Strat. Org. Pron. Gram. Task Ethos Tone Str. Weak. Rec.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Debatrix</td><td>√</td><td></td><td></td><td></td><td>△</td><td>△</td><td></td><td>△</td><td></td><td></td><td>△</td><td>△</td><td>△</td></tr><tr><td>InspireScore</td><td>√</td><td></td><td></td><td>△</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Debatable Intelligence</td><td>√</td><td></td><td></td><td></td><td></td><td>△</td><td></td><td>△</td><td></td><td></td><td>△</td><td></td><td></td></tr><tr><td>MA Debate Chatbot (CT)</td><td>√</td><td></td><td></td><td></td><td>△</td><td>△</td><td></td><td></td><td>△</td><td></td><td>√</td><td>△</td><td></td></tr><tr><td>PolyDebate</td><td>√ √</td><td>V</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>△√</td><td>√</td><td>√</td></tr></table>

✓=covered; △=partial; blank=not covered. T=text; A=audio; V=video. Arg.=argumentation/logos; Opp.=opponent targeting; Pathos=emotional/non-verbal appeal; Strat.=rhetorical/argumentative strategies; Org.=organization/relevance; Pron.=pronunciation/fluency; Gram.=grammar/lexis; Task=task fulfillment/stage appropriateness; Ethos=credibility/audience appeal; Tone=language/interpersonal appropriateness; Str.=strengths; Weak.=weaknesses; Rec.=recommendations.

## 5.3 AI Judge Feedback Quality and Ablation

We next evaluated whether the AI judge produces complete, specific, actionable, and grounded feedback. Using the same GPT-5.4 mini model, we compared the full PolyDebate judge against a generic baseline and four ablations that remove the rubric, the skill card, the multimodal evidence, or the structured feedback schema. The evaluation used 100 multimodal samples (transcript, audio, and video) whose weakness labels follow the ten subcategories in Table 2.

Table 3 shows the full judge performing best on all metrics: against the generic judge, it substantially improves weighted rubric coverage and yields more specific, actionable, and grounded feedback. The ablations confirm each component’s contribution. Removing the rubric sharply reduces coverage, while removing the skill card or the multimodal evidence reduces weakness F1. Therefore, the full judge gives the most comprehensive formative diagnosis while preserving high feedback quality.

## 5.4 User Perception Study

Finally, we ran a lightweight study with ten students interested in English learning or debate. Each participant tried both the Unity 3D and web platform versions and then completed the same 5-point Likert questionnaire over six dimensions: usability, AI opponent usefulness, AI judge feedback usefulness, skill-card support, gameful motivation, and overall value.

Table 3. AI judge feedback quality and component ablation. Coverage and weakness F1 are percentages; specificity, actionability, and groundedness use a 1–5 scale.
<table><tr><td>Variant</td><td>W.Cov. W.F1 Spec. Act. Grd.</td><td></td><td></td><td></td><td></td></tr><tr><td>Generic LLM feedback judge</td><td>46.7</td><td>80.6</td><td>4.0</td><td>4.1</td><td>4.1</td></tr><tr><td>Full w/o rubrics</td><td>51.3</td><td>64.3</td><td>4.4</td><td>4.5</td><td>4.7</td></tr><tr><td>Full w/o skill card</td><td>81.7</td><td>59.6</td><td>4.4</td><td>4.4</td><td>4.6</td></tr><tr><td>Full w/o multimodal evidence</td><td>71.7</td><td>32.2</td><td>4.2</td><td>4.2</td><td>4.1</td></tr><tr><td>Full w/o feedback schema</td><td>76.2</td><td>74.8</td><td>4.8</td><td>4.8</td><td>4.8</td></tr><tr><td>Full PolyDebate judge</td><td>99.2</td><td>85.9</td><td>4.9</td><td>4.9</td><td>4.9</td></tr></table>

W.Cov.=weighted rubric coverage; W.F1=weakness-label F1; Spec.=specificity; Act.=actionability; Grd.=groundedness.

![](images/14478eafe396da03ef010d005721041c8e370bb42909f773c33d35d336436652.jpg)  
Fig. 4. User-perception results for the Unity 3D and web platform versions of PolyDebate. Bars report mean 5-point Likert ratings from ten students across six questionnaire dimensions.

Figure 4 summarizes the mean scores: both versions were rated positively on all dimensions. The web version scored higher on usability, feedback usefulness, skill-card support, and overall value, while the Unity version scored higher on gameful motivation, and AI opponent usefulness was similar across versions. The two are thus complementary: the web version suits accessible, repeated practice, while the Unity version supports immersive, game-like engagement.

## 6 Conclusion and Future Work

We presented PolyDebate, a game-orchestrated multimodal system for English debate practice and evaluation that connects staged 1v1 interaction, skill-card guidance, props and coins, an AI opponent, multimodal evidence, rubric-aligned evaluation, and structured feedback. PolyDebate is available in both an immersive Unity 3D version and a web platform version that share one workflow. Our evaluation shows that PolyDebate produces skill-aware opponent responses, broadens assessment beyond text-only evaluation, and presents the practice loop in a usable and motivating form.

Future work will explore learner-profile-driven practice that uses prior-round feedback, additional formats such as team-based and multi-role debate, and classroom deployment to examine learning gains.

Acknowledgments. This research work was conducted at the Research Institute for Artificial Intelligence of Things (RIAIoT) and The Institute for Higher Education Research and Development (IHERD) of PolyU. This work was supported in part by the Hong Kong Research Grants Council Theme-based Research Scheme (No. T43- 518/24-N), PolyU Internal Research Fund (No. BDZ3), and PolyU LTC Project (Grant TDLEG25-28/IICA/P/05, No. 48EM).

## References

1. Lee, E.Y., Ngagaba Gogo, D.I., An, G.H., Lee, S., Lim, K.: ChatGPT-based debate game application utilizing prompt engineering. In: Proc. RACS. pp. 1–6 (2023). https://doi.org/10.1145/3599957.3606244

2. Cassim, F., Yang, J.C.: GAI pedagogical agents and teacher-guided prompt use in EFL debate training. In: Proc. ICCE (2025)

3. Park, B., Seo, K.: Assessing critical thinking through a multi-agent LLM-based debate chatbot. In: Proc. CHI EA. pp. 80:1–80:13 (2025). https://doi.org/10.1145/3706599.3721207

4. Slonim, N., Bilu, Y., Alzate, C., Bar-Haim, R., Bogin, B., et al.: An autonomous debating system. Nature 591(7850), 379–384 (2021). https://doi.org/10.1038/s41586-021-03215-w

5. Zhang, Y., Yang, X., Feng, S., Wang, D., Zhang, Y., Song, K.: Can LLMs beat humans in debating? a dynamic multi-agent framework for competitive debate. In: Proc. ICASSP. pp. 19377–19381 (2026). https://doi.org/10.1109/ICASSP55912.2026.11460889

6. Aryan, P.: LLMs as debate partners: Utilizing genetic algorithms and adversarial search for adaptive arguments (2024), arXiv:2412.06229

7. Liang, J., Ye, R., Han, M., Lai, R., Zhang, X., Huang, X., Wei, Z.: Debatrix: Multi-dimensional debate judge with iterative chronological analysis based on LLM. In: Findings ACL. pp. 14575–14595 (2024). https://doi.org/10.18653/v1/2024.findings-acl.868

8. Li, M., Wang, Z., Li, H., Liu, J.: R-Debater: Retrieval-augmented debate generation through argumentative memory. In: Proc. AAMAS. pp. 910–919 (2026). https://doi.org/10.65109/XWXH6253

9. Sternlicht, N., Gera, A., Bar-Haim, R., Hope, T., Slonim, N.: Debatable intelligence: Benchmarking LLM judges via debate speech evaluation. In: Proc. EMNLP. pp. 18850–18869 (2025). https://doi.org/10.18653/v1/2025.emnlp-main.953

10. Wang, F., Li, J., Zhu, K., Jiang, C.: InspireDebate: Multi-dimensional subjectiveobjective evaluation-guided reasoning and optimization for debating. In: Proc. ACL. pp. 27525–27544 (2025). https://doi.org/10.18653/v1/2025.acl-long.1335

11. Prajwal, K.R., Mukhopadhyay, R., Namboodiri, V.P., Jawahar, C.V.: A lip sync expert is all you need for speech to lip generation in the wild. In: Proc. ACM MM. pp. 484–492 (2020). https://doi.org/10.1145/3394171.3413532

12. Zhang, J., Kumar, R., Ravi, S., Danescu-Niculescu-Mizil, C.: Conversational flow in Oxford-style debates. In: Proc. NAACL-HLT. pp. 136–141 (2016). https://doi.org/10.18653/v1/N16-1017

13. Chang, J.P., Chiam, C., Fu, L., Wang, A., Zhang, J., Danescu-Niculescu-Mizil, C.: ConvoKit: A toolkit for the analysis of conversations. In: Proc. SIGDIAL. pp. 57–60 (2020). https://doi.org/10.18653/v1/2020.sigdial-1.8