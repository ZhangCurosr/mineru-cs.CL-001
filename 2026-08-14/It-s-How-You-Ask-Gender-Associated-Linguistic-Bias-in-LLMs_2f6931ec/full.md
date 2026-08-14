# It’s How You Ask: Gender-Associated Linguistic Bias in LLMs

Katherine Van Koevering Data Science and AI Institute Johns Hopkins University kvankoe1@jh.edu

Anjalie Field   
Computer Science Department   
Johns Hopkins University   
anjalief@jhu.edu

## Abstract

Professional communication is increasingly mediated by LLMs — but do these models serve all users equally? We show that when prompts contain linguistic features more commonly used by women (hedges, tag questions, collective reference), they systematically elicit shorter, less sophisticated, and less formal responses across three document types and four models. These effects persist after controlling for prompt complexity and feature carry-over. Explicit gender cues like sign-off names are encoded in the same representational space as linguistic dialect — suggesting shared underlying mechanisms — yet linguistic register is far more influential, producing large, consistent effects where names produce none. Our results further reveal that post-hoc mitigation is challenging: because these patterns are culturally embedded and outside conscious control, users cannot easily avoid them through strategic self-presentation, and mechanistic analysis reveals that linguistic features are encoded in early transformer layers and entangled with other features. Our work calls for upstream consideration of the influences of linguistic variation to mitigate disparate impacts of LLM-mediated workplace communication.

## 1 Introduction

People increasingly rely on LLMs to support professional communication, including drafting and revising emails, job applications, and other workplace documents (Zhao et al., 2024; Bassignana et al., 2025). Additionally, the workplace has long been a site of gender inequality, with extensive evidence that women face systematic disadvantages in professional contexts (Heilman, 2012; Crompton, 1987). This shift raises a pressing question: how might these tools influence gender equality in the workplace?

Men and women consistently differ in how they write and speak (Argamon et al., 2003), including in workplace email and formal documents (Colley & Todd, 2002; Atifi & Marcoccia, 2020; Prabhakaran et al., 2014). Women’s language is more likely to include hedges (e.g., maybe, I think), tag questions (e.g., isn’t it?), collective reference (e.g., we, our), and expressive adjectives (e.g., lovely, wonderful).<sup>1</sup> Crucially, these patterns are largely unconscious and culturally embedded (Eckert, 2000), meaning speakers cannot easily avoid or modify them. Given these systematic differences, if LLMs respond to gender-associated language patterns in prompts, this could systematically differentiate how LLMs perform for users with varying linguistic patterns - potentially disadvantaging users with women-associated linguistic patterns in professional contexts.

Substantial research has demonstrated that LLMs encode social biases (Cheng et al., 2023; Bianchi et al., 2023; Wan et al., 2023; Bai et al., 2025; Wan & Chang, 2025). Yet existing bias studies focus predominantly on how models depict people — in narratives, profiles, or decision-making scenarios — leaving a critical gap: whether models perform differentlyfor different users. Simultaneously, models are known to be sensitive to subtle prompt variation, including formatting (Sclar et al., 2024) and linguistic features (Hofmann et al., 2024; Deas et al., 2023), but this work has focused on more explicit language artifacts or dialects, while gender-associated linguistic differences mark a much more subtle effect.

<table><tr><td>Prompt: Compose an email to schedule a Mid-year review with your team Response:Subject: Mid-Year Review Schedule Dear Team, As we approach the midpoint of the year, I would like to take this opportunity to evaluate our progress, discuss challenges, and set new goals for the second half of the year. I am scheduling a mid-year review for our team and would like to invite each of you to participate. The purpose of this review is to: - Evaluate our performance against the goals set at the beginning of the year - Discuss any challenges or obstacles we have faced and identify so...</td><td>Prompt: Let&#x27;s compose an email together to arrange our mid-year appraisal with our team Response: I’d be happy to help you compose an email to arrange your mid-year appraisal with your team. To start, let&#x27;s begin with a basic structure. Here&#x27;s a draft: Subject: Mid-Year Appraisal - Scheduling a Meeting Dear [Team Members], I hope this email finds you well. As we approach the middle of the year, I would like to schedule our mid-year appraisals to discuss our goals and progress so far. This will be a great opportunity for us to reflect on our accomplishments, set new targets, and align our obj..</td></tr></table>

Figure 1: An example of two modified prompts and responses to each. The left prompt is direct with no collective pronouns, presenting linguistic features documented as more common for men. The right prompt uses collective pronouns, presenting linguistic features documented as more common for women. The response to the left prompt uses more complex language than the right prompt.

In this work, we investigate this risk directly. We construct a controlled evaluation framework using real workplace prompts from the WildChat corpus (Zhao et al., 2024), systematically perturbing them to include women-associated linguistic features, and measure how these differences shape model outputs across dimensions of length, complexity, formality, and style. We carefully control for style-mirroring as a confound, and conduct an interpretability analysis to reveal the internal mechanisms underlying observed differences.

Women-associated language in prompts leads to responses that are shorter, less sophisticated, more readable, and less formal, as illustrated in Figure 1. These effects are robust across task types and cannot be fully explained by models mirroring input style. Strikingly, these implicit linguistic cues have a stronger influence on model behavior than explicit gender markers such as names, yet the underlying information is encoded in nearly identical locations in the model, suggesting that LLMs may use linguistic register as a proxy for gender to produce stereotyped outputs. Because the patterns that trigger this bias are unconscious and culturally embedded, users cannot easily avoid them, and current debiasing approaches focused on explicit markers like names and pronouns are insufficient to address it. Instead, our work calls for more nuanced upstream consideration of how models respond to linguistic features of its users, and how these responses may lead to disparity in downstream tasks among groups of users.

## 2 Designing Prompts with Gender-associated Linguistic Features

We design a controlled prompt manipulation experiment to test whether gender-associated linguistic patterns in prompts affect LLM outputs in professional communication tasks. Specifically, we manipulate prompt text to mirror documented women-associated and men-associated language patterns. In order to do this, we first identify gendered linguistic features from prior work, although we note that these features are not gender exclusive and results will hold for anyone who uses these linguistic patterns. Next, we collect realworld prompts that attempt a variety of professional tasks. Finally, we use GPT-4 to inject our linguistic features into these prompts, to manipulate them into having more women associated linguistic features (WALF) or more men associated linguistic features (MALF). We then verify that our injection was successful and that these features also appear in real-world prompts.

## 2.1 Methodology

Identification of gender-associated linguistic features We first identify language patterns that sociolinguistic studies have consistently associated with men and women.

Gendered variation in language is well-documented across spoken and written English (Argamon et al., 2003). Drawing on Lakoff’s foundational framework (Rahadiyanti, 2020; Yanti et al., 2021; Lakoff, 1973) and subsequent empirical validation (Ruanan & Jun, 2020), we operationalize four women-associated and two men-associated feature classes.

Hedges (e.g., maybe, perhaps, I think, sort of) are uncertainty markers that soften assertions and reduce directness. Women use hedges at substantially higher rates than men across conversational and written and spoken contexts (Ginarti et al., 2022; Aziz et al., 2022; Salman et al., 2023). Tag questions (e.g., isn’t it?, don’t you think?) are clause-final interrogatives appended to declaratives. Women use them at roughly twice the rate of men in empirical studies (Aziz et al., 2022; Salman et al., 2023). Expressive adjectives (e.g., lovely, wonderful) are affectively charged modifiers that Lakoff identified as characteristic of women’s speech (Lakoff, 1973). They are used significantly more frequently by women than men (Putri et al., 2020). Collective reference (e.g., we, our, together) are favored by women while men demonstrate greater use of individual reference (I, my) and self-mentions (Jie, 2024; Argamon et al., 2003). For men-associated language, we operationalize the complementary patterns: minimized hedging with direct assertions, absence of tag questions, individual rather than collective reference, and neutral rather than expressive adjectives. Men additionally show greater use of quantifiers and determiners (Rubin & Greene, 1992)

Data Description We sampled prompts from WildChat-4.8M (Zhao et al., 2024), a largescale corpus of real user–LLM interactions. We identified writing requests using regular expressions matching trigger phrases: “write/compose/draft an email” for emails; “write [verb] + job application or cover letter” for job applications; and “write [verb] + resignation letter” for resignation letters. Write verbs included generate, write, compose, create, draft, revise, edit, rewrite, and update. We sampled 427 valid prompts distributed across three categories: 200 emails, 200 job applications, and 27 resignation letters (limited by corpus availability).<sup>2</sup>

Injection An ideal experiment would exhibit a paired test. A given prompt would be written using men-associated language and women-associated language and then the differences in responses would be compared. We simulate this, by modifying the prompt artificially to set up this paired test. Each valid prompt was rewritten into two versions using GPT-4. The WALF version was constructed by injecting hedges, tag questions, collective nouns, and expressive adjectives while preserving the core task semantics. The MALF version removed hedges and tag questions, replaced collective with individual reference, and substituted neutral adjectives for expressive ones.<sup>3</sup> This setup gives us two versions of each prompt that ask for the same task, but with subtle differences in language.

## 2.1.1 Validation

To assess whether our rewriting pipeline preserved the underlying task while changing linguistic register, we ran two human validation studies. First, we sampled 60 matched prompt pairs (30 WALF and 30 MALF) from the rewritten set, constrained to prompts under 500 characters. Four annotators rated whether each rewritten prompt preserved the same underlying task as its original WildChat source on a 5-point scale from fundamentally different things to identical things; we exclude one annotator identified as an outlier who misunderstood instructions, leaving three annotators for the reported statistics. Majority vote judged 57/60 pairs (95.0%) to preserve the same task, and 91.7% of pooled ratings were at least somewhat similar to the original prompt (score ≥ 3).

Second, we conducted a realism check on rewritten prompts. Three annotators rated 54 rewritten prompts, 20 matched real WildChat prompts, and 5 wildcard WildChat prompts selected to be lexically dissimilar from the rewritten prompts, which served as calibration items to span a wider range of prompt realism. Prompts were rated on a 5-point realism scale from unintelligible or incoherent to this seems like a real prompt. Rewritten prompts were rated less realistic than real WildChat prompts overall, but remained broadly plausible. Real prompts had mean realism $4 . 3 7 \pm 1 . 2 1$ , while rewritten prompts had mean realism 3.84 ± 1.22 on the 5-point scale. We report the detailed distribution and inter-annotator agreement in Appendix A. These validation studies support that the rewritten prompts largely preserve task semantics while introducing controlled register variation. Additional details on the human validation studies, including annotation instructions, rating distributions, agreement statistics, and representative prompt pairs, are provided in Appendix A; example original/WALF/MALF prompt triples are shown in Table 6.

To verify our injection schema, we check for these features in the modified prompts using custom regex. Across all categories, WALF prompts contained significantly more hedges, tag questions, expressive adjectives, and collective nouns than MALF prompts (all $p < 0 . 0 0 1 )$ (Appendix Table 8). Additionally, we find that these features do appear in real-world prompts in both the Wildchat and Mila datasets of AI usage queries (Bassignana et al., 2025; Zhao et al., 2024) (Table 1).

Table 1: Percentage of all full-sentence features in Wildchat and Mila datasets with given features. All features we study are present in real prompts, although frequency varies significantly.

## 3 Effects of Gender-associated Features in Prompts on LLM Responses

<table><tr><td>Feature</td><td>Wildchat</td><td>Mila</td></tr><tr><td>Expressive</td><td>14.7</td><td>4.7</td></tr><tr><td>Collective</td><td>4.4</td><td>2.4</td></tr><tr><td>Hedges</td><td>5.1</td><td>2.1</td></tr><tr><td>Tag questions</td><td>0.66</td><td>0.36</td></tr><tr><td>Quantifiers</td><td>57.7</td><td>32.2</td></tr><tr><td>Determiners</td><td>64.3</td><td>69.7</td></tr></table>

We next investigate whether differences in prompt language affect model responses. We set up a variety of established metrics to measure response complexity, politeness, clout, and formality. We then run paired t-tests to identify whether the

MALF and WALF prompts elicit substantially different responses.

## 3.1 Methodology

Complexity We measure six complexity metrics. Word count and tokens capture response length and lexical breadth, both validated predictors of writing quality (Rossetti & Waes, 2022; Kapoor et al., 2025). Lexical sophistication is measured as mean word length, a simple proxy for lexical complexity (He & Yiu, 2022; Paetzold & Specia, 2016). Readability uses the Flesch Reading Ease score (Flesch, 1948) and grade level uses the Flesch-Kincaid Grade Level formula (Kincaid et al., 1975); both are widely validated across educational and professional contexts (Imperial & Ong, 2021; Zhang & Zhang, 2022; Rets et al., 2020; Crossley, 2024). Finally, type-token ratio (TTR) measures lexical diversity as the ratio of unique word types to total tokens, with higher values indicating greater sophistication (Xia et al., 2016; Carrasco-Farre, 2022; Yu et al., 2024).´

Style We measure three stylistic features. Politeness density is the proportion of politeness markers per response, operationalized using a computational classifier grounded in Brown and Levinson’s politeness theory (Brown & Levinson, 1987; Nguyen et al., 2016; Aljanaideh et al., 2020). Formality uses the F-measure (Heylighen & Dewaele, 1999), which captures the ratio of noun-like to verb-like constructions, with higher scores indicating more formal register (Jumanto, 2015). Clout score is a LIWC-derived measure of confidence and social authority, where higher scores reflect language associated with leadership and assertiveness (Kacewicz et al., 2014).

## 3.2 Results

Complexity A consistent pattern emerges across models and categories: WALF prompts elicit responses that are more readable, lower in lexical sophistication, and require lower grade levels — suggesting models generate plainer, more accessible language in response to women-associated linguistic features. Effects are strongest for emails and job applications and weakest for resignation letters, where only readability shows a significant difference in some models (Table 2). As LLMs become more ubiquitous in professional settings, this model behavior could exacerbate existing gender biases. 4

Style Among stylistic features, the most consistent effect is for formality, which shows significant differences across all three writing categories (email: $\mathrm { { p } < 0 . 0 \check { 0 } 1 }$ ; job application: $\mathrm { \tt ~ p ~ } < 0 . 0 0 1$ ; resignation letter: $\mathrm { ~ p ~ } = \ 0 . 0 3 3 )$ with MALF prompts consistently eliciting more formal re-

Table 2: Effect of prompt linguistic feature (WALF vs MALF) on response complexity metrics, by model and document category. Cells show direction (M or W) and significance of a paired t-test; - = not significant. $^ { * } p < . 0 5 , ^ { * * } p < . 0 1$ $^ { * * * } p ^ { ^ { \mathbf { \hat { } } } } <$ .001. Resignation letter: $n \ = \ 2 7$ per linguistic feature per model. Results that support the hypothesis that MALF prompts result in more complex responses than WALF prompts are bolded.
<table><tr><td colspan="2">Metric</td><td>GPT</td><td>Gemma</td><td>Mistral</td><td>Llama</td></tr><tr><td rowspan="6">Emaiil</td><td>Word count</td><td>M***</td><td></td><td></td><td></td></tr><tr><td>Sophistication</td><td> $\mathbf { M } ^ { * * * }$ </td><td> $\mathbf { M } ^ { * * * }$ </td><td> $\mathbf { M } ^ { * * * }$ </td><td> $\mathbf { M } ^ { \ast \ast \ast }$ </td></tr><tr><td>Tokens</td><td></td><td> $\mathbf { M } ^ { * * * }$ </td><td>W*</td><td> $\mathbf { M } ^ { \ast \ast \ast }$ </td></tr><tr><td>Readability</td><td> $\mathbf { W } ^ { * * * }$ </td><td> $\mathbf { W } ^ { * * * }$ </td><td>W***</td><td> $\mathbf { W } ^ { * * * }$ </td></tr><tr><td>Grade level TTR</td><td> $\mathbf { M } ^ { * * * }$ </td><td> $\mathbf { M } ^ { * * * }$ </td><td></td><td> $\mathbf { M } ^ { \ast \ast \ast }$ </td></tr><tr><td></td><td> $W ^ { * * * }$ </td><td></td><td></td><td>W*</td></tr><tr><td rowspan="6">opt on</td><td>Word count</td><td></td><td></td><td></td><td></td></tr><tr><td>Sophistication</td><td>M*</td><td>M***</td><td></td><td></td></tr><tr><td>Tokens</td><td></td><td>M**</td><td>W*</td><td>M*</td></tr><tr><td>Readability</td><td>W**</td><td>W***</td><td></td><td>W***</td></tr><tr><td>Grade level</td><td></td><td> $\mathbf { M } ^ { * * * }$ </td><td></td><td> $\mathbf { M } ^ { \ast \ast \ast }$ </td></tr><tr><td>TTR</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Resn ester</td><td>Word count</td><td></td><td></td><td></td><td></td></tr><tr><td>Sophistication</td><td>M*</td><td></td><td></td><td>M**</td></tr><tr><td>Tokens</td><td></td><td></td><td></td><td></td></tr><tr><td>Readability</td><td></td><td></td><td></td><td> $\mathbf { W } ^ { * * }$ </td></tr><tr><td></td><td>Grade level</td><td></td><td></td><td></td><td> $\mathbf { M } ^ { * * }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>TTR</td><td></td><td></td><td></td><td></td></tr></table>

sponses (Appendix E, Table 9). By contrast, politeness density and clout score show no significant differences in any category — a striking null result given that politeness is 7–60× higher in WALF prompts. This suggests models calibrate politeness to the writing task and genre rather than to the register of the user’s request. We additionally report a preliminary LLM-as-a-judge evaluation on a subset of job-domain responses in the Appendix.

## 4 Testing Mirroring Effects

The results above demonstrate clear complexity differences between responses to WALF and MALF prompts. We now test how much of these differences can be explained by simply mirroring the prompt style or features. If we cannot explain from mirroring, then this

suggests that the model is introducing complexity differences based on gender-associated linguistic cues that are not directly signified in prompts. We use GPT-4 for these tests.

## 4.1 Methodology

Prompt Complexity and Style One straightforward explanation is that the linguistic manipulation made prompts more or less complex or formal, and models simply mirrored those properties in their responses. To test this, we fit separate standardized OLS regressions within each document category, predicting each response-level metric from the corresponding prompt-level metric. For complexity, this includes regressions such as response word count on prompt word count and response readability on prompt readability. For style, this includes regressions such as response formality on prompt formality, response politeness density on prompt politeness density, and response clout on prompt clout.

Feature Carry-Over A second explanation is that carried-over linguistic features themselves drive complexity differences — for example, responses containing more hedges might naturally score differently on readability. We test whether response-level feature carry-over mediates the prompt-condition effect using bootstrap mediation analysis with 2,000 resamples. The treatment is prompt condition (WALF vs. MALF), the outcomes are response-level complexity metrics, and the mediators are response-level counts of hedges, tag questions, collective nouns, and expressive adjectives, entered jointly. Full mediation would indicate that complexity differences are explained entirely by feature carry-over; partial or absent mediation would suggest the model is adjusting its overall register — not merely echoing injected features back.

Table 3: Prompt-level complexity predicting response-level complexity $( R ^ { 2 } ,$ within category).
<table><tr><td>Category</td><td>Word count</td><td>Sophistication</td><td>Unique words</td><td>Readability (Flesch)</td><td>Grade level</td><td>TTR</td></tr><tr><td>Email</td><td>0.065</td><td>0.003</td><td>0.099</td><td>0.080</td><td>0.003</td><td>0.105</td></tr><tr><td>Job application</td><td>0.188</td><td>0.032</td><td>0.341</td><td>0.254</td><td>0.074</td><td>0.256</td></tr><tr><td>Resignation letter</td><td>0.331</td><td>0.017</td><td>0.234</td><td>0.048</td><td>0.058</td><td>0.260</td></tr></table>

Standardised β<sup>ˆ</sup>: <sup>∗</sup> p < 0.05, <sup>∗∗</sup> p < 0.01, <sup>∗∗∗</sup> p < 0.001; unmarked = ns.

## 4.2 Results

Prompt Complexity and Style We find weak relationships throughout (Table 3). Even the strongest predictors — unique word count in job applications $( \mathrm { R } ^ { 2 } = \mathrm { 0 } . 3 4 1 )$ and readability in job applications $( \mathrm { R } ^ { 2 } = 0 . 2 5 \dot { 4 } )$ — explain at most 34% of response complexity variance. The same pattern holds for stylistic measures, where R² values range from 0.001 to 0.081 (median $= 0 . 0 \dot { 2 } 9 )$ , with formality in emails as the strongest predictor at just 3.7% despite prompt formality differing by 15 F-measure points between conditions. Simple prompt mirroring cannot account for the observed response differences.

Table 4: $R ^ { 2 }$ of the full four-predictor OLS model (hedges, tag questions, collective nouns, expressive adjectives) predicting response-level complexity, by document category.
<table><tr><td>Metric</td><td>Email</td><td>Job application</td><td>Resignation Letter</td></tr><tr><td>Word count</td><td>0.342</td><td>0.126</td><td>0.226</td></tr><tr><td>Sophistication</td><td>0.037</td><td>0.028</td><td>0.125</td></tr><tr><td>Unique words</td><td>0.334</td><td>0.129</td><td>0.193</td></tr><tr><td>Reađability (Flesch)</td><td>0.006</td><td>0.068</td><td>0.094</td></tr><tr><td>Grade level</td><td>0.085</td><td>0.026</td><td>0.141</td></tr><tr><td>TTR</td><td>0.037</td><td>0.024</td><td>0.147</td></tr></table>

Feature Carry-Over We find only partial mediation for few outcomes. Response-level linguistic features predict complexity moderately in isolation (e.g., email word count R² = 0.342), but substantial complexity differences remain unexplained by feature carry-over alone (Table 4).

## 5 Analysis of Model Encoding

The behavioral results establish that models respond differently to WALF and MALF prompts, producing outputs that differ systematically in complexity and style. To understand the internal basis of these effects, we first attempt an experiment on implicit versus explicit gender encoding by assigning a random name to prompts to see if the gender of the name affects out results. We then apply mechanistic interpretability methods to Llama-3.2- 3B-Instruct. We investigate two questions: (1) where in the network linguistic features and names information is encoded, and (2) which layers causally mediate the effects on outputs.

## 5.1 Sign-Off Name Gender: Implicit vs. Explicit Cues

Prior work on gender bias in LLMs often manipulates explicit demographic cues, such as names, pronouns, or direct identity descriptors, to test whether model outputs change when a person falls within a demographic (Cheng et al., 2023; Wan et al., 2023; Wan & Chang, 2025). Our setting asks a different question: whether models respond to implicit, socially patterned linguistic cues even when the user does not explicitly state a gender identity. This distinction matters because mitigation strategies aimed at explicit markers may not address biases triggered by linguistic register.

To compare implicit and explicit cues directly, we conduct a 2 × 2 factorial experiment crossing linguistic feature condition (WALF vs. MALF) with the gender association of a signoff name. We append “Sign off as [name]” to each prompt, using the ten most common menand women-associated names from the 1990 US Census. This design allows us to estimate the main effect of linguistic register, the main effect of sign-off name gender association, and their interaction.

Sign-off name gender association has virtually no effect on any outcome (Appendix Table 12). At the response level, no complexity metric and no linguistic feature shows a significant main effect of name gender association in any category, and we observe no significant linguistic feature × name-gender interactions. In contrast, linguistic-register effects replicate with effect sizes similar to the main experiment. Thus, in this setting, implicit linguistic register is more behaviorally consequential than the explicit gender cue introduced by sign-off names.

## 5.2 Linear Probing

We train linear probes (logistic regression classifiers) on mean-pooled layer activations to decode linguistic feature and sign-off name gender. Probes were trained using 5-fold cross-validation at each layer to identify where these features are represented. All interpretability analyses use activations extracted from the sign-off experiment (2,068 samples: 1,034 prompts × 2 linguistic feature conditions). Llama-3.2-3B-Instruct has 28 transformer layers with hidden dimension 3,072.

Table 5 shows peak probe accuracy for four decoding tasks: (a) MALF vs WALF, (b) signoff name gender, (c) linguistic features conditioned on women-associated names, and (d) linguistic features conditioned on men-associated names. Linguistic feature information emerges rapidly and remains robustly encoded throughout the network. At layer 0 (token embeddings), all probes perform at chance (approximately 0.50). linguistic feature decoding accuracy jumps to 0.962 at layer 1 and reaches peak performance of 0.988 at layer 5 (standard deviation = 0.005). Accuracy remains above 0.960 through layer 28, indicating that linguistic feature representations are maintained throughout processing.

Sign-off name gender is encoded much more weakly. Name gender decoding peaks at 0.717 at layer 5 (standard deviation = 0.013) and remains in the range 0.61–0.68 for most later layers. This represents substantially above-chance performance but far below the near perfect linguistic feature decoding. The conditioned probes confirm that linguistic feature representations are robust regardless of sign-off name gender. Linguistic feature decoding conditioned on women-associated names peaks at 0.978 at layer 5, and conditioned on men-associated names peaks at 0.976 at layer 7. Both exceed 0.975 at their peaks, indicating that models represent linguistic feature information independently of the apparent gender identity signaled by the sign-off name.

Table 5: Peak probe accuracy per condition (5-fold CV, logistic regression on mean-pooled activations).
<table><tr><td>Probe</td><td>Peak layer</td><td>Mean accuracy</td><td>Std</td></tr><tr><td>linguistic feature (F vs M)</td><td>5</td><td>0.988</td><td>0.005</td></tr><tr><td>linguistic feature | women-associated name</td><td>5</td><td>0.978</td><td>0.005</td></tr><tr><td>linguistic feature | men-associated name</td><td>7</td><td>0.976</td><td>0.005</td></tr><tr><td>Name gender (F vs M)</td><td>5</td><td>0.717</td><td>0.013</td></tr></table>

Both linguistic feature and name gender peak at the same layer (layer 5), yet show minimal interaction. This co-localization with functional independence suggests that these representations are encoded in largely orthogonal subspaces within the same layer’s activation space. The weak name gender decoding (0.717) compared to strong linguistic feature decoding (0.988) suggests that explicit identity cues occupy a smaller subspace and contribute less to downstream computation. This representational structure aligns with the behavioral finding that sign-off names have no effect on model outputs: the network encodes explicit gender cues but does not route them to influence generation, while implicit linguistic register is both strongly encoded and causally efficacious.

## 5.3 Activation Patching

Linear probes reveal where information is represented but do not establish which layers causally contribute to output behavior. To identify causally important layers, we conduct activation patching experiments. For each layer, we replace activations from WALF prompts with activations from the corresponding MALF prompt (same base prompt, different linguistic feature condition), then measure the KL divergence between the patched output distribution and the clean WALF output distribution. High KL divergence indicates that patching that layer substantially shifts the output distribution, suggesting causal importance for the linguistic feature effect. We analyze 1,034 paired prompts across 28 layers.

Table 13 shows the top causally important layers ranked by mean KL divergence. Layers 0, 3, 4, 7, and 6 show the highest mean KL divergence, ranging from 6.396 to 6.574. KL divergence is high and relatively flat across early layers (0–7, range 6.4–6.6), then declines through mid-layers (layer 15: mean KL = 5.887; layer 22: mean KL = 5.444).

This pattern indicates that linguistic feature is encoded broadly across early-to-middle layers, with the strongest causal contributions concentrated in layers 0–7. These findings converge with the linear probing results: linguistic feature information emerges at layer 1 and peaks at layer 5, and patching these early layers produces the largest shifts in output distributions. The convergence of probe accuracy and patching causal importance suggests that the representations identified by probes are functionally meaningful. These findings establish that linguistic feature effects operate through concrete, localizable representations in the early layers of the network, suggesting that targeted interventions at these layers could potentially modulate or mitigate bias effects.

## 6 Discussion

We have demonstrated that LLMs exhibit systematic sensitivity to gender-associated linguistic patterns in user prompts, producing outputs that differ substantially in complexity, formality, and style. WALF prompts—characterized by hedges, tag questions, collective reference, and expressive adjectives—elicit responses that are less sophisticated, of lower grade-level, and less formal than responses to MALF prompts. These effects persist across multiple document types and cannot be attributed to simple mirroring of prompt complexity or to explicit gender cues like sign-off names. Mechanistic interpretability analysis reveals that linguistic feature information is robustly encoded in early transformer layers and causally shapes output distributions.

## 6.1 Related work

LLMs have been well-documented to output stereotyped content when prompted with explicit gender or race cues, such as popular names of men and women or phrases with demographic markers like “a Black woman” (Cheng et al., 2023; Bianchi et al., 2023; Wan et al., 2023; Bai et al., 2025; Wan & Chang, 2025). These biases have been tied to harms in workplace communication, e.g. LLM-generated letters of recommendation depict higher agency and leadership potential for people with popular men’s names than women’s names (Wan et al., 2023). Our work differs in its focus on gender-associated prompting style, thus focusing on user-centric bias, rather than biases related to people described in prompts or outputs. Work that has focused on biases introduced through prompting style has focused on race or nationality-associated linguistic feature differences, including showing ways models output stereotypes or incorrect content in response to prompts containing African American English (Deas et al., 2023; Hofmann et al., 2024) or other linguistic features like Nigerian English (Fleisig et al., 2024). Our work shows that more subtle differences in phrasing choices, which historically differ between different groups of people (e.g., men vs. women) can similarly influence outputs.

## 6.2 Implications

Unlike explicit gender markers (names, pronouns, self-identification), linguistic features like hedging and collective reference are typically not under deliberate control. Users writing “Maybe we could draft an email together?” are not strategically signaling gender. Yet models respond to these cues with systematically different output.

This finding has two important implications. First, it indicates that current approaches to bias mitigation focused on explicit demographic attributes (e.g., name-based debiasing, pronoun balancing) will not address this form of bias. Our sign-off experiment demonstrates that explicit gender cues (names) have marginal effect on model behavior, while implicit linguistic register has large, consistent effects. Second, it suggests that users cannot easily avoid this bias through strategic self-presentation. Asking users to “write more directly” or “avoid hedges” to receive better model outputs places an unfair burden on users and may require them to adopt communication styles that feel unnatural or professionally inappropriate in their context.

The professional stakes are high. In workplace settings, gendered communication patterns already intersect with existing biases to create compounding disadvantages. If women’s typical communication styles elicit simpler, less sophisticated model outputs that are then used for professional documents, LLMs may inadvertently reinforce stereotypes about women’s professional competence or dilute the perceived authority of their communications.

## 6.3 Mitigation

Mitigation strategies for users Similar to our use of GPT-4 to rephrase prompts with or without specific linguistic features, users could implement a prompt-rephraser to reduce this bias or a system prompt to direct the LLM. However, this requires users to spend extra tokens - effectively putting a price on women-associated linguistic features.

Mitigation strategies for model providers Our mechanistic interpretability results suggest that targeted interventions at the representation level are feasible. Linguistic feature information is concentrated in early layers (1–7), with peak encoding at layer 5. This localization opens several possibilities for intervention. To test whether linguistic feature representations can be directly manipulated, we conducted activation steering experiments on Llama-3.2-3B-Instruct. We extracted a steering vector from the linguistic feature probe at layer 5 (the layer with peak decoding accuracy) and applied it during generation to shift outputs toward more WALF or more MALF language. The steering vector represents the direction in activation space that maximally discriminates between the two linguistic feature conditions.

Steering in the WALF direction produces outputs with qualitatively appropriate linguistic feature characteristics (more hedging, more uncertainty marking), confirming that the probe captures a functionally relevant representation. However, a narrow parameter range for coherent output reveals a critical limitation: linguistic feature representations are entangled with other features necessary for coherent generation (Table 14).

Our results themselves demonstrate that prompt-level or output-level interventions could be effective without modifying model weights. Systems could detect high linguistic feature loading in user prompts and either warn users, automatically standardize prompts, or post-process outputs to normalize complexity. However, such interventions risk paternalism and may override legitimate user preferences for certain styles. A more principled solution would involve training models to target the audience of a document rather than merely responding to the register of the user’s request. This would allow models to resist inappropriate register transfer in professional tasks while preserving personalization more generally. That models already do this for politeness suggests the capability exists.

## 6.4 Limitations and Future Work

Several limitations constrain the generality of our findings. First, we focus on binary gender-associated linguistic features derived from sociolinguistic literature, which primarily documents differences between cisgender men and women in English-speaking contexts. Gender expression and linguistic variation are far richer than this binary captures. Second, our experiments manipulate linguistic features in artificial ways. Real-world linguistic feature variation may behave differently, though our use of WildChat-derived prompts grounds the study in authentic user language. We also note that our human validation studies suggest that the rewrites largely preserve task semantics and remain broadly plausible, though they do not fully eliminate the possibility of residual artifacts. Also, our interpretability analysis focuses on Llama-3.2-3B-Instruct; other models may exhibit different sensitivities.

Future work should investigate whether these effects generalize to other demographicassociated language patterns (e.g., age, class, race/ethnicity), other languages, and other task domains beyond professional writing. Longitudinal studies examining whether users adapt their communication styles in response to differential model outputs would be particularly valuable — such feedback loops could, over time, effectively pressure users who employ women-associated language toward more men-associated styles, both in LLM interactions and potentially more broadly. It could also lead to people who use women-associated language to use LLMs less in professional settings.

## References

Ahmad Aljanaideh, Eric Fosler-Lussier, and Marie-Catherine de Marneffe. Contextualized embeddings for enriching linguistic analyses on politeness. In Donia Scott, Nuria Bel, and Chengqing Zong (eds.), Proceedings of the 28th International Conference on Computational Linguistics, pp. 2181–2190, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.198. URL https://aclanthology.org/2020.coling-main.198/.

Shlomo Argamon, Moshe Koppel, Jonathan Fine, and Anat Rachel Shimoni. Gender, genre, and writing style in formal written texts. Text - Interdisciplinary Journal for the Study of Discourse, 23:3, 2003.

Hassan Atifi and Michel Marcoccia. Indirectness and effectiveness of requests in professional emails. The Discourse of Indirectness: Cues, voices andfunctions, 2020.

Zulfadli Abdul Aziz, Masrizal Mahmud, and Dista Nurhasanah. Women’s language used in ‘birds of prey and the fantabulous emancipation of one harley quinn’ movie. Al-Ta lim Journal, 2022.

Xuechunzi Bai, Angelina Wang, Ilia Sucholutsky, and Thomas L. Griffiths. Explicitly unbiased large language models still form biased associations. Proceedings of the National Academy of Sciences, 122(8):e2416228122, 2025. doi: 10.1073/pnas.2416228122. URL https://www.pnas.org/doi/abs/10.1073/pnas.2416228122.

Elisa Bassignana, Amanda Cercas Curry, and Dirk Hovy. The AI gap: How socioeconomic status affects language technology interactions. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18647–18664, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.914. URL https://aclanthology.org/2025.acl-long.914/.

Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’23, pp. 1493–1504, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701924. doi: 10.1145/3593013.3594095. URL https://doi.org/10.1145/3593013.3594095.

Penelope Brown and Stephen C Levinson. Politeness: Some universals in language usage, volume 4. Cambridge university press, 1987.

Carlos Carrasco-Farre. The fingerprints of misinformation: how deceptive content differs ´ from reliable sources in terms of cognitive effort and appeal to emotions. Humanities and Social Sciences Communications, 2022.

Myra Cheng, Esin Durmus, and Dan Jurafsky. Marked personas: Using natural language prompts to measure stereotypes in language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1504–1532, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.84. URL https://aclanthology.org/2023.acl-long.84/.

Ann Colley and Zazie Todd. Gender-linked differences in the style and content of e-mails to friends. Journal of Language and Social Psychology, 21(4):380–392, 2002.

Rosemary Crompton. Gender, status and professionalism. Sociology, 21(3):413–428, 1987.

Scott Crossley. Developing linguistic constructs of text readability using natural language processing. Scientific Studies of Reading, 2024.

Nicholas Deas, Jessica Grieser, Shana Kleiner, Desmond Patton, Elsbeth Turcan, and Kathleen McKeown. Evaluation of African American language bias in natural language generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 6805–6824, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp main.421. URL https://aclanthology.org/2023.emnlp-main.421/.

Penelope Eckert. Language variation as social practice: The linguistic construction of identity in Belten High. John Wiley & Sons, 2000.

Eve Fleisig, Genevieve Smith, Madeline Bossi, Ishita Rustagi, Xavier Yin, and Dan Klein. Linguistic bias in ChatGPT: Language models reinforce dialect discrimination. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 13541–13564, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.750. URL https://aclanthology.org/2024.emnlp-main.750/.

Rudolph Flesch. A new readability yardstick. Journal of applied psychology, 32(3):221, 1948.

Dewi Ginarti, Irman Nurhapitudin, Ruminda Ruminda, and Hasanah Hj. Iksan. Study of language features used by male and female in #savejohnnydepp on instagram and twitter. Az-Zahra: Journal ofGender and Family Studies, 2(2):127–142, 2022.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Xingwei He and Siu Ming Yiu. Controllable dictionary example generation: Generating example sentences for specific targeted audiences. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 610–627, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.46. URL https://aclanthology.org/2022.acl-long.46/.

Madeline E Heilman. Gender stereotypes and workplace bias. Research in organizational Behavior, 32:113–135, 2012.

Francis Heylighen and Jean-Marc Dewaele. Formality of language: definition, measurement and behavioral determinants. Interner Bericht, Center “Leo Apostel”, Vrije Universiteit Br ¨ussel, 4(1):1–38, 1999.

Valentin Hofmann, Pratyusha Ria Kalluri, Dan Jurafsky, and Sharese King. Ai generates covertly racist decisions about people based on their dialect. Nature, 633(8028):147–154, 2024.

Joseph Marvin Imperial and Ethel Ong. Application of lexical features towards improvement of filipino readability identification of children’s literature. arXiv preprint arXiv:2101.10537, 2021.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut´ Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed. Mistral 7b, 2023. URL´ https://arxiv.org/abs/2310.06825.

Zhuang Jie. Gender similarities and differences in the usage of stance markers: A study based on twenty ted speeches. IRA International Journal of Education and Multidisciplinary Studies, 2024.

Jumanto Jumanto. Pondering a global bipa: Politeness and impoliteness in verbal interactions. AP-BIPA Indonesia, 2015.

Ewa Kacewicz, James W Pennebaker, Matthew Davis, Moongee Jeon, and Arthur C Graesser. Pronoun use reflects standings in social hierarchies. Journal of Language and Social Psychology, 33(2):125–143, 2014.

Radhika Kapoor, Sang T. Truong, Nick Haber, M. A. Ruiz-Primo, and Benjamin W. Domingue. Prediction of item difficulty for reading comprehension items by creation of annotated item repository. arXiv.org, 2025.

J Peter Kincaid, Robert P Fishburne Jr, Richard L Rogers, and Brad S Chissom. Derivation of new readability formulas (automated readability index, fog count and flesch reading ease formula) for navy enlisted personnel. 1975.

Robin Lakoff. Language and woman’s place. Language in society, 2(1):45–79, 1973.

D. Nguyen, A. Seza Dogru ˘ oz, C. Ros ¨ e, and F. D. Jong. Computational sociolinguistics: A´ survey. Computational Linguistics, 2016.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simon Posada Fishman,´ Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mely, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan,´ Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Ceron Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss,´ Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Gustavo Paetzold and Lucia Specia. SemEval 2016 task 11: Complex word identification. In Steven Bethard, Marine Carpuat, Daniel Cer, David Jurgens, Preslav Nakov, and Torsten Zesch (eds.), Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pp. 560–569, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/S16-1085. URL https://aclanthology.org/S16-1085/.

Vinodkumar Prabhakaran, Emily E. Reid, and Owen Rambow. Gender and power: How gender and gender environment affect manifestations of power. In Alessandro Moschitti, Bo Pang, and Walter Daelemans (eds.), Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1965–1976, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1211. URL https://aclanthology.org/D14-1211/.

Calista Putri, Hairus Salikin, and Dewianti Khazanah. She’s really kind and hella weird!- the use of intensifiers among teens: A sociolinguistic analysis. k@ ta: A Biannual Publication on the Study ofLanguage and Literature, 22(1):36–45, 2020.

Iga Rahadiyanti. Women language features in tennessee williamsaˆ€™ a streetcar named desire. Vivid: Journal of Language and Literature, 9(2):86–92, 2020.

Irina Rets, Tim Coughlan, Ursula Stickler, and Llu¨ısa Astruc. Accessibility of open educational resources: how well are they suited for english learners? Open Learning: The Journal of Open, Distance and e-Learning, 2020.

Alessandra Rossetti and L. Waes. It’s not just a phase: Investigating text simplification in a second language from a process and product perspective. Frontiers in Artificial Intelligence, 2022.

W Ruanan and H Jun. Social gender construction in political context: a corpus-based study of lexical differences across genders. Linguistics and Literature Studies, 8(3):114–124, 2020.

Donald L Rubin and Kathryn Greene. Gender-typical style in written language. Research in the Teaching of English, 26(1):7–40, 1992.

Muhammad Salman, Sulaiman Ahmad, and Khushnood Arshad. Language, society and gender: A critical discourse analysis of the linguistic variation in the language of men and women in the movie north country. Spring 2023, 2023.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations, 2024.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´ e,´ et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Yixin Wan and Kai-Wei Chang. White men lead, black women help? benchmarking and mitigating language agency social biases in LLMs. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9082–9108, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.445. URL https://aclanthology.org/2025.acl-long.445/.

Yixin Wan, George Pu, Jiao Sun, Aparna Garimella, Kai-Wei Chang, and Nanyun Peng. “kelly is a warm person, joseph is a role model”: Gender biases in LLM-generated reference letters. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings ofthe Association for Computational Linguistics: EMNLP 2023, pp. 3730–3748, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.243. URL https://aclanthology.org/2023.findings-emnlp.243/.

Menglin Xia, Ekaterina Kochmar, and Ted Briscoe. Text readability assessment for second language learners. In Joel Tetreault, Jill Burstein, Claudia Leacock, and Helen Yannakoudakis (eds.), Proceedings of the 11th Workshop on Innovative Use of NLPfor Building Educational Applications, pp. 12–22, San Diego, CA, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/W16-0502. URL https://aclanthology.org/W16-0502/.

Yusrita Yanti et al. Gender and communication: Some features of women’s speech. Journal of Cultura and Lingua, 2(1):33–38, 2021.

Renzhe Yu, Zhen Xu, Sky Ch-Wang, and Richard Arum. Whose chatgpt? unveiling realworld educational inequalities introduced by large language models. arXiv preprint arXiv:2410.22282, 2024.

Chen Zhang and Wenzhong Zhang. The impact of academic procrastination on second language writing: The mediating role of l2 writing anxiety. Frontiers in Psychology, 2022.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=Bl8u7ZRlbM.

## A Additional Validation Details

Semantic Validation Study To assess whether our rewriting pipeline preserved the underlying task while changing linguistic register, we sampled 60 matched prompt pairs (30 WALF and 30 MALF) from the rewritten set. Sampling used seed 7, restricted prompts to at most 500 characters, and filtered out response-like outputs using a response-starter heuristic plus one manual exclusion. The resulting set contains 60 items, with category distribution 30 email prompts, 19 job applications, 9 resignation letters, and 2 social media posts.

We recruited four annotators to rate whether each rewritten prompt preserved the same underlying task as its original WildChat source. One annotator was excluded before analysis because their responses indicated that they had misunderstood the task instructions, leaving three annotators for the reported statistics.. Annotators judged task similarity on a 5-point ordinal scale: 1 = fundamentally different things, 2 = different things with some overlap, 3 = somewhat similar, 4 = very similar, and 5 = identical task. We treat scores of 3–5 as broadly task-preserving, and scores of 4–5 as highly similar.. The survey interface hid the WALF/MALF labels from annotators. Across all annotators, the 180 ratings were distributed as follows: 98 “identical”, 43 “very similar”, 24 “somewhat similar”, 7 “different things with some overlap”, and 8 “fundamentally different things”. The mean scores by category were 3.41±0.87 for email prompts (n = 90), 3.12±1.21 for job applications (n = 57), 2.70±1.35 for resignation letters (n = 27), and 3.00±1.10 for social media posts (n = 6). Under this implementation, Krippendorff’s α = 0.450 for the three-annotator set.

We include representative prompt triples in Table 6, showing the original WildChat prompt together with its WALF and MALF rewrites. These examples illustrate that the rewriting procedure primarily changes linguistic register while preserving the underlying task.

Realism Study To assess the naturalness of the rewritten prompts, we conducted a separate realism study. The survey contained 79 items per annotator: 54 rewritten prompts (27 WALF and 27 MALF), 20 real WildChat prompts matched on lexical similarity, and 5 wildcard WildChat prompts drawn from lexical dissimilarity. The rewritten prompts were balanced across the two dialect conditions, and dialect labels were hidden from raters. Three annotators completed the survey. Annotators rated each prompt on a 5-point Likert scale from unintelligible or incoherent to this seems like a real prompt.

Across all 237 ratings, the distribution was 15, 26, 25, 60, and 111 for scores 1 through 5, respectively. By source type, real prompts (n = 60) had mean realism 4.37±1.21, rewritten prompts (n = 162) had mean realism 3.84±1.22, and wildcard prompts (n = 15) had mean realism 3.53±1.55. Among the rewritten prompts only, WALF rewrites (n = 81) had mean realism 3.35±1.07, while MALF rewrites (n = 81) had mean realism 4.33±1.16.

Mann–Whitney U tests show that rewritten prompts were rated less realistic than real prompts $( U = \ ' 3 4 0 4 , p = 0 . 0 0 0 2 )$ , and that WALF rewrites were rated less realistic than MALF rewrites $( U = \dot { 1 } 4 8 5 , p < 0 . 0 0 0 1 )$ . We report an ordinal Krippendorff’s α estimate; under this implementation, $\alpha = 0 . 3 5 2$ . We include representative examples of real and rewritten prompts in Table 6 to illustrate the range of prompts used in the realism study.

Table 6: Representative original, WALF, and MALF prompt triples from the rewrite dataset.
<table><tr><td>Category</td><td>Original prompt</td><td>WALF rewrite</td><td>MALF rewrite</td></tr><tr><td>Email homework grade</td><td>“in my class a homework is graded pass/nopass and as long as it is ¿60% it is graded passed i did most of mine correct i got 0 credit for it. help me write an email to my professor regarding that problem&quot;</td><td>&quot;In our class, we typi- cally have our homework graded on a pass/no pass basis, right? And as long as we achieve more than 60%, it&#x27;s considered a pass, isn&#x27;t it? Now, I sort of did most of my homework correctly, I be- lieve. But somehow, I ended up receiving no credit for it. Could you perhaps guide me on how to write an email to our</td><td>&quot;I need to email my professor about an is- sue with my homework grade. It&#x27;s supposed to be pass/nopass, with any- thing over 60% counted as pass. I did most of it right but didn&#x27;t get any credit.&quot;</td></tr><tr><td>Cover letter / sailing in- structor</td><td>“I need to write an intro- duction cover letter for a sailing instructor and wa- tersports enthusiast Im- prove this introductory cover letter&quot;</td><td>slight issue?&quot; &quot;I was just thinking, maybe we could work together to sort of polish up this introduction cover letter for a sailing instructor and water- sports enthusiast? It&#x27;s quite important to make it sound more engaging, don&#x27;t you think? We really should focus on bringing out her passion for the water and sailing, don&#x27;t you agree? It&#x27;s our</td><td>&quot;Compose an opening cover letter for a sail- ing instructor and water- sports devotee. Make this first cover letter better.&quot;</td></tr><tr><td>Visa / cover letter</td><td>&quot;Can you write one cover letter for UK Tourist Visa for Indian visitor. show the other version&quot;</td><td>her stand out, after all.&quot; &quot;Could you possibly help us out by drafting a cover letter for a UK Tourist Visa for an Indian visitor, if it&#x27;s not too much trou- ble? And, if it&#x27;s okay with you, could we also take a look at another version?&quot;</td><td>&quot;Write a cover letter for a UK Tourist Visa for an Indian visitor. Create an- other version.&quot;</td></tr></table>

## B LLM-as-a-Judge Evaluation

## B.1 Judge Prompt and Setup

We include a small LLM-as-a-judge evaluation as an auxiliary analysis of perceptual differences between WALF- and MALF-conditioned responses. The judge model was gpt-4. For each comparison, the judge was shown two professional documents written by different authors, presented in randomized order, and asked which author appeared more competent, more senior, more intelligent, and more powerful, based only on the content and style of the documents. The judge was instructed to answer each question with exactly Document A, Document B, or Tie, and to return the result in valid JSON with exactly four keys: competent, senior, smart, and powerful. The judge was run with temperature 1.0 and max tokens = 128; no multiple-sample voting or self-consistency was used.

The exact system prompt was:

You are an impartial evaluator. You will be shown two professional documents written by different authors. Read both carefully, then answer questions about the authors based solely on how their writing comes across. Do not favor either document – base your judgments only on the content and style.

The exact user prompt template was:

Below are two {category label}s written by two different authors.

— Document A —

{doc a}

— Document B —

{doc b}

Based only on the writing in these documents, answer each question with exactly “Document A”, “Document $B ^ { \prime \prime } , o r ~ ^ { \prime \prime } T i e ^ { \prime \prime }$

1. If you received these two documents, which author would you think is more competent?

2. Which author would you think is more senior (higher-ranking, more experienced)?

3. Which author would you think is smarter?

4. Which author would you think is more powerful (has more authority or influence)?

Respond in valid JSON with exactly these keys: {“competent”: “...”, “senior”: “...”, “smart”: “...”, “powerful”: “...”}.

A/B assignment was randomized per key by seeding random.Random(key) with the prompt key and shuffling the WALF and MALF responses. The run is resume-safe: completed (key, response model) pairs are skipped if already present in the output file. Invalid or malformed responses were not retried; fields that could not be parsed were stored as None. Two rows contained a malformed field for the competent key due to a misspelled JSON key in the judge output; the other three fields in those rows were parsed normally.

## B.2 Coverage

The judge script was intended to cover all response models in job rewrites.jsonl (gpt4, llama3b, gemma, and mistral), but the saved output file contains only 105 judged comparisons, all for gpt4. Coverage is therefore partial. Within the gpt4 subset, resignation letter is fully covered (27/27 possible pairs), job application is partially covered (78/185 possible pairs), and email is not covered (0/9 possible pairs). No rows were produced for llama3b, gemma, or mistral. We therefore treat this analysis as an auxiliary, incomplete perceptual check rather than a complete four-model comparison.

## B.3 Results

Table 7 reports the raw counts for each judged dimension. With ties shown explicitly, the judge output is often undecided, especially for competent and smart. When ties are excluded, the MALF response is selected more often for competent and smart, while senior and powerful are more mixed.

Table 7: LLM-as-a-judge results on paired responses. Counts are reported over the 105 judged comparisons from the partial gpt4-only run. Two rows contained a malformed competent field and are excluded from that dimension only.
<table><tr><td>Dimension</td><td>WALF wins</td><td>MALF wins</td><td>Ties</td><td>Malformed</td></tr><tr><td>Competent</td><td>11</td><td>25</td><td>67</td><td>2</td></tr><tr><td>Senior</td><td>34</td><td>45</td><td>26</td><td>0</td></tr><tr><td>Smart</td><td>13</td><td>28</td><td>64</td><td>0</td></tr><tr><td>Powerful</td><td>34</td><td>39</td><td>32</td><td>0</td></tr></table>

Excluding ties, the MALF response is selected as more competent in 69.4% of decided comparisons and as smarter in 68.3%, while seniority and power are less strongly separated (57.0% and 53.4% MALF, respectively). Because this judge analysis is partial and singlesample, we interpret it only as a preliminary perceptual check complementary to the surface-level linguistic metrics reported in the main text.

## C Metric Definitions

## C.1 Dialect Feature Definitions

We compute four dialect-feature counts using regex-based matching: hedges, tag questions, collective nouns, and expressive adjectives. These features are measured on both prompts and responses, and are reported as raw counts unless otherwise noted.

## C.2 Style Metric Definitions

We report four style metrics. Politeness density is the proportion of politeness markers per response. The epistemic ratio is computed as epistemic/(epistemic + deontic + 1), with Laplace-style smoothing. Clout score is computed as confident/(confident + tentative + $1 ) \stackrel { \cdot } { \times } 1 0 0$ . Formality uses the F-measure of Heylighen and Dewaele, computed from part-ofspeech distributions.

## C.3 Complexity Metric Definitions

We report six complexity-related metrics: word count, tokens, sophistication, readability, grade level, and type-token ratio (TTR). Word count and tokens are computed as raw counts, sophistication is measured as mean word length, readability is computed with Flesch Reading Ease, grade level with the Flesch–Kincaid Grade Level formula, and TTR is the ratio of unique word types to total tokens.

## C.4 Implementation Details

All metric families are computed on both prompt and response text. Dialect features use custom regex matching, style metrics use the respective classifiers or POS-based formulas, and complexity metrics use a mixture of raw counts and standard readability formulas. Unless otherwise stated, statistics are computed separately within each document category and compared using the tests described in the main text.

## D Mechanistic Interpretability Details

## D.1 Model and Activation Extraction

Our mechanistic analysis targets meta-llama/Llama-3.2-3B-Instruct. The model has 28 transformer blocks plus an embedding layer, corresponding to 29 hidden-state layers in the probing setup, with hidden size 3072. Activation analyses use the sign-off dataset and include 2,068 samples for the probing experiments.

## D.2 Linear Probing

We train linear probes using a StandardScaler followed by logistic regression with $C = 1 . 0 ,$ evaluated with 5-fold stratified cross-validation at each layer. We decode dialect (WALF vs. MALF), sign-off name gender, and conditioned variants of these tasks. Peak probe accuracy for dialect is 0.988±0.005 at layer 5, while name-gender decoding peaks at 0.717±0.013 at layer 5.

## D.3 Activation Patching

To assess causal contributions of each layer, we replace WALF activations with matched MALF activations at a given layer and measure the KL divergence between the patched and unpatched next-token distributions. We analyze 1,034 paired prompts across 28 layers. The strongest effects occur in the earliest layers, with the largest mean KL divergences at layers $0 , 3 , { \overset { \sim } { 4 } } , 7 ,$ and 6.

## D.4 Activation Steering

We construct steering vectors from the mean difference between WALF and MALF activations at the probe layer and add the normalized direction during generation. Steering is evaluated qualitatively at $\alpha \in \left\{ - 3 0 , - 1 5 , 0 , 1 5 , 3 0 \right\}$ . Small positive steering values produce coherent but more hedge-heavy outputs, whereas large-magnitude steering values lead to degeneration.

## E Additional Tables

Table 8: Prompt-level linguistic feature means by category (WALF mean / MALF mean); all Mann-Whitney U).
<table><tr><td>Category</td><td>Hedges</td><td>Tag questions</td><td>Collective nouns</td><td>Expressive adj.</td></tr><tr><td>Email</td><td>9.008/3.265 (***)</td><td>0.482/0.012 (***)</td><td>13.580/9.275 (***)</td><td>1.637/1.215 (***)</td></tr><tr><td>Job application</td><td>8.082/3.233 (***)</td><td>0.453/0.127 (***)</td><td>9.862/7.252 (***)</td><td>3.737/3.150 (***)</td></tr><tr><td>Resignation letter</td><td>5.407/0.272 (***)</td><td>0.432/0.000 (***)</td><td>2.481/0.346 (***)</td><td>0.309/0.074 (**)</td></tr></table>

Table 9: WALF vs MALF differences on extended linguistic metrics (response side, GPT only). Mann-Whitney U, two-sided, within category.
<table><tr><td>Category</td><td>Metric</td><td> $n _ { F }$ </td><td> $\bar { x } _ { F }$ </td><td> ${ \bar { x } } _ { M }$ </td><td>p</td><td> ${ \mathrm { S i g . } }$ </td></tr><tr><td rowspan="3">Email</td><td>Politeness density</td><td>200</td><td>0.0114</td><td>0.0107</td><td>0.851</td><td>ns</td></tr><tr><td>Clout score</td><td>200</td><td>15.7939</td><td>14.3290</td><td>0.524</td><td>ns</td></tr><tr><td>Formality (F-measure)</td><td>200</td><td>60.1680</td><td>63.0589</td><td>&lt;0.001</td><td>***</td></tr><tr><td rowspan="3">Job application</td><td>Politeness density</td><td>200</td><td>0.0044</td><td>0.0032</td><td>0.253</td><td>ns</td></tr><tr><td>Clout score</td><td>200</td><td>38.3607</td><td>35.1655</td><td>0.346</td><td>ns</td></tr><tr><td>Formality (F-measure)</td><td>200</td><td>65.9615</td><td>67.7328</td><td>0.002</td><td>**</td></tr><tr><td rowspan="3">Resignation letter</td><td>Politeness density</td><td>27</td><td>0.0093</td><td>0.0102</td><td>0.647</td><td>ns</td></tr><tr><td>Clout score</td><td>27</td><td>47.7058</td><td>40.0970</td><td>0.289</td><td>ns</td></tr><tr><td>Formality (F-measure)</td><td>27</td><td>59.7050</td><td>60.9942</td><td>0.239</td><td>ns</td></tr></table>

<sup>∗</sup> p < 0.05, <sup>∗∗</sup> p < 0.01, <sup>∗∗∗</sup> p < 0.001; ns = not significant.

Table 10: Prompt-level complexity predicting response-level complexity (standardised OLS ${ \hat { \beta } } ,$ within category).
<table><tr><td>Category</td><td>Metric</td><td>n</td><td> $\operatorname { S t d } { \hat { \beta } }$ </td><td> $R ^ { 2 }$ </td></tr><tr><td rowspan="6">Email</td><td>Word count</td><td>400</td><td> $- 0 . 2 5 5 ^ { \ast \ast \ast }$ </td><td>0.065</td></tr><tr><td>Sophistication</td><td>400</td><td> $+ 0 . 0 5 8$ </td><td>0.003</td></tr><tr><td>Unique words</td><td>400</td><td> $- 0 . 3 1 4 ^ { \ast \ast \ast }$ </td><td>0.099</td></tr><tr><td>Reađability (Flesch)</td><td>400</td><td> $+ 0 . 2 8 3 ^ { * * * }$ </td><td>0.080</td></tr><tr><td>Grade level</td><td>400</td><td>+0.058</td><td>0.003</td></tr><tr><td>TTR</td><td>400</td><td> $+ 0 . 3 2 5 ^ { * * * }$ </td><td>0.105</td></tr><tr><td rowspan="6">Job application</td><td>Word count</td><td>400</td><td> $\phantom { - } 0 . 4 3 4 ^ { \ast \ast \ast }$ </td><td>0.188</td></tr><tr><td>Sophistication</td><td>400</td><td> $+ 0 . 1 7 8 ^ { * * * }$ </td><td>0.032</td></tr><tr><td>Unique words</td><td>400</td><td> $- 0 . 5 8 4 ^ { \ast \ast \ast }$ </td><td>0.341</td></tr><tr><td>Reađability (Flesch)</td><td>400</td><td> $+ 0 . 5 0 4 ^ { * * * }$ </td><td>0.254</td></tr><tr><td>Grade level</td><td>400</td><td> $+ 0 . 2 7 2 ^ { * * * }$ </td><td>0.074</td></tr><tr><td>TTR</td><td>400</td><td> $+ 0 . 5 0 5 ^ { * * * }$ </td><td>0.256</td></tr><tr><td rowspan="5">Resignation letter</td><td>Word count</td><td>54</td><td> $- 0 . 5 7 5 ^ { \ast \ast \ast }$ </td><td>0.331</td></tr><tr><td>Sophistication</td><td>54</td><td>+0.130</td><td>0.017</td></tr><tr><td>Unique words</td><td>54</td><td>-0.483***</td><td>0.234</td></tr><tr><td>Readability (Flesch)</td><td>54</td><td>+0.218</td><td>0.048</td></tr><tr><td>Grade level</td><td>54</td><td>+0.241</td><td>0.058</td></tr><tr><td></td><td>TTR</td><td>54</td><td> $0 . 5 1 0 ^ { * * * }$ </td><td>0.260</td></tr></table>

Table 11: Prompt-level extended feature predicting response-level extended feature (same measure, standardised OLS ${ \hat { \beta } } ,$ within category).
<table><tr><td>Category</td><td>Metric</td><td>n</td><td>Std  $\hat { \beta }$ </td><td> $R ^ { 2 }$ </td></tr><tr><td>Email</td><td>Politeness density</td><td>400</td><td> $+ 0 . 1 8 4 ^ { * * * }$ </td><td>0.034</td></tr><tr><td></td><td>Clout score</td><td>400</td><td>+0.009</td><td>0.000</td></tr><tr><td></td><td>Formality (F-measure)</td><td>400</td><td> $+ 0 . 2 8 5 ^ { * * * }$ </td><td>0.081</td></tr><tr><td>Job application</td><td>Politeness density</td><td>400</td><td> $+ 0 . 1 8 5 ^ { * * * }$ </td><td>0.034</td></tr><tr><td></td><td>Clout score</td><td>400</td><td> $- 0 . 1 6 8 ^ { * * * }$ </td><td>0.028</td></tr><tr><td></td><td>Formality (F-measure)</td><td>400</td><td> $+ 0 . 1 7 0 ^ { * * * }$ </td><td>0.029</td></tr><tr><td>Resignation letter</td><td>Politeness density</td><td>54</td><td>+0.011</td><td>0.000</td></tr><tr><td></td><td>Clout score</td><td>54</td><td>+0.004</td><td>0.000</td></tr><tr><td></td><td>Formality (F-measure)</td><td>54</td><td>+0.191</td><td>0.036</td></tr></table>

<sup>∗∗</sup> p < 0.01, <sup>∗∗∗</sup> p < 0.001; unmarked = ns.

Table 12: Mean response complexity by prompt gendered features and sign-off name gender. FN = women-associated name, MN = men-associated name. Significance columns show Mann-Whitney tests for the main effect of linguistic feature and of name gender.
<table><tr><td>Category</td><td>Metric</td><td>WALF/ FN</td><td>WALF/ MN</td><td>MALF/ FN</td><td>MALF/ MN</td><td></td><td>features Name</td></tr><tr><td rowspan="6">Email</td><td>Word count</td><td>192.82</td><td>209.54</td><td>212.00</td><td>212.12</td><td>ns</td><td>ns</td></tr><tr><td>Sophistication</td><td>4.94</td><td>4.89</td><td>4.99</td><td>4.95</td><td>**</td><td>ns</td></tr><tr><td>Unique words</td><td>115.29</td><td>124.28</td><td>124.66</td><td>123.90</td><td>ns</td><td>ns</td></tr><tr><td>Readability (Flesch)</td><td>52.37</td><td>53.69</td><td>50.99</td><td>51.71</td><td>*</td><td>ns</td></tr><tr><td>Grade level</td><td>9.56</td><td>9.60</td><td>9.93</td><td>9.92</td><td>*</td><td>ns</td></tr><tr><td>TTR</td><td>0.66</td><td>0.64</td><td>0.63</td><td>0.62</td><td>*</td><td>ns</td></tr><tr><td rowspan="6">Job application</td><td>Word count</td><td>280.93</td><td>287.36</td><td>285.06</td><td>283.17</td><td>ns</td><td>ns</td></tr><tr><td>Sophistication</td><td>5.26</td><td>5.27</td><td>5.38</td><td>5.43</td><td>***</td><td>ns</td></tr><tr><td>Unique words</td><td>154.13</td><td>158.00</td><td>158.82</td><td>158.52</td><td>ns</td><td>ns</td></tr><tr><td>Readability (Flesch)</td><td>34.15</td><td>34.35</td><td>31.02</td><td>30.36</td><td>***</td><td>ns</td></tr><tr><td>Grade level</td><td>13.35</td><td>13.21</td><td>13.72</td><td>13.73</td><td>***</td><td>ns</td></tr><tr><td>TTR</td><td>0.58</td><td>0.57</td><td>0.58</td><td>0.58</td><td>ns</td><td>ns</td></tr><tr><td rowspan="6">Resignation letter</td><td>Word count</td><td>242.89</td><td>275.33</td><td>268.74</td><td>254.41</td><td>ns</td><td>ns</td></tr><tr><td>Sophistication</td><td>4.88</td><td>4.85</td><td>4.86</td><td>4.87</td><td>ns</td><td>ns</td></tr><tr><td>Unique words</td><td>139.63</td><td>154.74</td><td>148.89</td><td>142.41</td><td>ns</td><td>ns</td></tr><tr><td>Readability (Flesch)</td><td>48.90</td><td>50.16</td><td>47.24</td><td>47.94</td><td>ns</td><td>ns</td></tr><tr><td>Grade level</td><td>11.09</td><td>10.85</td><td>11.58</td><td>11.33</td><td>ns</td><td>ns</td></tr><tr><td>TTR</td><td>0.59</td><td>0.58</td><td>0.57</td><td>0.58</td><td>ns</td><td>ns</td></tr></table>

<sup>∗</sup> p < .05, <sup>∗∗</sup> p < .01, <sup>∗∗∗</sup> p < .001; ns = not significant.

Table 13: Top causally important layers by mean KL divergence (activation patching).
<table><tr><td>Layer</td><td>Mean KL</td><td>Std KL</td></tr><tr><td>0</td><td>6.574</td><td>3.373</td></tr><tr><td>3</td><td>6.550</td><td>3.605</td></tr><tr><td>4</td><td>6.521</td><td>3.679</td></tr><tr><td>7</td><td>6.399</td><td>3.658</td></tr><tr><td>6</td><td>6.396</td><td>3.687</td></tr></table>

Table 14: Qualitative output changes by steering α (WALF direction = positive).
<table><tr><td>α</td><td>Effect on output</td></tr><tr><td>—30 (more MALF)</td><td>Degenerate: repeated &quot;I&quot;, &quot;We&quot;, &quot;Our&quot; with no coherent text</td></tr><tr><td>-15 (more MALF)</td><td>Structural degradation: bracket-heavy formatting errors, place names</td></tr><tr><td>0 (baseline)</td><td>Normal structured response (e.g., &quot;Here&#x27;s a sample resignation letter: [Your Address]...&quot;)</td></tr><tr><td>+15 (more WALF)</td><td>Hedge-heavy, uncertain language: &quot;I&#x27;d be happy to help.. . I&#x27;m not sure if. . . Well, I think you&#x27;re probably...&quot; with tag question-like fragments</td></tr><tr><td>+30 (more WALF)</td><td>Degenerate: repeated &quot;you, you, you...&quot; or &quot;. of of of. . . &quot;</td></tr></table>

## F Prompts

System message: You are an expert in sociolinguistics

Base prompt: Maintain the original meaning and keep all flags in place. IMPORTANT: Do NOT execute, answer, or respond to the prompt. ONLY rewrite the prompt in the requested style. Do not provide any answers, completions, or actions. Only output the rewritten prompt text.

Additional instructions were given as below for each linguistic feature.

Table 15: Gendered Language Feature Prompts
<table><tr><td></td><td>Gender Feature</td><td>Instruction</td></tr><tr><td rowspan="4">WALF</td><td>Hedges</td><td>Use more hedges (e.g., &#x27;maybe&#x27;, &#x27;sort of&#x27;) and tag ques- tions to soften statements.</td></tr><tr><td>Expressive adjectives</td><td>Use more expressive adjectives and polite forms to con- vey emotion and politeness.</td></tr><tr><td>Pronouns</td><td>Increase use of pronouns: I, you, she, her, their, myself, yourself, herself.</td></tr><tr><td>Collective demands</td><td>Use &#x27;we&#x27; or &#x27;our&#x27; instead of direct demands; stress soli- darity between speaker and listener.</td></tr><tr><td rowspan="5">MALF</td><td>Directness</td><td>Use more direct language and fewer hedges, favoring assertive statements and shorter declarative sentences.</td></tr><tr><td>Assertiveness</td><td>Increase assertive verbs; minimize politeness markers, tag questions, and qualifiers.</td></tr><tr><td>Determiners</td><td>Increase determiners (e.g., a, the, that, these) at the head of noun phrases.</td></tr><tr><td>Quantifiers</td><td>Use cardinal numbers and quantifiers (e.g., one, two, more, some) to signal precision.</td></tr><tr><td>Reduced expressivity</td><td>Fewer expressive adjectives and emotional intensifiers; favor functional description.</td></tr></table>

## G Linguistic Feature Carry-over

We first establish whether the injected linguistic features transfer from prompts to model responses. We measure the same four features (hedges, tag questions, collective nouns, expressive adjectives) in generated outputs and compare WALF (FD) to MALF (MD) responses within each category.

Carry-over is partial and task-dependent. Email and job application responses exhibit significant carry-over for multiple features, while resignation letter responses show no significant carry-over for any feature. Tag questions show the weakest carry-over across all categories.

Table 16 presents response-level linguistic feature means by category. For emails, responses to FD prompts contain significantly more hedges (FD mean = 1.521 vs. MD mean = 1.008, p ¡ 0.001) and expressive adjectives (FD mean = 0.469 vs. MD mean = 0.296, p = 0.009). Tag questions and collective nouns do not carry over (both p ¿ 0.10).

Job application responses show the most comprehensive carry-over. All four features exhibit significant differences: hedges (FD mean = 1.062 vs. MD mean = 0.849, p = 0.008), tag questions (FD mean = 0.016 vs. MD mean = 0.000, p = 0.045), collective nouns (FD mean = 1.245 vs. MD mean = 0.639, p = 0.003), and expressive adjectives (FD mean = 0.609 vs. MD mean = 0.442, p = 0.003).

Resignation letter responses show no significant carry-over for any feature (all p ¿ 0.47). This null result may reflect the highly constrained nature of resignation letters as a genre, which imposes strong stylistic conventions that override prompt-level variation. It may also reflect the low sample size and thus low power of the test.

Table 16: Response-level linguistic feature means by category (Mann-Whitney U, linguistic feature main effect).
<table><tr><td>Category</td><td>Metric</td><td>FD mean</td><td>MD mean</td><td>Direction</td><td>p</td><td>Sig</td></tr><tr><td>Email</td><td>Hedges</td><td>1.521</td><td>1.008</td><td>F &gt; M</td><td>0.0001</td><td>***</td></tr><tr><td>Email</td><td>Tag questions</td><td></td><td></td><td></td><td>0.2554</td><td>ns</td></tr><tr><td>Email</td><td>Collective nouns</td><td></td><td></td><td></td><td>0.1304</td><td>ns</td></tr><tr><td>Email</td><td>Expressive adj.</td><td>0.469</td><td>0.296</td><td>F &gt; M</td><td>0.0087</td><td>**</td></tr><tr><td>Job application</td><td>Hedges</td><td>1.062</td><td>0.849</td><td>F&gt; M</td><td>0.0075</td><td>**</td></tr><tr><td>Job application</td><td>Tag questions</td><td>0.016</td><td>0.000</td><td>F &gt; M</td><td>0.0452</td><td>*</td></tr><tr><td>Job application</td><td>Collective nouns</td><td>1.245</td><td>0.639</td><td>F&gt; M</td><td>0.0028</td><td>**</td></tr><tr><td>Job application</td><td>Expressive adj.</td><td>0.609</td><td>0.442</td><td>F &gt; M</td><td>0.0027</td><td>**</td></tr><tr><td>Resignation letter</td><td>Hedges</td><td></td><td></td><td></td><td>0.5949</td><td>ns</td></tr><tr><td>Resignation letter</td><td>Tag questions</td><td></td><td></td><td></td><td>1.0000</td><td>ns</td></tr><tr><td>Resignation letter</td><td>Collective nouns</td><td></td><td></td><td></td><td>0.4749</td><td>ns</td></tr><tr><td>Resignation letter</td><td>Expressive adj.</td><td></td><td></td><td></td><td>0.9651</td><td>ns</td></tr></table>