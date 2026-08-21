# ContractScrub: A benchmark for final review of legal contracts

Yejin Bang<sup>∗1</sup>, Kirsty Fielding<sup>∗1</sup>, Brandan Oliver<sup>∗1</sup>, Brian Birke<sup>∗1</sup>, Nabeel Seedat<sup>∗1,2</sup>, Andrew M. Bean<sup>∗1,2</sup>

<sup>1</sup>Thomson Reuters Foundational Research, London, UK, <sup>2</sup>Imperial College London, UK

Correspondence: {first.last}@thomsonreuters.com

## Abstract

Legal work, with its heavy reliance on processing large amounts of text, is often considered one of the domains most exposed to the use of LLMs. Contract “scrubbing,” the final review of transactional agreements for errors and inconsistencies, is a particularly suitable task for automation, because it is routine, painstaking work requiring detailed attention to long documents. Scrubbing also seems to align naturally with the general capabilities expected of frontier LLMs around long-context reasoning, consistency checking, and named entity recognition (NER). Despite the economic value and potential for automation, no formal evaluations of LLMs performing contract scrubbing have been conducted. We introduce ContractScrub, the first benchmark designed to evaluate contract scrubbing capabilities, comprising contracts hand-crafted by experienced lawyers over diverse error categories such as misuse of defined terms, incorrect references, and inconsistent language. Frontier models perform surprisingly poorly with only one model reaching 0.75 macro average recall despite strong performance on seemingly related general benchmarks, demonstrating the practical limits of current models and the importance of narrowly targeted, domain-specific benchmarks for measuring real-world impact.

Dataset: https://huggingface.co/tri-fair-lab/contract\_scrub

## 1 Introduction

The practice of law ranks highly among the professions where AI is expected to have the most potential for economic impact [1, 2]. Benchmarks such as LEXam [3] and Stanford Legal Bench [4] are typically used to assess progress in legal capabilities like reasoning. However, translation between capability evaluations and real-world impact can often be limited [5, 6]. By directly testing models on economically valuable tasks, narrowly targeted benchmarks with high ecological validity ofer a more reliable measure of the potential of current legal AI systems.

Transactional law practices involve the negotiation, drafting, and execution of legally binding agreements and navigation of the complex relationships of the parties. Contracts, as a subset of transactional law, place heavy demands on precision and nuance. Small details can afect legal interpretation, a famous example being the lack of an Oxford comma in the statutes governing Maine employ-

![](images/14b872320eb99673bb658ed6a3f99947cdb3e1e81e0eeb0a2633bc9b63134e02.jpg)  
Figure 1 An example of errors that should be found in the scrubbing process.

Table 1 Scrub Review Categories. Each tested contract contains diferent categories of errors.
<table><tr><td>Category</td><td>Explanation</td></tr><tr><td>Defined Terms</td><td>Words or phrases formally defined in the agreement with a designated meaning, often capitalized, underlined, bolded, italicized, or in quotations.</td></tr><tr><td>Undefined Capitalized Terms</td><td>Terms that are capitalized or otherwise treated as a defined term in an agreement but not formally defined.</td></tr><tr><td>Uncapitalized Defined Terms Incorrectly Capitalized Terms In</td><td>Formally defined terms appearing in lowercase when they should be capitalized. Terms that have a definition but are capitalized in a context where they are</td></tr><tr><td>Context Unused Defined Terms</td><td>not being used as a defined term. Defined terms that are not used outside of the specific instance where they are</td></tr><tr><td>Terms Defined Multiple Times</td><td>defined. Terms that are defined more than one time in an agreement.</td></tr><tr><td>Incorrect Section, Article, or Para- graph References</td><td>Internal cross-references that assign an incorrect section, article, or paragraph of the agreement.</td></tr><tr><td>Incorrect Party References</td><td>Instances where a party is referred to by the wrong party name or role.</td></tr><tr><td>Inconsistent Language</td><td>Language in the agreement that directly contradicts itself or other language elsewhere in the agreement.</td></tr></table>

ment contracts, which led to a multi-million dollar settlement <sup>1</sup>. Contract scrubbing occupies a peculiar position in transactional law practices, widely recognized as essential but also generally tedious to carry out. At the very end of (and often throughout) a deal, when negotiations are complete but before signatures are exchanged, lawyers and paralegals will “scrub” a contract; making a final pass to remove any outstanding errors or inconsistencies. This exercise ofers high ecological validity as it is a discrete, well-scoped task with a clear ground truth directly replicating work performed by professionals under realistic conditions.

Though critically necessary, scrubbing is meticulous, repetitive work often performed under intense pressure from the demands of closing a deal, making this routine task error-prone in practice. At the same time, many elements of contract scrubbing (e.g. identifying defined terms, checking consistency of usage, and verifying section references) seem to fit naturally within the capabilities of LLMs around long-context reasoning, consistency checking, and named entity recognition (NER). Legal practitioners frequently approach this risk-control function as essential but tedious work. If the scrub could be automated reliably, practitioners could redirect their attention to other matters: identifying issues that demand legal judgment, advising clients on risk, and advancing negotiations. Attorneys could also perform automated scrubbing more frequently throughout the negotiation of a contract, reducing the risk of last-minute changes disrupting an otherwise settled deal.

The clear potential for AI-assisted contract scrubbing makes it a natural domain for a targeted, task-specific benchmark, but, to the best of our knowledge, no such benchmark exists. Current contract-related benchmarks focus largely on reasoning about contracts rather than reviewing them for precision and consistency. CUAD [7] tests whether models can identify legally significant provisions such as governing law, exclusivity, and non-compete clauses. Liu, Li, Ma, Zhao, and Du introduced ContractEval, a benchmark focused on evaluating LLMs’ ability to identify clause-level legal risks in commercial contracts. And, general-purpose tasks in NLP like "needle-in-a-haystack" [9], are more similar to the actual work required for scrubbing, but do not share the idiosyncrasies of the legal domain.

To address this gap, we introduce ContractScrub, the first benchmark designed to evaluate LLM performance on the various (including final) review stages of the contract or deal lifecycle. ContractScrub comprises 3,014 annotated tasks across 44 contracts drawn from CUAD [7], including scrubbing elements and errors, e.g., defined term inconsistencies, capitalization errors, and cross-reference failures. The contracts are hand annotated by experienced lawyers to include representative test issues.

We evaluate 9 frontier and open-weight models of varying families and sizes on ContractScrub and find that contract scrubbing remains a substantially harder task for LLMs than its individual components might suggest. The best-performing model, GPT-5.5, reaches a macro-average recall score of only 0.750, and all F1 scores are below 0.650. Performance is uneven across issue types: models handle categories with explicit lexical signals (e.g., defined term; $\mu = . 8 3 5 )$ much better than those requiring inference of intent within context (e.g., Incorrect Capitalization in Context; $\mu = . 4 2 7 )$ . Beyond the practical value of filling the measurement gap, our benchmark ofers a valuable theoretical insight. Despite many of the tasks required for scrubbing sharing a strong similarity with conventional NLP tasks, most frontier models fall short of the performance that would be expected based on their general capabilities, highlighting the value of domain-specific benchmarking as a practice. Furthermore, enabling reasoning yields only moderate gains, concentrated in categories that require full-document term consistency rather than deeper legal interpretation. These findings underscore the need for targeted benchmarks in professional domains, and we release ContractScrub to contribute to that efort.

## 2 Related Work

Legal Benchmarks for LLMs Evaluating the legal capabilities of LLMs is an active area of research, with recent benchmarks aimed at measuring broad legal competence across diverse tasks. LEXam [3], Stanford LegalBench [4], LawBench [10], and LexGLUE [11] exemplify this general-purpose approach, evaluating LLMs in collections of legal problems that test skills such as legal knowledge, reasoning, and interpretation. These suites are valuable for measuring overall progress in legal AI, but their tasks typically probe narrow skills in isolation rather than end-to-end review of a single long document. Contract scrubbing is one such workflow: it requires sustained attention to long, highly structured documents, where small textual inconsistencies can have legal or commercial consequences, or both.

A closer line of work does evaluate LLMs in contract-specific tasks. CUAD [7] tests whether models can identify important clauses and legal attributes in thousands of annotated examples of commercial contracts, while MAUD [12] provides a similarly large-scale benchmark for locating deal-points in merger agreements. Both cast contract review largely as reading comprehension over predefined legal categories, requiring models to locate relevant provisions or determine whether particular deal points are present — e.g “is there a noncompete clause?” ContractNLI [13] evaluates contractual reasoning asking whether contract provisions entail, contradict, or leave undetermined a set of hypotheses. The Lease benchmark of Leivaditi, Rossi, and Kanoulas evaluates NER and red-flag detection on residential lease agreements, and ContractEval [8] extends this to legal-risk identification across a wider set of commercial agreements. Both move closer to practical review, but still assess classification of provisions against a fixed taxonomy of risk types. In contrast, ContractScrub evaluates defect identification as a recall-sensitive, full-document task. Rather than answering a supplied question, classifying a provision against a known taxonomy, or evaluating a predefined hypothesis, the model is expected to surface drafting, consistency, and document-hygiene defects in a contract, which, if missed, can have material legal and practical consequences. In short, prior contract benchmarks largely evaluate “is X here?” under a known schema. ContractScrub evaluates over the whole document “what is wrong with this relative to the contract’s own internal conventions?”

Related Non-Legal Tasks Contract scrubbing requires the joint application of several general-purpose LLM capabilities, including long-context reasoning, precise localization, consistency checking, and referential understanding, in a structured legal setting. Existing non-legal benchmarks probe related capabilities, but generally in more isolated settings. Long-context benchmarks [15, 16] evaluate whether models can reason over extended inputs, a prerequisite for contract scrubbing because relevant errors may be sparsely distributed across a lengthy document. Needle-in-a-Haystack-style evaluations [9, 17] require models to find a single embedded fact within a long context, which parallels the localization demands of scrubbing but not the requirement to surface all relevant defects based on the broader context of the document. FaithEval [18] includes inconsistency detection over short passages, resembling conflicting-definition or inconsistent-term errors. However, it treats consistency primarily as a binary classification problem, rather than requiring models to find, locate, and contextualize inconsistencies across a long document. IdentifyMe [19] benchmarks entity reference resolution, a capability relevant to detecting incorrect party names and section references,

Figure 2 ContractScrub construction pipeline. Contracts are sourced and screened for existing structural issues (Stage 1), annotated for existing drafting errors (Stage 2), and augmented with targeted additional errors (Stage 3) to produce the gold answer. All steps are carried out by experienced lawyers.  
![](images/76526e9ed1e15f7c4a05180b55fcbc3add5079a88461c3dd5a99e3b161ba3195.jpg)  
but operates on general-domain text. Each of these benchmarks isolates an individual component capability. ContractScrub is designed to evaluate their collective application to comprehensive defect identification and localization in full-length contracts.

## 3 ContractScrub Benchmark

Contract review involves reading and understanding a contract thoroughly to identify errors, analyse risks, and ensure consistency. Throughout the negotiation process and certainly at the final stage of review, attorneys conduct a dedicated pass to eliminate residual errors and inconsistencies – a process commonly known as “scrubbing.” Seemingly small errors in contracts are time-intensive to identify and review and can be highly consequential, requiring legal practitioners to catch them in the contract-drafting lifecycle. However, they remain dificult to identify under the time pressures typically accompanying such a routine task (that is, nearing deal closing or contract execution) and because manual review is repetitive, subject to fatigue, and complicated by multiple negotiated drafts, schedules, and amendments. Automating or augmenting legal practitioners at this stage with LLMs ofers a practical path to freeing practitioners for higher-value work by reducing avoidable drafting mistakes, improving first-pass review, supporting junior attorneys, accelerating quality control, and allowing senior lawyers to spend more time on issues requiring their judgment and experience, all the while upholding the high standards commensurate with the legal profession.

## 3.1 Task Overview

ContractScrub is the first benchmark evaluating LLM performance on the scrubbing pass. The task is defined as follows. Let ${ \mathcal { C } } = \{ c _ { 1 } , \ldots , c _ { n } \}$ be a corpus of n contracts. For each contract $c _ { i } ,$ , a gold annotation $R _ { i }$ is a multiset of tuples, where each tuple $\mathbf { r } = \left( \kappa , \mathbf { f } \right)$ consists of a category label $\kappa \in \kappa$ and a category-specific field vector f. The category set K comprises nine elements covering one defined-term extraction category and eight drafting-error categories (Table 1). The field vector f encodes the category-specific fields for each instance — for example, f = (term, location) for most term-level categories, and f = (location , location ) for relational categories such as inconsistent terms. The task is designed to simulate the scrubbing pass as performed by actual attorneys in practice: given contract $c _ { i }$ and an instruction prompt I as input, a model must produce a predicted multiset $\hat { R } _ { i }$ of tuples drawn from the same schema. Formally, each model M produces a predicted annotation:

$$
\hat { R } _ { i } = \mathcal { M } ( I , c _ { i } ) ,
$$

Performance is measured by comparing $\hat { R } _ { i }$ against the gold $R _ { i }$ across all categories and contracts.

## 3.2 Dataset

To systematically evaluate scrubbing capability of LLMs, we construct dedicated annotated data (Figure 2). The schema of categories is designed by licensed attorneys with professional experience practicing and litigating contract law. Then, the further annotation and creation of the dataset is conducted by 9 diferent lawyers with experience in various forms of commercial, corporate, and contract law. All the lawyers have been practicing law for at least 8 years, with 8 having more than 10 years experience and 6 having more than 15 years.

ContractScrub consists of 3,014 annotated tasks across 9 categories drawn from 44 contracts. Each contract $c _ { i }$ has corresponding gold annotation $R _ { i }$ , which consists of diferent categories explained in Table 1. The guiding principles in creating this benchmark are improving contract hygiene (precision and consistency), reducing the risk of misinterpretation and ambiguity, and preserving client confidence by uncovering drafting defects that can have outsized impacts on contractual meaning, clarity, and outcomes.

Scrub Categories Each gold answer covers nine annotation categories listed in Table 1: one defined-term extraction category and eight drafting-error categories. The categories were selected because they are common, concrete, and potentially impactful on a contract with these types of issues. These errors are also often overlooked because they are embedded in otherwise innocuous contract language and surfaced only with both deep contextual awareness and exacting attention to detail.

Each category carries distinct legal implications. To illustrate, consider the ‘Uncapitalized Defined Terms’. Where a contracting party defines "Representative" narrowly, encompassing only company oficers and legal counsel, but subsequently uses "representative" in lowercase within a confidentiality provision, a counterparty may reasonably interpret the inconsistency as intentional and apply the broader, ordinary meaning of the term. The result is that the countererparty may share sensitive information with a substantially wider group than the drafting party intended. By contrast, some errors may be characterized as more mundane in nature, with comparatively limited legal consequence in isolation. Nonetheless, even errors that appear minor can introduce ambiguity about agreed terms, generate friction between contracting parties, delay deal execution, and erode client confidence – outcomes that carry real commercial cost regardless of their legal characterization. See Appendix A for implication of each category.

Construction Pipeline The process consists of three stages as shown in Figure 2: (i) selecting and reviewing source contracts for structural flaws; (ii) annotating existing issues; and (iii) inserting additional issues to create the final gold answers. Source contracts are drawn from the open-source CUAD dataset [7], which is in turn drawn from $\mathrm { E D G A R ^ { 2 } }$ , an open repository of documents from publicly-owned US companies. ContractScrub does not use CUAD’s labels: each contract is independently reviewed and annotated by legal subject matter experts (SMEs) for categories relevant to scrubbing contracts. We describe the pipeline in greater detail:

1. Contract sourcing and review. Contracts are selected from the source pool to span a range of subject matter and drafting styles. Each candidate is reviewed by an SME for fundamental drafting flaws that would render it unusable, and for suitable length - short enough to validate within reasonable efort, but long enough to host the full set of targeted issues without becoming structurally invalid, which in practice means approximately 10-15 pages. Where necessary, contracts were also revised to improve overall coherence by, for example, removing empty exhibits or improving consistency. This process yields an initial corpus $\hat { \mathcal { C } } = \{ \hat { c } _ { 1 } , \hdots , \hat { c } _ { n } \}$

2. Annotation of existing issues. For each $\hat { c } _ { i }$ , the SME records every defined term and any pre-existing instances of error categories $k \in \mathcal { K }$ already present in the contract as tuples $\scriptstyle ( \kappa , \mathbf { f } )$ . The quality of contracts in this dataset varies widely, and some contracts will have several existing errors to be identified.

3. Insertion of new issues. The SME then purposefully introduces additional drafting errors across the error categories (Table 1), reflecting realistic mistakes seen in transactional practices. These annotations aim for broad and approximately balanced representation across error types $\kappa .$ while preserving the coherence and legal plausibility of the edited contract. Each inserted issue is recorded as a tuple $\scriptstyle ( \kappa , \mathbf { f } )$ under the same schema as pre-existing issues. Every occurrence is logged separately: repeated terms or errors in the same section are not de-duplicated. The final result is a new contract, $c _ { i } \in \mathcal { C }$ for each original contract ${ \hat { c } } _ { i } ,$ and gold answers, $R _ { i }$ , including all defined terms, any identified pre-existing issues and all SME-inserted issues.

After the data creation pipeline, two of the lawyers performed targeted reviews of the contracts and gold answers for quality, and suggested changes to the contracts, gold answers, and task prompts as necessary to increase alignment.

## 3.3 Metrics

We evaluate performance via multiset comparison between gold and predicted tuples. For each category $k \in \mathcal { K }$ tuples from $R _ { i }$ and $\hat { R } _ { i }$ are normalized and matched across all contracts; true positives $( T P _ { k } )$ are matched pairs, false positives $( F P _ { k } )$ are unmatched tuples in $\hat { R } _ { i }$ , and false negatives $( F N _ { k } )$ are unmatched tuples in $R _ { i } .$ , with counts pooled across all $c _ { i } \in \mathcal { C }$

We focus on recall, $\begin{array} { r } { R _ { k } = \frac { T P _ { k } } { T P _ { k } + F N _ { k } } } \end{array}$ , as the primary metric, though we also report precision and F1 scores. For contract scrubbing, the potential costs of false negatives are significantly higher than the cost of false positives, since it is much easier to check whether a flagged issue is real than to identify issues that were not previously known. Practically, since the CUAD contracts can also contain pre-existing errors, using recall allows us to focus primarily on the known issues that experts have identified and inserted without being impacted by any potential remaining unknown issues.

Overall performance is reported as the macro-average across all $| \kappa | = 9$ categories:

$$
\mathrm { M a c r o - R } = { \frac { 1 } { | { \boldsymbol { K } } | } } \sum _ { k \in { \boldsymbol { K } } } R _ { k } .
$$

We additionally report a word-only variant in which location fields are removed from each tuple prior to comparison. This isolates errors of identification – whether the model found the correct term, reference, or party pair – from errors of localization, i.e., whether it cited the correct section.

## 3.4 Models

We evaluate a range of proprietary and open-source models: GPT-5.5, GPT-5.2, o4-mini; Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5; Qwen 3.5-397B; and Gemini 3.1 pro, Gemini 2.5 pro, Gemma-4-26B. This selection spans multiple model families and capability tiers – from frontier flagship models to smaller, eficient variants – enabling us to assess how model scale and family afect performance on the legal document scrubbing task. We provide a full list of the models tested and their hyperparameters in Appendix F.

## 3.5 Implementation Details

Each model is prompted to scrub a given contract, with instructions describing each of the nine categories and the output and reference formats prompt (See the full prompts in Appendix G). We prompt the models to identify issues from each category in separate instances to help reduce the competing task demands. Each response is expected to return a single JSON object with one key per category $k \in \mathcal { K }$ Model outputs are parsed into category-specific tuples – for example, (term, location) for defined terms and (wrong reference, correct reference, location) for incorrect references. These tuples are compared deterministically against gold annotation $R _ { i }$ as multisets, so repeated occurrences are scored independently as the same term or error type may occur multiple times in diferent locations, and each occurrence imposes a separate burden on a reviewer.

Normalisation before scoring To avoid penalizing superficial formatting diferences, all tuple fields are normalized on both the gold and prediction sides before comparison. We apply normalization as follow: (1) Lower-casing of Terms: Term and word fields are lowercased and stripped of whitespace (‘Licensor’ and ‘licensor’ match). (2) Location field Canonicalization: Location fields are canonicalized by collapsing parenthetical section levels while preserving major/minor dots: e.g., 1(a)(i) → 1ai, 1.1(h)(vii) → 1.1hvii, so $1 . 1 ( \mathsf { d } ) \neq 1 1 ( \mathsf { d } )$ . Special labels such as P/Preamble, R/Recitals, Exhibit X, Schedule X, and Signature Block are also canonicalized. (3) Symmetric Tuple Matching: Paired-location categories, such as Conflicting Definitions and Inconsistent Terms, are compared order-independently: a gold tuple containing locations (1e, 7a) is treated as equivalent to a predicted tuple containing (7a, 1e).

Table 3 Recall score comparison across models and scrub categories. Color heatmap applied to all metric rows. Recall Overall shown without heatmap for reference. All models use reasoning variants.
<table><tr><td rowspan=1 colspan=10>ModelMetricGemini   Claude  Gemini Claude            Claude  Qwen3.5GPT-5.53.1 ProSonnet 4.62.5 Pro           GPT-5.2            (397B) o4-miniOpus 4.7Haiku 4.5Overall Recall               0.750   0.744    0.686    0.632   0.616    0.589    0.445    0.438   0.409</td></tr><tr><td rowspan=1 colspan=1>Defined Terms</td><td rowspan=1 colspan=2>0.905   0.901</td><td rowspan=1 colspan=2>0.862    0.862</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.876</td><td rowspan=1 colspan=3>0.753    0.793   0.730</td></tr><tr><td rowspan=1 colspan=1>Undef. Capitalized Terms</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.440</td><td rowspan=1 colspan=1>0.367</td><td rowspan=1 colspan=1>0.319</td><td rowspan=1 colspan=1>0.447</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.216</td><td rowspan=1 colspan=1>0.171</td></tr><tr><td rowspan=1 colspan=1>Uncapitalized Defined Terms</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.703</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.306</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>0.227</td></tr><tr><td rowspan=1 colspan=1>Incorr. Capitalized in Context</td><td rowspan=1 colspan=1>0.744</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.527</td><td rowspan=1 colspan=1>0.357</td><td rowspan=1 colspan=1>0.481</td><td rowspan=1 colspan=1>0.419</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.349</td><td rowspan=1 colspan=1>0.101</td></tr><tr><td rowspan=1 colspan=1>Unused Defined Terms</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>0.931</td><td rowspan=1 colspan=1>0.896</td><td rowspan=1 colspan=1>0.782</td><td rowspan=1 colspan=1>0.871</td><td rowspan=1 colspan=1>0.822</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1>0.485</td><td rowspan=1 colspan=1>0.703</td></tr><tr><td rowspan=2 colspan=1>Terms Defined Multiple TimesIncorr. Sec./Art./Para. Refs</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.763</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.526</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.526</td></tr><tr><td rowspan=1 colspan=1>0.760</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.687</td><td rowspan=1 colspan=1>0.533</td><td rowspan=1 colspan=1>0.533</td></tr><tr><td rowspan=1 colspan=1>Incorrect Party References</td><td rowspan=1 colspan=1>0.562</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.531</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1>0.315</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.308</td></tr><tr><td rowspan=1 colspan=1>Inconsistent Language</td><td rowspan=1 colspan=1>0.713</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.713</td><td rowspan=1 colspan=1>0.678</td><td rowspan=1 colspan=1>0.575</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>0.460</td><td rowspan=1 colspan=1>0.379</td></tr></table>

## 4 Results

Table 2 Main Results. Precision (P), Recall (R), F1 scores, and cost for all evaluated models. The best score in each column is in bold; the second best is underlined. Cost column includes the mean price (\$) and time (sec.) per contract.
<table><tr><td>Model</td><td>R</td><td>P</td><td>F1</td><td>Cost ($ | s)</td></tr><tr><td>GPT-5.5</td><td>0.750</td><td>0.580</td><td>0.632</td><td>1.38 | 533</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.744</td><td>0.616</td><td>0.655</td><td>0.19 | 76</td></tr><tr><td>Claude Sonnet 4.6</td><td>0.686</td><td>0.620</td><td>0.637</td><td>1.52 | 258</td></tr><tr><td>Gemini 2.5 Pro</td><td>0.632</td><td>0.527</td><td>0.557</td><td>0.13 | 48</td></tr><tr><td>Claude Opus 4.7</td><td>0.616</td><td>0.644</td><td>0.621</td><td>0.66 | 23</td></tr><tr><td>GPT-5.2</td><td>0.589</td><td>0.540</td><td>0.553</td><td>0.32 | 192</td></tr><tr><td>Qwen 3.5 (397B)</td><td>0.438</td><td>0.268</td><td>0.316</td><td>0.04 | 194</td></tr><tr><td>Claude Haiku 4.5</td><td>0.445</td><td>0.592</td><td>0.492</td><td>0.68 | 108</td></tr><tr><td>04-mini</td><td>0.409</td><td>0.548</td><td>0.453</td><td>0.22 | 301</td></tr><tr><td colspan="5">Without Reasoning</td></tr><tr><td>GPT-5.5</td><td>0.643</td><td>0.463</td><td>0.525</td><td>0.42 | 22</td></tr><tr><td>Claude Opus 4.7</td><td>0.526</td><td>0.478</td><td>0.490</td><td>0.42 | 5</td></tr><tr><td>Gemma 4 (26B)</td><td>0.365</td><td>0.315</td><td>0.326</td><td>0.01 | 26</td></tr></table>

Overall The benchmark is reasonably challenging, with GPT-5.5 having a recall score of 0.750, and the precision and F1 scores across models nearly all less than 0.650 as shown in Table 2. Most models have higher recall than precision, but there are tradeofs between the two scores, with the model having the highest recall, GPT-5.5, actually having being fourth in precision. Focusing on recall, performance drops of quickly from the top few models to the weaker ones. The most expensive model, GPT-5.5, costs only \$1.38 per contract, significantly less than a lawyer or paralegal, though at scale the cost diferences to other models may be more relevant. Runtime is also a consideration, with the top performance of GPT-5.5 also requiring the most latency, taking nearly 9 minutes to finish while Gemini 3.1 Pro returns within 90 seconds.

As expected, smaller and older models generally lag behind their larger counterparts. However, Qwen3.5-397b (.438 recall) performs poorly despite the large parameter count, comparable to the much smaller o4-mini (.409) and Haiku 4.5 (.445). This suggests that raw model scale does not straightforwardly translate to legal document analysis ability.

Performance by Category As shown in Table 3, model performance varies widely across the different scrub categories. The gap in recall scores

between the easiest (Defined Terms) and hardest category (Undefined Capitalized Terms) is 0.484. Tasks which involve understanding legal relationships between diferent parts of the contracts tend to be more dificult than those which only require spotting repetitions or omissions. Easier categories – Defined Terms (mean recall .835), Terms Defined Multiple Times (.689), Unused Defined Terms (.781) – have explicit lexical signals and require less legal reasoning. In contrast, Incorrect Party References (.362), Incorrect Capitalization in Context (.427), and Undefined Capitalized Terms (.351) require inferring intent from context rather than matching surface form. Catching capitalized undefined terms requires the ability to flag inconsistencies against an internally maintained registry of definitions in combination with understanding which capitalizations are legally significant. Even the strongest performing model, GPT-5.5, achieves only 0.514 on Undefined Capitalized Terms and 0.562 on Party References.

![](images/f2f7b3cedaba73b4044e165bf969c099252644c59d0de136e64f41b92215717a.jpg)  
(a) Correlation of Subtask Performance. Task categories show clusters, between those related to definitions, with less correlation between the other categories (Pearson R).

![](images/3cda1fbd37c9814f9de6a44f09ab5845b97200ed22a820ae1bb27167b1f00c2c.jpg)  
(b) Long Distance References. Recall vs. reference distance, by model.

Correlations (Pearson’s R) between the category scores, shown in Figure 3a, are also relatively low, but positive. There is a weak cluster of categories around incorrect usage of definitions, with uncapitalized defined terms, unused defined terms, incorrectly capitalized terms, and incorrect references forming a group, but overall the diferent subtasks appear to measure mostly independent capabilities.

Reasoning As some of the tasks involved in contract scrubbing may appear not to require heavy reasoning, we tested whether turning of reasoning impacts model performance for the top models. The scores without reasoning are shown at the end of Table 2. For the two models we tested in both modes, including reasoning increased performance moderately at the cost of slower and more expensive inference. We include an analysis by issue category in Appendix B.

Long-Distance Referencing In reviewing the errors made by the models, we observed a qualitative tendency to struggle more with identifying incorrect section references that were further away (in terms of characters) from the section that they were meant to reference. As a targeted quantitative assessment, we manually annotated the distance in characters between the references and the referred sections for half of the “Incorrect Section References” gold standard items. As shown in Figure 3b, the general tendency is that predicted recall decreases as the distance between two target entities grows, with the efect becoming most pronounced past ten thousand characters (roughly 5-6 pages).

## 5 Discussion

Contract scrubbing in deployment Contract scrubbing is a practical task requiring reasoning over long documents with close attention to detail. Although the mechanics of the task (e.g. spotting capitalization errors and checking for duplicative definitions) seem simple, legal knowledge and judgement are still an important aspect. Such foundational tasks, though routine, are also load-bearing: errors at this level propagate upward, undermining careful negotiations, expected outcomes, and client relationships. Full automation of contract scrubbing requires a high bar for model performance, likely to be higher than the 75% recall currently attained. However, the speed and low cost of all of the evaluated models points to potential for integrating LLMs into existing workflows alongside experts, as well as adding more frequent scrubbing into contract lifecycles to reduce the issues that remain to be found at the end.

Sources of task difficulty for LLMs The dificulty of contract scrubbing stems from the simultaneous demands of several core tasks: (1) long-context reasoning, as errors must be detected across lengthy documents; (2) structured consistency checking, which requires the model to maintain and query an implicit index of defined terms throughout the document; (3) sparse error detection, where errors are rare, demanding high precision to avoid hallucination while retaining suficient sensitivity to catch true positives; (4) heterogeneous error types, meaning a single context includes various types of errors; and (5) document-internal context dependence, where correctness is determined not by external legal standards but by conventions established within the document itself.

A number of existing benchmarks focus on measuring one or more of these abilities in more generic and isolated contexts, and no benchmark combines all of them, and the scores on our benchmark are lower than might be expected based on other existing benchmarks. For example, our inconsistent term detection category resembles the inconsistency detection evaluated in FaithEval [18], where models determine whether a passage contains internal contradictions. On FaithEval, GPT-4 and Claude Sonnet 3.5 achieved 89.4% and 92.2%, respectively, while more recent models from the same family achieve only 75.0% and 68.6% recall on inconsistency detection in ContractScrub. Similarly, our term-finding tasks are similar in concept to named entity recognition or needle in a haystack tasks, where models search for specific target words or phrases with long documents. Chen et al. evaluate retrieval from long contexts, where Claude 3 achieves 98.28% at 128K tokens; Gemini 1.5 likewise reports near-perfect recall on needle-in-a-haystack probes [20]. We do find in our results that performance across the tasks related to defined terms is correlated, but all of the tasks prove to be substantially harder in our setting than standard needle in a haystack. In addition to requiring many capabilities at once, we expect that the domain specific terminology and structure of legal contracts adds in a layer of dificulty further lowering scores.

## Broader implications

Our findings in ContractScrub highlight two important themes. First, they demonstrate a disconnect between perceived dificulty of a task within a professional domain and its dificulty for LLMs. This echoes observations that models can struggle on tasks that appear relatively simple [21, 22], even when they are capable of passing the SAT or the bar exam. Building on this, the gap between model performance on ContractScrub and on existing benchmarks targeting individual capabilities suggests that evaluations of isolated abilities may not fully capture how models perform when those abilities are exercised jointly. Targeted benchmarks may focus on skills that seem cognitively significant to humans, while potentially overlooking factors that humans find trivial but which actually pose challenges for an LLM. Our analysis of the relationship between the distance between inconsistencies and model performance highlights one such example, where many models struggled with checking section references across longer spans. Contract scrubbing also requires following conventions internal to the context of a document rather than external world or domain-specific knowledge, which we believe is relatively under-represented in current benchmark eforts.

Limitations ContractScrub has a few limitations worth considering. First, while the benchmark has a large number of issues for the models to identify, they are drawn from only 44 contracts, a more modest scale. This size is suficient to surface meaningful performance diferences across models, but a larger corpus would reduce the efects of idiosyncrasies in any particular contract and better represent the universe of corporate contracts. Second, the benchmark’s dataset contains only English-language contracts with an approximate length of 10–15 pages. This design choice reflects the practical setting of a closing-stage scrub in many real use cases, but limits generalisability to other legal traditions, languages, and deal types involving diferent document lengths such as shorter term sheets or multi-hundred-page complex financings. Third, our evaluation protocol requires models to produce structured JSON output, which may depress raw performance scores relative to a free-form setting and introduces sensitivity to instruction-following ability as a confounder. That said, structured output is arguably a realistic requirement for any production scrubbing tool, where downstream parsing and issue tracking depend on machine-readable responses; the restriction, therefore, reflects a genuine constraint of the deployment context rather than an arbitrary evaluation choice.

## 6 Conclusion

We introduce ContractScrub, a novel evaluation for assessing the ability of LLMs to scrub legal contracts for errors. We found that performance varied widely among models, with many scoring well on tasks that are primarily lexical in nature and scoring poorly when the tasks require more complex legal reasoning on top of contextual document analysis skills. These findings contrast with results on related general-domain benchmarks, where frontier models are generally very capable. Our results reinforce the case for narrowly targeted, ecologically valid, benchmarks in professional domains. As AI systems are increasingly deployed in professional settings, benchmarks that directly measure economically relevant, task-specific performance are important for both scientific progress and responsible deployment. We ofer ContractScrub as one step in that direction.

## References

[1] M. Lane and A. Saint-Martin. The impact of Artificial Intelligence on the labour market: What do we know so far? Jan. 2021. doi: 10.1787/7c895724-en.

[2] Ed Felten, Manav Raj, and Robert Seamans. How will Language Modelers like ChatGPT Afect Occupations and Industries? 2023. arXiv: 2303.01157 [econ.GN]. url: https://arxiv.org/abs/2303. 01157.

[3] Yu Fan et al. “Lexam: Benchmarking legal reasoning on 340 law exams”. In: (May 2025).

[4] Neel Guha et al. “Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models”. In: Advances in neural information processing systems 36 (2023). Ed. by Alice Oh et al., pp. 44123–44279. doi: 10.52202/075280-1915.

[5] Reva Schwartz et al. “Reality Check: A New Evaluation Ecosystem Is Necessary to Understand AI’s Real World Efects”. In: arXiv preprint arXiv:2505.18893 (2025).

[6] Andrew M. Bean et al. Measuring what Matters: Construct Validity in Large Language Model Benchmarks. Nov. 2025. doi: 10.48550/arxiv.2511.04703. arXiv: 2511.04703 [cs.CL]. url: https://arxiv.org/abs/ 2511.04703.

[7] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. “CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review”. In: Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track. Ed. by Joaquin Vanschoren and Sai Kit Yeung. Mar. 2021. doi: 10.48550/arxiv.2103.06268. url: https://arxiv.org/pdf/2103.06268.

[8] Shuang Liu, Zelong Li, Ruoyun Ma, Haiyan Zhao, and Mengnan Du. “ContractEval: Benchmarking LLMs for Clause-Level Legal Risk Identification in Commercial Contracts”. In: Proceedings of the Natural Legal Language Processing Workshop 2025. Ed. by Nikolaos Aletras et al. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 291–291. isbn: 979-8-89176-338-8. doi: 10.18653/v1/2025.nllp-1.19. url: https://aclanthology.org/2025.nllp-1.19/.

[9] Greg Kamradt. LLMTest\_NeedleInAHaystack: Pressure Testing LLMs. https://github.com/gkamradt/ LLMTest\_NeedleInAHaystack. GitHub repository. 2023.

[10] Zhiwei Fei et al. “Lawbench: Benchmarking legal knowledge of large language models”. In: Proceedings of the 2024 conference on empirical methods in natural language processing. Ed. by Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen. Association for Computational Linguistics, 2024, pp. 7933–7962. url: https://aclanthology.org/2024.emnlp-main.452.pdf.

[11] Ilias Chalkidis et al. “LexGLUE: A Benchmark Dataset for Legal Language Understanding in English”. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics. Ed. by Smaranda Muresan, Preslav Nakov, and Aline Villavicencio. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 4310–4330. doi: 10.18653/v1/2022.acl- long.297. url: https://aclanthology.org/2022.acl-long.297/.

[12] Steven Wang et al. “MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding”. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Ed. by Houda Bouamor, Juan Pino, and Kalika Bali. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 16369–16382. doi: 10.18653/v1/2023.emnlp-main.1019. url: https://aclanthology.org/2023.emnlp-main.1019/.

[13] Yuta Koreeda and Christopher Manning. “ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts”. In: Findings of the Association for Computational Linguistics: EMNLP 2021. Ed. by Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih. Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 1907–1919. doi: 10.18653/v1/2021.findings-emnlp.164. url: https://aclanthology.org/2021.findings-emnlp.164/.

[14] Spyretta Leivaditi, Julien Rossi, and Evangelos Kanoulas. “A benchmark for lease contract review”. In: arXiv preprint arXiv:2010.10386 abs/2010.10386 (Oct. 2020). issn: 2331-8422. doi: 10.48550/arxiv. 2010.10386. url: https://arxiv.org/pdf/2010.10386.

[15] Cheng-Ping Hsieh et al. “RULER: What’s the Real Context Size of Your Long-Context Language Models?” In: First Conference on Language Modeling. 2024. url: https://openreview.net/forum?id= kIoBbc76Sy.

[16] Zhan Ling et al. “Longreason: A synthetic long-context reasoning benchmark via context expansion”. In: arXiv preprint arXiv:2501.15089 (Jan. 2025). issn: 2331-8422. doi: 10.48550/arxiv.2501.15089. url: https://arxiv.org/pdf/2501.15089.

[17] Pei Chen et al. “LongLeader: A Comprehensive Leaderboard for Large Language Models in Long-context Scenarios”. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Ed. by Luis Chiruzzo, Alan Ritter, and Lu Wang. Albuquerque, New Mexico: Association for Computational Linguistics, Apr. 2025, pp. 8734–8750. isbn: 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.439. url: https://aclanthology.org/2025.naacl-long.439/.

[18] Yifei Ming et al. “Faitheval: Can your language model stay faithful to context, even if" the moon is made of marshmallows"”. In: International Conference on Learning Representations. Vol. 2025. 2025, pp. 29430–29456.

[19] Kawshik Manikantan, Makarand Tapaswi, Vineet Gandhi, and Shubham Toshniwal. “IdentifyMe: A Challenging Long-Context Mention Resolution Benchmark for LLMs”. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers). Ed. by Luis Chiruzzo, Alan Ritter, and Lu Wang. Albuquerque, New Mexico: Association for Computational Linguistics, Apr. 2025, pp. 768–777. isbn: 979-8-89176-190-2. doi: 10.18653/v1/2025.naacl-short.64. url: https://aclanthology.org/2025. naacl-short.64/.

[20] Google Cloud. The Needle in the Haystack Test and How Gemini 1.5 Pro Solves It. Google Cloud Blog. 2024.

[21] Marianna Nezhurina, Lucia Cipolina-Kun, Mehdi Cherti, and Jenia Jitsev. “Alice in wonderland: Simple tasks showing complete reasoning breakdown in state-of-the-art large language models”. In: arXiv preprint arXiv:2406.02061 (June 2024). doi: 10.48550/arxiv.2406.02061. url: https://arxiv.org/pdf/ 2406.02061.

[22] Fabrizio Dell’Acqua et al. Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Efects of Artificial Intelligence on Knowledge Worker Productivity and Quality. SSRN Scholarly Paper. Rochester, NY, Sept. 2023. doi: 10.2139/ssrn.4573321. Social Science Research Network: 4573321. (Visited on 05/22/2026).

## Appendix

<table><tr><td colspan="2">PART 1 Additional Context and Experiments</td><td></td></tr><tr><td>A</td><td>Potential Implications of Each Error Category</td><td>13</td></tr><tr><td>B</td><td>Reasoning Ablation</td><td>15</td></tr><tr><td>C</td><td>Term-only Scoring Ablation</td><td>15</td></tr><tr><td>D</td><td>Per-category Precision &amp; Recall</td><td>16</td></tr><tr><td>PART 2</td><td>Reproducibility Details</td><td></td></tr><tr><td>E</td><td>Data Access and Usage Notes</td><td>17</td></tr><tr><td>F</td><td>Inference Details</td><td>17</td></tr><tr><td>G</td><td>Prompt Templates</td><td>17</td></tr></table>

## PART 1: Additional Context and Experiments

## A Potential implications of each error category

Table 4 Scrub error category counts in ContractScrub and per-contract averages

<table><tr><td>Category</td><td></td><td>Total Avg/Contract</td></tr><tr><td>Defined Terms</td><td>1,505</td><td>34.2</td></tr><tr><td>Undefined Capitalized Terms</td><td>689</td><td>15.7</td></tr><tr><td>Uncapitalized Defined Terms</td><td>317</td><td>7.2</td></tr><tr><td>Unused Defined Terms</td><td>202</td><td>4.6</td></tr><tr><td>Incorrect Section, Article, or Paragraph References</td><td>150</td><td>3.4</td></tr><tr><td>Incorrect Party References</td><td>130</td><td>3.0</td></tr><tr><td>Incorrectly Capitalized Terms In Context</td><td>129</td><td>2.9</td></tr><tr><td>Terms Defined Multiple Times</td><td>97</td><td>2.2</td></tr><tr><td>Inconsistent Language</td><td>87</td><td>2.0</td></tr></table>

Each category has diferent potential implications for the parties agreeing to the contract. Not all consequences are strictly "legal" in nature such that a law is broken or a contract is breached. Courts may, in certain circumstances, correct obvious clerical or scrivener’s errors where the parties’ mutual intent is evident from the surrounding context. Nevertheless, such errors remain undesirable, and those that introduce substantive ambiguity are of greater legal concern. From a practical standpoint, however, the threshold for harm is lower. Any error that delays deal execution or introduces friction between parties, including ostensibly minor clerical mistakes that create confusion about the operative terms, represents a failure with real professional consequences, up to and including reputational damage and client attrition.

Below, we describe what each category represents and provide examples of possible impacts that could arise from missing this type of error:

Undefined Capitalized Terms: Terms that are capitalized or otherwise treated as a defined term in an agreement but not formally defined. For example, an agreement provides: “Party A shall comply with all applicable requirements in the Approved Specification,” but Approved Specification is not defined. This matters because an undefined capitalized term indicates that a contracting party specifically intended a special contractual meaning. As a result, an ambiguity is created and a party may later dispute what specification was approved and whether it was applicable, whether a breach occurred, and whether outside evidence is admissible to contradict, vary, or add to the terms of the agreement.

Uncapitalized Defined Terms Formally defined terms appearing in lowercase when they should be capitalized. For example, defining “Representative” to mean only company oficers and legal counsel, rather than the everyday meaning of anyone acting on another’s behalf. If the confidentiality section of an agreement then says that a party can disclose confidential information to “representatives” in lowercase, a counterparty may treat the inconsistency as intentional and apply the broader, everyday definition. The result is that the other party could be permitted to share sensitive information with a much wider group of people than was intended.

Incorrectly Capitalized Terms in Context Terms that have a definition but are capitalized in a context where they are not being used as a defined term. For example, an agreement provides: “ ‘Services’ shall mean the software implementation services provided by Vendor to Customer as described in Exhibit A.” The agreement later provides that: “Vendor shall not provide similar Services to any competitor of Customer.” This matters because the later capitalized term can change the scope of a covenant, restriction, exclusion, or permission. Here, the later erroneous capitalization improperly restricts what should be “services” to the implementation services in Exhibit A whereas a drafter could have intended a broader scope of similar services generally provided by the Vendor.

Unused Defined Terms Defined terms that appear only once in the context where they are defined and are not used elsewhere in the agreement. For example, an agreement provides: “‘Change Order’ means additional or diferent specifications from the project terms set out in Scope of Construction.” However, the term “Change Order” never appears again in the agreement. Because courts aim to avoid any interpretation of contractual language that renders it surplusage, disputes over intent and scope of additional or diferent specifications can arise if a party attributes meaning to the stranded definition and can invite an opportunity to seek the admission of extrinsic evidence to provide competing interpretations of the agreement.

Terms Defined Multiple Times Terms that are defined more than one time in an agreement with conflicting or inconsistent definitions. A term that is merely repeated with the same meaning is NOT an error — only flag a term when its multiple definitions genuinely conflict. For example, if the Efective Date is defined in two places with conflicting dates in an agreement, the parties may disagree about which one controls. Consider a company hired to manage a property that is liable for any accidents occurring while the agreement is in efect. If the agreement defines the Efective Date as both March 1 and March 30, and an accident occurs on March 15, it becomes genuinely unclear who is responsible for that incident, the property owner or the management company. What should be a straightforward question can turn into a costly dispute.

Incorrect Section, Article, or Paragraph References Internal cross-references that assign an incorrect section, article, or paragraph of the agreement. Agreements are often reorganized during negotiation, with sections added, removed, or reordered. If an internal reference isn’t updated to reflect those changes, it may end up pointing to the wrong section entirely. For example, suppose an agreement states that a party will face enhanced damages for breaches of “Section 3,” which covers confidentiality. During negotiation, the sections are reshufled and Section 3 now covers product liability instead, but the reference is never updated. That party is now potentially exposed to enhanced damages for product liability incidents, a potentially broader and more expensive risk.

Incorrect Party References Instances where a party is referred to by the wrong party name or role, such as “Licensor” instead of “Licensee” or “Receiving Party” instead of “Disclosing Party”. Referencing the wrong party name in a contract can shift responsibilities in ways that a party may not want. If a clause requires a specific action, such as paying for shipping, but names the wrong party, the obligation could fall on that party regardless of what was originally discussed during contract negotiations. This type of error can end up costing a party time and money and potentially lead to a dispute over who is actually responsible.

Inconsistent Language Language in the agreement that directly contradicts itself or other language elsewhere in the agreement. When two provisions in a contract directly contradict each other, it creates uncertainty about which one actually applies. For example, if one section states that payment is due 30 days after receiving an invoice, but another states 45 days, neither party can be fully confident about when payment is expected. For the seller, this kind of ambiguity can complicate cash flow planning; they may be counting on payment at 30 days, while the buyer believes they have until 45. What starts as a drafting oversight can quickly become a source of friction or dispute.

Recall by category, with and without reasoning  
![](images/2798198b690723aed9ef5b1cc20475ed4a98d65029edccb509d87dee524c73f2.jpg)  
Figure 4 Recall by categories for two models with and without reasoning. Solid lines show per-section recall across nine defined-term and reference-checking tasks; dashed horizontal lines indicate each condition’s mean.

## B Reasoning Ablation

As some of the tasks involved in contract scrubbing may appear not to require heavy thinking, we tested whether turning of reasoning impacts model performance for the top models. The scores with and withou reasoning are shown in Figure 4.

Across the two models tested, enabling reasoning markedly improved performance for Claude Opus 4.7 (+0.090 avg. recall) and GPT-5.5 (+0.107). The efect was particularly pronounced for Unused Defined Term category and Uncapitlized Defined Term categories. Although these categories rely on lexical signals rather than deep legal interpretation, verifying them requires holistic reasoning over the full document rather than local pattern-matching, which likely explains why reasoning-enabled variants benefit more. By contrast, categories that depend on inferring intent, for example, Incorrect Party References, saw little to no improvement from reasoning, suggesting that current reasoning traces help with structured cross-referencing more than with the deeper semantic judgments these categories demand in this task. For this reason, we report the main results with reasoning enabled.

## C Term-only Scoring Ablation

To distinguish between failure to identify issues in each contract and failure to provide their locations, we conducted a scoring ablation. Rather than multiset comparison across tuples from $R _ { i }$ and ${ \hat { R } } _ { i } ,$ we limit the comparison to the categories, $\kappa ,$ and the elements of f which are terms, allowing mismatches in the locations.

By construction, scores matching only on the terms are higher than the scores requiring both term and location matches (Table 5). The diferences are mostly on the order of five percentage points, though Qwen 3.5 sees much larger improvements and actually passes Claude 4.5 Haiku, indicating that it struggles more with location references within the document than the other models.

Table 5 Recall score comparison across models and scrub categories. Color heatmap applied to all metric rows. Recall Overall shown without heatmap for reference. All models use reasoning variants.
<table><tr><td rowspan="2">Metric</td><td colspan="8">Model</td></tr><tr><td>GPT-5.5</td><td>Gemini 3.1 Pro</td><td>Claude Sonnet 4.6 ‡</td><td>Gemini 2.5 Pro</td><td>Claude GPT-5.2 Opus 4.7</td><td>Claude Haiku 4.5</td><td>Qwen3.5 (397B)</td><td>o4-mini</td></tr><tr><td>Overall Recall</td><td>0.799</td><td>0.793</td><td>0.746</td><td>0.698</td><td>0.661</td><td>0.631</td><td>0.478 0.529</td><td>0.466</td></tr><tr><td>Defined Terms</td><td>0.940</td><td>0.941</td><td>0.919</td><td>0.944</td><td>0.890 0.933</td><td>0.809</td><td>0.877</td><td>0.791</td></tr><tr><td>Undef. Capitalized Terms</td><td>0.552</td><td>0.460</td><td>0.392</td><td>0.347</td><td>0.505 0.582</td><td>0.165</td><td>0.253</td><td>0.205</td></tr><tr><td>Uncapitalized Defined Terms</td><td>0.924</td><td>0.915</td><td>0.776</td><td>0.776</td><td>0.656</td><td>0.707 0.369</td><td>0.533</td><td>0.297</td></tr><tr><td>Incorr. Capitalized in Context</td><td>0.798</td><td>0.736</td><td>0.620</td><td>0.395</td><td>0.519</td><td>0.481 0.217</td><td>0.519</td><td>0.140</td></tr><tr><td>Unused Defined Terms</td><td>0.960</td><td>0.955</td><td>0.941</td><td>0.837</td><td>0.906</td><td>0.876 0.629</td><td>0.540</td><td>0.772</td></tr><tr><td>Terms Defined Multiple Times</td><td>0.794</td><td>0.845</td><td>0.866</td><td>0.794</td><td>0.794</td><td>0.753 0.588</td><td>0.680</td><td>0.608</td></tr><tr><td>Incorr. Sec./Art./Para. Refs</td><td>0.793</td><td>0.840</td><td>0.820</td><td>0.813</td><td>0.807</td><td>0.713 0.727</td><td>0.573</td><td>0.580</td></tr><tr><td>Incorrect Party References</td><td>0.631</td><td>0.654</td><td>0.631</td><td>0.677</td><td>0.208</td><td>0.008 0.323</td><td>0.254</td><td>0.338</td></tr></table>

## D Per-category Precision & F1

We provide per-categoy precision and F1 performance of all models we tested for completeness.

## D.1 Per-category Precision

Table 6 Precision score comparison across models and scrub categories. Color heatmap applied to all metric rows. Precision Overall shown without heatmap for reference.
<table><tr><td rowspan=1 colspan=10>ModelMetric                                                                                                     Qwen3.5Gemini    Claude    Gemini  Claude              ClaudeGPT-5.53.1 Pro               2.5 Pro            GPT-5.2Sonnet 4.6 $             (397B) o4-miniOpus 4.7Haiku 4.5Overall Precision              0.580   0.616     0.620     0.527   0.644    0.540    0.592    0.268    0.548</td></tr><tr><td rowspan=1 colspan=1>Defined Terms</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.916</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.575</td><td rowspan=1 colspan=1>0.906</td></tr><tr><td rowspan=1 colspan=1>Undef. Capitalized Terms</td><td rowspan=1 colspan=1>0.632</td><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=1>0.615</td><td rowspan=1 colspan=1>0.570</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.185</td><td rowspan=1 colspan=1>0.608</td></tr><tr><td rowspan=1 colspan=1>Uncapitalized Defined Terms</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=1>0.299</td><td rowspan=1 colspan=1>0.408</td><td rowspan=1 colspan=1>0.325</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.375</td><td rowspan=1 colspan=1>0.406</td><td rowspan=1 colspan=1>0.073</td><td rowspan=1 colspan=1>0.369</td></tr><tr><td rowspan=1 colspan=1>Incorr. Capitalized in Context</td><td rowspan=1 colspan=1>0.490</td><td rowspan=1 colspan=1>0.556</td><td rowspan=1 colspan=1>0.523</td><td rowspan=1 colspan=1>0.354</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.470</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.232</td></tr><tr><td rowspan=1 colspan=1>Unused Defined Terms</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.640</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.665</td><td rowspan=1 colspan=1>0.359</td><td rowspan=1 colspan=1>0.582</td></tr><tr><td rowspan=1 colspan=1>Terms Defined Multiple Times</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>0.871</td><td rowspan=1 colspan=1>0.816</td><td rowspan=1 colspan=1>0.673</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.765</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.463</td><td rowspan=1 colspan=1>0.797</td></tr><tr><td rowspan=1 colspan=1>Incorr. Sec./Art./Para. Refs</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.611</td><td rowspan=1 colspan=1>0.829</td><td rowspan=1 colspan=1>0.725</td><td rowspan=1 colspan=1>0.786</td><td rowspan=1 colspan=1>0.362</td><td rowspan=1 colspan=1>0.734</td></tr><tr><td rowspan=1 colspan=1>Incorrect Party References</td><td rowspan=1 colspan=1>0.376</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.443</td><td rowspan=1 colspan=1>0.228</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.423</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.377</td></tr><tr><td rowspan=1 colspan=1>Inconsistent Language</td><td rowspan=1 colspan=1>0.246</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>0.312</td><td rowspan=1 colspan=1>0.314</td><td rowspan=1 colspan=1>0.393</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.327</td></tr></table>

## D.2 Per-category F1

Table 7 F1 score comparison across models and scrub categories. Color heatmap applied to all metric rows. F1 Overall shown without heatmap for reference.
<table><tr><td rowspan=1 colspan=10>ModelMetric                                              Claude    Gemini  Claude              Claude  Qwen3.5GeminiGPT-5.53.1 Pro                                    GPT-5.2 Haiku 4.5 (397B) o4-miniSonnet 4.6 $2.5 ProOpus 4.7</td></tr><tr><td rowspan=1 colspan=10>Overall F1                    0.632   0.655     0.637     0.557   0.621    0.553    0.492    0.316   0.453</td></tr><tr><td rowspan=1 colspan=1>Defined Terms</td><td rowspan=1 colspan=1>0.891</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.861</td><td rowspan=1 colspan=1>0.873</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.816</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.809</td></tr><tr><td rowspan=1 colspan=1>Undef. Capitalized Terms</td><td rowspan=1 colspan=1>0.567</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>0.465</td><td rowspan=1 colspan=1>0.408</td><td rowspan=1 colspan=1>0.518</td><td rowspan=1 colspan=1>0.556</td><td rowspan=1 colspan=1>0.229</td><td rowspan=1 colspan=1>0.199</td><td rowspan=1 colspan=1>0.267</td></tr><tr><td rowspan=1 colspan=1>Uncapitalized Defined Terms</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1>0.517</td><td rowspan=1 colspan=1>0.436</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.470</td><td rowspan=1 colspan=1>0.349</td><td rowspan=1 colspan=1>0.119</td><td rowspan=1 colspan=1>0.281</td></tr><tr><td rowspan=1 colspan=1>Incorr. Capitalized in Context</td><td rowspan=1 colspan=1>0.591</td><td rowspan=1 colspan=1>0.616</td><td rowspan=1 colspan=1>0.525</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.571</td><td rowspan=1 colspan=1>0.443</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>0.036</td><td rowspan=1 colspan=1>0.141</td></tr><tr><td rowspan=1 colspan=1>Unused Defined Terms</td><td rowspan=1 colspan=1>0.831</td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>0.767</td><td rowspan=1 colspan=1>0.704</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>0.630</td><td rowspan=1 colspan=1>0.413</td><td rowspan=1 colspan=1>0.637</td></tr><tr><td rowspan=1 colspan=1>Terms Defined Multiple Times</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.697</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.714</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.634</td></tr><tr><td rowspan=1 colspan=1>Incorr. Sec./Art./Para. Refs</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>0.618</td></tr><tr><td rowspan=1 colspan=1>Incorrect Party References</td><td rowspan=1 colspan=1>0.451</td><td rowspan=1 colspan=1>0.558</td><td rowspan=1 colspan=1>0.513</td><td rowspan=1 colspan=1>0.498</td><td rowspan=1 colspan=1>0.213</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1>0.361</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>0.339</td></tr><tr><td rowspan=1 colspan=1>Inconsistent Language</td><td rowspan=1 colspan=1>0.366</td><td rowspan=1 colspan=1>0.443</td><td rowspan=1 colspan=1>0.494</td><td rowspan=1 colspan=1>0.386</td><td rowspan=1 colspan=1>0.405</td><td rowspan=1 colspan=2>0.426     0.442</td><td rowspan=1 colspan=1>0.286</td><td rowspan=1 colspan=1>0.351</td></tr></table>

## PART 2: Reproducibility Details

## E Data Access and Usage Notes

The dataset will be made publicly available upon acceptance. The CUAD dataset is used under a CC-BY-4.0 license.

## F Inference Details

We evaluated 10 models on the contract scrub benchmark. Details regarding hyperparameters and compute resources are listed in Table 8. Empty cells indicate that an option is not available for a particular model.

## Table 8 Model Inference Details

<table><tr><td>Name</td><td>Provider</td><td>Temperature</td><td>Reasoning Effort</td></tr><tr><td>Claude Opus 4.7</td><td>Bedrock</td><td>一</td><td>High</td></tr><tr><td>Claude Sonnet 4.6</td><td>Bedrock</td><td>0.6</td><td>High</td></tr><tr><td>Claude Haiku 4.5</td><td>Bedrock</td><td>0.6</td><td>High</td></tr><tr><td>GPT 5.5</td><td>OpenAI</td><td>一</td><td>Medium</td></tr><tr><td>GPT 5.2</td><td>OpenAI</td><td></td><td>Medium</td></tr><tr><td>04 mini</td><td>Azure</td><td>0.6</td><td>一</td></tr><tr><td>Qwen 3.5 397B A17B</td><td>Self-hosted</td><td>0.6</td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>Vertex AI</td><td>0.6</td><td></td></tr><tr><td>Gemini 2.5 Pro</td><td>Vertex AI</td><td>0.6</td><td></td></tr><tr><td>Gemma 4 26B A4B</td><td>Vertex AI</td><td>0.6</td><td></td></tr></table>

## G Prompt Templates

We provide the full prompt text used to test each model in the main results. The prompts for each of the nine categories were built from an overall prompt template, with a targeted section inserted for each category. Each category has four items, including ‘title’, ‘definition’, ‘instruction’ and ‘schema’.

```yaml
Overall Scoring Prompt Template
You are a legal editor reviewing an agreement for drafting errors and issues.
Review the agreement below and identify ONLY the following category:
{title} - {definition}
{instructions}
If you find no items in this category, respond with an empty list.
Follow these rules for the location format. This is critical:
- Always give the most specific sub-section reference, NEVER just the parent section number.
- Concatenate sub-section letters/numerals together; preserve dots between major.minor section
numbers.
- If a defined term appears in section 1(a), the location is "1a", NOT "1".
- If something is in section 11(b)(i), the location is "11bi", NOT "11" or "11b".
- If something is in section 7(a)(i), the location is "7ai", NOT "7" or "7a".
- If something is in Attachment X Section Y, the location is "attachment xy", NOT "attach
ment x.y".
- For nested numbering: section 1.1(d) is "1.1d", section 3.7(a) is "3.7a", section 1.1(h)(vii) is
"1.1hvii".
- Use "P" for the Preamble (the introductory section that identifies the parties, date and gen
eral background).
- Use "Recitals" for the WHEREAS clauses; if the item is in a specific recital clause, append its
letter/number, e.g. "Recitals c" for recital (c).
- Use "Exhibit A", "Exhibit B" and "Schedule A", "Schedule B" etc. for exhibits and sched
ules.
- Use "Signature Block" for the section where parties execute the agreement.
Respond with JSON only - no explanation, no markdown fences. Use this exact schema:
{schema}
```

<table><tr><td>Defined Terms</td></tr><tr><td>title: Defined Terms</td></tr><tr><td>definition: Words or phrases that are formally defined in the agreement with a designated meaning within</td></tr><tr><td>the context of the agreement, often set out as a word or words that are capitalized, underlined, bolded,</td></tr><tr><td>italicized, or in quotations.</td></tr><tr><td></td></tr><tr><td>instruction: For each term, provide the section where it is defined.</td></tr></table>

schema: {"defined\_terms": [{"term": "...", "location": "..."}]}

Undefined Capitalized Terms

title: Undefined Capitalized Terms

definition: Terms that are capitalized or otherwise treated as a defined term in an agreement but not formally defined. For example, an agreement provides: “Party A shall comply with all applicable requirements in the Approved Specification,” but Approved Specification is not defined. This matters because an undefined capitalized term indicates that a contracting party specifically intended a special contractual meaning. As a result, an ambiguity is created and a party may later dispute what specification was approved and whether it was applicable, whether a breach occurred, and whether outside evidence is admissible to contradict, vary, or add to the terms of the agreement.

1. Terms of art — words or terms with a specific, precise, technical, and specialized meaning commonly understood and used within a particular field, profession, or discipline that is germane or applicable to the contract.

2. Proper nouns used in common parlance — names of specific people, places, brands, or things that have been adopted into everyday, informal speech and are widely understood by the general population, even outside of their original specialized or formal context.

3. Morphological variants of defined terms — instances where a defined term appears in a grammatically inflected form that is not itself defined.

4. Scientific and technical nomenclature — standardized words or terms with a specific, precise, and specialized meaning commonly understood and used in scientific and technical fields.

5. References to agreement sections, exhibits, and schedules.

6. The title of the agreement and the names of any associated documents as they appear in the agreement’s header or title section.

7. Titles, roles, and positions.

instruction: List each occurrence separately with the section where each term is located.

schema: {"undefined\_capitalized\_terms": [{"term": "...", "location": "..."}]}

Uncapitalized Defined Terms

title: Uncapitalized Defined Terms

definition: Formally defined terms appearing in lowercase when they should be capitalized. For example, defining “Representative” to mean only company oficers and legal counsel, rather than the everyday meaning of anyone acting on another’s behalf. If the confidentiality section of an agreement then says that a party can disclose confidential information to “representatives” in lowercase, a counterparty may treat the inconsistency as intentional and apply the broader, everyday definition. The result is that the other party could be permitted to share sensitive information with a much wider group of people than was intended.

instruction: List each occurrence separately with the section where each term is located.

schema: {"uncapitalized\_defined\_terms": [{"term": "...", "location": "..."}]}

Incorrectly Capitalized Terms In Context

title: Incorrectly Capitalized Terms In Context

definition: Terms that have a definition but are capitalized in a context where they are not being used as a defined term. For example, an agreement provides: “‘Services’ shall mean the software implementation services provided by Vendor to Customer as described in Exhibit A.” The agreement later provides that: “Vendor shall not provide similar Services to any competitor of Customer.” This

matters because the later capitalized term can change the scope of a covenant, restriction, exclusion, or permission. Here, the later erroneous capitalization improperly restricts what should be “services” to the implementation services in Exhibit A whereas a drafter could have intended a broader scope of similar services generally provided by the Vendor.

instruction: List each occurrence separately with the section where each term is located.

schema: {"incorrectly\_capitalized\_terms\_in\_context": [{"term": "...", "location": "..."}]}

Unused Defined Terms

title: Unused Defined Terms

definition: Defined terms that appear only once in the context where they are defined and are not used elsewhere in the agreement. For example, an agreement provides: “‘Change Order’ means additional or diferent specifications from the project terms set out in Scope of Construction.” However, the term “Change Order” never appears again in the agreement. Because courts aim to avoid any interpretation of contractual language that renders it surplusage, disputes over intent and scope of additional or diferent specifications can arise if a party attributes meaning to the stranded definition and can invite an opportunity to seek the admission of extrinsic evidence to provide competing interpretations of the agreement.

instruction: List each occurrence separately with the section where each term is located.

schema: {"unused\_defined\_terms": [{"term": "...", "location": "..."}]}

Terms Defined Multiple Times

title: Terms Defined Multiple Times

definition: Terms that are defined more than one time in an agreement with conflicting or inconsistent definitions. A term that is merely repeated with the same meaning is NOT an error — only flag a term when its multiple definitions genuinely conflict. For example, if the Efective Date is defined in two places with conflicting dates in an agreement, the parties may disagree about which one controls. Consider a company hired to manage a property that is liable for any accidents occurring while the agreement is in efect. If the agreement defines the Efective Date as both March 1 and March 30, and an accident occurs on March 15, it becomes genuinely unclear who is responsible for that incident, the property owner or the management company. What should be a straightforward question can turn into a costly dispute.

instruction: List each occurrence separately along with each of the sections where the multiple definitions are located.

schema: {"terms\_defined\_multiple\_times": [{"term": "...", "location1": "...", "location2": "..."}]}

Incorrect Section, Article, or Paragraph References

title: Incorrect Section, Article, or Paragraph References

definition: Internal cross-references that assign an incorrect section, article, or paragraph of the agreement. Agreements are often reorganized during negotiation, with sections added, removed, or reordered. If an internal reference isn’t updated to reflect those changes, it may end up pointing to the wrong section entirely. For example, suppose an agreement states that a party will face enhanced damages for breaches of “Section 3,” which covers confidentiality. During negotiation, the sections are reshufled and Section 3 now covers product liability instead, but the reference is never updated. That party is now potentially exposed to enhanced damages for product liability incidents, a potentially broader and more expensive risk.

instruction: For each, list the incorrect reference used, the correct reference, and the section where the error is located. Use only the reference identifier (e.g. “9.1”), without descriptive section names. List each reference as a SEPARATE item: if one cross-reference covers several sections, report each pair on its own (e.g. “9.1 or 9.2 → 10.1 or 10.2” becomes two items, 9.1 → 10.1 and 9.2 → 10.2), and if the same error recurs in multiple places, report each location separately.

schema: {"incorrect\_section\_article\_paragraph\_references": [{"wrong": "correct": 11 II ... "location": "..."}]}

## Incorrect Party References

title: Incorrect Party References

definition: Instances where a party is referred to by the wrong party name or role, such as “Licensor” instead of “Licensee” or “Receiving Party” instead of “Disclosing Party”. Referencing the wrong party name in a contract can shift responsibilities in ways that a party may not want. If a clause requires a specific action, such as paying for shipping, but names the wrong party, the obligation could fall on that party regardless of what was originally discussed during contract negotiations. This type of error can end up costing a party time and money and potentially lead to a dispute over who is actually responsible.

instruction: List each item identifying the incorrect party reference, the correct party reference, and the section where the error is located.

schema: {"incorrect\_party\_references": [{"wrong": "...", "correct": "...", "location": "..."}]}

## Inconsistent Language

title: Inconsistent Language

definition: Language in the agreement that directly contradicts itself or other language elsewhere in the agreement.

When two provisions in a contract directly contradict each other, it creates uncertainty about which one actually applies. For example, if one section states that payment is due 30 days after receiving an invoice, but another states 45 days, neither party can be fully confident about when payment is expected. For the seller, this kind of ambiguity can complicate cash flow planning; they may be counting on payment at 30 days, while the buyer believes they have until 45. What starts as a drafting oversight can quickly become a source of friction or dispute.

instruction: For each occurrence, identify both sections where the conflicting language appears.

schema: {"inconsistent\_terms": [{"section1": "...", "section2": "..."}]}