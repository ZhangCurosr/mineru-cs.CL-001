# When Readability and Source Retention Diverge: An Evaluability Gap in AI Translation

CHENCHEN MAO, Lehigh University, United States

HANJING SHI, Lehigh University, United States

HAIYAN JIA, Lehigh University, United States

EMILY WEGRZYN, Lehigh University, United States

DOMINIC DIFRANZO, Lehigh University, United States

Readable AI output can leave an evaluability gap: even when the source is shown, an overall-quality judgment may not reflect what an output preserves. We investigated how source-text condition and output rendering relate to perceived translation quality, and how output and system appraisals relate to trust and stated disclosure willingness in a plain-text interface. A focal 2×2 comparison (� = 306) using TransLingo examined simple generated narratives and complex literary-philosophical prose alongside LLM-generated readability-oriented outputs and researcher-revised fidelity-oriented outputs. A descriptive stimulus audit indicated greater source retention in fidelity-oriented outputs in both source-text conditions. Factorial analyses showed a significant rendering-by-source-textcondition interaction in perceived quality. Participants rated fidelity-oriented outputs higher than readability-oriented outputs for the simple narratives, whereas no reliable rendering diference emerged for the complex prose. A corresponding source-conditiondependent pattern was observed for perceived intelligence, agency-oriented anthropomorphic attribution, and task-performance trust. A separate theory-ordered appraisal-structure SEM characterized concurrent associations among perceived quality, perceived intelligence, agency-oriented anthropomorphic attribution, task-performance trust, and stated disclosure willingness across six domains, with task-performance trust as the proximal correlate of stated willingness. The observed rating pattern distinguishes source access from source evaluability: for the complex stimuli, displaying the source did not ensure that one overall-quality rating reflected diferences in retained content. It also separates support for evaluating translation output from data-handling support for decisions about what personal text to entrust to a system.

CCS Concepts: • Do Not Use This Code → Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper.

## ACM Reference Format:

Chenchen Mao, Hanjing Shi, Haiyan Jia, Emily Wegrzyn, and Dominic DiFranzo. 2018. When Readability and Source Retention Diverge: An Evaluability Gap in AI Translation. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 24 pages. https://doi.org/XXXXXXX.XXXXXXX

Authors’ Contact Information: Chenchen Mao, chm321@lehigh.edu, Lehigh University, Bethlehem, Pennsylvania, United States; Hanjing Shi, hasa23@ lehigh.edu, Lehigh University, Bethlehem, Pennsylvania, United States; Haiyan Jia, haj616@lehigh.edu, Lehigh University, Bethlehem, Pennsylvania, United States; Emily Wegrzyn, emw224@lehigh.edu, Lehigh University, Bethlehem, Pennsylvania, United States; Dominic DiFranzo, djd219@lehigh.edu, Lehigh University, Bethlehem, Pennsylvania, United States.

## 1 Introduction

Machine translation is widely used across many domains, including medicine and legal services [49], online chat [57] and the workplace [5]. Translation interfaces are often treated as utilities: a user enters text, receives output, and moves on. Seen this way, trusting a translation system means little more than judging whether the output is usable. But the text that users submit to these systems is often personal — private messages, workplace requests, or material about their own circumstances — and often entered at moments when wording matters and verification is dificult [49]. When the input is about the user themselves, entering it is itself an act of disclosure, and the system becomes not merely an information source but a disclosure interface. Trust then becomes a question of what users are willing to entrust to a system that does not merely transmit their text but may reshape it. This view is reflected in AI-mediated communication research, which treats translation as a case in which a computational agent modifies messages toward a communication goal [17]. For HCI, AI translation is therefore both an output-evaluation problem and an interface through which users entrust information to a computational mediator.

LLM-based translation makes this reformulation especially important. Early machine translation often produced disfluent, awkward output, but contemporary LLMs often generate highly fluent text [2]. Fluency alone may therefore no longer reliably distinguish a good rendering from a poor one. What still varies, and what a prompt can shift substantially, is readability: an LLM can be asked to make a text easier to read [37], but readability gains can come at the cost of faithfulness. Evidence of this tension has emerged across tasks in which LLMs restate source content. In scientific summarization, LLM-generated summaries may read clearly while overgeneralizing the original claims [38]. In question answering, fluent answers can appear more trustworthy even when they are not faithful to the supporting evidence [9]. Similarly, when LLMs are asked to simplify text to a lower reading level, greater degrees of simplification make it more dificult to preserve the original meaning fully [4]. Although these findings come from diferent tasks, they point to a common concern for LLM-based translation: readability-oriented generation may weaken the model’s adherence to the source. We call the resulting possibility an evaluability gap: users may see the source without being able to recognize, through one overall-quality judgment, what the output retained or altered.

Whether users register the divergence between readability and faithfulness, however, may depend on the source text itself. We therefore treat diferences between the tested source-text conditions as an open research question (RQ1b).

Because translation can involve personal text, how users perceive the system may matter alongside how they judge its output. Our setting is a plain-text translation interface without an avatar, voice, persona, or multi-turn conversation. We therefore examine whether users’ output evaluations are associated with anthropomorphic or agentic perceptions of the system.

One such system appraisal is perceived intelligence. We therefore examine whether perceived quality is positively associated with perceived intelligence [22], and whether perceived intelligence is, in turn, positively associated with perceived anthropomorphism.

The tension between readability and faithfulness, and the possible role of source-text complexity, motivates our overarching question:

When translation output is more readable but retains less source content—in an interface with no overt anthropomorphic design cues—how do users evaluate its quality, does the source-text dependence of that evaluation extend to their appraisals of the system itself, and how are those evaluations connected to perceptions of the system, trust, and disclosure willingness?

We address this question in a focal 2×2 comparison of source-text condition (simple versus complex) and prepared output rendering: readability-oriented output that prioritizes readability and fidelity-oriented output revised toward source fidelity. We first compare perceived quality across the output renderings and ask whether that relationship difers between the source-text conditions (RQ1a, RQ1b). In exploratory analyses, we then ask whether the same rendering-by-source-text pattern appears in users’ appraisals of the system itself — perceived intelligence, agencyoriented anthropomorphism, and task-performance trust (RQ2). We then estimate a theoretically ordered pattern of associations among the concurrently measured post-interaction appraisals: perceived quality is associated with perceived intelligence; perceived intelligence is associated with perceived anthropomorphism; perceived intelligence and perceived anthropomorphism are associated with trust; and trust is associated with willingness to disclose across six personal-information domains (H1–H5; Figure 1).

This study makes three contributions:

(1) It distinguishes source visibility from source evaluability. Although every participant saw the source and output side by side, the greater source retention documented by the stimulus audit was reflected in overall-quality ratings only for the simple sources. Factorial analyses identified this source-dependent rendering pattern in perceived quality and showed that it extended to perceived intelligence, agency-oriented anthropomorphism, and task-performance trust. No reliable rendering contrast emerged for stated disclosure willingness in either source-text condition.

(2) It provides a theory-ordered structural account of the appraisal system surrounding translation output. A complementary appraisal-structure SEM described concurrent associations among perceived quality, perceived intelligence, agency-oriented anthropomorphism, trust, and disclosure willingness. Trust was positively associated with disclosure willingness in every domain. Adding direct anthropomorphism-to-disclosure paths did not improve model fit, supporting the more parsimonious specification in which trust was the appraisal most proximally associated with stated disclosure willingness.

(3) It separates output evaluation from information entrustment. Trust was positively associated with stated disclosure willingness across all six domains $( \beta = . 3 6 3 - . 5 6 6 .$ , all FDR-adjusted $\textstyle p \ < \ . 0 0 1 )$ . In contrast, although the rendering conditions difered on four appraisal measures for the simple sources, they did not difer reliably in stated disclosure willingness. The model-implied associations between rendering and disclosure were correspondingly small for the simple sources $( \beta = . 0 5 0 \mathrm { - } . 0 7 8 )$ and near zero for the complex sources. This pattern motivates treating support for evaluating an output and support for deciding what personal text to submit as distinct interface-design problems.

These findings expose a limit of source-visible translation interfaces: placing source evidence on screen did not ensure that users’ holistic judgments reflected it. Source-linked change information and explicit data-handling information address diferent decisions and require separate evaluation.

## 2 Related Work and Hypothesis Development

## 2.1 When Readability and Source Retention Diverge

Translation quality is commonly judged through a combination of source adequacy—how faithfully the output preserves the meaning of the source—and target-language fluency [16, 51]. Martindale and Carpuat [28] found that users reacted strongly to disfluent MT but were much less sensitive to fluent output that was not fully adequate. As fluent LLM output

Manuscript submitted to ACM

![](images/0a861300d67e4ad258688a08d5bbf28d7619bb4a86a6bc3cfc2c504e69a28017.jpg)  
Fig. 1. Hypothesized model. Output rendering (readability-oriented vs. fidelity-oriented) is compared on perceived quality (RQ1a), with the rendering–quality relationship examined across the two source-text conditions (RQ1b). Perceived quality is expected to be positively associated with perceived intelligence (H1); perceived intelligence with perceived anthropomorphism (H2) and trust (H4); perceived anthropomorphism with trust (H3); and trust with disclosure willingness across six domains (H5a–H5f): tastes and interests, atitudes and opinions, work or studies, economic and social status, interpersonal relations and self-concept, and physical appearance and sex. Paths among the post-task appraisals represent hypothesized associations rather than temporal or causal efects.

becomes common [2], fluency may become less diagnostic among outputs that already read smoothly. Adequacy then remains an important source of quality diferences that users may not readily detect.

This is also where adequacy and readability can come apart. An LLM can be prompted to enhance the readability of a text, making it more accessible to particular readers such as students [37], but rendering a dificult source into simple language can diverge from faithfulness to that source. This tension is not specific to translation; it appears wherever an LLM restates source content for a reader. In scientific summarization, Peters and Chin-Yee [38] show tha LLM-generated summaries can read as clear and accessible while overgeneralizing the claims of the original papers. Similarly, in conversational question answering, Chiesurin et al. [9] find that LLMs can produce fluent, user-aligned answers that raise perceived trustworthiness even when those answers are not faithful to the underlying evidence. These findings raise the possibility of a trade-of between readability and source retention in LLM-generated output.

How users perceive the quality of output that diverges in readability and adequacy in this way, however, is not theoretically settled. Two accounts pull in opposite directions. Research on processing fluency shows that text which is easier to process is judged as more credible and of higher quality, independently of its actual content [1, 39] — by this account, a readability-oriented rendering should be perceived as higher in quality. By a competing account, adequacy — how faithfully the output preserves the meaning of the source — has long been recognized as a core dimension of translation quality [16, 51]; when users can compare the output against the source, a rendering that preserves more of its meaning may be recognized as more adequate, and therefore as higher in quality. By this account, a fidelity-oriented rendering should be perceived as higher in quality. Because these accounts predict opposite efects, we leave the relationship between output rendering and perceived quality as our first open research question:

RQ1a: How do perceived-quality ratings difer between readability-oriented and fidelity-oriented outputs?

The relationship between output rendering and perceived translation quality may difer across source-text conditions. Complex texts require more background knowledge and cognitive efort to understand [7, 52]. Under such conditions, users may rely more heavily on processing fluency—the subjective ease with which information is understood—as a cue when evaluating quality [1]. For example, greater readability has been shown to increase readers’ confidence in Manuscript submitted to ACM

financial disclosures through experienced processing fluency [41]. Applied to machine translation, this account suggests that a readability-oriented rendering may be evaluated more favorably when the source text is dificult to process.

Source-text complexity may also make fidelity more dificult to evaluate. When users struggle to understand the source itself, they may be less able to determine whether a translation accurately preserves its meaning. The relative advantage of a more faithful rendering may therefore become less apparent as source-text complexity increases. Both mechanisms imply that complexity may increase the importance of readability while making fidelity harder to recognize. However, it remains unclear whether either mechanism is strong enough to alter users’ overall quality judgments, and our study does not measure the mechanisms separately. We therefore examine whether the rendering–quality relationship difers across the tested source-text conditions:

RQ1b: Does the relationship between output rendering and perceived translation quality difer across the tested source-text conditions?

## 2.2 From Perceived Quality to Perceived Intelligence

Beyond how users judge the translation itself, output quality may also relate to how they perceive the system behind it. In this study, we adopt the definition by Moussawi and Koufaris [32], characterizing perceived intelligence as the extent to which an agent appears eficient, useful, goal-directed, and autonomous, with the specific ability to generate efectual outputs and process natural language. Perceived translation quality concerns users’ evaluation of the specific outputs presented to them, whereas perceived intelligence concerns the broader system capabilities they infer from those outputs.

Prior research suggests that users draw on a system’s observable performance when judging its intelligence. Koda and Maes [22] found that the perceived intelligence of a poker-playing agent depended more on its demonstrated competence than on its facial appearance. Similarly, Zieger et al. [56] experimentally varied the reliability of in-vehicle agents and found that highly reliable agents were perceived as more intelligent than less reliable agents. Consistent with these findings, a review of human–robot interaction research identified successful task performance as one factor commonly associated with higher perceived intelligence [48].

In our setting, the most directly observable aspect of system performance is the perceived quality of its translation output. We therefore expect perceived translation quality to be positively associated with how intelligent the system is perceived to be.

H1: Higher perceived translation quality is positively associated with higher perceived intelligence of the system.

## 2.3 Anthropomorphism: Beyond Designed Social Cues

Judging a system as intelligent is not only an assessment of what it can do; it may also be associated with how the system is categorized—from a tool that processes text to an agent that appears to understand and reshape it. Prior work describes this humanlike interpretation as anthropomorphism, broadly characterized as the attribution of human traits and characteristics to computers and other interactive technologies [34]. In the present study, we focus on an agency-oriented form of anthropomorphism: perceiving a nonhuman system as possessing humanlike capacities to understand, anticipate, and plan [15, 50].

This expectation is consistent with the three-factor theory of anthropomorphism [11], which identifies elicited agent   
knowledge, efectance motivation, and sociality motivation as three determinants of anthropomorphism. Elicited agent   
knowledge is particularly relevant here: it refers to the activation of knowledge ordinarily used to understand human   
agents. When a system appears capable of understanding, anticipating, and planning, users may draw on this knowledge Manuscript submitted to ACM

and interpret the system in more humanlike terms. Perceived intelligence may therefore be positively associated with perceived anthropomorphism even when the interface contains no explicitly designed humanlike cues.

Much prior research has examined anthropomorphism in interfaces containing designed social cues, including humanlike appearance, names, voices, and behavioral features [36, 44, 50]. This includes studies reporting positive associations between perceived intelligence and anthropomorphism [26, 31–33, 47]. Because social cues were also present in these settings, it remains unclear whether the association holds when the system’s textual output is the primary basis for evaluation.

H2: Higher perceived intelligence of the system is positively associated with greater perceived anthropomorphism.

## 2.4 Anthropomorphism, Perceived Intelligence, and Trust

Perceiving a system as more agent-like may also be associated with how far users are willing to rely on it. Experimental work has manipulated anthropomorphic cues and found efects on trust. Waytz et al. [50] varied an autonomous vehicle’s humanlike features and found greater trust when those features were present, while Natarajan and Gombolay [35] identified anthropomorphism as one of the strongest predictors of trust and compliance in a multi-factor robot study. Survey evidence reports a similar positive association [14]. Focus-group research further suggests that consumers regard humanness and intelligence as important bases for judging whether an AI service can be trusted [47].

H3: Greater perceived anthropomorphism is positively associated with higher trust in the system.

Perceived intelligence may also be associated with trust alongside its association with perceived anthropomorphism. In a survey of employee beliefs about AI, Lu et al. [24] found that trust mediated the relationship between perceived intelligence and acceptance of AI use. Chen et al. [8] found that perceived intelligence was positively associated with both cognitive and emotional trust. Focus-group evidence similarly indicates that consumers rely on perceived intelligence and anthropomorphism as key determinants of AI trustworthiness [47]

H4: Higher perceived intelligence is positively associated with higher trust in the system.

## 2.5 Trust and Disclosure Willingness

The related work and hypotheses above concern the appraisals associated with trust in a translation system. The practical significance of trust lies in users’ willingness to provide personal text to the system [18].

Research on privacy decision-making ofers an account of what makes users willing to do so. The extended privacy calculus model of Dinev and Hart [10] conceptualizes disclosure as a decision under uncertainty, in which privacy concerns inhibit online transactions while Internet trust and personal interest can together outweigh perceived privacy risks. In this model, trust is one factor that can make disclosure more likely. Supporting this role, Metzger [30] identified trust as an important antecedent of disclosure to commercial websites, while Krasnova et al. [23] found that trust in a network provider reduced the perceived risk associated with self-disclosure.

These findings come from e-commerce sites and social platforms, but the same decision arises whenever users provide personal text to a system—including a translation interface. We therefore expect trust in the translation system to be positively associated with disclosure willingness, and because that willingness may vary with the topic being disclosed, we examine the association separately across six personal-information domains.

H5a–H5f: Higher trust in the system is associated with greater willingness to self-disclose across each of the six personal-information domains: tastes and interests, attitudes and opinions, work or studies, economic and social status, interpersonal relations and self-concept, and physical appearance and sex.

Manuscript submitted to ACM

Because these appraisals were collected after participants viewed the same prepared outputs, we also ask, as an exploratory question, whether the source-condition dependence examined for perceived quality (RQ1a,RQ1b) extends to the appraisals of the system itself:

RQ2: Do the relationships between output rendering and perceived intelligence, perceived anthropomorphism, and task-performance trust also difer across the tested source-text conditions?

## 3 Methods

## 3.1 Experiment Design

The experiment contained a focal 2 × 2 comparison. The first factor was source-text condition, contrasting simple and complex texts. The second factor was output rendering, comparing a readability-oriented rendering—produced by an LLM and oriented toward making the output easy to read—with a fidelity-oriented rendering of that same output, revised toward closer source fidelity. All outputs were presented as TransLingo translations; the prepared-output contrast changed the text rather than adding a visible social cue.

## 3.2 Selection and Preparation of Source Texts and Translations

3.2.1 Source Texts. To construct the source-text conditions, we varied the source text because its characteristics shape the complexity of the translation task. We based our approach on the work of Wood [52], who argues that task complexity includes both component and coordinative complexity. This concept was recently extended to Human-AI decision-making contexts by Salimzadeh et al. [45]. We operationalized component complexity—the number of distinct information cues to be processed—through vocabulary; simple texts contained common, frequently repeated words (e.g., village, flower, children), while complex texts featured abstract, low-frequency words (e.g., paternal afection, solecism, steadfastness). We addressed coordinative complexity—the nature of relationships between task inputs and outputs— through sentence structure, using short sentences averaging 15 words for simple texts and long sentences averaging 72 words for complex texts. The simple texts were generated for this study using ChatGPT-4 with instructions to create content easily understood by primary school students. In contrast, the complex texts were extracted directly from Aurelius’s Meditations [3]. We assessed participants’ perceptions of this contrast using the study-specific grade-level item described below.

Two source texts were used at each complexity level, four in total. Each participant was presented with both texts at their assigned complexity level and rated each one separately, so every participant contributed two readability judgments and two translation-quality judgments. Because each complexity level is represented by texts drawn from a single provenance, source complexity is confounded with genre, topic, provenance, period style, and length: the simple texts are contemporary LLM-generated narrative prose, whereas the complex texts are drawn from a nineteenth-century English translation of Stoic philosophy [3]. The source-text comparison should therefore be read as a contrast between simple generated narrative and complex literary-philosophical prose rather than as a pure manipulation of linguistic complexity.

To assess the intended source-text contrast, participants rated the level of education required to read and understand the input texts using a measure adapted from the Dale–Chall Readability Formula [6]. The simple texts were rated as accessible to an average seventh- or eighth-grade student, whereas the complex texts were rated as requiring approximately college-level comprehension. A Mann–Whitney � test showed a significant diference between the two conditions, $U = 6 0 7 . 5 0 , Z = - 1 8 . 2 0 , p < . 0 0 1$ , supporting the intended contrast in perceived source-text complexity.

Manuscript submitted to ACM

3.2.2 Output Rendering Condition. Output rendering had two levels: a readability-oriented rendering and a fidelity oriented rendering. For the readability-oriented rendering condition, ChatGPT-4 paraphrased the source text in a way that prioritized making the final English output easy to read, while preserving the general topic of the source. For the fidelity-oriented rendering condition, a researcher revised the readability-oriented output for closer faithfulness to the source — restoring content that had been simplified or omitted, while keeping the output readable.  
To assess whether the two rendering conditions difered in the intended directions on readability and source retention, we conducted a stimulus-level audit on the source–output pairs used in the study. Each pair was scored on four metrics: the change in Flesch reading-ease score from source to output (ΔFlesch), which captures readability gain [21]; contentword Jaccard overlap, which captures how much source content survives in the output; BERTScore F1, which captures semantic similarity to the source [55]; and COMET, a learned metric for translation quality [40].
<table><tr><td>Source</td><td>Rendering</td><td>∆Flesch</td><td>Jaccard</td><td>BERTScore</td><td>COMET</td></tr><tr><td>Simple</td><td>Readability-oriented</td><td>14.08</td><td>0.377</td><td>0.937</td><td>0.869</td></tr><tr><td>Simple</td><td>Fidelity-oriented</td><td>4.83</td><td>0.713</td><td>0.976</td><td>0.914</td></tr><tr><td>Complex</td><td>Readability-oriented</td><td>70.43</td><td>0.138</td><td>0.854</td><td>0.670</td></tr><tr><td>Complex</td><td>Fidelity-oriented</td><td>56.40</td><td>0.452</td><td>0.910</td><td>0.753</td></tr></table>

Table 1. Stimulus-level audit of the prepared-output contrast across the four focal cells. ΔFlesch = change in Flesch reading-ease score from source to output; Jaccard = content-word Jaccard overlap with source; BERTScore = BERTScore F1 against source; COMET = learned MT quality metric. Higher ΔFlesch indicates a larger readability gain; higher Jaccard and BERTScore indicate closer source retention, while higher COMET indicates higher estimated MT quality.

Table 1 summarizes diferences among the prepared outputs. First, the readability-oriented outputs made the source texts easier to read. Their reading-ease gains were larger than those of the fidelity-oriented outputs for both simple texts (14.08 vs. 4.83) and complex texts (70.43 vs. 56.40). Second, compared with the readability-oriented outputs, the fidelity oriented outputs retained a larger proportion of the source content words (Jaccard), had higher semantic similarity to the source (BERTScore), and received higher estimated translation-quality scores (COMET) at both complexity levels. Thus, both conditions produced readable English, but they emphasized diferent goals: the readability-oriented condition simplified the source more aggressively, whereas the fidelity-oriented condition retained more of its original content. These metrics describe relative diferences among the stimuli rather than independently validated translation adequacy.

## 3.3 Experiment Procedure

Participants first completed a pre-survey reporting their demographics, first language, and other languages spoken; language proficiency was not assessed using a standardized test. They were then introduced to TransLingo, a prototype AI-mediated translation system. Data were collected in two separate periods: one evaluated the readability-oriented rendering and the other evaluated the fidelity-oriented rendering. Within each period, participants were randomly assigned to either the simple or complex source-text condition, resulting in a 2 × 2 design defined by rendering and source-text complexity. Each participant evaluated two texts in the assigned complexity condition. Thus, source-tex complexity was randomized within each period, whereas the rendering comparison was conducted between periods, as discussed further in the Limitations.

Manuscript submitted to ACM

Inside TransLingo, participants were told that two test texts had been randomly picked from the internet. Because the sample was English-speaking, the task translated English to Chinese and then back to English, allowing participants to evaluate the final English output.

After reviewing both the source and translated texts in their assigned condition, the post-survey button became active. The post-survey measured perceived input-text complexity, perceived translation quality, perceived intelligence, anthropomorphism, trust in the system, and willingness to disclose personal information while using the system.

3.3.1 Ethical Consideration. The design involved mild deception: participants were told they were testing a working prototype, whereas all input texts and translations had been prepared in advance and held constant within condition. This was necessary to hold stimulus quality fixed rather than allowing it to vary with live model output, and to maintain experimental realism. At the end of the study, participants were debriefed and informed that the materials had been selected or prepared in advance, why this information had been withheld beforehand, and whom to contact with questions or to withdraw. The study procedures received approval from the Institutional Review Board (IRB) at Anonymous University

## 3.4 TransLingo Interface Overview

TransLingo was the web interface used in the experiment. Participants selected the pre-made source text assigned by the interface, clicked the ’translate’ button, and then saw the source text and translated output together.

As illustrated in Figure 2, the interface presented the English source, intermediate Chinese text, and final English output in separate panels.

## 3.5 Variables and Measurements

All post-survey constructs were measured after participants had reviewed both source texts and both translated outputs in their assigned condition.

For the factorial analyses, item responses for perceived intelligence, anthropomorphism, and trust were averaged into composite scores; in the appraisal-structure SEM, the same items were modeled as indicators of their respective latent factors.

3.5.1 Perceived source-text readability (manipulation check). For each of the two source texts, participants answered "What level of education is needed for someone to read and understand this text?" on a 7-point scale adapted from grade-level categories associated with the Dale–Chall Readability Formula [6], ranging from 1 ("easily understood by an average 4th-grade student or lower") to 7 ("easily understood by an average college graduate student and above"). The two ratings were averaged. This study-specific subjective item was used as a manipulation check, not as an objective application of the formula or a validated psychometric scale.

3.5.2 Perceived translation quality. For each of the two texts, participants were shown the source text together with the final English output in their assigned rendering condition and answered: "Please rate the overall quality of the translation compared to the original source text provided above." Responses used a 7-point scale labeled Extremely poor (1), Poor (2), Fair (3), Average (4), Good (5), Very good (6), and Excellent (7). Perceived quality was measured with a single item per text; the two ratings were strongly correlated, $r = . 8 2 6 ,$ and were averaged into an observed composite $( M = 5 . 3 7 , S D = 1$ .18; Spearman–Brown coeficient = .904). Perceived quality entered the structural model as an observed composite rather than as a latent variable because it was a single item administered twice, not a multi-item

![](images/5a4614ac1209d482a6d1945a621624a6ada543e587b8fb5f33c0209c49efd796.jpg)

Welcome to TransLingo, your all-inclusive machine translation tool. During this beta test, TransLingo will initially translate English text to Chinese and then translate Chinese text back to English.

## Here's how to get started

1. Click on Text1. The system will randomly grab a text from an online service

2. Click on the Translate Button. You will see Text1 translated to Chinese. Then, it will be translated back to English

3. Repeat the above steps for Text2

4. After you have clicked both Texts and observed both translations, you can click the Post Survey link to go to the survey page.

Tex12

## English to Chinese

Fig. 2. Snapshot of the TransLingo interface used in the experiment. Participants saw the English source, an intermediate Chinese text, and the final English output.

scale. The Spearman–Brown coeficient reflects consistency across the two text-level ratings; the item does not identify whether participants weighted readability, adequacy, or another quality dimension.

3.5.3 Perceived intelligence. Perceived intelligence was measured using four items adapted from Moussawi and Koufaris [32]. References to the original conversational agent and its task were replaced with TransLingo and machine-translationspecific language. For example, participants rated the statement, “The machine translation system (TransLingo) can understand the input text accurately.” Responses were recorded on a 7-point scale from 1 (“strongly disagree”) to 7 (“strongly agree”). The four items were modeled as indicators of a latent perceived-intelligence factor (� = .914, $C R = . 9 1 5 , A V E = . 7 3 0 )$

3.5.4 Anthropomorphism. Anthropomorphism was measured using four items adapted from Waytz et al. [50]. The system referent and task context were rewritten for machine translation while retaining the capacities assessed by the source measure. For example, participants were asked, “How well do you think this machine translation system (TransLingo) can plan and choose the best translation options available?” Responses used the original 0–10 format, with higher values indicating greater attribution of the specified capacity.

Manuscript submitted to ACM

The four items were modeled as indicators of a latent anthropomorphism factor $( \alpha = . 8 4 4 , C R = . 8 5 8 , A V E = . 6 0 6 ) .$ This operationalization primarily captures cognitive agency—including perceived understanding, anticipation, and planning—rather than humanlike appearance, personality, or emotional experience.

3.5.5 Trust. Trust was measured using seven items adapted from the self-reported trust measure reported by Waytz et al. [50]. Vehicle-specific referents and performance scenarios were replaced with TransLingo and translation-related tasks. For example, participants were asked, “How much do you trust this machine translation system’s (TransLingo) ability to provide accurate translations in the next task?” Responses used the original 0–10 format, with higher scores indicating greater trust.

The seven items were modeled as indicators of a latent trust factor $( \alpha = . 9 5 3 , C R = . 9 5 6 , A V E = . 7 5 5 )$ . Because the items concern translation accuracy, reliability, and task performance, the construct represents performance-related trust in TransLingo rather than trust in its privacy or data-handling practices.

3.5.6 Disclosure willingness. Disclosure willingness was measured using a 36-item battery adapted from Ma et al. [27], building on earlier research on self-disclosure [19, 43]. The original social-media context was replaced with the machine-translation context. For every topic, participants were asked:

“How comfortable do you feel using TransLingo to translate and disclose this aspect of yourself?”

For example, one item concerned participants’ favorite foods and preferred ways of preparing food. Responses ranged from 1 (“very uncomfortable”) to 4 (“very comfortable”).

The battery contained six topics within each of six domains: tastes and interests, attitudes and opinions, work or studies, economic and social status, interpersonal relationships and self-concept, and physical appearance and sex. Following the instrument’s content-sampling structure, the six responses within each domain were averaged to form a domain score. The resulting six domain means were used as separate observed outcome scores in the factorial analyses and were entered into the appraisal-structure SEM as six separate observed endogenous outcomes rather than as indicators of a single disclosure factor. Higher scores indicate greater comfort with disclosing information from the corresponding domain while using TransLingo. Internal consistency was high across the six domain composites (� = .887–.926).

3.5.7 Study Variables and Numerical Coding. The analyses covered the four cells of the focal comparison: simple/readability oriented, simple/fidelity-oriented, complex/readability-oriented, and complex/fidelity-oriented.

For the factorial ANOVAs, output rendering and source-text condition were entered as two-level factors using sum-to-zero contrasts. This coding allowed the Type III tests to estimate marginal main efects averaged over the other factor.

For the appraisal-structure SEM, output rendering was dummy-coded as 0 = readability-oriented and 1 = fidelityoriented, and source-text condition was coded as 0 = simple and 1 = complex. Their interaction was calculated as the product of the two dummy-coded variables. Under this coding, the rendering coeficient represents the rendering contrast within the simple-source reference condition, whereas the source-condition coeficient represents the source contrast within the readability-oriented reference condition.

## 3.6 Data Analysis Methods

The primary analyses used the retained 2×2 focal sample. We conducted two complementary analyses that addressed diferent questions.

Manuscript submitted to ACM

First, condition diferences were examined using separate 2×2 factorial ANOVAs for perceived quality, perceived intelligence, anthropomorphism, task-performance trust, and each of the six disclosure domains. Perceived quality was represented by the mean of the two text-level ratings. Perceived intelligence, anthropomorphism, and trust were represented by their respective composite means for these factorial analyses, and each disclosure outcome was represented by the mean of its six constituent items. Each ANOVA included output rendering, source-text condition, and their interaction.

Because the four cells were unequal in size, omnibus efects were evaluated using Type III tests with sum-to-zero contrasts. Planned rendering contrasts within each source-text condition were estimated using HC3 heteroskedasticity robust standard errors and confidence intervals. Hedges’ � was used as the standardized efect size for these simple contrasts. Perceived quality was the single focal outcome corresponding to RQ1 and was not multiplicity-adjusted. Benjamini–Hochberg false-discovery-rate correction was applied across the three system-appraisal interaction tests and separately within each six-test family for the disclosure outcomes. HC3-robust omnibus tests were used as robustness checks for the Type III interaction tests.

Because each source-text condition was represented by two texts, the rendering contrasts were additionally estimated separately for each text using HC3-robust inference. For perceived quality, a long-form three-way diagnostic included both text-level ratings from each participant and used participant-clustered robust standard errors.

Second, we estimated an appraisal-structure SEM to characterize the theory-ordered concurrent associations among perceived quality, perceived intelligence, anthropomorphism, task-performance trust, and stated disclosure willingness Perceived intelligence, anthropomorphism, and trust were modeled as latent factors using their item-level indicators. Perceived quality was entered as an observed composite of the two text-level ratings. The six disclosure domains were represented by separate observed domain means and were modeled simultaneously as endogenous outcomes, with their residual covariances freely estimated.

The SEM specified associations from output rendering, source-text condition, and their interaction to perceived quality; from perceived quality to perceived intelligence (H1); from perceived intelligence to anthropomorphism (H2) and trust (H4); from anthropomorphism to trust (H3); and from trust to stated disclosure willingness across the six domains (H5a–H5f). Because the post-task appraisal constructs were measured concurrently, the paths among them were interpreted as theory-ordered associations rather than causal or temporal efects.

The SEM was estimated in R using the lavaan package [42], with MLR estimation and full-information maximum likelihood for missing data. No values were missing among the variables used in the retained analytic sample. Indirec associations were tested using 5,000-sample percentile-bootstrap confidence intervals with fixed random seeds. Because lavaan does not combine nonparametric bootstrapping with MLR standard errors, bootstrap intervals were obtained from ordinary maximum-likelihood refits of the same model. Benjamini–Hochberg correction was applied within each six-efect family across the disclosure domains.

## 3.7 Recruitment and Participants

An a priori power analysis was conducted in G\*Power for a fixed-efects 2 × 2 factorial ANOVA, focusing on the one-degree-of-freedom interaction. Assuming a medium efect size (Cohen’s (� = .25)), (� = .05), and 95% statistical power, the analysis indicated a minimum required sample size of 210 participants. The final analytic sample (� = 306) exceeded this requirement; the calculation applies to the rendering-by-source interaction rather than to the SEM or indirect associations. Data were collected through Prolific between August 2023 and February 2025. Participants received \$6.00 for completing the study. Responses were screened for data quality, attention-check performance, Manuscript submitted to ACM

valid experimental-condition coding, and reported language background. Two respondents whose reported language information indicated Chinese were excluded. The final analytic sample comprised 306 participants across the focal $2 \times 2$ comparison: simple/readability-oriented $( n = 8 0 )$ , complex/readability-oriented $( n = 8 8 )$ , simple/fidelity-oriented $( n = 6 8 )$ , and complex/fidelity-oriented $( n = 7 0 )$

Participants self-identified as 155 women (50.7%), 141 men (46.1%), 8 non-binary or third gender participants (2.6%), and 2 who preferred not to say (0.7%). After one invalid age response was treated as missing, ages ranged from 18 to 72 $( M = 3 6 . 6 , S D = 1 2 . 5 )$ . The largest education groups were bachelor’s degree (40.8%), some college (19.6%), high school graduate (14.1%), master’s degree (9.2%), and associate degree (8.8%). Race and ethnicity were collected as a multi-select item; the most frequent selections were White or Caucasian (73.2%), Black or African American (16.7%), Hispanic, Latino/a, or Spanish Origin (10.5%), and Asian (8.2%). Most participants reported English as their first language (304, 99.3%), and 44 (14.4%) reported fluency in at least one additional language. Demographic balance checks found no detectable diferences across cells in age, gender, race/ethnicity, education, or bilingual status (al $ { p } > . 0 5 )$ .

## 4 Findings

We first report the measurement properties of the latent constructs, followed by the factorial condition analyses and the appraisal-structure SEM.

## 4.1 Measurement Model and Discriminant Validity

The three-factor confirmatory factor analysis of perceived intelligence, anthropomorphism, and trust showed adequate fit, robust $\chi ^ { 2 } ( 8 7 ) = 2 2 3 . 5 9 , p < . 0 0 1 , \mathrm { C F I } = . 9 3 9 , \mathrm { T L I } = . 9 2 7 , \mathrm { R M S E A } = . 0 7 2 , 9 0 \% \mathrm { C T } \left[ . 0 6 2 , . 0 8 1 \right] , \mathrm { S R M R } = . 0 3 8 ,$ All standardized loadings were significant and ranged from .577 to .937. Reliability and convergent validity met conventional benchmarks for all three constructs: perceived intelligence $( \alpha = . 9 1 4 , C R = . 9 1 5 , A V E = . 7 3 0 )$ , anthropomorphism $( \alpha = . 8 4 4 , C R = . 8 5 8 , A V E = . 6 0 6 )$ , and trust $( \alpha = . 9 5 3 , C R = . 9 5 6 , A V E = . 7 5 5 )$

Because the appraisal constructs were substantially intercorrelated, we assessed their discriminant validity using HTMT. The HTMT values were .617 between perceived intelligence and anthropomorphism, .705 between perceived intelligence and trust, and .871 between anthropomorphism and trust. The last value is below the .90 criterion but above the more conservative .85 threshold. As an additional test, constraining the latent correlation between anthropomorphism and trust to unity significantly worsened model fit, robust $\Delta \chi ^ { 2 } ( 1 ) = 1 9 . 4 8 , p < . 0 0 1$ . These results support treating anthropomorphism and trust as distinct but closely related constructs

## 4.2 Factorial Condition Diferences Across Outcomes

We used 2×2 factorial ANOVAs to test whether the rendering contrast difered between the simple- and complex-source conditions. Table 2 reports the cell descriptives and interaction tests.

4.2.1 Perceived Quality (RQ1a and RQ1b). The rendering contrast depended on source-text condition, $F ( 1 , 3 0 2 ) = 4 . 5 1 $ $\dot { p } = . 0 3 4 , \eta _ { \phi } ^ { 2 } = . 0 1 5 \left( \mathrm { R Q 1 b } \right)$ . For the simple sources, the fidelity-oriented condition received higher quality ratings than the readability-oriented condition, $\Delta M = . 5 8 4 , 9 5 \% \mathrm { C I } \ |$ [.248, $. 9 2 1 ] , p < . 0 0 1$ , Hedges’ $g = . 5 6 7 \left( \mathrm { R Q 1 a } \right)$ . For the complex sources, the two rendering conditions did not difer, $\Delta M = . 0 1 8 , 9 5 \% \mathrm { C I } [ - . 3 7 6 , . 4 1 3 ] , \ L _ { P } = . 9 2 7 , g = . 0 1 4$

The simple-source contrast was significant for both Text 1, $\Delta M = . 5 6 3 , 9 5 \% \mathrm { C I } [ . 2 1 7 , . 9 0 9 ] , p = . 0 0 1$ , and Text 2, $\Delta M = . 6 0 5 , 9 5 \% \mathrm { C I } [ . 2 3 1 , . 9 7 9 ] , \ L _ { P } = . 0 0 2$ , whereas neither complex-source contrast was significant (both $p \ge . 8 1 8 )$ A direct three-way test found no evidence that the rendering-by-source-condition interaction difered between the two texts, $b = - . 0 9 9 , 9 5 \% \operatorname { C I } { \left[ - . 4 2 7 , . 2 3 0 \right] } , p = . 5 5 6$ . The pattern was therefore consistent across the stimuli employed, although using only two texts at each complexity level limits generalization beyond them.

4.2.2 System Appraisals (RQ2). The same rendering-by-source-condition pattern appeared for perceived intelligence, anthropomorphism, and task-performance trust. The interactions were significant for perceived intelligence, $F ( 1 , 3 0 2 ) =$ 5.66, $p = . 0 1 8 , \eta _ { \mathscr { p } } ^ { 2 } = . 0 1 8$ , FDR-adjusted $q = . 0 2 7$ ; anthropomorphism, $F ( 1 , 3 0 2 ) = 1 1 . 5 6 , \ : p < . 0 0 1 , \ : \eta _ { \phi } ^ { 2 } = . 0 3 7 , q = . 0 0 2 ;$ and trust, $F ( 1 , 3 0 2 ) = 4 . 2 0 , \ : p = . 0 4 1 , \ : \eta _ { \phi } ^ { 2 } = . 0 1 4 , q = . 0 4 1$ . The trust interaction was small and close to the conventional significance threshold and should therefore be interpreted cautiously.

For the simple sources, the fidelity-oriented condition received higher ratings of perceived intelligence, $\Delta M = . 4 0 4 .$ 95% CI $[ . 1 4 6 , . 6 6 2 ] , p = . 0 0 2 , g = . 4 8 9 ;$ anthropomorphism, $\Delta M = 1 . 0 5 1 , 9 5 \% \mathrm { C I } [ . 4 0 1 , 1 . 7 0 1 ] , \dot { p } = . 0 0 2 , g = . 5 2 1 ;$ and trust, $\Delta M = . 7 3 0 { } _ { \mathrm { ; } }$ 95% CI [.132, 1.329], $p = . 0 1 7 , g = . 3 9 6$ . No reliable rendering diferences emerged for these outcomes under the complex sources (all $p \geq . 1 1 9 )$ . The point estimate for anthropomorphism was negative in this condition $( \Delta M = - . 4 9 7 , 9 5 \% \mathrm { C I } [ - 1 . 1 2 4 , . 1 2 9 ] , \ L _ { \hat { P } } = . 1 1 9$ , Hedges $g = - . 2 5 3 )$ , but the confidence interval included zero. Unlike perceived quality, perceived intelligence, anthropomorphism, and trust were each rated once at the system level after participants had reviewed both texts. Consequently, text-specific rendering contrasts could not be estimated for these outcomes.

Averaged across rendering conditions, perceived intelligence was lower in the complex-source condition than in the simple-source condition, $F ( 1 , 3 0 2 ) = 1 5 . 9 8 , \ : p < . 0 0 1 , \eta _ { \mathnormal { p } } ^ { 2 } = . 0 5 0$ . This marginal diference should be interpreted alongside the significant interaction, which indicates that its magnitude varied across rendering conditions.

4.2.3 Disclosure Willingness. No rendering-by-source-condition interaction was significant for any of the six disclosure domains (all $p \geq . 0 7 6 ;$ all FDR-adjusted $q \geq . 3 6 0 )$ . Within the simple-source condition, the unadjusted contrasts reached $\mathit { p } < . 0 5$ for work or studies, $\Delta M = . 2 4 3 , p = . 0 4 2$ , and economic and social status, $\Delta M = . 2 9 2 , p = . 0 3 7$ , but neither remained significant after correction across the six domains (both $q = . 1 2 6 )$ . No rendering contrast was significant within the complex-source condition (all $p \ge . 5 2 6 )$ . Thus, no reliable condition diference was detected for stated disclosure willingness. Across all outcomes, HC3-robust omnibus tests reproduced the same pattern of significant and nonsignificant rendering-by-source-condition interactions as the Type III tests.

## 4.3 How the Appraisals Are Organized: The Structural Mode

The factorial analyses above tested whether the observed outcome scores difered across the study conditions. We next estimated the appraisal-structure SEM to characterize the theory-ordered concurrent associations among perceived quality, perceived intelligence, anthropomorphism, task-performance trust, and stated disclosure willingness. Perceived quality remained an observed composite, whereas perceived intelligence, anthropomorphism, and trust were modeled as latent factors. Figure 3 presents the complete structural specification

The condition predictors were retained in the perceived-quality equation to present the complete structural specification. RQ1a and RQ1b were formally evaluated using the factorial analyses in Section 4.2.1. The corresponding coeficients in Table 3 are included for completeness and are not interpreted as independent confirmatory tests of the research questions. Because the condition predictors were dummy-coded, the rendering and source-condition coeficients are conditional on the reference level of the other factor rather than marginal main efects.

The appraisal-structure model showed acceptable fit, robust $\chi ^ { 2 } ( 2 5 4 ) = 5 1 6 . 0 4 , \dot { p } < . 0 0 1 , \mathrm { C F I } = . 9 4 0 , \mathrm { T L I } = . 9 3 0 ,$ $\mathrm { R M S E A } = . 0 5 8 , 9 0 \% \mathrm { C I } \left[ . 0 5 1 , . 0 6 5 \right] , \mathrm { a n d } \mathrm { S R M R } = . 0 4 4 .$

Manuscript submitted to ACM

Table 2. Condition descriptives and rendering-by-source-condition interaction tests.
<table><tr><td>Outcome</td><td>Simple readability M (SD)</td><td>Simple fidelity M (SD)</td><td>Complex readability M (SD)</td><td>Complex fidelity M (SD)</td><td>F</td><td> $\eta _ { p } ^ { 2 }$ </td><td></td><td>FDR q</td></tr><tr><td>Perceived quality</td><td>5.269 (1.003)</td><td>5.853 (1.051)</td><td>5.210 (1.364)</td><td>5.229 (1.141)</td><td>4.51</td><td>.034*</td><td>.015</td><td></td></tr><tr><td>Perceived intelligence</td><td>5.809 (.985)</td><td>6.213 (.572)</td><td>5.639 (1.059)</td><td>5.543 (.905)</td><td>5.66</td><td>.018*</td><td>.018</td><td> $. 0 2 7 ^ { * }$ </td></tr><tr><td>Agency-oriented anthropomorphism</td><td>5.769 (2.099)</td><td>6.820 (1.889)</td><td>6.219 (1.889)</td><td>5.721 (2.042)</td><td>11.56</td><td> $< . 0 0 1 ^ { \ast \ast \ast }$ </td><td>.037</td><td> $. 0 0 2 ^ { * * }$ </td></tr><tr><td>Task-performance trust</td><td>6.698 (1.860)</td><td>7.429 (1.807)</td><td>6.734 (1.997)</td><td>6.573 (1.867)</td><td>4.20</td><td> $. 0 4 1 ^ { * }$ </td><td>.014</td><td> $. 0 4 1 ^ { * }$ </td></tr><tr><td>Tastes and interests</td><td>3.213 (.693)</td><td>3.382 (.574)</td><td>3.256 (.636)</td><td>3.317 (.563)</td><td>0.58</td><td>.447</td><td>.002</td><td>.447</td></tr><tr><td>Attitudes and opinions</td><td>2.650 (.915)</td><td>2.885 (.790)</td><td>2.750 (.791)</td><td>2.807 (.743)</td><td>0.90</td><td>.344</td><td>.003</td><td>.412</td></tr><tr><td>Work or studies</td><td>2.904 (.785)</td><td>3.147 (.651)</td><td>3.032 (.713)</td><td>3.081 (.702)</td><td>1.39</td><td>.240</td><td>.005</td><td>.360</td></tr><tr><td>Economic and social status</td><td>2.421 (.873)</td><td>2.713 (.809)</td><td>2.545 (.771)</td><td>2.512 (.719)</td><td>3.17</td><td>.076</td><td>.010</td><td>.360</td></tr><tr><td>Interpersonal relationships</td><td>2.673 (.806)</td><td>2.865 (.745)</td><td>2.778 (.700)</td><td>2.762 (.699)</td><td>1.51</td><td>.220</td><td>.005</td><td>.360</td></tr><tr><td>Physical appearance</td><td>2.402 (.868)</td><td>2.645 (.816)</td><td>2.511 (.786)</td><td>2.533 (.646)</td><td>1.49</td><td>.223</td><td>.005</td><td>.360</td></tr></table>

Note. The table reports only the focal rendering-by-source-condition interaction from each Type III 2×2 ANOVA; $d f _ { 1 } = 1$ and <sup>�</sup>� = 302. Partial $\eta _ { p } ^ { 2 }$ is the standardized omnibus efect size. Marginal main efects are omitted from the table for brevity; the perceived-intelligence source-condition efect reported in the text was estimated from the same $\mathrm { T y p e }$ III model. Planned simple rendering contrasts and their standardized efect sizes (Hedges’ �) are reported in the text. Perceived quality was the single focal outcome and was not multiplicity-adjusted. Benjamini–Hochberg correction was applied separately across the three system-appraisal interactions and the six disclosure-domain interactions. $^ { * * * } p < . 0 \dot { 0 } 1 , ^ { * * } p < . 0 1 , ^ { * } \dot { p } < . 0 5$

![](images/bc079b07540c1f945bacf1a56f0621987122485681f11f6ca0ca3cf0253256c7.jpg)  
Fig. 3. Standardized path estimates from the appraisal-structure model. The condition-to-quality paths are included to present the complete structural specification; RQ1a and RQ1b were formally evaluated using the factorial analyses reported in Section 4.2.1. Perceived intelligence, anthropomorphism, and trust were modeled as latent constructs (circles). The six disclosure domains were modeled as separate observed domain-mean outcomes (rectangles). The nonsignificant source-condition path is omited for visual clarity but is reported in Table 3. Values shown are standardized coeficients. ∗ $* * p < . 0 0 1 , * * p < . 0 1 , * p < . 0 5$

4.3.1 Associations among Output and System Appraisals (H1–H4). Perceived quality was positively associated with perceived intelligence $( \mathrm { H 1 : ~ } b \ = \ 1 . 0 2 5 , \beta \ = \ . 7 7 0 , \gamma \ < \ . 0 0 1 )$ . Perceived intelligence was positively associated with anthropomorphism $( \mathrm { H } 2 ; b = . 5 4 6 , \beta = . 6 5 0 , p < . 0 0 1 )$ and trust $( \mathrm { H } 4 ; b = . 4 0 0 , \beta = . 2 9 0 , \beta < . 0 0 1 )$ . Anthropomorphism was also positively associated with trust $( \mathrm { H } 3 ; b = 1 . 1 0 6 , \beta = . 6 7 1 , \gamma < . 0 0 1 )$

4.3.2 Disclosure Willingness Across Six Domains (H5a–H5f ). The disclosure submodel modeled trust as the proximal predictor of disclosure willingness across six personal domains. Trust was positively associated with disclosure Manuscript submitted to ACM willingness in all six: tastes and interests (H5a: $\beta = . 4 5 3 )$ , attitudes and opinions (H5b: $\beta = . 4 7 4 )$ , work or studies $( \mathrm { H } 5 \mathrm { c } \colon \beta = . 5 6 6 )$ , economic and social status (H5d: $\beta = . 3 6 3 )$ , interpersonal relationships $( \mathrm { H } 5 \mathbf { e } \colon \beta = . 4 5 7 )$ , and physical appearance (H5f: $\beta = . 4 6 6 )$ , all $p < . 0 0 1$ and all significant after Benjamini–Hochberg FDR correction.

<table><tr><td>Path</td><td>b</td><td>SE</td><td> $\beta$ </td><td>95% CI</td><td>p</td></tr><tr><td>Output rendering → Perceived quality</td><td>.584</td><td>.169</td><td>.247</td><td>[.254, .915]</td><td>&lt; .001</td></tr><tr><td>Source-text condition → Perceived quality</td><td>-.059</td><td>.183</td><td>-.025</td><td>[-.416, .299]</td><td>.749</td></tr><tr><td>Rendering × source condition → Perceived quality</td><td>-.566</td><td>.260</td><td>-.202</td><td>[-1.076,-.056]</td><td>.030</td></tr><tr><td>Perceived quality → Perceived intelligence</td><td>1.025</td><td>.095</td><td>.770</td><td>[.839, 1.210]</td><td>&lt; .001</td></tr><tr><td>Perceived intelligence → Anthropomorphism</td><td>.546</td><td>.062</td><td>.650</td><td>[.425, .666]</td><td>&lt; .001</td></tr><tr><td>Perceived intelligence → Trust</td><td>.400</td><td>.101</td><td>.290</td><td>[.202, .599]</td><td>&lt; .001</td></tr><tr><td>Anthropomorphism → Trust</td><td>1.106</td><td>.171</td><td>.671</td><td>[.771, 1.442]</td><td>&lt; .001</td></tr><tr><td>Trust → Tastes and interests</td><td>.130</td><td>.019</td><td>.453</td><td>[.092, .168]</td><td>&lt; .001</td></tr><tr><td>Trust → Attitudes and opinions</td><td>.178</td><td>.023</td><td>.474</td><td>[.133, .223]</td><td>&lt; .001</td></tr><tr><td>Trust → Work or studies</td><td>.187</td><td>.020</td><td>.566</td><td>[.148, .227]</td><td>&lt; .001</td></tr><tr><td>Trust → Economic and social status</td><td>.133</td><td>.024</td><td>.363</td><td>[.087, .180]</td><td>&lt; .001</td></tr><tr><td>Trust → Interpersonal relationships</td><td>.155</td><td>.021</td><td>.457</td><td>[.114, .197]</td><td>&lt; .001</td></tr><tr><td>Trust → Physical appearance</td><td>.169</td><td>.022</td><td>.466</td><td>[.126, .212]</td><td>&lt; .001</td></tr></table>

Table 3. Appraisal-structure model. Reported � values are unadjusted; all six trust–disclosure associations remained significant after Benjamini–Hochberg FDR correction (all FDR-adjusted $\textstyle p < . 0 0 1 )$

The indirect association between perceived anthropomorphism and disclosure willingness through trust was significant in all six domains $( \beta = . 2 4 4 - . 3 8 0 ;$ ; all 5,000-sample percentile-bootstrap confidence intervals excluded zero). Adding direct anthropomorphism-to-disclosure paths did not improve model fit, robust $\Delta \chi ^ { 2 } ( 6 ) = 3 . 6 7 , p = . 7 2 2 ,$ , and none of the direct paths was individually significant. These results support retaining the more parsimonious statistical specification.

The appraisal-structure model yielded small model-implied indirect associations between rendering and disclosure willingness for the simple texts $( \beta = . 0 5 0 \mathrm { - } . 0 7 8 ;$ all bootstrap confidence intervals excluded zero; all FDR-adjusted � values were .004). The corresponding associations were negligible for the complex texts $( \beta = . 0 0 9 - . 0 1 4 ;$ all confidence intervals included zero; all FDR-adjusted � values were .927), consistent with the near-zero model-implied rendering contrast for perceived quality in the complex-source condition. This contrast is the sum of the rendering and interaction coeficients in Table $3 , b = . 5 8 4 + \left( - . 5 6 6 \right) = . 0 1 8 .$ These conditional indirect associations do not imply a reliable total condition diference in disclosure, which was not detected in the factorial analyses. Because the appraisal variables were measured concurrently and rendering was compared across collection periods, these estimates describe model-implied associations rather than causal mediation.

## 5 Discussion

## 5.1 The Evaluability Gap in Source-Visible Translation

For the simple narratives, ratings followed the source-retention ordering documented by the stimulus audit: fidelityoriented outputs received higher ratings. For the complex literary-philosophical prose, no rendering diference was detected, although the audit indicated greater source similarity for fidelity-oriented outputs. We refer to this mismatch as an evaluability gap: the source was visible, but overall-quality ratings did not reflect the diference in retained content for the complex stimuli.   
Manuscript submitted to ACM

Earlier machine-translation research identified a related gap between fluency and adequacy. Users react strongly to disfluent output but may overlook adequacy errors in fluent translations [28]. Sentence-level judgments can also compress diferences that become visible when evaluators see document context or inspect errors directly [12, 25]. Ou finding extends this concern to readability-oriented rewriting in a source-visible interface. Showing the source made evidence available, but did not ensure that the holistic rating reflected the audited retention diference for the complex stimuli. This distinction between access and evaluability is the central HCI contribution

Several explanations remain plausible. Comparing a dificult source with its output requires efort, which may increase reliance on ease of reading [1, 39]. Participants may instead have changed the weight assigned to readability and source retention, or the overall-quality item may have combined dimensions that should have been measured separately. The source-text conditions also difered in genre, provenance, period style, topic, and length. Separating these explanations requires matched stimuli and distinct measures of processing efort, readability, and adequacy.

The source-dependent pattern also appeared in system appraisals (RQ2). For the simple texts, the fidelity-oriented outputs — which the stimulus audit showed retained more source content — received higher ratings of perceived intelligence, agency-oriented attribution, and task-performance trust; for the complex texts, the same audited retention advantage was accompanied by no reliable diferences in any of the three appraisals, and all three rendering-by-sourcecondition interactions were reliable.

## 5.2 Output Appraisals and Agency-Oriented Atribution

Perceived output quality was associated with perceived intelligence $( \beta = . 7 7 0 , p < . 0 0 1 )$ , and perceived intelligence was associated with agency-oriented anthropomorphic attribution $( \beta = . 6 5 0 , p < . 0 0 1 )$

Anthropomorphism research has often examined systems that signal humanness through appearance or interaction design [20, 36, 44]. TransLingo omitted those overt cues, although its brand, prototype framing, and task instructions could still shape expectations. In this plain-text utility, ratings of understanding, anticipation, and planning nevertheless covaried with perceived intelligence. A cue-present comparison would help determine how explicit social design changes this relationship.

Elicited agent knowledge ofers one interpretation: behavior associated with human agents can activate a humanoriented reading of a nonhuman system [11]. The items in this study capture cognitive agency rather than emotional or relational humanness, and efectance and sociality motivations were not measured. The ordering is therefore best read as a theory-guided account of concurrent judgments.

Perceived intelligence and agency-oriented attribution were both associated with task-performance trust $( \beta = . 2 9 0$ and $\beta = . 6 7 1$ , respectively; both $\textstyle p < . 0 0 1 )$ , while attribution and trust remained closely related $( \mathrm { H T M T } = . 8 7 1 )$ . The unity-constraint test supported treating them as distinct constructs, but their overlap argues against reading them as sharply separated stages. The measures describe a connected appraisal structure around this plain-text translation interface.

## 5.3 Trust and Disclosure Willingness

Task-performance trust was the construct most directly associated with stated disclosure willingness in the retained model $( \beta = . 3 6 3 \mathrm { - } . 5 6 6 )$ , all FDR-adjusted $\textstyle p < . 0 0 1 )$ ). The exploratory addition of direct agency-attribution paths did not improve fit, robust $\Delta \chi ^ { 2 } ( 6 ) = 3 . 6 7 , p = . 7 2 2$ . Prior studies range from users’ responses to virtual agents in a disclosure context [29], to disclosure intentions in chatbot experiments [13], to information revealed in a field interaction [53]. Within our concurrent self-reports, the model without direct attribution-to-willingness paths was more parsimonious.

Participants reported willingness rather than disclosing personal information. In these data, task-performance trust was the proximal statistical correlate of stated willingness, consistent with evidence that trust may not translate into actual chatbot disclosure [54].

Output evaluation and information entrustment occur in the same interface, yet our results distinguish them. For the simple texts, rendering contrasts appeared in quality and system appraisals but not in disclosure willingness. Interfaces should therefore support judging what an output preserved separately from deciding whether to submit personal text.

## 5.4 Implications for Design and Evaluation

The evaluability gap may also arise in other source-restating tasks. Scientific summaries can be easy to read while extending claims beyond source evidence [38], and explanations can raise confidence without improving accuracy assessment [46]. These tasks need not share one psychological mechanism, but they raise a related design question: how can an interface help users inspect what a fluent output preserved or altered? For translation, this motivates testing source–output alignment, omission or simplification indicators, and separate readability and fidelity assessments.

The results also raise a caution for human-in-the-loop evaluation and preference-based optimization of LLMs in translation and other source-restating tasks. If evaluators repeatedly select the output that feels better overall without separately examining source retention, the resulting feedback may favor readable outputs even when they omit or alter source content. In our own stimulus audit, reference-based metrics (BERTScore, COMET) registered th complex-condition retention diference that participants’ overall ratings did not, illustrating that automatic source-based measures can remain sensitive where holistic human judgments are not. Evaluation pipelines could therefore test separate readability and fidelity judgments, stratification by source dificulty, and source-based metrics or error-focused review alongside overall preference.

Translation performance and personal-data entrustment require diferent design support. Task-performance trust — a construct concerning accuracy and reliability rather than privacy or data handling — was the appraisal most closely associated with stated disclosure willingness, and no reliable rendering contrast emerged for disclosure itself. This separation motivates testing explicit information about data retention, model-training use, and third-party sharing rather than expecting users to infer data safety from translation quality or general trust in the system.

## 6 Limitations and Future Work

The sample comprised English-speaking participants who evaluated the system through back-translation. They could compare the English source with the final English output, unlike users who rely on translation because they cannot read one side of the language pair. The fidelity-oriented rendering received higher ratings only for the simple sources; no rendering diference was detected for the complex sources. This design does not establish how the pattern would change when users cannot evaluate the target language. Follow-up work should test that comparison directly.

Source complexity was represented by only two texts at each level and was confounded with genre, provenance, period style, and length. The simple sources were contemporary generated narratives, whereas the complex sources were literary-philosophical prose. The observed interaction therefore applies to these stimuli and does not isolate which text properties produced it. A stronger test would vary complexity across a larger set of texts matched on these other properties. Replication with independently generated system outputs would test whether the pattern persists beyond these prepared renderings.

Data collection spanned August 2023 to February 2025, a period in which baseline familiarity with generative AI was changing. Prior AI familiarity was not measured. Rendering condition was aligned with the data-collection period and Manuscript submitted to ACM

therefore was not independently randomized across periods; period-related variation is treated as a design boundary rather than as a temporal finding.

All appraisal constructs were collected in the same post-survey, and anthropomorphism and trust were closely related. Common-method efects may therefore contribute to their associations, and the SEM cannot establish their temporal ordering. Repeated-interaction studies should examine how these appraisals change after translation errors, correction, and recovery.

Self-disclosure was measured as stated willingness rather than actual disclosure behavior. The study also did not measure how participants perceived the intimacy, valence, or sensitivity of individual topics. Future work should measure these perceptions and observe what information users actually choose to translate.

## 7 Conclusion

This study provides evidence of an evaluability gap in the tested AI translation setting. Fidelity-oriented rendering received higher ratings of perceived quality, intelligence, agency-oriented attribution, and task-performance trust for the simple sources, but no reliable rendering diference was detected for the complex sources, despite greater source retention under both conditions. In addition, no reliable rendering diference emerged in stated disclosure willingness under either source-text condition. The appraisal-structure model clarified this distinction: participants who reported greater task-performance trust also reported greater disclosure willingness, but the model-implied indirect association between rendering and disclosure was small for the simple sources and negligible for the complex sources. The HCI contribution is a distinction between source access and source evaluability: showing the source can leave users without enough support to recognize what an output has preserved. Because translation quality and stated disclosure willingness did not respond to the rendering contrast in the same way, source-linked change support and data-handling information should be evaluated as separate interventions in consequential translation tasks.

## A Measurement Items

This appendix reports the wording presented to participants.

## A.1 Perceived Intelligence

PI1 The machine translation system (TransLingo) can understand the input text accurately.

PI2 The machine translation system (TransLingo) can provide translations in an understandable manner.

PI3 The machine translation system (TransLingo) can find and process the necessary information for accurate translations.

PI4 The machine translation system (TransLingo) is able to provide me with a useful and accurate translation.

Answer options: 1 = Strongly disagree, to 7 = Strongly agree.

## A.2 Anthropomorphism

ANT1 How smart does this MT system (TransLingo) seem?

ANT2 How well do you think this machine translation system (TransLingo) can understand the context of the text?

ANT3 How well do you think this machine translation system (TransLingo) could anticipate what is about to be written,

before it is actually written?

Manuscript submitted to ACM

ANT4 How well do you think this machine translation system (TransLingo) can plan and choose the best translation options available?

Answer options: 0 = Not at all, to 10 = Very much.

## A.3 Trust

TR1 How accurate do you think the average translation is when utilizing this machine translation system (TransLingo)?

TR2 How much would you trust this machine translation system (TransLingo) to perform accurately in busy, contextrich, real-time language scenarios?

TR3 How much would you trust this machine translation system (TransLingo) to perform accurately in simpler, less context-heavy language scenarios?

TR4 How confident would you feel in the translation quality of this system (TransLingo) if you were actively using it?

TR5 How much do you trust this machine translation system’s (TransLingo) ability to provide accurate translations in the next task?

TR6 How much do you trust this machine translation system’s (TransLingo) ability to provide accurate translations throughout this task?

TR7 How at ease do you feel relying on this machine translation system (TransLingo) for your communication needs? Answer options: 0 = Not at all, to 10 = Very much.

## A.4 Disclosure Willingness

For each topic listed below, participants were asked:

“How comfortable do you feel using TransLingo to translate and disclose this aspect of yourself?”

## A.4.1 Tastes and Interests.

• My favorite foods, the ways I like food prepared.

• The kind of music, books, movies, or TV shows that I cannot bear.

• My favorite singers, movie stars, writers, places, and brands.

• People or organizations that I strongly dislike.

• The kind of party or social gathering that I like the best.

• The type of social gathering that would bore me or that I would not enjoy.

## A.4.2 Atitudes and Opinions.

• Political or religious views that I admire.

• My indiference or dislikes about certain political or religious views.

• Positive comments about the progressing of society.

• My criticism about persisting social problems such as poverty and injustice.

• Support for certain controversial choices such as dropping out of school.

• Skepticism about the choices that people make, e.g., career choices and marriage.

## A.4.3 Work or Studies.

• What I enjoy and get the most satisfaction from in my present work or study.

• Pressures and strains in my work or study.

Manuscript submitted to ACM

• What I feel are my special strong points for my work or study.

• What I find to be the most boring and unenjoyable in work or study.

• Things that I accomplish or achieved in work or study.

• Frustrations about how my work or study is not being valued at all.

## A.4.4 Economic and Social Status.

• Items or signals about how wealthy I am, e.g., luxury trips and accessories.

• Pressing need for money right now, e.g., outstanding bills and debts.

• Optimism about my future financial worth, e.g., getting job ofers.

• Pessimistic views about my own future employment prospects and salaries.

• Connections to people who have high social status.

• Feelings of inferiority economically or socially compared with others around me.

## A.4.5 Interpersonal Relations and Self-Concept.

• Having good times with my significant other.

• Disappointments or bad experiences I have had in a romantic relationship.

• Missing home or old friends.

• Things that I dislike or resent about my friends.

• Things that make me especially proud of myself.

• What I dislike the most about myself.

## A.4.6 Physical Appearance and Sex.

• Confidence in my sexual adequacy.

• Disappointments in past sexual activity or fear of the first sexual experience.

• My standards about the attractiveness of an ideal partner.

• Things I do not like about my appearance—nose, eyes, hair, skin, etc.

• Happiness about being fit, healthy, and attractive.

• Eforts to improve my physical appearance, such as exercise, diet, or surgery.

Answer options: 1 = Very uncomfortable; 2 = Somewhat uncomfortable; 3 = Somewhat comfortable; 4 = Very comfortable.

## References

[1] Adam L. Alter and Daniel M. Oppenheimer. 2009. Uniting the tribes of fluency to form a metacognitive nation. Personality and Social Psychology Review 13, 3 (2009), 219–235. doi:10.1177/1088868309341564

[2] Duygu Ataman, Alexandra Birch, Nizar Habash, Marcello Federico, Philipp Koehn, and Kyunghyun Cho. 2025. Machine Translation in the Era of Large Language Models:A Survey of Historical and Emerging Problems. Information 16, 9 (Sept. 2025), 723. doi:10.3390/info16090723

[3] M. Aurelius and G. Long. 1923. The thoughts of the emperor marcus aurelius antoninus. M. Hopkinson, London, UK. https://books.google.com/ books?id=HLnacDFs30IC

[4] Abdullah Barayan, Jose Camacho-Collados, and Fernando Alva-Manchego. 2025. Analysing Zero-Shot Readability-Controlled Sentence Simplification. In Proceedings ofthe 31st International Conference on Computational Linguistics, Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert (Eds.). Association for Computational Linguistics, Abu Dhabi, UAE, 6762–6781. https: //aclanthology.org/2025.coling-main.452

[5] Joop Bindels, Aletta G Dorst, and Mark Pluymaekers. 2024. Machine Translation in the Workplace: Deciding on the Whether, Why and How. Global Advances in Business Communication 11, 1 (2024), Article 4. https://commons.emich.edu/gabc/vol11/iss1/4/

Manuscript submitted to ACM

[6] Jeanne Sternlicht Chall and Edgar Dale. 1995. Readability Revisited: The New Dale-Chall Readability Formula. Brookline Books, Brookline, MA, USA. Google-Books-ID: 2nbuAAAAMAAJ.

[7] Siew H. Chan, Qian Song, and Lee J. Yao. 2015. The moderating roles of subjective (perceived) and objective task complexity in system use and performance. Computers in Human Behavior 51 (Oct. 2015), 393–402. doi:10.1016/j.chb.2015.04.059

[8] Aihui Chen, Yini Zhang, and Nan Feng. 2025. Thinking twice and weighing outcomes: comparing the dual influence of intelligent healthcare features on word-of-mouth. Behaviour & Information Technology 0, 0 (2025), 1–16. doi:10.1080/0144929X.2025.2502478 \_eprint: https://doi.org/10.1080/0144929X.2025.2502478.

[9] Sabrina Chiesurin, Dimitris Dimakopoulos, Marco Antonio Sobrevilla Cabezudo, Arash Eshghi, Ioannis Papaioannou, Verena Rieser, and Ioannis Konstas. 2023. The Dangers of trusting Stochastic Parrots: Faithfulness and Trust in Open-domain Conversational Question Answering. In Findings ofthe Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics, Toronto, Canada, 947–959. doi:10.18653/v1/ 2023.findings-acl.60

[10] Tamara Dinev and Paul Hart. 2006. An Extended Privacy Calculus Model for E-Commerce Transactions. Information Systems Research 17, 1 (2006) 61–80. https://www.jstor.org/stable/23015781

[11] Nicholas Epley, Adam Waytz, and John T. Cacioppo. 2007. On seeing human: A three-factor theory of anthropomorphism. Psychological Review 114, 4 (2007), 864–886. doi:10.1037/0033-295X.114.4.864

[12] Lukas Fischer and Samuel Läubli. 2020. What’s the diference between professional human and machine translation? A blind multi-language study on domain-specific MT. In Proceedings ofthe 22nd annual conference ofthe european association for machine translation. European Association for Machine Translation, Lisboa, Portugal, 215–224. https://aclanthology.org/2020.eamt-1.23/

[13] Huijian Fu, Lixia Shang, Weican Lin, and Helen S. Du. 2026. Are Consumers Willing to Disclose Information to AI-Based Chatbot? The Roles of Anthropomorphism and Regulatory Focus. Journal of Consumer Behaviour 25, 1 (2026), 394–408. doi:10.1002/cb.70076 \_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/cb.70076.

[14] Sofia Gomes, João M. Lopes, and Elisabete Nogueira. 2025. Anthropomorphism in artificial intelligence: a game-changer for brand marketing. Future Business Journal 11, 1 (Jan. 2025), 2. doi:10.1186/s43093-025-00423-y

[15] Heather M. Gray, Kurt Gray, and Daniel M. Wegner. 2007. Dimensions of Mind Perception. Science 315, 5812 (Feb. 2007), 619–619. doi:10.1126/ science.1134475

[16] Lifeng Han. 2018. Machine Translation Evaluation Resources and Methods: A Survey. https://arxiv.org/abs/1605.04515 Conference poster presented at IPRC 2018; arXiv:1605.04515.

[17] Jefrey T. Hancock, Mor Naaman, and Karen Levy. 2020. AI-mediated communication: Definition, research agenda, and ethical considerations Journal ofComputer-Mediated Communication 25, 1 (2020), 89–100. doi:10.1093/jcmc/zmz022

[18] Sidney M Jourard. 1971. Self-disclosure: An experimental analysis ofthe transparent self. John Wiley, New York, NY, USA.

[19] Sidney M. Jourard and Paul Lasakow. 1958. Some factors in self-disclosure. The Journal ofAbnormal and Social Psychology 56, 1 (1958), 91–98. doi:10.1037/h004335

[20] Joohee Kim and Il Im. 2023. Anthropomorphic response: Understanding interactions between humans and artificial intelligence agents. Computers in Human Behavior 139 (Feb. 2023), 107512. doi:10.1016/j.chb.2022.107512

[21] J. Peter Kincaid, Robert P. Fishburne, Jr., Richard L. Rogers, and Brad S. Chissom. 1975. Derivation of New Readability Formulas (Automated Readability Index, Fog Count and Flesch Reading Ease Formula) for Navy Enlisted Personnel. Research Branch Report 8-75. Naval Technical Training Command, Research Branch, Millington, TN. https://eric.ed.gov/?id=ED108134

[22] T. Koda and P. Maes. 1996. Agents with faces: the efect of personification. In Proceedings 5th IEEE International Workshop on Robot and Human Communication. RO-MAN’96 TSUKUBA. IEEE, Piscataway, NJ, USA, 189–194. doi:10.1109/ROMAN.1996.568812

[23] Hanna Krasnova, Sarah Spiekermann, Ksenia Koroleva, and Thomas Hildebrand. 2010. Online Social Networks: Why We Disclose. Journal of Information Technology 25, 2 (June 2010), 109–125. doi:10.1057/jit.2010.6

[24] Cheng-Chieh Allan Lu, Chu-Chen Rosa Yeh, and Chih-Chien Steven Lai. 2025. The role of intelligence, trust and interpersonal job characteristics in employees’ AI usage acceptance. International Journal ofHospitality Management 126 (April 2025), 104032. doi:10.1016/j.ijhm.2024.104032

[25] Samuel Läubli, Rico Sennrich, and Martin Volk. 2018. Has machine translation achieved human parity? A case for document-level evaluation. In Proceedings ofthe 2018 conference on empirical methods in natural language processing. Association for Computational Linguistics, Brussels, Belgium, 4791–4796. doi:10.18653/v1/D18-1512

[26] Ning Ma, Ruslana Khynevych, Yunqiang Hao, and Yahui Wang. 2025. Efect of anthropomorphism and perceived intelligence in chatbot avatars of visual design on user experience: accounting for perceived empathy and trust. Frontiers in Computer Science 7 (May 2025), 1531976. doi:10.3389/ fcomp.2025.1531976

[27] Xiao Ma, Jef Hancock, and Mor Naaman. 2016. Anonymity, intimacy and self-disclosure in social media. In Proceedings of the 2016 CHI conference on human factors in computing systems (Chi ’16). Association for Computing Machinery, San Jose, California, USA, 3857–3869. doi:10.1145/2858036. 2858414 Number of pages: 13 tex.address: New York, NY, USA.

[28] Marianna Martindale and Marine Carpuat. 2018. Fluency Over Adequacy: A Pilot Study in Measuring User Trust in Imperfect MT. In Proceedings of the 13th Conference ofthe Association for Machine Translation in the Americas (Volume 1: Research Track), Colin Cherry and Graham Neubig (Eds.). Association for Machine Translation in the Americas, Boston, MA, 13–25. https://aclanthology.org/W18-1803/

[29] Diana C. G. Mendes, Rita Costa, Gabriel Agrela, Sergi Bermúdez i Badia, and Mónica S. Cameirão. 2024. Evaluating the Impact of Anthropomorphism and Verbal Interaction on Users’ Perception of a Virtual Agent in Facilitating Self-Disclosure. In 2024 IEEE 12th International Conference on Serious Games and Applications for Health (SeGAH). IEEE, Piscataway, NJ, USA, 1–8. doi:10.1109/SeGAH61285.2024.10639572 ISSN: 2573-3060.

[30] Miriam J. Metzger. 2004. Privacy, Trust, and Disclosure: Exploring Barriers to Electronic Commerce. Journal ofComputer-Mediated Communication 9, 4 (July 2004), JCMC942. doi:10.1111/j.1083-6101.2004.tb00292.x

[31] Sara Moussawi and Raquel Benbunan-Fich. 2021. The efect of voice and humour on users’ perceptions of personal intelligent agents. Behaviour & Information Technology 40, 15 (Nov. 2021), 1603–1626. doi:10.1080/0144929X.2020.1772368

[32] Sara Moussawi and Marios Koufaris. 2019. Perceived intelligence and perceived anthropomorphism of personal intelligent agents: Scale development and validation. In Proceedings ofthe 52nd Hawaii International Conference on System Sciences. Hawaii International Conference on System Sciences, Honolulu, HI, USA, 115–124. doi:10.24251/hicss.2019.015

[33] Sara Moussawi, Marios Koufaris, and Raquel Benbunan-Fich. 2021. How perceptions of intelligence and anthropomorphism afect adoption o personal intelligent agents. Electronic Markets 31, 2 (June 2021), 343–364. doi:10.1007/s12525-020-00411-w

[34] Cliford Nass and Youngme Moon. 2000. Machines and mindlessness: Social responses to computers. Journal ofSocial Issues 56, 1 (2000), 81–103 doi:10.1111/0022-4537.00153 tex.eprint: https://spssi.onlinelibrary.wiley.com/doi/pdf/10.1111/0022-4537.00153.

[35] Manisha Natarajan and Matthew Gombolay. 2020. Efects of anthropomorphism and accountability on trust in human robot interaction. In Proceedings ofthe 2020 ACM/IEEE international conference on human-robot interaction (Hri ’20). Association for Computing Machinery, Cambridge, United Kingdom, 33–42. doi:10.1145/3319502.3374839 Number of pages: 10 tex.address: New York, NY, USA.

[36] Kristine L. Nowak and Christian Rauh. 2005. The Influence of the Avatar on Online Perceptions of Anthropomorphism, Androgyny, Credibility, Homophily, and Attraction. Journal ofComputer-Mediated Communication 11, 1 (2005), 153–178. doi:10.1111/j.1083-6101.2006.tb00308.x \_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1083-6101.2006.tb00308.x

[37] Guilherme Pascoal, Marlinde van den Bosch, Olga Viberg, Jacqueline Wong, and Carrie Demmans Epp. 2026. Improving Text Readability to Support Student Comprehension and Learning: An LLM-Powered Approach. In Two Decades ofTEL. From Lessons Learnt to Challenges Ahead, Kairit Tammets, Sergey Sosnovsky, Rafael Ferreira Mello, Gerti Pishtari, and Tanya Nazaretsky (Eds.). Springer Nature Switzerland, Cham, 291–305. doi:10.1007/978-3-032-03870-8\_20

[38] Uwe Peters and Benjamin Chin-Yee. 2025. Generalization bias in large language model summarization of scientific research. Royal Society Open Science 12 (2025), 241776. doi:10.1098/rsos.241776

[39] Rolf Reber and Norbert Schwarz. 1999. Efects of perceptual fluency on judgments of truth. Consciousness and Cognition 8, 3 (1999), 338–342. doi:10.1006/ccog.1999.038

[40] Ricardo Rei, Craig Stewart, Ana C. Farinha, and Alon Lavie. 2020. COMET: A Neural Framework for MT Evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 2685–2702. doi:10.18653/v1/2020.emnlp-main.213

[41] Kristina M. Rennekamp. 2012. Processing Fluency and Investors’ Reactions to Disclosure Readability. doi:10.2139/ssrn.1881847

[42] Y. Rosseel. 2012. Lavaan: An R package for structural equation modeling and more. Version 0.5-12 (BETA). Journal ofStatistical Software 48, 2 (2012) 1–36. doi:10.18637/jss.v048.i02

[43] Zick Rubin and Stephen Shenker. 1978. Friendship, proximity, and self-disclosure. Journal of Personality 46, 1 (1978), 1–22. doi:10.1111/j.1467- 6494.1978.tb00599.x tex.eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-6494.1978.tb00599.x.

[44] Maha Salem, Friederike Eyssel, Katharina Rohlfing, Stefan Kopp, and Frank Joublin. 2013. To Err is Human(-like): Efects of Robot Gesture on Perceived Anthropomorphism and Likability. International Journal ofSocial Robotics 5, 3 (Aug. 2013), 313–323. doi:10.1007/s12369-013-0196-9

[45] Sara Salimzadeh, Gaole He, and Ujwal Gadiraju. 2023. A Missing Piece in the Puzzle: Considering the Role of Task Complexity in Human-AI Decision Making. In Proceedings of the 31st ACM Conference on User Modeling, Adaptation and Personalization (UMAP ’23). Association for Computing Machinery, New York, NY, USA, 215–227. doi:10.1145/3565472.3592959

[46] Mark Steyvers, Heliodoro Tejeda, Aakriti Kumar, Catarina Belem, Sheer Karny, Xinyue Hu, Lukas W. Mayer, and Padhraic Smyth. 2025. What large language models know and what people think they know. Nature Machine Intelligence 7 (2025), 221–231. doi:10.1038/s42256-024-00976-7

[47] Indrit Troshani, Sally Rao Hill, Claire Sherman, and Damien Arthur. 2021. Do we trust in AI? Role of anthropomorphism and intelligence. Journal of Computer Information Systems 61, 5 (2021), 481–491. doi:10.1080/08874417.2020.1788473 tex.eprint: https://doi.org/10.1080/08874417.2020.1788473

[48] Inara Tusseyeva, Anara Sandygulova, and Matteo Rubagotti. 2024. Perceived Intelligence in Human-Robot Interaction: A Review. IEEE Access 12 (2024), 151348–151359. doi:10.1109/ACCESS.2024.3478751

[49] Lucas Nunes Vieira, Minako O’Hagan, and Carol O’Sullivan. 2021. Understanding the societal impacts of machine translation: a critical review of the literature on medical and legal use cases. Information, Communication & Society 24, 11 (Aug. 2021), 1515–1532. doi:10.1080/1369118X.2020.1776370

[50] Adam Waytz, Joy Heafner, and Nicholas Epley. 2014. The mind in the machine: Anthropomorphism increases trust in an autonomous vehicle Journal ofExperimental Social Psychology 52 (May 2014), 113–117. doi:10.1016/j.jesp.2014.01.005

[51] John S. White and Theresa A. O’Connell. 1993. Evaluation of Machine Translation. In Human Language Technology: Proceedings ofa Workshop Held at Plainsboro, New Jersey, March 21-24, 1993. Association for Computational Linguistics, Stroudsburg, PA, USA, 206–210. doi:10.3115/1075671.107571

[52] Robert E Wood. 1986. Task complexity: Definition of the construct. Organizational Behavior and Human Decision Processes 37, 1 (Feb. 1986), 60–82. doi:10.1016/0749-5978(86)90044-0

[53] Yuqian Xu, Hongyan Dai, and Wanfeng Yan. 2026. Identity Disclosure and Anthropomorphism in Voice Chatbot Design: A Field Experiment Management Science 72, 1 (2026), 223–241. doi:10.1287/mnsc.2022.03833 Published online August 26, 2024.

[54] Mingxin Zhang and Hou Zhu. 2026. Actual Self-disclosure to Anthropomorphic AI Chatbots: A Contextual Privacy Calculus Approach. AIS Transactions on Human-Computer Interaction 18, 1 (2026), 31–60. doi:10.17705/1thci.00237

[55] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating text generation with BERT. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=SkeHuCVFDr

[56] Scott Zieger, Jiayuan Dong, Skye Taylor, Caitlyn Sanford, and Myounghoon Jeon. 2023. Happiness and high reliability develop afective trust in in-vehicle agents. Frontiers in Psychology 14 (2023), 1129294. doi:10.3389/fpsyg.2023.1129294

[57] Mehmet Şahin and Derya Duman. 2014. Multilingual Chat through Machine Translation: A Case of English-Russian. Meta 58, 2 (March 2014), 397–410. doi:10.7202/1024180ar