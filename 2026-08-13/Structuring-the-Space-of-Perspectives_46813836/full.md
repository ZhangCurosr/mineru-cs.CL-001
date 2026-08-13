# Structuring the Space of Perspectives

Agnese Daffara University of Stuttgart agnese.daffara@ims.uni-stuttgart.de

Sebastian Padó University of Stuttgart pado@ims.uni-stuttgart.de

Tanise Ceron Bocconi University tanise.ceron@unibocconi.it

## Abstract

The same event can be reported from different perspectives depending on the experiences, background, and beliefs of the writer or speaker. A variety of NLP areas engage with perspectives, spanning from text analy sis to algorithm optimization. A wide range of operative concepts (such as stances, sentiment, frames, and arguments) has been used to capture perspectives in texts, however the precise relationships among those concepts remain unclear. Arguably, a deeper theoretical understanding of these concepts would empower more effective research on perspectives. In this paper, we address this gap by reviewing the space of perspectives in NLP and defining a set of properties that help distinguishing perspective-related concepts. Our analysis leads us to posit a hierarchy which organizes these concepts linearly along a single axis. Finally, we show how this principled conceptual hierarchy can help researchers navigate the field and select operationalizations of perspective that align with their specific research objectives.

## 1 Introduction

Every day we are exposed to a wide variety of information through different text types, from personalrelated content like product reviews and social media posts, to official productions like political debates and news articles. While some texts explicitly express personal opinions, others aim to report factual information. Yet, all of them project some perspective onto the world. Indeed, perspectivetaking is a prerequisite of human communication (Graumann and Kallmeyer, 2008). The NLP community has long studied perspectives, but there is a growing interest in recent years<sup>1</sup>.

![](images/7d53d01b5cd09ccf697d6e5fd33684d96635ebf2500d48ef1481133a3f6e003a.jpg)  
Figure 1: A model of perspective with key concepts related to perspective. The scheme arranges them on a specificity axis, ranging from ideological concepts to specific linguistic devices, as described in §4.3. Extratextual factors are shown in a separate box.

This trend has unfolded across many (sub)communities and under a variety of labels (cf. Figure 1). The diversity of concepts and frameworks makes it challenging to clarify the connections between different studies, to identify gaps in existing research, and to uncover the theoretical backgrounds and assumptions underlying individual studies. Imagine, for example, that a researcher aims to build a multi-perspective news recommender or to diversity opinions in language models’ generated text: what concepts and what research should they be aware of? As Reuver et al. (2021a) note, bridging perspective frameworks from the domain of Natural Language Processing (NLP) is essential to advance democratic values in information consumption and dissemination.

The Term Perspective A perspective generally indicates a viewpoint on reality. Historically, the term has been used in multiple ways across research fields. In narrative theory, it simply refers to the positioning of a viewpoint within a particular character or narrator in the discourse (Sanders and Redeker, 1993). From a cognitive viewpoint, it has been demonstrated that reality itself is a collection of perspectives (Basile et al., 2022). In discourse studies and linguistics, it has been widely studied how semantic and syntactic choices trigger perspectivization (or vantage point taking), by representing a state of affair in terms of actor roles and their viewpoints; e.g., whether the perpetrator of a murder is presented as the responsible agent (Graumann and Kallmeyer, 2008). Recently, the research program of perspectivism adopted the term to denote the entirety of socio-demographic, cultural, and individual traits in annotations and modeling (Frenda et al., 2024). In this paper, we address perspectives directly embedded in the text and discuss extra-textual traits as additional factors (cf. Figure 1). Building on its generic usage in NLP literature, we adopt perspective as an umbrella term, comprising related concepts in a way to be further clarified.

Relevance in NLP The interest in identifying perspectives in texts has mainly two aims: i) perspective recognition; ii) perspective generation. Regarding the first aim, perspective traditionally denotes a political orientation (Pang et al., 2008; Küçük and Can, 2020); e.g., LIBERAL vs. CONSERVATIVE. However, it is commonly expanded in its meaning to indicate other related notions, like opinions (Morante et al., 2020), stances (Klebanov et al., 2010; Roy and Goldwasser, 2023), frames (Liu et al., 2019; Alashri et al., 2015), sentiment (Yu and Hatzivassiloglou, 2003; Greene and Resnik, 2009), and claims (Chen et al., 2019).

Concerning the second aim, the rise of LLMs has introduced new research questions about perspectives: what kind of data and annotations are used for training? Are they biased toward specific viewpoints (Lin et al., 2025; Ceron et al., 2025)? Can these models preserve multiple and minor perspectives in summarization (Van Der Meer, 2024) and reproduce different opinions in generation (Hu et al., 2025)? Generating diverse perspectives in AI applications is desirable to promote alignment with democratic values and ensure that certain opinions are not underrepresented (Sorensen et al., 2024b).

Previous Work A number of surveys focus on perspective-related concepts and tasks. This enables a detailed investigation, but limits the possibility of connecting and organizing them – the gap we address in this paper. Among others, Pang et al. (2008) provide an overview on opinion mining and sentiment analysis, Munezero et al. (2014) on subjectivity-related terms; Küçük and Can (2020) on stance detection, Doan and Gulla (2022) on political perspective detection, Otmakhova et al. (2024) on media frames, and Rodrigo-Ginés et al. (2024) on media bias (cf. Appendix A.2 for more references).

All these concepts relate to how a viewpoint is expressed in language and are thus characterized by some shared linguistic features. A line of work in linguistics (Hunston and Thompson, 2000; Benamara et al., 2017) uses the term evaluation to denote “the speaker or writer’s attitude or stance towards, viewpoint on, or feelings about the entities or propositions that she or he is talking about” (Hunston and Thompson 2000, p. 5). This clarifies the linguistic surface of perspective and roots the discussion in the domain of evaluation, as opposed to factual knowledge.

Contributions and Structure Our paper aims at improving the state of the art in perspective-related research by asking two research questions:

• RQ1: What concepts are used to identify perspectives in text?

• RQ2: How are these concepts related?

To answer these questions, we first introduce and characterize the space of perspective-related concepts based on the literature (§3). We then organize and structure the conceptual space with a property annotation<sup>2</sup>, clustering analysis and Principal Component Analyis (PCA) (§4). The outcome of the analysis is the model of perspective shown in Figure 1: we find that clusters of perspective-related concepts form a hierarchy along a linear scale of conceptual and linguistic specificity. To make this insights actionable, we propose a decision tree (Figure 5) to help researchers make informed choices among the discussed concepts that match their specific research goals (§5).

## 2 Paper Collection

Our paper is not a full survey, but rather a conceptual organization, supported by a synthesized literature review with the goal of capturing the space of perspective-related concepts together with their definitions and characteristic properties. To understand the concepts related to perspectives in NLP, we first conduct a regular expression based search on the ACL Anthology bibliography and select 60 papers (see Appendix A.1 for details).

Since this search did not capture some foundational work that contributed to the conceptualization of perspective in NLP, neither the literature from neighbor areas (e.g, communication science), we continue collecting papers manually on Google Scholar, by following citation trails in references and related works. Papers are included in the analysis if they: i) provide a definition of perspective; ii) use perspective in a conceptual way that distinguishes it from similar work; iii) establish a conceptual grounding for a certain term. After some iterations, we organize all papers in a bottom-up fashion, which resulted in a final set of relevant concepts. The final number of collected papers is 227 (full list in Appendix A.2).

## 3 Perspective-related Concepts

Based on our analysis of the extant literature, we collect a total of 15 concepts relevant for perspectives. For each concept, we characterize it and briefly discuss relevant textual features and methods used in computational modeling ("Perspective Signals"). Discriminative properties of each concept are marked in bold; linguistic levels are indicated in italics. Section 3.9 discusses how these concepts relate to each other and provides two unified examples. Further definitions and additional examples can be found in Appendix B.2.

## 3.1 Morals and Values

We include morals and values because they are increasingly studied in NLP as the foundation of perspective given their proximity with ideological positions (Graham et al., 2009). Values motivate arguments by influencing the positions adopted and the justifications offered for them (Kiesel et al., 2022; Atkinson and Bench-Capon, 2021; Van Der Meer, 2024). They form “the basis for all processes of evaluation” (Van Dijk, 1998), being "the intrinsic goods or ideals that individuals pursue or cherish"

(Sorensen et al., 2024a), such as FREEDOM and EQUALITY. Morals are frequently studied together in the NLP literature, but they origin from distinct frameworks (Dönmez and Falenska´ , 2026); they are societal and encode a shared judgment of what is right or wrong (Vida et al., 2023; Graham et al., 2009; Entman, 1993), e.g., the moral principle that causing harm is wrong is accepted across cultures.

Perspective Signals Values are grounded in psychological frameworks (e.g., Schwartz (1992)) and are often operationalized through social surveys, such as the World Values Survey (Inglehart et al., 2000). While values are difficult to spot in text because they are high-level constructs, some evaluation marks can be recognized, such as the mention of goals or (non-)achievements (Hunston and Thompson, 2000). Morals are often studied with reference to the Moral Foundations Theory (MFT) (Graham et al., 2009), which identifies five virtue/vice dimensions: CARE/HARM, FAIRNESS/CHEATING, LOYALTY/- BETRAYAL, AUTHORITY/SUBVERSION, and PU-RITY/DEGRADATION. The Moral Foundations Dictionary (Graham et al., 2009) maps lexical items to these dimensions. Morals and values are also operationalized as frames – some values, such as SECURITY, overlap with Media Frames labels; at the semantic level, morality frames associate the MFT dimensions with agents and objects (Roy et al., 2021). Recent work on morals and values in NLP focuses mainly on generating pluralistic values (Sorensen et al., 2024a) and assessing LLMs ideological (Ceron et al., 2024; Benkler et al., 2023) and moral (Abdulhai et al., 2024) alignment or inducing it through reinforcement learning.

## 3.2 Ideology

While morals and values provide the overarching beliefs guiding decisions, ideology is the coherent system that organizes them into a stable set of ideas shared by a group. Ideology is stable because it is anchored in a shared system of core values that function as evaluative criteria across topics and contexts; just like the grammar of a language, which is the reference point for its rules (Van Dijk 1998, p. 56). Because it is anchored in these group-level commitments, ideology operates at a higher level of abstraction than stance or sentiment (which are granular and variable), and can be interpreted as a cluster of aligned opinions (Van Dijk, 1998; Doan and Gulla, 2022). In NLP, ideology is used in political domains (Pang et al., 2008). It manifests as (i) a position in a debate (e.g. PRO-ISRAEL vs. PRO-PALESTINE), (ii) a political leaning on a spectrum (e.g. LEFT vs. RIGHT), or (iii) a party affiliation (e.g. DEMOCRATICS vs. REPUBLICANS).

The task of ideology bias detection classifies texts as BIASED or UNBIASED (or on a scale inbetween), measuring the degree of ideological skew (Rodrigo-Ginés et al., 2024), which arguably assumes the existence of neutral, factual texts (Vargas et al., 2023). Note that while bias detection identifies whether a text is ideologically skewed without specifying orientation, ideology detection assigns it to a specific group. Early work framed the task as binary classification between two opposing views, such as the Israeli–Palestinian conflict (Lin et al., 2006). Political leaning and party detection are instead typically modelled as multi-class, regression or scaling problems, situating texts along a political scale (Budak et al., 2016; Kiesel et al., 2019; Ceron et al., 2022), for example from CONSERVATIVE to LIBERAL.

Perspective Signals At the lexical level, the most informative signals are one-sided terms and sticky bigrams (Klebanov et al., 2010; Recasens et al., 2013; Monroe et al., 2008), for example, illegal aliens is linked to conservative discourse (Webson et al., 2020). Early approaches exploited these via bag-of-words (Lin et al., 2006; Laver et al., 2003; Slapin and Proksch, 2008) and n-grams (Hardisty et al., 2010); because such patterns overlap with topic distributions, perspectives have also been modelled via LDA (§3.8), though this risks reducing ideology to surface frequencies. At the semantic and syntactic levels, factive verbs, lexical entailments, hedges (Greene and Resnik, 2009; Recasens et al., 2013), and constructions like the passive voice convey ideological positioning without overt evaluation. At the pragmatic level, metaphor (Sengupta et al., 2024) and rhetorical strategies (Huguet Cabot et al., 2020) carry meaning beyond literal content (cf. Table 1).

Neural approaches capture signals across all levels through word embeddings (Iyyer et al., 2014; Gangula et al., 2019a; Li and Goldwasser, 2019; Alzhrani, 2022), at the cost of interpretability (Martinez et al., 2024). Recent work has leveraged encoder-based LLMs (Baly et al., 2020) and decoder-based ones (Kim et al., 2023; Da San Martino et al., 2023), enriched with social network relations (Li and Goldwasser, 2019; Baly et al.,

<table><tr><td>Phenomenon</td><td>Example</td></tr><tr><td>Lexical</td><td></td></tr><tr><td>One-sided terms</td><td>pro-life</td></tr><tr><td>Sticky bigrams Evaluative language</td><td>illegal aliens</td></tr><tr><td>Subj. intensifiers</td><td>suitable fantastic</td></tr><tr><td>Semantic</td><td></td></tr><tr><td>Factive verbs</td><td>reveal</td></tr><tr><td>Assertive verbs</td><td>say</td></tr><tr><td>Lexical entailments</td><td>murdered</td></tr><tr><td>Hedges</td><td>possibly</td></tr><tr><td>Syntactic</td><td></td></tr><tr><td>Passive constructions</td><td>mistakes were made</td></tr><tr><td>Pragmatic</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Metaphor</td><td>real beef</td></tr><tr><td>Implicature</td><td>[implicit blame]</td></tr></table>

Table 1: Signals of ideological bias by level of linguistic analysis (Yano et al., 2010; Recasens et al., 2013; Greene and Resnik, 2009; Sengupta et al., 2024).

2020), Wikipedia (Li and Goldwasser, 2021; Feng et al., 2021), or political speeches (Jakob et al., 2024). Hybrid methods combine text with knowledge graphs (Zhang et al., 2022).

## 3.3 Stances

Stance indicates an ideological position towards a target. It is usually treated together with sentiment as an evaluative and affective concept. However, it may also be epistemic if there is no affective component (Kiesling et al., 2018; Küçük and Can, 2020). Typical labels are AGAINST/NEGATIVE, NEUTRAL/NEITHER, and PRO/FAVOR/POSITIVE. While binary ideology detection is sometimes referred to as stance detection, there are important distinctions between the two: (i) ideology is a stable overarching system, whereas stance is individually variable (cf. §3.2), and (ii) ideology is targetgeneric, whereas stance is target-specific, requiring one or more explicit or inferable targets (Küçük and Can, 2020): entities (e.g. Donald Trump), policy issues (e.g. border control), events (e.g. the approval of a new policy), or claims (e.g. "we need to increase border control to ensure national security"). Stance also overlaps with entity-based or aspect-based sentiment analysis, which seeks to identify emotional attitude toward a target (§3.4); however, stance reflects evaluative alignment rather than emotional tone. As Hasan and Ng (2012) note, the same document may have negative sentiment expressions but a positive stance (cf. Figure 2).

Perspective Signals Stance is typically conveyed through the structure of argumentation, which makes it difficult to capture for simple bag-ofwords approaches. Pioneer stance detection tasks integrated lexical subjectivity and polarity features with parse trees and discourse-level argumentative features (Wiebe et al., 2005; Somasundaran and Wiebe, 2010; Hasan and Ng, 2012; Bar-Haim et al., 2017; Anand et al., 2011). These features remain latent when using neural networks (Roy and Goldwasser, 2023; Mohammad et al., 2016). The detection can be enriched by integrating semantic information like entities, their roles, and associated sentiment (Roy and Goldwasser, 2023).

## 3.4 Sentiment and Emotions

We have seen that ideology reflects stable, overarching belief systems shared by a group (§3.2) and stance captures the author’s ideological position towards a target (§3.3). In contrast, sentiment and opinions indicate a subjective response with an affective component. In fact, they are traditionally linked to the area of subjectivity detection, where they were originally theorized as private states, i.e., mental states that are not accessible to objective observation or verification (Quirk et al., 1985).

Sentiment can be interpreted in two ways: (i) in a general sense, it refers to instances of perspective expressed in a text, which is why sentiment analysis and opinion mining are traditionally treated as equivalent tasks (Pang et al., 2008); (ii) in a more fine-grained view, it reflects the polar orientation (POSITIVE, NEUTRAL, NEGATIVE) of an opinion, whereas an opinion represents the full perspective expression (Munezero et al., 2014). For example, the sentence “illegal aliens are ruining the country” is a fully opinion expression including a NEGATIVE sentiment. Sentiment can also be referred to specific targets or entities (entity-based) or to specific attributes of the target (aspect-based) (Rønningstad et al., 2024; Küçük and Can, 2020).

Emotions are affective states that go beyond polar orientation to specify the type of emotional response (such as FEAR, ANGER, or SADNESS) typically grounded in psychological models such as Ekman’s basic emotions (Ekman, 1992) or Plutchik’s wheel (Plutchik, 1980; Plaza-del-Arco et al., 2024).

Perspective Signals Early approaches to sentiment analysis and emotion detection used subjectivity features as a proxy. At the lexical level, they relied on static methods: unigrams (Pang et al., 2002), pre-compiled subjectivity lexicons (Liu et al., 2005; Yu and Hatzivassiloglou, 2003; Wilson, 2005; Mohammad and Turney, 2013) and bootstrapped patterns (Riloff and Wiebe, 2003). At the semantic and discourse levels, dynamic methods exploit contextual meaning (Wilson et al., 2005), WordNet relations (Choi and Wiebe, 2014), constituents (Kim and Hovy, 2004), and dependency patterns combined with discourse-level cues (Hasan and Ng, 2012). A key resource supporting this line of research is the MPQA (Multi-Perspective Question Answering) corpus (Wiebe et al., 2005), comprising news articles annotated with subjectivity, entityand event-level sentiment (Deng and Wiebe, 2015).

## 3.5 Opinions

Munezero et al. (2014) provide an overview of the definitions and representations of opinion. First, there is a shared intuition that opinion involves some degree of uncertainty, as it is not factual but instead tied to personal beliefs, just like sentiment. Second, opinion is a structured construct that can be decomposed into various sub-components (see below). Note that also sentiment can be structured into event-level components; the key distinction between these two concepts seems to lay in the facts that (i) an opinion can lack a sentiment, like in the sentences “Bin Laden is hiding in Pakistan” or "I believe the word is flat" (Kim and Hovy, 2004) and (ii) an opinion can be identified with the linguistic expression itself (e.g., "Mary said the dress is beautiful")<sup>3</sup>, while sentiment tends to denote an abstract attitude mapped onto polar labels (e.g. POSITIVE).

Perspective Signals Opinion mining often employs a structured conceptualization at the semantic level. According to Kim and Hovy (2004), an opinion consists of four elements: (i) the topic, (ii) the holder, (iii) the claim, and (iv) optionally, the sentiment. Other proposed components include the features of the target (aspects) and the time when the opinion is expressed (Liu et al., 2010). These conceptualizations are extended by van Son et al. (2016), one of the few proposals for a structured representation of perspective. They proposed an annotation scheme with: (i) the event structure, (ii) the attribution (relationship between the source and the target), (iii) the factuality (certainty, polarity and time), and (iv) the opinion (sentiment). The framework was later used to annotate a corpus of news items about Covid-19 (Morante et al., 2020), where perspectives were defined as “relations between the source of a statement (i.e., the author or another entity [...]) and a target in that statement (i.e., an entity, event, or (micro-)proposition)”. In this sense, structured opinions were seen as perspective expressions operating at the semanticpragmatic level, with sentiment as an ideological sub-component. Table 2 summarizes discriminative properties of opinion-related concepts.

![](images/dea2a0f885318d03e7d7c5195dd129bf2c95ff90a533ad9adbe077e77aa1b949.jpg)  
Table 2: Discriminative properties of opinion-related concepts describing whether they have sub-components, are polar, have a target, are stable, and have an affective tone. yes; optional; no. All properties are discussed in the literature review (marked in bold).

## 3.6 Claims and Arguments

Argumentation lies at the core of perspective and can be understood as a basis for representing it (Van Der Meer, 2024). Therefore, we include in our review the notions of claim (a statement functioning as the minimal unit of argumentation) and argument (a set of statements composed of premises and conclusions) (Govier, 2005). While opinions describe what people think about a topic or product (cf. §3.5), arguments explain why they hold these opinions (Lauscher et al., 2022). Although opinion mining and argument mining are distinct tasks, their boundaries can be blurry, because opinion mining may also involve identifying argumentative motivations behind a sentiment (Cabrio and Villata, 2018). Since argumentation provides the foundational layer for perspective detection, in the next paragraph we focus on how it is leveraged for higher-level perspective detection.

Perspective Signals In supervised approaches, argumentation n-grams involving the discourse level have been shown to be more effective than lexical sentiment features for stance detection, as they encode reasoning patterns (Somasundaran and Wiebe, 2010) (cf. §3.3). Indeed, the MPQA corpus annotates trigger expressions of positive and negative argumentation (e.g., be important to, would be better, cannot imagine, we don’t need).

In unsupervised approaches, a perspective can be seen as an aggregation of arguments. Clusters of similar arguments (and therefore similar opinions) can reveal overarching belief systems and be interpreted as ideological groups (Abu-Jbara et al., 2012, 2013; Chen et al., 2017) (cf. §3.2) or can group texts by frame (De Vreese, 2005; Reimers et al., 2019), where a frame is defined as “a set of arguments that shares an aspect” (Ajjour et al., 2019) (cf. §3.7). Only a few contributions propose a more structured approach. Carlebach et al. (2020) use a five-step pipeline for perspective-oriented news aggregation: topic modeling, hypothesis extraction, semantic similarity on hypotheses, premise extraction, and textual entailment. Chen et al. (2019) extract distinct arguments associated with a claim, which collectively constitute a spectrum of perspectives (perspectrum). Overall, leveraging argumentation structure for perspective detection is still an open research direction (Lauscher et al., 2022).

## 3.7 Frames

According to Boydstun et al. (2013), framing means “portraying an issue from one perspective to the necessary exclusion of alternative perspectives”. In some work, frame and perspective are used as synonyms (Alashri et al., 2015; Liu et al., 2019). When we frame something, we do three things: (i) selecting: choosing what to present and what not to; (ii) focussing: highlighting or emphasizing some parts; (iii) embedding: presenting some information as the part and some other as the whole (Van Hulst et al., 2025). These processes take place at the cognitive level (mental representations of the world), the semantic level (choosing what linguistic structures to use), and the communicative level (impact on the audience) (Otmakhova et al., 2024). However, defining framing is "notoriously slippery" (Boydstun et al., 2013; Field et al., 2018) because there are various ways of analyzing it: one can use, for example, topiclike dimensions such as media frames (Entman, 1993), semantic patterns such as semanticframes (Fillmore, 1976) and connotationframes (Rashkin et al., 2016), narrative dimensions such as narrative frames (e.g. HERO, VICTIM) (Otmakhova et al., 2024), or morality dimensions such as morality frames (e.g. CARE/HARM (Roy et al., 2021) (cf.

§3.1). These different ways of detecting framing communicate with each other, for example, the narrative role PERPETUATOR can be associated with the semantic role AGENT. In this section, we focus on the first two types.

Media frames or communication frames are framing dimensions found in media. They can be issue-generic or issue-specific and the labels can be defined in an inductive or deductive fashion (De Vreese, 2005). A famous annotation framework is the Media Frame Corpus (Card et al., 2015), including 15 labels (e.g., MORALITY, ECONOMIC, HEALTH AND SAFETY), widely adopted in media studies (Khanehzar et al., 2019, 2021; Mendelsohn et al., 2021; Mulder et al., 2021; Gilardi et al., 2023; Piskorski et al., 2023). Semanticframes, on the other hand, study framing through linguistic structures and semantic roles (e.g. KILLING: the KILLER or CAUSE causes the death of a VICTIM). The theory of semanticframes was introduced by Fillmore (1976) and operationalized as FrameNet (Baker et al., 1998). Framing is a perspectivization where cognitive dispositions are induced and perpetuated through language (Minnema et al., 2022b). The main limitation is that semantic frames are primarily used to investigate specific issues, such as femicides (Minnema et al., 2022a) and car crashes (Te Brömmelstroet, 2020), as the specificity of the patterns impedes cross-domain generalization.

Perspective Signals Early work in media frame detection relied on topic models, subtracting framing features from topic representations (cf. §3.8). Traditional supervised classification techniques mainly leveraged n-grams and lexical features (Baumer et al., 2015). Following approaches explored embedding-based lexicon expansion (Field et al., 2018), neural networks (Naderi and Hirst, 2017; Liu et al., 2019; Morstatter et al., 2018), fine-tuning of pre-trained models (Liu et al., 2019; Khanehzar et al., 2019; Mendelsohn et al., 2021; Kwak et al., 2020) and prompting (Piskorski et al., 2023; Gilardi et al., 2023). Entity-level framing has been explored to characterize latent personas (Card et al., 2016) and analyze the portrayal of social actors (Ziems and Yang, 2021; Roy and Goldwasser, 2023; Masini and Van Aelst, 2017).

Semantic frames are linguistically more interpretable and directly tied to concrete linguistic schemes that involve lexical units and semantic roles. These frames can influence the way information is conveyed; for example, in reporting a femicide, the choice of dead over murdered can shift responsibility onto the victim and obscure agency. A few multilingual detection tools exist in NLP, including LOME (Xia et al., 2021) and SocioFillmore (Minnema et al., 2022b).

## 3.8 Topics

The choice of which topics to present contributes to perspective, because it is inherently linked to selection and framing bias (Rodrigo-Ginés et al., 2024). Besides this, topic modeling has been used for perspective detection, leveraging unsupervised methods such as LDA to uncover latent semantic structures (Lin et al., 2008; Ahmed and Xing, 2010; Nguyen et al., 2013; Tsur et al., 2015; Ajjour et al., 2019). Here, a perspective is seen as an aggregation of documents or text segments that share topical distributions. This approach is conceptually related to argument clustering (cf. §3.6), but relies primarily on lexical patterns rather than argumentative structures and fine-grained discourse signals.

Perspective Signals The ideological dimension is obtained from the topic representation as a latent variable: texts are assigned one weight for their topic and another one for their ideology, so that the latter can be isolated. This process is used to find political ideologies (Lin et al., 2008; Ahmed and Xing, 2010; Vilares and He, 2017; Nguyen et al., 2013; Roberts et al., 2014) and frames (DiMaggio et al., 2013; Tsur et al., 2015; Ajjour et al., 2019) and can also enhance opinion mining (Draws et al., 2020). Despite their decent performance, topic models risk oversimplifying perspective by focusing too heavily on lexical distributions rather than higher-level linguistic features. The alternative is combining them with semantic and discourse features to capture viewpoints more holistically (Carlebach et al., 2020) (cf. §3.6).

## 3.9 Interactions Among Concepts

This section summarizes how the concepts can be combined for perspective detection and analysis, as a theoretically motivated account open to future empirical investigation. Figure 2 provides two examples annotated with all the discussed concepts.

Political ideology is the prominent conceptualization of perspective in NLP (Pang et al., 2008). It is an overarching belief system rooted in a shared system of values that aggregates multiple finegrained viewpoints on entities, topics, and issues (§3.2). For example, a CONSERVATIVE ideology will be the sum of specific positions on various topics (e.g., AGAINST migration, AGAINST public health, etc.). These positions, intended as stances, sentiments or opinions, can serve as proxies for ideology detection (Grefenstette et al., 2004; Lin et al., 2006; van Son et al., 2014; Bhatia and Deepak, 2018; Zhang et al., 2022). In contrast to ideology, these concepts are tied to specific situations and are variable across time and topics; e.g., a structured opinion will have a specific time, location, holder, and be linked to a specific event (cf. §3.5). Such fine-grained beliefs are encoded in argumentation, making this dimension the core nucleus of perspective and a concrete level where perspective can be identified. For example, to fully understand why someone has certain thoughts towards migration (opinion), a certain position towards the topic (stance), and a certain affective attitude (sentiment), we must consider the reasons deeply encoded in argumentation. Accordingly, some studies propose to find perspectives by clustering similar arguments and corresponding opinions (cf. §3.6).

![](images/52931836b46d3a9a19f7a03b0c98f9c72a6eff094e928b269ca12c4d9384fd6c.jpg)  
Figure 2: Examples illustrating all perspective-related concepts across two texts.

In parallel, the choice of what information to present and how to present it is a good proxy for perspective. Media frames present a text under a particular light, and semantic frames induce perspectivization. Both can be combined with the abstract beliefs discussed above to represent perspectives holistically, aggregating the ideological belief and their concrete realizations in language and cognition (Card et al., 2015; Alashri et al., 2015; Mendelsohn et al., 2021; Tsur et al., 2015; Field et al., 2018; Card et al., 2015; Morstatter et al., 2018; Draws et al., 2022; Blokker et al., 2022).

## 4 Structuring the Space of Perspective-related Concepts

In this section, we carry out an analysis to investigate the organization of the perspective-related concepts (RQ2). We aim at inducing a propertydriven structure over these concepts from expert judgments. To do so, we first manually annotate them with a set of properties, then cluster the resulting distributions by similarity, and finally construct a hierarchy based on the clusters.

## 4.1 Annotating Conceptual Properties

We first annotate each concept along four properties that we characterize during the literature review. The definitional properties discussed in §3, such as polar and affective, summarized in Table 2 for opinion-related terms, and later formalized in the decision tree (Figure 5), are binary and conceptspecific: they apply only to subsets of concepts and are not gradable, making them unsuitable for a comparison across all concepts. Therefore, we inductively derive four additional dimensions from the literature in $\ S 3$ that are both universal (applicable to every concept in the survey) and gradient (varying continuously across concepts). These properties are: (i) strength of linguistic cues: how strongly the concept is associated with specific linguistic elements; (ii) granularity (scope): the typical scope or localization of the concept within a text; (iii) entity-specificity: how strongly the concept is tied to specific entities; (iv) number ofdiscrete classes: in classification, how many classes are used. This property annotation is performed independently by the three authors in their capacity as experts for the literature discussed above. We acknowledge the limitations of having 3 annotators only, but we judge it to be the best choice given the gained familiarity with the literature, and therefore, the required expertise to perform the type of annotations reliably. Based on a codebook (cf. Appendix B), they locate concepts on a Likert scale from 1 to 5 with respect to each property.

<table><tr><td>Concept</td><td>Ling. cues</td><td>Granularity</td><td>Entity-spec.</td><td>Disc. classes</td><td>Cluster</td><td></td></tr><tr><td>Values</td><td> $1 . 3 3 \pm \mathrm { 0 . 5 8 }$ </td><td> $1 . 0 0 _ { \pm 0 . 0 0 }$ </td><td> $1 . 0 0 _ { \pm 0 . 0 0 }$ </td><td> $3 . 0 0 _ { \pm 1 . 7 3 }$ </td><td>1</td><td rowspan="4">Val. &amp; ideology</td></tr><tr><td>Morals</td><td> $2 . 0 0 \pm 1 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 3 3 \pm 0 . 5 8$ </td><td> $3 . 0 0 \pm 1 . 7 3$ </td><td>1</td></tr><tr><td>Political ideology</td><td> $3 . 0 0 \pm 1 . 0 0$ </td><td> $1 . 6 7 \pm 0 . 5 8$ </td><td> $3 . 0 0 \pm 2 . 0 0$ </td><td> $2 . 0 0 \pm 1 . 7 3$ </td><td>1</td></tr><tr><td>Ideology bias</td><td> $1 . 6 7 \pm 0 . 5 8$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 0 0 \pm 1 . 7 3$ </td><td> $1 . 0 0 _ { \pm 0 . 0 0 }$ </td><td>1</td></tr><tr><td>Political leaning</td><td> $2 . 3 3 \pm 0 . 5 8$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 3 3 \pm 1 . 5 3$ </td><td> $1 . 3 3 \pm 0 . 5 8$ </td><td>1</td></tr><tr><td>Sentiment</td><td> $3 . 3 3 \pm 0 . 5 8$ </td><td> $3 . 3 3 \pm 1 . 1 5$ </td><td> $3 . 3 3 \pm 0 . 5 8$ </td><td> $1 . 3 3 \pm 0 . 5 8$ </td><td>2</td><td rowspan="4">Sent. &amp; stances</td></tr><tr><td>Polarity</td><td> $3 . 6 7 \pm 0 . 5 8$ </td><td> $3 . 3 3 \pm 1 . 1 5$ </td><td> $3 . 3 3 \pm 1 . 1 5$ </td><td> $1 . 0 0 _ { \pm 0 . 0 0 }$ </td><td>2</td></tr><tr><td>Stances</td><td> $2 . 6 7 _ { \pm 1 . 5 3 }$ </td><td> $3 . 0 0 \pm 2 . 0 0$ </td><td> $3 . 6 7 _ { \pm 1 . 5 3 }$ </td><td> $1 . 0 0 _ { \pm 0 . 0 0 }$ </td><td></td></tr><tr><td>Emotions</td><td> $3 . 3 3 \pm 0 . 5 8$ </td><td> $3 . 0 0 \pm 1 . 7 3$ </td><td> $3 . 6 7 _ { \pm 1 . 5 3 }$ </td><td> $2 . 6 7 _ { \pm 0 . 5 8 }$ </td><td>22</td></tr><tr><td>Media frames</td><td> $2 . 6 7 _ { \pm 0 . 5 8 }$ </td><td> $3 . 0 0 \pm 1 . 0 0$ </td><td> $2 . 0 0 \pm 1 . 0 0$ </td><td> $3 . 6 7 _ { \pm 0 . 5 8 }$ </td><td></td><td rowspan="2">Topics &amp; MF</td></tr><tr><td>Topics</td><td> $3 . 6 7 \pm 0 . 5 8$ </td><td> $2 . 0 0 \pm 1 . 0 0$ </td><td> $2 . 3 3 \pm 1 . 5 3$ </td><td> $4 . 6 7 \pm 0 . 5 8$ </td><td>33</td></tr><tr><td>Arguments</td><td> $2 . 3 3 \pm 1 . 1 5$ </td><td> $4 . 3 3 \pm 0 . 5 8$ </td><td> $4 . 0 0 \pm 0 . 0 0$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td>4</td><td rowspan="4"></td></tr><tr><td>Opinions</td><td> $2 . 0 0 _ { \pm 1 . 0 0 }$ </td><td> $4 . 0 0 _ { \pm 0 . 0 0 }$ </td><td> $4 . 0 0 _ { \pm 0 . 0 0 }$ </td><td> $5 . 0 0 _ { \pm 0 . 0 0 }$ </td><td>4</td></tr><tr><td>Claims</td><td> $3 . 6 7 _ { \pm 2 . 3 1 }$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 3 3 \pm 0 . 5 8$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td></td></tr><tr><td>Semantic frames</td><td> $4 . 6 7 \pm 0 . 5 8$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 2 . 0 0$ </td><td> $4 . 6 7 \pm 0 . 5 8$ </td><td>44</td></tr><tr><td>Average IRR (ρ)</td><td>0.31</td><td>0.77</td><td>0.26</td><td>0.80</td><td></td><td></td></tr></table>

Table 3: Per-concept scores for four properties: (i) strength of linguistic cues, (ii) granularity, (iii) entity-specificity, and (iv) number of discrete classes. $\mathrm { M e a n } \pm \mathrm { s t d }$ . dev. across annotators is reported (higher=red, lower=green). IRR: inter-rater reliability (Spearman’s ρ).

Results Table 3 reports the resulting score averages over the three annotators. The colors indicate the standard deviation (higher=red, lower=green). We report average Spearman ρ for each property as a measure of inter-rater reliability (IRR).

The property with the highest IRR is number of discrete classes $( \rho = 0 . 8 0 )$ , followed by granularity $( \rho = 0 . 7 7 )$ . Strength of linguistic cues and entityspecificity have significantly lower agreement $( \rho =$ 0.31 and 0.26). Arguments and claims are difficult to agree on in strength of linguistic cues because they are not systematically associated with certain recurrent linguistic cues, but at the same time they are identified with the linguistic expression itself. Stances in turn can also be ambiguous because they can be interpreted either as the ideological position toward an issue or as its concrete instantiation (similar to opinions). Entity-specificity raises problems for concepts that can optionally refer to specific entities or not; for example, semantic frames involve entities and their roles, but they may or may not be necessary for annotation and detection. In sum, we take our result to be sufficiently robust for a clustering analysis.

## 4.2 Clustering Perspective Concepts

To group the concepts in Table 3, we apply agglomerative hierarchical clustering on annotatoraggregated scores, specifically average linkage with Euclidean distance (Nielsen, 2016). Average linkage defines the distance between two clusters as the mean of all pairwise distances between their members, and iteratively merges the closest pair. This results in four compact groups, as shown in Figure 3.

We refer to them as follows: values and ideology, the most abstract and global level, comprising overarching belief systems (§3.1) and political ideologies (§3.2); sentiment and stances, including specific beliefs that express personal positions (§3.4, §3.3); topics and media frames, including topical dimensions (§3.7, §3.8); argumentation, including arguments and claims (§3.6), semantic frames (§3.7), and opinions (§3.5), all identifiable with linguistic structures.

![](images/0dc6dc6f1a4a2f1a0fa9af6b01f60a06a6bac29881fb57b018ab1c906441e12f.jpg)

Figure 3: Dendrogram of concepts clusters obtained with hierarchical clustering. The dashed line marks the four-cluster cut.  
![](images/24b395954e9728756da577390457d01064df62cfda45091132e74e51e76670a5.jpg)  
Figure 4: Average scores for each conceptual cluster on the four properties.

The spider chart in Figure 4 shows the average scores for the clusters on each property. The chart demonstrates the properties we consider are strongly correlated at the cluster level: the ranking of the clusters regarding the different properties agrees almost perfectly. The only exception is that the sentiment and stances cluster shows a lower class count than expected. This is presumably the case because the cluster includes concepts whose interpretation varies depending on their theoretical definition and operationalization, notably emotion (Scarantino, 2016). However, we consider this variation is minor for our ordering purposes.

## 4.3 A Model of Perspective

Given the strong correlation between the four properties which we see in Figure 4, we investigate whether the perspective-related concepts can be reduced to a single axis by running a Principal Component Analysis (PCA) of the annotations (cf. results in Appendix B.3). We find that this is largely the case: the first principal component (PC1) explains 62% of the variance and shows a positive loading with each of the properties. When we represent all concepts purely in terms of their value on the dimension formed by PC1, we recover the four clusters almost perfectly, with the only outlier a swap between topics and stances (cf. Figure 7). This likely happens because the two concepts received mid-point scores for all properties, apart from the class number: when giving more importance to such property, they get pulled apart (cf. PC2 in Figure 6).

This result supports our interpretation that there is a latent linear ordering underlying the concepts. The axis identified by PC1 captures both linguistic and conceptual aspects, and can be interpreted as a dimension of (generic) specificity. Figure 1 is informed by this analysis and shows our model: the space is represented as a set of concentric circles, ranging from ideological beliefs (outer) to linguistic instantiations (inner). At one end, we find generic concepts, such as ideology, which are not bound to specific situations, but underlie other finegrained perspectives; they emerge throughout the document mainly with lexical cues, and map onto a few labels. At the other end, we find argumentationrelated concepts, which are instead more specific both in what they express, i.e., precise situations and entities, and in how they are expressed: well localized in language spans, signalled by semantic and syntactic patterns, and mapping to a wide or open-ended class range.

Additional Factors So far, we have not considered information about the writer, annotator, or media source, even though they are indicators of perspective (Frenda et al., 2024). We include them in a separate box, since they describe the (extralinguistic) context of the text rather than its content. Indeed, metadata may not match the perspectives expressed in the text (Baly et al., 2020), and it is important to distinguish between grouping emerging from texts and groupings based on external data (Vitsakis et al., 2024). We identify three types of extra-textual factors, based on the perspective holder: (i) the author’s characteristics, (ii) the annotator’s characteristics and (iii) the media source. These include socio-demographic, political and cultural background (e.g., political orientation, gender, country, social affiliations, editorial stance), as well as annotation-related information (e.g., IAA).

Previous Hierarchies Some other works in NLP aim at organizing the conceptual space of perspective. Doan and Gulla (2022) divide the methods for perspective detection into: (i) political ideologies/leaning/party detection, (ii) political stance/framing detection, and (iii) political viewpoint extraction. Klebanov et al. (2010) distinguish four levels of perspective, from less to more abstract: (i) opinions, (ii) stances on specific issues, (iii) ideological positions, and (iv) demographic factors and life of the author (e.g., place of birth, religion, culture, political tradition). Most similar to our proposal is the hierarchy by Van Der Meer (2024), comprising three levels of abstraction: (i) stances, (ii) arguments, and (iii) values.

Our model makes a number of contributions: (i) we include subjectivity-related concepts that have traditionally been treated separately (Pang et al., 2008); (ii) we include the linguistic level, following the idea that perspectives can be identified through arguments (Van Der Meer, 2024) and semantic patterns (Minnema et al., 2022b); (iii) we annotate conceptual properties, providing an empirical support for the hierarchy; (iv) we distinguish content from context-related factors.

## 5 Discussion

The goal of our study was to clarify and structure the space of concepts relevant for research on perspectives in NLP. We ask two research questions: RQ1, what concepts are used in perspective identification. We identify 15 concepts and characterize both their definition and operationalization based on a literature analysis (§3, cf. Table 5). In RQ2, we ask how these concepts are related. The analysis of our expert annotations of four properties found that perspective-related concepts can be organized along a single dimension of linguistic and conceptual specificity that captures most of the variance between the concepts (§4, cf. Figure 1).

## 5.1 Outlook: Actionability

Figure 1 orders them in terms of specificity but does not provide guidance for choosing which one(s) to use in a hypothetical application. In Figure 5, we present a decision tree that leverages the features from §3 to guide concept selection. The following scenarios illustrate how the tree can be used.

Scenario 1 I am studying a corpus of news that is fully topic- and issue-agnostic. I want to detect political perspectives for building a diverse news recommender. Following the decision tree, I decide to consider perspectives that emerge from the text. I aim to capture generic beliefs (emerges from the text > denotes a mental state or belief), without identifying explicit targets or relying on affective features, as the dataset is unstructured and mostly comprises factual news. The proposed operationalization is political ideologies. If I want to derive perspectives bottom-up from concrete language patterns (... > denotes a mental state or belief = NO > is about information selection = NO), I could start from opinion mining: as discussed, perspectives can be inferred by aggregating minimal positions on smaller topics (§3.6). For diversification purposes, these opinions should then be reduced to a small number of meaningful clusters or categories.

Scenario 2 I am analyzing a corpus of news articles from different outlets covering the same event, with the goal of comparing how it is presented across sources. In this case, I have more flexibility, as I am not constrained by a fixed topic or application, and I do not need to cluster articles. My focus is on how information is organized and empathized rather than on what content is conveyed. Following the tree (... > is about information selection > involves emphasis), the proposed device is frames. In case I care about linguistic patterns and event structures, I could work with semantic frames.

Scenario 3 I am building a politically-aligned LLM-based persona. I could leverage metadata about authors’ demographics from a corpus to guide the alignment (emergesfrom the text = NO). Otherwise, the persona can be aligned with a broader political leaning which emerges from a consistent pattern of opinions across multiple issues. If my analysis is more granular, I could control for stances toward specific issues or targets (... > requires a target) (e.g., PRO migration, PRO samesex marriage, AGAINST gun control). If I care about the affective tone (... > is affective), I may consider sentiment or emotions as complementary dimensions.

## 5.2 Future Research Directions

Newspapers make editorial decisions at multiple levels of perspective, including how to frame events, which arguments to use, what topics to cover, and which stances to adopt. Despite lacking such deliberate mechanisms, LLMs convey perspectives emerging from training data in a comparable way. These viewpoints, embedded in textual choices, often go unnoticed by readers. We claim that detecting, controlling, and communicating these layers with transparency is worth-while to support people’s access to information and promote critical engagement.

![](images/999f41a71b54ffd1df24755aab8f7fe2942cbaf333eb483a93f8c938da776e7a.jpg)  
Figure 5: Decision tree for choosing what perspective concept(s) to adopt [is shared]. = yes, = no. Discriminative characteristics are marked in bold in the literature review (§3)

By surveying perspective concepts, we have shown how analyzing argument structures jointly with information selection and presentation can map specific opinions to beliefs at different levels of granularity, up to ideology and values, while framing can reveal hidden over-emphasizing signals. Exploring how these levels can be integrated into a coherent representation is a promising research direction in support of critical social analysis. Possible outcomes of this paper include a comprehensive annotation scheme, a modeling recipe, or an evaluation protocol for perspectives in text.

On a more operational level, our conceptual hierarchy also carries direct implications for how we evaluate and audit language models. Rather than treating perspective bias as a monolithic property, the specificity axis offers a diagnostic lens: bias in LLMs may manifest differently at different levels, from systematic skews in ideological framing that pervade entire outputs, to more localized choices in argumentation structure or semantic framing that subtly shift responsibility or salience. Benchmarks for perspective diversity in generated text could be designed to probe each level independently, yielding a richer picture of where training data or alignment procedures introduce distortions.

At the same time, the normative framing underlying much of this work – that greater perspective diversity is inherently desirable – deserves scrutiny. Diversity of perspectives is a meaningful democratic value when it reflects the genuine range of informed viewpoints on a contested issue; it becomes problematic when operationalized in ways that treat fringe or harmful positions as simply another point on a spectrum to be represented. Our hierarchy may help draw this distinction more precisely: diversity at the level of values and ideology calls for different normative criteria than diversity at the level of claims or arguments, where factual accuracy and logical coherence impose additional constraints beyond mere representational balance. Navigating this tension – between pluralism and epistemic responsibility – is as important as the development of methods to evaluate generated text.

## References

Marwa Abdulhai, Gregory Serapio-Garcia, Clé- ment Crepy, Daria Valter, John Canny, and Natasha Jaques. 2024. Moral foundations of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17737–17752.

Amjad Abu-Jbara, Pradeep Dasigi, Mona Diab, and Dragomir Radev. 2012. Subgroup detection in ideological discussions. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 399–409.

Amjad Abu-Jbara, Ben King, Mona Diab, and Dragomir Radev. 2013. Identifying opinion subgroups in arabic online discussions. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 829–835.

Amr Ahmed and Eric Xing. 2010. Staying informed: supervised and semi-supervised multiview topical analysis of ideological perspective. In Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing, pages 1140–1150.

Yamen Ajjour, Milad Alshomary, Henning Wachsmuth, and Benno Stein. 2019. Modeling frames in argumentation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2922–2932.

Khalid Al Khatib, Hinrich Schütze, and Cathleen Kantner. 2012. Automatic detection of point of view differences in Wikipedia. In Proceedings of COLING 2012, pages 33–50.

Saud Alashri, Sultan Alzahrani, Lenka Bustikova, David Siroky, and Hasan Davulcu. 2015. What animates political debates? analyzing ideological perspectives in online debates between opposing parties. In Proceedings of the ASE/IEEE International Conference on Social Computing (SocialCom-15), Stanford, CA. Academy of Science and Engineering.

Rubayyi Alghamdi and Khalid Alfalqi. 2015. A survey of topic modeling in text mining. Inter-

national Journal ofAdvanced Computer Science and Applications, 6(1).

Mohammad Ali and Naeemul Hassan. 2022. A survey of computational framing analysis approaches. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9335–9348.

Khudran M Alzhrani. 2022. Political ideology detection of news articles using deep neural networks. Intelligent Automation & Soft Computing, 33(1).

Pranav Anand, Marilyn Walker, Rob Abbott, Jean E Fox Tree, Robeson Bowmani, and Michael Minor. 2011. Cats rule and dogs drool!: Classifying stance in online debate. In Proceedings of the 2nd Workshop on Computational Approaches to Subjectivity and Sentiment Analysis (WASSA 2.011), pages 1–9.

Lora Aroyo and Chris Welty. 2015. Truth is a lie: Crowd truth and the seven myths of human annotation. AI Magazine, 36(1):15–24.

Katie Atkinson and Trevor Bench-Capon. 2021. Value-based argumentation. Journal of Applied Logics, 8(6):1543–1588.

Christian Baden and Nina Springer. 2017. Conceptualizing viewpoint diversity in news discourse. Journalism, 18(2):176–194.

Collin F Baker, Charles J Fillmore, and John B Lowe. 1998. The Berkeley FrameNet project. In COLING 1998 Volume 1: The 17th International Conference on Computational Linguistics.

Razan Baltaji, Babak Hemmatian, and Lav Varshney. 2024. Conformity, confabulation, and impersonation: Persona inconstancy in multi-agent LLM collaboration. In Proceedings ofthe 2nd Workshop on Cross-Cultural Considerations in NLP, pages 17–31, Bangkok, Thailand. Association for Computational Linguistics.

Ramy Baly, Giovanni Da San Martino, James Glass, and Preslav Nakov. 2020. We can detect your bias: Predicting the political ideology of news articles. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4982–4991, Online. Association for Computational Linguistics.

Roy Bar-Haim, Indrajit Bhattacharya, Francesco Dinuzzo, Amrita Saha, and Noam Slonim. 2017. Stance classification of context-dependent claims. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 251–261.

Emma Barker, Monica Paramita, Adam Funk, Emina Kurtic, Ahmet Aker, Jonathan Foster, Mark Hepple, and Robert Gaizauskas. 2016. What’s the issue here?: Task-based evaluation of reader comment summarization systems. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 3094–3101, Portorož, Slovenia. European Language Resources Association (ELRA).

Valerio Basile, Tommaso Caselli, Alexandra Balahur, and Lun-Wei Ku. 2022. Bias, subjectivity and perspectives in natural language processing. Frontiers in Artificial Intelligence, 5:926435.

Eric Baumer, Elisha Elovic, Ying Qin, Francesca Polletta, and Geri Gay. 2015. Testing and comparing computational approaches for identifying the language of framing in political news. In Proceedings of the 2015 conference of the North American chapter of the Association for Computational Linguistics: human language technologies, pages 1472–1482.

Farah Benamara, Maite Taboada, and Yannick Mathieu. 2017. Evaluative language beyond bags of words: Linguistic insights and computational applications. Computational Linguistics, 43(1):201–264.

Noam Benkler, Drisana Mosaphir, Scott Friedman, Andrew Smart, and Sonja Schmer-Galunder. 2023. Assessing LLMs for moral value pluralism. In Proceedings of the NeurIPS workshop ’AI meets Moral Philosophy and Moral Psychology’.

Rodney Benson. 2009. What makes news more multiperspectival? A field analysis. Poetics, 37(5-6):402–418.

Sumit Bhatia and P Deepak. 2018. Topic-specific sentiment analysis can help identify political ideology. In Proceedings of the 9th workshop on computational approaches to subjectivity, sentiment and social media analysis, pages 79–84.

Vibhu Bhatia, Vidya Prasad Akavoor, Sejin Paik, Lei Guo, Mona Jalal, Alyssa Smith, David Assefa Tofu, Edward Edberg Halim, Yimeng Sun, Margrit Betke, Prakash Ishwar, and Derry Tanti Wijaya. 2021. OpenFraming: Open-sourced tool for computational framing analysis of multilingual data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 242– 250, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Nico Blokker, Tanise Ceron, André Blessing, Erenay Dayanik, Sebastian Haunss, Jonas Kuhn, Gabriella Lapesa, and Sebastian Padó. 2022. Why justifications of claims matter for understanding party positions. In Proceedings of the 2nd Workshop on Computational Linguisticsfor Political Text Analysis, Potsdam, Germany.

Katarina Boland, Pavlos Fafalios, Andon Tchechmedjiev, Stefan Dietze, and Konstantin Todorov. 2022. Beyond facts–a survey and conceptualisation of claims in online discourse analysis. Semantic Web, 13(5):793–827.

Mihaela Bošnjak and Mladen Karan. 2019. Data set for stance and sentiment analysis from user comments on Croatian news. In Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing, pages 50–55, Florence, Italy. Association for Computational Linguistics.

Amber E Boydstun, Justin H Gross, Philip Resnik, and Noah A Smith. 2013. Identifying media frames and frame dynamics within and across policy issues. In New Directions in Analyzing Text as Data Workshop, London, pages 27–28.

Ceren Budak, Sharad Goel, and Justin M Rao. 2016. Fair and balanced? quantifying media bias through crowdsourced content analysis. Public Opinion Quarterly, 80(S1):250–271.

Elena Cabrio and Serena Villata. 2018. Five years of argument mining: A data-driven analysis. In IJCAI, volume 18, pages 5427–5433.

Jaime Guillermo Carbonell. 1979. Subjective Understanding: Computer Models of Belief Systems. Ph.D. thesis, Yale University.

Dallas Card, Amber Boydstun, Justin H Gross, Philip Resnik, and Noah A Smith. 2015. The

media frames corpus: Annotations of frames across issues. In Proceedings ofthe 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 438–444.

Dallas Card, Justin Gross, Amber Boydstun, and Noah A. Smith. 2016. Analyzing framing through the casts of characters in the news. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, pages 1410–1420, Austin, Texas. Association for Computational Linguistics.

Mark Carlebach, Ria Cheruvu, Brandon Walker, Cesar Ilharco Magalhaes, and Sylvain Jaume. 2020. News aggregation with diverse viewpoint identification using neural embeddings and semantic understanding models. In Proceedings of the 7th Workshop on Argument Mining, pages 59–66.

Tanise Ceron, Nico Blokker, and Sebastian Padó. 2022. Optimizing text representations to capture (dis)similarity between political parties. In Proceedings of the 26th Conference on Computational Natural Language Learning (CoNLL), pages 325–338, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Tanise Ceron, Neele Falk, Ana Baric, Dmitry Niko-´ laev, and Sebastian Padó. 2024. Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in llms. Transactions of the Association for Computational Linguistics, 12:1378–1400.

Tanise Ceron, Dmitry Nikolaev, Dominik Stammbach, and Debora Nozza. 2025. What is the political content in LLMs’ pre-and post-training data? arXiv preprint arXiv:2509.22367.

Harris Chan, Jamie Kiros, and William Chan. 2020. Multichannel Generative Language Model: Learning All Possible Factorizations Within and Across Channels. In Findings of the Associationfor Computational Linguistics: EMNLP 2020, pages 4208–4220, Online. Association for Computational Linguistics.

Yi-Ting Chang, Yun-Zhu Song, Yi-Syuan Chen, and Hong-Han Shuai. 2023. Beyond detection:

A defend-and-summarize strategy for robust and interpretable rumor analysis on social media. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 11538–11556, Singapore. Association for Computational Linguistics.

Iti Chaturvedi, Erik Cambria, Roy E Welsch, and Francisco Herrera. 2018. Distinguishing between facts and opinions for sentiment analysis: Survey and challenges. Information Fusion, 44:65–77.

Hung-Ting Chen and Eunsol Choi. 2025. Openworld evaluation for retrieving diverse perspectives. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8508–8528.

Sihao Chen, Daniel Khashabi, Wenpeng Yin, Chris Callison-Burch, and Dan Roth. 2019. Seeing things from a different angle: Discovering diverse perspectives about claims. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 542–557.

Wei Chen, Xiao Zhang, Tengjiao Wang, Bishan Yang, and Yi Li. 2017. Opinion-aware knowledge graph for political ideology detection. In Proceedings ofIJCAI, volume 17, pages 3647– 3653.

Wei-Fan Chen, Khalid Al Khatib, Benno Stein, and Henning Wachsmuth. 2021. Controlled neural sentence-level reframing of news articles. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2683–2693, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yoonjung Choi and Janyce Wiebe. 2014. +/- effectwordnet: Sense-level lexicon acquisition for opinion inference. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1181– 1191.

Giovanni Da San Martino, Firoj Alam, Maram Hasanain, Rabindra Nath Nandi, Dilshod Azizov, and Preslav Nakov. 2023. Overview of the

CLEF-2023 CheckThat! lab task 3 on political bias of news articles and news media. In CLEF (Working Notes), pages 250–259.

Agnese Daffara, Sourabh Dattawad, Sebastian Padó, and Tanise Ceron. 2025. Generalizability of media frames: Corpus creation and analysis across countries. Proceedings ofthe 14th Joint Conference on Lexical and Computational Semantics (\*SEM 2025), page 83–99.

Rohan Das, Aditya Chandra, I-Ta Lee, and Maria Leonor Pacheco. 2024. Media framing through the lens of event-centric narratives. In Proceedings of the The 6th Workshop on Narrative Understanding, pages 85–98, Miami, Florida, USA. Association for Computational Linguistics.

Pradeep Dasigi, Weiwei Guo, and Mona Diab. 2012. Genre independent subgroup detection in online discussion threads: A study of implicit attitude using textual latent semantics. In Proceedings ofthe 50th Annual Meeting ofthe Association for Computational Linguistics (Volume 2: Short Papers), pages 65–69.

Claes H De Vreese. 2005. News framing: Theory and typology. Information design journal+ document design, 13(1):51–62.

Nicholas Deas and Kathleen McKeown. 2025. Summarization of opinionated political documents with varied perspectives. In Proceedings of the 31st International Conference on Computational Linguistics, pages 8088–8108, Abu Dhabi, UAE. Association for Computational Linguistics.

Lingjia Deng and Janyce Wiebe. 2015. Mpqa 3.0: An entity/event-level sentiment corpus. In Proceedings of the 2015 conference of the North American chapter of the association for computational linguistics: human language technologies, pages 1323–1328.

Paul DiMaggio, Manish Nag, and David Blei. 2013. Exploiting affinities between topic modeling and the sociological perspective on culture: Application to newspaper coverage of us government arts funding. Poetics, 41(6):570–606.

Tu My Doan and Jon Atle Gulla. 2022. A survey on political viewpoints identification. Online Social Networks and Media, 30:100208.

Tim Draws, Oana Inel, Nava Tintarev, Christian Baden, and Benjamin Timmermans. 2022. Comprehensive viewpoint representations for a deeper understanding of user interactions with debated topics. In Proceedings ofthe 2022 Conference on Human Information Interaction and Retrieval, pages 135–145.

Tim Draws, Jody Liu, and Nava Tintarev. 2020. Helping users discover perspectives: Enhancing opinion mining with joint topic models. In 2020 International Conference on Data Mining Workshops (ICDMW), pages 23–30. IEEE.

Tim Draws, Nava Tintarev, and Ujwal Gadiraju. 2021. Assessing viewpoint diversity in search results using ranking fairness metrics. ACM SIGKDD Explorations Newsletter, 23(1):50–58.

Esra Dönmez and Agnieszka Falenska. 2026.´ Structuring the space of sociotechnical alignment: A specification framework and systematic literature review. SocArXiv/3uafr\_v1.

Paul Ekman. 1992. An argument for basic emotions. Cognition and Emotion, 6(3/4):169–200.

Heba Elfardy, Mona Diab, and Chris Callison-Burch. 2015. Ideological perspective detection using semantic features. In Proceedings of the Fourth Joint Conference on Lexical and Computational Semantics, pages 137–146.

Robert M Entman. 1993. Framing: Toward clarification of a fractured paradigm. Journal of communication, 43(4):51–58.

Lisa Fan, Marshall White, Eva Sharma, Ruisi Su, Prafulla Kumar Choubey, Ruihong Huang, and Lu Wang. 2019. In plain sight: Media bias through the lens of factual reporting. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6343–6349.

Shangbin Feng, Zilong Chen, Wenqian Zhang, Qingyao Li, Qinghua Zheng, Xiaojun Chang, and Minnan Luo. 2021. Kgap: Knowledge graph augmented political perspective detection in news media. arXiv:2108.03861.

Anjalie Field, Doron Kliger, Shuly Wintner, Jennifer Pan, Dan Jurafsky, and Yulia Tsvetkov.

2018. Framing and agenda-setting in russian news: a computational analysis of intricate political strategies. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3570–3580.

Charles J Fillmore. 1976. Frame semantics and the nature of language. Annals of the New York Academy ofSciences, 280(1):20–32.

Simona Frenda, Gavin Abercrombie, Valerio Basile, Alessandro Pedrani, Raffaella Panizzon, Alessandra Teresa Cignarella, Cristina Marco, and Davide Bernardi. 2024. Perspectivist approaches to natural language processing: a survey. Language Resources and Evaluation, pages 1–28.

Simona Frenda, Alessandro Pedrani, Valerio Basile, Soda Marem Lo, Alessandra Teresa Cignarella, Raffaella Panizzon, Cristina Marco, Bianca Scarlini, Viviana Patti, Cristina Bosco, and Davide Bernardi. 2023. EPIC: Multi-perspective annotation of a corpus of irony. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13844–13857, Toronto, Canada. Association for Computational Linguistics.

Rama Rohit Reddy Gangula, Suma Reddy Duggenpudi, and Radhika Mamidi. 2019a. Detecting political bias in news articles using headline attention. In Proceedings of the 2019 ACL workshop BlackboxNLP: analyzing and interpreting neural networks for NLP, pages 77–84.

Rama Rohit Reddy Gangula, Suma Reddy Duggenpudi, and Radhika Mamidi. 2019b. Detecting political bias in news articles using headline attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 77–84, Florence, Italy. Association for Computational Linguistics.

Matthew Gentzkow, Jesse M Shapiro, and Matt Taddy. 2019. Measuring group differences in high-dimensional choices: method and application to congressional speech. Econometrica, 87(4):1307–1340.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences, 120(30):e2305016120.

Trudy Govier. 2005. A practical study of argument, 6th edition. Thomson Wadsworth.

Jesse Graham, Jonathan Haidt, and Brian A Nosek. 2009. Liberals and conservatives rely on different sets of moral foundations. Journal ofpersonality and social psychology, 96(5):1029.

Carl Friedrich Graumann and Werner Kallmeyer. 2008. Perspective and perspectivation in discourse: An introduction. In Perspective and perspectivation in discourse, pages 1–11. John Benjamins Publishing Company.

Stephan Greene and Philip Resnik. 2009. More than words: Syntactic packaging and implicit sentiment. In Proceedings ofhuman language technologies: The 2009 annual conference of the north american chapter of the association for computational linguistics, pages 503–511.

Gregory Grefenstette, Yan Qu, James G Shanahan, and David A Evans. 2004. Coupling niche browsers and affect analysis for an opinion mining application. Proceedings ofRecherche d’Information Assistée par Ordinateur (RIAO).

Omama Hamad, Khaled Shaban, and Ali Hamdi. 2024. ASEM: Enhancing empathy in chatbot through attention-based sentiment and emotion modeling. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 1588– 1601, Torino, Italia. ELRA and ICCL.

Jiyoung Han, Youngin Lee, Junbum Lee, and Meeyoung Cha. 2019. The fallacy of echo chambers: Analyzing the political slants of usergenerated news comments in Korean media. In Proceedings of the 5th Workshop on Noisy Usergenerated Text (W-NUT 2019), pages 370–374, Hong Kong, China. Association for Computational Linguistics.

Eric Hardisty, Jordan Boyd-Graber, and Philip Resnik. 2010. Modeling perspective using adaptor grammars. In Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing, pages 284–292.

Kazi Saidul Hasan and Vincent Ng. 2012. Predicting stance in ideological debate with rich linguistic knowledge. In Proceedings ofCOLING 2012: Posters, pages 451–460.

Shirley Anugrah Hayati, Minhwa Lee, Dheeraj Rajagopal, and Dongyeop Kang. 2024. How far can we extract diverse perspectives from large language models? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5336–5366, Miami, Florida, USA. Association for Computational Linguistics.

Natali Helberger. 2021. On the democratic role of news recommenders. In Algorithms, automation, and news, pages 14–33. Routledge.

Natali Helberger, Kari Karppinen, and Lucia D’acunto. 2018. Exposure diversity as a design principle for recommender systems. Information, communication & society, 21(2):191–207.

Zhe Hu, Hou Pong Chan, Jing Li, and Yu Yin. 2025. Debate-to-write: A persona-driven multiagent framework for diverse argument generation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 4689–4703, Abu Dhabi, UAE. Association for Computational Linguistics.

Nannan Huang, Lin Tian, Haytham Fayek, and Xiuzhen Zhang. 2023. Examining bias in opinion summarisation through the perspective of opinion diversity. In Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pages 149–161, Toronto, Canada. Association for Computational Linguistics.

Pere-Lluís Huguet Cabot, Verna Dankers, David Abadi, Agneta Fischer, and Ekaterina Shutova. 2020. The Pragmatics behind Politics: Modelling Metaphor, Framing and Emotion in Political Discourse. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4479–4488, Online. Association for Computational Linguistics.

Susan Hunston and Geoffrey Thompson. 2000. Evaluation in text: Authorial stance and the construction ofdiscourse. Oxford University Press, UK.

Ronald Inglehart, Miguel Basanez, Jaime Diez-Medrano, Loek Halman, and Ruud Luijkx. 2000. World values surveys and european values surveys, 1981-1984, 1990-1993, and 1995-1997. Ann Arbor-Michigan, Institute for Social Research, ICPSR version.

Mohit Iyyer, Peter Enns, Jordan Boyd-Graber, and Philip Resnik. 2014. Political ideology detection using recursive neural networks. In Proceedings of the 52nd annual meeting of the Association for Computational Linguistics (volume 1: long papers), pages 1113–1122.

Charlott Jakob, Pia Wenzel, Salar Mohtaj, and Vera Schmitt. 2024. Augmented political leaning detection: Leveraging parliamentary speeches for classifying news articles. In Proceedings of the 4th Workshop on Computational Linguistics for the Political and Social Sciences: Long and short papers, pages 126–133, Vienna, Austria. Association for Computational Linguistics.

Kristen Johnson and Dan Goldwasser. 2016. “all i know about politics is what i read in twitter”: Weakly supervised models for extracting politicians’ stances from twitter. In Proceedings of COLING 2016, the 26th international conference on computational linguistics: technical papers, pages 2966–2977.

Wan Ju Kang, Jiyoung Han, Jaemin Jung, and James Thorne. 2024. XFACT team0331 at PerspectiveArg2024: Sampling from bounded clusters for diverse relevant argument retrieval. In Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024), pages 182–188, Bangkok, Thailand. Association for Computational Linguistics.

Shima Khanehzar, Trevor Cohn, Gosia Mikolajczak, Andrew Turpin, and Lea Frermann. 2021. Framing unpacked: A semi-supervised interpretable multi-view model of media frames. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2154–2166.

Shima Khanehzar, Andrew Turpin, and Gosia Mikolajczak. 2019. Modeling political framing across policy issues and contexts. In Proceedings of The 17th Annual Workshop of the Australasian Language Technology Association, pages 61–66.

Johannes Kiesel, Milad Alshomary, Nicolas Handke, Xiaoni Cai, Henning Wachsmuth, and Benno Stein. 2022. Identifying the human values behind arguments. In Proceedings of the

60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4459–4471, Dublin, Ireland. Association for Computational Linguistics.

Johannes Kiesel, Maria Mestre, Rishabh Shukla, Emmanuel Vincent, Payam Adineh, David Corney, Benno Stein, and Martin Potthast. 2019. Semeval-2019 task 4: Hyperpartisan news detection. In Proceedings of the 13th International Workshop on Semantic Evaluation, pages 829– 839.

Scott F Kiesling, Umashanthi Pavalanathan, Jim Fitzpatrick, Xiaochuang Han, and Jacob Eisenstein. 2018. Interactional stancetaking in online forums. Computational Linguistics, 44(4):683– 718.

Kang-Min Kim, Mingyu Lee, Hyun-Sik Won, Min-Ji Kim, Yeachan Kim, and SangKeun Lee. 2023. Multi-stage prompt tuning for political perspective detection in low-resource settings. Applied Sciences, 13(10):6252.

Soo-Min Kim and Eduard Hovy. 2004. Determining the sentiment of opinions. In Proceedings of the 20th international conference on computational linguistics, pages 1367–1373, Geneva, Switzerland.

Beata Beigman Klebanov, Eyal Beigman, and Daniel Diermeier. 2010. Vocabulary choice as an indicator of perspective. In Proceedings of the ACL 2010 conference short papers, pages 253–257.

Dilek Küçük and Fazli Can. 2020. Stance detection: A survey. ACM Computing Surveys (CSUR), 53(1):1–37.

Haewoon Kwak, Jisun An, and Yong-Yeol Ahn. 2020. A systematic media frame analysis of 1.5 million new york times articles from 2000 to 2017. In Proceedings of the 12th ACM Conference on Web Science, pages 305–314.

Anne Lauscher, Henning Wachsmuth, Iryna Gurevych, and Goran Glavaš. 2022. Scientia potentia est—on the role of knowledge in computational argumentation. Transactions ofthe Associationfor Computational Linguistics, 10:1392– 1422.

Michael Laver, Kenneth Benoit, and John Garry. 2003. Extracting policy positions from political texts using words as data. American political science review, 97(2):311–331.

Sophie Lecheler, Mario Keer, Andreas RT Schuck, and Regula Hänggli. 2015. The effects of repetitive news framing on political opinions over time. Communication Monographs, 82(3):339–358.

Chang Li and Dan Goldwasser. 2019. Encoding social information with graph convolutional networks forPolitical perspective detection in news media. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 2594–2604, Florence, Italy. Association for Computational Linguistics.

Chang Li and Dan Goldwasser. 2021. MEAN: Multi-head entity aware attention networkfor political perspective detection in news media. In Proceedings of the Fourth Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda, pages 66–75, Online. Association for Computational Linguistics.

Luyang Lin, Lingzhi Wang, Jinsong Guo, and Kam-Fai Wong. 2025. Investigating bias in llm-based bias detection: Disparities between llms and human perception. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10634–10649.

Wei-Hao Lin, Theresa Wilson, Janyce Wiebe, and Alexander G Hauptmann. 2006. Which side are you on? identifying perspectives at the document and sentence levels. In Proceedings of the Tenth Conference on Computational Natural Language Learning (CoNLL-X), pages 109–116.

Wei-Hao Lin, Eric Xing, and Alexander Hauptmann. 2008. A joint topic and perspective model for ideological discourse. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2008, Antwerp, Belgium, September 15-19, 2008, Proceedings, Part II 19, pages 17–32. Springer.

Bing Liu, Minqing Hu, and Junsheng Cheng. 2005. Opinion observer: analyzing and comparing opinions on the web. In Proceedings of the 14th international conference on World Wide Web, pages 342–351.

Bing Liu and Lei Zhang. 2012. A survey of opinion mining and sentiment analysis. In Mining text data, pages 415–463. Springer.

Bing Liu et al. 2010. Sentiment analysis and subjectivity. Handbook of natural language processing, 2(2010):627–666.

Siyi Liu, Sihao Chen, Xander Uyttendaele, and Dan Roth. 2021. MultiOpEd: A corpus of multi-perspective news editorials. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4345–4361, Online. Association for Computational Linguistics.

Siyi Liu, Lei Guo, Kate Mays, Margrit Betke, and Derry Tanti Wijaya. 2019. Detecting frames in news headlines and its application to analyzing news framing trends surrounding U.S. gun violence. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 504–514, Hong Kong, China. Association for Computational Linguistics.

Yuhan Liu, Shangbin Feng, Xiaochuang Han, Vidhisha Balachandran, Chan Young Park, Sachin Kumar, and Yulia Tsvetkov. 2024. P<sup>3</sup>Sum: Preserving author‘s perspective in news summarization with diffusion language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2154–2173, Mexico City, Mexico. Association for Computational Linguistics.

Felicia Loecherbach, Judith Moeller, Damian Trilling, and Wouter van Atteveldt. 2020. The unified framework of media diversity: A systematic literature review. Digital Journalism, 8(5):605–642.

Matej Martinc, Nina Perger, Andraž Pelicon, Matej Ulcar, Andreja Vezovnik, and Senja Pollak. 2021.ˇ EMBEDDIA hackathon report: Automatic sentiment and viewpoint analysis of Slovenian news corpus on the topic of LGBTIQ+. In Proceedings of the EACL Hackashop on News Media Content Analysis and Automated Report Generation, pages 121–126, Online. Association for Computational Linguistics.

Manuel Nunez Martinez, Sonja Schmer-Galunder, Zoey Liu, Sangpil Youm, Chathuri Jayaweera, and Bonnie J. Dorr. 2024. Balancing transparency and accuracy: A comparative analysis of rule-based and deep learning models in political bias classification. In Proceedings of the Second Workshop on Social Influence in Conversations (SICon 2024), pages 102–115, Miami, Florida, USA. Association for Computational Linguistics.

Andrea Masini and Peter Van Aelst. 2017. Actor diversity and viewpoint diversity: Two of a kind? Communications, 42(2):107–126.

Julie Mastrine. 2022. How to spot 16 types of media bias. allsides.com.

Daniel G McDonald and John Dimmick. 2003. The conceptualization and measurement of diversity. Communication Research, 30(1):60–79.

Denis McQuail. 1992. Media performance: Mass communication and the public interest. Sage.

Chiara Meluzzi, Erica Pinelli, Elena Valvason, and Chiara Zanchi. 2021. Responsibility attribution in gender-based domestic violence: A study bridging corpus-assisted discourse analysis and readers’ perception. Journal of pragmatics, 185:73–92.

Julia Mendelsohn, Ceren Budak, and David Jurgens. 2021. Modeling framing in immigration discourse on social media. In Proceedings of the 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2219–2263, Online. Association for Computational Linguistics.

Stefano Menini and Sara Tonelli. 2016. Agreement and disagreement: Comparison of points of view in the political domain. In Proceedings ofCOL-ING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 2461–2470.

Wiktoria Mieleszczenko-Kowszewicz, Kamil Kanclerz, Julita Bielaniewicz, Marcin Oleksy, Marcin Gruza, Stanislaw Wozniak, Ewa Dzieciol, Przemyslaw Kazienko, and Jan Kocon. 2023. Capturing human perspectives in nlp: Questionnaires, annotations, and biases. In

Proceedings of the Workshop on Perspectivist Approaches to NLP at ECAI.

Jeremiah Milbauer, Ziqi Ding, Zhijin Wu, and Tongshuang Wu. 2023. NewsSense: Referencefree verification via cross-document comparison. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 422–430, Singapore. Association for Computational Linguistics.

Gosse Minnema, Sara Gemelli, Chiara Zanchi, Tommaso Caselli, and Malvina Nissim. 2022a. Dead or murdered? predicting responsibility perception in femicide news reports. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1078–1090, Online only. Association for Computational Linguistics.

Gosse Minnema, Sara Gemelli, Chiara Zanchi, Tommaso Caselli, and Malvina Nissim. 2022b. SocioFillmore: A tool for discovering perspectives. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics: System Demonstrations, pages 240–250, Dublin, Ireland. Association for Computational Linguistics.

Gosse Minnema, Sara Gemelli, Chiara Zanchi, Viviana Patti, Tommaso Caselli, Malvina Nissim, et al. 2021. Frame semantics for social NLP in Italian: Analyzing responsibility framing in femicide news reports. In Proceedings of the Eighth Italian Conference on Computational Linguistics, Milan, Italy.

Gosse Minnema, Huiyuan Lai, Benedetta Muscato, and Malvina Nissim. 2023. Responsibility perspective transfer for Italian femicide news. In Findings of the Association for Computational Linguistics: ACL 2023, pages 7907–7918, Toronto, Canada. Association for Computational Linguistics.

Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. Semeval-2016 task 6: Detecting stance in tweets. In Proceedings of the 10th international workshop on semantic evaluation (SemEval-2016), pages 31–41.

Saif M Mohammad and Peter D Turney. 2013. NRC emotion lexicon. Technical report, National Research Council Canada.

Burt L Monroe, Michael P Colaresi, and Kevin M Quinn. 2008. Fightin’words: Lexical feature selection and evaluation for identifying the content of political conflict. Political Analysis, 16(4):372–403.

Roser Morante, Chantal van Son, Isa Maks, and Piek Vossen. 2020. Annotating perspectives on vaccination. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4964–4973.

Fred Morstatter, Liang Wu, Uraz Yavanoglu, Stephen R Corman, and Huan Liu. 2018. Identifying framing bias in online news. ACM Transactions on Social Computing, 1(2):1–18.

Mats Mulder, Oana Inel, Jasper Oosterman, and Nava Tintarev. 2021. Operationalizing framing to support multiperspective recommendations of opinion pieces. In Proceedings of the 2021 ACM conference onfairness, accountability, and transparency, pages 478–488.

Myriam Munezero, Calkin Suero Montero, Erkki Sutinen, and John Pajunen. 2014. Are they different? affect, feeling, emotion, sentiment, and opinion detection in text. IEEE transactions on affective computing, 5(2):101–111.

Benedetta Muscato, Chandana Sree Mala, Marta Marchiori Manerba, Gizem Gezici, and Fosca Giannotti. 2024. An overview of recent approaches to enable diversity in large language models through aligning with human perspectives. In Proceedings of the 3rd Workshop on Perspectivist Approaches to NLP (NLPerspectives).

Nona Naderi and Graeme Hirst. 2017. Classifying frames at the sentence level in news articles. Policy, 9:4–233.

Preslav Nakov, Firoj Alam, Shaden Shaar, Giovanni Da San Martino, and Yifan Zhang. 2021. A second pandemic? analysis of fake news about COVID-19 vaccines in Qatar. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 1010–1021, Held Online. INCOMA Ltd.

Philip M Napoli. 1999. Deconstructing the diversity principle. Journal of communication, 49(4):7–34.

Viet-An Nguyen, Jordan L Ying, and Philip Resnik. 2013. Lexical and hierarchical topic regression. Proceedings of NeurIPS, 26.

Frank Nielsen. 2016. Hierarchical clustering. In Introduction to HPC with MPIfor Data Science, pages 195–211. Springer.

Olubusayo Olabisi, Aaron Hudson, Antonie Jetter, and Ameeta Agrawal. 2022. Analyzing the dialect diversity in multi-document summaries. In Proceedings of the 29th International Conference on Computational Linguistics, pages 6208– 6221, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Yulia Otmakhova and Lea Frermann. 2025. Narrative media framing in political discourse. In Findings of the Association for Computational Linguistics: ACL 2025, pages 9167–9196, Vienna, Austria. Association for Computational Linguistics.

Yulia Otmakhova, Shima Khanehzar, and Lea Frermann. 2024. Media framing: A typology and survey of computational approaches across disciplines. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15407– 15428, Bangkok, Thailand. Association for Computational Linguistics.

Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan. 2002. Thumbs up? sentiment classification using machine learning techniques. In Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing, pages 79–86.

Bo Pang, Lillian Lee, et al. 2008. Opinion mining and sentiment analysis. Foundations and Trends® in information retrieval, 2(1–2):1–135.

Eli Pariser. 2011. The Filter Bubble: What the Internet Is Hiding from You. Penguin Group.

ChaeHun Park, Wonsuk Yang, and Jong Park. 2019. Generating sentential arguments from diverse perspectives on controversial topic. In Proceedings of the Second Workshop on Natural Language Processingfor Internet Freedom: Censorship, Disinformation, and Propaganda, pages

56–65, Hong Kong, China. Association for Computational Linguistics.

Michael Paul and Roxana Girju. 2010. A twodimensional topic-aspect model for discovering multi-faceted topics. In Proceedings ofthe AAAI conference on artificial intelligence, volume 24, 1, pages 545–550.

Erica Pinelli and Chiara Zanchi. 2021. Genderbased violence in italian local newspapers: How argument structure constructions can diminish a perpetrator’s responsibility. In Discourse Processes between Reason and Emotion: A Post-disciplinary Perspective, pages 117–143. Springer.

Jakub Piskorski, Nicolas Stefanovitch, Giovanni Da San Martino, and Preslav Nakov. 2023. Semeval-2023 task 3: Detecting the category, the framing, and the persuasion techniques in online news in a multi-lingual setup. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2343–2361.

Flor Miriam Plaza-del-Arco, Alba A. Cercas Curry, Amanda Cercas Curry, and Dirk Hovy. 2024. Emotion analysis in NLP: Trends, gaps and roadmap for future directions. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 5696–5710, Torino, Italia. ELRA and ICCL.

Joan Plepi, Charles Welch, and Lucie Flek. 2024. Perspective taking through generating responses to conflict situations. In Findings ofthe Associationfor Computational Linguistics ACL 2024, pages 6482–6497.

Robert Plutchik. 1980. A general psychoevolutionary theory of emotion. In Theories of emotion, pages 3–33. Elsevier.

Randolph Quirk, Sidney Greenbaum, Geoffrey Leech, and Jan Svartvik. 1985. A Comprehensive Grammar ofthe English Language. Longman, London.

Hannah Rashkin, Sameer Singh, and Yejin Choi. 2016. Connotation frames: A data-driven investigation. In Proceedings ofthe 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 311– 321.

Marta Recasens, Cristian Danescu-Niculescu-Mizil, and Dan Jurafsky. 2013. Linguistic models for analyzing and detecting biased language. In Proceedings of the 51st annual meeting of the Association for Computational Linguistics (volume 1: long papers), pages 1650–1659.

Nils Reimers, Benjamin Schiller, Tilman Beck, Johannes Daxenberger, Christian Stab, and Iryna Gurevych. 2019. Classification and clustering of arguments with contextualized word embeddings. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 567–578, Florence, Italy. Association for Computational Linguistics.

Myrthe Reuver, Antske Fokkens, and Suzan Verberne. 2021a. No NLP task should be an island: Multi-disciplinarity for diversity in news recommender systems. In Proceedings of the EACL Hackashop on News Media Content Analysis and Automated Report Generation, pages 45–55, Online. Association for Computational Linguistics.

Myrthe Reuver, Nicolas Mattis, Marijn Sax, Suzan Verberne, Nava Tintarev, Natali Helberger, Judith Moeller, Sanne Vrijenhoek, Antske Fokkens, and Wouter van Atteveldt. 2021b. Are we human, or are we users? the role of natural language processing in human-centric news recommenders that nudge users to diverse content. In Proceedings of the 1st Workshop on NLP for Positive Impact, pages 47–59, Online. Association for Computational Linguistics.

Myrthe Reuver, Suzan Verberne, and Antske Fokkens. 2024. Investigating the robustness of modelling decisions for few-shot cross-topic stance detection: A preregistered study. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 9245–9260, Torino, Italia. ELRA and ICCL.

Ellen Riloff and Janyce Wiebe. 2003. Learning extraction patterns for subjective expressions. In Proceedings of the 2003 conference on Empirical methods in natural language processing, pages 105–112.

Margaret E Roberts, Brandon M Stewart, Dustin Tingley, Christopher Lucas, Jetson Leder-Luis, Shana Kushner Gadarian, Bethany Albertson,

and David G Rand. 2014. Structural topic models for open-ended survey responses. American journal ofpolitical science, 58(4):1064–1082.

Francisco-Javier Rodrigo-Ginés, Jorge Carrillo-de Albornoz, and Laura Plaza. 2024. A systematic review on media bias detection: What is media bias, how it is expressed, and how to detect it. Expert Systems with Applications, 237:121641.

Egil Rønningstad, Roman Klinger, Erik Velldal, and Lilja Øvrelid. 2024. Entity-level sentiment: More than the sum of its parts. In Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pages 84–96.

Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Kirk, Hinrich Schuetze, and Dirk Hovy. 2024. Political compass or spinning arrow? towards more meaningful evaluations for values and opinions in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15295–15311.

Shamik Roy and Dan Goldwasser. 2023. “a tale of two movements’: Identifying and comparing perspectives in# blacklivesmatter and# bluelivesmatter movements-related tweets using weakly supervised graph-based structured prediction. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10437–10467.

Shamik Roy, María Leonor Pacheco, and Dan Goldwasser. 2021. Identifying morality frames in political tweets using relational learning. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 9939–9958.

Sebastian Ruder, John Glover, Afshin Mehrabani, and Parsa Ghaffari. 2018. 360<sup>◦</sup> stance detection. In Proceedings of the 2018 Conference of the North American Chapter ofthe Association for Computational Linguistics: Demonstrations, pages 31–35, New Orleans, Louisiana. Association for Computational Linguistics.

Sougata Saha and Rohini Srihari. 2024. Turiya at PerpectiveArg2024: A multilingual argument retriever and reranker. In Proceedings of the

11th Workshop on Argument Mining (ArgMining 2024), pages 159–163, Bangkok, Thailand. Association for Computational Linguistics.

Abdelrhman Saleh, Ramy Baly, Alberto Barrón-Cedeño, Giovanni Da San Martino, Mitra Mohtarami, Preslav Nakov, and James Glass. 2019. Team QCRI-MIT at SemEval-2019 task 4: Propaganda analysis meets hyperpartisan news detection. In Proceedings ofthe 13th International Workshop on Semantic Evaluation, pages 1041– 1046, Minneapolis, Minnesota, USA. Association for Computational Linguistics.

José Sanders and Gisela Redeker. 1993. Linguistic perspective in short news stories. Poetics, 22(1- 2):69–87.

Olufunke O. Sarumi, Béla Neuendorf, Joan Plepi, Lucie Flek, Jörg Schlötterer, and Charles Welch. 2024. Corpus considerations for annotator modeling and scaling. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1029–1040, Mexico City, Mexico. Association for Computational Linguistics.

Andrea Scarantino. 2016. The philosophy of emotions and its impact on affective science. In Michael Lewis, Jeanette Haviland-Jones, and Lisa Feldman Barrett, editors, The Handbook of Emotions, 4th edition. Guildford University Press, New York.

Shalom H Schwartz. 1992. Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. In Advances in experimental social psychology, volume 25, pages 1–65. Elsevier.

Holli A Semetko and Patti M Valkenburg. 2000. Framing european politics: A content analysis of press and television news. Journal of communication, 50(2):93–109.

Meghdut Sengupta, Roxanne El Baff, Milad Alshomary, and Henning Wachsmuth. 2024. Analyzing the use of metaphors in news editorials for political framing. In Proceedings ofthe 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3621–3631, Mexico City, Mexico. Association for Computational Linguistics.

Xu Sheng, Fumiyo Fukumoto, Jiyi Li, Go Kentaro, and Yoshimi Suzuki. 2023. Learning disentangled meaning and style representations for positive text reframing. In Proceedings of the 16th International Natural Language Generation Conference, pages 424–430, Prague, Czechia. Association for Computational Linguistics.

Chongyang Shi, Yijun Yin, Qi Zhang, Liang Xiao, Usman Naseem, Shoujin Wang, and Liang Hu. 2023. Multiview clickbait detection via jointly modeling subjective and objective preference. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11807–11816, Singapore. Association for Computational Linguistics.

Mohammad Shokri, Vivek Sharma, Elena Filatova, Shweta Jain, and Sarah Levitan. 2024. Subjectivity detection in English news using large language models. In Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pages 215–226, Bangkok, Thailand. Association for Computational Linguistics.

Jonathan B Slapin and Sven-Oliver Proksch. 2008. A scaling model for estimating time-series party positions from texts. American Journal of Political Science, 52(3):705–722.

David A Snow and Robert D Benford. 2000. Clarifying the relationship between framing and ideology in the study of social movements: A comment on Oliver and Johnston. Mobilization, 5(2):55–60.

Swapna Somasundaran and Janyce Wiebe. 2010. Recognizing stances in ideological on-line debates. In Proceedings of the NAACL HLT 2010 Workshop on Computational Approaches to Analysis and Generation ofEmotion in Text, pages 116–124, Los Angeles, CA. Association for Computational Linguistics.

Taylor Sorensen, Liwei Jiang, Jena D Hwang, Sydney Levine, Valentina Pyatkin, Peter West, Nouha Dziri, Ximing Lu, Kavel Rao, Chandra Bhagavatula, et al. 2024a. Value kaleidoscope: Engaging AI with pluralistic human values, rights, and duties. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19937–19947.

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, et al. 2024b. Position: a roadmap to pluralistic alignment. In Proceedings of the 41st International Conference on Machine Learning, pages 46280–46302.

Aleksandra Sorokovikova, Michael Becker, and Ivan P Yamshchikov. 2024. Echo-chambers and idea labs: Communication styles on twitter. In Proceedings of the Second Workshop on Natural Language Processing for Political Sciences @ LREC-COLING 2024, pages 91–95, Torino, Italy.

Alexander Spangher, Nanyun Peng, Sebastian Gehrmann, and Mark Dredze. 2024. Do LLMs plan like human writers? Comparing journalist coverage of press releases with LLMs. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 21814–21828, Miami, Florida, USA. Association for Computational Linguistics.

Christian Stab and Iryna Gurevych. 2017. Parsing argumentation structures in persuasive essays. Computational Linguistics, 43(3):619–659.

Cass R. Sunstein. 2007. Republic.com 2.0. Princeton University Press, USA.

Marco Te Brömmelstroet. 2020. Framing systemic traffic violence: Media coverage of Dutch traffic crashes. Transportation research interdisciplinary perspectives, 5:100–109.

Thibaut Thonet, Guillaume Cabanac, Mohand Boughanem, and Karen Pinel-Sauvagnat. 2016. Vodum: a topic model unifying viewpoint, topic and opinion discovery. In Proceedings ofECIR, pages 533–545, Padova, Italy. Springer.

Isidora Tourni, Lei Guo, Taufiq Husada Daryanto, Fabian Zhafransyah, Edward Edberg Halim, Mona Jalal, Boqi Chen, Sha Lai, Hengchang Hu, Margrit Betke, Prakash Ishwar, and Derry Tanti Wijaya. 2021. Detecting frames in news headlines and lead images in U.S. gun violence coverage. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4037– 4050, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Amine Trabelsi and Osmar R Zaıane. 2014. Finding arguing expressions of divergent viewpoints in online debates. In Proceedings of EACL, pages 35–43.

Oren Tsur, Dan Calacci, and David Lazer. 2015. A frame of mind: Using statistical models for detection of framing and agenda setting campaigns. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1629–1638.

Mathieu Valette. 2024. What does perspectivism mean? an ethical and methodological countercriticism. In Proceedings of the 3rd Workshop on Perspectivist Approaches to NLP (NLPerspectives)@ LREC-COLING 2024, pages 111–115.

Michiel Van Der Meer. 2024. Facilitating opinion diversity through hybrid NLP approaches. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 4: Student Research Workshop), pages 272–284, Mexico City, Mexico. Association for Computational Linguistics.

Michiel van der Meer, Neele Falk, Pradeep K. Murukannaiah, and Enrico Liscio. 2024a. Annotator-centric active learning for subjective NLP tasks. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 18537–18555, Miami, Florida, USA. Association for Computational Linguistics.

Michiel van der Meer, Piek Vossen, Catholijn M. Jonker, and Pradeep K. Murukannaiah. 2024b. An empirical analysis of diversity in argument summarization. In Proceedings ofthe 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2028–2045, St. Julian’s, Malta. Association for Computational Linguistics.

Teun A Van Dijk. 1998. Ideology: A multidisciplinary approach. Sage.

Merlijn Van Hulst, Tamara Metze, Art Dewulf, Jasper De Vries, Severine Van Bommel, and Mark Van Ostaijen. 2025. Discourse, framing

and narrative: three ways of doing critical, interpretive policy analysis. Critical policy studies, 19(1):74–96.

Chantal van Son, Tommaso Caselli, Antske Fokkens, Isa Maks, Roser Morante, Lora Aroyo, and Piek Vossen. 2016. Grasp: A multilayered annotation scheme for perspectives. In LREC 2016, Tenth International Conference on Language Resources and Evaluation, pages 1177– 1184. European Language Resources Association (ELRA).

Chantal van Son, Marieke van Erp, Antske Fokkens, and Piek Vossen. 2014. Hope and fear: How opinions influence factuality. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 3857–3864, Reykjavik, Iceland. European Language Resources Association (ELRA).

Francielle Vargas, Kokil Jaidka, Thiago Pardo, and Fabrício Benevenuto. 2023. Predicting sentencelevel factuality of news and bias of media outlets. In Proceedings ofthe 14th International Conference on Recent Advances in Natural Language Processing, pages 1197–1206.

Karina Vida, Judith Simon, and Anne Lauscher. 2023. Values, ethics, morals? on the use of moral concepts in NLP research. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 5534–5554, Singapore. Association for Computational Linguistics.

David Vilares and Yulan He. 2017. Detecting perspectives in political debates. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 1573–1582.

Nikolas Vitsakis, Amit Parekh, and Ioannis Konstas. 2024. Voices in a crowd: Searching for clusters of unique perspectives. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 12517– 12539.

Piek Vossen and Antske Fokkens, editors. 2022. Creating a More Transparent Internet: The Perspective Web. Studies in Natural Language Processing. Cambridge University Press.

Sanne Vrijenhoek, Mesut Kaya, Nadia Metoui, Judith Möller, Daan Odijk, and Natali Helberger.

2021. Recommenders with a mission: assessing diversity in news recommendations. In Proceedings ofthe 2021 conference on human information interaction and retrieval, pages 173–183.

Herun Wan, Shangbin Feng, Zhaoxuan Tan, Heng Wang, Yulia Tsvetkov, and Minnan Luo. 2024. DELL: Generating reactions and explanations for LLM-based misinformation detection. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 2637–2667, Bangkok, Thailand. Association for Computational Linguistics.

Albert Webson, Zhizhong Chen, Carsten Eickhoff, and Ellie Pavlick. 2020. Are “undocumented workers” the same as “illegal aliens”? Disentangling denotation and connotation in vector spaces. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4090–4105, Online. Association for Computational Linguistics.

Janyce Wiebe, Theresa Wilson, Rebecca Bruce, Matthew Bell, and Melanie Martin. 2004. Learning subjective language. Computational linguistics, 30(3):277–308.

Janyce Wiebe, Theresa Wilson, and Claire Cardie. 2005. Annotating expressions of opinions and emotions in language. Language resources and evaluation, 39:165–210.

T Wilson. 2005. Recognizing contextual polarity in phrase-level sentiment analysis. In Proceedings of HLT/EMNLP.

Theresa Wilson. 2008. Annotating subjective content in meetings. In Proceedings ofthe Sixth International Conference on Language Resources and Evaluation (LREC’08), Marrakech, Morocco. European Language Resources Association (ELRA).

Theresa Wilson, Paul Hoffmann, Swapna Somasundaran, Jason Kessler, Janyce Wiebe, Yejin Choi, Claire Cardie, Ellen Riloff, and Siddharth Patwardhan. 2005. Opinionfinder: A system for subjectivity analysis. In Proceedings of HLT/EMNLP 2005 Interactive Demonstrations, pages 34–35.

Patrick Xia, Guanghui Qin, Siddharth Vashishtha, Yunmo Chen, Tongfei Chen, Chandler May,

Craig Harman, Kyle Rawlins, Aaron Steven White, and Benjamin Van Durme. 2021. LOME: Large ontology multilingual extraction. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 149– 159, Online. Association for Computational Linguistics.

Hanqi Yan, Qinglin Zhu, Xinyu Wang, Lin Gui, and Yulan He. 2024. Mirror: Multiple-perspective self-reflection method for knowledge-rich reasoning. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7086– 7103, Bangkok, Thailand. Association for Computational Linguistics.

Tae Yano, Philip Resnik, and Noah A Smith. 2010. Shedding (a thousand points of) light on biased language. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon’s Mechanical Turk, pages 152–158.

Yangfan Ye, Xiachong Feng, Xiaocheng Feng, Weitao Ma, Libo Qin, Dongliang Xu, Qing Yang, Hongtao Liu, and Bing Qin. 2024. Globe-Summ: A challenging benchmark towards unifying multi-lingual, cross-lingual and multidocument news summarization. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10803– 10821, Miami, Florida, USA. Association for Computational Linguistics.

Hong Yu and Vasileios Hatzivassiloglou. 2003. Towards answering opinion questions: Separating facts from opinions and identifying the polarity of opinion sentences. In Proceedings of the 2003 conference on Empirical methods in natural language processing, pages 129–136.

Chiara Zanchi, Serena Coschignano, and Gosse Minnema. 2021. Numeri in arrivo anziché uomini e donne in partenza: Come i frame ci rendono inconsapevolmente ingiusti. In Notizie ai margini: IX rapporto Carta di Roma 2021, pages 30–36. Associazione Carta di Roma.

Wenqian Zhang, Shangbin Feng, Zilong Chen, Zhenyu Lei, Jundong Li, and Minnan Luo. 2022. KCD: Knowledge walks and textual cues enhanced political perspective detection in news

media. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4129–4140, Seattle, United States. Association for Computational Linguistics.

Chenye Zhao, Yingjie Li, Cornelia Caragea, and Yue Zhang. 2024. Zerostance: Leveraging chatgpt for open-domain stance detection via dataset generation. In Findings of the Association for Computational Linguistics ACL 2024, pages 13390–13405.

He Zhao, Dinh Phung, Viet Huynh, Yuan Jin, Lan Du, and Wray Buntine. 2021. Topic modelling meets deep neural networks: A survey. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, pages 4713– 4720.

Yilun Zhao, Zhenting Qi, Linyong Nan, Lorenzo Jaime Flores, and Dragomir Radev. 2023. LoFT: Enhancing faithfulness and diversity for table-to-text generation via logic form control. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 554–561, Dubrovnik, Croatia. Association for Computational Linguistics.

Xiaofan Zheng, Minnan Luo, and Xinghao Wang. 2025. Unveiling fake news with adversarial arguments generated by multimodal large language models. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 7862–7869, Abu Dhabi, UAE. Association for Computational Linguistics.

Caleb Ziems and Diyi Yang. 2021. To protect and to serve? analyzing entity-centric framing of police violence. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 957–976, Punta Cana, Dominican Republic. Association for Computational Linguistics.

## A Paper Collection

## A.1 ACL regular expression search

The RegEx search is conducted on ACL titles and abstracts in May 2025, using the following query: PERSPECTIVE[S] OR VIEWPOINT[S] AND DIVER-SITY OR NEWS. We filter out papers where (i) perspective is used generically (e.g., to denote research angles or methods); (ii) the notion of perspective is not central to the paper’s contribution; or (iii) the focus is on perspective-taking in a narrative sense (e.g., deictic shifts), retaining 60 papers from the original 139 retrieved.

## A.2 Collected references

The references in Table 4 include both papers from the ACL search and additional foundational work collected manually via citation chaining, as described in §2. We report all consulted references, organized by thematic area.

## B Property Annotation and Analysis

## B.1 Annotation Procedure

The annotation was conducted in two iterations. In the first iteration, the three annotators independently assigned scores to each concept on the four dimensions described in the codebook below (§B.2). After completing the first round, annotators shared their scores and discussed cases of disagreement, focusing on cases where the operationalization was ambiguous. In the second iteration, annotators revised their scores in light of the discussion, without being required to reach consensus: final scores reflect each coder’s independent expert judgment. Inter-rater reliability was computed on each property using Spearman ρ (cf. §4).

## B.2 Codebook

## Instructions

For each perspective-related concept listed in Table 3, independently assign a score from 1 to 5 on each of the four dimensions below. Scores reflect your expert judgment based on the NLP literature. In case of doubt, consult the concept definitions and examples in Table 5 at the end of this section. Do not discuss your scores with other annotators until all annotations in the current iteration are complete.

## Dimensions

We characterize perspective-related concepts along four dimensions, selected because they jointly capture the key aspects that distinguish concepts in the literature and determine how they are operationalized in annotation and detection tasks.

For each concept, assign a score on a Likert scale from 1 to $5 \left( 1 = \mathrm { l o w } ; 5 = \mathrm { h i g h } \right)$

1. Strength of linguistic cues: how strongly the concept is associated with specific linguistic elements [1 = weakly associated; 5 = strongly associated]. This dimension captures to what extent a concept is signalled by identifiable surface features: some concepts leave strong lexical and syntactic traces, while others require holistic document-level inference with no reliable surface cues.

2. Granularity (scope): the typical scope or localization of the concept within a text [1 = very broad/document-level; 5 = very narrow/phrase- or clause-level]. This dimension determines the annotation unit: some concepts are inherently local, while others are distributed across an entire document.

3. Entity-specificity: how strongly the concept is tied to a specific entity (e.g., politician, policy, event) [1 = not tied to a specific entity; 5 = directly tied to a specific entity]. This dimension distinguishes target-generic from target-specific concepts, determining whether entity recognition is a prerequisite for annotation and detection.

4. Number ofdiscrete classes: in a typical classification task, how many classes the concept comprises [1 = few classes, e.g., binary; 5 = many classes]. This dimension reflects the complexity of the label space: binary or ternary concepts organize reality in a coarsegrained fashion, while open-ended concepts require finer distinctions.

## Decision Rules

• If a concept can be operationalized in multiple ways (e.g., sentiment as binary emotional polarity or as the perspective expression itself), refer to the examples in Table 3 to choose a specific interpretation.

• Score each concept independently; do not let your score on one dimension influence another.

## Anchor Examples and Concept Definitions

For reference, the following examples illustrate prototypical high and low scores across dimensions:

• High across dimensions (score ≈ 5): claims — triggered by specific linguistic patterns, tied to a specific proposition, open-ended label space.

<table><tr><td>Domain</td><td>References</td></tr><tr><td>surveys</td><td>Grounding work &amp; discourse/narrative: Sanders and Redeker (1993), Graumann and Kallmeyer (2008), Van Der Meer (2024), Van Hulst et al. (2025); ideology: Van Dijk (1998); bias: Rodrigo-Ginés et al. (2024); subjectivity: Quirk et al. (1985); sentiment: Pang et al. (2008), Pang et al. (2002), Munezero et al. (2014), Liu and Zhang (2012), Chaturvedi et al. (2018); emotions: Ekman (1992), Plutchik (1980); stance: Küçük and Can (2020), Kiesling et al. (2018); political ideology: Doan and Gulla (2022); frames: Entman (1993). Semetko and Valkenburg (2000), Ali and Hassan (2022), Fillmore (1976), Baker et al. (1998), Rashkin et al. (2016), Otmakhova et al. (2024), Otmakhova and Frermann (2025); perspectivism: Basile et al. (2022), Frenda et al. (2024), Valette (2024), Mieleszczenko-Kowszewicz et al. (2023), Aroyo and Welty</td></tr><tr><td>Communication stud- ies</td><td>Faleńska (2026), Vida et al. (2023); evaluative language: Hunston and Thompson (2000), Benamara et al. (2017); topics: Alghamdi and Alfalqi (2015), Zhao et al. (2021) conceptual: Loecherbach et al. (2020), McQuail (1992), Napoli (1999), Sunstein (2007), McDonald and Dimmick (2003), Snow and Benford (2000), Benson (2009), Baden and Springer (2017), Helberger et al. (2018), Helberger (2021), Pariser (2011); applications: Masini and Van Aelst (2017), Mulder et al.</td></tr><tr><td>Morals &amp; values Private states</td><td>(2021), Draws et al. (2021), Vrijenhoek et al. (2021), Vossen and Fokkens (2022), Draws et al. (2022), Sorokovikova et al. (2024) Sorensen et al. (2024a), Röttger et al. (2024), Benkler et al. (2023), Abdulhai et al. (2024), Roy et al. (2021), Kiesel et al. (2022)</td></tr><tr><td></td><td>subjectivity: Carbonell (1979), Riloff and Wiebe (2003), Wiebe et al. (2004), Wiebe et al. (2005), Shokri et al. (2024), Wilson (2008); polarity/stance: Wilson (2005), Wilson et al. (2005), Somasundaran and Wiebe (2010), Choi and Wiebe (2014), Bošnjak and Karan (2019), Liu et al. (2005), Deng and Wiebe (2015); sentiment: Martinc et al. (2021), Yu and Hatzivassiloglou (2003), Rønningstad et al. (2024), Liu</td></tr><tr><td>Media bias</td><td>et al. (2010); emotions: Mohammad and Turney (2013); opinion: Kim and Hovy (2004), Grefenstette et al. (2004), Morante et al. (2020), van Son et al. (2016) Yano et al. (2010), A1 Khatib et al. (2012), Recasens et al. (2013), Fan et al. (2019), Vargas et al. (2023), Mastrine (2022), Lin et al. (2025), Da San Martino et al. (2023), Saleh et al. (2019), van Son et al. (2014)</td></tr><tr><td>Polar ideology stance</td><td>&amp; Lin et al. (2006), Lin et ai. (2008), Ahmed and Xing (2010), Hardisty et al. (2010), Klebanov et al. (2010), Hasan and Ng (2012), Elfardy et al. (2015), Johnson and Goldwasser (2016), Zhao et al. (2024), Reuver et al. (2024), Roy and Goldwasser (2023), Anand et al. (2011), Mohammad et al. (2016)</td></tr><tr><td>Political leaning</td><td>Gentzkow et al. (2019), Monroe et al. (2008), Budak et al. (2016), Gangula et al. (2019b), Baly et al. (2020), Feng et al. (2021), Zhang et al. (2022), Alzhrani (2022), Kim et al. (2023), Han et al. (2019), Li and Goldwasser (2021), Kiesel et al. (2019), Li and Goldwasser (2019), Ceron et al. (2022), Martinez et al. (2024), Laver et al. (2003), Slapin and Proksch (2008), Jakob et al. (2024), Huguet Cabot et al.</td></tr><tr><td>Frames</td><td>(2020), Bhatia and Deepak (2018), Webson et al. (2020), Greene and Resnik (2009), Iyyer et al. (2014) media frames: Baumer et al. (2015), Lecheler et al. (2015), Alashri et al. (2015), Card et al. (2015), Card et al. (2016), Naderi and Hirst (2017), Field et al. (2018), Morstatter et al. (2018), Khanehzar et al. (2019), Mendelsohn et al. (2021), Khanehzar et al. (2021), Gilardi et al. (2023), Piskorski et al. (2023), Bhatia et al. (2021), Das et al. (2024), Tourni et al. (2021), Liu et al. (2019), Nakov et al. (2021), Daffara et al. (2025), Kwak et al. (2020), Ziems and Yang (2021), Sengupta et al. (2024), Blokker et al. (2022); semantic frames: Te Brömmelstroet (2020), Xia et al. (2021), Minnema et al. (2021), Minnema et al.</td></tr><tr><td>Clusters as perspec- tives</td><td>(2022b), Meluzzi et al. (2021), Pinelli and Zanchi (2021), Zanchi et al. (2021), Minnema et al. (2023), Minnema et al. (2022a) Chen et al. (2017); sub-groups: Dasigi et al. (2012), Abu-Jbara et al. (2013), Abu-Jbara et al. (2012); claims/args: Bar-Haim et al. (2017), Chen et al. (2019), Ajjour et al. (2019), Trabelsi and Zaiane (2014), Carlebach et al. (2020), Boland et al. (2022), Stab and Gurevych (2017); LDA: Boydstun et al. (2013), Tsur et al. (2015), Nguyen et al. (2013), Roberts et al. (2014), Paul and Girju (2010), Thonet et al. (2016), Menini and Tonelli (2016), Vilares and He (2017); arg. retrieval: Saha and Srihari (2024), Kang et al.</td></tr><tr><td>Multi-view generation</td><td>(2024); other: Vitsakis et al. (2024), De Vreese (2005), Reimers et al. (2019), DiMaggio et al. (2013), Draws et al. (2020) Chen and Choi (2025), Plepi et al. (2024); personas: Hu et al. (2025), Baltaji et al. (2024); chatbots: Hamad et al. (2024); diversity: Hayati et al. (2024), Park et al. (2019); reasoning: Yan et al. (2024); table-to-text: Zhao et al. (2023); multichannel: Chan et al. (2020); writing: Spangher et al. (2024); fake news: Zheng et al. (2025), Wan et al. (2024); reframing: Sheng et al. (2023), Chen et al. (2021);</td></tr><tr><td>Multi-view summa- rization News recommenda- tion</td><td>alignment: Sorensen et al. (2024b) Barker et al. (2016), Deas and McKeown (2025), Liu et al. (2021), Liu et al. (2024), van der Meer et al. (2024b), Huang et al. (2023), Olabisi et al. (2022), Ye et al. (2024); fake news: Chang et al. (2023) Reuver et al. (2021b), Reuver et al. (2021a), Milbauer et al. (2023); stance: Ruder et al. (2018); clickbait: Shi et al. (2023)</td></tr></table>

Table 4: All references consulted, organized by thematic area.

• Low across dimensions (score ≈ 1): ideology bias — no reliable surface cue, documentlevel, no entity required, binary label.

Table 5 provides definitions and examples for each concept to be annotated. Consult it when the scope of a concept is unclear before assigning scores.

## B.3 Principal Component Analysis (PCA)

We perform PCA on averaged annotations as a validation of our conceptual model shown in Figure 1. PC1 and PC2 explain 83.4% of the variance. As shown in Figure 6, the distribution of concepts along these two latent dimensions corresponds to the four clusters recognized in §4.2. Notably, PC1 alone explains 61.9% of the variance and is positively correlated with each of the four properties (cf. Figure 7). PC2 instead is dominated by the class number and, negatively, by the strength of linguistic cues. Overall, these findings validate the cluster analysis and the correlation between properties, supporting the linear organization of the concepts in clusters along a single axis of specificity.

![](images/751e457a0918a35a0df17ebfe6ab804e5ceb3e025603d49964a694584b3095c1.jpg)

![](images/5ec835fa816ed3e5fec659018a49da96bd9fd577d3e99611074882a10aeb8085.jpg)  
Figure 7: Perspective concepts scores and property loadings on PC1. The four properties all positively correlate with PC1, validating our linear model in Figure 1.  
Figure 6: Perspective concepts along PC1 and PC2. The structure validates the clusters found in §4.2.

<table><tr><td>Term</td><td>Definition</td><td>Example (class/value)</td></tr><tr><td>Argument</td><td>A set of statements about a controversial topic, made up of premises and conclusions (Govier, 2005).</td><td>Marijuana should not be legalized. That&#x27;s because sustained use of mar- ijuana worsens a person&#x27;s memory, and nothing that adversely affects</td></tr><tr><td>Claim</td><td>A component of an argument, either the premise or the conclu- sion (Govier, 2005) or the central assertion (Stab and Gurevych,</td><td>galized. (Govier, 2005) Animals should not be used for sci- entific or commercial testing. (Chen</td></tr><tr><td>Semantic frames</td><td>2017; Boland et al., 2022). In linguistics, structures of meaning consisting of semantic roles and lexical units (Fillmore, 1976).</td><td>et al., 2019) ABUSE, RAPE, CAUSE_MOTION,</td></tr><tr><td>Media frames</td><td>Topic-like dimensions that organize reality to shape understand-</td><td>USE_FIREARM MORALITY, PUBLIC SENTIMENT,</td></tr><tr><td>Topic</td><td>ing and promote specific views (Card et al., 2015). An interpretable semantic concept emerging from probabilistic word distributions (Alghamdi and Alfalqi, 2015).</td><td>CULTURAL IDENTITY ECONOMY, POLITICS</td></tr><tr><td>Opinion</td><td>An idea or belief about a target that contributes to a viewpoint (Munezero et al., 2014; Thonet et al., 2016), typically found in</td><td>Mary said the dress is beautiful. (Kim and Hovy, 2004)</td></tr><tr><td>Stance</td><td>subjective texts (Pang et al., 2008). The evaluation of a target, which can align the author with or</td><td>IN FAVOUR-AGAINST</td></tr><tr><td>Sentiment</td><td>against others (Küçük and Can, 2020). A lasting feeling or disposition toward something (Munezero</td><td>POSITIVE—NEGATIVE</td></tr><tr><td>Emotions</td><td>et al., 2014). Affective states specifying the type of subjective response</td><td>FEAR, ANGER, JOY, SADNESS</td></tr><tr><td>Polarity</td><td>(Plutchik, 1980). The orientation of sentiment towards positive or negative (Pang et al., 2008).</td><td>POSITIVE-NEGATIVE</td></tr><tr><td>Ideological bias</td><td>It occurs when a text is unbalanced towards a particular ideology</td><td>BIASED-NOT BIASED</td></tr><tr><td>Political ideology</td><td>(Yano et al., 2010). The political orientation of a text in terms of ideological values</td><td>PALESTINIAN/ISRAELI, CONSER-</td></tr><tr><td>Political leaning</td><td>(Pang et al., 2008). The political orientation of a text in terms of political spectrum</td><td>VATIVE-LIBERAL LEFT-RIGHT</td></tr><tr><td>Morals</td><td>(Doan and Gulla, 2022). Societal-level norms used to judge between “right&quot; and “wrong&#x27;</td><td>CARE/HARM, FAIRNESS/BE-</td></tr><tr><td>Values</td><td>(Vida et al., 2023). The individual ideals that people pursue (Sorensen et al., 2024a)</td><td>TRAYAL FREEDOM, EQUALITY</td></tr></table>

Table 5: Perspective-related terms with definitions and examples. We report example labels if the concept is commonly mapped onto label sets for text classification, and example values when it it is open-ended and commonly denotes language spans in mining tasks.