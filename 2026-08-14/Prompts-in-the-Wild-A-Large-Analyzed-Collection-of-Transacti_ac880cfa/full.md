# Prompts in the Wild: A Large Analyzed Collection of Transactional Prompts in Code

Victoria Basmov<sup>1,2</sup> Yoav Goldberg<sup>1,2</sup> Reut Tsarfaty<sup>1</sup> <sup>1</sup>Bar-Ilan University <sup>2</sup>Allen Institute for Artificial Intelligence {vikasaeta, yoav.goldberg, reut.tsarfaty} @gmail.com

## Abstract

The behavior of contemporary generative Large Language Models (LLMs) is directly shaped by prompts, unstructured texts that describe the desired output and model behavior. In this paper we argue that prompts are linguistic objects that merit investigation in their own right. To this end, we collect 57.5K unique samples of prompts from GitHub. Specifically, we focus on transactional prompts: reproducible natural language instructions that are integrated into software. To enable the empirical, quantitative study of prompts, we introduce a structured ontology, capturing the properties of prompts as well as their formal and semantic components. Based on this ontology, we transform prompts from unstructured raw texts into richly structured linguistic objects. Analysis of these structured data reveals significant diversity of usage patterns across languages, domains, tasks, and modalities, in a typical Zipf-like distribution where some clearly prevail and others, more diverse, appear in the long tail. To validate the reliability of the ontology-based annotation of the prompts, we perform a comprehensive error analysis across all fields, providing a detailed assessment of annotation quality. We release the dataset together with a browsing and exploration interface<sup>1</sup>.

## 1 Introduction

Prompts, the instructions humans give to large language models (LLMs), constitute the primary interface for guiding models’ behavior. Despite growing practical interest in prompt engineering and prompting strategies, prompts are still treated largely as informal and intuitive artifacts rather than objects of systematic scientific inquiry. While models are analyzed in great depth, the usage of prompts, natural language utterances which directly shape models behavior, remains largely ad-hoc (Villamizar et al.,

2025). A systematic study of prompts may reveal crucial aspects of LLMs usage patterns: what languages are used in prompts and how? what structural and semantic patterns do they follow? what tasks are they used to solve? what common practices emerge in prompt design? and a lot more. However, tapping into these questions and investigating them empirically, requires injecting into prompts structure that would allow for quantitative, rigorous analyses. Moreover, formalizing prompt structure is essential for the linguistic analysis of prompts (Jeoung et al., 2025; Leidinger et al., 2023); research of sensitivity to linguistic and structural prompt variation (Cuellar et al., 2026; Arabzadeh and Bagheri, 2025; Wahle et al., 2024); for tools and methods of structure-aware and linguistically informed automated prompt optimization (Santos et al., 2025; Hidalgo et al., 2025; Khattab et al., 2023; Saletta and Ferretti, 2024; Khattab et al., 2022; Murthy et al., 2025; Juneja et al., 2025); multilingual prompt engineering (Vatsal et al., 2025; Zhang et al., 2025; Kmainasi et al., 2024); and other areas of prompt research and downstream tasks.

We aim to establish prompts as first-order objects of scientific study. A phenomenon becomes a scientific object of study once practical relevance and sustained research are complemented by a sharedformalframework, which prompts still lack. While prompts are already gaining interest, not only as tools for using LLMs but also as independent objects of study (Pister et al., 2024; Mao et al., 2025; Vir et al., 2025; Zheng et al., 2024a; Villamizar et al., 2025), prompt research lacks common terminology, structure, and large-scale empirical grounds. This work aims to fill this gap.

To facilitate the study of prompts, we collected a dataset of 57.5K unique prompts from public GitHub repositories. We specifically focus on transactional prompts,<sup>2</sup> prompts that are intended to perform reproducible, parameterized tasks, as part of a larger software-based workflow. Unlike casual (“interactive") prompts, which are ad-hoc and oneof interactions with LLMs as part of a user-LLM conversation, transactional prompts run within predefined automated workflows and are refined to be robust and repeatable. Studying transactional prompts ofers vital insights into real-world LLM usage in software applications.<sup>3</sup>

Unlike conventional programming, LLM instructions realized in prompts are specified in unstructured texts that convey complex, hierarchical, and multi-faceted messages, using natural language to encode a mixture of instructions and information. To explore the expressive power of this new “natural programming language” and the structural, compositional, and linguistic mechanisms through which this semantic range is expressed, it is helpful to introduce structure into otherwise unstructured prompt texts. To this end, we define an ontology underlying prompt structure.

The ontology provides a systematic framework for investigating the complex semantics encoded in prompts and the diversity of expressive means employed to express it (Section 3). It helps to uncover underlying semantic and structural patterns within widely diverse prompt data, providing a foundation for the empirical investigation of prompts as a “programming language” for LLMs across diverse applications, languages, and modalities.

Our analysis of the resulting structured prompts (Section 5) reveals a rich diversity of prompt usage across languages, modalities, tasks and domains, exhibiting a Zipf-like distribution — typical of linguistic phenomena (Piantadosi, 2014; Linders and Louwerse, 2022) — with a prominent head and a much more diverse and nuanced long tail.

Importantly, our data collection and ontology are not intended to be final or exhaustive. Rather, they represent a starting point paving the way for further investigation by linguists, prompt researchers, and engineers, who can contribute complementary perspectives to the research.

The prompt collection we deliver is accompanied by an online user interface for browsing, searching, and exploring prompts by their various characteristics and components, as defined by the underlying ontology. Alongside the collection and the ontology, it is intended as a practical resource to inspire further investigation into this topic.

In sum, this work treats prompts as first-class objects for empirical scientific and linguistic investigation, and makes four main contributions:

(i) a large-scale dataset of 57.5K transactional prompts gathered from GitHub;

(ii) a structured prompt-ontology that captures the primary prompt features and components;

(iii) empirical analysis of the structured prompts, highlighting patterns in the way programmers use LLMs; and

(iv) a user interface for browsing and searching prompts by their properties and components to support further research.

We aim for these resources to facilitate linguistic and practical investigations into how humans interact with LLMs and how prompts’ structure interacts with the semantic space they construct when using the prompting language as part of software engineering — pragmatic, syntactic, and lexical variations they use, how they structure prompts to elicit outputs with specific forms and content, the strategies they employ to overcome the inherent ambiguity and underspecification of natural language, how they choose the language of the prompt (English vs. other languages), are all prompts similar stylistically due to style convergence induced by LLMs, in what ways prompting language diverges from ordinary human language, and other potential perspectives.

## 2 Collecting Transactional Prompts

Following standard practices (Liu et al., 2025; Li et al., 2022; Mir et al., 2021; Alon et al., 2019; Raychev et al., 2016), we collect prompts from Github repositories <sup>4</sup>, by looking for files that either invoke the chat.completion.create API or the PromptTemplate constructor from the LangChain package<sup>5</sup>, and attempt to extract the contents of the messages (for completion.create) or template (for PromptTemplate) parameters from each call-site.

The immediate content is often a formatted string or a variable name, which we then aim to resolve to the actual prompt content via static analysis of the code, recursively tracking string values across variable assignments and function calls. The cases where this resolution succeeds are then filtered using a set of heuristics to retain only semanticallycontentful prompts, and the filtered results are deduplicated. For each resulting prompt we further retain its metadata such as the repository name, file-path and URL, as well as the last commit date for the prompt. This process resulted in 57,640 unique prompts. Of these, 36,916 came from chat.completion.create and 20,724 from PromptTemplate. Details of prompt text extraction are available in the Appendix A. Details of filtering and deduplication are described in Appendix B.

## 3 The Prompts Ontology

In order to analyze prompts more deeply, we introduce an ontology outlining their main components and dimensions. Figure 1 shows a bird’s-eye view of our proposed ontology, with the specific fields explained shortly.

The ontology categories are grounded in three complementary sources: (1) Inherent prompt properties (e.g., prompts are texts written in a language; prompts by definition formulate a task. prompts serve to elicit certain output), (2) prior literature, e.g., prompting techniques (Jr et al., 2025; Schulhof et al., 2025), context-grounded vs. parametric prompting (Zhou et al., 2024; Sun et al., 2026), and (3) recurring lexical and structural components we identified through manual inspection of sampled prompts, such as explicit language mentions, semantically distinct instruction blocks etc. The categories capture orthogonal dimensions of the data and are not intended to constitute a mutually exclusive or collectively exhaustive taxonomy. Rather, the ontology is deliberately non-restrictive and extensible; dataset users may adopt, adapt, extend, or replace it as appropriate for their purposes. The utility of the proposed ontology is directly demonstrated by its application in the data annotation and the analysis we present (Section 5).

Languages: detected languages used in prompt texts and any explicit language mentions.

Task and domain: we track the tasks for which the prompt is intended, and its application domain. These have both coarse-grained and fine-grained categories.

Input characteristics: At the outset, prompts specify one or more of (1) overall high-level instructions (“answer the question provided by the user)"; (2) a question/task to be solved (“how many apples did John eat?"); (3) supporting context for 2. Each of these can be either hard-coded in the prompt or be a variable provided as input in each invocation. We identify the cases where each of these information types are read as input. For each identified input slot, we also retain information about its language, structure and modality, if available.

Output characteristics: Each prompt’s expected output is annotated for modality and for the requested structure, language and answer paradigm.

Prompt structure: Each prompt is represented as a list of role-messages (“System", “User", “Assistant"), and their associated texts.<sup>6</sup> For each message we list, beyond its text and role, also its detected language and languages explicitly mentioned within it. We also further break the message text into a sequence of individual instructions. Each instruction is associated with one of 42 semantic kinds (e.g. “role specification", “audience specification", “input content description". See Appendix E for the complete list). We explicitly mark negative instructions, and distinguish between central and auxiliary instructions.

Prompting techniques: For each prompt we extract a list of prompting techniques with records specifying which techniques are used in the prompt, with supporting evidence spans. The prompting techniques come from a pre-specified list of 12 techniques (e.g. “use of sections", “structured outputs", see Figure 1).

![](images/5547b2452b4b18e0673e65f63e01e3b41eebc6fda1b40b9f989ab602f60ab2c1.jpg)  
Figure 1: The Prompt Ontology Underlying the Empirical Analysis and the Structured Collection

Meta-data: Additionally, each object in the data includes an ID, a GitHub URL of the source file, timestamp of the last update, full prompt text (a concatenation of all individual message texts in the prompt) and its translation into English if not in English already.

For some fields (input and output structure, modality and variability, prompting techniques, instruction kinds, role), we defined the possible value inventories, although in some cases (e.g., structure fields) we instructed the model to enrich predefined values with additional detail (e.g., “Dictionary of items (‘From’: string, ‘To’: string)” rather than just “Dictionary”). For the remaining fields, values were generated by the LLM during annotation. For fields with especially large inventories (e.g., class, domain, input and output type), the values were then grouped into classes using the algorithm described in Appendix G.

## 4 Annotation, Quality Control and Error Analysis

The ontology annotation is performed using an LLM-based process. The prompts used for annotating the prompt collection, and the detailed data-model, are listed in Appendix J.

We iteratively tested, manually evaluated and refined annotation prompts on sample data. Concretely, we jointly performed human-in-the-loop checks: we sampled ≈ 100 items per major metadata category and manually inspected the automatic labels and their evidence spans. Any disagreements between the authors were resolved through peer discussions. These checks were used to refine the annotation prompts until labels became consistent, and the manual inspections showed consistently high agreement between the automatic labels and human judgments across categories.

To assess the resulting annotation quality, we manually evaluate additional 100 randomly selected data points (50 from each source) and perform error analysis across all fields.<sup>7</sup> In the error analysis, predicted labels and evidence spans were evaluated against the annotation guidelines provided to the model in the respective prompts (see Appendix J). Detected errors were then grouped into recurring categories and quantified to characterize the main failure modes. The error analysis was conducted by a single expert, due to the substantial time required for this kind of fine-grained manual review.

The error analysis shows generally strong performance across most fields, with many categories exceeding 90% accuracy (e.g., Prompt Language 93.0%, Domain 90.6%, Context Language/Modality 97%, Central vs. Meta 96.4%, Prompting Techniques 98.5%). However, several fields remain more challenging, especially Output Type (60.4%) and Directions Text (69.4%), with moderate error rates in Answer Paradigm (80.2%), Instruction Kinds (89.8%), and Context Structure (81.9%). The observed errors fall into several recurring groups: (1) hallucinations, where the model assigns attributes not grounded in the prompt (e.g., inventing output types, domains, or languages); (2) confusions between related categories, such as conflating prompt language with explicit language mentions, labeling restrictions as negative instructions, or reporting output structure instead of context or question structure; (3) omissions, where relevant elements are not detected (e.g., missing tasks, ignored placeholders, undetected context or question units); and (4) segmentation and granularity errors, including failure to split complex instruction blocks or grouping multiple output types into a single “complex” type. Notably, many errors arise when indirect inference is required—e.g., coreference resolution, common-sense reasoning, or domain knowledge (inferring the output type, not mentioned directly, from few-shot examples, recognizing a Python function signature as code etc.). The results suggest that performance degrades primarily in cases involving implicit information and fine-grained annotation distinctions.

## 5 Analysis

## 5.1 Language, Modality and Domain

Language. Which languages are prompts written in? Naturally, English is predominant, but to what extent? And how diverse are the other languages? Our analysis shows that the dataset encompasses prompt messages in 62 languages. English is overwhelmingly dominant, accounting for 84.66% of identifiable cases.<sup>8</sup> The other languages that constitute above 1% are Chinese, Korean, Spanish, Japanese and Portuguese. The following seven highly represented languages are mostly (but not only) European: French, Russian, German, Indonesian, Vietnamese, Polish, Italian and Dutch (see Figure 6 Appendix I).<sup>9</sup> The long-tail languages occurring below 100 times in the data, with Hindi at the beginning and Tagalog at the very end, are shown in Figure 7 (Appendix I).

Multilinguality. Next, we examine the presence of multilingualism. Only 6.3% of prompts (3,649) are multilingual, and of these, over 99% include English. Despite this English dominance in mixedlanguage queries, 8.19% of the total dataset (4721 prompts) consists of entirely non-English text.

Language Mentions. Beyond the language used to write the prompt, we also annotate the prompts for explicit language mentions—instances where a language is referenced but not necessarily used (e.g., “Translate this to Afrikaans"). This oftenoverlooked dimension of multilinguality reveals even greater linguistic diversity. In terms of total mentions, we observe 15,331 references to natural human languages. In terms of language diversity, while English leads (9,510 mentions), the remaining references cover 151 languages and dialects, a far broader range than is found in the languages directly used in prompt texts. The listing and distribution of these non-English mentioned languages is detailed in Figure 8 (Appendix I).

Modality. While it is obvious that in LLMs text is the dominant input and output modality, what other combinations of input and output modalities do we see in the data? And can we find even greater diversity in the long tail? Where is the predominance of text more pronounced: in input or in output?

In the overwhelming majority of cases, the input context modality (77.82% of the cases) and the output modality (over 97% of the cases) is text.<sup>10</sup> The distributions of non-text or mixed modality for input and output are shown on the Figures 9 and 10 (Appendix I) with images accounting for a much greater share than audio and video.

The frequent input-output modality combinations all have textual output and difer only in input: the prevailing combination is text→text (77.31% of all the data); with the other categories— ungrounded or undefined (18.98%), image→text (1.98%), image+text→text (0.66%), audio→text (0.23%) — lag far behind (Fig 11, Appendix I).

Domain. In which domains are transactional prompts most frequently used? Which domains are leading and which are in the long tail? Prompts with a specific domain that can be identified<sup>11</sup> constitute 57.38% of the prompts. This resulted in 77 distinct domains with a long-tail, Zipf-like distribution. The top leading domains are, in order: education & instruction, software development, business & commerce, healthcare & medical, technology, media & entertainment, finance & banking, creative writing & content-creation, human resources, arts & culture. Together, they cover almost half (49.58%) of all the domains in the data, and appear in 63.63% of the prompts with specific domains.<sup>12</sup> Mid-frequency domains include, for example, hospitality & food service, design & arts, personal services and philosophy, while low-frequency domains include urban development, historical studies, politics and others. See Appendix D for the full list of domains.

## 5.2 Structure and Semantics

Instruction Kinds. A prompt (or a message therein) can be interpreted as a sequence of instructions given to the LLM. But what is the semantic or functional structure of prompts? That is, what are the semantic types of instructions and their order in transactional prompts? Our ontological structure treats each message as a sequence of instruction items, each labeled with its semantic function.

Overall, the dataset includes 39,4875 such instruction blocks, 6.85 per prompt on average. The top 10 most frequent types are:

• Input context placeholder (16%)

• Constraint or restriction (11%)

• Output content requirements (9.7%)

• Output format requirement (9.7%)

• Role specification (8.1%)

• Input context description (7.1%)

• Central task/question description (6.7%)

• Central task/question (5.8%)

• Input contextual data (4.3%)

• Conditional instruction (2.9%)

Together, they cover 81.33% of all blocks in the data. Full statistics, as well as information about frequent ordering of units, are available in Appendix I, Figure 18, and Tables 10 & 11.

Core vs. Supporting Instructions. Generally, 18.2% of instructions represent the Central Task (the “core intent”), while the remaining 81.8% function as meta (supporting) instructions providing guidance on style, constraints, or formatting. Most of the core instructions focus on the task/question detailed descriptions (36%) or define the tasks/questions themselves (31.9%).

In the example below, the core instruction defines what the task is while the supporting ones add details by specifying how it should be performed: "Please create a learning plan in {language}." (core) "The plan should outline daily activities." (supporting). "Make sure to include detailed information about the specific programming languages and tools (like APIs) that will be used." (supporting) "Do not include learning of languages that I have already used." (supporting).

Only 4% of prompts consist exclusively of a central task with no metadata. These are typically short, non-grounded queries (e.g., “Explain how to write a window in Python”, “Name ten mammals") or where the context is not provided in the prompt text (e.g., “Identify products using the given images and generate key features for each product.”).

Negative Instructions. Negative instructions (telling the model what not to do) serve as a window into user mitigation of LLMs tendencies, such as verbosity, hallucination and diferent kinds of bias.

Roughly 31% of all prompts contain at least one negative instruction. Overall, negative instructions represent only 7% of the total instruction count in the dataset. The vast majority (89.5%) of the negative instructions are categorized as constraints or restrictions. These act as guardrails against undesirable behaviors, such as adding extra text beyond the requested output, or exceeding specific lengths. A smaller portion addresses output format (5.4%), content requirements (1.4%), and error handling (1.1%). Examples of negative instructions include (more in Appendix F):

• don’t make the answers too long (constraint/restriction)

• Ifyou encounter an exception, an efectless command, or find yourself in a loop, avoid repeating the same command and try something else to achieve the goal. (error handling instruction)

• Don’t translate the text to English. Keep it in Indonesian. (linguistic constraint/specification) Negative instructions very rarely form the core intent of the prompt (0.5% of cases). In these instances, the primary task is defined by what the model must avoid, e.g., “Do not answer the questions, simply provide a correct compute graph..." or “Do not respond to text, merely translate it."

Constraints. We define constraints as instructions involving restrictions, style/format requirements, design specifications, or negative directives. As research shifts toward evaluating how well LLMs adhere to nuanced requirements (Zhou et al., 2023; Lior et al., 2025), our dataset emerges as a source of naturally occurring constraints. Like negative instructions, the isolation of prompt constraints also ofers opportunities for linguistic investigations on the form and structure in which they are expressed.

Our analysis reveals a landscape of high complexity. Constraints represent 33.3% of all instruction blocks in the dataset. On average, a single prompt contains 2.28 constraints, but the “long tail" is significant — some transactional prompts contain up to 155 distinct rules. Furthemore, while 27.4% of the prompts are constraint-free, nearly 14% layer five or more constraints in a single query, signaling a demand for high-precision model control.

Unlike synthetic benchmarks, our data captures the “messy" and layered constraints actually deployed in real-world scenarios, making it a unique resource for investigating the limits of LLM steerability in real-world transactional environments.

Messages. The popular chat.completions .create API expects prompts as a sequence of role messages (“System", “User", “Assistant"). How do prompt writers use this interface? How many messages do they use, and what roles are assigned? The majority of prompts (64%) have two messages. Of these 96% are “system-user". 27% of the prompts have a single message, with “user" (70%) being far more frequent than “system" (28%). For three-message prompts (4%) the most popular sequences are “system-user-user” (26.41%), “system-user-assistant” (22.01%), “system-systemuser” (18.45%), “system-assistant-user” (13.07%). For prompts with more than three messages, the majority (52%) includes alternations of the “user” and “assistant” messages, alternatively with a system prompt at the beginning. (See details in Figure 16 Appendix I and Figure 17 Appendix I). It thus appears that the “system-user” duo has become the standard unit in transactional prompts. This suggests that developers largely view the System role as a static configuration layer and the User role as the dynamic input, rather than utilizing the API for complex, multi-turn role-play within a single template.

## 5.3 What Are Prompts Being Used For?

Tasks. What tasks are typically performed using LLMs? Are LLMs used more for standard NLP tasks or for other, non-NLP tasks? From these tasks (NLP vs. non-NLP), to what extent either is more pronounced? Are the tasks used across domains or are some of them limited to a specific domain? The distribution of NLP tasks in the data covers both NLP tasks (involving language understanding, generation and analysis) and tasks outside classical NLP (like data processing, analysis, multimodal, and structured-data tasks).

It is clear from the data that NLP tasks prevail: the top 4 (question answering, general text generation, information extraction, and summarization) cover over 48% of the data (see Figure 13, Appendix I). Non-NLP tasks occur mostly at moderate to low frequencies. In the long tail (tasks with under 30 cases) we see non-NLP tasks, such as education design, game strategy, state tracking, data privacy, policy generation, and system integration. The top 10 in long-tail and mid-frequency tasks are shown in Table 9 (Appendix I). We further see that each task appears in multiple domains, from 4 (game strategy) to 77 (question answering), and no task is purely domain-specific.

Inputs. It is common for prompts to have a question or main task that needs to be answered, either based on a context that is also provided, or based on parametric knowledge. Either of these components can be hard-coded into the prompt, or be a varying input to the prompt. In our data, 97.9% of the prompts included a question or a main task, and 89.2% included a supporting context. From these, the question/task is expected as input 70% of the time, and was hard-coded in the prompt for the remaining 30%. In contrast, a context is provided as input 94% of the time and is only hard coded 6% of the time. This suggests an (expected) tendency to perform the same task over varying contexts rather than performing varying tasks over a fixed context.

Grounding. LLMs may be expected to respond either based on their internal parametric knowledge, or based on grounding context provided to them. Which of these options is more prevalent in realworld transactional usage? And, in the case of grounded prompts, what kinds of contexts are used for grounding?

Our analysis suggests that 89.25% of all cases are grounded, i.e., performed based on a certain context rather than merely based on the LLM’s parametric knowledge.<sup>13</sup> In the vast majority of cases the type of the grounding context is text (that may include a variety of subtypes such as document, paragraph, sentence, abstract, tweet, proverb, caption, description, instruction, etc.). Other frequent types of context include a question, dialogue/conversation, code, table, image, numeric context, json. The distribution of top 10 input types across the top 10 tasks is shown in Figure 14 (Appendix I).

## 5.4 Prompting Techniques

Which prompting techniques are adopted by prompt writers, and to what extent?

Role assignment is by far the most frequently used technique, accounting for over 45% of all instances (Figure 19, Appendix I). Explicitly defining the role of the AI was one of the first well-known techniques, and was recommended by model developers. The data shows that this remains a widely adopted practice in prompt engineering, despite reports of its limited efectiveness in more recent models (Zheng et al., 2024b; Kim et al., 2025).

The next most commonly-used technique is structured output (12.9%), that is, requesting output in machine parseable formats (json, XML etc.). This technique is closely related to the output demonstrations technique (3.34%) including examples of how the output should be organized. Together, these constitute 16.24% of the cases. Also related is the sections technique (7.83%) dividing the prompt into clearly defined, marked sections. These demonstrate the significance of clearly-structured input and output specifications.

The next in frequency is decomposition via prompt (10.34%) meaning that the prompt specifies how to break the task down into sub-tasks or manageable steps. This technique is complemented by the much less frequent decomposition via LLM (0.56%) where the same decomposition is expected to be done by the LLM. Together, these techniques form 10.9% of the annotated techniques, suggesting the importance of solving complex tasks by breaking them down into more manageable units.

## 6 User Interface

To empower researchers to explore the prompt collection beyond our set of analyses, we provide a web-based UI designed for exploratory discovery and deep-dive analyses into subsets of the data. Users can filter prompts by ontology fields; search the dataset with semantic similarity; see and aggregate counts; inspect individual prompts and their annotated sections; and download prompts and ontological data for their filtered subsets. Additional information on the UI can be found in Appendix C.

## 7 Related Work

As interest in prompts as distinctly designed artifacts has grown, several datasets have emerged, encompassing both user prompts and transactional prompts.

User prompts are prompts that are intended to be used directly by users, and have significantly different characteristics than the transactional prompts we study herein. LMSYS-Chat-1M (Zheng et al., 2024a) is a dataset of 1M user-LLM conversations collected across 154 languages via the Vicuna demo and Chatbot Arena. WildChat (Zhao et al., 2024) is a multilingual, dataset of 1 million timestamped user-LLM conversations (over 2.5 million turns) collected via a ChatGPT/GPT-4 chatbot with explicit user consent, annotated with demographic metadata (state, country, and hashed IPs) to enable behavioral analysis. PROMPTEVALS (Vir et al., 2025) is a dataset of 2,087 LLM prompt templates from the LangChain Prompt Hub,<sup>14</sup> a repository of user-contributed prompts, containing a mix of user-prompts and transactional prompts. PROMPTEVALS is intended for training and evaluating “assertion guardrails", and has prompts spanning multiple domains including IT, finance, and healthcare. DevGPT (Xiao et al., 2023) is a dataset of 29,778 ChatGPT prompts and responses from software developers, collected from shared Chat-GPT conversations on GitHub and Hacker News for analysis of developers’ interactions with ChatGPT and their implications for AI-assisted programming. These datasets are diferent than ours in that they do not address transactional prompts in software.

Other researchers have studied Non-LLM prompts, for instance DifusionDB (Wang et al., 2023) and VidProM (Wang and Yang, 2024) that compile large-scale prompts for text-to-image and text-to-video generation, respectively. As proposed herein, such collection too would merit from structured representation and empirical analysis, compatible to ours, which is reserved for future research.

Finally, for Transactional Prompts, Pister et al. (2024) introduced PromptSet, a dataset of developer prompts with similar size and collection method to our own. However, they invest less efort in extraction and cleanup compared to us. As a result, as reported by Tafreshipour et al. (2025) and verified by us, the data contains many incomplete prompts or prompt fragments that are hard or impossible to analyze. They also do not provide analysis or structuring of the prompts beyond this raw string data. In spite of these limitations, researchers use PromptSet for exploration of diferent aspects of transactional prompts (Villamizar et al., 2025; Tafreshipour et al., 2025; Mao et al., 2025), as well as for development of prompt optimization tools (Rzig et al., 2025). Notably, Mao et al. (2025) construct their own small dataset, derived from PromptSet, to analyze real-world prompts that combine static content with dynamic placeholders such as “input”. After filtering, cleaning and deduplication, they extract 2,163 such prompts, in which they identified key components and categorized them into one of six semantic categories.

This highlights the interest in transactional prompts research and a clear community demand for larger, higher-quality resources for such research like the corpus presented in this work.

## 8 Conclusions

We present a large, real-world collection of transactional prompts; an ontology that captures both the structural components of prompts and their descriptive characteristics; and a web interface for their systematic exploration. These resources enable a range of applications, including linguistic and structural analysis of prompt texts, uncovering common conventions and “unspoken norms” of prompt composition, comparing recommended versus actual practices, and supporting multiple downstream applications, such as instruction-following and promptsensitivity research, more realistic benchmarking, structure-aware and linguistically-informed automated prompt optimization, multilingual prompt engineering and others. We present a preliminary empirical analysis that exemplifies the utility of the provided framework and illustrates some of the kinds of insights this data makes possible, and hope it inspires further interest in the systematic and methodological study of prompts as scientific and linguistic objects in the community.

## Limitations

Our search for prompts in GitHub files was limited to certain patterns (the chat.completion.create API call and the LangChain PromptTemplate class), whereas many other patterns are possible. Furthermore, we only considered Python files. Thus, we do not present an unbiased sample of transactional prompts, and the observed trends in our analysis may be biased towards a subset of the prompt space defined by users who chose to use these APIs. Additionally, while prompts are continuously created and updated, our dataset represents a snapshot at the time of collection. Therefore, rather than being exhaustive, our work paves the way to future eforts that could expand the collection, potentially making it dynamic by regularly incorporating new entries over time, and include broader search patterns and additional programming languages.

The prompt structuring annotations were performed by an LLM, and, though annotation prompts were iteratively tested, manually evaluated and refined on sample data, each annotation result could not be manually verified individually. As a result, the data may still contain some errors or noise despite the eforts to ensure annotation quality.

Finally, the analysis presented in this work only scratches the surface, and many more quantitative and qualitative investigations are possible, including detailed linguistic analysis, diachronic comparisons (changes over time), cross-lingual or domain and task-specific exploration. We encourage the community to contribute to expanding the data and improving annotation accuracy and to continue and deepen the research in the field of transactional prompts.

## Acknowledgments

Work on this project was supported by a VATAT grant from the Planning and Budgeting Committee of the Council for Higher Education in Israel, Kamin grant by the Israel Innovation Authority (IIA) and ISF grant number 670/23.

## References

Uri Alon, Shaked Brody, Omer Levy, and Eran Yahav. 2019. code2seq: Generating sequences from structured representations of code. In International Conference on Learning Representations.

Negar Arabzadeh and Ebrahim Bagheri. 2025. VAP3: Variation-aware prompt performance prediction. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval.

Jaime E. Cuellar, Óscar Moreno-Martínez, Paula Sofía Torres Rodríguez, Jaime Andrés Pavlich-Mariscal, Andrés Felipe Micán Castiblanco, and Juan Guillermo Torres Hurtado. 2026. Trusting ChatGPT? When a subtle variation in the prompt can significantly alter the results. Journal of Artificial Intelligence and Technology.

Nicolás Hidalgo, Pablo Alzaga Sáez, Nicolas Meneses, Víctor Reyes, and Erika Rosas. 2025. Prompt’s evolution for language model-driven data generation. Applied Sciences.

Sullam Jeoung, Yueyan Chen, Yi Zhang, Shuai Wang, Haibo Ding, and Lin Lee Cheong. 2025. PromptPrism: A linguistically-inspired taxonomy for prompts. ArXiv, abs/2505.12592.

E. G. Santana Jr, Gabriel Benjamin, Melissa Araujo, Harrison Santos, David Freitas, Eduardo Almeida, Paulo Anselmo da M. S. Neto, Jiawei Li, Jina Chun, and Iftekhar Ahmed. 2025. Which prompting technique should i use? An empirical investigation of prompting techniques for software engineering tasks. Preprint, arXiv:2506.05614.

Gurusha Juneja, Gautam Jajoo, Nagarajan Natarajan, Hua Li, Jian Jiao, and Amit Sharma. 2025. Task facet learning: A structured approach to prompt optimization. Preprint, arXiv:2406.10504.

Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. Demonstrate-search-predict: Composing retrieval and language models for knowledgeintensive NLP. arXiv preprint arXiv:2212.14024.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2023. DSPy: Compiling declarative language model calls into self-improving pipelines. Preprint, arXiv:2310.03714.

Junseok Kim, Nakyeong Yang, and Kyomin Jung. 2025. Persona is a double-edged sword: Rethinking the impact of role-play prompts in zero-shot reasoning tasks. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 848–862, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam. 2024. Native vs non-native language prompting: A comparative analysis. ArXiv, abs/2409.07054.

Alina Leidinger, Robert van Rooij, and Ekaterina Shutova. 2023. The language of prompting: What linguistic properties make a prompt successful? ArXiv, abs/2311.01967.

Zhiyu Li, Shuai Lu, Daya Guo, Nan Duan, Shailesh Jannu, Grant Jenks, Deep Majumder, Jared Green, Alexey Svyatkovskiy, Shengyu Fu, and Neel Sundaresan. 2022. Automating code review activities by largescale pre-training. Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering.

Guido Linders and Max Louwerse. 2022. Zipf’s law revisited: Spoken dialog, linguistic units, parameters, and the principle of least efort. Psychonomic Bulletin & Review, 30.

Gili Lior, Asaf Yehudai, Ariel Gera, and Liat Ein-Dor. 2025. WildIFEval: Instruction following in the wild. Preprint, arXiv:2503.06573.

Jingjing Liu, Zeming Liu, Zihao Cheng, Mengliang He, Xiaoming Shi, Yuhang Guo, Xiangrong Zhu, Yuanfang Guo, Yunhong Wang, and Haifeng Wang. 2025. RepoDebug: Repository-level multi-task and multi-language debugging evaluation of large language models. Preprint, arXiv:2509.04078.

Yuetian Mao, Junjie He, and Chunyang Chen. 2025. From prompts to templates: A systematic prompt template analysis for real-world LLMapps. Preprint, arXiv:2504.02052.

A. M. Mir, E. Latoskinas, and G. Gousios. 2021. Many-Types4Py: A benchmark python dataset for machine learning-based type inference. In IEEE/ACM 18th International Conference on Mining Software Repositories (MSR), pages 585–589. IEEE Computer Society.

Rithesh Murthy, Ming Zhu, Liangwei Yang, Jielin Qiu, Juntao Tan, Shelby Heinecke, Caiming Xiong, Silvio Savarese, and Huan Wang. 2025. Promptomatix: An automatic prompt optimization framework for large language models. Preprint, arXiv:2507.14241.

Steven T. Piantadosi. 2014. Zipf’s word frequency law in natural language: A critical review and future directions. Psychonomic Bulletin & Review, 21(5):1112–1130.

Kaiser Pister, Dhruba Jyoti Paul, Ishan Joshi, and Patrick Brophy. 2024. PromptSet: A programmer’s prompting dataset. In Proceedings of the 1st International Workshop on Large Language Models for Code, LLM4Code ’24, page 62–69. ACM.

Veselin Raychev, Pavol Bielik, and Martin Vechev. 2016. Probabilistic model for code with decision trees. pages 731–747.

Dhia Elhaq Rzig, Dhruba Jyoti Paul, Kaiser Pister, Jordan Henkel, and Foyzul Hassan. 2025. An empirically-grounded tool for automatic prompt linting and repair: A case study on bias, vulnerability, and optimization in developer prompts. ArXiv, abs/2501.12521.

Martina Saletta and Claudio Ferretti. 2024. Exploring the prompt space of large language models through evolutionary sampling. Proceedings of the Genetic and Evolutionary Computation Conference.

Gabriel Machado Santos, Rita Maria Silva Julia, and Marcelo Zanchetta do Nascimento. 2025. Diverse prompts: Illuminating the prompt space of large language models with MAP-Elites. 2025 IEEE Congress on Evolutionary Computation (CEC), pages 1–8.

Sander Schulhof, Michael Ilie, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei Si, Yinheng Li, Aayush Gupta, HyoJung Han, Sevien Schulhof, Pranav Sandeep Dulepet, Saurav Vidyadhara,

Dayeon Ki, Sweta Agrawal, Chau Pham, Gerson Kroiz, Feileen Li, Hudson Tao, Ashay Srivastava, and 12 others. 2025. The prompt report: A systematic survey of prompt engineering techniques. Preprint, arXiv:2406.06608.

Kaiser Sun, Fan Bai, and Mark Dredze. 2026. Task matters: Knowledge requirements shape LLM responses to context-memory conflict. Preprint, arXiv:2506.06485.

Mahan Tafreshipour, Aaron Imani, Eric Huang, Eduardo Almeida, Thomas Zimmermann, and Iftekhar Ahmed. 2025. Prompting in the wild: An empirical study of prompt evolution in software repositories. Preprint, arXiv:2412.17298.

Shubham Vatsal, Harsh Dubey, and Aditi Singh. 2025. Multilingual prompt engineering in large language models: A survey across NLP tasks. ArXiv, abs/2505.11665.

Hugo Villamizar, Jannik Fischbach, Alexander Korn, Andreas Vogelsang, and Daniel Mendez. 2025. Prompts as software engineering artifacts: A research agenda and preliminary findings. Preprint, arXiv:2509.17548.

Reya Vir, Shreya Shankar, Harrison Chase, Will Fu-Hinthorn, and Aditya Parameswaran. 2025. PROMPTEVALS: A dataset of assertions and guardrails for custom production large language model pipelines. Preprint, arXiv:2504.14738.

Jan Philip Wahle, Terry Ruas, Yang Xu, and Bela Gipp. 2024. Paraphrase types elicit prompt engineering capabilities. ArXiv, abs/2406.19898.

Wenhao Wang and Yi Yang. 2024. VidProM: A millionscale real prompt-gallery dataset for text-to-video difusion models. Preprint, arXiv:2403.06098.

Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. 2023. DifusionDB: A large-scale prompt gallery dataset for text-to-image generative models. Preprint, arXiv:2210.14896.

Tao Xiao, Christoph Treude, Hideaki Hata, and Kenichi Matsumoto. 2023. DevGPT: Studying developer-ChatGPT conversations. 2024 IEEE/ACM 21st International Conference on Mining Software Repositories (MSR), pages 227–230.

Lechen Zhang, Yusheng Zhou, Tolga Ergen, Lajanugen Logeswaran, Moontae Lee, and David Jurgens. 2025. Cross-lingual prompt steerability: Towards accurate and robust LLM behavior across languages. ArXiv, abs/2512.02841.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024. WildChat: 1M ChatGPT interaction logs in the wild. Preprint, arXiv:2405.01470.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. 2024a. LMSYS-Chat-1M: A large-scale real-world LLM conversation dataset. Preprint, arXiv:2309.11998.

Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, and David Jurgens. 2024b. When "a helpful assistant" is not really helpful: Personas in system prompts do not improve performances of large language models. Preprint, arXiv:2311.10054.

Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. Preprint, arXiv:2311.07911.

Sizhe Zhou, Sha Li, Yu Meng, Yizhu Jiao, Heng Ji, and Jiawei Han. 2024. Establishing knowledge preference in language models. ArXiv, abs/2407.13048.

## A Details of GitHub Prompt Collection

To identify Python files containing prompts, we used the GitHub Code Search API to systematically query repositories for code involving promptrelated functions. Specifically, we searched for files that invoked chat.completions.create, a common method used in prompt construction for language models, and the langchain PromptTemplate class, a class used to generate prompts from a string template and variables. For each search result we collected metadata such as repository name, file path, and URL. This way we end up with 95806 objects from 95434 filepaths from 51393 repositories.

Building on our initial URL collection, we then implement an extraction pipeline that pulls actual prompt text out of each discovered file. We iterate through each file record, and retrieve file contents via GitHub REST API, decoding the Base64 encoded result into plain Python source code. We then parse that source with Python’s ast module to locate all occurrences of our target API call, chat.completions.create, or the langchain PromptTemplate class. Then we employ a multistep process aiming to extract the full contents of the "messages" or "template" parameter (for chat.completions.create or PromptTemplate respectively) - even when they’re built up across several statements. Specifically, using the ast module, we track variable assignments and resolve all arguments and keyword values used within the API call (whenever possible). If the API call is inside a function, we find where that function is called and replace its parameters with the actual values passed into the function at each call site - using both the current file and related imports.

Next, we check for remaining unresolved variables. If the entire messages field or a specific content field inside a messages list is a variable placeholder, the actual values of these variables are looked up in the current file and other related files in the repository. At every step, if a variable is reassigned to diferent values before diferent calls, our extraction logic will capture each distinct value, yielding multiple versions of the prompt.

Figure 2 illustrates some stages of this process.

Finally, for each prompt, we look up the date of most recent commit that changed any of the lines contributing to it, in order to estimate when the prompt was last modified. To ensure correctness of the extraction pipeline, we manually evaluated a subset of 1,000 prompts before scaling to the full dataset. This extraction process leaves us initially with 145553 objects.

Next we perform filtering and deduplication (see Appendix B for details).

## B Filtering And Deduplication Details

We filter out objects where the extracted texts are empty, contain invalid values (e.g., ‘error’, ‘n/a’, ‘nan’), or consist only of unresolved variables or placeholders, identifiable via string matching or regular expressions. Next, we apply a series of additional heuristics to filter out prompts that lack readable or meaningful content. Specifically, we remove prompts that consist solely of punctuation or whose language cannot be reliably detected by the langdetect library. For prompts identified as English, we use spaCy to parse the text and check for the presence of verbs or auxiliaries—signals of syntactic structure and potential informativeness. Prompts with such features are retained. Prompts in clearly detected non-English languages are also kept. These heuristics help exclude most empty, malformed, or placeholder-based prompts while preserving those that exhibit valid language or meaningful structure.

The deduplication procedure is as follows. Exact repeats, defined as objects with the same file path, extracted prompt text, and timestamp, are removed after the first instance. Prompt texts that occur more than once but in diferent files or at diferent times are retained, but marked as duplicates. This yields a dataset of 85,209 objects. In this version of the dataset, we identify 8,169 groups of duplicate prompts. Group sizes range from 2 to 580 prompts. Most groups (63.55%) contain only 2 duplicate prompt instances, followed by 15.79% with 3 instances, 6.24% with 4 instances, 7.28% with 5–7 instances, and the remaining 7.14% with 8 or more instances.<sup>15</sup>

Additionally, we create a fully deduplicated version of the dataset in which cross-file repeats are removed after the first instance. The analysis and statistics reported in this article are based on the fully deduplicated dataset version.<sup>16</sup>

![](images/f498160e64b5842227ccd292f1cb6dcc9047b67157fbd6297f7548a7c5935571.jpg)  
Figure 2: Prompt Extraction Flow. This figure illustrates prompt text extraction by tracing variables across the repository. Static variables (such as topic and mood\_prompt) are successfully resolved, while dynamic variables requiring runtime execution (such as tweets\_prompt) remain unresolved in the final extracted text.

## C User Interface: Layout and Functionality

The layout and functionality of each UI component are as follows. At the top are a semantic free-text search field, a filter box showing all active filters, Show prompts and Download prompts buttons. Below them, the page displays a set of boxes for diferent ontology fields (task, domain, language, modality, prompting-techniques, etc.). Each box lists all available values for the field with the corresponding prompt counts, which update dynamically as filters are applied. It shows coarse categories by default. Clicking an eye icon next to each coarse category reveals its fine-grained subcategories.

Users can select multiple values in each box and switch between match-all and match-any mode. Clicking the checkbox next to a value selects or deselects it. The "Apply filters" button applies the selected filters. The filter box ofers per-filter removal and a Clear allfilters button. Counts across all boxes update dynamically as filters change.

The Show prompts button opens a paginated drawer containing a stack of prompt cards. Each card displays the prompt text and includes a Show spans toggle that marks structural components - directions, context, question/task, output description and diferent semantic kinds of instruction blocks - using colors. A color legend on each card explains the span colors. Hovering a legend entry highlights the corresponding spans in the prompt for convenience.

Free-text search uses embedding-based semantic similarity: the system embeds each query with the embeddinggemma-300m-ONNX model, which is also used to precompute embeddings for all the prompts, and retrieves relevant prompts by cosine similarity.

The header displays the number of prompts matching the current filters. Clicking the Download prompts button exports the full dataset entries for the selected prompts.

Figures 3 and 4 illustrate some features of the user interface.

## D Domains

Below all the domains in our data are listed along with their number and percentage.

1. education & instruction - 4182 (8.42%)

2. software development - 3863 (7.78%)

3. business & commerce - 2790 (5.62%)

4. healthcare & medical - 2485 (5.00%)

5. technology - 2468 (4.97%)

6. media & entertainment - 2040 (4.11%)

![](images/76daa659e2d01f134d48779638a8386615b6c8be0147a6bf658d68804266fcc3.jpg)

Figure 3: User Interface. The top section features a free-text search field, a filter box displaying currently active filters, and buttons for prompt display and download. Below, ontology field boxes list available values alongside dynamically updating counts. The Languages box on the right demonstrates selected values. The Domains box on the left shows displayed subcategories. The header shows the total number of currently selected prompts.  
![](images/cd4f261a502f06269dda481e9c3de0904d01e4f3401be2a4027e54902ab14297.jpg)  
Figure 4: User Interface. Paginated prompts view with displayed spans.

7. finance & banking - 1933 (3.89%)

8. creative writing & content creation - 1728 (3.48%)

16. legal & regulatory - 1052 (2.12%)

9. human resources - 1607 (3.24%)

10. arts & culture - 1522 (3.07%)

11. food & beverages - 1302 (2.62%)

18. gaming - 1005 (2.02%)

12. personal development - 1281 (2.58%)

13. artificial intelligence & machine learning - 1175 (2.37%)

14. digital media - 1054 (2.12%)

15. other - 1054 (2.12%)

17. research, scholarship & publications - 1031 (2.08%)

19. travel & leisure - 984 (1.98%)

20. customer support - 898 (1.81%)

21. retail & consumer goods - 804 (1.62%)

22. language services - 800 (1.61%)

23. data management - 776 (1.56%)

24. data analytics - 653 (1.32%)

25. marketing & advertising - 645 (1.30%)

26. security & cybersecurity - 624 (1.26%)

27. government & policy - 564 (1.14%)

28. physical sciences - 515 (1.04%)

29. mathematics - 502 (1.01%)

30. cultural studies - 463 (0.93%)

31. geography & locations - 455 (0.92%)

32. sports - 449 (0.90%)

33. computer engineering & architecture - 408 (0.82%)

34. hospitality & food service - 372 (0.75%)

35. design & arts - 366 (0.74%)

36. manufacturing & industry - 304 (0.61%)

37. agriculture & ecology - 264 (0.53%)

38. information retrieval - 248 (0.50%)

39. communication & language - 244 (0.49%)

40. personal services - 241 (0.49%)

41. philosophy - 233 (0.47%)

42. religion & spirituality - 220 (0.44%)

43. transportation - 219 (0.44%)

44. project management - 218 (0.44%)

45. document management - 218 (0.44%)

46. hardware & engineering - 215 (0.43%)

47. academic services & administration - 214 (0.43%)

48. sustainability & environment - 201 (0.40%)

49. social communication - 194 (0.39%)

50. user experience & design - 189 (0.38%)

51. safety - 182 (0.37%)

52. biological sciences - 160 (0.32%)

53. home & interior design - 157 (0.32%)

54. logistics & supply chain - 139 (0.28%)

55. social issues & policies - 135 (0.27%)

56. veterinary services - 131 (0.26%)

57. energy management - 123 (0.25%)

58. assessment & testing - 119 (0.24%)

59. recreation & leisure - 109 (0.22%)

60. it operations - 93 (0.19%)

61. community & volunteering - 88 (0.18%)

62. public services - 86 (0.17%)

63. scientific analysis - 85 (0.17%)

64. environmental management - 84 (0.17%)

65. data management & analysis - 82 (0.17%)

66. quality assurance - 81 (0.16%)

67. security & defense - 70 (0.14%)

68. environmental science - 68 (0.14%)

69. administrative services - 66 (0.13%)

70. general & miscellaneous - 64 (0.13%)

71. data security and quality - 59 (0.12%)

72. urban development - 54 (0.11%)

73. process modeling & monitoring - 51 (0.10%)

74. research & development - 42 (0.08%)

75. languages - 30 (0.06%)

76. historical studies - 20 (0.04%)

77. politics - 3 (0.01%)

## E Instruction Block Kinds

In this section we provide the full list of 42 semantic kinds of instruction blocks used in the ontology:

\- input context placeholder

\- constraint/restriction

\- output content requirement

\- output format requirement

\- role specification

\- input context description

\- central task/question

\- central task/question description

\- input contextual data

\- conditional instruction

\- question/task data/placeholder

\- reasoning instructions

\- question/task description

\- style specification

\- central task/question placeholder

\- examples

\- expertise/skills requirements

\- assistant response

\- example clarification

\- evaluation criteria

\- linguistic constraint/specification

\- audience specification

\- function call instruction

\- scope specification

\- error handling instruction

\- design specification

\- scene setting

\- interaction guideline

\- default behavior instruction

\- encouragement

\- instruction to avoid errors

\- date reference

\- confirmation request

\- greeting

\- prompt variable/placeholder

\- disclaimer requirement

\- placeholder

\- prompt

\- input format specification

\- command instruction

\- clarification instruction

\- other

## F Negative Instructions: Examples

Below are additional examples of negative instructions of various semantic types found in the dataset (see §5.2 for details).Their respective semantic kinds are given in parentheses.

\- “Response Format: Response should be always in clean json format — don’t use the word json or any extra.” (output format requirement)

\- “The lyrics should be narrative-driven, avoiding simplistic rhyming patterns.” (output content requirements)

\- “do not get confused between the symbols like decimal(".") and comma(",")” (instruction to avoid errors)

\- “Ensure your style of speech is not influenced by the style and prose of the other users.” (style specification)

\- “Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI.” (role specification)

## G Value Clustering Procedure

We first reduced surface-form term variation by grouping near-duplicate terms with fuzzy string matching (fuzz.ratio from fuzzywuzzy) and merging terms whose similarity exceeded a fixed threshold (e.g., 94). These coarse synonym groups were then refined with an LLM (o3-mini), which was prompted to identify subsets of representative terms that were full synonyms or duplicates and to merge only those cases for which it had high confidence. Any LLM-identified group was expanded to include all terms from the corresponding synonym groups identified previously via fuzzy string matching. Terms not assigned to any group remained singletons. This procedure was repeated for a fixed number of iterations or until the groups stabilized.

For clustering, we next applied the LLM to an initial batch of up to 500 representative terms resulting from the synonym consolidation step, and asked it to group them into a limited number of classes, assigning each term to exactly one class and producing informative labels. Terms that were unassigned or assigned to multiple classes were marked as unclassified and carried over to subsequent batches. The remaining terms were then assigned to previously created classes batch-wise, while the LLM was allowed to introduce a limited number of new classes per batch. At the end of the process, any still-unclassified terms were assigned to other. Clusters above a size threshold (e.g., >100 terms) were reclustered using the same procedure used to obtain the initial class list. Finally, highly similar class names were merged by fuzzy matching, small clusters (e.g., fewer than five terms) were included into other, and hallucinated terms not present in the original list were removed.

## H Error Analysis: Charts and Tables

Tables 1–7 summarize the results of manual error analysis of (see Section 4). Table 8 and Figure 5 report the annotation accuracy per ontology field based on the error analysis.

<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Error Type</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>%</td></tr><tr><td rowspan=3 colspan=1>PromptLanguage</td><td rowspan=1 colspan=1>Correct</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>147</td><td rowspan=1 colspan=1>93.04%</td></tr><tr><td rowspan=1 colspan=1>Hallucinated language</td><td rowspan=1 colspan=1>The model assigns a specific naturallanguage (typically, English) toplaceholders whose specific linguisticvalue is unknown.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>6.96%</td></tr><tr><td rowspan=1 colspan=1>Total errors</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>6.96%</td></tr><tr><td rowspan=6 colspan=1></td><td rowspan=1 colspan=1>Correct</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>126</td><td rowspan=1 colspan=1>88.11%</td></tr><tr><td rowspan=1 colspan=1>Conflation with promptlanguage</td><td rowspan=1 colspan=1>The model confuses the language of theprompt with an explicit languagemention.</td><td rowspan=1 colspan=1>E.g. for a prompt in English the modelhallucinates a explicit mention ofEnglish</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>7.69%</td></tr><tr><td rowspan=1 colspan=1>Placeholder mentionignored</td><td rowspan=1 colspan=1>A placeholder language mention isignored.</td><td rowspan=1 colspan=1>E.g. {target_languge} is not labeled asa languge mention.</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2.10%</td></tr><tr><td rowspan=1 colspan=1>Not a language mention</td><td rowspan=1 colspan=1>A non-language mention labeled as alanguage mention.</td><td rowspan=1 colspan=1>E.g. &quot;Chinese medicine&quot; labeled as alanguge mention</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1.40%</td></tr><tr><td rowspan=1 colspan=1>Hallucinated mention</td><td rowspan=1 colspan=1>The model invents a non-existentlanguage mention.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.70%</td></tr><tr><td rowspan=1 colspan=1>Total errors</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>11.89%</td></tr></table>

Table 1: Error Analysis Summary: Language

<table><tr><td colspan="1" rowspan="1">Field</td><td colspan="1" rowspan="1">Error Type</td><td colspan="1" rowspan="1">Description</td><td colspan="1" rowspan="1">Example</td><td colspan="1" rowspan="1">Count</td><td colspan="1" rowspan="1">%</td></tr><tr><td colspan="1" rowspan="6">Task</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">125</td><td colspan="1" rowspan="1">89.93%</td></tr><tr><td colspan="1" rowspan="1">Wrong task class</td><td colspan="1" rowspan="1">The fine-grained task is correct, but thetask class is wrong</td><td colspan="1" rowspan="1">E.g., sentiment analysis assigned to"emotion detection," althoughsentiment analysis does not necessarilyinvolve emotions.</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1.44%</td></tr><tr><td colspan="1" rowspan="1">Bias toward commontasks</td><td colspan="1" rowspan="1">The model incorrectly identifies acommon task.</td><td colspan="1" rowspan="1">E.g., QA or general text generation areidentified instead of a less commoncorrect task.</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">5.04%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated task</td><td colspan="1" rowspan="1">The model invents a task when itcannot be determined from the prompt,or adds an irrelevant task to anotherwise correct list.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">2.88%</td></tr><tr><td colspan="1" rowspan="1">Missing task</td><td colspan="1" rowspan="1">A relevant task is omitted.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.72%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">10.07%</td></tr><tr><td colspan="1" rowspan="4">Domain</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">115</td><td colspan="1" rowspan="1">90.55%</td></tr><tr><td colspan="1" rowspan="1">Undefined domain</td><td colspan="1" rowspan="1">The model fails to identify a domaineven though it can be inferred from theprompt text.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">6.30%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated domain</td><td colspan="1" rowspan="1">The model assigns a domain thatcannot be inferred from the prompt text,or adds an irrelevant domain to anotherwise correct list.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">3.15%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">9.45%</td></tr><tr><td colspan="1" rowspan="4">Central vs.MetaInstructions</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">702</td><td colspan="1" rowspan="1">96.43%</td></tr><tr><td colspan="1" rowspan="1">Meta misclassified ascentral</td><td colspan="1" rowspan="1">A meta-instruction labeled as a centralinstruction.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">1.79%</td></tr><tr><td colspan="1" rowspan="1">Central misclassified asmeta</td><td colspan="1" rowspan="1">A central instruction labeled as ameta-instruction.</td><td colspan="1" rowspan="1">Typical of central tasks incorporatedinto role assignment, e.g. "You are asearch assistant designed to help usersby summarizing web pages.",</td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">1.79%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">3.57%</td></tr><tr><td colspan="1" rowspan="4">NegativeInstructions</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">718</td><td colspan="1" rowspan="1">98.63%</td></tr><tr><td colspan="1" rowspan="1">Restriction misclassifiedas negative</td><td colspan="1" rowspan="1">Restrictions or constraints withoutexplicit negation labeled as negativeinstructions.</td><td colspan="1" rowspan="1">E.g., "Your answer must be within thescope of the information provided" or“Restrict the questions to the contextinformation provided" are labeled asnegative instructions.</td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">0.82%</td></tr><tr><td colspan="1" rowspan="1">Ignoring negation</td><td colspan="1" rowspan="1">The model fails to recognize a negationexplicitly stated in the instruction.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.55%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">1.37%</td></tr><tr><td colspan="1" rowspan="10">InstructionSemanticKinds</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">654</td><td colspan="1" rowspan="1">89.84%</td></tr><tr><td colspan="1" rowspan="1">Too complex instructionblocks</td><td colspan="1" rowspan="1">Complex instructions containingmultiple elements of distinct types .</td><td colspan="1" rowspan="1">E.g., "Write a summary ofapproximately 200 words, that giveskey insights for articles: {url_list}"identified as one instruction, while itincludes a central task, a lengthrestriction, an output contentrequirement and an input contextplaceholder.</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">2.20%</td></tr><tr><td colspan="1" rowspan="1">Output format vs.content</td><td colspan="1" rowspan="1">Output content requirements aremislabeled as output formatrequirements and vice versa.</td><td colspan="1" rowspan="1">E.g., "The answer has to contain ONLYthe translation itself' labeled as aformat requirement.</td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">1.79%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled output prefix</td><td colspan="1" rowspan="1">Output prefixes at the end of a promptare mislabeled as central task or outputformat requirement.</td><td colspan="1" rowspan="1">E.g. such phrases as "Answer:","Output:", "Assistant:" at the end of theprompt.</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">0.96%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled evaluationcriteria</td><td colspan="1" rowspan="1">Evaluation criteria labeled as anothertype.</td><td colspan="1" rowspan="1">E.g. "A higher Shelf Life Scoreindicates that the product is sellingfaster" labeled as example clarification(even though the prompt contains noexamples).</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">0.96%</td></tr><tr><td colspan="1" rowspan="1">Conditional instructionmisclassified asrestriction</td><td colspan="1" rowspan="1">Conditional instructions labeled asrestrictions.</td><td colspan="1" rowspan="1">Typical of instructions to admitunanswerability, e.g. "If you don'tknow the answer, say None"</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.55%</td></tr><tr><td colspan="1" rowspan="1">Task/questiondescription mislabled asrole specification</td><td colspan="1" rowspan="1">This is typical of cases wheretask/question description is expressedas an assertion in the second person(rather than instruction or question).</td><td colspan="1" rowspan="1">E.g. "You are extracting data from apublic financial document"</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">0.41%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled rolespecification</td><td colspan="1" rowspan="1">Role specification islabeled as anothertype</td><td colspan="1" rowspan="1">E.g "Your name is {ai_name}" labeledas otput content requirement.</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">0.41%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">2.88%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">74</td><td colspan="1" rowspan="1">10.16%</td></tr><tr><td colspan="1" rowspan="6">ContextEvidence</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">208</td><td colspan="1" rowspan="1">91.63%</td></tr><tr><td colspan="1" rowspan="1">Missing context</td><td colspan="1" rowspan="1">A context unit not detected by themodel.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">3.52%</td></tr><tr><td colspan="1" rowspan="1">Output formatmislabeled as context</td><td colspan="1" rowspan="1">Output format demonstrations orrequirements labeled as context.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">3.08%</td></tr><tr><td colspan="1" rowspan="1">Question mislabeled ascontext</td><td colspan="1" rowspan="1">Input question labeled as context.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.44%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">1.32%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">19</td><td colspan="1" rowspan="1">8.37%</td></tr><tr><td colspan="1" rowspan="6">ContextType</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">206</td><td colspan="1" rowspan="1">90.75%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated type</td><td colspan="1" rowspan="1">The model assigns a type that cannot bedetermined from the prompt text.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">3.52%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled type(inferable from prompt)</td><td colspan="1" rowspan="1">An undefined or incorrect type wherethe correct type can be inferred fromthe prompt.</td><td colspan="1" rowspan="1">E.g. the type is mentioned elsewhere inthe prompt or implied by inputexamples.</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">3.08%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled type(inferable fromparametric knowledge)</td><td colspan="1" rowspan="1">An undefined or incorrect type wherethe correct type can be inferred fromgeneral knowledge.</td><td colspan="1" rowspan="1">E.g. the model fails to classify the typeof a Python function signature as"code".</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">1.76%</td></tr><tr><td colspan="1" rowspan="1">Short text instead of text</td><td colspan="1" rowspan="1">The model predicts “short text" wherethe prompt does not specify contextlength.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.88%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">9.25%</td></tr><tr><td colspan="1" rowspan="7">ContextStructure</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">186</td><td colspan="1" rowspan="1">81.94%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated structure</td><td colspan="1" rowspan="1">The model assigns a structure where itcannot be determined from the prompt.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">24</td><td colspan="1" rowspan="1">10.57%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled structure(inferable from prompt)</td><td colspan="1" rowspan="1">An undefined or incorrect structurewhere the correct structure can beinferred from the prompt.</td><td colspan="1" rowspan="1">E.g., the structure is mentionedelsewhere in the prompt ordemonstrated in input examples.</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">3.52%</td></tr><tr><td colspan="1" rowspan="1">Mislabeled structure(inferable fromparametric knowledge)</td><td colspan="1" rowspan="1">An undefined or incorrect structureeven though the correct structure can beinferred from general knowledge.</td><td colspan="1" rowspan="1">E.g., the structure label of a chat historyshould "list" because it containsmultiple items- dialogue turns.</td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">2.20%</td></tr><tr><td colspan="1" rowspan="1">Output structure insteadof context structure</td><td colspan="1" rowspan="1">The model reports the output structureinstead of the context structure.</td><td colspan="1" rowspan="1">E.g., "dictionary" where dictionary isthe expected output structure.</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.88%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.88%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">41</td><td colspan="1" rowspan="1">18.06%</td></tr><tr><td colspan="1" rowspan="5">ContextLanguage</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">221</td><td colspan="1" rowspan="1">97.36%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated language</td><td colspan="1" rowspan="1">The model assigns a language wherenone applies (e.g., numeric inputlabeled as a specific language).</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.88%</td></tr><tr><td colspan="1" rowspan="1">Undefined language(inferable fromexamples)</td><td colspan="1" rowspan="1">The model fails to deternine a languagewhere it can be inferred from inputexamples.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.88%</td></tr><tr><td colspan="1" rowspan="1">Undefined language(inferable from commonsense)</td><td colspan="1" rowspan="1">Undefined language though it can beinferred from general knowledge andcommon sense.</td><td colspan="1" rowspan="1">E.g., a placeholder value in a Japaneseprompt is lmost ikely also in Japanese.</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.88%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">2.64%</td></tr><tr><td colspan="1" rowspan="6">ContextModality</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">221</td><td colspan="1" rowspan="1">97.36%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated modality(misreading the prompt)</td><td colspan="1" rowspan="1">Wrong modality label based on anincorrect interpretation of the prompt.</td><td colspan="1" rowspan="1">E.g.., "context about a video" labeledas video modality.</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">1.32%</td></tr><tr><td colspan="1" rowspan="1">Modality undefined(inferable from commonsense)</td><td colspan="1" rowspan="1">Undefined modality where it can beinferred from general knowledge.</td><td colspan="1" rowspan="1">E.g., a placeholder inside a formattedstring is clearly text.</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.44%</td></tr><tr><td colspan="1" rowspan="1">Modality undefined(explicitly stated inprompt)</td><td colspan="1" rowspan="1">The model returns "undefined" wherethe modality is directly stated in theprompt.</td><td colspan="1" rowspan="1">E.g., instructions such as "Analyze thetext. . ." clearly indicate textual context</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.44%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated modality</td><td colspan="1" rowspan="1">The model assigns a modality thatcannot be determined from the prompttext.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.44%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">2.64%</td></tr><tr><td colspan="1" rowspan="5">ContextVariability</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">95</td><td colspan="1" rowspan="1">95.00%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated variability</td><td colspan="1" rowspan="1">The model assigns a variability labelwhere it cannot be determined from theprompt.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">3.00%</td></tr><tr><td colspan="1" rowspan="1">Incorrect variability dueto misidentified context</td><td colspan="1" rowspan="1">Wrong variability type as a result ofincorrectly identified context.</td><td colspan="1" rowspan="1">E. g., predicting "none" (meaning thatcontext is missing) when context isactually present but was not identified</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">1.00%</td></tr><tr><td colspan="1" rowspan="1">Incorrect variability dueto ignored placeholderin the context</td><td colspan="1" rowspan="1">Predicting "fixed" where contextincludes a placeholder (i.e. the correctlabel is "varying").</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">1.00%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">5.00%</td></tr><tr><td colspan="1" rowspan="5">DirectionsText</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100</td><td colspan="1" rowspan="1">69.44%</td></tr><tr><td colspan="1" rowspan="1">Meta-instructionslabeled as directions</td><td colspan="1" rowspan="1">Some of the meta-instructions areincorrectly labeled as directions.</td><td colspan="1" rowspan="1">E.g., instructions to admitunanswerability, reasoning instructionsetc.</td><td colspan="1" rowspan="1">41</td><td colspan="1" rowspan="1">28.47%</td></tr><tr><td colspan="1" rowspan="1">Input questionmislabeled as directions</td><td colspan="1" rowspan="1">The input question and directionssometimes overlap. This only counts asan error when clear directions, distinctfrom the question, are also present.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1.39%</td></tr><tr><td colspan="1" rowspan="1">Missing directions</td><td colspan="1" rowspan="1">Directions clearly present in the text arenot detected by the model.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.69%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">44</td><td colspan="1" rowspan="1">30.56%</td></tr><tr><td colspan="1" rowspan="2">DirectionsLanguage</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">144</td><td colspan="1" rowspan="1">100.00%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0.00%</td></tr><tr><td colspan="1" rowspan="6">QuestionEvidence</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">137</td><td colspan="1" rowspan="1">91.95%</td></tr><tr><td colspan="1" rowspan="1">Missing question unit</td><td colspan="1" rowspan="1">A question unit was not detected by themodel.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">4.03%</td></tr><tr><td colspan="1" rowspan="1">Directions mislabeled asquestion</td><td colspan="1" rowspan="1">Directions labeled as question units.Only counts as an error when distinctfrom the question.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1.34%</td></tr><tr><td colspan="1" rowspan="1">Role instructionsmislabeled as question</td><td colspan="1" rowspan="1">Role instructions labeled as questionunits. Only counts as an error whenthey are distinct.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.67%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">2.01%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">8.05%</td></tr><tr><td colspan="1" rowspan="3">QuestionType</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">142</td><td colspan="1" rowspan="1">95.30%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated type</td><td colspan="1" rowspan="1">The model assigns a type when itcannot be determined based on theprompt text</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">4.70%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">4.70%</td></tr><tr><td colspan="1" rowspan="5">QuestionStructure</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">138</td><td colspan="1" rowspan="1">92.62%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated structure</td><td colspan="1" rowspan="1">The model assigns a structure where itcannot be determined from the prompt.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">4.03%</td></tr><tr><td colspan="1" rowspan="1">Output structuremislabeled as questionstructure</td><td colspan="1" rowspan="1">The model reports the output structureinstead of the question structure.</td><td colspan="1" rowspan="1">E.g., "dictionary"where dictionary isthe expected output structure.</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">2.68%</td></tr><tr><td colspan="1" rowspan="1">Undefined or wrongstructure (inferablefrom prompt)</td><td colspan="1" rowspan="1">The model predicts an undefined orincorrect structure even though thecorrect structure can be inferred fromthe prompt.</td><td colspan="1" rowspan="1">E.g., the structure is mentionedelsewhere in the prompt ordemonstrated in input examples</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.67%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">7.38%</td></tr><tr><td colspan="1" rowspan="3">QuestionLanguage</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">147</td><td colspan="1" rowspan="1">98.66%</td></tr><tr><td colspan="1" rowspan="1">Undefined language(inferable from commonsense)</td><td colspan="1" rowspan="1">The model fails to identify a languagewhere it can be inferred from generalknowledge or common sense.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1.34%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1.34%</td></tr><tr><td colspan="1" rowspan="4">QuestionVariability</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">94</td><td colspan="1" rowspan="1">94.00%</td></tr><tr><td colspan="1" rowspan="1">Incorrect variability dueto misidentified question</td><td colspan="1" rowspan="1">Errors caused by undetected ormislabeled question units.</td><td colspan="1" rowspan="1">E.g. the model failed to identify aquestion unit with a placeholder; as aresult "fixed" was predicted instead of"varying".</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">4.00%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">2.00%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">6.00%</td></tr><tr><td colspan="1" rowspan="9">OutputType</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">67</td><td colspan="1" rowspan="1">60.36%</td></tr><tr><td colspan="1" rowspan="1">Short text instead of text</td><td colspan="1" rowspan="1">The model returns "short text" whenthe prompt does not specify length.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">15.32%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated type</td><td colspan="1" rowspan="1">The model returns an output type thatcannot be inferred from the prompt.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">9.01%</td></tr><tr><td colspan="1" rowspan="1">Erroneous complextypes</td><td colspan="1" rowspan="1">Multiple types are grouped whileshould be split.</td><td colspan="1" rowspan="1">E.g., "complex: code, explanation"though these are distinct output units.</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">6.31%</td></tr><tr><td colspan="1" rowspan="1">Ignored types</td><td colspan="1" rowspan="1">The model omits one or more outputtypes present in the prompt.</td><td colspan="1" rowspan="1">E.g., the model returns only "text"while output includes text and JSON.</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">3.60%</td></tr><tr><td colspan="1" rowspan="1">Too specific</td><td colspan="1" rowspan="1">The model hallucinates a more specifictype than indicated in the prompt</td><td colspan="1" rowspan="1">E.g. the model returns “article"whenthe prompt only suggests that the outputis text, without specifying the type.</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.90%</td></tr><tr><td colspan="1" rowspan="1">Type mismatch</td><td colspan="1" rowspan="1">The prompt explicitly specifies a type,but the model predicts a different one.</td><td colspan="1" rowspan="1">E.g., prompt says “sentence", modelreturns “paragraph”.</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.90%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">3.60%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">44</td><td colspan="1" rowspan="1">39.64%</td></tr><tr><td colspan="1" rowspan="3">OutputStructure</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">106</td><td colspan="1" rowspan="1">95.50%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated structure</td><td colspan="1" rowspan="1">The model hallucinates a structurewhere it cannot be determined.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">4.50%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">4.50%</td></tr><tr><td colspan="1" rowspan="3">OutputLanguage</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">104</td><td colspan="1" rowspan="1">93.69%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated outputlanguage</td><td colspan="1" rowspan="1">The model hallucinates a languagewhere it cannot be determined</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">6.31%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">6.31%</td></tr><tr><td colspan="1" rowspan="4">OutputModality</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">107</td><td colspan="1" rowspan="1">96.40%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated modality</td><td colspan="1" rowspan="1">The model hallucinates a modality thatcannot be determined.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">2.70%</td></tr><tr><td colspan="1" rowspan="1">Undefined modality</td><td colspan="1" rowspan="1">The modality is "undefined" while itcan be inferred from the prompt text.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.90%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">3.60%</td></tr><tr><td colspan="1" rowspan="8">AnswerParadigm</td><td colspan="1" rowspan="1">Correct</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">89</td><td colspan="1" rowspan="1">80.18%</td></tr><tr><td colspan="1" rowspan="1">Free generation insteadof language or styletransfer</td><td colspan="1" rowspan="1">The model returns “free generation"when the prompt explicitly requeststranslation or style transfer.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">6.31%</td></tr><tr><td colspan="1" rowspan="1">Free generation insteadof summary/paraphrase</td><td colspan="1" rowspan="1">The model returns “free generation"when the prompt explicitly requestssummarization or paraphrasing.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">2.70%</td></tr><tr><td colspan="1" rowspan="1">Binary answer instead offree generation</td><td colspan="1" rowspan="1">The model outputs a binary answerwith a "don't know" option when thetask requires free generation.</td><td colspan="1" rowspan="1">This error is typical of prompts withinstructions to admit ananswerability,e.g. “If you don't know, say None"</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">2.70%</td></tr><tr><td colspan="1" rowspan="1">Undefined answerparadigm</td><td colspan="1" rowspan="1">The the answer paradigm is “undefined"when it is inferrable from the prompt.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">2.70%</td></tr><tr><td colspan="1" rowspan="1">Hallucinated answerparadigm</td><td colspan="1" rowspan="1">The model invents an answer paradigmwhere it cannot be determined.</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">0.90%</td></tr><tr><td colspan="1" rowspan="1">Other</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">4.50%</td></tr><tr><td colspan="1" rowspan="1">Total errors</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">22</td><td colspan="1" rowspan="1">19.82%</td></tr></table>

Table 2: Error Analysis Summary: Task&Domain

Table 3: Error Analysis Summary: Instruction Sequences

Table 4: Error Analysis Summary: Input Context

Table 5: Error Analysis Summary: Input Directions&Question

Table 6: Error Analysis Summary: Output
<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Error Type</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>%</td></tr><tr><td rowspan=10 colspan=1>PromptingTechniques</td><td rowspan=1 colspan=1>Correct</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1182</td><td rowspan=1 colspan=1>98.50%</td></tr><tr><td rowspan=1 colspan=1>Ignored role assignment</td><td rowspan=1 colspan=1>The model ignores explicit roleassignment.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.33%</td></tr><tr><td rowspan=1 colspan=1>Ignored outputdemonstrations</td><td rowspan=1 colspan=1>The model ignores examples ordemonstrations.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.33%</td></tr><tr><td rowspan=1 colspan=1>Ignored decomposition</td><td rowspan=1 colspan=1>The model ignores decompositioninstructions in the prompt.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.25%</td></tr><tr><td rowspan=1 colspan=1>Ignored sections</td><td rowspan=1 colspan=1>The model overlooks explicitsegmentation into sections.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.25%</td></tr><tr><td rowspan=1 colspan=1>Hallucinated audiencespecification</td><td rowspan=1 colspan=1>The model invents an audiencespecification not present in the prompt .</td><td rowspan=1 colspan=1>E.g., &quot;use simple words that a3-year-old understands&quot;: &quot;a 3-year-old&quot;is misinterpreted as the audience.</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.08%</td></tr><tr><td rowspan=1 colspan=1>Hallucinated promptchaining</td><td rowspan=1 colspan=1>The model assumes a previous outputwas generated by another prompt, whileno evidence supports this.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.08%</td></tr><tr><td rowspan=1 colspan=1>Hallucinated structuredoutput</td><td rowspan=1 colspan=1>The model hallucinates structuredoutput instructions.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.08%</td></tr><tr><td rowspan=1 colspan=1>Few-shot vs. outputdemonstrationsconfusion</td><td rowspan=1 colspan=1>The model confuses few-shot(input-output) examples with outputdemonstrations.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.08%</td></tr><tr><td rowspan=1 colspan=1>Total errors</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>1.50%</td></tr></table>

Table 7: Error Analysis Summary: Prompting Techniques

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Number of Evaluated Units</td></tr><tr><td rowspan=2 colspan=1>Language</td><td rowspan=1 colspan=1>Prompt Language</td><td rowspan=1 colspan=1>93.04%</td><td rowspan=1 colspan=1>158</td></tr><tr><td rowspan=1 colspan=1>Explicit Language Mentions</td><td rowspan=1 colspan=1>88.11%</td><td rowspan=1 colspan=1>143</td></tr><tr><td rowspan=2 colspan=1>Task&amp;Domain</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>89.93%</td><td rowspan=1 colspan=1>139</td></tr><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>90.55%</td><td rowspan=1 colspan=1>127</td></tr><tr><td rowspan=3 colspan=1>Instruction Sequences</td><td rowspan=1 colspan=1>Instruction Kinds</td><td rowspan=1 colspan=1>89.84%</td><td rowspan=1 colspan=1>728</td></tr><tr><td rowspan=1 colspan=1>Central vs. Meta</td><td rowspan=1 colspan=1>96.43%</td><td rowspan=1 colspan=1>728</td></tr><tr><td rowspan=1 colspan=1>Negative Instructions</td><td rowspan=1 colspan=1>98.63%</td><td rowspan=1 colspan=1>728</td></tr><tr><td rowspan=6 colspan=1>Input Context</td><td rowspan=1 colspan=1>Context Evidence</td><td rowspan=1 colspan=1>91.63%</td><td rowspan=1 colspan=1>227</td></tr><tr><td rowspan=1 colspan=1>Context Type</td><td rowspan=1 colspan=1>90.75%</td><td rowspan=1 colspan=1>227</td></tr><tr><td rowspan=1 colspan=1>Context Structure</td><td rowspan=1 colspan=1>81.94%</td><td rowspan=1 colspan=1>227</td></tr><tr><td rowspan=1 colspan=1>Context Language</td><td rowspan=1 colspan=1>97.36%</td><td rowspan=1 colspan=1>227</td></tr><tr><td rowspan=1 colspan=1>Context Modality</td><td rowspan=1 colspan=1>97.36%</td><td rowspan=1 colspan=1>227</td></tr><tr><td rowspan=1 colspan=1>Context Variability</td><td rowspan=1 colspan=1>95.00%</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=2 colspan=1>Input Directions</td><td rowspan=1 colspan=1>Directions Text</td><td rowspan=1 colspan=1>69.44%</td><td rowspan=1 colspan=1>144</td></tr><tr><td rowspan=1 colspan=1>Directions Language</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>144</td></tr><tr><td rowspan=5 colspan=1>Input Question</td><td rowspan=1 colspan=1>Question Evidence</td><td rowspan=1 colspan=1>91.95%</td><td rowspan=1 colspan=1>149</td></tr><tr><td rowspan=1 colspan=1>Question Type</td><td rowspan=1 colspan=1>95.30%</td><td rowspan=1 colspan=1>149</td></tr><tr><td rowspan=1 colspan=1>Question Structure</td><td rowspan=1 colspan=1>92.62%</td><td rowspan=1 colspan=1>149</td></tr><tr><td rowspan=1 colspan=1>Question Language</td><td rowspan=1 colspan=1>98.66%</td><td rowspan=1 colspan=1>149</td></tr><tr><td rowspan=1 colspan=1>Question Variability</td><td rowspan=1 colspan=1>94.00%</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=5 colspan=1>Output</td><td rowspan=1 colspan=1>Output Type</td><td rowspan=1 colspan=1>60.36%</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>Output Structure</td><td rowspan=1 colspan=1>95.50%</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>Output Language</td><td rowspan=1 colspan=1>93.69%</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>Output Modality</td><td rowspan=1 colspan=1>96.40%</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>Answer Paradigm</td><td rowspan=1 colspan=1>80.18%</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>Prompting Techniques</td><td rowspan=1 colspan=1>Prompting Techniques</td><td rowspan=1 colspan=1>98.50%</td><td rowspan=1 colspan=1>1200</td></tr></table>

Table 8: Annotation Accuracy Per Field (based on error analysis results)

![](images/a7d6c36c1f49c7b5ee529dd155098087ba796b8c127c1d60a6a04e27e755ea1e.jpg)  
Figure 5: Annotation Accuracy Per Field (based on error analysis results)

## I Analysis: Charts and Tables

Tables 9-11 and Figures 6-19 below illustrate the results of the analysis presented in Section 5.

<table><tr><td rowspan=1 colspan=6>Task Cases</td></tr><tr><td rowspan=1 colspan=2>Top 10 tasks</td><td rowspan=1 colspan=2>Mid-frequency Tasks</td><td rowspan=1 colspan=2>Long-tail Tasks</td></tr><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Count</td></tr><tr><td rowspan=1 colspan=1>question_answering</td><td rowspan=1 colspan=1>14176</td><td rowspan=1 colspan=1>ranking</td><td rowspan=1 colspan=1>428</td><td rowspan=1 colspan=1>state_tracking</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>general_text_generation</td><td rowspan=1 colspan=1>11359</td><td rowspan=1 colspan=1>code_transformation</td><td rowspan=1 colspan=1>393</td><td rowspan=1 colspan=1>system_integration</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>information_extraction</td><td rowspan=1 colspan=1>6764</td><td rowspan=1 colspan=1>text_analysis</td><td rowspan=1 colspan=1>387</td><td rowspan=1 colspan=1>style_analysis</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1>summarization</td><td rowspan=1 colspan=1>6496</td><td rowspan=1 colspan=1>image_processing</td><td rowspan=1 colspan=1>253</td><td rowspan=1 colspan=1>game_strategy</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>classification</td><td rowspan=1 colspan=1>4969</td><td rowspan=1 colspan=1>diagnosis</td><td rowspan=1 colspan=1>228</td><td rowspan=1 colspan=1>knowledge_management</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>code_generation</td><td rowspan=1 colspan=1>2908</td><td rowspan=1 colspan=1>dialogue management</td><td rowspan=1 colspan=1>217</td><td rowspan=1 colspan=1>task_formulation</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>explanatory_and_instructional_generation</td><td rowspan=1 colspan=1>2157</td><td rowspan=1 colspan=1>creative_and_narrative_generation</td><td rowspan=1 colspan=1>184</td><td rowspan=1 colspan=1>data_management</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>planning</td><td rowspan=1 colspan=1>2134</td><td rowspan=1 colspan=1>data_cleaning</td><td rowspan=1 colspan=1>181</td><td rowspan=1 colspan=1>natural_language_understanding</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>dialogue_and_response_generation</td><td rowspan=1 colspan=1>1864</td><td rowspan=1 colspan=1>parsing</td><td rowspan=1 colspan=1>174</td><td rowspan=1 colspan=1>policy_generation</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>recommendation</td><td rowspan=1 colspan=1>1710</td><td rowspan=1 colspan=1>speech_processing</td><td rowspan=1 colspan=1>158</td><td rowspan=1 colspan=1>information_fusion</td><td rowspan=1 colspan=1>3</td></tr></table>

Table 9: Task Distribution in the Dataset: Top, Mid-Frequency, and Long-Tail

![](images/67194257ce15ee0eabed4e183608e73d4c57aa3ad52b0b8481bdde3f8df0db04.jpg)  
Figure 6: Most frequent prompt languages in the dataset (top 14, ≥1% each)

![](images/863847ad2ac253b7245ec27cb7079f4158f4969d1778148fde899bca8cd2b6fa.jpg)  
Figure 7: Long-tail languages occurring below 100 times in the data

![](images/108b2193a90cb93c00fcc8717425b0349234b79b8403c3bed32ce2b170b99d51.jpg)  
Figure 8: Explicit Language Mentions (excluding English). Abbreviation key: Armenian = HY; Bahasa Indonesia = ID; Basque = EU; Galician = GL; Igbo = IG; Kannada = KN; Kurdish = KU; Kyrgyz = KY; Latvian = LV; Luxembourgish = LB; Macedonian = MK; Maltese = MT; Minang = MIN; Pashto = PS; Punjabi = PA; Tagalog = TL; Welsh = CY; Albanian = SQ; Amharic = AM; Burmese = MY; Corsican = CO; Georgian = KA; Gujarati = GU; Haitian Creole = HT; Hausa = HA; Khmer = KM; Kirundi = RN; Lao = LO; Mongolian = MN; Sinhala = SI; Tajik = TG; Xhosa = XH; Zulu = ZU; Belarusian = BE; Breton = BR; Cebuano = CEB; Chichewa = NY; Ebonics = EN-EB; Esperanto = EO; Frisian = FY; Hawaiian = HAW; Hmong = HMN; Isan = TH-ISAN; Joseon = KO-JOSEON; Kinyarwanda = RW; Libras = SGN-BR; Luganda = LG; Malagasy = MG; Maori = MI; Norse = NON; Odia = OR; Ottoman Turkish = OTA; Samoan = SM; Sesotho = ST; Shayari = HI-POETIC; Shona = SN; Sindhi = SD; Sundanese = SU; Tatar = TT; Toki Pona = TOK; Uyghur = UG; Yiddish = YI; Acholi = ACH; Assamese = AS; Ateso = TEO; Baatonum = BBA; Bambara = BM; Biblical Greek = GRC; Bobo = BOB; Dendi = DDN; Elfish = ART; Finglish = FI-EN; Fongbe = FON; Fula = FF; Gaelic = GD; Jamaican Patois = JAM; Kenyan = EN-KE; Kyoto dialect (Japanese) = JA-KYOTO; Lugbara = LGG; Luhshootseed = LUT; Montenegrin = CNR; Native American = NAI; Nigerian = EN-NG; Runyankole = NYN; Scottish = EN-SC; Tetun = TET; Tunisian Darija = AEB; Turkmen = TK

![](images/dd669dbd15c5c7ace9f4806a27b729a6fbf867831eb1118965b43de263e283bd.jpg)  
Figure 9: Input non-text modality combinations

![](images/c615cad0b9b05a316cdcddbbc1592767b6ec84e2bfae00193d18d07a1b58a517.jpg)  
Figure 10: Output non-text modality combinations

![](images/28b463f87c47ca39017a71b460bbb97c805d6c795c2fdee3e3dbc807ecf7d466.jpg)  
Figure 11: Main input-output modality combinations

![](images/fee86589fd6584a3e49d39bf982fe4baa317c70e40a7b46a77bbccd7daf8b6d2.jpg)  
Figure 12: Long-tail input–output modality combinations (less than 0.1% each). The inner circle indicates the input; the outer circle shows the output. Text=TXT; image=IMG, audio=AUD, video=AUD, all 4 modalities=ALL.

![](images/4417874e6bd233335156410b1f8bd57f36b97e3238851efd4a22eb422f06368e.jpg)  
Figure 13: Top four tasks covering over 48% of the data

![](images/7614dfa39c492b7a0581813986d3a4d273db406337fc8cd4c04b76abb53b7088.jpg)  
Figure 14: Distribution of the top 10 input types across the top 10 tasks in the collection.

![](images/f436db565a7fd9829830cd77ee35d8444c4243ad3ec7717426999ce9051c8362.jpg)

Figure 15: Distribution of prompt text lengths in the dataset  
![](images/b8ba146180e19a33b22cf908d46736dc102816d6b22ef1c00831e2fdbc0a6395.jpg)  
Figure 16: Number of messages per prompt (for chat.completions.create data only).

![](images/9bf228b701c90197a062ecebe673d494c9af2d23ddcb9b729ca1ec1af4845f8c.jpg)  
Figure 17: Prompt role sequences by number of messages.

![](images/e223ba29bc4decd177980177a064ddf49b704e710a169a46ebe57c40c8b865f6.jpg)  
Figure 18: Semantic instruction type frequencies.

<table><tr><td rowspan=1 colspan=1>Sequence</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Num Blocks</td></tr><tr><td rowspan=1 colspan=1>output_content_requirements → constraint/restriction</td><td rowspan=1 colspan=1>10452</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>input_context_placeholder → output_content_requirements</td><td rowspan=1 colspan=1>9703</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description → role_specification</td><td rowspan=1 colspan=1>9230</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>input_context_placeholder→output_content_requirements→constraint/restriction</td><td rowspan=1 colspan=1>5038</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description → output_format_requirement → role_specification</td><td rowspan=1 colspan=1>3739</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>role_specification → output_format_requirement → input_context_description</td><td rowspan=1 colspan=1>3433</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description → role_specification → output_format_requirement →input_context_description</td><td rowspan=1 colspan=1>1983</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description → output_format_requirement → role_specification →input_context_description</td><td rowspan=1 colspan=1>1442</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>input_context_placeholder → output_content_requirements → constraint/restriction →conditional_instruction</td><td rowspan=1 colspan=1>1311</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description → output_format_requirement → role_specification →input_context_description → input_context_placeholder</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description → role_specification → output_format_requirement →input_context_description → input_context_placeholder</td><td rowspan=1 colspan=1>494</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>output_format_requirement → input_context_description → input_context_placeholder →output_content_requirements → constraint/restriction</td><td rowspan=1 colspan=1>426</td><td rowspan=1 colspan=1>5</td></tr></table>

Table 10: Most frequent sequences of semantic types for instruction block chains of length 2–5.

<table><tr><td rowspan=1 colspan=1>Combination</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Num Blocks</td></tr><tr><td rowspan=1 colspan=1>input_context_placeholder; role_specification</td><td rowspan=1 colspan=1>20687</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>input_context_placeholder; output_format_requirement</td><td rowspan=1 colspan=1>19111</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>constraint/restriction; input_context_placeholder</td><td rowspan=1 colspan=1>17657</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>constraint/restriction; input_context_placeholder; output_format_requirement</td><td rowspan=1 colspan=1>10344</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>constraint/restriction; input_context_placeholder; role_specification</td><td rowspan=1 colspan=1>10016</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>input_context_placeholder; output_format_requirement; role_specification</td><td rowspan=1 colspan=1>9969</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>constraint/restriction;      input_context_placeholder;      output_format_requirement;role_specification</td><td rowspan=1 colspan=1>5884</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description; constraint/restriction; input_context_placeholder; out-put_format_requirement</td><td rowspan=1 colspan=1>5825</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>constraint/restriction; input_context_placeholder; output_content_requirements; out-put_format_requirement</td><td rowspan=1 colspan=1>5591</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description; constraint/restriction; input_context_placeholder; out-put_format_requirement; role_specification</td><td rowspan=1 colspan=1>3462</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>constraint/restriction; input_context_placeholder; output_content_requirements; out-put_format_requirement; role_specification</td><td rowspan=1 colspan=1>3290</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>central_task/question_description; constraint/restriction; input_context_placeholder; out-put_content_requirements; output_format_requirement</td><td rowspan=1 colspan=1>3208</td><td rowspan=1 colspan=1>5</td></tr></table>

Table 11: Most frequent sets of semantic types for instruction block chains of length 2–5 (regardless of order).

![](images/5bb11e1415f9f9e87e8a833da1e6253eb3c6359bc8cd9bd4a4de1b5f64769145.jpg)  
Figure 19: Distribution of prompting techniques in the dataset.

## J Data Model And Annotation Prompts

To structure the annotated prompts as described in Section 3 we use the following data model:

from dataclasses import dataclass from typing import List, Dict, Optional, Literal

@dataclass class TaskInfo: task\_class: str task: str subtask: str

@dataclass class DomainInfo: domain\_class: str domain: str

@dataclass class Granular: fine\_category: str coarse\_category: str

@dataclass   
class LangInfo: language: str orig\_text: str translated\_text: Optional[str]   
@dataclass   
class ExplicitLangMention: language: str mention: str   
@dataclass   
class Evidence: text: str type: Literal["description", "direct\_content"]   
@dataclass   
class TypedInstruction: instruction\_kind: str instruction: str is\_central: bool is\_negative: bool negative\_instructions\_explanation: Optional[str]   
@dataclass   
class AnalyzedMessage: languages: List[LangInfo] explicit\_language\_mentions: List[ExplicitLangMention] instruction\_sequence: List[TypedInstruction]   
@dataclass   
class AnalyzedPromptMessage: role: str original\_text: str prompt\_text: str analyzed: AnalyzedMessage   
@dataclass   
class InputContextInfo: context\_evidence: Evidence context\_type: Granular context\_structure: Granular context\_modality: Literal["text", "audio", "image", "video", ↩→ "undefined"] context\_language: List[str]   
@dataclass   
class DirectionInfo: directions\_text: str direction\_language: List[str]   
@dataclass   
class InputQuestionInfo: question\_evidence: Evidence question\_language: List[str] question\_structure: Granular question\_type: Granular   
@dataclass   
class InputInfo: context\_variability: Literal["fixed", "varying", "none", ↩→ "undefined"] question\_variability: Literal["fixed", "varying", ↩→ "undefined"] direction:List[DirectionInfo] context: List[InputContextInfo] question: List[InputQuestionInfo]   
@dataclass   
class OutputUnitInfo: output\_type: Granular modality: Literal["text", "audio", "image", "video", ↩→ "undefined"] description: str description\_source: Literal["extracted", "generated", ↩→ "undefined"] structure: Granular answer\_paradigm: Granular output\_language: List[str]   
@dataclass   
class UsedPromptingTechnique: technique: str reasoning: str evidence: List[str]   
@dataclass   
class NonUsedPromptingTechnique: technique: str reasoning: str   
@dataclass   
class PromptData: res\_id: int github\_url: str is\_duplicate: bool update\_last: int duplicate\_id: Optional[str] prompt\_messages: List[AnalyzedPromptMessage] prompt\_text: str full\_translation: Optional[str] task: List[TaskInfo] domain: List[DomainInfo] input: InputInfo output: List[OutputUnitInfo] instruction\_sequence: List[TypedInstruction]

```yaml
central_instructions: List[str]
meta_instructions: List[str]
negative_instructions: List[str]
used_prompting_techniques:
↩→ List[UsedPromptingTechnique]
non_used_prompting_techniques:
↩→ List[NonUsedPromptingTechnique]
```

For each metadata category ( language, task, domain, input, output, instruction sequence, prompting techniques), we applied the same annotation pipeline. To reduce cost, we used the OpenAI Batch API with gpt-4.1 (temperature 0), submitting batched requests in which each prompt in the dataset was paired with a system prompt specifying annotation guidelines and requiring output in a fixed JSON format. Some categories required multiple prompts. The system prompts are specified below. Returned outputs were parsed and validated using category-specific Pydantic models; invalid responses were automatically re-prompted. Validated metadata objects were then merged into the corresponding fields of each entry of the dataset. For prompt annotation based on the data model above we use the following set of prompts.

```python
#strings to fill placeholders in the prompting technique system
prompt
technique_details = {
"chain_of_thought": {
"brief":
"The prompt explicitly asks the model to reason before
producing a final answer",
"detailed":
("Look for phrases that instruct the model to ‘think
through' or ‘reason step-by-step' or similar **explicit
instructions to the LLM to reason before delivering its
final answer**. "
"Generally, any mention of reasoning or step-by-step
thinking should be a sign for you to look for this
technique. "
"The reasoning should occur **before** the final
answer, not after.
"The instruction to provide an explanation as such doesn
not mean chain-of-though. Only if the explanation is
required **before final answer**. "
"""Ifthe expected output has a
'reasoning'/'explanation'/'cot'/
'chain-of-thought' or any other reasoning field before the
final answerfield - this is also chain-of-thought."""
"Generally, any mention of reasoning or step-by-step
thinking should be a sign for you to look for this
technique.
"IMPORTANT!! There is **no need for the phrase 'step
by step' to appear explicitly**."
),
"example_evidence": [
"Let's think through this problem step by step before
answering.",
"First, let's think about this logically",
"Let's work this out in a step by step way to be sure we
have the right answer",
"Before answering, briefly outline your reasoning for
this answer",
"Include a 'reasoning' field that explains the reason for
your answer",
```

"Output format: id: <question id>, reasoning: <explain   
your thinking process>, correct\_answer: <your final   
answer>",   
"Your output should have three fields, 'reasoning', 'x'   
and 'y', where 'reasoning' is a string explaining the   
answer, 'x' is ... and 'y' is ..."   
"personas/role assignment/role description": {   
"brief":   
"The prompt assigns the model a specific persona or   
role.",   
"detailed"   
("Check for instructions of the form ‘You are an   
expert. . . ', ‘Act as if you are. . . ', or "   
"‘Your role is. . . '. The evidence must name the persona   
or role explicitly. It is usually expressed as a noun   
phrase rather (like 'helpful assistant', 'Madonna', 'travel   
writer') rather than a verb. "   
"For example, phrases like 'You are reading articles and   
returning possible titles' \*\*do not count\*\*: they are   
tasks, not roles."   
),   
"example\_evidence": [   
"You are an expert doctor specializing in cardiology.",   
"Act as if you are a seasoned software engineer.",   
"Pretend you are a shepherd and write a limerick about   
llamas."   
]   
"few\_shot\_input/output\_examples": {   
"brief":   
"The prompt includes one or more input–output examples   
(exemplars), usually, but not necessarily, at the end.",   
"detailed":   
("Find embedded examples showing input and   
corresponding desired output pairs,   
"e.g. ‘Example 1: Input: X → Output: Y'. These act as   
demonstrations that guide the LLM to accomplish a task.   
"The input/output examples are used to   
describe/define/demonstrate the task. They are \*\*no   
used to demonstrate the format\*\*. So \*\*format   
demonstrartions do not count\*\*. "   
"Phrases like 'Follow this format', 'Here is what the   
format looks like' etc. \*\*are not few-shot evidence\*\*!   
Avoid them!"   
"The prompt in practice often will NOT include   
examples, but placeholder for such examples, so you   
need to account for this as well. "   
"They should form a dedicated section of the prompt. "   
"Make sure they include both \*\*input and output\*\*   
demonstrations. \*\*Output alone does not count\*\*. "   
"Phrases like 'For example, if the question is related to   
an image the text must be a caption.' do not count: look   
for examples of \*\*specific\*\* inputs and outputs."   
),   
"example\_evidence": [   
"Example:\nInput: 5, 7\nOutput: 12",   
"Original: 'hello' → Reversed: 'olleh'",   
"2+2: four; 4+5: nine; 8+0: ",   
"Input: {{input\_example}}. Expected output: {{output   
example}}.",   
"## Examples: {{examples}}"   
"output\_demonstrations": {   
"brief":   
"The prompt provides one or more example outputs   
(without inputs).",   
"detailed":

```python
("Look for sample outputs given in isolation, often
labeled ‘Sample output:' or similar, **without**
showing the corresponding inputs. "
"Such examples may appear in the prompt eihter
explicitly or as placeholders. Examples should
demonstrate **outputs only**. So input-output
demonstrations don't count."
"Make sure the outpts demonstrations **are not**
preceded by corresponding input examples."
),
"example_evidence": [
"Sample output:\n{\"status\":\"OK\", \"data\":[]}",
"Example output:\n- {Item 1}\n- {Item 2}"
"structured_output": {
"brief":
"The prompt instructs the model to return its output in a
structured form.",
"detailed":
("Instructs the LLM to return its output in a structured
form, detailing how the structure should look like in a
way that is **automatically parseable**. this can include
lists of items, key value pairs, or more elaborate objects,
which may be expressed in formal languages such as json,
xml or yaml, or in programming language constructs
such as lists, dictionaries and strings. Markdown formats
do not count as structured. Instructions to divide the
output into sections (e.g. 'Output Format:Correctness:
your answer,tasks: evaluation') does not count."
),
"example_evidence": [
"Return the answer as JSON: {\"name\": ..., \"age\":
...}",
"""Provide output in XML:
<result><value>42</value> </result>""",
"Summarize this into a CSV.", "Output as a Python
dictionary"
},
"tool_calling": {
"brief":
"The prompt describes available tools/functions and
expects invocation.",
"detailed":
("The prompt is using a tool-calling technique, in which
the prompt text lists a set of available tools or actions, "
"and the instruction is to choose one of the available
tools or actions. "
"The idea of choosing the one of the provided tools or
actions can be phrased in a diferent way (for example,
'decide' instead of 'choose'). "
"NOTE: just mentioning a function name is not enough,
the instruction should be specifically to choose one o
the tools, actions or functions to run. "
"However, any reference to provided **tools** in the
prompt is a strong indication of usage of this techinique.
),
"example_evidence": [
"Use the calculator by specifying {\"tool\": \"calc\",
\"input\": 2+2}.",
"Call search_api(query) to fetch results.",
"You are provided with the following tools",
"your output should indicate which tool to use",
"your output should specify which of these actions to
choose",
"""{instruc
tions}/n{status_prompt}/n{COT_PROMPT2}/n
{response}/n{memory_prompt}/nProvide the best next
action in the correct JSONformat. Action:"""
```

},   
"quote\_extraction": {   
"brief":   
"The prompt instructs grounding by extracting exact   
spans from context.",   
"detailed":   
("Find instructions asking to ‘quote', ‘extract', or ‘cite   
text \*\*directly from the given input\*\*. "   
"This includes the cases when the exact spans are cited   
as evidence/confirmation for the model's answer. "   
"The focus is on \*\*exact spans\*\*. "   
"Other types of extractions that are not exact spans \*\*do   
not count\*\*!"   
"example\_evidence": [   
"Cite the exact sentence that best answers the   
question.",   
"Extract the exact spans that follow these patterns:   
{patterns}."   
"Cite the exact spans from the input that confirm your   
answer."   
},   
"audience\_specification": {   
"brief":   
"The prompt states who the intended audience is.",   
"detailed":   
("Look for ‘for a beginner', ‘to a non-expert', 'for AI   
researchers' or naming another specific group for whom   
the output is intended. "   
"Audience specification is \*\*not role/persona   
assignment\*\*! "   
"\*\*It does no specify who you are.\*\* "   
"It specifies who \*\*your audience\*\* is."),   
"example\_evidence": [   
"Explain this concept for a high-school student.",   
"Write this guide aimed for web developers."   
"sections": {   
"brief":   
"The prompt is divided into clearly labeled sections.",   
"detailed":   
("Detect headings like ‘## Input', ‘### Task Description',   
or numbered segments or other forms indicating sections.   
The sections should be found in the prompt itself.   
Specifying desired sections of the expected output ('Your   
output example:Correctness: your answer\n, tasks:   
evaluation') does not count"   
),   
"example\_evidence": [   
"## Input\nThe first line contains. . . \n###   
Output\nPrint the result. . . ",   
"1. Problem Statement\n2. Constraints\n3. Example"   
]   
},   
"prompt\_chaining": {   
"brief":   
"The prompt is one step in a multi-step workflow.",   
"detailed":   
("Check for references to previous or next prompts, or   
instructions to pass output to another step.   
"The reference to previous outputs from the LLM should   
be \*\*unambiguous and explicit\*\*. "   
"Outputs from functions that don't envolve LLM queries   
\*\*do not count\*\*!"   
),   
"example\_evidence": [   
"Here are responses from various open-source models   
to the latest user query: {prev\_resps}",

"detailed":

#prompting techniques

"Use the result of the previous query as input here.", "Given a set of relevant quotes you extracted from a

"The prompt asks to decompose the task into steps, then solve each one.",

("Look for ‘Break the problem into sub-tasks', ‘Break the solution into steps "

"This is diferent from chain\_of\_thought where the LLM is asked to think/reason step by step before answering.

"Make a very clear distinction between thinking step by step vs. dividing a task into subtasks (only the latter one counnts!)"

"This is also diferent from cases where the prompt specifies the decomposition steps ('First do X then Y). These are \*not decomposition\_by\_LLM!\* "

"We are only looking for cases where the pompt instructs the LLM to decompose the problem/solution. 1 "Make a very clear distinction between decomposition is already specified in the prompt vs. where the LLM has to decompose (only the latter one counnts!)"

"example\_evidence": [

"Divide the solution into smaller steps and explain each step",

"First break the task into subtasks then accomplish them one by one."

"The prompt tells the model how exactly to decompose the task into steps.",

("Look for 'First do X, then Y', 'Here are the steps you should follow to solve the problem: A. <first step>, B. <second step>.."

"This is diferent from chain\_of\_thought where the LLM is asked to think/reason step by step before answering. "Make a very clear distinction between thinking step by step vs. dividing a task into subtasks (only the latter one counnts!)"

"This is also diferent from decomposition\_by\_LLM where the prompt tells the LLM to decompose the task by itself. This is \*not decomposition\_in\_prompt!\* " "Make a very clear distinction between decomposition by LLM itself vs. decompositon specified in the prompt (only the latter one counnts!)"

),   
"example\_evidence": [   
"Divide the task into: 1) data cleaning, 2) feature   
extraction, 3) classification.",   
"First list all entities, then identify relationships."   
]   
}   
}

def technique\_system\_prompt(technique\_name: str) -> str: details = technique\_details[technique\_name] evidence\_lines = "\n\t".join(f"- {e}" for e in details["example\_evidence"]) return f"""

You are an expert in prompting-technique analysis.

You will be given:

2) A list of tasks the prompt is intended for.

Reason step by step as described below, but output only the final answer.

Here are your resoning steps:

1. \*\*Study the detailed description\*\* of \*\*{technique\_name}\*\* and possible signals of its usage : {details['detailed']}

2. \*\*Locate candidate spans\*\* in the prompt that demonstrate use of {technique\_name}:

\- Scan for keywords or structures described below.

\- Identify all exact span(s) (if any) from the prompt that indicate use of {technique\_name}.

{technique\_name} \*\*very clearly and unambiguously\*\*).

Examples of typical evidence for {technique\_name}: {evidence\_lines}

3. \*\*Validate each found span\*\* (if any):

\- Confirm it fulfills the criteria for \*\*{technique\_name}\*\*.

4. \*\*Decide usage\*\*:

\- \`is\_used = true\` if at least one span was found, otherwise \`false\`.

5. Think and explain your decision before answering.

6. Return \*\*only\*\* a JSON object with three fields:

\- "reasoning": a string (max 100 words) where you briefly explain your decision before answering. Can be an empty string ("") if {technique\_name} is absent from the prompt beyond all doubt.

\- “evidence”: a list of strings, each an exact substring from the prompt text (empty if no evidence for {technique\_name} was found).

\- “is\_used”: a boolean (true \*\*if evidence is found\*\*, else false).

```twig
Example output:
{{
"reasoning": "<your reasoning steps>",
"evidence": {details['example_evidence']},
"is_used": true
}}
"nn
```

#promptfor discovering additional chain-of-thought cases reasoning\_field\_system\_prompt = """

You are an expert in prompting-technique analysis.

You will be given:

1) A raw prompt text.

2) A list of tasks the prompt is intended for.

Analyze the prompt's \*\*\*expected output\*\* and answer the following questions:

1. "Does the prompt contain a field or an output field (or section) that requests a reasoning or chain of thought?"

2 "If yes - does the answer based on this reasoning come before or after the reasoning? In other words, does the promp ask the model 1) to give an answer and then explain it, or 2) first think then give the answer"

Think and explain your decision before answering. Return \*\*only\*\* a JSON object with three fields:

\- "reasoning": a string (max 100 words) where you briefly explain your decision before answering. Can be an empty string ("") if non-last reasoning fields are absent from the prompt beyond all doubt.

\- “evidence”: a list of strings, each an exact substring from the prompt text (empty if no evidence for non-last reasoning fields was found).

\- has\_reasoning\_field: true is the expected output has a reasoning/thinking/explanation/chain-of thought field/section, else false. field/section, else false

\- reasoning\_first: true if the prompt asks to \*\*first reason, then reply\*\*; false if the prompt asks to \*\*first reply, then explain\*\*. If has\_reasonong field is false - this field is also false.

\*\*The reasoning field should be explicitly called so or similarly (reasoning/explanation/thinking/chain-of-though and the like.) \*\* Do not look for far-fetched implications.

## Example output:

"reasoning": "<your reasoning steps>",

"evidence": {<example evidence (found reasoning fields)>},

"has\_reasoning\_field": true|false,

"reasoning\_first":true|false

## translation\_system\_prompt = """

You are an expert at translating LLM prompts into English.   
You will be given a prompt text.

\- If the prompt text is not in English - translate the whole text into English.

\- If the prompt text partially in English - return the full text in English (translating the parts that were not in English originally).

## For example:

Prompt text: "<Text in Russian>! How are you? <Text in Russian>: {question}"

Your output: "Hello! How are you? Answer the question: {question}"

\- If the text is entirely in English already - return null.

Return ONLY the translated text (no extra commentary) or null .

## #instruction block list

## blocks = [

'audience specification', 'central task/question',

'central task/question placeholder', 'central task/question description',

'role specification', 'style specification', 'confirmation request',

'constraint/restriction', 'input context description',

'input contextual data', 'input context placeholder',

'evaluation criteria', 'date reference', 'default behavior instruction',

'design specification', 'disclaimer requirement',

'error handling instruction', 'example clarification', 'examples',

'expertise/skills requirements', 'function call instruction', 'question/task data/placeholder', 'question/task description', 'reasoning instructions', 'instruction to avoid errors',

'interaction guideline', 'language specification', 'scope specification',

'output content requirements', 'output format requirement', 'assistant response', 'conditional instruction', 'scene setting', 'encouragement', 'other'

## #instruction blocks

instruction\_blocks\_system\_prompt = f"""

You are proficient at breaking down LLM prompts into their atomic instruction blocks.

You will be given:

1) A JSON list of message objects, each with “role”,

“prompt\_text” and "message\_id" (for easier matching between input and output);

2) A list of tasks for the overall prompt.

Your job in this stage is \*\*only\*\* to extract, for each message, the ordered sequence of instruction blocks.

## 1. \*\*Block names\*\*:

\- To name the structural blocks, use only terms from this list:

{blocks}

\- If several terms apply, pick the single best fit.

\- If none fit, use \`Other(...)\`.

\- You may have multiple blocks of the same kind.

2. Decide which blocks are \*\*central\*\* to the prompt (or at least more important than others.)

This are the blocks for which "is\_central" will be set to 'true' in step 3.

3. \*\*Format\*\* each block as an object with \*\*five\*\* fields: - \`"instruction\_kind"\`: the block name

\`"instruction"\`: the exact substring from \`prompt\_text\` - \`"is\_central"\`: \`true\` if this block conveys the core task (contains the central task instruction) else \`false\`. You should try to select at least one central block if at all possible.

\- \`"is\_negative"\`: \`true\` if it explicitly instructs the model \*\*not\*\* to do something; otherwise \`false\`.

\- \`"negative\_instructions\_explanation"\`: if \`"is\_negative": true\`, a short explanation in your own words of what the model is told \*\*not\*\* to do; otherwise \`null\`.

Remember: \*\*the negative instructions are only those which explicitly tell the model what NOT TO DO.\*\*

4. \*\*Order\*\*: preserve the order in which blocks appear in the message.

## 5. \*\*Splitting\*\*:

\- Divide into pieces smaller than sentences if needed. Example:

{{ "instruction\_kind": "role specification",

"instruction": "As a helpful assistant", "is\_central": false, "is\_negative": false,

"negative\_instructions\_explanation": null }},

{{ "instruction\_kind": "task description",

"instruction": "summarize the text", "is\_central": true, "is\_negative": false,

"negative\_instructions\_explanation": null }}

## - Overlaps allowed. Example:

{{ "instruction\_kind": "task description",

"instruction": "Extract lists of named entities",

"is\_central": true, "is\_negative": false,

"negative\_instructions\_explanation": null }},

{{ "instruction\_kind": "output format requirement",

"instruction": "lists of named entities", "is\_central": false, "is\_negative": false,

"negative\_instructions\_explanation": null }},

```jinja
{{ "instruction_kind": "output content
requirement","instruction": "named entities",
"is_central": false, "is_negative": false,
"negative_instructions_explanation": null }}
5. **Empty**: if no blocks, return `"instruction_order": []`
**The output must carry each input message's `message_id
so you can map blocks back to messages.**
Be specific, precise and exhaustive.
**When you are done, go over your annotaion once again**.
Did you mark all the blocks correctly?
Did you mark at least one as is_central=true? **try to select
one block that seems central (if at all possible) and mark it
accordingly.**
**Output** exactly one JSON object, schema:
{
"instruction_data": [
{{
"message_id": "<same as input>",
"instruction_order": [
"instruction_kind": "<block name or Other(...)>",
"instruction": "<exact substring>",
"is_central": <true|false>,
"is_negative": <true|false>,
"negative_instructions_explanation": <string|null>
}},
}},
For example:
Input:
"messages": [
{
"message_id": 0,
"role": "system",
"prompt_text": "You are a helpful assistant skilled in
summarizing content concisely. Given a text passage,
extract the three most important points. Ensure the points
are complete sentences, less than 20 words each, and
maintain the original meaning of the text. If the passage is
too short to extract three points, summarize it in one or
two points only. Avoid adding any extra information not
present in the input text."
"message_id": 1,
"role": "user",
"prompt_text": "Input_text: {{{{text}}"
}}
"tasks": ["summarization"]
}}
{
"message_id": "0",
```

```csv
"instruction_order": [

"instruction_kind": "role specification",
"instruction": "You are a helpful assistant skilled in
summarizing content concisely.",
"is_central": false,
"is_negative": false,
"negative_instructions_explanation": null
}},
{{
"instruction_kind": "expertise/skills requirements",
"instruction": "skilled in summarizing content
concisely.",
"is_central": false,
"is_negative": false,
"negative_instructions_explanation": null
}},
"instruction_kind": "input context description",
"instruction": "Given a text passage",
"is_central": false,
"is_negative": false,
"negative_instructions_explanation": null
}},
{{
"instruction_kind": "task description",
"instruction": "extract the three most important points",
"is_central": true,
"is_negative": false,
"negative_instructions_explanation": null
}},
{{
"instruction_kind": "output format requirement",
"instruction": "Ensure the points are complete
sentences, less than 20 words each",
"is_central": false,
"is_negative": false,
"negative_instructions_explanation": null
}},
{{
"instruction_kind": "output content requirement",
"instruction": "maintain the original meaning of the
text",
"is_central": false,
"is_negative": false,
"negative_instructions_explanation": null
}},
"instruction_kind": "conditional instruction",
"instruction": "If the passage is too short to extract three
points, summarize it in one or two points only",
"is_central": false,
"is_negative": false,
"negative_instructions_explanation": null
}},
{{
"instruction. kind": "constraint/restriction"
"instruction": "Avoid any extra information not presen
in the input text",
"is_central": false,
"is_negative": true,
"negative_instructions_explanation": "The model is
instructed not to add information that isn't in the input
text."
}}
}}
{{
"message_id": "1",
"instruction_order": [
{{
```

"instruction\_kind": "input contextual data",   
"instruction": "Input\_text: {{text}}",   
"is\_central": false,   
"is\_negative": false,   
"negative\_instructions\_explanation": null   
}},   
{{   
"instruction\_kind": "input context placeholder",   
"instruction": "{{text}}",   
"is\_central": false,   
"is\_negative": false,   
"negative\_instructions\_explanation": null   
}}   
]   
}}   
]   
}}

## #identifying input context, question and directions input\_blocks\_prompt = """

You are proficient at analyzing LLM prompts. You are given: (1) the prompt text as a list of messages and (2) one or more standard NLP/AI tasks.

Identify three parts of the prompt: directions, context, and question.

## Definitions:

\- Directions: what the model must do given the context and question (e.g., “Answer the following. . . ”,

“Summarize the text. . . ”). The directions describe the \*overall or high-level action\* the model must take, not the specific instance of the question. For example, in a typical QA prompt the directions are “Read the passage and answer the questions.”

Important: the "directions" are usually 1-2 spans. They only include the main, high-level instructions, never details or nuances.

\- Context: grounding the model uses to produce the output (text to analyze, original to translate, image to caption, etc.). A context may come in many diferent forms, even in the form of a question, but should be identified as context if it functions as grounding/background. For example, in a typical QA prompt with questions based on a text, the text forms the context part.

Do NOT include examples, role specifications, meta-instructions, formatting hints. output prefixes (e.g., "Answer:").

Only the grounding information, in which the output is grounded.

\- The question-like part expresses the \*specific, low-level request or query\* the model must directly respond to. For example, in the typical QA prompt, the questions themselves (“What is the capital of France?”) form the question-like part.

The question-like part is \*obligatory\* (cannot be empty); context may be missing; the directions can be separated or merged with the question-like part (e.g., “Summarize the text below”).

Yu should extract them all \*from the prompt-text\*, not he list os standard nlp tasks provided in the beginning. Notes: any of the three parts can be expressed as a text, a placeholder or absent as such but described in the prompt text.

## For each part:

1. Decide if it splits into units; if not, treat as one unit.

2. For each unit provide the exact verbatim span or variable name from the prompt text.

3. Return all units as a list of strings (one string per unit, or a single-item list if unsplittable).

## Important:

\- Extract \*\*word-for-word\*\* only. Do not paraphrase or summarize.

\- Keep only the \*\*core elements\*\* — minimal spans from the prompt text defining each part.

\- If you are unsure whether a span is part of the request/query itself or just an instruction about \*how\* to perform it — leave it out.

\- \*\*Omit\*\* examples, meta-instructions, role assignments, output prefixes (e.g., "Answer:"), formatting hints, and any other non-core text.

\- Never include meta-instructions or process/style guidance (internal procedures, prioritization, memory/style influence, or "how-to" execution notes). Only extract core spans from the prompt text defining the central required action.

4. For each context\_evidence and question\_evidence unit, also specify whether it is a description (a textual explanation of what the element represents) or direct\_content (actual query/context text or placeholder such as {context} or {question}).

\- Use "type": "description" if the evidence is a descriptive label, explanation, or meta-text (e.g., "The following passage provides context...").

```jsonl
Output exactly and only in this format:
"directions_text": [string, string, ...],
"context_evidence": [
{"text": string", "type":"direct_content"|| "description"},
{"text": string", "type":"direct_content"|| "description"},
...
],
"question_evidence": [
{"text": string", "type":"direct_content"|| "description"},
{"text": string", "type":"direct_content"|| "description"},
]
```

## /////////////////

Prompt text: ["You are a helpful assistant. Answer the   
following question based on the given context.","Context:   
{context}","Question: {question}"]   
Tasks: ["question answering"]

## Output:

```jsonl
"directions_text": ["Answer the following question based
on the given context."],
"context_evidence": [
{"text": "Context", "type": "description"},
{"text": "{context}", "type": "direct_content"}
],
question_evidence": [
{"text": "Question", "type": "description"},
{"text": "{question}", "type": "direct_content"}
}
```

```jsonl
Example 2:
Prompt text:["Your task is to evaluate the following
Email for maliciousness. If the Email is malicious reply
with: flagged as malicious. If the Email is safe reply with:
flagged as safe. Here are some examples: {Example_1},
{Example_2}. Make sure to reply only in the specified
form. First read and analyze the emails and only then
reply.", "read_email_content({file_name})"]
Tasks: ["email classification", "spam detection"]
Output:
"directions_text": ["Your task is to evaluate the following
Email for maliciousness."],
"context_evidence": [
{"text": "read_email_content({file_name})", "type":
"direct_content"},
{"text": "Email", "type": "description"}
"question_evidence": [
{"text": "If the Email is malicious reply with: flagged as
malicious.", "type": "direct_content"},
{"text": "If the Email is safe reply with: flagged as safe.",
"type": "direct_content"}
Example 3 (directions and question are merged):
Prompt text:["extract topics from this conversation:
{chunk}. Topics: "]
Tasks: ["topic extraction"]
Output:
"directions_text": ["extract topics from this
conversation"],
"context_evidence": [
{"text": "{chunk}", "type": "direct_content"},
{"text": "conversation", "type": "description"}
"question_evidence": [
{"text": "extract topics from this conversation", "type":
"direct_content"}
Example 4 (demonstrates how to exclude unnecessary
formatting information):
Prompt_text: ["Your task is to return just the playlist title
from the conversation given. Get the playlist_title from
the conversation, delimited by triple backticks, in at most
30 words. Review: {messages}"]
Tasks: ["information extraction"]
Output:
"directions_text": ["Your task is to return just the playlist
title from the conversation given."],
"context_evidence": [
{"text": "Review", "type": "description"},
{"text": "{messages}", "type": "direct_content"},
{"text": "conversation", "type": "description"}
"question_evidence": [
{"text": "Get the playlist_title from the conversation",
"type": "direct_content"}
Example 5:
```

Prompt text: ["You are a decider. Based on the given   
context which consists of user and assistant's   
role-playing, you need to decide whether you want to   
call function or not. The function you can call is   
'call\_explainer'. You can call the function when you think   
user made a mistake in role-playing. Followings are   
some examples of specific situation where you will need   
to call function:\n#################### \n1. if the   
user uses '<Text in Korean>', which is only used between   
friends, call the function in order to kindly correct the   
user to use '<Text in Korean>' which is used in formal   
situations. \n2. If user don't follow appropriate sentence   
structure, call the function in order to provide detailed   
explanation about the sentence   
structure.\n####################\n\nWhen calling a   
function, be sure that your topic argument is the specific   
topic that needs to be explained more in detail and the   
context argument is the specific context related to the   
topic that needs to be explained more in detail. Also be   
careful not to just call function everytime. You should   
only call function when you think user needs additiona   
explanation about the topic's context."]   
Tasks: ["decision making"]   
Output:   
"directions\_text": [   
"Based on the given context which consists of user and   
assistant's role-playing, you need to decide whether you   
want to call function or not.",   
"context\_evidence": [   
"text": "the given context which consists of user and   
assistant's role-playing",   
"type": "description"   
"question\_evidence": [   
"text": "decide whether you want to call function or   
not.",   
"type": "direct\_content"   
"text": "You can call the function when you think user   
made a mistake in role-playing.",   
"type": "direct\_content"   
Example 6:   
Prompt text: ["Your job is to use patient reviews to answer   
questions about their experience at a hospital. Use the   
following context to answer questions.   
Be as detailed as possible, but don't make up any   
information that's not from the context.   
{context}"]   
Tasks: ["question answering"]   
Output:   
"directions\_text": [   
"Your job is to use patient reviews to answer questions   
about their experience at a hospital."   
"context\_evidence": [   
"text": "patient reviews",   
"type": "description"

```python
"text": "context", "type": "description"
"type": "description" },
"text": "{user_input}",
"text": "{context}", "type": "direct_content"
"type": "direct_content"
]
}
"question_evidence": [

"text": "questions about their experience at a hospital", #input context and question variability
"type": "description" input_variability_prompt = """
You are proficient at analyzing LLM prompts. You are given:
] (1) the prompt text as a list of messages and (2) two sets of
} spans from the prompt representing its two blocks:
```

Prompt text: ["You Write Python Function. You are a Senior Data Analyst with 10+ Years of Experience. This is a Critical Scenerio. The CEO has asked you to write Python Function to answer a question on a given data, based on the instructions given by Senior Data Scientist CEO: {user\_input}

Dataframe Head: {df\_head}

Data Scientist's Instructions:{instructions}

Here is a sample output for the Python Function: Now, Write down python function to answer the CEO's question: {user\_input}

Just Write the Python Function in markdown format, that's   
it."]   
Tasks: ["code generation"]   
Output:

"directions\_text": [   
"Now, Write down python function to answer the CEO's   
question: {user\_input}",   
"write Python Function to answer a question on a given   
data, based on the instructions given by Senior Data   
Scientist"   
],   
"context\_evidence": [   
"text": "Dataframe Head",   
"type": "description"   
},   
"text": "{df\_head}",   
"type": "direct\_content"   
},   
"text": "Data Scientist's Instructions",   
"type": "description"   
},   
"text": "{instructions}",   
"type": "direct\_content"   
}   
],   
"question\_evidence": [   
"text": "a question on a given data",   
"type": "description"   
},   
"text": "the CEO's question",

\- Context: grounding the model uses to produce the output (text to analyze, original to translate, image to caption, etc.). A context may come in many diferent forms, even in the form of a question, but should be identified as context if it functions as grounding/background. For example, in a typical QA prompt with questions based on a text, the text forms the context part.

\- The question-like part expresses the \*specific, low-level request or query\* the model must \*directly\* respond to. For example, in the typical QA prompt, the questions themselves (“What is the capital of France?”) form the question-like part.

Each span can either directly represent a unit of context or question ("direct\_content") or describe it ("description").

Read the prompt and the spans corresponding to the two blocks, and determine the following: 1. context\_variability.

\*\*Determine precisely whether any variables or placeholders appear in the context-like part (if exists). Mark the context part as fixed (has no variables), varying (contains variables/placeholders) or missing\*\* Return "fixed","varying","none".

If the answer cannot be determined, return "undefined". Important! Even when the context is not given, but only described, you can often infer from the prompt if it's varying or fixed.

\*\*\*Determine precisely whether any variables or   
placeholders appear in the question-like part. Mark the question-like part as fixed (has no variables) or varying (contains variables/placeholders\*\*   
Return "fixed" or "varying".

Important! Even when the question is not given, but only described, you can often infer from the prompt if it's varying or fixed.

Important! Sometimes the context is mentioned within the question. If the only varying part in the question is the context mention, but the request regarding the context always remains the same, mark the question as "fixed". See example 4.

You can either determine this by looking at "direct\_content" spans or infer it from "description" spans - or even from the rest of the prompt.

The output should be a JSON object with exactly these fields:

```jsonl
"context_variability": one of "fixed", "varying", "none",
or "undefined" (use "none" when the context_evidence is
empty; "undefined" if nonempty, but the variability
cannot be inferred.)
"question_variability": one of "fixed", "varying", or
"undefined"
////////////////////
Example 1:
Prompt text: ["You are a helpful assistant. Answer the
following question based on the given context.","Context:
{context}","Question: {question}"]
Blocks: {
"context_evidence": [
{"text": "Context", "type": "description"},
{"text": "{context}", "type": "direct_content"}
],
"question_evidence": [
{"text": "Question", "type": "description"},
{"text": "{question}", "type": "direct_content"}
]
}
Output:
{"context_variability": "varying",
"question_variability": "varying"}
```

## Example 2:

Prompt text:["Your task is to evaluate the following Email for maliciousness. If the Email is malicious reply with: flagged as malicious. If the Email is safe reply with: flagged as safe. Here are some examples: {Example\_1}, {Example\_2}. Make sure to reply only in the specified form. First read and analyze the emails and only then reply.", "read\_email\_content({file\_name})"] Blocks:

```jsonl
"context_evidence": [
{"text": "read_email_content({file_name})", "type":
"direct_content"},
{"text": "Email", "type": "description"}
],
"question_evidence": [
{"text": "If the Email is malicious reply with: flagged as
malicious.", "type": "direct_content"},
{"text": "If the Email is safe reply with: flagged as safe.",
"type": "direct_content"}
]
}
```

```jsonl
Example 3: Prompt_text:["You are a helpful assistant.",
"Based on the following transcript, please answer the
question: {question}"]
Blocks: "context_evidence":
[ {"text": "transcript", "type": "description"}],
"question_evidence":
[ {"text": "question", "type": "description"},
{"text": "{question}", "type": "direct_content"} ]
}
```

Output:

"context\_variability": "undefined",   
"question\_variability": "varying"   
}   
Example 4 (the only variable in the question is the   
context):   
Prompt\_text:["Please translate {speech2text(output.wav)}   
to malayalam"]   
Blocks:   
"context\_evidence": [{"text":   
"{speech2text(output.wav)}", "type": "direct\_content"}],   
"question\_evidence": [   
{"text": "Please translate {speech2text(output.wav)} to   
malayalam", "type": "direct\_content"}   
Output:   
"context\_variability": "varying",   
"question\_variability": "fixed"   
}   
  
#input language and structure   
input\_language\_and\_structure\_prompt = """   
You are proficient at analyzing LLM prompts. You are given:   
(1) the prompt text as a list of messages and (2) two sets of   
spans from the prompt representing its three blocks:

\- Directions: what the model must do given the context and question (e.g., “Answer the following. . . ”, “Summarize the text. . . ”). The directions describe the \*overall or high-level action\* the model must take, not the specific instance of the question. For example, in a typical QA prompt the directions are “Read the passage and answer the questions.”

\- Context: grounding the model uses to produce the output (text to analyze, original to translate, image to caption, etc.). A context may come in many diferent forms, even in the form of a question, but should be identified as context if it functions as grounding/background. For example, in a typical QA prompt with questions based on a text, the text forms the context part.

\- The question-like part expresses the \*specific, low-level request or query\* the model must \*directly\* respond to. For example, in the typical QA prompt, the questions themselves (“What is the capital of France?”) form the question-like part.

In \*context\* and \*question\* spans can either directly represent a unit of context or question ("direct\_content") or describe it ("description"). In \*directions\* the spans are always direct content.

Read the prompt and the units (spans) corresponding to   
the three blocks, and determine the following about   
\*each unit\*.   
1. language   
\*\*What natural human languages are used in this unit of   
directions, context or question\*\* Provide a list of natural human languages used in the unit. It may be a one-item list if only one language is used.

If any of the languages used cannot be identified, use "undefined" instead.

For direct text - identify the language(s) by looking at the text.

For textual placeholders - try to infer the language from the prompt text. Look for \*language mentions\* or \*strings in diferent languages\* in the prompt text. However, don't use unsupported guessing. Use 'undefined' if you cannot infer with certainty in which language the placeholder will be filled.

For descriptions - do not return the language of the description itself! Return the langiage od the   
context/question being described!   
Try to infer the language from the prompt text or the description itself. Look for \*language mentions\* or \*strings in diferent languages\* in the prompt text.   
However, don't use unsupported guessing. Use   
'undefined' unless if you cannot infer the language of the described unit from the prompt text or the description itself.

For descriptions and placeholders: please, do not assume the language is Engglish - unless there is clear evidence for it in the prompt.

Keep in mind that even if the prompt itself is in English, the context or question can still be in a diferent language. So avoid ungrounded assumptions.

For empty context return empty list.

2. structure (only for context and question units.)

\- Single item

\- Pair of items (type, typeB)

\- Tuple (typeA, typeB, . . . , typeN)

\- List of items (list of type A)

\- Dictionary of items (key1: typeA, key2: typeB, . . . ) (a pair is basically a tuple of two items; if a dictionary only includes one entry, it's still a dictionary) (you can expand this list if needed)

For direct text - identify the structure by looking at the text.

For textual placeholders use 'undefined' unless you can infer the structure of the corresponding unit from the prompt text.

3. For context units also provide context modality:

\*\*What is the modality of the context unit?\*\*

\- text

\- audio

\- image

For direct text - identify the modality by looking at the text.

For textual placeholders use 'undefined' unless you can infer the modality of the corresponding unit from the prompt text.

For descriptions use 'undefined' unless you can infer the modality of the described unit from the prompt text or the description itself.

For empty context - don't add anything.

The output should be two json fields added to each unit:

"\*\_language":[string,string,...]

and an additional "context\_modality" field for context units:

"context\_modality": one of "text", "audio", "image", "video", "undefined"

Important: even if the language, structure and/or modality aren't mentioned explicitly, you can often infer them from diferent signals like citations, examples etc.

Important: Do not guess. Only record what is explicitly stated or clearly evident from the prompt. If something cannot be determined with certainty, return "undefined".

Inference and guessing are not the same!

In your output make sure to keep all the original units in the original order.

////////////////////

Example 1:

Prompt text: ["{audio}","The user left us a message in the voice mail. Extract 1. Their name, 2. a fitting title that summarizes their message, and 3. their message. Output 1.their name, 2. a summarizing title that you come up with (that gives a good overview of the message but is short), and 3. the message, separated by only a hyphen(-) (no space). Only output those values. Nothing more. Example: John-Grocery Trip-I want to go to the store and buy groceries. Don't put a hyphen before, after or anywhere else in the output. Only in between the name, title and message.",

Blocks: {   
"direction": [   
"directions\_text": "Extract 1. Their name, 2. a fitting title   
that summarizes their message, and 3. their message."   
}   
],   
"context": [   
"context\_evidence": {   
"text": "{audio}",   
"type": "direct\_content"   
},   
"context\_evidence": {   
"text": "The user left us a message in the voice mail.",   
"type": "description"   
],   
"question": [   
"question\_evidence": {   
"text": "Extract 1. Their name, 2. a fitting title that   
summarizes their message, and 3. their message.",   
"type": "direct\_content"   
]   
Output:   
"direction": [

"directions\_text": "Extract 1. Their name, 2. a fitting title   
that summarizes their message, and 3. their message.",   
"direction\_language": ["English"]   
"context": [   
"context\_evidence": {   
"text": "{audio}",   
"type": "direct\_content"   
"context\_language": ["undefined"],   
"context\_structure": "Single item",   
"context\_modality":"audio"   
},   
"context\_evidence": {   
"text": "The user left us a message in the voice mail.",   
"type": "description"   
},   
"context\_language": ["undefined"],   
"context\_structure": "Single item",   
"context\_modality":"audio"   
"question": [   
"question\_evidence": {   
"text": "Extract 1. Their name, 2. a fitting title that   
summarizes their message, and 3. their message.",   
"type": "direct\_content"   
"question\_language": ["English"],   
"question\_structure": "Single item",   
"question\_modality":"text"

```yaml
Example 2:
Prompt text:["Your task is to evaluate the following email
for maliciousness. The emails content will be in French.
Each email is a dictionary with three fields: "Expéditeur",
"Destinataire", "Contenu". Respond only with one of the
exact labels below (no extra text). If the email is malicious,
reply: signalé comme malveillant. If the email is safe, reply:
signalé comme sûr. Here are some examples: {Example_1},
{Example_2}. First read and analyze the email content,
then reply.","read_email_content({file_name})"]
Blocks: {
"direction": [
"directions_text": "Your task is to evaluate the following
email for maliciousness.
],
"context": [
"context_evidence": {
"text": "read_email_content({file_name})",
"type": "direct_content"
},
"context_evidence": {
"text": "email",
"type": "description"
```

```csv
"question": [
"question_evidence": {
"text": "If the email is malicious, reply: signalé comme
malveillant.",
"type": "direct_content"
"question_evidence": {
"text": "If the email is safe, reply: signalé comme sûr.",
"type": "direct_content"
Output:
"direction": [
"directions_text": "Your task is to evaluate the following
email for maliciousness. ",
"direction_language": ["English"]
"context": [
"context_evidence": {
"text": "read_email_content({file_name})",
"type": "direct_content"
"context_language": ["French"],
"context_structure": "Dictionary of items ("Expéditeur:
string, "Destinataire": string, "Contenu": string)",
"context_modality": "text"
"context_evidence": {
"text": "email"
"type": "description"
"context_language": ["French"],
"context_structure": "Dictionary of items ("Expéditeur:
string, "Destinataire": string, "Contenu": string)",
"context_modality": "text"
"question": [
"question_evidence": {
"text": "If the email is malicious, reply: signalé comme
malveillant.",
"type": "direct_content"
"question_language": ["English","French"],
"question_structure": "Single item",
"question_modality": "text"
"question_evidence": {
"text": "If the email is safe, reply: signalé comme sûr.",
"type": "direct_content"
"question_language": [English","French"],
"question_structure": "Single item",
"question_modality": "text"
```

Example 3 (shows how language can be inferred from details):

Prompt text: ["You are a decider. Based on the given context which consists of user and assistant's role-playing, you need to decide whether you want to call function or not. The function you can call is 'call\_explainer'. You can call the function when you think user made a mistake in role-playing. Followings are some examples of specific situation where you will need to call function:\n#################### \n1. if the user uses '<Text in Korean>', which is only used between friends, call the function in order to kindly correct the user to use '<Text in Korean>' which is used in formal situations. \n2. If user don't follow appropriate sentence structure, call the function in order to provide detailed explanation about the sentence structure.\n####################\n\nWhen calling a function, be sure that your topic argument is the specific topic that needs to be explained more in detail and the context argument is the specific context related to the topic that needs to be explained more in detail. Also be careful not to just call function everytime. You should only call function when you think user needs additional explanation about the topic's context."]

```jsonl
Blocks: {
"direction": [
"directions_text": "Based on the given context which
consists of user and assistant's role-playing, you need to
decide whether you want to call function or not."
"context": [
"context_evidence": {
"text": "the given context which consists of user and
assistant's role-playing",
"type": "description"
],
"question": [
"question_evidence": {
"text": "decide whether you want to call function or not.",
"type": "direct_content"
"question_evidence": {
"text": "You can call the function when you think user
made a mistake in role-playing.",
"type": "direct_content"
]
Output:
"direction": [
"directions_text": "Based on the given context which
consists of user and assistant's role-playing, you need to
decide whether you want to call function or not.",
"direction_language": ["English"]
],
"context": [
"context_evidence": {
```

```python
"text": "the given context which consists of user and
assistant's role-playing",
"type": "description"
},
"context_language": ["Korean"],
"context_structure": "undefined",
"context_modality": "text"
],
"question": [
"question_evidence": {
"text": "decide whether you want to call function or not.",
"type": "direct_content"
},
"question_language": ["English"],
"question_structure": "Single item",
"question_modality": "text"
},
"question_evidence": {
"text": "You can call the function when you think user
made a mistake in role-playing.",
"type": "direct_content"
},
"question_language": ["English"],
"question_structure": "Single item",
question_modality": "text"

input type
nput_type_prompt = """
ou are proficient at analyzing LLM prompts. You are given:
1) the prompt text as a list of messages and (2) two sets of
pans from the prompt representing its two blocks:
```

\- Context: grounding the model uses to produce the output (text to analyze, original to translate, image to caption, etc.). A context may come in many diferent forms, even in the form of a question, but should be identified as context if it functions as grounding/background. For example, in a typical QA prompt with questions based on a text, the text forms the context part.

\- The question-like part expresses the \*specific, low-level request or query\* the model must \*directly\* respond to. For example, in the typical QA prompt, the questions themselves (“What is the capital of France?”) form the question-like part.

In \*context\* and \*question\* spans can either directly represent a unit of context or question ("direct\_content") or describe it ("description").

Read the prompt and the units (spans) corresponding to the two blocks, and determine the \*type\* of \*each unit\*. \*\*What kind of context/question unit is it?\*\* (please, consider the surrounding prompt text when answering: it may give you important hints):

You can expand this list if needed (for example, with other lengths and types of text - such as sentence, email, poem, tweet, outline, summary, article, dialogue etc. - other types of structured data, such as json, xml etc. ). Be precise, do not guess. Do try to say something more interesting than just "text".

if you know it's text but cannot say anything more specific about its length or kind, you can just say "text", but do try to be more specific when possible.

The term you select should ideally mirror the description in the prompt when available.

Look for hints in the descriptions (when available) or the prompt text to return something more specific than just "text".

For example, it is better to use a specific genre or form - e.g. "poem", "menu","sms" or "sentence", "html" etc. - than just "text" - but only when \*firmly supported by the prompt text\*!

But do not hallucinate or invent! Output \*only firmly grounded\* answers!

Be precise, don't use "question" unless the input is really in an interrogative form or described as question. Paragraph is genuinely a paragraph, not just any relatively short text.

Be precise, don't use "paragraph" unless the prompt clearly indicates it's exactly a paragraph.

Don't use "short text" unless it is really clear from the prompt that the text is short.

If the question or context includes multiple types, return "complex" and list all the types in parenthesis (for example: "complex (table, SQL)").

It's ok to use singular (e.g. email) even if the input includes several items of the same kind (e.g. several emails). Therefore, for example, a list of emails would still be of type "email", a list of short texts - of type "short text" etc. Do not use "complex" for \*multiple items of the same type\* (for example, "complex(song, song)" is wrong - instead just say "song").

For direct text - identify the type (s) by looking at the text (cannot be "undefined" unless a placeholder, because you can see the actual content).

For textual placeholders - try to infer the type from the prompt text. However, don't use unsupported guessing. Use 'undefined' if you cannot infer the type with certainty

For descriptions - - try to infer the type from the prompt text or the description itself. However, don't use unsupported guessing. Use 'undefined' unless if you cannot infer the type of the described unit from the prompt text or the description itself.

Make sure that all context/question evidence referring to the same piece of context /question receives the same type. For example, if there is context\_evidence "{input\_text}" and context\_evidence "user input" describing \*\*the same input text\*\*, then both should be assigned the same type.

question\_type - with rare exeption is either "question" (if interrogative or described as question) - or instruction (in rare cases can be also "complex" including both, "undefined" or something else.) In very rare cases can be something else.

Avoid confusing question type with tye output type it requests!

The output should a json fields added to each unit: "\*\_type": string (For empty contexts - don't add anything)

Important: even if the type isn't mentioned explicitly, you can often infer it from diferent signals like citations, examples etc.

Important: Do not guess. Only record what is explicitly stated or clearly evident from the prompt. If something cannot be determined with certainty, return "undefined".

Inference and guessing are not the same!

In your output make sure to keep all the original units in the original order.

///////////////////// Example 1: Prompt text: ["You are a university course advisor assistant. Use the background info to answer the student's question.", "You can use the following student info: Major: {major}, GPA: {gpa}.","Student question: {student\_question}"]

```jsonl
Blocks: {
"context": [
"context_evidence": {
"text": "Major",
"type": "description"
},
"context_evidence": {
"text": "{major}",
"type": "direct_content"
"context_evidence": {
"text": "GPA",
"type": "description"
},
"context_evidence": {
"text": "{gpa}",
"type": "direct_content"
"question": [
"question_evidence": {
"text": "Student question",
"type": "description"
},
"question_evidence": {
"text": "{student_question",
"type": "direct_content"
Output:
```

```csv
"context": [
"context_evidence": {
"text": "Major",
"type": "description"
"context_type": "short text"
"context_evidence": {
"text": "{major}",
"type": "direct_content"
"context_type": "short text"
"context_evidence": {
"text": "GPA",
"type": "description"
"context_type": "numeric"
"context_evidence": {
"text": "{gpa}",
"type": "direct_content"
"context_type": "numeric"
"question": [
"question_evidence": {
"text": "Student question",
"type": "description"
"question_type": "question"
"question_evidence": {
"text": "{student_question}",
"type": "direct_content"
"question_type": "question"
Example 2 (with a complex type - only use if really
necessary):
Prompt text: ["You are preparing nutritional guidance for th
user based on their personal details. Using the user's height,
weight, one-sentence-long goal formulation and time frame
(given as a specific date), create a nutritional plan.","User
info: {info}"]
Blocks: {
"context": [
"context_evidence": {
"text": "personal details",
"type": "description"
},
"context_evidence": {
"text": "the user's height, weight, one-sentence-long goa
formulation and time frame (given as a specific date)",
"type": "description"
},
```

"context\_evidence": {   
"text": "User info",   
"type": "description"   
"context\_evidence": {   
"text": "{info}",   
"type": "direct\_content"   
"question": [   
"question\_evidence": {   
"text": ", create a nutritional plan",   
"type": "direct\_content"   
Output:   
"context": [   
"context\_evidence": {   
"text": "personal details",   
"type": "description"   
},   
"context\_type": "complex (numeric, sentence, time/date)"   
"context\_evidence": {   
"text": "the user's height, weight, one-sentence-long goal   
formulation and time frame (given as a specific date)",   
"type": "description"   
},   
"context\_type": "complex (numeric, sentence, time/date)"   
"context\_evidence": {   
"text": "User info",   
"type": "description"   
"context\_type": "complex (numeric, sentence, time/date)"   
"context\_evidence": {   
"text": "{info}",   
"type": "direct\_content"   
"context\_type": "complex (numeric, sentence, time/date)"   
"question": [   
"question\_evidence": {   
"text": ", create a nutritional plan",   
"type": "direct\_content"   
"question\_type": "instruction"   
  
#output modality, language and structure   
outputs\_system\_prompt = """   
Task:   
You are skilled at analyzing text prompts for LLMs and   
identifying their \*\*expected output\*\*.

You will be provided with:

\- Prompt text (as a raw string)

\- A task or list of tasks corresponding to the prompt.

Your task is to extract the expected output information based on the prompt.

Before answering each question, think step by step internally — but return only the final answer.

## Steps to Follow:

## 1. Output Identification

Identify which spans in the prompt text specify or describe the \*\*output\*\* expected from the model (as opposed to input and other things).

This may include diferent specifications of the output format, style, content etc., output descriptions, output prefills and any other things from which you can infer what output is expected from the LLM.

## 2. Output Segmentation

Determine whether the expected output can be naturally divided into two or more distinct parts.

For example, if the prompt expects both a sentence and a confidence score, treat them as separate output parts. If there is only one unified output, treat it as a single part.

3. For each output part, provide the following:

## a. Output Modality

Identify the modality of the output:

\- text

\- audio

\- image

\- video

If the modality is unclear, return "undefined".

## b. Output Description

Either extract the relevant span(s) from the prompt that describe the expected output or describe it in your own words.

If neither is possible, return "undefined".

Avoid quoting the entire prompt.

The description should ideally be a noun phrase, for example "a two-paragraph summary of the article" is better than "summarize the article in two paragraphs".

## c. Output Description Source

Indicate how the description above was obtained:

\- "extracted" - only if it is a direct, verbatim phrase from the prompt

\- "generated" - in all other cases (if it was paraphrased, inferred from the prompt etc.).

## d. Output Language

For textual outputs (if the modelity is 'text') try to identify/infer based on the prompt text in which natural human language (if any) the output is expected. Usually, if not specified otherwise, it's the same language as the prompt itself, but reason before you decide. Return a list of languages. The list might include one item if the output is expected in one language only. For non-textual outputs return ["undefined"].

For code, numerical outputs, formulas and other outputs that are not in a natural language return "undefinde".

## e. Output Structure

Identify the output structure:

\- Single item: for one value

\- Pair of items (typeA, typeB): for exactly two items

\- Tuple (typeA, typeB, ..., typeN): for multiple items of diferent types

\- List of items (list of typeA): for multiple items of the same type

\- Dictionary of items (key1: typeA, key2: typeB, ...): for key-value mappings

! if the output is described in plural (e.g., fragments, articles), it is likely a list, pair, or tuple—not a single item.)

\*\*Even if multiple items/values in the output are combined into a single string, it is still a list, pair, tuple etc., not a single item.\*\*

\*\*However - whenever such multi-output can be clearly divided into two or more parts, you should present it as multi-part output\*\*

You may expand this list if needed based on the prompt. If the structure of the ouput cannot be determined based on the prompt, return "undefined".

## Output Format:

Return a JSON array named "output", where each item corresponds to one output unit and contains:

\- "modality": "text","audio","image","video" or "undefined".

\- "description": string

\- "description\_source": "extracted" or "generated"

\- "output\_language": list of strings

\- "structure": string

If a string field cannot be identified, return "undefined".

## Examples:

1. Single Output:

```json
"output": [
"modality": "text",
"description": "three follow-up questions that a teacher
could ask after reading the student's answer",
"description_source": "extracted",
"output_language": ["english"],
"structure": "list of items"
}
]
}
```

## 2. Multi-Part Output:

"output": [

"modality": "text",

"description": "customer contact details as a JSON object with keys first\_name in Japanese, last\_name in Japanese, phone",

"description\_source": "extracted",

"output\_language": ["japanese","undefined"],

"structure": "dictionary of items (first\_name: single item,

last\_name: single item, phone: single item)"

"modality": "text",

"description": "subscription status (yes or no)",

"description\_source": "extracted",

"output\_language":["english"],

"structure": "single item"

## #output type

output\_type\_system\_prompt = """

You are skilled at analyzing LLM prompt outputs and determining the type of each output unit.

Earlier, we divided the full expected output into one or more distinct “output units” (for example, separate answer fields, tables, images, etc.). Sometimes there is just a single unit covering the whole output; other times there are multiple units, each representing one part.

Now you will be given:

\- The full original prompt text.

\- The associated task(s).

And for the output unit in question:

\- This unit's modality.

\- This unit's description.

\- This unit's structure.

Your job is to assign an \*\*output\_type\*\* to that unit. Valid types include:

\- Short text / tweet

\- Code / SQL

\- Table

\- Row

\- Time / Date

\- Unknown

\*You can expand this list if needed\* (for example, with other lengths and types of text - such as sentence, email, poem, tweet, outline etc. - other types of structured data, such as json, xml etc. ). Be precise, do not guess. Do try to say something more interesting than just "text".

Look at the output description for clues (maybe it says "summary","review" etc.). But, of course, consider the whole prompt text.

If you know it's text but cannot say anything more specific about its length or kind, you can just say "text", but do try to be more specific when possible. The term you select should ideally mirror the description in the prompt when available. \*Be precise\*, \*avoid guessing\*.

Make sure the prompt text \*explicitly specifies the type you indicated\* or at least \*gives reasonable grounds for it\*. It's ok to use singular (e.g. email) even if the output includes several items of the same kind (e.g. several emails).

Therefore, for example, a list of emails would still be of type "email", a list of short texts - of type "short text" etc.

## Rules:

\- If the unit clearly matches one type, return that type as a plain string.

\- If it contains multiple heterogeneous types, return \`complex

(A and B)\`, naming each type in the order they appear.   
Always return a string.

\- If you cannot determine a type, return \`"undefined"\`.

Examples:

1) \*\*Single numeric unit\*\*

• Modality: text

• Description: "The population of the city {city}"

## • Structure: single item

→ \*\*Output\*\*: \`numeric

## 2) \*\*List of titles\*\* (we use a singular form for multiple units

## of the \*same\* type)

• Modality: text

• Description: "three possible titles for the movie: {plot}"

• Structure: list of items

→ \*\*Output\*\*: \`short text

## 3) \*\*Pair of values\*\*

• Modality: text

• Description: "city names and their populations: {cities}"

• Structure: pair of items

→ \*\*Output\*\*: \`complex (short text and numeric)

## 4) \*\*Image diagram\*\* (we use a singular form for multiple units of the \*same\* type)

• Modality: image

• Description: "diagram of the network architecture"

• Structure: single item

→ \*\*Output\*\*: \`image

## 5) \*\*Timestamp range\*\*

• Modality: text

• Description: "timestamps: {start} to {end}"

• Structure: single item

→ \*\*Output\*\*: \`Time / Date

## 6) \*\*A title and a body\*\*

• Modality: text

• Description:"a text, composed in the format of a

one-sentence title followed by an email body",

• Structure: "pair of items"

→ \*\*Output\*\*: \`complex (sentence, email)\`

## #answer paradigm

## answer\_paradigm\_system\_prompt = """

You are an expert at analyzing individual output units from LLM prompts and identifying their answer paradigm—how each output unit relates to the input.

You will be provided with, for a single output unit:

\- The full prompt text.

\- The associated task(s).

\- The output\_modality for that unit.

\- The output\_description for that unit.

\- The output\_structure for that unit.

Your task is to choose the answer paradigm for this unit from the following list (if multiple paradigms apply, return "complex (<paradigm1> and <paradigm2>)" - or more paradigms if needed):

• free\_generation

• binary\_answer(s)

• binary\_answer(s)\_with\_a\_"dont\_know"\_option

• one\_option\_from\_a\_set

• several\_options\_from\_a\_set

• extracted\_span(s)

• ordering\_OR\_ranking

• clustering\_OR\_grouping

• text\_completion

• summary(s)\_OR\_paraphrase(s)

• modality\_OR\_language\_OR\_style\_transfer

• embedding\_OR\_vector\_representation

• other (specify)

## Definitions:

\- \*\*free\_generation\*\*: the output is neither extracted from the input nor a transformation (translation, style-transfer, reorder) of it.

// Example 3: Multiple choice selection

- \*\*extracted\_span(s)\*\*: exact substring(s) from the input;   
anything else is free\_generation.

- \*\*binary\_answer(s)\*\*: yes/no, true/false, 1/0.

\- \*\*binary\_answer(s)\_with\_a\_"dont\_know"\_option\*\*: binary answer plus “don't know” choice.

\- \*\*one\_option\_from\_a\_set\*\*: pick one from a provided list. - \*\*several\_options\_from\_a\_set\*\*: pick multiple from a provided list.

- Others as named.

Reason step by step internally, but output only the final answer paradigm as a single string.

If the answer\_paradigm cannot be determined, return “undefined”.

Answer paradigm: "free\_generation"

## // Example 2: Extracted span

"output\_modality": "text",   
"output\_description": "The user's name as it appears in the   
input: {user\_name}",   
"output\_structure": "single item"

Answer paradigm: "extracted\_span(s)"

"output\_modality": "text",   
"output\_description": "Choose the best summary from the   
list: {summary1}, {summary2}, {summary3}",   
"output\_structure": "single item"   
Answer paradigm: "one\_option\_from\_a\_set"

## #tasks and domains

## tasks\_system\_prompt = """

You are skilled at analyzing LLM prompts and identifying their corresponding tasks and domains.

You are given a prompt text (as a raw string). Your task is to: 1. Identify the NLP/AI Task(s):

• Match the prompt to established NLP or AI task names (e.g., summarization, question answering, NLI, paraphrasing, simplification, text-generation, code-generation, code-fixing, planning, etc.).

• Use standard and general terms. Avoid overly specific labels like "medical QA" — instead, use "question answering".

• If multiple tasks apply, list them all.

• If the task cannot be identified, use "undefined".

2. Provide a Subtask for Each Task:

• Give a more granular description of what the task is doing in this case.

• For example, for task = "summarization", a possible subtask might be "article summarization".

• Every task must have a corresponding subtask.

3. Determine the Domain(s):

• Identify the domain of the prompt (e.g., medical, finance, news, legal, travel, etc.).

• Be specific and exhaustive. If unclear or unidentifiable, use "undefined".

• If multiple domains apply, list them all.

Output: Return a JSON object with exactly two fields:

\- "task": a list of objects, each with:

• "task": the standardized task name.

• "subtask": a brief specific description.

\- "domain": a list of objects, each with:

• "domain": the domain name.

If no task or domain can be identified, use "undefined" as the value of the corresponding fields.

\*\*Output only valid JSON.\*\*

```jsonl
For example:
"task": [
{ "task": "...", "subtask": "..." },
{ "task": " , "subtask": "..." }
],
"domain": [
{ "domain": "..." },
{ "domain": "..." }
]
}
```

## #language

language\_system\_prompt = """

You are an expert at analyzing text prompts, identifying their human languages and translating them to English. Note: The word "language" here refers to human languages like French, English, Chinese etc., not to programmimg languages!

You are given a prompt text (provided as a raw string). You have to produce a JSON object with exactly two fields:

1. “languages”: a list of objects, corresponding to detected human languages in the text.

Important! Make sure this is indeed a natural human language (like German, English etc.) and \*not a programming language\*.

Each object must include three subfields:

• “language”: the language name (e.g. “English”, “Chinese”).

• “orig\_text”: the exact substring of the prompt written in that language.

• “translated\_text”: if the language is not English, its translation of that substring into English; otherwise null.

If the same language appears in separate \*non-consecutive\* segments, include multiple objects — one per segment — in the order they occur.

\*Do not split \*consecutive segments in the same language\*.   
If they are consecutive they should all form one segment.   
\*Do not split if not necessary\*.

If the language of a certain span cannot be detected, return [{"language":"Undefined","orig\_text":"<span text>", "translated\_text":None}]\`.

Once again: \*\*Avoid programming languages\*\*.

2. “explicit\_language\_mentions”: a list of objects for each place the prompt explicitly names a natural human language. Important! Make sure this is indeed a \*natural human language\* (like German, English etc.) and \*not a programming language\* or just a mention unrelated to languges (like "Turkish food").

Once again: \*\*Avoid programming languages\*\*. Each object must include 2 subfields:

• “language”: the named language (e.g. “French”).

• “mention”: the exact substring from the prompt where it is named.

\- The "mention" span should include the language name and -if possible - some minimal surrunding context (for example, "respond in French").

\- The "mention" span should be an exact span from the prompt.

Important: the "mention" should be an \*exact span\* from the prompt! If no human natural language name (like

"English","French","Japanese" etc.) is used in the prompt, just return an empty list ("explicit\_language\_mentions": []).

Important: make sure \*the "mention" indeed contains the language name\* from the "language" field (for example, "French","Spanish" etc.)

Once again: \*\*Avoid programming languages\*\*.

If there are no explicit mentions, return an empty

list("explicit\_language\_mentions": []).

## \*\*Output only valid JSON.\*\*

```ipynb
For example:
"languages": [
"language": "Chinese",
"orig_text": "<Text in Chinese>",
"translated_text": "Hello, how was your day?"
},
"language": "English",
"orig_text": "Then share your feedback. Please, respond in
French",
"translated_text": null
},
"language": "Chinese",
"orig_text": "<Text in Chinese>",
"translated_text": "Goodbye!"
}
],
"explicit_language_mentions": [
"language": "French",
"mention": "Please respond in French"
}
]
}
```

## #text extraction

text\_extraction\_system\_prompt = """"

You are an expert at extracting readable prompt text from the "messages" field of the chat.completions.create OpenAI API call parameters .

You are given the contents subfield of one of the "messages". You need to extract readable text from this "content" field if any.

\- If an intelligible, readable prompt text cannot be extracted from a certain content field, return "None" (make sure you return it as a string).

\- However, even if very little readable text can be extracted, you should still extract it.

\- If there are variable names and/or placeholders inside the prompt text, include them in braces (or any other form which fits).

\- If a variable name is enclosed into markers <VAR:...>, remove these wrapping markers but keep the variable name itself.

\- Also remove code artifacts (slashes, func, args, etc.), and anything that is not part of the text as such, substituting them with a clear placeholder where applies. For example, {'func': '<VAR:image\_b64>', 'args': ['screenshot.jpg'], 'keywords': {'quality': 'high'}}""} should become {'quality': 'high'} }""} should become

{image\_b64(screenshot.jpg,quality=high)}.

\- However, always keep any readable text even if surrounded by a lot of code remnants.

\- Be sure not to omit or invent any readable text. Only render faithfully any readable text from the original message content. - Always return your answer as a string.