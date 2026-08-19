# The Plot Thins: Uniformity and Linearity in Literary Summaries

Rebecca M. M. Hicke1 ©, Sil Hamilton² , David Mimno2 , and Ross Deans Kristensen-McLachlan³

1 Computer Science Department, Cornell University 2 Information Science Department, Cornell University 3 School of Communication and Culture, Aarhus University

## Abstract

Works of literature are complicated; they balance plot, suspense, surprise, and artistic expression. Summaries of literature prioritize plot, and therefore may deviate from their sources. Using a combination of manual and LLM-based annotation, we construct a dataset mapping sentences from 150 novel summaries to their respective source chapters. We find the task unexpectedly difficult for both human and model annotators. Using the sentence-to-chapter mappings, we then measure summary linearity, the degree to which it maintains the source's order of events, and uniformity, the degree to which a summary spreads attention equally across a source. By examining when and how summaries break linearity and uniformity, we identify differences in how literary works and summaries express plot, particularly with regard to the clarity and prominence with which narrative details are described.

Keywords: summarization, narrative understanding, plot extraction, cultural analytics, language models

## 1 Introduction

Literary works combine artistic expression with plot and events. In contrast, a summary of a literary work separates out just the plot, compressing it to only the events the summarizer finds most impactful or necessary. Modeling summarization is therefore valuable both because it helps characterize the relative significance of events, but also because, through its modifications and omissions, it highlights the artistic expression of the author. Naively, we might hypothesize that summaries follow two principles. First, linearity: the summary should refer to events in the order they occur in the original work. Importantly, this concept is independent from linear narrative — a summary can faithfully report the sequence of a story told non-linearly. Second, uniformity: the summary should allocate the same amount of space to describing each part of the original work; that is, the summary's description should be uniformly spread across the work.

At the same time, there are many reasons why a summary might deviate from linearity and uniformity. Indeed, the Wikipedia guidelines for plot summaries explicitly state that “a plot summary is not a recap."Breaks from both principles are permitted and sometimes encouraged. The guidelines specifically stress that “a plot summary... should not cover every scene or every moment of the story," and that many plots “[include] a lengthy middle section... on their way to the climatic encounter" full of events which “often clutter a plot summary with excessive and repetitive detail." On the subject of linearity, the guidelines say that “events can be reordered" if it “makes the plot easier to explain," implicitly acknowledging that narratives are not always optimized for coherence alone.1 Thus, the ways in which summaries break linearity and uniformity provide insight into the impact of artistic expression on the original works' execution of the plot.

![](images/23b01533536aae07ed3f0656cc79565056289ca425e2b04acaf5c2e469adfe9b.jpg)  
Figure 1: Our summary sentence to chapter alignment task. Novels are treated twice: they are first split into chapters, and then summarized. Summaries and chapters are then aligned

To examine how summaries represent literary texts, this work presents a corpus of 150 humanwritten summaries alongside the works they describe, as well as an automated method for aligning summary sentences with the chapters (or other delineated narrative segments) in the original work (cf. Figure 1).² We then use a combination of quantitative metrics and close readings to analyze the aligned summaries and characterize patterns in summary writing.

We find that, although summarization has long served as a task in NLP research, contemporary frontier LLMs struggle to reverse summarization, grounding summary sentences in their source texts. Nonetheless, using model-in-the-loop prompt development, we create a summary sentence-to-chapter mapping pipeline that achieves ～84% F1 on a dataset of four human-aligned summaries and texts. Examining the aligned data, we find that summarizers frequently violate linearity and uniformity, at least to a degree. Several reasons for these deviations emerge: some narrative elements or information may be explicitly described in summaries but subtly conveyed in the original works; some events are more consequential and interesting and thus emphasized in summaries; and authors of literary works may choose to reveal information in ways that heighten emotion or drama where summarizers prioritize comprehension.

## 2 Related Work

Stories, discourse, and plot. Literary texts are not required to relate story events in a clear and unambiguous manner [8]. In fact, literary texts — especially those considered high literary — may intentionally impede narrative comprehension for artistic purposes by employing stylistic devices such as retelling events non-chronologically or unreliable narrators. Implicit in this is the idea that the text and the story it describes may not always cleanly align with one another. Russian formalists in the mid-20th century (and the narratologists who later built on their work) drew a distinction between the text's story, or what happens, and its discourse, or how what happens is relayed to the reader [6; 22; 26]. One important story element is the plot, which is the causal ordering of character actions over chronological time [17]. The typical literary summary can be understood to be specifically describing these plot lines, and so can reveal what readers believe to be the causal structure underlying a narrative [19; 27]. Summaries have been used for a variety of research applications, including studying the differences between child and adult reading comprehension [12] and serving as transient structures for extracting plot-related entities like characters [25]. However, little prior work has examined how summaries typically operationalize plot — and why some story content is privileged over the rest [20] — which we address throughout this paper.

Summaries and computational narratology. Early computational work engaging with summaries largely sought to create systems capable of automatically producing journalistic headlines and scientific abstracts [21; 29]. Researchers hoped that such systems would eventually prove capable of parsing the semantic content of natural language more generally. Early summarization datasets were therefore usually drawn from news sources like CNN and NYT [10; 11; 24]. Until the 2020s, other common target domains included medical records [3], legislative documents [15], and short stories [13]. These texts shared characteristics that made them convenient for early statistical and neural approaches to natural language processing such as short mean document lengths and copious availability. With the introduction of the transformer architecture and the subsequent increase in tractable context lengths, machine learning researchers began using the summarization of long narrative texts as a test of text comprehension [14; 28; 30]. However, this research largely treats summaries as transient artifacts produced in pursuit of a different task. Works that do consider summary quality largely focus on logical consistency and clarity [4; 9]. Existing work on aligning summaries with literary exist either predominantly work on a sub-novel level due to technical limitations [2; 5] or focus on generating summaries [18]. Other prior work has explored how much of a book summaries cover, but reliance on n-gram overlap and BERT-based semantic similarity to align summaries with source texts limits utility [16].

## 3 Data

Our dataset contains 150 novels in the public domain under US law and their Wikipedia plot summaries. We select Wikipedia summaries specifically because our interest is in understanding how summaries represent and reorganize narratives, not in assessing how any individual reader summarizes a story. Because Wikipedia summaries are typically co-written by multiple authors, they will not necessarily be biased towards the writing style of an individual. We select works from the public domain to allow the release of our data for reproducibility and future research.

We gathered full novel texts from Project Gutenberg.3 To identify English-language novels available on Project Gutenberg with English-language summaries on Wikipedia, we first collected a candidate list of Gutenberg works in English. We then removed common single-word titles (which can overlap with generic Wikipedia pages) and searched for Wikipedia pages matching our candidate list containing subheadings like “Plot"and “Summary." This process yielded 80 candidate texts. We then sourced an additional 70 novels and their corresponding Wikipedia summaries from the BookSum dataset [16], selecting novels recently released into the public domain. The dataset includes several translated novels; we attempted to select standard English-language translations which should correspond to the English-language Wikipedia summaries.

We automatically split novels into chapters using the chapterize package and manually validate the resulting segmentation. We then standardized punctuation and removed illustration markers to ensure footnotes are placed in the correct chapter file. We extracted Wikipedia summaries manually,4 excluding paragraphs solely containing publication information and other similar metacommentary. We finally split summaries into sentences using the stock spaCy sentence tokenizer. We provide all novels and their corresponding Project Gutenberg IDs in Appendix A.

The novels contain between 3 to 86 chapters (with an average of 32 per book) and are between 9,780 to 488,101 tokens long, with an average of 147,173 tokens and a mean of 4,519 tokens per chapter. The summaries are 992 tokens on average, ranging from 243 to 3,534 tokens long.5

Table 1: Performance of Qwen 3.6 27B on the sentence-to-chapter mapping task. Note that scores vary considerably between novels with the F1 floating between \~70 and \~92.
<table><tr><td>Novel</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Kazan</td><td>92.3</td><td>92.3</td><td>92.3</td></tr><tr><td>Carmilla</td><td>93.2</td><td>87.3</td><td>90.2</td></tr><tr><td>Kim</td><td>78.9</td><td>90.9</td><td>84.5</td></tr><tr><td>Rookwood</td><td>70.3</td><td>70.3</td><td>70.3</td></tr><tr><td>Overall</td><td>83.7 [± 9.6]</td><td>85.2 [± 8.8]</td><td>84.3 [± 8.6]</td></tr></table>

## 4 Generating Sentence-to-Chapter Mappings

Having segmented books into chapters and summaries into sentences, our goal was next to create bipartite graphs where each edge connects one chapter to one sentence. A summary sentence is mapped to a novel chapter if it describes some event that is dramatized in that chapter. An individual sentence can map to any number of chapters, including zero, and vice versa.

We find these mappings are themselves a challenge to create. We began with an evaluation mapping dataset of four novels randomly sampled from the corpus — Rookwood by William Harrison Ainsworth, Kazan by James Oliver Curwood, Kim by Rudyard Kipling, and Carmilla by Joseph Sheridan Le Fanu — and their Wikipedia summaries. With two expert annotators, we found a high inter-annotator agreement (IAA) with a mean Cohen's κ of 0.74. Disagreements between annotators were resolved via discussion. This resulted in a final evaluation dataset containing 172 sentence-to-chapter mappings across all four novels. Despite high IAA, the annotators reported that the mapping task was challenging due to the length of both the summaries and the novel texts; each novel took approximately 1–4 hours per annotator to complete. While two (Carmilla and Kazan) were relatively straightforward, Rookwood's complex plot and Kim's complex language made their annotations more challenging.

This dataset was then used to design and evaluate prompts for mapping summary sentences to novel chapters. An initial prompt was drafted by one of the human annotators and then iteratively refined using Claude Opus 4.8 [1]. Each iteration of the mapping prompt or prompts was evaluated on the four-book annotated dataset. The model was instructed to group and categorize the errors from each run, then propose prompt edits to fix selected error types. The human annotator directed the model in choosing and designing prompt edits throughout this process.

Iterative development yielded a mapping pipeline consisting of three prompts. The first prompt performs the preliminary summary sentence to chapter mapping and is run once for every novel chapter. It accepts the entire novel summary with sentences indexed from 1-n, one novel chapter, and the set of summary sentence indices matched to previous chapters. We chose to pass the model a single chapter instead of the entire novel text because performance degraded with novel length. This prompt instructs the model to match a sentence to a chapter if an event described in it occurs in the provided chapter or if it describes a state, relationship, or characterization first introduced or directly dramatized in the chapter. The prompt also advises the model to avoid sentence-to-chapter matches under some conditions (e.g. an event is alluded to but not shown, a recurring character or setting merely appears, etc) and describes when a sentence may be matched to multiple chapters.

The next two prompts in the pipeline are run on the output of the first. The second prompt corrects the off-by-one errors common in the first prompt's output and is only run on summary sentences already matched to one or two adjacent novel chapters. It accepts a single summary sentence and its index, the chapters directly before and after the already matched chapter(s) as context, and the candidate chapter(s). Using the provided context, it instructs the model to determine whether one, both, or none of the candidate chapter(s) are true matches to the provided summary sentence. It returns a list of the chapter number(s) it determines are true matches which are used to update the existing sentence-to-chapter mapping. The third prompt is run only on sentence-to-chapter matches confirmed by the second prompt. This prompt accepts a single summary sentence and its index, the nearby summary sentences as context, a single chapter the sentence is matched to, and the index of the chapter the summary sentence would be matched to if the summary were perfectly linear. Given this information, the prompt double-checks whether this summary sentence matches the provided chapter. This prompt returns a binary judgment on whether the sentence and chapter represent a true match. Only matches confirmed by this prompt are included in the final summary sentence to chapter matches. The full text of all three prompts is provided in Section B.

![](images/57361383bca819f9813212e5048e1e7591e3cc398e9d96546d00708a6294daa4.jpg)

![](images/831aada4c771be0ca59af82cbe9b0c7300fe567247b5f14560133c693fff38a5.jpg)  
Figure 2: Linearity (a) and the proportion of chapters matched $\mathrm { t o } \geq 1$ summary sentence (b) across all 150 books. Solid orange lines mark the medians, dashed lines the quartiles. Solid black lines mark a random baseline, calculated from 200 random shuffles of the sentence and chapter matches respectively.

All prompts were run on Qwen 3.6 27B [23] quantized to 6 bits and loaded on a DGX Spark. Temperature was set to 0 and nucleus sampling (top\_p) was set to 1.0. The complete mapping pipeline achieved an F1 of 84.3% over all four novels in the evaluation dataset. However, performance varied dramatically by book (Table 1), suggesting that differences in summary or novel complexity may impact mapping accuracy. Using this pipeline, we mapped all 150 Wikipedia summaries to their respective source texts (see bipartite graphs of all mappings in Section C). We consider the performance strong enough to proceed with analysis, but caution that individual novels’ alignments may be prone to error.

## 5 Summary Linearity

We first evaluate adherence to the linearity principle. Specifically, we order the sentence-to-chapter matches by summary sentence index, sorting the chapter indices matched to each sentence. Then, we define linearity (l) as the Kendall's Tau correlation coefficient between the list of chapter indices ordered by the summary sentence they map to and the sorted version of this list. Intuitively, this captures the extent to which the summary differs from its equivalent in perfect narrative order.

Most summaries in the dataset are predominantly, but not perfectly, linear (Fig. 2a); only 6.7% (10 / 150) have $l = 1 . 0 _ { : }$ , but 62.7% (94 / 150) have $l \geq 0 . 9$ . Summaries with linearity values near the median (0.92) describe events in almost the same order as the original narrative with a small number of changes. Further inspection reveals several reasons for these “out-of-order" sentence-to-chapter matches: summaries may begin by framing the narrative using information only revealed later in the book (e.g. Sentence 1 of Section D.1; Section C, Fig. 135); single sentences may describe recurring events or a set of related events, which take place throughout the novel (e.g. Sentence 8 of Section D.2; Section C, Fig. 138); summaries may separate plot lines, even if they occur concurrently in the narrative (e.g. Sentences 12 + 13 of Section D.3; Section C, Fig. 71); or summaries may adjust the order in which information is revealed to increase clarity (e.g. Sentence 50 of Section D.3; Section C, Fig. 71). Some out-of-order matches may also be artifacts of the mapping pipeline (e.g. Sentence 14 of Section D.3; Section C, Fig. 71).

![](images/f30e66ea450cbceb6a353b036527f91442aedc3fb9d7733cc175e17ab221dd06.jpg)  
Figure 3: Bipartite graphs representing the sentence-to-chapter mappings of two summaries with unusually low linearities. Note the highly intersecting lines, indicating nonlinearity.

Although most summaries are primarily linear, the distribution is left-tailed (Fig. 2a), with clear outliers. By far the least linear summary is of The Expedition of Humphry Clinker by Tobias Smollett, with l ≈ 0.10 (Fig. 3a). Observation suggests two reasons for this summary's low linearity. First, it begins with several sentences that describe overarching plot points which cover the majority of the novel, and are therefore matched to a large number of chapters throughout the narrative. Second, the summary is, unusually, broken up by character, meaning that sequential sentences in the summary often reference events that take place in very different parts of the novel.

In contrast, the other outlier summaries often display more extreme versions of the same summary-writing choices identified in the more linear summaries. For example, the summary of L’Assommoir by Émile Zola $( l \approx 0 . 4 6 , \mathrm { F i g . 3 b ) }$ separates a secondary plotline from the main narrative (Section D.4, Sentences 15 and 16), causing a large number of out-of-order alignments. Similarly, the summary of The Marrow of Tradition by Charles W. Chesnutt $( l \approx 0 . 5 1$ ; Section C, Fig. 42) both describes overarching plot elements in single summary sentences and separates interwoven plotlines. The summary of Anne of Green Gables by L. M. Montgomery $( l \approx 0 . 5 8 ;$ Section C, Fig. 122) contains many individual sentences that refer to recurring events or character traits, and the summary of The Black Moth by Georgette Heyer $( l \approx 0 . 6 2 ;$ Section C, Fig. 97) begins with framing information only revealed much later in the narrative, separates out subplots, and describes overarching plot elements in individual summary sentences. The drop in linearity of another notable outlier, the summary of The Sound and the Fury by William Faulkner (l ≈ 0.52; Section C, Fig. 72), is likely due both to its famously complex narrative and to errors by the mapping pipeline, which struggled with Faulkner's experimental style.

Across the dataset, proportionally longer summaries tend to be more linear — linearity is moderately positively correlated with the log-ratio of summary length to book length $( r \approx 0 . 3 2$ $p < 1 0 ^ { - 3 } ) . ^ { \bar { 6 } }$ For example, the summary of Ulysses by James Joyce is both unusually long and unusually linear $( l \approx 0 . 9 3 ;$ Section C, Fig. 106). We note that books that receive considerable attention from Wikipedia editors may have longer summaries, which may contribute to this result.

Overall, we find that summaries usually follow the same ordering of events as the narratives they seek to describe. However, deviations from this ordering appear to be caused by a number of summarization techniques intended to make the narrative easier to parse for readers. These may therefore represent choices in summary writing.

![](images/dd3ea49cdafa11a4e990c802d6534cef64286f98329ab60eed2ff81720e9a0b5.jpg)  
Figure 4: Bipartite graphs representing the sentence-to-chapter mappings of two summaries with large log-ratios of summary-to-book length and a small proportion of matched chapters

## 6 Summary Uniformity

We next study whether the summaries adhere to the uniformity principle. Since each summary sentence theoretically describes an important aspect of the novel's narrative, we assume that the proportion of summary sentences matched to an individual chapter reflects how much important or relevant information that chapter contains. We proceed with this analysis using three metrics.

## 6.1 Proportion of Matched Chapters

We first calculate the proportion of chapters in each novel matched to at least one summary sentence. We find that, on average, only 79% of chapters are matched to ≥1 summary sentence and only 12% (18 / 150) of summaries cover all novel chapters. The distribution of the proportion of chapters matched $( p _ { c } )$ is broad (Fig. 2b), ranging from 0.31 to 1.0.

Proportionally longer summaries are more likely to match a larger proportion of chapters; there is a strong positive correlation between $p _ { c }$ and the log-ratio of the summary length to book length $( r \approx 0 . 5 5 , p < 1 0 ^ { - 1 1 } )$ . Intuitively, it makes sense that summaries with more “space" to discuss a book's contents are likely to cover a greater proportion of novel chapters. Summary length may also reflect the information density of the novel itself — a summary may need to be comparatively longer to accurately describe a denser narrative. Both explanations likely contribute to this relationship.

However, summary-to-book density does not entirely explain variation in $p _ { c } ;$ linear regression shows that the log-ratio of summary-to-book length explains only $R ^ { 2 } \approx 0 . 3 1$ of the variation in the proportion of novel chapters matched $( p < 1 0 ^ { - 1 2 } )$ . Some summaries have comparatively high summary-to-book length ratios but low $p _ { c }$ values. One example is the summary of The Murder of Roger Ackroyd by Agatha Christie, which maps sentences to only 56% of the novel's chapters (Fig. 4a). This work is, famously, characterized by a revelation at the end that calls into question the reliability of the narrator. Indeed, many of the un-matched chapters depict the detective revealing clues to the narrator but withholding their meaning; the summary either ignores smaller clues or presents them with their true meaning, revealed later in the novel. Similarly, the summary of Flatland: A Romance of Many Dimensions by Edwin Abbott Abbott has a high summary-to-book length log-ratio but only maps to 59% of the novel's chapters (Fig. 4b). However, many of these skipped chapters involve the narrator describing aspects of life in his world to the readers. Although the summary says that the narrator “guides the readers through some of the implications of life in two dimensions,"(Section D.5) the alignment algorithm does not successfully match this sentence to the relevant chapters, perhaps because the summary description is so broad. These examples suggest that some chapters are skipped not because they are irrelevant, but because the summary introduces information differently, either by streamlining (as in The Murder of Roger Ackroyd) or by abstracting to the intention of the chapters (as for Flatland).

![](images/e7f7a60c569a42eab6a8d650c3b7e752a3756dff950ae76a97a1f657ac1c5c6d.jpg)  
Gini Coefficient (Prop. Sents Matched to Chaps)

![](images/44c9ba2d1363dd999df43e6a25d88168b51fd5c770e876f92c9f4990bc009cfe.jpg)  
Avg. Sentence-to-Chapter Match Position  
Figure 5: Gini coefficient for sentence-to-chapter matches (a) and the average position of sentenceto-chapter matches (b) across all 150 novels. Baselines calculated from 200 random shuffles of the chapter matches.

## 6.2 Uniformity of Sentence-to-Chapter Match Distribution

In addition to skipping chapters, another violation of uniformity is to dwell on particular chapters. To quantify this, we calculate the Gini coefficient [7] of the proportion of summary sentences matched to each chapter $( G _ { c } )$ . Values closer to 0.0 indicate greater uniformity, and values closer to 1.0 indicate unequal focus. For this metric, we only consider chapters linked to at least one sentence; otherwise the results are mostly indistinguishable from the proportion of matched chapters.

The median $G _ { c }$ is 0.28 with a relatively normal distribution (Fig. 5a); all novels have $G _ { c } < 0 . 5$ We find that summaries with $G _ { c }$ values near the median have an uneven but largely continuous distribution of sentence-to-chapter matches, with only small differences between the most and least matched chapters (e.g. Section C, Figs. 26, 33 and 88).

On the most uniform end of the distribution, three summaries have $G _ { c } ~ < ~ 0 . 1$ . These are unusual for different reasons. The summary of Adam Bede by George Eliot $( G _ { c } \approx 0 . 0 4 )$ only refers to 41% of the novel's 56 chapters, but all but one of those chapters matches to exactly one of the 15 summary sentences (Section D.6). These sentences are in strictly linear order $( l = 1 . 0 )$ .

![](images/e4300e4b3bfdbd882a7f64a773ef1aaa5d241fec801179e8be378cf265b8fb89.jpg)  
Figure 6: The relationship between linearity (l) and the Gini coefficient of sentence-to-chapter matches $( G _ { c } )$ . One summary with $l < 0 . 5$ is cut from the figure. Note the lack of correlation.

![](images/38da7c198b98be3e93284305cf1ba30ad5902aeb1969c45e79297641cdb482b9.jpg)  
Figure 7: Bipartite graphs representing the sentence-to-chapter mappings of two summaries with large $G _ { c }$ values. Note the relative lack of intersecting lines, indicating linearity and proportionality

Metamorphosis by Franz Kafka $( G _ { c } \approx 0 . 0 4 )$ and A Christmas Carol by Charles Dickens $( G _ { c }$ ≈ 0.07) are both short texts (split into three and five sections respectively) each of which is paired with a similar number of summary sentences (Section C, Fig. 107 + Fig. 60). Again, both of these summaries are completely linear $( l = 1 . 0 )$ , although linearity and $G _ { c }$ are not significantly correlated more broadly (Fig. 6).

In contrast, the three summaries with the largest $G _ { c }$ values — of Anthem by Ayn Rand $( G _ { c }$ ≈ 0.44), O Pioneers! by Willa Cather $( G _ { c } \approx 0 . 4 3 )$ , and The Phantom of the Opera by Gaston Leroux $( G _ { c } \approx 0 . 4 2 )$ all map disproportionately many sentences to a subset of chapters. Some of these chapters contain important information or significant parts of the narrative. For example, the first 11 sentences of Anthem's 37-sentence summary are narrative framing, all of which is introduced in the novella's first chapter (Fig. 7a; Section D.7). ～ 20 of the 42 summary sentences for The Phantom of the Opera describe Chapter 13, Christine's first abduction by the Phantom, and another \~20% describe the second-to-last chapter (27), which contains much of the climax (Fig. 7b; Section D.8). However, disproportionate focus on a single chapter is often simply a quirk. In the summary of O Pioneers!, 12 of 60 sentences are matched only with Chapter 18; however, 11 of these sentences are an unmarked direct quote of the chapter's beginning (Fig. 8a; Section D.9).

Some evidence suggests summaries refer to events that are uniformly distributed over tokens rather than chapters: longer chapters are somewhat more likely to have summary sentences matched to them. In 127 books there is a slight positive Pearson's R correlation between chapter length and the proportion of summary sentences matched to that chapter, with an average of $r \approx 0 . 2 4$ . After Benjamini-Hochberg correction, 14% (21 / 150) correlations are significant $( p < 0 . 0 1 )$ . For example, the first chapter of Anthem has 3× more tokens than the novel's mean (6,357 vs. 2,004). Similarly, Chapter 13 of The Phantom of the Opera is over double the novel average (9,644 vs. 4,092). However, Chapter 27 of the same book is only slightly longer than average (4,228 tokens) yet is matched to more sentences, indicating many noteworthy events.

![](images/5618af7588942ceaf60f6a7f58e9b72090208882032fef079842af2a5b487f02.jpg)  
Figure 8: Bipartite graphs representing the sentence-to-chapter mappings of O Pioneers! which features larger $G _ { c }$ values. Note a few chapters receive a disproportionate number of sentences.

![](images/5c5b8c60c19143e0f8aff5712e7b8c642e7561c4c21e79a5919697cb2a91f47a.jpg)  
Figure 9: Bipartite graphs representing the sentence-to-chapter mappings of two summaries with near median $\mu _ { m }$ values. Note the Gini coefficient appears to be correlated with summary length.

In contrast to linearity, we find $G _ { c }$ is moderately positively correlated with summary length $( r \approx 0 . 4 2 , p < 1 0 ^ { - 6 } )$ — longer summaries are somewhat more likely to dwell on certain chapters. Interestingly, $G _ { c }$ is not significantly correlated with the log-ratio of summary-to-book length or novel length, only summary length.

## 6.3 Position of Sentence-to-Chapter Matches Across Novels

Finally, having confirmed that summary sentence-to-chapter matches are not equally distributed throughout novels, we look at where in the narrative the matches are concentrated. For each summary, we calculate the position in the novel of the average sentence-to-chapter match. Specifically, if there are $n _ { c }$ chapters and the set M of sentence-to-chapter matches, where each match m consists of the sentence index $( s _ { m } )$ and the chapter index $( c _ { m } )$ , we calculate:

$$
\mu _ { m } = \frac { \Sigma _ { m \in M } \frac { c _ { m } - 1 } { n _ { c } - 1 } } { | M | } .
$$

A summary that only references the first chapter would have a value close to 0.0, while a summary that only references the final chapter would be close to 1.0.

In this dataset, $\mu _ { m }$ ranges from 0.32 to 0.71 with a median of 0.52, suggesting that on average sentence-to-chapter matches are evenly spread between novel halves, with slightly more focus placed on novels’second halves (Fig. 5b). Novels with $\mu _ { m }$ values near the median still differ in their distribution of summary attention. For example, the summary of Vile Bodies by Evelyn Waugh $( \mu _ { m } \approx 0 . 5 2 )$ distributes summary attention mostly evenly across each chapter $( G _ { c } \approx 0 . 1 8 ;$ Fig. 9a). In contrast, the summary of The House of Mirth by Edith Wharton $( \mu _ { m } \approx 0 . 5 1 )$ distributes attention very unevenly per-chapter $( G _ { c } \approx 0 . 4 1 ; \mathrm { F i g . 9 b ) }$ , but evenly across the novel halves.

![](images/9cc3089368b6d049497fe3058c0db0838e7bad9eb3d2b87b90138407bf35e7de.jpg)  
Figure 10: Bipartite graph representing the sentence-to-chapter mapping of one summary with low $\mu _ { m }$ value. Note the summary appears to place more focus on the first half of the novel.

![](images/03b9895e3bb7e6a75b85454be344c778215ce404a1c0c04d54aba1c414350b3b.jpg)  
Figure 11: Bipartite graphs representing the sentence-to-chapter mappings of two summaries with high $\mu _ { m }$ values. Note the relative uniform assortment of sentence to chapter pairs.

Summaries with low $\mu _ { m }$ values place more focus on the first halves of novels. We find that some of these summaries skip the novels’ endings. For example, the summaries of The Triumph of the Scarlet Pimpernel by Baroness Orczy $( \mu _ { m } \approx 0 . 3 2 )$ , Beauvallet by Georgette Heyer $( \mu _ { m }$ ≈ 0.35), and The Golovlyov Family by Mikhail Saltykov-Shchedrin $( \mu _ { m } \approx 0 . 3 5 )$ do not cover the last 12, 3, and 6 chapters of the novels respectively (Section C, Figs. 98, 123 and 132). All of these summaries end with open-ended sentences (Sections D.10 to D.12) — the summary of The Triumph of the Scarlet Pimpernel ends with “Lady Blakeney is kidnapped yet again and taken to France and imprisoned as bait for Sir Percy." — highlighting an unexpected summarization trend. Another group of summaries with low $\mu _ { m }$ values include descriptions of novels’ endings, but skip some other series of chapters in the novels' second halves. These include the summaries of Anne of Green Gables by L. M. Montgomery $( \mu _ { m } \approx 0 . 3 8 )$ , David Copperfield by Charles Dickens $( \mu _ { m } \approx 0 . 3 8 )$ Main Street by Sinclair Lewis $( \mu _ { m } \approx 0 . 4 0 )$ , and Captain Blood by Rafael Sabatini $( \mu _ { m } \approx 0 . 3 3 )$ (Section C, Figs. 58, 113, 122 and 130). In these summaries, there is often a summary sentence matched to a chapter(s) preceding the unmatched stretch that describes a series of events or a period of time (Sections D.13 to D.16); for example, the ninth sentence of the summary of Main Street, which is matched to chapters 8–10, states that “despite her efforts, these ventures are ineffective and she is constantly derided by the leading cliques."After this, six novel chapters go unmatched. These unmatched novel sections may represent a stretch of narrative that subtly repeats events described in these broad summary sentences or may simply capture less “eventful" or “interesting" swathes of the novels. A final category of front-weighted summaries include extensive description of the novels' first chapters either because they include detailed novel framing (e.g. Anthem by Ayn Rand — Fig. 7a, Section D.7) or are particularly eventful (e.g. The Wind in the Willows by Kenneth Grahame — Fig. 10a, Section D.17). Low $\mu _ { m }$ values may also be impacted by the existence of other eventful chapters in the novels’ first halves (again, e.g. The Wind in the Willows) and the novel properties described here may coexist.

Summaries often have high $\mu _ { m }$ values for the inverse of the reasons described above. Some summaries ignore chapters at or near novels' beginnings. For example, in Flatland by Edwin Abbott Abbott $( \mu _ { m } \approx 0 . 7 1 )$ , chapters 5–12 are not matched with any part of the summary (Fig. 4b). The second summary sentence (Section D.5) — “The narrator is a square, a member of the caste of gentlemen and professionals, who guides the readers through some of the implications of life in two dimensions." — broadly describes these chapters' content, but in too general terms for the sentence mapping pipeline to identify. Similarly, the first four chapters of Nostromo: A Tale of the Seaboard by Joseph Conrad $( \mu _ { m } \approx 0 . 6 7 )$ are not matched with any summary sentence due to mistakes by the mapping pipeline (Section C, Fig. 48); although several summary sentences describe events from these chapters (Section D.18) they are presented with different context and in a different order than in the novel, likely causing the errors. Another group of summaries, including those of This Side of Paradise by F. Scott Fitzgerald $( \mu _ { m } \approx 0 . 6 7 )$ , Heart of Darkness by Joseph

![](images/e6a6ef79dcca2048dc922a3015002250476913582fbb0fd55dadc8ab2294571b.jpg)  
Figure 12: Mean off-diagonal distance for all 150 novels. Note measured off-diagonal distances nearly all fall below the random baseline of 0.36, indicating plot editing. Baseline calculated from 200 random shuffles of the sentence matches.

Conrad $( \mu _ { m } \approx 0 . 6 2 )$ , and Fanshawe by Nathaniel Hawthorne $( \mu _ { m } \approx 0 . 6 2 )$ , pays considerable attention to novels' last chapters (Fig. 11a; Section C, Figs. 46 and 93), which appear to contain a large portion of the summarized events (Sections D.19 to D.21). Finally, some summaries appear to broadly describe the novels’ second halves in more detail. These include the summaries of A Tale of Two Cities by Charles Dickens $( \mu _ { m } \approx 0 . 6 1 )$ , Northanger Abbey by Jane Austen $( \mu _ { m } \approx 0 . 6 1 )$ ， The Good Soldier by Ford Maddox Ford $( \mu _ { m } \approx 0 . 6 1 )$ , and A High Wind in Jamaica by Richard Hughes $( \mu _ { m } \approx 0 . 6 2 )$ (Fig. 11b; Section C, Figs. 56, 75 and 101). Again, many summaries share multiple of the described characteristics, increasing their $\mu _ { m }$ value.

These three metrics — the proportion of matched chapters, the Gini coefficient of sentence-tochapter matches, and the average sentence-to-chapter match position — reveal that attention is not distributed equally in most summaries. Some variance in these metrics can be attributed to errors in the sentence-to-chapter mappings, as the specific examples cited above demonstrate, but many reveal true summarization choices. Some novel chapters are clearly more “eventful" according to the summaries, even when they are not longer texts. Often, the distribution of summary attention seems to reflect choices made by the summarizers or aspects of novel construction. Overall, we find that summaries do make choices about what text represents “interesting" narrative information, as the Wikipedia instructions encourage; novel text is not of equal importance.

## 7 Combining Linearity and Uniformity

We next describe each summary's distance from our naive uniform and linear summary in a single metric. While this flattens some of the nuance the previous metrics allowed, it facilitates a more direct comparison between summaries. Specifically, we look at the mean off-diagonal distance $\left( \mu _ { O D D } \right)$ of each summary by averaging the distance of each sentence-to-chapter match from the perfectly linear, even matching of summary sentences to novel chapters — represented as the diagonal of the sentence-to-chapter matrix. Mean off-diagonal distance is negatively correlated with linearity $( r \approx - 0 . 5 2 , p < 1 0 ^ { - 9 } )$ and the proportion of chapters matched $( r \approx - 0 . 4 3 , p < 1 0 ^ { - 6 } )$ confirming that summaries with higher $\mu _ { O D D }$ are generally both less linear and uniform.

The median off-diagonal distance for all human-written summaries is relatively low (μODD≈ 0.11), but ranges from 0.02 to 0.35 (Fig. 12). Summaries with low $\mu _ { O D D }$ values resemble the naive baseline, and are mostly linear with relatively even distribution of summary attention across the novel. Examples of this type of summary include those of Mike by P. G. Wodehouse $( \mu _ { O D D }$ ≈ 0.02), The Cave Girl by Edgar Rice Burroughs $\left( \mu _ { O D D } \approx 0 . 0 3 \right)$ , Sister Carrie: A Novel by Theodore Dreiser $( \mu _ { O D D } \approx 0 . 0 3 )$ , Around the World in Eighty Days by Jules Verne $( \mu _ { O D D } \approx$ 0.04), and Greenmantle by John Buchan $( \mu _ { O D D } \approx 0 . 0 4 )$ (Section C, Figs. 33, 37, 64, 148 and 161). The summaries with comparatively large μODD values, in contrast, are either non-linear, distribute attention unevenly across the novel they summarize, or combine both. Many of these summaries were highlighted as outliers or extremes in the previously discussed metrics: the summary of The Expedition of Humphry Clinker $( \mu _ { O D D } \approx 0 . 3 5 )$ is extremely non-linear (Fig. 3a), whereas the summaries of Beauvallet $( \mu _ { O D D } \approx 0 . 3 2 )$ , The Triumph of the Scarlet Pimpernel $( \mu _ { O D D } \approx 0 . 2 8 )$ Captain Blood $( \mu _ { O D D } \approx 0 . 2 7 )$ , and Main Street $( \mu _ { O D D } \approx 0 . 2 7 )$ are all predominantly linear but heavily non-uniform (Section C, Figs. 98, 113, 123 and 130).

![](images/4bb031b0fa7e131cad5950abb436d434c1572846764e326e6d15b9c5d01e7bc6.jpg)

![](images/88a99d752e817f6a485df73f5a39e0b41e200d0aa30464a7c69f7a953964f7a7.jpg)  
Gini Coefficient (Prop. Chaps Matched to Sents)  
Figure 13: Proportions of matched summary sentences (a) and the Gini coefficient of chapterto-sentence matches (b) for all 150 human-authored summaries. Note nearly all sentences were matched with chapters, indicating the pipeline found success in alignment. Baselines calculated from 200 random shuffles of the chapter and sentence matches respectively.

We find that $\mu _ { O D D }$ is also moderately negatively correlated with the log-ratio of summaryto-book length $( r \approx - 0 . 3 6 , p < 1 0 ^ { - 4 } )$ , suggesting that the less “space" the summary has to describe the book, the less linear and even the summaries will be. It seems that the more a summary condenses the novel, the more changes they make to how the narrative is delivered.

## 8 Differences in Summary Sentences

Having examined how summaries project onto novels, we now study the reverse. Here, we are interested in whether summary sentences are comparable units of information. We look at two metrics: the proportion of summary sentences matched to at least one novel chapter $( p _ { s } )$ and the Gini coefficient of the proportion of novel chapters matched to each summary sentence $( G _ { s } )$

We find all sentences are matched to at least one novel chapter $( p _ { s } = 1 . 0 )$ in 52% (78 / 150) of summaries and at least 90% of sentences are matched in 91% (136 / 150) of summaries. However, the distribution of $p _ { s }$ is again left-tailed (Fig. 13a). Unmatched summary sentences are typically mapping errors, e.g. most of the unmatched sentences in the summary The Sound and the Fury by William Faulkner $( p _ { s } \approx 0 . 6 0 ;$ Section C, Fig. 72) describe matchable events (Section D.22) but the stylistic complexity of the novel appears to confound the mapping pipeline. Similarly, the unmatched sentences in the summaries of Pollyanna by Eleanor H. Porter $( p _ { s } \approx 0 . 7 9 )$ , The Seven Dials Mystery by Agatha Christie $( p _ { s } \approx 0 . 8 0 )$ , Adam Bede by George Eliot $( p _ { s } \approx 0 . 8 0 )$ , and Wolfbane by Frederik Pohl and C. M. Kornbluth $( p _ { s } \approx 0 . 8 1 )$ all appear to be alignment errors (Sections D.6 and D.23 to D.25). However, a smaller subset of the unmatched sentences deliver meta-commentary about the novel and its history or framing information never directly stated in the novel. For example, the three unmatched sentences in the summary of Germinal by Émile Zola (Section D.26) all fall into this category, as do the single unmatched sentences in the summaries of Ulysses by James Joyce (Section D.27) and Micromegas by Voltaire (Section D.28). We see that summaries may contain information more than a mere synthesis of novel events despite the Wikipedia plot summary guideline suggesting summaries “avoid commentary."

![](images/309adf86d91a3a046d972df3c6f24ef14bcc66c546235354f0595adf66b824e5.jpg)

![](images/93a4a44c026ec0960a8f1169ce5572a9d3c3816d58c58de594a8ebba34656034.jpg)  
Figure 14: Bipartite graphs representing the sentence-to-chapter mappings of two summaries with low $G _ { s }$ values.

$G _ { s }$ values are distributed more widely than the $G _ { c }$ values (Fig. 13b, Fig. 5a), ranging from 0.00 to 0.58, although the median $G _ { s }$ value is less than the median $G _ { c } ( 0 . 2 2 \mathrm { v s . } 0 . 2 8 )$ . Summaries with very low $G _ { s }$ values, like those of Metamorphosis by Franz Kafka $( G _ { s } \approx 0 . 0 0 )$ , Heart of Darkness by Joseph Conrad $( G _ { s } \approx 0 . 0 0 )$ , and Ulysses by James Joyce $( G _ { s } \approx 0 . 0 1 )$ (Figs. 14a and 14b; Section C, Fig. 46), usually match each sentence with 1–2 chapters and are often strongly linear. Indeed, we find there is negative correlation between linearity and $G _ { s } ( r \approx - 0 . 5 0 , p < 1 0 ^ { - 8 } )$ suggesting summaries with even distributions of novel chapters across sentences are usually more linear. Summaries with low $G _ { s }$ values also tend to be more uniform; $G _ { s }$ is negatively correlated with the proportion of novel chapters matched $( r \approx - 0 . 4 0 , p < 1 0 ^ { - 5 } )$ . It is also positively correlated with mean off-diagonal distance $( r \approx 0 . 4 3 , p < 1 0 ^ { - 6 } )$ (the combination of linearity and uniformity) suggesting a uniform distribution of novel chapters across summary sentences reflects summary's adherence to a naive summary structure. Finally, $G _ { s }$ is strongly negatively correlated with the log-ratio of summary-to-book length $( r \approx - 0 . 6 0 , p < 1 0 ^ { - 1 3 } )$ , again suggesting the more summary space' there is per book text, the more standard the summaries will be.

## 9 Conclusion

In this work, we examine summaries of literary works as valuable and interesting interpretations of their sources and as cultural artifacts themselves. We find that summaries vary in form and structure, seemingly impacted by both summarization choices driven by stylistic desire and the internal structure of the works they summarize. Moreover, we find summaries often break from expectations of linearity (the expectation that summaries describe events in the same order as source works) and uniformity (the expectation that summaries attend equally to all of the source) to different degrees. The reasons that summarizers choose to break these principles (e.g. to emphasize significant events or to clarify events and themes) provide insight into both the role of the summary and the artistic choices the original authors made in constructing their narratives.

Next steps. Our work hints at a number of productive research threads to follow on. In constructing our dataset, we found that mapping summary sentences to their origins in the source texts was surprisingly challenging for both human annotators and frontier LLMs. While our work contributes a small evaluation dataset and examples of automated mappings, we have identified opportunities to expand both the dataset and mapping pipeline. Moreover, further formalizations of summary-to-source mapping may allow for more fine-grained analysis of this topic.

Additionally, while we restrict our attention to Wikipedia summaries, expanding to solo humanauthored or AI-authored summaries would be a promising way to characterize the distribution of summarization choices. Finally, while our close-reading analysis of the alignment results offers a reason why summary writers violate linearity and uniformity, we believe there is an opportunity for further work in predicting how summarizers will respond based on the source work.

## Acknowledgments

We would like to thank Axel Bax, Federica Bologna, Kiara Liu, Sanghoon Oh, Andrea Wang, Matthew Wilkens, and Shengqi Zhu for their thoughtful feedback. This work was supported in part by the Cornell Foundational AI PhD Fellowship, the Schmidt Sciences Humanities and AI Virtual Institute, and the Danish National Research Foundation via TEXT: Center for Contemporary Cultures of Text, grant number DNRF193. We additionally acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC); nous remercions le Conseil de recherches en sciences naturelles et en génie du Canada (CRSNG) de son soutien.

## References

[1] Anthropic. "System Card: Claude Opus 4.8". Tech. rep. Anthropic, 2026. URL: https : //www-cdn.anthropic.com/0b4915911bb0d19eca5b5ee635c80fef830a37ea.pdf.

[2] Bamman, David and Smith, Noah A. “New alignment methods for discriminative book summarization". In: arXiv preprint arXiv:1305.1319 (2013).

[3] Baumel, Tal, Cohen, Raphael, and Elhadad, Michael. "Topic concentration in query focused summarization datasets". In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. 1. 2016.

[4] Chang, Yapei, Lo, Kyle, Goyal, Tanya, and Iyyer, Mohit. "Booookscore: A systematic exploration of book-length summarization in the era of llms". In: International Conference on Learning Representations. Vol. 2024. 2024, pp. 56059–56093.

[5] Chaudhury, Atef, Tapaswi, Makarand, Kim, Seung Wook, and Fidler, Sanja. “The shmoop corpus: A dataset of stories with loosely aligned summaries". In: arXiv preprint arXiv:1912.13082 (2019).

[6] Genette, Gérard. Narrative discourse: An essay in method. Vol. 3. Cornell University Press, 1983.

[7] Gini, Corrado. Variabilità e mutabilità: contributo allo studio delle distribuzioni e delle relazioni statistiche.[Fasc. I.] Tipogr. di P. Cuppini, 1912.

[8] Gius, Evelyn and Vauth, Michael. “Towards an event based plot model. a computational narratology approach". In: Journal of Computational Literary Studies 1, no. 1 (2022).

[9] Goyal, Tanya, Li, Junyi Jessy, and Durrett, Greg. “SNaC: Coherence error detection for narrative summarization”. In: Proceedings of the 2022 conference on empirical methods in natural language processing. 2022, pp. 444–463.

[10] Grusky, Max, Naaman, Mor, and Artzi, Yoav. "Newsroom: A dataset of 1.3 million summaries with diverse extractive strategies". In: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers). 2018, pp. 708–719.

[11] Hermann, Karl Moritz, Kocisky, Tomas, Grefenstette, Edward, Espeholt, Lasse, Kay, Will, Suleyman, Mustafa, and Blunsom, Phil. "Teaching machines to read and comprehend". In: Advances in neural information processing systems 28 (2015).

[12] Johnson, Nancy S. "What do you do if you can't tell the whole story? The development of summarization skills". In: Children's language (2014), pp. 315–383.

[13] Kazantseva, Anna and Szpakowicz, Stan. "Summarizing short stories". In: Computational Linguistics 36, no. 1 (2010), pp. 71–109.

[14] Kim, Yekyung, Chang, Yapei, Karpinska, Marzena, Garimella, Aparna, Manjunatha, Varun, Lo, Kyle, Goyal, Tanya, and Iyyer, Mohit. "Fables: Evaluating faithfulness and content selection in book-length summarization". In: arXiv preprint arXiv:2404.01261 (2024).

[15] Kornilova, Anastassia and Eidelman, Vladimir. “BillSum: A corpus for automatic summarization of US legislation". In: Proceedings of the 2nd Workshop on New Frontiers in Summarization. 2019, pp. 48–56.

[16] Kryściński, Wojciech, Rajani, Nazneen, Agarwal, Divyansh, Xiong, Caiming, and Radev, Dragomir. “BookSum: A Collection of Datasets for Long-form Narrative Summarization". 2021. arXiv: 2105.08209[cs.CL].

[17] Kukkonen, Karin. "Plot". In: The Living Handbook of Narratology, ed. by Peter Hühn et al. Viewed 25 January 2014, revised 24 March 2014. Hamburg University / Interdisciplinary Center forNarratology, 2014. URL: https://www-archiv.fdm.uni-hamburg.de/ lhn/node/115.html.

[18] Ladhak, Faisal, Li, Bryan, Al-Onaizan, Yaser, and McKeown, Kathleen. "Exploring content selection in summarization of novel chapters". In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020, pp. 5043–5054.

[19] Lehnert, Wendy G. “Plot units and narrative summarization". In: Cognitive Science 5, no. 4 (1981), pp. 293–331. DOI: 10.1016/S0364-0213(81)80016-X.

[20] Mani, Inderjeet. “Plot Basics". In: Narrative and Generative AI: A Computational Account. Springer, 2025, pp. 123–149.

[21] Mani, Inderjeet and Maybury, Mark T. Advances in automatic text summarization. MIT press, 1999.

[22] Prince, Gerald. Narratology: The form and functioning of narrative. en. Vol. 108. Walter de Gruyter, 2012.

[23] Qwen Team. “Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model". Apr. 2026. URL:https://qwen.ai/blog?id=qwen3.6-27b.

[24] Sandhaus, Evan. “The New York Times Annotated Corpus". In: (2008).

[25] Srinivasan, Vardhini and Power, Aurelia. “Character extraction and character type identification from summarised story plots". In: Journal of Computer-Assisted Linguistic Research 6 (2022), pp. 19–41.

[26] Tomashevsky, Boris. Russian Formalist Criticism: Four Essays. University of Nebraska Press, 1965.

[27] Trabasso, Tom and van den Broek, Paul. "Causal thinking and the representation of narrative events". In: Journal of Memory and Language 24, no. 5 (1985), pp. 612–630. DOI: 10 . 1016/0749-596X(85)90049-X.

[28] Wu, Jeff, Ouyang, Long, Ziegler, Daniel M, Stiennon, Nisan, Lowe, Ryan, Leike, Jan, and Christiano, Paul. “Recursively summarizing books with human feedback". In: arXiv preprint arXiv:2109.10862 (2021).

[29] Zhang, Haopeng, Yu, Philip S, and Zhang, Jiawei. “A systematic survey of text summarization: From statistical methods to large language models". In: ACM Computing Surveys 57, no. 11 (2025), pp. 1–41.

[30] Zhao, Chao, Brahman, Faeze, Song, Kaiqiang, Yao, Wenlin, Yu, Dian, and Chaturvedi, Snigdha. “NarraSum: A Large-Scale Dataset for Abstractive Narrative Summarization". 2023. arXiv: 2212.01476[cs.CL].

## A Novel Dataset

<table><tr><td>Author</td><td>Title</td><td>Publication Date</td><td>ID</td></tr><tr><td>Abbott, Edwin Abbott</td><td>Flatland: A Romance of Many Dimensions</td><td>1884</td><td>45506</td></tr><tr><td>Ainsworth, William Harrison</td><td>Rookwood</td><td>1834</td><td>23564</td></tr><tr><td>Alcott, Louisa May</td><td>Little Women</td><td>1868-9</td><td>514</td></tr><tr><td>Alcott, Louisa May</td><td>Moods</td><td>1864</td><td>28203</td></tr><tr><td>Alcott, Louisa May</td><td>Work: A Story of Experience</td><td>1873</td><td>4770</td></tr><tr><td>Austen, Jane</td><td>Emma</td><td>1816</td><td>158</td></tr><tr><td>Austen, Jane</td><td>Mansfield Park</td><td>1814</td><td>141</td></tr><tr><td>Austen, Jane</td><td>Northanger Abbey</td><td>1818</td><td>121</td></tr><tr><td>Austen, Jane</td><td>Persuasion</td><td>1818</td><td>105</td></tr><tr><td>Austen, Jane</td><td>Pride and Prejudice</td><td>1813</td><td>1342</td></tr><tr><td>Baum, L. Frank</td><td>Rinkitink in Oz</td><td>1916</td><td>25581</td></tr><tr><td>Baum, L. Frank</td><td>The Wonderful Wizard of Oz</td><td>1900</td><td>55</td></tr><tr><td>Bellamy, Edward</td><td>Equality</td><td>1897</td><td>7303</td></tr><tr><td>Braddon, M. E.</td><td>Lady Audley's Secret</td><td>1862</td><td>8954</td></tr><tr><td>Brontë, Charlotte</td><td>Jane Eyre</td><td>1847</td><td>1260</td></tr><tr><td>Brontë, Charlotte</td><td>Shirley</td><td>1849</td><td>30486</td></tr><tr><td>Brontë, Charlotte</td><td>Villette</td><td>1853</td><td>9182</td></tr><tr><td>Brontë, Emily</td><td>Wuthering Heights</td><td>1847</td><td>768</td></tr><tr><td>Buchan, John</td><td>Greenmantle</td><td>1916</td><td>559</td></tr><tr><td>Buchan, John</td><td>Huntingtower</td><td>1922</td><td>3782</td></tr><tr><td>Burroughs, Edgar Rice</td><td>The Cave Girl</td><td>1925</td><td>69191</td></tr><tr><td>Burnett, Frances Hodgson</td><td>A Little Princess</td><td>1905</td><td>146</td></tr><tr><td>Burney, Fanny</td><td>Evelina</td><td>1778</td><td>6053</td></tr><tr><td>Butler, Samuel Carroll, Lewis</td><td>The Way of All Flesh</td><td>1903</td><td>2084 11</td></tr><tr><td></td><td>Alice's Adventures in Wonder- land</td><td>1865</td><td></td></tr><tr><td>Cather, Willa Cather, Willa</td><td>My Ántonia</td><td>1918</td><td>242</td></tr><tr><td>Chesnutt, Charles W.</td><td>O Pioneers!</td><td>1913</td><td>24</td></tr><tr><td></td><td>The Marrow of Tradition</td><td>1901</td><td>11228</td></tr><tr><td>Christie, Agatha</td><td>The Murder of Roger Ackroyd</td><td>1926</td><td>69087</td></tr><tr><td>Christie, Agatha</td><td>The Seven Dials Mystery</td><td>1929</td><td>75288</td></tr><tr><td>Collins, Wilkie</td><td>Armadale</td><td>1866</td><td>1895</td></tr><tr><td>Cooper, James Fenimore</td><td>The Deerslayer</td><td>1841</td><td>3285</td></tr><tr><td>Cooper, James Fenimore</td><td>The Last of the Mohicans</td><td>1826</td><td>940</td></tr><tr><td>Conrad, Joseph</td><td>Heart of Darkness</td><td>1902</td><td>219</td></tr><tr><td>Conrad, Joseph</td><td>Nostromo: A Tale of the Seaboard</td><td>1904</td><td>2021</td></tr><tr><td>Conrad, Joseph</td><td>The Secret Agent: A Simple Tale</td><td>1907</td><td>974</td></tr><tr><td>Conrad, Joseph</td><td>Victory: An Island Tale</td><td>1915</td><td>6378</td></tr><tr><td>Crane, Stephen</td><td>Maggie: A Girl of the Streets</td><td>1893</td><td>447</td></tr><tr><td>Crane, Stephen</td><td>The Red Badge of Courage</td><td>1895</td><td>73</td></tr><tr><td>Curwood, James Oliver</td><td>Kazan</td><td>1914</td><td>10084</td></tr><tr><td>Defoe, Daniel</td><td>The Further Adventures of Robinson Crusoe</td><td>1719</td><td>561</td></tr><tr><td>Dickens, Charles</td><td>A Christmas Carol</td><td>1843</td><td>19337</td></tr><tr><td>Dickens, Charles</td><td>David Copperfield</td><td>1850</td><td>766</td></tr><tr><td>Dickens, Charles</td><td>Little Dorrit</td><td>1857</td><td>963</td></tr><tr><td>Dickens, Charles</td><td>Oliver Twist</td><td>1838</td><td>730</td></tr><tr><td>Dickens, Charles</td><td>A Tale of Two Cities</td><td>1859</td><td>98</td></tr><tr><td>Doyle, Arthur Conan</td><td>The Hound of the Baskervilles</td><td>1902</td><td>2852</td></tr><tr><td>Doyle, Arthur Conan</td><td>A Study in Scarlet</td><td>1888</td><td>244</td></tr><tr><td>Doyle, Arthur Conan</td><td>The Valley of Fear</td><td>1914</td><td>3289</td></tr><tr><td>Dreiser, Theodore</td><td>Sister Carrie: A Novel</td><td>1900</td><td>233</td></tr><tr><td>Du Maurier, George</td><td>Trilby</td><td>1894</td><td>39858</td></tr><tr><td>Dumas, Alexandre</td><td>The Three Musketeers</td><td>1844</td><td>1257</td></tr><tr><td>Eliot, George</td><td>Adam Bede</td><td>1859</td><td>507</td></tr><tr><td>Eliot, George</td><td>Middlemarch</td><td>1871-2</td><td>145</td></tr><tr><td>Eliot, George</td><td>The Mill on the Floss</td><td>1860</td><td>6688</td></tr><tr><td>Eliot, George</td><td>Romola</td><td>1862-3</td><td>24020</td></tr><tr><td>Falkner, John Meade</td><td>Moonfleet</td><td>1898</td><td>10743</td></tr><tr><td>Faulkner, William</td><td>The Sound and the Fury</td><td>1929</td><td>75170</td></tr><tr><td>Fitzgerald, F. Scott</td><td>This Side of Paradise</td><td>1920</td><td>805</td></tr><tr><td>Flaubert, Gustave</td><td>Madame Bovary</td><td>1857</td><td>2413</td></tr><tr><td>Ford, Ford Madox</td><td>The Good Soldier</td><td>1915</td><td>2775</td></tr><tr><td>Forster, E. M.</td><td>Howards End</td><td>1910</td><td>2891</td></tr><tr><td>Forster, E. M.</td><td>A Room with a View</td><td>1908</td><td>2641</td></tr><tr><td>Forster, E. M.</td><td>Where Angels Fear to Tread</td><td>1905</td><td>2948</td></tr><tr><td>Galdos, Benita Perez</td><td>Marianela</td><td>1878</td><td>48818</td></tr><tr><td>Gaskell, Elizabeth</td><td>Mary Barton</td><td>1848</td><td>2153</td></tr><tr><td>Gilman, Charlotte Perkins</td><td>Herland</td><td>1979</td><td>32</td></tr><tr><td>Gissing, George</td><td>Demos</td><td>1886</td><td>4309</td></tr><tr><td>Glasgow, Ellen</td><td>Virginia</td><td>1913</td><td>26316</td></tr><tr><td>Goldsmith, Oliver</td><td>The Vicar of Wakefield</td><td>1766</td><td>2667</td></tr><tr><td>Goncharov, Ivan</td><td>Oblomov</td><td>1859</td><td>54700</td></tr><tr><td>Grahame, Kenneth</td><td>The Wind in the Willows</td><td>1908</td><td>289</td></tr><tr><td>Haggard, H. Rider</td><td>She: A History of Adventure</td><td>1887</td><td>3155</td></tr><tr><td>Hammett, Dashiell</td><td>The Maltese Falcon</td><td>1930</td><td>77600</td></tr><tr><td>Hardy, Thomas</td><td>Jude the Obscure</td><td>1895</td><td>153</td></tr><tr><td>Hardy, Thomas</td><td>The Return of the Native</td><td>1878</td><td>122</td></tr><tr><td>Harrison, Harry</td><td>Deathworld</td><td>1960</td><td>28346</td></tr><tr><td>Hawthorne, Nathaniel</td><td>Fanshawe</td><td>1828</td><td>7085</td></tr><tr><td>Hawthorne, Nathaniel</td><td>The House of the Seven Gables</td><td>1851</td><td>77</td></tr><tr><td>Hawthorne, Nathaniel</td><td>The Scarlet Letter</td><td>1850</td><td>25344</td></tr><tr><td>Hemingway, Ernest</td><td>A Farewell to Arms</td><td>1929</td><td>75201</td></tr><tr><td>Hesse, Hermann</td><td>Siddhartha</td><td>1922</td><td>2500</td></tr><tr><td>Heyer, Georgette</td><td>Beauvallet</td><td>1929</td><td>75547</td></tr><tr><td>Heyer, Georgette</td><td>The Black Moth: A Romance of the XVIIIth Century</td><td>1921</td><td>38703</td></tr><tr><td>Howells, William Dean</td><td>The Rise of Silas Lapham</td><td>1885</td><td>154</td></tr><tr><td>Hudson, W. H.</td><td>Green Mansions: A Romance of the Tropical Forest</td><td>1904</td><td>942</td></tr><tr><td>Hughes, Richard</td><td>A High Wind in Jamaica</td><td>1929</td><td>75530</td></tr><tr><td>Hugo, Victor</td><td>Ninety-Three</td><td>1874</td><td>49372</td></tr><tr><td>Jackson, Helen Hunt</td><td>Ramona</td><td>1884</td><td>2802</td></tr><tr><td>James, Henry</td><td>The Turn of the Screw</td><td>1898</td><td>209</td></tr><tr><td>James, Henry</td><td>What Maisie Knew</td><td>1897</td><td>7118</td></tr><tr><td>Joyce, James</td><td>Ulysses</td><td>1922</td><td>4300</td></tr><tr><td>Kafka, Franz</td><td>Metamorphosis</td><td>1915</td><td>5200</td></tr><tr><td>Keene, Carolyn</td><td>The Hidden Staircase</td><td>1930</td><td>77602</td></tr><tr><td>Kipling, Rudyard</td><td>Kim</td><td>1901</td><td>2226</td></tr><tr><td>Le Fanu, Joseph Sheridan</td><td>Carmilla</td><td>1872</td><td>10007</td></tr><tr><td>Lawrence, D. H.</td><td>Sons and Lovers</td><td>1913</td><td>217</td></tr><tr><td>Lindsay, David</td><td>A Voyage to Arcturus</td><td>1920</td><td>1329</td></tr><tr><td>Leroux, Gaston</td><td>The Phantom of the Opera</td><td>1910</td><td>175</td></tr><tr><td>Lewis, Sinclair</td><td>Arrowsmith</td><td>1925</td><td>70875</td></tr><tr><td>Lewis, Sinclair</td><td>Babbitt</td><td>1922</td><td>1156</td></tr><tr><td>Lewis, Sinclair</td><td>Main Street</td><td>1920</td><td>543</td></tr><tr><td>Lofting, Hugh</td><td>The Voyages of Doctor Dolit- tle</td><td>1922</td><td>1154</td></tr><tr><td>London, Jack</td><td>The Valley of the Moon</td><td>1913</td><td>1449</td></tr><tr><td>London, Jack</td><td>White Fang</td><td>1906</td><td>910</td></tr><tr><td>Mason, A. E. W.</td><td>Clementina</td><td>1901</td><td>13567</td></tr><tr><td>Montgomery, L. M.</td><td>Anne of Green Gables</td><td>1908</td><td>45</td></tr><tr><td>Orczy, Baroness</td><td>The Triumph of the Scarlet Pimpernel</td><td>1922</td><td>65695</td></tr><tr><td>Peacock, Thomas Love</td><td>Gryll Grange</td><td>1861</td><td>21514</td></tr><tr><td>Poe, Edgar Allan</td><td>The Narrative of Arthur Gor- don Pym of Nantucket</td><td>1838</td><td>51060</td></tr><tr><td>Pohl, Frederik &amp; C. M. Korn- bluth</td><td>Wolfbane</td><td>1959</td><td>51845</td></tr><tr><td>Porter, Eleanor H.</td><td>Pollyanna</td><td>1913</td><td>1450</td></tr><tr><td>Radcliffe, Ann</td><td>The Mysteries of Udolpho</td><td>1794</td><td>3268</td></tr><tr><td>Rand, Ayn</td><td>Anthem</td><td>1938</td><td>1250</td></tr><tr><td>Rice, Alice Hegan</td><td>Sandy</td><td>1905</td><td>14079</td></tr><tr><td>Sabatini, Rafael</td><td>Captain Blood</td><td>1922</td><td>1965</td></tr><tr><td>Salten, Felix</td><td>Bambi, a Life in the Woods</td><td>1923</td><td>63849</td></tr><tr><td>Saltykov-Shchedrin, Mikhail</td><td>The Golovlyov Family</td><td>1880</td><td>44237</td></tr><tr><td>Sand, George</td><td>Indiana</td><td>1832</td><td>63445</td></tr><tr><td>Sayers, Dorothy L. &amp; Robert Eustace</td><td>The Documents in the Case</td><td>1930</td><td>77601</td></tr><tr><td>Scott, Walter</td><td>Ivanhoe: A Romance</td><td>1819</td><td>82</td></tr><tr><td>Scott, Walter</td><td>Kenilworth</td><td>1821</td><td>1606</td></tr><tr><td>Shelley, Mary Wollstonecraft</td><td>Frankenstein</td><td>1818</td><td>84</td></tr><tr><td>Sinclair, Upton</td><td>Oil!</td><td>1926-7</td><td>70379</td></tr><tr><td>Smollett, Tobias</td><td>The Expedition of Humphry Clinker</td><td>1771</td><td>2160</td></tr><tr><td>Spyri, Johanna</td><td>Heidi</td><td>1880-1</td><td>1448</td></tr><tr><td>Stevenson, Robert Louis</td><td>Catriona</td><td>1893</td><td>589</td></tr><tr><td>Stevenson, Robert Louis</td><td>Kidnapped</td><td>1886</td><td>421</td></tr><tr><td>Stevenson, Robert Louis</td><td>Treasure Island</td><td>1883</td><td>120</td></tr><tr><td>Stoker, Bram</td><td>Dracula</td><td>1897</td><td>45839</td></tr><tr><td>Stratton-Porter, Gene</td><td>Freckles</td><td>1904</td><td>111</td></tr><tr><td>Thackeray, William Make- peace</td><td>Vanity Fair</td><td>1848</td><td>599</td></tr></table>

You are a literary alignment assistant. You will be given a set of numbered   
summary sentences and the full text of ONE chapter from a novel. For each   
summary sentence, decide whether it should be matched to this chapter.   
Judge only from the text provided. Do not rely on any prior knowledge of this   
novel, its plot, or its characters; base every decision solely on the chapter   
text and summary below.   
INPUTS   
SUMMARY SENTENCES (one per line, each prefixed with its integer id, e.g.   
"1. ..."):   
[summary]   
CHAPTER TEXT:   
[text]

<table><tr><td>Author</td><td>Title</td><td>Publication Date</td><td>ID</td></tr><tr><td>Verne, Jules</td><td>Around the World in Eighty Days</td><td>1873</td><td>103</td></tr><tr><td>Verne, Jules</td><td>From the Earth to the Moon</td><td>1865</td><td>83</td></tr><tr><td>Verne, Jules</td><td>A Journey to the Centre of the Earth</td><td>1864</td><td>18857</td></tr><tr><td>Verne, Jules</td><td>Twenty Thousand Leagues Under the Sea</td><td>1870</td><td>164</td></tr><tr><td>Voltaire</td><td>Candide</td><td>1759</td><td>19942</td></tr><tr><td>Voltaire</td><td>Micromégas</td><td>1752</td><td>30123</td></tr><tr><td>Warner, Gertrude Chandler</td><td>The Box-Car Children</td><td>1924</td><td>42796</td></tr><tr><td>Waugh, Evelyn</td><td>Vile Bodies</td><td>1930</td><td>77900</td></tr><tr><td>Webb, Frank J.</td><td>The Garies and Their Friends</td><td>1857</td><td>11214</td></tr><tr><td>Wells, H. G.</td><td>Bealby: A Holiday</td><td>1915</td><td>59769</td></tr><tr><td>Wells, H. G.</td><td>The Invisible Man</td><td>1897</td><td>5230</td></tr><tr><td>Wells, H. G.</td><td>The Time Machine</td><td>1895</td><td>35</td></tr><tr><td>Wharton, Edith</td><td>The House of Mirth</td><td>1905</td><td>284</td></tr><tr><td>Wilkins, Mary E.</td><td>The Jamesons</td><td>1899</td><td>17792</td></tr><tr><td>Wodehouse, P. G.</td><td>Mike</td><td>1909</td><td>7423</td></tr><tr><td>Zamiatin, Evgenii Ivanonich</td><td>We</td><td>1924</td><td>61963</td></tr><tr><td>Zola, Émile</td><td>L&#x27;Assommoir</td><td>1877</td><td>8600</td></tr><tr><td>Zola, Émile</td><td>Germinal</td><td>1885</td><td>56528</td></tr></table>

## B Alignment Prompts

## B.1 Prompt 1

ALREADY-MATCHED IDS (a JSON array of sentence ids that were matched to EARLIER   
chapters of this same novel; may be empty):   
[old\_ids]   
WHAT COUNTS AS A MATCH   
A sentence matches this chapter if at least one thing it describes is realized   
here:   
(a) An event it describes actually occurs in this chapter - it is dramatized in   
the chapter's narrative (including inside a flashback or scene that is   
shown, not merely referred to); OR   
(b) For sentences that describe a state, relationship, or characterization   
rather than an event (e.g. "the village is poor and isolated," "Mara   
distrusts her brother"), this chapter is where that state is first   
established or directly shown.   
Match on identity of the event or state, NOT on wording. The summary will   
describe things more abstractly or in different words than the chapter. You are   
matching the same underlying event/state, not identical phrasing. Do not   
require a literal or word-for-word match.   
WHAT DOES NOT COUNT   
Do NOT match a sentence to this chapter if the relevant event is only:   
- recalled, summarized, or discussed by characters after the fact,   
- foreshadowed or anticipated, or   
- alluded to without being shown.   
Also do NOT match on presence or framing alone:   
- A recurring character or place merely APPEARING here is NOT a match. Only   
the specific event or state the sentence describes counts - not the   
character's presence. A character introduced earlier who simply turns up   
again in a separate, later scene is NOT a re-match unless the sentence's   
own event actually happens here.   
- A sentence that merely asserts setting, time/date, or who narrates (e.g.   
"the story is set in England in 1737," "Laura narrates her childhood") and   
dramatizes no event here matches NO chapter on setting alone.   
The test is whether the event or state the sentence describes is actually   
happening or shown in this chapter - not whether the chapter mentions it or   
merely features the same character.   
MATCHING ACROSS MULTIPLE CHAPTERS   
A summary sentence may match more than one chapter. Match this chapter if   
EITHER:   
- it bundles several distinct events that may occur in DIFFERENT, possibly   
NON-ADJACENT chapters, and at least one of them actually occurs here.   
Match EVERY chapter where any of its events occurs, even chapters far apart   
- e.g. "X happens, and later Y happens" matches BOTH X's chapter and Y's   
chapter; OR   
it describes a single continuing event (a journey, a search, an illness, an   
evolving relationship) that is ACTIVELY depicted - advancing, developing,   
or being shown - here. A continuing event matches the FULL run of   
consecutive chapters in which it is actively shown, not only the chapter   
where it begins.

Match a chapter only where one of the sentence's OWN events is actually   
occurring or shown - NOT where the same character merely reappears in a   
separate matter, and NOT where the event is only continuing in the background,   
recalled, or referred to.   
For ids listed in the ALREADY-MATCHED IDS above: an event they describe   
occurred in an earlier chapter. Match such an id again here if that SAME event   
is still actively unfolding (the next consecutive step of an ongoing event),   
or if a DIFFERENT event from the sentence occurs here. Answer NO if this   
chapter merely recalls, refers back to, foreshadows, or revisits the event in   
a separate later scene without it actively unfolding now.   
BORDERLINE CASES   
When a match is genuinely uncertain, decide by the test above: answer YES if   
the chapter shows the event or state itself, and NO only if the chapter merely   
refers to it or does not contain it. Do not default to NO when the event is   
plausibly shown here.   
EXAMPLES OF THE DISTINCTION   
Summary "Tom leaves home and travels to the city," in a chapter that shows   
Tom packing and boarding the train → YES.   
The same sentence, in a later chapter where Tom only tells a friend how he   
once left home → NO (recalled, not happening).   
Summary "Tom leaves home, and years later returns for his father's   
funeral." The departure occurred in an earlier chapter (its id is already   
matched); this chapter dramatizes the funeral return → YES (a distinct   
second event occurs here).   
OUTPUT   
For EVERY summary sentence id, output "YES" or "NO". Return ONLY a single valid   
JSON object mapping every id (in ascending order) to "YES" or "NO" - no other   
text, commentary, or reasoning.   
Example:   
{ "1": "YES", "2": "NO", "3": "YES" }

## B.2 Prompt 2

You are an intelligent literary assistant. A summary sentence has been   
tentatively matched to a chapter, but a neighbouring chapter can look similar   
(it may lead up to or follow the event). Decide which chapter(s) the event   
ACTUALLY takes place in, using \*\*only\*\* the chapter texts provided.   
SUMMARY SENTENCE (id [sentence-id]):   
[sentence]   
NEARBY SUMMARY CONTEXT (for reference only; do NOT judge these):   
[context-chapters]   
CANDIDATE CHAPTERS (adjacent):   
[chapters]

TASK:   
Among chapters {choices}, return EVERY chapter in which the event the sentence   
describes ACTUALLY happens - i.e. is dramatized or actively advancing in the   
narrative. Often this is a single chapter; but when the event genuinely unfolds   
across two ADJACENT chapters, include BOTH - do NOT force a single choice when   
the event truly continues across the chapter boundary. Exclude only a chapter   
that merely leads up to, recalls, foreshadows, or follows the event without it   
actually occurring there.   
OUTPUT FORMAT:   
Return ONLY {"chapters": [...]} - a JSON list of the chosen chapter number(s)   
from {choices}: one number, or two adjacent numbers if the event truly spans   
them.

## B.3 Prompt 3

You are an intelligent literary assistant. Re-check ONE summary-sentence-to  
chapter match using \*\*only\*\* the chapter text provided.   
SUMMARY SENTENCE (id [sentence-id]):   
[sentence]   
NEARBY SUMMARY CONTEXT (for reference only; do NOT judge these):   
[context-chapters]   
CHAPTER [chapter-index]:   
[chapter-text]   
NOTE: Summaries are usually written in chapter order, so by position this   
sentence would fall around chapter [expected-index]; this is chapter [chapter  
index], which is out of that order. That does NOT make it wrong - flashbacks,   
recurring events, and foreshadowing fulfilled later are genuine. Judge ONLY   
from the chapter text above, not from the order.   
TASK:   
Answer YES if the event the sentence describes is ACTUALLY happening -   
dramatized or actively advancing - in THIS chapter. Answer NO if the chapter   
merely mentions, recalls, foreshadows, or leads up to it, or if the event does   
not appear at all.   
The match must be the SPECIFIC occurrence the sentence describes - the same   
participants doing the same thing. Answer NO if this chapter shows only a   
similar or parallel event involving DIFFERENT characters, or shows the same   
characters merely present or continuing/pursuing an ongoing situation whose   
defining event (e.g. its initiation) happens elsewhere.   
OUTPUT FORMAT:   
Return ONLY {"match": "YES"} or {"match": "NO"}.

## C Bipartite Summary Sentence to Chapter Graphs

![](images/30b64906bb58faa01ed3b4395550b00a2f5eb46f49a9267e0bfb3834ba115340.jpg)  
Figure 15

![](images/b10090ae52a5a7b906d90e36b3a4f60c53f40ee2d018a9aff126c849a7cc8eeb.jpg)  
Figure 16

![](images/45d7a57dfdb0d0573d83c92ca954e9949ea1ee1cdd2d6c245830091e6dde24fe.jpg)  
Figure 17

![](images/27e63cf1e3af937caeb43de4b2f5d89d31f56ed30d614ace1f37286dee72be5f.jpg)  
Figure 18

![](images/d3eb2e917c013f80f93c0e8be2819e6460fe19cb694e27b3f8ef2bed27f3ac62.jpg)  
Figure 19

![](images/601060f63ba73306d8df401e0c337d73ff38630cd75ac691abbdaf4d3216d5c3.jpg)  
Figure 20

![](images/a0a95bc3ced114b69cd44fdf70c801591ce6c16dc93fed82ee53f34cdb833612.jpg)  
Figure 21

![](images/ace1e857f027733f85f82748e407b74fe02c50dc51c60c369d4f77d7c4b791fd.jpg)  
Figure 22

![](images/d015565baf3a538034687aaec596fb28925069bd5e85e2ad0336ff65d69e4cd5.jpg)  
Figure 23

![](images/d6daeff88f27b43f3154fe9a310bfbfaae06dd73c01f7da29ebeedff89b75a14.jpg)  
Figure 24

![](images/97c3d06a6969ad69bba9013174e854f2f2ea07381405f8bcc5829307bd1d8d5e.jpg)  
Figure 25

![](images/8186faa414c49646a655118c6bd7cfb56f5be8b6570b4c46d86e56c6d42346ed.jpg)  
Figure 26

![](images/a2a13cc63ca4d63ed016d958c2b08810dae00dadcbb5b780481dc598278dd581.jpg)  
Figure 27

![](images/ce14b9b6d9daac0371020639ef9a684d40d384282e8affa11c57399c1a1c0e3c.jpg)  
Figure 28

![](images/bbdbc24be9dc03828ad9ff561cf2f135aafa9e3d08433a997526796746d1e76d.jpg)  
Figure 29

![](images/62c587bb2b7da152d7b1332d9ebecd77e7c06f72a067e1e210b2c9baeb801e45.jpg)  
Figure 30

![](images/4413268ab340d2176a417718d753983bbb1e8893556615f9570bc8c67ef7d255.jpg)  
Figure 31

![](images/9ae0442cc59751d3acc7efe80b7238ebea784b91a42415797c0cc14bec54c392.jpg)  
Figure 32

![](images/7ace5982ce5eba64d1fb25911decc05e74252d90074ec33a8160a96d671a3ebe.jpg)  
Figure 33

![](images/39560ef597150d3294b4569916a10cd994fd153b8d1c2aef2463324f7397fb4e.jpg)  
Figure 34

![](images/745f3df15865a46b7da214cd8c26ae71bac3f38cc9aae3f85e5906a157578c26.jpg)  
Figure 35

![](images/3cb989a8e2244f0128ce121086aa28e469a6f7f830c0697f2ef0b7a119d311e1.jpg)  
Figure 36

![](images/ff186f2b129eee0717d6e3f0489e2542c46a9bbf05e84696da1e768df0ebb515.jpg)  
Figure 37

![](images/31319e9d1d96d8716b02b56edca21c42fb41501f5f2d306f503b27da930886a4.jpg)  
Figure 38

![](images/f5227d72fbbc79eaeb2d6882f0c788b100bffc6ea9e1e41d416423ebe5925dda.jpg)  
Figure 39

![](images/b541426eb37648682dce5c9f3556b283ed7c96ca0058fb2dc38f0773341986c3.jpg)  
Figure 40

![](images/065b60a225b0076363646dfe1bc07838917938607996cc823bf3cf03f4d520ea.jpg)  
Figure 41

![](images/8d2c9203bd49d59be8a1cdcbcd81f9bac41c560bba28f2ccdad57a0f136b6ae5.jpg)  
Figure 42

![](images/2b0f4f536bc091c27001761081ff7e5b8b50faf2a5a3aed80e8218341b3f3e88.jpg)  
Figure 43

![](images/0cbe8aaef20e0cf60b660c290cc85b42d0cc2b4991df1384274e2fcaf0cdb477.jpg)  
Figure 44

![](images/cc3ec9f906ae1128bff105bc1eec3a5fdc1957bef8f4161e8e5cebb43d84710b.jpg)  
Figure 45

![](images/7ad2c87c3755b7bcce65472db2c598d737e82b1c0f22cc2800b7a4033daeedd3.jpg)  
Figure 46

![](images/9c54fbebfb66581d9b11760a668294decdd0efd2fdc77530390178af4073bbc1.jpg)  
Figure 47

![](images/6d2bc788c72673c50ff83f4fa79370e30fc8d3c872e349f027da328635d21912.jpg)  
Figure 48

![](images/de4016ef27bf1de9f8bedba06ebecb4ceea1006646d107277b75d0abaf3a29b1.jpg)  
Figure 49

![](images/beecb109db93c57867c3adc8d0eb7d2a0769ffbfc1c9cce6f9605e068f4e9bb1.jpg)  
Figure 50

![](images/77738570258159269965d643000da5602cb949918b0164c72f70ac3974c9bee5.jpg)  
Figure 51

![](images/f1f2096566b6cd71351c14b58f975feeb20a040a2cdbd5e567934837ff7c974c.jpg)

![](images/4215ea1085d12e88bc51228172296689e57fa63652f14da8ee0599f18e161105.jpg)  
Figure 53

Figure 52  
![](images/f966192c014b7d99cebd141417a8de24daae13be89f07fb22ae0d7b3c5efea6a.jpg)  
Figure 54

![](images/32368d66821e0c1ffcd32a57248bfb0331a3ea224435f86d575ac660d51bf773.jpg)  
Figure 55

![](images/3c83fa13393431e4bfbdb5f0b9096a70570fa660799493e5faf8af0af1503780.jpg)  
Figure 56

![](images/5035de086fa5bb1571f3ab3dc8ddf3f2b005a09098aacdf956ca89485e9d6f5d.jpg)  
Figure 57

![](images/7bf42d1e98b53ef51e5766cfedd9cc55fa0e97d8b0f146910a0b8053a6233cb9.jpg)  
Figure 58

![](images/8c10af35c968c0bd9d819543fbaddee648db795a23f2287fd9b24b7caac7be2e.jpg)  
Figure 59

![](images/802789fa2c826252d12d85632f3481627cd8648c89ca0e846388d89ad45d16b8.jpg)  
Figure 60

![](images/0968f3fe28e6f502f95f7e02c41508d9e9e2f2fd48e135e68e9e2eb19920de5a.jpg)  
Figure 61

![](images/a9b86f0eabfb5cd690a8a737ed98c731e0b6d8437eefbc9770697df3a8971747.jpg)

![](images/481789b70a3121842f81d354bb8420d3b197aeba96ff832fa744a26798c27d83.jpg)  
Figure 63

Figure 62  
![](images/2ec181b468085f1b4ee5351936ba09ca6ae9539257d0c6a617159090763ba885.jpg)  
Figure 64

![](images/2da815bc49de5019f19dcab70afd2feefdda7c84b75cc665871cfa7966776068.jpg)  
Figure 65

![](images/b0c42f14cb78f38758f27113101abc5804a4be945d4a85c804f4c13d3f1fe01e.jpg)  
Figure 66

![](images/53dd45ffcf40a0799790d8fd9fff36080407d3c72f37b4c7a911c3af237b4023.jpg)  
Figure 67

![](images/05c4ed921ca1df75b1d3d61a4ca5d7c2061c5950a4918fc53c56dae267435e72.jpg)  
Figure 68

![](images/67ed7a7bbf5f8a066a9e4c9a3474572d9ef5c6af189169f58c893ae3be7e1fd4.jpg)  
Figure 69

![](images/de65a32b732122e10832c98332cb091064051eabad86f149747160384f343433.jpg)  
Figure 70

![](images/857dfc39a372bc721eae010064fb9e2602f205f81b511ec3c06a14a7573b194b.jpg)  
Figure 71

![](images/3be8ea25e154203039915b62462c60f5f9f3a3ba53e8b16b0145aa90a8cfb0ca.jpg)

![](images/424e358fa7f330175b7f6ee34b48a3acbd9ffceecb805825ef0e9b27d687e1d7.jpg)  
Figure 73

Figure 72  
![](images/3e3c1ee89f70fad4080d902de468d1f6dd1038f4b048fee62411854cd4118c25.jpg)  
Figure 74

![](images/4da3c2723b6d0014b3f339a547f5efdb6f0065301901e604ab1adad3ac6bec97.jpg)  
Figure 75

![](images/13187b671d5ae8728c87a1cfefa0910d6cb43bbbd597bce5c11f9be657fceb76.jpg)  
Figure 76

![](images/9f69b599af7823303fc9a3529d23db546e039a2d14eeab91b8e3ed232700414b.jpg)  
Figure 77

![](images/1531d2b07dda5a87aaf22ececf42a3e0e599db06ef1b732a7c61f4a0b32073e8.jpg)  
Figure 78

![](images/03c5ab8e8687b44275dd03730b7f35293eef649a2dbabd735cc7ce44110bc88a.jpg)  
Figure 79

![](images/2fe503de96e5a34ce244590af438ed9a077fa4b0be5066d58f93e2dae346d71d.jpg)  
Figure 80

![](images/6c4e1b9a6a4d30808ff81466bc3fee144563a920df30d927415f2c8a37290e04.jpg)  
Figure 81

![](images/9ab2d2d4dfcababd97dd5b45030175e6fd913d159b775830863e4faf1c9894ae.jpg)  
Figure 82

![](images/1441cf5bc99dfddd26dfced5a55e2d13d5430b708794712b67bc55630bf9985a.jpg)  
Figure 83

![](images/b0e4f60459c0b271e0721e83fe29542ef3891c027d0a0d8ed53a9e6cedd0a7ac.jpg)  
Figure 84

![](images/46a2d8b8e453b8ca56b965fe647524129a2f1e3553af2eb4e6d2aaec9229fc72.jpg)  
Figure 85

![](images/02a55671acfd723caa093ea7ca2f93f1341f5e1f69e5a48a799a0c63e3a1279e.jpg)  
Figure 86

![](images/6bbe61b6947a0dc245f99bee60ec56c2a4bd0dd0d3742632081c7ea459cdde97.jpg)  
Figure 87

![](images/9a9e070dcd50488f1b1e8bcb1d932bdf438aa6569638e8a96a5f940b62fca653.jpg)  
Figure 88

![](images/8084200c8b6db83d1566a0c3811d375c0a9d17678d8b5052edf44cfa447fff92.jpg)  
Figure 89

![](images/622f57d4817e2cfe51dc17ea2fe767118a0bf72883efe7fc7d727e7acb9f30b4.jpg)  
Figure 90

![](images/4ed670f72d012581431d6b475a1579ae8a997b8949bb7f0c6387f244515acf14.jpg)  
Figure 91

![](images/ae8863aea8f61474a6514c76b9ff8d101bafdb0ab44ece9e8561a2416747db55.jpg)

![](images/369854f35e74c954d8834e7e48e7199190c9d3939d29433624d54369cbd080b7.jpg)  
Figure 93

Figure 92  
![](images/f01154350200410b6d3e2d78528e5759eb9839c3cd08c4b4cb7595f9ea15dfbe.jpg)  
Figure 94

![](images/555dd1f8843c51b7a8811c4e4821f8d8b9ebd7e65bdcbd7c401375a7490637c1.jpg)  
Figure 95

![](images/c60862b1e350f49fc889dd235c5308173e2ce12284f67f9973ec1aff8a78be2b.jpg)  
Figure 96

![](images/47f2eb756b3d0e5d583d9c580ed6d6f653d16eb0d5293d40287867a17ec8143c.jpg)  
Figure 97

![](images/34f69acb724ac3f4b5125fde2ef2ba407e0d9657e97bef79dad19aed10c7096c.jpg)  
Figure 98

![](images/eb346fd4bd4fe11406c1a27cf40cc72c2b964ff87870b3efe6a31f0e3d7fe597.jpg)  
Figure 99

![](images/d1f5fd45b395a6c7e937589170ad8fb36572c1fe07ff7b915657efe625818e7a.jpg)  
Figure 100

![](images/c7a0c2752587defdadd3811e0714db5d3f7bb2c2fb45df5aee83c16c7ec13490.jpg)  
Figure 101

![](images/c32730c7c3bd83b0f27a86886849ef197b3c4cd16b65a2c5d304bd9b67a3c867.jpg)  
Figure 102

![](images/a3e769d840a0711daff8d5bfb6a691da41274b8fe4df99228c2dc2fe5fc6c4dc.jpg)  
Figure 103

![](images/09d157d9c4c67a4e982d49b65ce025121cc5a471532e8efa29c48a5dc49e8ba4.jpg)  
Figure 104

![](images/6b38b32d9e199a2cefdd7975401a6aaab17f1e780a7af34bcb02e7c760eb7095.jpg)  
Figure 105

![](images/755f9834a2f18de91c7f4a3eb62eb800872c2b88524c2b543178cb84d96701e7.jpg)  
Figure 106

![](images/48ded7610f27b3581c5bd0d5112c915a32488812c2e1c959d1ee2f4f6be577d4.jpg)  
Figure 107

![](images/ed2a9ca0b3c2cf120ce6dd8756bebf0ffe4c222acece59d0aedffa3a05a7d6fb.jpg)  
Figure 108

![](images/d2083ac79a88f84a3d35bc2fa3b2d7a3fd5d66e4cf5bbd589afe75ae445eedc5.jpg)  
Figure 109

![](images/d5676c0b1d9893e00a1a3a7fb7829620e516ae8cfb7153d013d7ec2a9efcaaf6.jpg)  
Figure 110

![](images/5219170be9a8927e59dfc23c4be7c787fe20d0995b52a80f2fd371471bb4926e.jpg)  
Figure 111

![](images/e692c300c75f01f7ea24ae685baede5b8cd6ebd1b1655329438f91e198d6c659.jpg)  
Figure 112

![](images/a4b11b2a7fb94634f27bd4a61369f629d7b5cf88efd8494312c765b6720b79f6.jpg)  
Figure 113

![](images/b268ee6116f38e1a29d2d17a76731de9904a49c97df99cbe0640db4da8d91ce1.jpg)  
Figure 114

![](images/d92e0575a711e790f3c5c680bd8c29c849152c3518017b12900c1af09d7d67b4.jpg)  
Figure 115

![](images/78a5dc18d75982a92791e04ad269d2c9cf980f9ae0e03c5f0bef3999d3ae2176.jpg)  
Figure 116

![](images/214a50dbc547b15501d35ee0a53ec7d2d73de092cdbbb3c88cc20ff3253933c2.jpg)  
Figure 117

![](images/22f790ffaa506167cacde01c8c7be3e9849ae2298be65e73add2a9d0ec0648c5.jpg)  
Figure 118

![](images/2d10c9cfa618db8d7b231f49ea9ca3c54f204c45004f6df89bdbc85e9912b52c.jpg)  
Figure 119

![](images/4bad4cafc69fce5b80d248e0b994672b9e387fa3bd28c16ad3807a018f16be82.jpg)  
Figure 120

![](images/8efab7eb012239d1b8f4fdb765f85525220223016e75e4adc1dc9e4cae09d2b6.jpg)  
Figure 121

![](images/72ca4e5c3473387197a239c0e6634b2fbb48a589734f4c68fefa37342e9015bc.jpg)  
Figure 122

![](images/b51a48184de26f5d62c71876396eac25c8fe419d496b8e50149cfe0352b7ed36.jpg)  
Figure 123

![](images/1ceab0b992791c3d3342b7f065d24553b890c90d0b23baa1859088246b6e8f4d.jpg)  
Figure 124

![](images/11b3ad84122e8d627527299195d97c49ab432ba4ee6795e0a608e0f24e436aa1.jpg)  
Figure 125

![](images/9a025d4c3a1f1e061e5ff6238b6c2f8619f45e28243e697ed931e7834bb0fb1c.jpg)  
Figure 126

![](images/634dabfecfc66278d278ab1e5055a5f8036b4e8dbb08cafa35771cbf702f7b28.jpg)  
Figure 127

![](images/349f4e6906ca5de2b0d9b9702f57d04654fd6c8c1d2b2b270b08e39583660645.jpg)  
Figure 128

![](images/5454c2159b89d8e2f0939bb2db42e2d7db880957f737318b799cb231e50b235e.jpg)  
Figure 129

![](images/d57fa53285f5f97398cd402104d3b423c80567cfcd3959e030ee0a13bc638506.jpg)  
Figure 130

![](images/77363ce0f514af67a645f90e61b08b0b1b7ce94f3b3b96c100e97f61c001026d.jpg)  
Figure 131

![](images/dc2e7cb5483df8cf163aff58f11e11084d8691e4e5aeedde92fb4f24298223bf.jpg)  
Figure 132

![](images/b21eedc99a1bde5d88e0416859f200cd6774fcd24b9822228cfc9e48f436db9d.jpg)  
Figure 133

![](images/d5f109ac9c4b117023277f6ce8bb5d958c427bef4e040db1253f562b30196f3e.jpg)  
Figure 134

![](images/afde6f795fba289778d22e255bde3780e646725508d474eaa59a1140e846e998.jpg)  
Figure 135

![](images/a96a98eca30ea801d1ed22a5febf38087407272841891687c69f03056199b037.jpg)  
Figure 136

![](images/d9061a752205c81eceb8b7a71de05f0b6285738aae3b5fb53cf456e28dc6dbc6.jpg)  
Figure 137

![](images/0d8476c35d083db56550fc2277a05918f8a42223566af611d0a189dfcf7af287.jpg)  
Figure 138

![](images/19966dc19db3bc4633348e2f681c060c85ea90b1f930fd405a36bdc75e240286.jpg)  
Figure 139

![](images/29025d2e5d789076b5e7b92382f9679b20e99b71cca137e5c2cf523803325f34.jpg)  
Figure 140

![](images/f1218c60aaf2796716519944b2ceef8ad020772f50311c9e551f142517f6e095.jpg)  
Figure 141

![](images/c5827a2f047deb7af5a13002c950cb5e2e6797f8a6adbc842b591425b84be5b2.jpg)  
Figure 142

![](images/c6c6cd011a837e4b5ad8586f1b4a11ac9db175af149c49ece9debc8d94274b6f.jpg)  
Figure 143

![](images/46ccced844e7808107b36f5abd933de23b87ebb37efaeb315f10122f92a3b7b6.jpg)  
Figure 144

![](images/bcd58529435d69a270fee8de673d9436d953f19cf36427586ca41a92f0598159.jpg)  
Figure 145

![](images/03019a67ab88256d5b6321c0a9b3492940e7e30d2fb421502f6a0b3c44158977.jpg)  
Figure 146

![](images/6955a455ee0e3a5c2f4ba29321633de57eef448610208716ed2cba77107953a3.jpg)  
Figure 147

![](images/b8d1735cec00b96ccc30027743a98fb647c429312eaafad64c6e847884f60e27.jpg)  
Figure 148

![](images/525ab9b14d78fbbb370e13c8f6002a92007456791caff04473a2e90d100e702f.jpg)  
Figure 149

![](images/7affa57830f60de18afca3ae5eca499aabd80ba7fd78a0f9d3c6dfafc1c82014.jpg)  
Figure 150

![](images/9961d4be0fb9684b203b73ac1d2aedf06847b43921a5c2d56f05dc32261dc622.jpg)  
Figure 151

![](images/c5a32f474f325f6e45edc997aca827a1a7f625d30721cf0f674eb28dc4fb88c1.jpg)  
Figure 152

![](images/a8c2d5f4867679f41e3a2e7ee8a379ed671332aa2c7cfcb4bcf1a63b565131db.jpg)  
Figure 153

![](images/4b4853d99cc09e4b4d689942420e85291c210da3eb1d6dbe56cf0a6d5a5f6943.jpg)  
Figure 154

![](images/f8df3f2d878b7c1dbec67a28db2341e887b1849f67ca005a46d6c770775b08d1.jpg)  
Figure 155

![](images/474b2c427d991038c9a66e93ec83ff80c300de5dd70a270a578e1b328a5d6b24.jpg)  
Figure 156

![](images/f03976393aa06a65b4ffb18e7ec23e2a8a227e086add20c1b1c57a47b991649e.jpg)  
Figure 157

![](images/5669b8bdde08911535d802e81d95d460dd4c29bb35dd6e1aa6276a40b8b0f0cb.jpg)  
Figure 158

![](images/66a57061fc7e1cb00b7f8feecffc55d18cd7ff6b4b2f11c25b068dd1ea559f5b.jpg)  
Figure 159

![](images/f50e2af7e23d7e8a1f3410eb62b11886faa608091d35310ead38bb280066c45e.jpg)  
Figure 160

![](images/d1b28fe68fc5aa9adeff176e7d0c26ddea0f5ff7a3019f18a5f9840cc9243182.jpg)  
Figure 161

![](images/75d3df36298a91ecc356d80630d12c23e8164573c3542be845d6bca01e750b51.jpg)  
Figure 162

![](images/8161a5db7d770536a4f171825932f2bfdba89d3faa7dca5e3b4135be18bf307d.jpg)  
Figure 163

![](images/49c1139b63ad62623d6487866bfa59f62d24789dfb69ae86657a3f0bc5f5dab9.jpg)  
Figure 164

## D Summary Texts

This section includes the Wikipedia summaries referenced in the body of the paper in the order of appearance. Summary sentences that were not matched with any chapter of the text after alignment appear in red.

## D.1 Ivanhoe: A Romance by Walter Scott

1. Protagonist Wilfred of Ivanhoe is disinherited by his father Cedric of Rotherwood for supporting the Norman King Richard and for falling in love with the Lady Rowena, a ward of Cedric and descendant of the Saxon Kings of England.

2. Cedric planned to have Rowena marry the powerful Lord Athelstane, a pretender to the Crown of England by his descent from the last Saxon King, Harold Godwinson.

3. Ivanhoe accompanies King Richard on the Third Crusade, where he is said to have played a notable role in the Siege of Acre.

4. The book opens with a scene of Norman knights and prelates seeking the hospitality of Cedric.

5. They are guided there by a pilgrim, known at that time as a palmer.

6. That same night, Isaac of York, a Jewish moneylender, seeks refuge at Rotherwood on his way to the tournament at Ashby.

7. Following the night's meal, the palmer observes one of the Normans, the Templar Brian de Bois-Guilbert, issue orders to his Saracen soldiers to capture Isaac.

8. The palmer then assists in Isaac's escape from Rotherwood, with the additional aid of the swineherd Gurth.

9. Isaac of York offers to repay his debt to the palmer with a suit of armour and a war horse to participate in the tournament at Ashby-de-la-Zouch Castle, on his inference that the palmer was secretly a knight.

10. The palmer is taken by surprise, but accepts the offer.

11. The tournament is presided over by Prince John.

12. Also in attendance are Cedric, Athelstane, Lady Rowena, Isaac of York, his daughter Rebecca, Robin of Locksley and his men, Prince John's advisor Waldemar Fitzurse, and numerous Norman knights.

13. On the first day of the tournament, in a bout of individual jousting, a mysterious knight, identifying himself only as "Desdichado" (described in the book as Spanish, taken by the Saxons to mean "Disinherited"), defeats Bois-Guilbert.

14. The masked knight declines to reveal himself despite Prince John's request, but is nevertheless declared the champion of the day and is permitted to choose the Queen of the Tournament.

15. He bestows this honour upon Lady Rowena.

16. On the second day, at a melee, Desdichado is the leader of one party, opposed by his former adversaries.

17. Desdichado's side is soon hard-pressed and he himself beset by multiple foes until rescued by a knight nicknamed Le Noir Faineant ('the Black Sluggard'), who thereafter departs in secret.

18. When forced to unmask himself to receive his coronet (the sign of championship), Desdichado is identified as Wilfred of Ivanhoe, returned from the Crusades.

19. This causes much consternation to Prince John and his court who now fear the imminent return of King Richard.

20. Ivanhoe is severely wounded in the competition yet his father does not move quickly to tend to him.

21. Instead, Rebecca, a skilled physician, tends to him while they are lodged near the tournament and then convinces her father to take Ivanhoe with them

to their home in York when he is fit for that trip.

22. The conclusion of the tournament includes feats of archery by Locksley, such as splitting a willow reed with his arrow.

23. Prince John's dinner for the local Saxons ends in insults.

24. In the forests between Ashby and York, Isaac, Rebecca and the wounded Ivanhoe are abandoned by their guards, who fear bandits and take all of Isaac's horses.

25. Cedric, Athelstane and the Lady Rowena meet them and agree to travel together.

26. The party is captured by de Bracy and his companions and taken to Torquilstone, the castle of Front-de-BIuf.

27. The swineherd Gurth and Wamba the jester manage to escape, and then encounter Locksley, who plans a rescue.

28. The Black Knight, having taken refuge for the night in the hut of local friar, the Holy Clerk of Copmanhurst, volunteers his assistance on learning about the captives from Robin of Locksley.

29. They then besiege the Castle of Torquilstone with Robin's own men, including the friar and assorted Saxon yeomen.

30. Inside Torquilstone, de Bracy expresses his love for the Lady Rowena but is refused.

31. Brian de Bois-Guilbert tries to rape Rebecca and is thwarted.

32. He then tries to seduce her and is rebuffed.

33. Front-de-Bïuf tries to wring a hefty ransom from Isaac of York, but Isaac refuses to pay unless his daughter is freed.

34. When the besiegers deliver a note to yield up the captives, their Norman captors demand a priest to administer the Final Sacrament to Cedric; whereupon Cedric's jester Wamba slips in disguised as a priest, and takes the place of Cedric, who escapes and brings important information to the besiegers on the strength of the garrison and its layout.

35. On his way out, Cedric meets the Saxon crone Ulrica, who vows revenge on Front-de-Bïuf and advises Cedric to tell the besiegers.

36. The besiegers storm the castle.

37. The castle is set aflame during the assault by Ulrica, the daughter of the original lord of the castle, Lord Torquilstone, as revenge for her father's death.

38. Front-de-Bïuf is killed in the fire while de Bracy surrenders to the Black Knight, who identifies himself as King Richard and releases de Bracy.

39. Bois-Guilbert escapes with Rebecca while Isaac is captured by the Clerk of Copmanhurst.

40. The Lady Rowena is saved by Cedric, while the still-wounded Ivanhoe is rescued from the burning castle by King Richard.

41. In the fighting, Athelstane is wounded and presumed dead while attempting to rescue Rebecca, whom he mistakes for Rowena.

42. Following the battle, Locksley plays host to King Richard.

43. Word is conveyed by de Bracy to Prince John of the King's return and the fall of Torquilstone.

44. In the meantime, Bois-Guilbert rushes with his captive to the nearest Templar Preceptory, where Lucas de Beaumanoir, the Grand Master of the Templars, takes umbrage at Bois-Guilbert's infatuation and subjects Rebecca to a trial for witchcraft.

45. At Bois-Guilbert's secret request, she claims the right to trial by combat; and Bois-Guilbert, who had hoped to fight as Rebecca's champion, is devastated when the Grand Master orders him to fight on behalf of the Templestowe.

46. Rebecca then writes to her father to procure a champion for her.

47. Cedric organizes Athelstane's funeral at Coningsburgh, in the midst of which the Black Knight arrives with Ivanhoe.

48. Cedric, who had not been present at Locksley's carousal, is ill-disposed towards the knight upon learning his true identity, but Richard calms Cedric and reconciles him with his son.

49. During this conversation, Athelstane emerges - not dead, but laid in his coffin alive by monks desirous of the funeral money.

50. Over Cedric's renewed protests, Athelstane pledges his homage to the Norman King Richard and urges Cedric to allow Rowena to marry Ivanhoe, to which Cedric finally agrees.

51. Soon after this reconciliation, Ivanhoe receives word from Isaac beseeching him to fight on Rebecca's behalf.

52. Ivanhoe, riding day and night, arrives in time for the trial by combat; however, both horse and man are exhausted, with little chance of victory.

53. Bois-Guilbert refuses to fight but Ivanhoe accuses him of breaking his word and the Templar reacts fiercely.

54. His face becomes flushed and he is ready for combat.

55. The two knights make one charge at each other with lances, Bois-Guilbert appearing to have the advantage.

56. Ivanhoe and his horse go down, but Bois-Guilbert also falls though barely touched.

57. Ivanhoe quickly gets up to finish the fight with his sword, but Bois-Guilbert does not rise and dies a victim of his own contending passions.

58. Ivanhoe and Rowena marry and live a long and happy life together.

59. Fearing further persecution, Rebecca and her father plan to quit England for Granada.

60. Before leaving, Rebecca comes to Rowena shortly after the wedding to bid her a solemn farewell.

61. Ivanhoe's military service ends with the death of King Richard five years later.

## D.2 Oil! by Upton Sinclair

1. James Arnold "Dad" Ross and his son, James Jr. ("Bunny") are introduced as they drive through southern California to meet with the Watkins family, who are leasing out some oil property they own.

2. They find out that the family is deadlocked about how the properties run and proceeds should be divided.

3. While Dad and Bunny go quail hunting on the Watkins' goat ranch, they find oil.

4. At Bunny's urging, Dad tries to prevent the elder Watkins from beating his daughter Ruth, trying to convince them that he has received a "third revelation" which prohibits parents from beating their children.

5. The plan backfires when Eli, Ruth's brother, interjects himself into the discussion and claims that he has received the revelation.

6. As drilling begins at the Watkins ranch, Bunny begins to realize his father's business methods are not entirely ethical.

7. After a worker is killed in an accident and an oil well is destroyed in a blowout, Dad's workforce goes on strike.

8. Bunny is torn between loyalty to Dad and his friendship to Ruth and her rebellious brother Paul, who support the workers.

9. Paul is drafted into World War I and, when the conflict is over, remains in Siberia to fight the rising Bolsheviks.

10. Back home, Bunny enrolls in college, and he becomes increasingly involved with socialism through a classmate, Rachel Menzies.

11. Paul returns home and tells of his travels, explaining he has become a communist.

12. Bunny accompanies Dad to the seaside mansion of his business associate Vernon Roscoe.

13. Dad and Roscoe flee the country to avoid being subpoenaed by Congress in the Teapot Dome scandal.

14. Before Dad goes away, Bunny proposes parting ways with his father and earning his own way in the world; Dad is confused and hurt, but not unsupportive.

15. Overseas, Dad meets and marries Mrs. Olivier, a widow and spiritualist, but soon passes away from pneumonia.

16. Bunny decides to dedicate his life and inheritance to social justice while Roscoe moves to get control of the bulk of Dad's estate.

17. Bunny and his sister Bertie are swindled out of most of their inheritance by Roscoe and Mrs. Olivier.

18. Bunny marries Rachel and they dedicate themselves to establishing a socialist institution of learning; Eli, by now a successful evangelist, falsely claims that Paul underwent a deathbed conversion to Christianity.

## D.3 Carmilla by Joseph Sheridan Le Fanu

|1. The story is presented as part of the casebook of Dr. Hesselius.

2. A woman named Laura narrates, beginning with her childhood in a "picturesque

and solitary" castle amid an extensive forest in Styria, where she lives with her father, a wealthy English widower retired from service to the Austrian Empire.

3. When she was six, Laura had a vision of a beautiful visitor in her bedchamber.

4. She later claims to have been punctured in her breast, although no wound was found.

5. All the household assure Laura that it was just a dream, but they step up security as well and there is no subsequent vision or visitation.

6. Twelve years later, Laura's father tells her of a letter from his friend, General Spielsdorf.

7. The General was supposed to visit them with his niece, Bertha Rheinfeldt, but she died under mysterious circumstances.

8. The General promises to discuss the circumstances in detail when they meet later.

9. Laura, saddened by the loss of a potential friend, longs for a companion.

10. A carriage accident outside Laura's home unexpectedly brings Carmilla, a girl of Laura's age, into the family's care.

11. Both girls instantly recognise each other from the "dream" they both had when they were young.

12. Carmilla appears injured after her carriage accident, but her mysterious mother informs Laura's father that her journey is urgent and cannot be delayed.

13. She arranges to leave Carmilla with Laura and her father until she can return in three months.

14. Before leaving, she notes that Carmilla will not disclose any information whatsoever about her family, her past, or herself.

15. Carmilla and Laura grow to be close friends, but occasionally Carmilla's mood abruptly changes.

16. She sometimes makes romantic advances towards Laura.

17. Carmilla refuses to tell anything about herself, despite questioning by Laura.

18. Her secrecy is not the only mysterious thing about Carmilla; she never joins the household in its prayers, she sleeps much of the day, and she

seems to sleepwalk outside at night.

19. Meanwhile, young women and girls in the nearby towns have begun dying from an unknown malady.

20. When the funeral procession of one such victim passes by the two girls, Laura joins in the funeral hymn.

21. Carmilla bursts out in rage and scolds Laura, complaining that the hymn hurts her ears.

22. When a shipment of restored heirloom paintings arrives, Laura finds a portrait of her ancestor, Countess Mircalla Karnstein, dated 1698.

23. The portrait resembles Carmilla exactly, down to the mole on her neck.

24. Carmilla suggests that she might be descended from the Karnsteins, though the family died out centuries before.

25. During Carmilla's stay, Laura has nightmares of a large, cat-like beast entering her room.

26. The beast springs onto the bed and Laura feels something like two needles, an inch or two apart, darting deep into her breast.

27. The beast then takes the form of a female figure and disappears through the door without opening it.

28. In another nightmare, Laura hears a voice say, "Your mother warns you to beware of the assassin," and a light reveals Carmilla standing at the foot of her bed, her nightdress drenched in blood.

29. Laura's health declines, and her father has a doctor examine her.

30. He finds a small, blue spot, an inch or two below her collar, where the creature in her dream bit her, and speaks privately with her father, only asking that Laura never be unattended.

31. Her father sets out with Laura in a carriage for the ruined village of Karnstein, three miles distant.

32. They leave a message behind asking Carmilla and a governess to follow once the perpetually late-sleeping Carmilla awakes.

33. En route to Karnstein, Laura and her father encounter Spielsdorf, who tells them his story.

34. At a costume ball, Spielsdorf and Bertha had met a beautiful young woman named Millarca and her enigmatic mother.

35. Bertha was immediately taken with Millarca.

36. The mother convinced Spielsdorf that she was an old friend of his and asked that Millarca be allowed to stay with them for three weeks while she attended to a secret matter of great importance.

37. Bertha fell mysteriously ill, suffering the same symptoms as Laura.

38. After consulting with a specially ordered priestly doctor, Spielsdorf realised that Bertha was being visited by a vampire.

39. He hid with a sword and waited until a large, black creature crawled onto Bertha's bed and spread itself onto her throat.

40. He leapt from his hiding place and attacked the creature, which had then taken the form of Millarca.

41. She fled through the locked door, unharmed.

42. Bertha died before the morning dawned.

43. Upon arriving at Karnstein, Spielsdorf asks a woodman where he can find the tomb of Mircalla Karnstein.

44. The woodman says the tomb was relocated long ago by a Moravian nobleman who vanquished the vampires haunting the region.

45. While Spielsdorf and Laura are alone in the ruined chapel, Carmilla appears.

46. Spielsdorf attacks her with an axe.

47. Carmilla disarms Spielsdorf and disappears.

48. Spielsdorf explains that Carmilla is also Millarca, both anagrams for the original name of the vampire Mircalla, Countess Karnstein.

49. The party is joined by Baron Vordenburg, the descendant of the hero who rid

the area of vampires.

50. Vordenburg, an authority on vampires, has discovered that his ancestor was romantically involved with Mircalla before she died.

51. Using his forefather's notes, he locates Mircalla's hidden tomb.

52. An imperial commission exhumes the body of Mircalla.

53. Immersed in blood, it seems to be breathing faintly, its heart beating, its eyes open.

54. A stake is driven through its heart, and it gives a corresponding shriek; then, the head is struck off.

55. The body and head are burned to ashes, which are thrown into a river.

56. Afterwards, Laura's father takes his daughter on a year-long tour through Italy to regain her health and recover from the trauma, but she never fully does.

## D.4 L'Assommoir by Émile Zola

1. L'Assommoir begins with Gervaise and her two young sons being abandoned by Lantier, who takes off for parts unknown with another woman.

2. Though at first she swears off men altogether, eventually she gives in to the advances of Coupeau, a teetotal roofer, and they are married.

3. The marriage sequence is one of the most famous set-pieces of Zola's work; the account of the wedding party's impromptu and chaotic trip to the Louvre is one of the novelist's most famous passages.

4. Through a combination of happy circumstances, Gervaise is able to realise her dream and raise enough money to open her own laundry.

5. The couple's happiness appears to be complete with the birth of a daughter, Anna, nicknamed Nana (the protagonist of Zola's later novel of the same title).

6. However, later in the story, we witness the downward trajectory of Gervaise's life from this happy high point.

7. Coupeau is injured in a fall from the roof of a new hospital he is working on, and during his lengthy convalescence he takes first to idleness, then to gluttony, and eventually to drink.

8. In only a few years, Coupeau becomes a vindictive, wife-beating alcoholic, with no intention of trying to find more work.

9. Gervaise struggles to keep her home together, but her excessive pride leads her to a number of embarrassing failures and before long everything is going downhill.

10. Gervaise becomes infected by her husband's newfound laziness and, in an effort to impress others, spends her money on lavish feasts and accumulates uncontrolled debt.

11. The home is further disrupted by the return of Lantier, who is warmly welcomed by Coupeau - by this point losing interest in both Gervaise and life itself, and becoming seriously ill.

12. The ensuing chaos and financial strain is too much for Gervaise, who loses her laundry-shop and is sucked into a spiral of debt and despair.

13. Eventually, she too finds solace in drink and, like Coupeau, slides into heavy alcoholism.

14. All this prompts Nana - already suffering from the chaotic life at home and getting into trouble on a daily basis - to run away from her parents' home and become a casual prostitute.

15. Gervaise's story is told against a backdrop of a rich array of other welldrawn characters with their own vices and idiosyncrasies.

16. Notable amongst these being Goujet, a young blacksmith, who spends his life in unconsummated love for the hapless laundress.

17. Eventually, sunk by debt, hunger and alcohol, Coupeau and Gervaise both

die.

18. The latter's corpse lies for two days in her unkempt hovel before it is noticed by her disdaining neighbors.

## D.5 Flatland: A Romance of Many Dimensions by Edwin Abbott Abbott

1. The story describes a two-dimensional world inhabited by geometric figures (flatlanders); women are line segments, while men are polygons with various numbers of sides.

2. The narrator is a square, a member of the caste of gentlemen and professionals, who guides the readers through some of the implications of life in two dimensions.

3. The first half of the story goes through the practicalities of existing in a two-dimensional universe, as well as a history leading up to the year 1999 on the eve of the 3rd Millennium.

4. On New Year's Eve, the Square dreams of a visit to a one-dimensional world, "Lineland", inhabited by men, who are lines, while the women are "lustrous points".

5. These points and lines are unable to see the Square as anything other than a set of points on a line.

6. Thus, the Square attempts to convince the realm's monarch of a second dimension but cannot do so.

7. In the end, the monarch of Lineland tries to kill the Square rather than tolerate him any further.

8. Following this vision, the Square is visited by a sphere.

9. Similar to the "points" in Lineland, he is unable to see the threedimensional object as anything other than a circle (more precisely, a disk).

10. The Sphere then levitates up and down through Flatland, allowing the Square to see the circle expand and contract between a great circle and small circles.

11. The Sphere then tries further to convince the Square of the third dimension by dimensional analogies (a point becomes a line, a line becomes a square).

12. The Square is still unable to comprehend the third dimension, so the Sphere resorts to deeds: he gives information about the "insides" of the house, moves a tablet through the third dimension, and even goes inside the Square for a moment.

13. Still unable to comprehend the third dimension, the Square is taken by the Sphere to the third dimension, Spaceland.

14. This Sphere visits Flatland at the turn of each millennium to introduce a new apostle to the idea of a third dimension in the hope of eventually educating the population of Flatland.

15. From the safety of Spaceland, they can oversee the leaders of Flatland, acknowledging the Sphere's existence and prescribing the silencing.

16. After this proclamation is made, many witnesses are massacred or imprisoned (according to caste), including the Square's brother.

17. After the Square's mind is opened to new dimensions, he tries to convince the Sphere of the theoretical possibility of the existence of a fourth dimension and higher spatial dimensions.

18. The Sphere at first scoffs at the idea of higher dimensions, just as the Square had done, showing that his comprehension is not as broad as he had thought.

19. Still, the Sphere returns his student to Flatland in disgrace.

20. The Square then has a dream in which the Sphere revisits him, this time to introduce him to a zero-dimensional space, Pointland, of whom the Point (sole inhabitant, monarch, and universe in one) perceives any communication as a thought originating in his own mind (cf.

21. Solipsism).

22. The Square recognises the ignorance of the monarchs of Pointland and Lineland correspond with his own (and the Sphere's) previous ignorance of the existence of higher dimensions than their own.

23. Once returned to Flatland, the Square cannot convince anyone of Spaceland's existence, especially after official decrees are announced that anyone preaching the existence of three dimensions will be imprisoned (or executed, depending on caste).

24. For example, he tries to convince his relative of the third dimension but cannot move a square "upward," as opposed to forward or sideways.

25. Eventually, the Square himself is imprisoned for just this reason, with only occasional contact with his brother, who is imprisoned in the same facility.

26. He cannot convince his brother, even after all they have both seen.

27. Seven years after being imprisoned, "A. Square" writes out the book Flatland as a memoir, hoping to keep it as posterity for a future generation that can see beyond their two-dimensional existence.

## D.6 Adam Bede by George Eliot

1. Adam, a local carpenter much admired for his integrity and intelligence, is in love with Hetty.

2. She is attracted to Arthur, the local squire's charming grandson and heir, and falls in love with him.

3. When Adam interrupts a tryst between them, Adam and Arthur fight.

4. Arthur agrees to give up Hetty and leaves Hayslope to return to his militia.

5. After he leaves, Hetty Sorrel agrees to marry Adam but shortly before their marriage, discovers that she is pregnant.

6. In desperation, she leaves in search of Arthur but cannot find him.

7. Unwilling to return to the village on account of the shame and ostracism she would have to endure, she delivers her baby with the assistance of a friendly woman she encounters.

8. She subsequently abandons the infant in a field but not being able to bear the child's cries, she tries to retrieve the infant.

9. However, she is too late, the infant having already died of exposure.

10. Hetty is caught and tried for child murder.

11. She is found guilty and sentenced to hang.

12. Dinah enters the prison and pledges to stay with Hetty until the end.

13. Her compassion brings about Hetty's contrite confession.

14. When Arthur Donnithorne, on leave from the militia for his grandfather's funeral, hears of her impending execution, he races to the court and has the sentence commuted to penal transportation.

15. Ultimately, Adam and Dinah, who gradually become aware of their mutual love, marry and live peacefully with his family.

## D.7 Anthem by Ayn Rand

1. Equality 7-2521, a 21-year-old man writing by candlelight in a tunnel under the earth, tells the story of his life up to that point.

2. He exclusively uses plural pronouns ("we", "our", "they") to refer to himself and others.

3. He was raised like all children in his society, away from his parents in collective homes: the Home of Infants from birth until five years old, then

the Home of Students from five to fifteen.

4. He believes he has a "curse" that makes him learn quickly and ask many questions.

5. He excels at the Science of Things and dreams of becoming a Scholar, but when the Council of Vocations assigns his Life Mandate at fifteen, he is assigned to be a Street Sweeper.

6. Equality 7-2521 accepts his street sweeping assignment as penance for his Transgression of Preference in secretly desiring to be a Scholar.

7. He works with the handicapped Union 5-3992 and International 4-8818, the latter of whom is Equality 7-2521's only friend (which is another Transgression of Preference, because all are supposedly equal in their society).

8.Despite International 4-8818's protests that any exploration unauthorized by a Council is forbidden, Equality 7-2521 explores an underground tunnel near the City Theatre tent, and finds metal tracks.

9. Equality 7-2521 believes the tunnel is from the Unmentionable Times of the distant past.

10. He begins sneaking away from his community at night to use the tunnel as a laboratory for scientific experiments, using garbage he has taken from the Home of the Scholars.

11. He is using stolen paper from the Home of the Clerks to write his journal by candlelight, using candles stolen from the larder at the Home of the Street Sweepers.

12. While cleaning a road at the edge of the city, Equality 7-2521 meets Liberty 5-3000, a 17-year-old Peasant girl who works in the fields.

13. He commits another transgression by thinking constantly of her, instead of waiting to be assigned a woman at the annual Time of Mating, in which men aged twenty and over, and women of eighteen and over, are assigned to each other solely for breeding.

14. She has dark eyes and golden hair, and he names her "The Golden One".

15. When he speaks to her, he discovers that she also thinks of him.

16. He reveals his secret name for her, and Liberty 5-3000 tells Equality 7-2521 she has named him "The Unconquered".

17. Continuing his scientific work, Equality 7-2521 rediscovers electricity.

18. In the ruins of the tunnel, he finds a glass box with wires that gives off light when he passes electricity through it.

19. He decides to take his discovery to the World Council of Scholars; he thinks such a great gift to mankind will outweigh his many transgressions and lead to him being made a Scholar.

20. However, one night, his absence from the Home of the Street Sweepers is noticed.

21. He is whipped and held in the Palace of Corrective Detention.

22. The night before the World Council of Scholars is set to meet, he easily escapes; essentially walking out of the hall as there are no guards because no one has ever attempted escape before.

23. The next day, he presents his work to the World Council of Scholars.

24. Horrified that he has done unauthorized research, they assail him as a "wretch" and a "gutter cleaner" and say he must be punished.

25. They want to destroy his discovery so it will not disrupt the plans of the World Council and the Department of Candles.

26. Equality 7-2521 seizes the box, cursing the council before fleeing into the Uncharted Forest that lies outside the city.

27. In the forest, Equality 7-2521 sees himself as damned for having left his fellow men, but he enjoys his freedom.

28. No one will pursue him into this forbidden place.

29. He only misses Liberty 5-3000.

30. On his second day of living in the forest, Liberty 5-3000 appears; she

followed him into the forest and vows to stay with him forever.

31. They live together in the forest and try to express their love for one another, but they lack the words to speak of love as individuals.

32. They find a house from the Unmentionable Times in the mountains and decide to live in it.

33. While reading books from the house's library, Equality 7-2521 discovers the word "I" and tells Liberty 5-3000 about it.

34. Her first words to him are, ÒI love you.

35. ó Having rediscovered individuality, they give themselves new names from the books: Equality 7-2521 becomes "Prometheus" and Liberty 5-3000 becomes "Gaea".

36. Months later, Gaea is pregnant with Prometheus's child.

37. Prometheus wonders how men in the past could have given up their individuality; he plans a future in which they will regain it.

## D.8 The Phantom of the Opera by Gaston Leroux

1. In the 1880s, in Paris, the Palais Garnier Opera House is believed to be haunted by an entity known as the 'Phantom of the Opera', or simply the 'Opera Ghost', after stagehand Joseph Buquet is found hanged, the noose around his neck missing.

2. At a gala performance for the retirement of the opera house's managers, a young, little-known Swedish soprano, Christine Daae, is called upon to sing in place of the opera's leading soprano, Carlotta, who is ill.

3. Christine's performance is a success.

4. Among the audience is the Vicomte Raoul de Chagny, who recognizes her as his childhood playmate and recalls his love for her.

5. He attempts to visit her backstage, where he hears a man complimenting her from inside her dressing room.

6. He investigates the room once Christine leaves, only to find it empty.

7. At Perros-Guirec, Christine meets with Raoul, who confronts her about the voice he heard in her room.

8. Christine says she has been tutored by the "Angel of Music", whom her father used to tell her and Raoul about.

9. When Raoul suggests that she might be the victim of a prank, she storms off.

10. Christine visits her father's grave one night, where a mysterious figure appears and plays the violin for her.

11. Raoul attempts to confront the figure but is struck and knocked out in the process.

12. Back at the Palais Garnier, the new managers receive a letter from the Phantom demanding that they allow Christine to perform the lead role of Marguerite in Faust and that box five be left empty for his use, lest they perform in a house with a curse on it.

13. The managers assume his demands are a prank and ignore them.

14. Soon after, Carlotta ends up croaking like a toad, and a chandelier drops into the audience, killing a spectator.

15. The Phantom, having abducted Christine from her dressing room, reveals himself as a deformed man called Erik.

16. Erik intends to hold her prisoner in his lair with him for a few days.

17. Still, she causes him to change his plans when she unmasks him and, to the horror of both, beholds his skull-like face.

18. Fearing that she will leave him, he decides to hold her permanently.

19. However, when Christine requests her release after two weeks, he agrees on the condition that she wear his ring and be faithful to him.

20. On the roof of the Opera House, Christine tells Raoul about her abduction and makes Raoul promise to take her away to where Erik can never find her, even if she resists.

21. Raoul says he will act on his promise the next day.

22. Unbeknownst to Christine and Raoul, Erik is watching them and overheard their whole conversation.

23. The following night, the enraged and jealous Erik abducts Christine during a production of Faust and tries to force her to marry him.

24. Raoul is led by a mysterious Opera House regular, 'the Persian', into Erik's secret lair in the bowels of the building.

25. Still, they end up trapped in a mirrored room by Erik, who threatens that unless Christine agrees to marry him, he will kill them and everyone in the Opera House by using explosives.

26. Under duress, Christine agrees to marry Erik.

27. Erik initially tries to drown Raoul and the Persian, using the water which would have been used to douse the explosives.

28. Still, Christine begs, promising him she would not kill herself after becoming his bride.

29. Erik releases Raoul and 'the Persian' from his torture chamber.

30. When Erik is alone with Christine, he lifts his mask to kiss her on her forehead and is eventually given a kiss back.

31. Erik reveals he has never kissed anyone, including his own mother, who would run away if he ever tried to kiss her.

32. Moved, he and Christine cry together.

33. She also holds his hand and says, "Poor, unhappy Erik", which reduces him to "a dog ready to die for her".

34. He allows 'the Persian' and Raoul to escape, though not before making Christine promise that she will visit him on his death day and return the ring he gave her.

35. He also makes 'the Persian' promise that afterward, he will go to the newspaper and report his death, as he will die soon "of love."

36. Later, Christine returns to Erik's lair, and per his request, returns the ring and buries him 'somewhere he will never be found'.

37. Afterward, a local newspaper runs the note: "Erik is dead".

38. Christine and Raoul then elope together, never to return.

39. The epilogue reveals that Erik was born deformed and is the son of a construction business owner.

40. He ran away from his native Normandy to work in fairs and caravans, schooling himself in the circus arts across Europe and Asia, and eventually building trick palaces in Persia and Turkey.

41. Returning to France, he started his own construction business.

42. After being subcontracted to work on the Palais Garnier's foundations, Erik discreetly built his secret lair with hidden passages and other tricks that allowed him to spy on the managers.

## D.9 O Pioneers! by Willa Cather

1. On a windy January day in Hanover, Nebraska, Alexandra Bergson is with her five-year-old brother Emil, whose little kitten has climbed a telegraph pole and is afraid to come down.

2. Alexandra asks her neighbor and friend Carl Linstrum to retrieve the kitten.

3. Later, Alexandra finds Emil in the general store with Marie Tovesky.

4. They are playing with the kitten.

5. Marie lives in Omaha and is visiting her uncle Joe Tovesky.

6. Alexandra's father is dying, and it is his wish that she run the farm after he is gone.

7. Alexandra and her brothers Oscar and Lou later visit Ivar, known as Crazy Ivar because of his unorthodox views.

8. For instance, he sleeps in a hammock, believes in killing no living thing and goes barefoot summer and winter.

9. But he is known for healing sick animals.

10. Alexandra is concerned about their hogs as the hogs of many of their neighbors are dying.

11. Crazy Ivar advises her to keep their hogs clean rather than letting them live in filth and to give them fresh, clean water and good food.

12. This simply confirms Oscar's and Lou's opinion that Ivar deserves the name Crazy Ivar.

13. Alexandra, however, starts making plans for where she will relocate the hogs.

14. After years of crop failure, many of the Bergson's neighbors are selling out, even if it means taking a loss.

15. Then they learn the Linstrums have also decided to leave.

16. Oscar and Lou want to leave too, but neither their mother nor Alexandra will.

17. After visiting villages downwards to see how they are getting on, Alexandra talks her brothers into mortgaging the farm to buy more land, in hopes of ending up as rich landowners.

18. Sixteen years later, the farms are now prosperous.

19. Alexandra and her brothers have divided up their inheritance, and Emil has just returned from college.

20. The Linstrum farm has failed, and Marie, now married to Frank Shabata, has bought it.

21. That same day, the Bergsons are surprised by a visit from Carl Linstrum, whom they have not seen for thirteen years.[2] [Note: Carl says it has been sixteen years, but this is a textual error.

22. John Bergson died sixteen years earlier, and Carl's family left during the drought that occurred three years later.[citation needed]] Having failed at a job in Chicago, he is on his way to Alaska, but decides to stay with Alexandra for a while.

23. Carl notices the growing flirtatious relationship between Emil and Marie.

24. Lou and Oscar suspect that Carl wants to marry Alexandra, and are resentful at the idea that Carl might end up owning Alexandra's share of the farm, which they still view as belonging to them and which would otherwise be inherited by their children.

25. This causes problems between Alexandra and her brothers, and they stop speaking to each other.

26. Carl, recognizing a problem, decides to leave for Alaska.

27. At the same time, Emil announces he is leaving to travel through Mexico.

28. Alexandra is left alone.

29. Winter has settled down over the Divide again; the season in which Nature recuperates, in which she sinks to sleep between the fruitfulness of autumn and the passion of spring.

30. The birds have gone.

31. The teeming life that goes on down in the long grass is exterminated.

32. The prairie dog keeps his hole.

33. The rabbits run shivering from one frozen garden patch to another and are hard put to it to find frost-bitten cabbage stalks.

34. At night the coyotes roam the wintry waste, howling for food.

35. The variegated fields are all one color now; the pastures, the stubble, the roads, and the sky are the same leaden gray.

36. The hedgerows and trees are scarcely perceptible against the bare earth, whose slaty hue they have taken on.

37. The ground is frozen so hard that it bruises the foot to walk on the roads or in the plowed fields.

38. It is like an iron country, and the spirit is oppressed by its rigor and melancholy.

39. One could easily believe that in that dead landscape the germs of life and fruitfulness were extinct forever.

40. Alexandra spends the winter alone, except for occasional visits from Marie, whom she visits with Mrs. Lee, Lou's mother-in-law.

41. She also has an increased number of mysterious dreams she has had since girlhood.

42. These dreams are about a strong, god-like male figure who carries her over the fields.

43. Emil returns from Mexico City.

44. His best friend, Amedee, is now married with a young son.

45. At a fair at the French church, Emil and Marie kiss for the first time.

46. They later confess their illicit love, and Emil determines to leave for law school in Michigan.

47. Before he leaves, Amedee dies from a ruptured appendix, and as a result both Emil and Marie realize what they value most.

48. Before leaving for Michigan, Emil stops by Marie's farm to say one last goodbye, and they fall into a passionate embrace beneath the white mulberry tree.

49. They stay there for several hours, until Marie's husband, Frank, finds them and shoots them in a drunken rage.

50. He flees to Omaha, where he later turns himself in for the crime.

51. Ivar discovers Emil's abandoned horse, leading him to search for the boy and discover the bodies.

52. After Emil's death Alexandra is distraught, in shock, and slightly dazed.

53. She goes off in a rainstorm.

54. Ivar goes looking for her and brings her back home, where she sleeps fitfully and dreams about death.

55. She then decides to visit Frank in Lincoln where he is incarcerated.

56. While in town she walks by Emil's university campus, comes upon a polite young man who reminds her of Emil, and feels better.

57. The next day she talks to Frank in prison.

58. He is bedraggled and can barely speak properly, and she promises to do what she can to see him released; she bears no ill will toward him.

59. She then receives a telegram from Carl, telling her that he is back.

60. They decide to marry, unconcerned with the approval of her brothers.

## D.10 The Triumph of the Scarlet Pimpernel by Baroness Orczy

1. The story starts in Paris in April 1794, year II of the French Revolution.

2. Theresia Cabarrus is a beautiful but shallow Spaniard who is betrothed to Citizen Tallien the popular Representative in the Convention and one of Robespierre's inner circle.

3. She is credited with exercising a mellowing influence over Tallien, whom she met in Bordeaux but although she is engaged to be married to him, what little love she has appears to be lavished on another.

4. Bertrand Moncrif is a good-looking but impulsive young man who appears determined to martyr himself in opposition to the revolutionary government.

5. To this end, he has gathered the siblings of his long-term sweetheart, Regine de Serval, into his plan to denounce Robespierre at one of the Fraternal suppers.

6. Despite warnings from Regine he insists on carrying through his plans which inevitably go awry and the wrath of the mob is soon turned towards the small group.

7. After a timely intervention on the part of the Scarlet Pimpernel, using the guise of the coal heaver Rateau (who also appears in several short stories in The League of the Scarlet Pimpernel - The Cabaret de la Liberte, Needs Must and A Battle of Wits), the de Servals are saved from a lynching while Moncrif lies unconscious and unseen under a table.

8. In England, Moncrif and the de Servals are finally free to resume an almost normal life.

9. Theresia arrives at Dover dressed in men's clothes and claiming she has been driven out of France by her association with Bertrand, in fear of her life.

10. An obviously staged row between the Spaniard and Chauvelin outside Sir Percy's cottage fails to persuade our hero that she is up to anything but mischief, but he seems to relish the prospect of such an intelligent and wily adversary and promises not to reveal her true identity to anyone for he "is a lover of sport."

11. With her plans to seduce Percy scuppered, Theresia turns her attention to Sir Percy's wife Marguerite and uses an all too willing Bertrand to set the trap.

12. Lady Blakeney is kidnapped yet again and taken to France and imprisoned as bait for Sir Percy.

## D.11 Beauvallet by Georgette Heyer

1. The year is 1586 and 35-year-old Sir Nicholas Beauvallet (great-great-great-grandson of Simon the Coldheart)-'El Beauvallet'to the Spanish and called 'Mad Nick' by his men - is one of the most daring pirates of the Elizabethan era.

2. With the blessing of the Queen, this friend and former associate of Sir Walter Raleigh sails the seas in his ship The Venture with the intention of plundering any Spanish ships that come his way.

3. It is while engaged in one such enterprise that he takes the galleon on which the retired and ailing Governor Don Manuel de Rada y Sylva is returning home, accompanied by his daughter Dominica.

4. Having plundered their own ship, Beauvallet promises to take them with him the rest of the way to Spain.

5. Dona Dominica's haughty spirit appeals to Beauvallet and he vows to come back to claim her within the year.

6. Upon landing in England, he rides on a visit to his elder brother Gerard who, lacking heirs himself, reminds Nicholas that it is his duty to continue the family line.

7. Three months later, Sir Nicholas visits a relation in France and then rides south towards the Spanish border, accompanied by his servant Joshua.

8. There he meets the young Chevalier Claude de Guise, on a mission to deliver a secret message to the Spanish King.

9. On catching the Chevalier trying to steal his horse, Beauvallet kills him in a sword fight and assumes his identity.

10. Travelling as a Frenchman provides Beauvallet with a convenient disguise in a rigidly Catholic land where the English are abhorred as Protestant heretics.

11. Having arrived in Madrid, he learns that Dominica's father has died and that she is now under the protection of her uncle, Don Rodriguez.

12. But even as he makes contact with Dominica's new family, Beauvallet arouses the suspicion of the French ambassador, M. de Lauviniere, who makes enquiries about him.

13. Henceforward Sir Nicholas must engineer his escape with his chosen bride while avoiding the clutch of the Inquisition, or imprisonment as a spy, and the jealous intrigues of Dominica's other suitors.

## D.12 The Golovlyov Family by Mikhail Saltykov-Shchedrin

1. Arina Petrova, matriarch of the Golovlyov family, runs a large estate (4,000 serfs) in Russia.

2. She learns that her first born son, Stepan/Styopka/The Dolt has squandered the land and house she gave to him.

3. She was a practical and strict noblewoman, and she banished her drunken husband Vladmir Mihailitch to his room for several decades while she ran the estate.

4. Arina sent Stepan to college, where he was the class clown.

5. He worked in a series of government jobs, but lost them all due to laziness.

6. He returns home after losing his estate.

7. Arina's second child is Anna, who ran off and married a musician named Ulanov.

8. Anna has twin girls Anninka and Lubinka.

9. Ulanov soon abandons his family, and Anna dies of an illness 3 months later.

10. Arina hoped to be rid of her children by giving them estates.

11. She was very upset when Anna died ('throwing her two brats on to my shoulders') and when Stepan returned.

12. Her third son is Porphyry/Iudushka/Bloodsucker; he is an obsequious, scheming son.

13. Her fourth son is Pavel; he is normal and unremarkable in any way.

14. She keeps her family on a very tight financial leash, and they live at poverty level despite their wealth.

15. Stepan, having nowhere to go, sadly travels back home.

16. Arina declares that she hates him, and says "he has been nothing but a worry and a disgrace to me all his life.

17. ó She wonders who she is saving her money for.

18. Stepan is let back into the estate, but becomes depressed and runs away one winter evening.

19. He is found alive but never speaks again; he dies shortly thereafter.

20. Serfdom is abolished by the Tsar.

21. Vladimir Mihailitch dies and Arina divides her estate between her 2 remaining children.

22. Pavel dies from alcoholism having refused to make a will.

23. This means everything goes to Porphyry.

24. Arina leaves the big house and moves in with her orphaned twin granddaughters to the Pogorelka estate.

25. Porphyry gets married and has 2 sons.

26. He becomes very religious, but it is only for show.

27. While Arina is very strict with the girls, her energy for managing an estate is waning.

28. The girls demand to leave, and she lets them.

29. Arina becomes depressed living in an empty house in rural Russia.

30. She begins visiting her son for good food and conversation.

31. Porphyry's wife dies, and he takes a lover, Yevpraxeya.

32. The twins write back that they have both become successful provincial actresses and can support themselves.

33. Porphyry's first son, Volodya, kills himself.

34. Porphyry's second son, Pyotr/Petenka, arrives unexpectedly to beg for money.

35. He was an infantry treasurer that has gambled the unit's money away.

36. Pyotr explains that Volodya killed himself because Porphyry refused to support him and his new wife (Volodya informed, but didn't ask permission for the marriage).

37. Porphyry refuses to pay the debt, sending Pyotr away to await trial.

38. Arina falls ill.

39. Porphyry calls for the twins as Arina dies.

40. Prophyry's son Pyotr is banished to Siberia and he dies on the way there.

41. Anninka arrives to settle some paperwork since Arina died.

42. The twins want to continue their acting careers, and have no interest in moving back.

43. Porphyry has become a compulsive talker.

44. Anninka's visit with Porphyry is awful and she leaves as soon as the papers are signed.

45. Yevpraxeya becomes pregnant with Porphyry's child.

46. He is afraid of a scandal, so he ignores the pregnancy and denies any involvement.

47. She gives birth to a boy; Porphyry feels guilty for having a child out of wedlock, so he sends the baby to the orphanage without Yevpraxeya's knowledge.

48. Yevpraxeya, deprived of her child, decides to ruin Porphyry's life.

49. She begins to complain incessantly and refuses to listen to Porphyry's constant babbling.

50. She takes several lovers.

51. Porphyry becomes a recluse and begins to lose his mind.

## D.13 Anne of Green Gables by L. M. Montgomery

1. Anne Shirley, a young orphan from the fictional community of Bolingbroke, Nova Scotia (based upon the real community of New London, Prince Edward Island), is sent to live with Marilla and Matthew Cuthbert, unmarried siblings in their fifties and sixties, after a childhood spent in strangers'homes and orphanages.

2. Marilla and Matthew had originally sought to adopt a boy from the orphanage to help Matthew run their farm at Green Gables, which is set in the fictional town of Avonlea (based on Cavendish, Prince Edward Island).

3. Through a misunderstanding, the orphanage sends Anne instead.

4. Anne is fanciful, imaginative, eager to please, and dramatic.

5. She is also adamant her name should always be spelled with an "e" at the end.

6. However, she is defensive about her appearance, despising her red hair, freckles, and pale, thin frame, but liking her nose.

7. She is talkative, especially when it comes to describing her fantasies and dreams.

8. At first, stern Marilla says Anne must return to the orphanage, but after much observation and consideration, along with kind, quiet Matthew's encouragement, Marilla decides to let her stay.

9. Anne takes much joy in life and adapts quickly, thriving in the close-knit farming village.

10. Her imagination and talkativeness soon brighten up Green Gables.

11. The book recounts Anne's struggles and joys in settling into Green Gables (the first real home she's ever known): the country school where she quickly excels in her studies; her friendship with Diana Barry, the girl living next door (her best or "bosom friend" as Anne fondly calls her); her budding literary ambitions; and her rivalry with her classmate Gilbert Blythe, who teases her about her red hair.

12. For that, he earns her instant hatred, although he apologizes several times.

13. As time passes, however, Anne realizes she no longer hates Gilbert, but her pride and stubbornness keep her from speaking to him.

14. The book also follows Anne's adventures in Avonlea.

15. Episodes include playtime with her friends Diana, calm, placid Jane Andrews, and beautiful, boy-crazy Ruby Gillis.

16. She has run-ins with the unpleasant Pye sisters, Gertie and Josie, and frequent domestic "scrapes" such as dyeing her hair green while intending to dye it black, and accidentally getting Diana drunk by giving her what she thinks is raspberry cordial but which turns out to be currant wine.

17. At sixteen, Anne goes to Queen's Academy to earn a teaching license, along with Gilbert, Ruby, Josie, Jane, and several other students, excluding Diana, much to Anne's dismay.

18. She obtains her license in one year instead of the usual two and wins the Avery Scholarship awarded to the top student in English.

19. This scholarship would allow her to pursue a Bachelor of Arts (B.A.) degree at the fictional Redmond College (based on the real Dalhousie College) on the mainland in Nova Scotia.

20. Near the end of the book, however, tragedy strikes when Matthew dies of a heart attack after learning that all of his and Marilla's money has been lost in a bank failure.

21. Out of devotion to Marilla and Green Gables, Anne gives up the scholarship to stay at home and help Marilla, whose eyesight is failing.

22. She plans to teach at the Carmody school, the nearest school available, and return to Green Gables on weekends.

23. In an act of friendship, Gilbert Blythe gives up his teaching position at the Avonlea School in favor of Anne, to work at the White Sands School instead, knowing that Anne wants to stay close to Marilla after Matthew's death.

24. After this kind act, Anne and Gilbert's friendship is cemented, and Anne looks forward to what life will bring next.

## D.14 David Copperfield by Charles Dickens

1. The story follows the life of David Copperfield from childhood to maturity.

2. David was born in Blunderstone, Suffolk, England, six months after the death of his father.

3. David spends his early years residing in a small house called the Rookery.

4. His loving and childish mother, Clara Copperfield, and their kindly housekeeper, Clara Peggotty, bring him up here; they call him Davy.

5. When he is seven years old, his mother surprises David by marrying Edward Murdstone while the boy is visiting Peggotty's family in Yarmouth.

6. Her brother, a fisherman named Mr Peggotty, lives in a beached barge, with his adopted niece and nephew Emily and Ham, and an elderly widow, Mrs Gummidge. "

7. Little Em'ly" is somewhat spoiled by her fond foster father, and David is in love with her.

8. Here he is known as "Master Copperfield.

9." On his return, David discovers his mother has married and is immediately given good reason to dislike his stepfather, Murdstone, who believes exclusively in stern, harsh parenting, calling it "firmness".

10. David has similar feelings for Murdstone's sister Jane, who moves into the house soon afterwards.

11. Between them, they tyrannise David and his poor mother, making their lives miserable.

12. When David falls behind in his studies, Murdstone thrashes him, and David bites his hand; in consequence, he is sent away to Salem House, a boarding school, under a ruthless headmaster named Mr Creakle.

13. There he is befriended by two quite different older boys, James Steerforth and Tommy Traddles.

14. He develops an impassioned admiration for Steerforth, perceiving him as someone noble, who could do great things if he would, and one who pays attention to him.

15. David goes home for the holidays to learn that his mother has given birth to a baby boy.

16. Shortly after David returns to Salem House, his mother and her baby die, and David returns home immediately.

17. Peggotty marries the local carrier, Mr Barkis.

18. Murdstone sends David to work for a wine merchant in London - a business of which Murdstone is a joint owner.

19. After some months, David's friendly but spendthrift landlord, Wilkins Micawber, is arrested for debt and sent to the King's Bench Prison, and the rest of Mr. Micawber's family soon moves to the Prison too.

20. David visits the Micawbers regularly at the Prison, and boards nearby.

21. When Micawber's release is imminent, the Micawbers decide they will soon move to Plymouth.

22. David realises that will leave him alone in London, where no one cares about him.

23. He makes up his mind to run away to Dover to find his only known remaining relative, his eccentric and kind-hearted great-aunt Betsey Trotwood.

24. She had come to Blunderstone at his birth, only to depart in ire upon learning that he was not a girl.

25. However, she takes it upon herself to raise David, despite Murdstone's attempt to regain custody of him.

26. She encourages him to 'be as like his sister, 'Betsey Trotwood' as he can be - that is, to meet the expectations she had for the girl who was never born.

27. David's great-aunt renames him "Trotwood Copperfield" and addresses him as "Trot", one of several names others call David in the novel.

28. David's aunt sends him to a better school than the last he attended.

29. It is run by kind Dr. Strong, whose methods inculcate honour and self-reliance in his pupils.

30. During term, David lodges with the lawyer Mr Wickfield and his daughter Agnes, who becomes David's friend and confidante.

31. Wickfield's clerk, Uriah Heep, also lives at the house.

32. By devious means, Uriah Heep gradually gains a complete ascendancy over the aging and alcoholic Wickfield, to Agnes's great sorrow.

33. Heep, as he maliciously confides to David, aspires to marry Agnes.

34. Ultimately with the aid of Micawber, who has been employed by Heep as a secretary, his fraudulent behaviour is revealed. (

35. At the end of the book, David encounters him in prison, convicted of attempting to defraud the Bank of England.)

36. After completing school, David apprentices to be a proctor.

37. During this time, due to Heep's fraudulent activities, his aunt's fortune has diminished.

38. David toils to make a living.

39. He works mornings and evenings for his former teacher Dr Strong as a secretary, and also starts to learn shorthand, with the help of his old school-friend Traddles, upon completion reporting parliamentary debate for a newspaper.

40. With considerable moral support from Agnes and his own great diligence and hard work, David ultimately finds fame and fortune as an author, writing fiction.

41. David's romantic but self-serving school friend, Steerforth, also re-acquainted with David, goes on to seduce and dishonour Emily, offering to marry her off to his manservant Littimer before deserting her in Europe.

42. Her uncle Mr Peggotty manages to find her with the help of Martha, who had grown up in their part of England and then settled in London.

43. Ham, who had been engaged to marry Emily before the tragedy, dies in a fierce storm off the coast while rescuing shipwreck victims.

44. Steerforth was aboard the ship and also dies.

45. Mr Peggotty takes Emily to a new life in Australia, accompanied by Mrs Gummidge and the Micawbers, where all eventually find security and happiness.

46. Meanwhile, David has fallen in love with Dora Spenlow, and marries her.

47. Their marriage proves troublesome for David in the sense of everyday practical affairs, but he never stops loving her.

48. Dora dies early in their marriage after a miscarriage.

49. After Dora's death, Agnes encourages David to return to normal life and his profession of writing.

50. While living in Switzerland to dispel his grief over so many losses, David realises that he loves Agnes.

51. Upon returning to Britain, after a failed attempt to conceal his feelings, David finds that Agnes loves him too.

52. They quickly marry, and in this marriage he finds true happiness.

53. David and Agnes then have at least five children, including a daughter named after his great-aunt, Betsey Trotwood.

## D.15 Main Street by Sinclair Lewis

1. Carol Milford, the daughter of a judge, grew up in Mankato, Minnesota, and became an orphan in her teenage years.

2. In college, she reads a book on village improvement in a sociology class and begins to dream of redesigning villages and towns.

3. After college, she attends a library school in Chicago and is exposed to many radical ideas and lifestyles.

4. She becomes a librarian in Saint Paul, Minnesota, the state capital, but finds the work unrewarding.

5. She marries Will Kennicott, a doctor from the small town of Gopher Prairie.

6. When they marry, Will convinces her to live in his hometown.

7. Carol is filled with disdain for the town's physical ugliness and smug conservatism and immediately formulates plans to remake Gopher Prairie.

8. She speaks with its members about progressive changes, joins women's clubs, distributes literature, and holds a party to liven up Gopher Prairie's inhabitants.

9. Despite her efforts, these ventures are ineffective and she is constantly derided by the leading cliques.

10. She finds some comfort and companionship with a variety of social outsiders in the town, but these companions all fail to live up to her expectations.

11. After a political meeting of the Nonpartisan League is broken up by local authorities, Carol leaves her husband and moves to Washington, D.C. to become a clerk in a wartime government agency but she eventually returns.   
12. Nevertheless, she does not feel defeated.

## D.16 Captain Blood by Rafael Sabatini

1. The protagonist is the sharp-witted Dr. Peter Blood, a fictional Irish physician who had had a wide-ranging career as a soldier and sailor (including a commission as a captain under the Dutch admiral Michiel de Ruyter) before settling down to practice medicine in the town of Bridgwater,Somerset.

2. The story is told from the perspective of an omniscient narrator, who enables the reader to see the thoughts and views of many different characters.

3. The narrator-perhaps meant to be Sabatini himself-claims to have acquired the story from the ship's logs of Blood's longtime companion Jeremy Pitt.

4. The book opens with Blood attending to his geraniums while the town prepares to fight for James Scott, 1st Duke of Monmouth.

5. He wants no part in the Monmouth Rebellion, but while attending to some of the rebels wounded at the Battle of Sedgemoor, Peter is arrested.

6. During the Bloody Assizes, he is convicted by the infamous Judge Jeffreys of treason on the grounds that "if any person be in actual rebellion against the King, and another person-who really and actually was not in rebellion-does knowingly receive, harbour, comfort, or succour him, such a person is as much a traitor as he who indeed bore arms."

7. The sentence for treason is death by hanging, but King James II, for purely financial reasons, has the sentence for Blood and other convicted rebels commuted to transportation to penal servitude in the Caribbean.

8. Upon arrival on the island of Barbados, Blood is bought by Colonel William Bishop, initially for forced work in the Colonel's prison farms but later hired out by Bishop when Blood's skills as a physician prove superior to those of the local doctors.

9. During his period of servitude, Blood wins the pity and sympathy of Arabella, Colonel Bishop's niece.

10. When a Spanish force attacks and raids Bridgetown, Blood escapes with a number of other convicts (including former shipmaster Jeremy Pitt, the one-eyed giant Edward Wolverstone, former gentleman Nathaniel Hagthorpe and two Royal Navy veterans, former petty officer Nicholas Dyke and former master gunner Ned Ogle).

11. The escapees capture the Spaniards' ship and sail away to become some of the most successful pirates in the Caribbean, hated and feared by the Spanish but always sparing English ships.

12. Colonel Bishop, humiliated by Blood's superior abilities and daring escape, devotes himself to capturing and executing Blood.

13. After the Glorious Revolution, Blood is pardoned.

14. As a reward for saving the colony of Jamaica from a French assault, he is appointed its governor in place of Colonel Bishop, who had abandoned his post to hunt for Blood, and the novel ends with the implication that Blood will not only marry Arabella but will also generously forgive Bishop.

## D.17 The Wind in the Willows by Kenneth Grahame

1. The novel's characters are anthropomorphized animals.

2. With the arrival of spring and fine weather outside, the good-natured Mole loses patience with spring cleaning, exclaiming, "Hang spring cleaning!"

3. He leaves his underground home and comes up at the bank of the river, which he has never seen before.

4. Here he meets Water Rat, a water vole, who takes Mole for a ride in his row boat.

5. They get along well and spend many more days boating. "

6. Ratty" teaches Mole the ways of the river, and the two friends live together in Ratty's riverside home.

7. One summer day, Rat and Mole disembark near the grand Toad Hall and pay a visit to Toad.

8. Toad is wealthy, jovial, friendly and kindhearted, but sometimes arrogant and rash; he regularly becomes obsessed with fads, only to abandon them abruptly.

9. His current craze is his horse-drawn caravan.

10. When a passing automobile scares his horse and causes the caravan to overturn into a ditch, Toad's craze for caravan travel is immediately replaced by an obsession with motorcars.

11. Mole goes to the Wild Wood on a snowy winter's day, hoping to meet the elusive but wise and virtuous Badger.

12. Mole becomes lost in the woods, succumbs to fright and hides among the sheltering roots of a tree.

13. Rat finds him as snow begins to fall in earnest.

14. As they attempt to find their way home, Mole barks his shin on the bootscraper on Badger's doorstep.

15. Badger welcomes Rat and Mole into his large, cosy underground home, providing them with hot food, dry clothes and reassuring conversation.

16. Badger learns from his visitors that Toad has crashed seven cars, has been in hospital three times, and has been issued numerous fines resulting in great expense.

17. With the arrival of spring, the three of them place Toad under house arrest with themselves as guards.

18. However, Toad pretends to be sick, tricking Ratty into leaving for a doctor, and escapes.

19. Badger and Mole continue residing at Toad Hall in the hope that Toad may return.

20. Toad orders lunch at The Red Lion Inn and then sees a motorcar pull into the courtyard.

21. He takes the car, drives it recklessly, and is caught by the police.

22. He is sent to prison for 20 years, most of which is for "gross impertinence" to his captors.

23. In prison, Toad gains the sympathy of the gaoler's daughter.

24. With her help, he escapes, disguised as a washerwoman.

25. After a long series of misadventures he returns to the hole of the Water Rat.

26. Rat hauls Toad inside and informs him that Toad Hall has been taken over by weasels, stoats and ferrets from the Wild Wood, who have driven out Mole and Badger.

27. Armed to the teeth, Badger, Rat, Mole and Toad enter through the tunnel and pounce upon the unsuspecting Wild-Wooders, who are holding a celebratory party.

28. Having driven away the intruders, Toad holds a banquet to mark his return, during which he behaves both quietly and humbly.

29. He makes up for his earlier excesses by seeking out and compensating those he has wronged, and the four friends live happily ever after.

30. In addition to the main narrative, the book contains several independent short stories featuring Rat and Mole, such as an encounter with the wild god Pan while they are searching for Otter's son Portly, and Ratty's meeting with a Sea Rat.

31. These appear for the most part between the chapters chronicling Toad's adventures, and they are often omitted from adaptations of the story as well as from some abridged versions.

## D.18 Nostromo: A Tale of the Seaboard by Joseph Conrad

1. Charles Gould is a native Costaguanero of English descent who owns an important silver-mining concession near the key port of Sulaco.

2. He is tired of the political instability in Costaguana and its concomitant corruption, and uses his wealth to support Ribiera's government, which he believes will finally bring stability to the country after years of misrule and tyranny by self-serving dictators.

3. Instead, Gould's refurbished silver mine and the wealth it has generated inspires a new round of revolutions and self-proclaimed warlords, plunging Costaguana into chaos.

4. Among others, the forces of the revolutionary General Montero invade Sulaco after securing the inland capital.

5. Gould, adamant that his silver mine should not become spoil for his enemies, orders Nostromo, the trusted "Capataz de Cargadores" (Head Longshoreman) of Sulaco, to take the mine's most recent load of silver offshore, and arranges for the mine complex to be destroyed by dynamite if the coup leaders try to take it.

6. Nostromo is an Italian expatriate who has risen to his position through his bravery and daring exploits. ("

7. Nostromo" is Italian for "shipmate" or "boatswain", but the name could also be considered a corruption of the Italian phrase "nostro uomo" or "nostr'uomo", meaning "our man").

8. Nostromo's real name is Giovanni Battista Fidanza ÑFidanza meaning "trust" in archaic Italian.

9. Nostromo is a commanding figure in Sulaco, respected by the wealthy Europeans and seemingly limitless in his abilities to command power among the local population.

10. He is, however, never admitted to become a part of upper-class society, but is instead viewed by the rich as their useful tool.

11. He is believed by Charles Gould and his own employers to be incorruptible, and it is for this reason that Nostromo is entrusted with removing the silver from Sulaco to keep it from the revolutionaries.

12. Accompanied by the young journalist Martin Decoud, Nostromo sets off to smuggle the silver out of Sulaco.

13. However, the lighter on which the silver is being transported is struck at night in the waters off Sulaco by a transport carrying the invading revolutionary forces under the command of Colonel Sotillo.

14. Nostromo and Decoud manage to save the silver by putting the lighter ashore on Great Isabel.

15. Decoud and the silver are deposited on the deserted island of Great Isabel in the expansive bay off Sulaco, while Nostromo scuttles the lighter and manages to swim back to shore undetected.

16. Back in Sulaco, Nostromo's power and fame continues to grow as he daringly rides over the mountains to summon the army which ultimately saves Sulaco's powerful leaders from the revolutionaries and ushers in the independent state of Sulaco.

17. In the meantime, left alone on the deserted island, Decoud eventually loses his mind.

18. He takes the small lifeboat out to sea and there shoots himself, after first weighing his body down with some of the silver ingots so that he would sink into the sea.

19. His exploits during the revolution do not bring Nostromo the fame he had hoped for, and he feels slighted and used.

20. Feeling that he has risked his life for nothing, he is consumed by resentment, which leads to his corruption and ultimate destruction, for he has kept secret the true fate of the silver after all others believed it lost at sea.

21. He finds himself becoming a slave of the silver and its secret, even as he slowly recovers it ingot by ingot during nighttime trips to Great Isabel.

22. The fate of Decoud is a mystery to Nostromo, which combined with the fact of the missing silver ingots only adds to his paranoia.

23. Eventually a lighthouse is constructed on Great Isabel, threatening Nostromo's ability to recover the treasure in secret.

24. The ever resourceful Nostromo manages to have a close acquaintance, the widower Giorgio Viola, named as its keeper.

25. Nostromo is in love with Giorgio's younger daughter, but ultimately becomes engaged to his elder daughter Linda.

26. One night while attempting to recover more of the silver, Nostromo is shot and killed, mistaken for a trespasser by old Giorgio.

## D.19 This Side of Paradise by F. Scott Fitzgerald

1. Amory Blaine, a young Midwesterner, believes that he has a great destiny, but the precise nature of this destiny eludes him.

2. He attends a preparatory school where he becomes a football quarterback.

3. He grows estranged from his eccentric mother Beatrice Blaine and becomes the protege of Monsignor Thayer Darcy, a Catholic priest.

4. During his sophomore year at Princeton, he returns to Minneapolis over Christmas break and falls in love with Isabelle Borge, a wealthy debutante whom he first met as a boy.

5. Amory and Isabelle embark upon a romance.

6. While at Princeton, Amory deluges Isabelle with letters, but she becomes disenchanted with him due to his criticism, and they break up on Long Island.

7. Following their separation, Amory accompanies a Princeton classmate to an apartment occupied by two New York showgirls of easy virtue.

8. He considers staying the night with the showgirls, but his conscience and an apparition compel him to leave.

9. After four years at Princeton, he enlists in the United States Army amid World War I. He ships overseas to serve in the muddy trenches of the Western front.

10. While overseas, his mother Beatrice dies, and most of his family's wealth disappears due to a series of failed investments.

11. After the armistice with Imperial Germany in November 1918, Amory settles in New York City amid the flowering of the Jazz Age.

12. Rebounding from Isabelle, he becomes infatuated with Rosalind Connage, a cruel and narcissistic flapper.

13. Desperate for a job, Amory obtains employment with an advertising agency but detests the work.

14. His relationship with Rosalind deteriorates as she prefers a rival suitor, Dawson Ryder, a man of wealth and status.

15. Rejected by Rosalind due to his lack of financial prospects, Amory quits his advertising job and goes on a drinking binge until the start of prohibition in the United States.

16. When Amory travels to visit an uncle in Maryland, he meets Eleanor Savage, a beautiful and reckless atheist.

17. Eleanor chafes under the religious conformity and gender limitations imposed on her by contemporary society in Wilsonian America.

18. Amory and Eleanor spend a lazy summer conversing about love.

19. On their final night together before Amory returns to New York City, Eleanor attempts suicide by riding her horse over a cliff in order to prove her disbelief in any deity.

20. At the last moment, she leaps to safety as her horse plummets over the precipice, and Amory realizes that he does not love her.

21. Returning to New York City, Amory learns of Rosalind's engagement to his wealthy rival Dawson Ryder, and he declares that Rosalind is now dead to him.

22. The death of his beloved mentor, Monsignor Darcy, further dispirits him.

23. Homeless, Amory wanders from New York City to his alma mater Princeton.

24. Accepting a car ride from a wealthy upper-class man driven by his working-class chauffeur in a Locomobile, Amory speaks out in favor of socialism in the United States-although he admits he is still formulating his thoughts as he is talking.

25. While riding in the Locomobile, Amory continues his argument about their time's societal ills and articulates his disillusionment with the current historical era.

26. He announces his hope to stand alongside those in the younger generation and to bring forth a new age in America.

27. Both the upper-class and working-class men in the car denounce his views, but when Amory discovers that the upper-class man is the father of a Princeton classmate who died in World War I, they reconcile.

28. Amory parts ways with his travel companions, and the upper-class man tells him: "Good luck to you and bad luck to your theories."

29. Approaching Princeton, Amory recognizes his selfishness as well as his overindulgence in drink and beauty.

30. He realizes that he must transcend these flaws to become a better man.

31. Wandering through a graveyard at twilight, he reflects upon his inevitable mortality and finds solace in the fact that future generations may one day ponder his life.

32. He thinks of the next generation-inheriting disillusionment and a loss of faith, yet still chasing love and success.

33. After midnight, he stands alone gazing at Princeton's gothic towers and feels a newfound freedom.

34. He stretches out his arms and proclaims, "I know myself . ..

35. but that is all."

## D.20 Heart of Darkness by Joseph Conrad

1. The novella opens on "the sea-reach of the Thames" where Charles Marlow tells his friends that "when the Romans first came here, nineteen hundred years ago" they would have sensed "the savagery, the utter savagery" surrounding them.

2. Marlow then relates how he became captain of a river steamboat for an ivory trading company.

3. He tells of his fascination as a child for "the blank spaces" on maps, particularly in Africa.

4. The image of a river on the map particularly drew his attention.

5. In a flashback, Marlow makes his way to Africa by taking passage on a steamer.

6. He travels 30 miles (50 km) up the river to where his company's station is.

7. Work on a railway is taking place.

8. Marlow explores a narrow ravine, and is horrified to find himself in a place full of critically ill Africans who worked on the railroad and are now dying.

9. Marlow must wait for ten days in the company's devastated Outer Station.

10. Marlow meets the company's chief accountant, who tells him of a Mr. Kurtz, who is in charge of a very important trading post, and is described as a respected first-class agent.

11. The accountant predicts that Kurtz will go far.

12. Marlow departs with 60 men to travel to the Central Station, where the steamboat that he will command is based.

13. At the station, he learns that his steamboat has been wrecked in an accident.

14. The general manager informs Marlow that he could not wait for Marlow to arrive, and tells him of a rumour that Kurtz is ill.

15. Marlow fishes his boat out of the river and spends months repairing it.

16. Delayed by the lack of tools and replacement parts, Marlow is frustrated by the time it takes to perform the repairs.

17. He learns that Kurtz is resented, not admired, by the manager.

18. Once underway, the journey to Kurtz's station takes two months.

19. The journey pauses for the night about eight miles (13 km) below the Inner Station.

20. In the morning the boat is enveloped by a thick fog.

21. The steamboat is later attacked by a barrage of arrows, and the helmsman is killed.

22. Marlow sounds the steam whistle repeatedly, frightening the attackers away.

23. After landing at Kurtz's station, a man boards the steamboat: a Russian wanderer who strayed into Kurtz's camp.

24. Marlow learns that the natives worship Kurtz and that he has been very ill.

25. The Russian tells of how Kurtz opened his mind and how he admires Kurtz for his power and his willingness to use it.

26. Marlow suspects that Kurtz has gone mad.

27. Marlow observes the station and sees a row of posts topped with the severed heads of natives.

28. Around the corner of the house, Kurtz appears with supporters who carry him as a ghost-like figure on a stretcher.

29. The area fills with natives ready for battle, but Kurtz shouts something and they retreat.

30. His entourage carries Kurtz to the steamer and lays him in a cabin.

31. The manager tells Marlow that Kurtz has harmed the company's business in the region because his methods are "unsound".

32. The Russian reveals that Kurtz believes the company wants to kill him, and Marlow confirms that hangings were discussed.

33. After midnight, Kurtz returns to shore.

34. Marlow finds Kurtz crawling back to the station house.

35. Marlow threatens to harm Kurtz if he raises an alarm, but Kurtz only laments that he did not accomplish more.

36. The next day they prepare to journey back down the river.

37. Kurtz's health worsens during the trip.

38. The steamboat breaks down, and while stopped for repairs, Kurtz gives Marlow a packet of papers, including his commissioned report and a photograph, telling him to keep them from the manager.

39. When Marlow next speaks with him, Kurtz is near death; Marlow hears him weakly whisper, "The horror!

40. The horror!"

41. A short while later, the manager's boy announces to the crew that Kurtz has died (the famous line "Mistah Kurtz-he dead" would become the epigraph of T. S. Eliot's poem "The Hollow Men").

42. The next day Marlow pays little attention to Kurtz's pilgrims as they bury "something" in a muddy hole.

43. Returning to Europe, Marlow is embittered and contemptuous of the "civilised" world.

44. Several callers come to retrieve the papers Kurtz entrusted to him, but Marlow withholds them or offers papers he knows they have no interest in.

45. He gives Kurtz's report to a journalist, for publication if he sees fit.

46. Marlow is left with some personal letters and a photograph of Kurtz's fiancee.

47. When Marlow visits her, she is deep in mourning although it has been more than a year since Kurtz's death.

48. She presses Marlow for information, asking him to repeat Kurtz's final words.

49. Marlow tells her that Kurtz's final word was her name.

## D.21 Fanshawe by Nathaniel Hawthorne

1. Dr. Melmoth, the president of fictional Harley College, takes into his care Ellen Langton, the daughter of his friend, Mr. Langton, who is at sea.

2. Ellen is a young, beautiful girl and attracts the attentions of the college boys, especially Edward Walcott, a strapping though immature student, and Fanshawe, a reclusive, meek intellectual.

3. While out walking, the three young people meet a nameless character called 'the angler', a name he gets for appearing an expert fisherman.

4. The angler asks for a word with Ellen, tells her something in secret, and apparently flusters her.

5. Walcott and Fanshawe become suspicious of his intentions.

6. We learn that the angler is an old friend of the reformed Inn owner, Hugh Crombie.

7. The two had been at sea together, where Mr. Langton had been the angler's mentor and caretaker.

8. Langton and the angler had a falling out, however, and, thinking that Langton has been killed at sea, the angler undertakes to marry Ellen in order to inherit her father's considerable wealth.

9. Thus in his secret meeting with Ellen, the angler instructs her to sneak out of Melmoth's home and follow him, telling her he has information about her father's whereabouts.

10. His real aim, though, is to kidnap her, to tell her of her father's death, and to manipulate her into marrying him.

11. When the various men (Melmoth, Edward, Fanshawe) learn that she is not in her chamber, they go searching for her.

12. The search reveals the nature of each: Melmoth, an aged scholar unused to physical labor, enlists the help of Walcott, who is the most skilled rider and the most likely to be able to contend with the angler in a fight.

13. Fanshawe, who lags behind the search because of his weak constitution and his slow horse, is given information by an old woman in a cabin (where another old woman, Widow Butler, who turns out to be the angler's mother, has just died) that allows him to reach the angler and Ellen first.

14. The angler has taken Ellen to a craggy cliff and cave, where he intends to hold her captive.

15. Ellen has finally realized the angler's intentions.

## D.22 The Sound and the Fury by William Faulkner

20. Over the following years, Mr. Compson dies from alcoholism, leaving Jason and his mother to run the house.

21. Roskus, Dilsey's husband, also dies.

22. Benjy's former plot of land, sold to pay for Quentin's tuition, becomes a golf course.

23. Benjy is castrated after chasing a schoolgirl he believes to be Caddy.

24. Caddy leaves Miss Quentin with Jason and her mother, but they refuse to let Caddy live with them.

25. Instead, Caddy sends them money every month, which Jason keeps for himself.

26. Most of the novel takes place over Easter weekend in 1928.

27. On Good Friday, Jason torments Miss Quentin, whom he blames for costing him the job at Head's bank, and unsuccessfully uses Caddy's money to short-sell cotton futures.

28. On Holy Saturday, Benjy celebrates his 33rd birthday.

29. Dilsey's grandson Luster looks after him while searching for a quarter so he can attend a traveling carnival; he ultimately receives one from Miss Quentin.

30. That night, Miss Quentin steals Jason's money.

31. She then escapes the Compson house by climbing down the same tree Caddy climbed as a child.

32. On Easter, Dilsey takes her family and Benjy to a black church, where they listen to a sermon from Reverend Shegog.

33. Meanwhile, Jason attempts to chase after Miss Quentin, who has left town with a man from the carnival.

34. After catching up with the carnival in the nearby town of Mottson, Jason provokes a fight with one of the workers and injures himself in the confrontation.

35. Having failed to find Miss Quentin, he hires a black man to drive him home.

36. Back in Jefferson, Luster takes Benjy to the town cemetery, but travels the wrong way around a Confederate monument in the town square.

37. Jason returns to find Benjy crying over this disruption to his routine; he intervenes and takes Benjy the right way.

## D.23 Pollyanna by Eleanor H. Porter

1. The title character is Pollyanna Whittier, an eleven-year-old orphan who goes to live in the fictional town of Beldingsville, Vermont, with her wealthy but stern and cold spinster Aunt Polly Harrington, who does not want to take in Pollyanna but feels it is her duty to her late sister Jennie.

2. Pollyanna's philosophy of life centers on what she calls "The Glad Game", an optimistic and positive attitude she learned from her father.

3. The game consists of finding something to be glad about in every situation, no matter how bleak it may be.

4. It originated in an incident one Christmas when Pollyanna, who was hoping for a doll in the missionary barrel, found only a pair of crutches inside.

5. Making the game up on the spot, Pollyanna's father taught her to look at the good side of thingsÑin this case, to be glad about the crutches because she did not need to use them.

6. With this philosophy, and her own sunny personality and sincere,

7. The Glad Game shields her from her aunt's stern attitude: when Aunt Polly puts her in a stuffy attic room without carpets or pictures, she exults at the beautiful view from the high window; when she tries to "punish" her niece for being late to dinner by sentencing her to a meal of bread and milk in the kitchen with the servant Nancy, Pollyanna thanks her rapturously because she likes bread and milk, and she likes Nancy.

8. Soon Pollyanna teaches some of Beldingsville's most troubled inhabitants to "play the game" as well, from Mrs. Snow, a querulous invalid, to Mr. Pendleton, a miserly bachelor who lives all alone in a cluttered mansion.

9. Aunt Polly, too-finding herself helpless before Pollyanna's buoyant refusal to be downcast-gradually begins to thaw, although she resists the Glad Game longer than anyone else.

10. Eventually, however, even Pollyanna's robust optimism is put to the test when she is struck by a car and loses the use of her legs.

11. At first, she does not realize the seriousness of her injury, but her spirits plummet when she learns she will probably never walk again.

12. After that, she lies in bed, unable to find anything to be glad about.

13. Then the townspeople begin calling at Aunt Polly's house, eager to let Pollyanna know how much her encouragement has improved their lives; and Pollyanna decides she can still be glad that she at least had her legs.

14. The novel ends with Aunt Polly marrying her former lover Dr. Chilton and Pollyanna being sent to a specialist in spinal injuries, where she learns to walk again and is able to appreciate the use of her legs far more as a result of being temporarily disabled and unable to walk well.

## D.24 The Seven Dials Mystery by Agatha Christie

1. Sir Oswald and Lady Coote host a party at the stately home Chimneys, which they have rented for the season.

2. The guest list includes Gerry Wade, Jimmy Thesiger, Ronny Devereux, Bill Eversleigh and Rupert "Pongo" Bateman

3. Since Wade has a bad habit of oversleeping, the others play a joke on him by placing eight alarm clocks in his room and timing them to go off at intervals.

4. The next morning, a footman finds Wade dead in his bed, with chloral on his nightstand.

5. Thesiger notices that one of the eight alarm clocks is missing.

6. It is later found in a hedge.

7. Lord Caterham and his daughter Lady Eileen 'Bundle' Brent move back into Chimneys.

8. When Bundle drives to London to see Eversleigh, Ronny Devereux jumps out in front of her car.

9. Before he dies, Devereux mutters "Seven Dials..." and "Tell...Jimmy Thesiger."

10. Bundle gets his body to a doctor, who tells her that her car did not hit Devereux; he was shot.

11. Seven Dials turns out to be a seedy nightclub and gambling den in the Seven Dials area of London.

12. Bundle recognises the doorman as Alfred, a footman from Chimneys.

13. Alfred tells her that he left Chimneys for far higher wages offered by Mosgorovsky, owner of the club.

14. Alfred takes Bundle into a secret room, where she hides in a cupboard and witnesses a meeting of six people wearing hoods with clock faces.

15. They talk of the always-missing "Number Seven", and about an upcoming party at Wyvern Abbey, where a scientist called Eberhard will offer a secret formula for sale to the British Air Minister.

16. At the party, the formula is stolen, but retrieved by Wade's stepsister; and Jimmy Thesiger is shot in his right arm.

17. Thesiger tells how he fought a man who climbed down the ivy.

18. The next morning, Battle finds a charred left-handed glove with teeth marks in the fireplace.

19. He theorises that the thief threw the gun onto the lawn from the terrace and then climbed back into the house via the ivy.

20. Bundle's father reports that Bauer, the footman who replaced Alfred, is missing.

21. Thesiger rings up Bundle and Gerry Wade's stepsister Loraine and tells them to meet him and Eversleigh at the Seven Dials club.

22. Bundle shows Thesiger the room where the Seven Dials meet.

23. Loraine finds Eversleigh unconscious in the car and they take him into the club.

24. Thesiger says that he will go for a doctor.

25. Someone knocks Bundle unconscious and she comes round in Eversleigh's arms.

26. Mr Mosgorovsky takes them into the meeting of the Seven Dials, where Number Seven is revealed as Superintendent Battle.

27. He reveals that they are a group of people doing secret service work for the government.

28. Battle tells Bundle that the association has succeeded with their main target, an international criminal whose stock in trade is the theft of secret formulae: Jimmy Thesiger was arrested that afternoon with his accomplice, Loraine Wade.

29. Battle explains that Thesiger killed Wade and Devereux when they got onto his track.

30. Devereux took the eighth clock from Wade's room to see if anyone reacted to there being "seven dials".

31. At Wyvern Abbey, Thesiger stole the formula, passed it to Loraine, then shot himself in his right arm and disposed of his left-hand glove using his teeth.

32. Eversleigh feigned unconsciousness in the car outside the Seven Dials club.

33. Thesiger never went for a doctor, but hid in the club, and knocked Bundle unconscious.

34. Bundle takes Wade's place in the Seven Dials and marries Bill Eversleigh.

## D.25 Wolfbane by Frederik Pohl and C. M. Kornbluth

10. They find he doesn't entirely fit in there either, but hope he may get collected by an 'Eye', giving them a chance to measure this process in detail.

11. This eventually happens and they find, as expected, that his disappearance was facilitated by the Pyramid on Everest.

12. Glenn Tropile has been sent to the Pyramid's planet.

13. We then learn his fate.

14. To the Pyramids, the human race is nothing more than a useful source of 'Components' for a complex world-machine devoted mostly to feeding these artificial and semi-organic beings.

15. Tropile is suspended in a fluid-filled tank and 'wired in'to the vast computer system.

16. Later, he is linked to seven other humans as a 'Snowflake' - eight minds joined together to facilitate more complex tasks than a single human Component could manage.

17. In this condition, Tropile wakes, manages to retain his sanity, and wakes the other humans.

18. They eventually merge with one another to form a sophisticated collective mind.

19. The freed Snowflake then spies on the Pyramids, finding that they have been traveling for some two million years, and have collected many species as Components, but seem locked in meaningless rituals surrounding an alien creature at the world's North Pole.

20. They later realize that this is the last survivor of the race that created the Pyramids.

21. In the meantime, they have modified the collection process of human Components so that it selects persons known to at least one of the eight people composing the Snowflake (which has become almost as ruthless and inhuman as the alien Pyramids).

22. These humans are intended as 'mice,' disrupters of the planet, and later as an army with which to fight the Pyramids.

23. Roget Germyn is one of them, as is Tropile's wife.

24. Facing a philosophical dead-end, the 'Snowflake' decides to separate its component minds to study the problem.

25. Restored to individual identity, Glenn Tropile becomes horrified at what he has done.

26. However, the majority of the others want to carry on.

27. While they are arguing, one of their number is taken over by the mind of the alien at the North Pole, who warns them that the Pyramids have noticed them and plan to wipe out half the planet to get rid of them.

28. They initially re-merge their personalities as the Snowflake and expedite their plans.

29. Tropile decides he must physically disconnect himself from the Snowflake and leave to lead the humans.

30. They manage to defeat the Pyramids, but not before the remainder of the Snowflake is also destroyed.

31. The humans free many of the other Components and ship them back to Earth.

32. Tropile now finds he is a hero of sorts, but does not fit the role (though he never truly fit any role in which he was placed - Sheep, Wolf, or Component).

33. He also sees that there is a need for someone to wire themselves back into the alien planet's surviving systems, to re-kindle the 'sun' every five years and perhaps return the Earth to its original orbit.

34. He doesn't want to do it alone, but most of the people he knows are either unwilling or unsuitable.

35. On the last page, though, his wife agrees to join him.

36. He expects that there will later be others, that "[t]he ring of fire [will] grow."

## D.26 Germinal by Émile Zola

1. The novel's central character is Etienne Lantier, previously seen in L'Assommoir (1877), and originally to have been the central character in Zola's "murder on the trains" thriller La Bote humaine (1890) before the overwhelmingly positive reaction to Germinal persuaded him otherwise.

2. The young migrant worker arrives at the forbidding coal mining town of Montsou in the bleak area of the far north of France to earn a living as a miner.

3. Sacked from his previous job on the railways for assaulting a superior, Etienne befriends the veteran miner Maheu, who finds him somewhere to stay and gets him a job pushing the carts down the pit.

4. Etienne is portrayed as a hard-working idealist but also a naive youth; Zola's genetic theories come into play as Etienne is presumed to have inherited his Macquart ancestors' traits of hotheaded impulsiveness and an addictive personality capable of exploding into rage under the influence of drink or strong passions.

5. Zola keeps his theorizing in the background and Etienne's motivations are much more natural as a result.

6. He embraces socialist principles, reading large amounts of working class movement literature and fraternizing with Souvarine, a Russian anarchist and political emigre who has also come to Montsou to seek a living in the pits, and Rasseneur, a pub owner.

7. Etienne's simplistic understanding of socialist politics and their rousing effect on him are very reminiscent of the rebel Silvere in the first novel in the cycle, La Fortune des Rougon (1871).

8.While this is going on, Etienne also falls for Maheu's daughter Catherine, also employed pushing carts in the mines, and he is drawn into the relationship between her and her brutish lover Chaval, a prototype for the character of Buteau in Zola's later novel La Terre (1887).

9. The complex tangle of the miners' lives is played out against a backdrop of severe poverty and oppression, as their working and living conditions continue to worsen throughout the novel; eventually, pushed to breaking point, the miners decide to strike and Etienne, now a respected member of the community and recognized as a political idealist, becomes the leader of the movement.

10. While the anarchist Souvarine preaches violent action, the miners and their families hold back, their poverty becoming ever more disastrous, until they are sparked into a ferocious riot, the violence of which is described in explicit terms by Zola, as well as providing some of the novelist's best and most evocative crowd scenes.

11. The rioters are eventually confronted by police and the army that repress the revolt in a violent and unforgettable episode.

12. Disillusioned, the miners go back to work, blaming Etienne for the failure of the strike; then, Souvarine sabotages the entrance shaft of one of the Montsou pits, trapping Etienne, Catherine and Chaval at the bottom.

13. The ensuing drama and the long wait for rescue are among some of Zola's best scenes, and the novel draws to a dramatic close.

14. After Chaval is killed by Etienne, Catherine and Etienne are finally able to be lovers before Catherine dies in his arms.

15. Etienne is eventually rescued and fired but he goes on to live in Paris with Pluchart, an organizer for the International.

## D.27 Ulysses by James Joyce

1. At 8 a.m., Malachi "Buck" Mulligan, a boisterous medical student, calls an aspiring writer, Stephen Dedalus, up to the roof of the Sandycove Martello tower, where they live.

2. There is tension between Stephen and Mulligan: Stephen overheard Mulligan make a cruel remark about Stephen's recently deceased mother, and Mulligan has invited an English student, Haines, whom Stephen dislikes, to stay with them.

3. The three men eat breakfast and walk to the shore, where Mulligan demands from Stephen the key to the tower and a loan.

4. The three make plans to meet at a pub, The Ship, at 12:30pm.

5. Departing, Stephen decides that he will not return to the tower that night, as Mulligan, the "usurper", has taken it over.

6. Stephen is teaching a history class on the victories of Pyrrhus of Epirus.

7. After class, one student, Cyril Sargent, stays behind so that Stephen can show him how to do a set of algebraic exercises.

8. Stephen looks at Sargent's ugly face and tries to imagine Sargent's mother's love for him.

9. He then visits unionist school headmaster Garrett Deasy, from whom he collects his pay.

10. Deasy asks Stephen to take his long-winded letter about foot-and-mouth disease to a newspaper office for printing.

11. The two discuss Irish history and Deasy lectures on what he believes is the role of Jews in the economy.

12. As Stephen leaves, Deasy jokes that Ireland has "never persecuted the Jews" because the country "never let them in".

13. This episode is the source of some of the novel's best-known lines, such as Dedalus's claim that "history is a nightmare from which I am trying to awake" and that God is "a shout in the street".

14. Stephen walks along Sandymount Strand for some time, mulling various philosophical concepts, his family, his life as a student in Paris, and his mother's death.

15. As he reminisces he lies down among some rocks, watches a couple whose dog urinates behind a rock, scribbles some ideas for poetry and picks his nose.

16. This chapter is characterised by a stream of consciousness narrative style that changes focus wildly.

17. Stephen's education is reflected in the many obscure references and foreign phrases employed in this episode, which have earned it a reputation for being one of the book's most difficult chapters.

18. The narrative shifts abruptly.

19. The time is again 8 a.m., but the action has moved across the city and to the second protagonist of the book, Leopold Bloom, a part-Jewish advertising canvasser.

20. The episode opens with the line "Mr. Leopold Bloom ate with relish the inner organs of beasts and fowls."

21. After starting to prepare breakfast, Bloom decides to walk to a butcher to buy a mutton kidney.

22. Returning home, he prepares breakfast and brings it with the mail to his wife Molly as she lounges in bed.

23. One of the letters is from her concert manager Blazes Boylan, with whom she is having an affair.

24. Bloom reads a letter from their daughter Milly Bloom, who tells him about her progress in the photography business in Mullingar.

25. The episode closes with Bloom reading a magazine story titled "Matcham's Masterstroke", by Mr. Philip Beaufoy, while defecating in the outhouse.

26. While making his way to Westland Row post office Bloom is tormented by the knowledge that Molly will welcome Boylan into her bed later that day.

27. At the post office he surreptitiously collects a love letter from one 'Martha Clifford' addressed to his pseudonym, 'Henry Flower'.

28. He meets an acquaintance, and while they chat, Bloom attempts to ogle a woman wearing stockings, but is prevented by a passing tram.

29. Next, he reads the letter from Martha Clifford and tears up the envelope in an alley.

30. He wanders into a Catholic church during a service and muses on theology.

31. The priest has the letters I.N.R.I. or I.H.S. on his back; Molly had told Bloom that they meant I have sinned or I have suffered, and Iron nails ran in.

32. He buys a bar of lemon soap from a chemist.

33. He then meets another acquaintance, Bantam Lyons, who mistakenly takes him to be offering a racing tip for the horse Throwaway.

34. Finally, Bloom heads towards the baths.

35. The episode begins with Bloom entering a funeral carriage with three others, including Stephen's father.

36. They drive to Paddy Dignam's funeral, making small talk on the way.

37. The carriage passes both Stephen and Blazes Boylan.

38. There is discussion of various forms of death and burial.

39. Bloom is preoccupied by thoughts of his dead infant son, Rudy, and the suicide of his own father.

40. They enter the chapel for the service and subsequently leave with the coffin cart.

41. Bloom sees a mysterious man wearing a Mackintosh raincoat during the burial.

42. Bloom continues to reflect upon death, but at the end of the episode rejects morbid thoughts to embrace "warm fullblooded life".

43. At the office of the Freeman's Journal, Bloom attempts to place an ad.

44. Although initially encouraged by the editor, he is unsuccessful.

45. Stephen arrives bringing Deasy's letter about foot-and-mouth disease, but Stephen and Bloom do not meet.

46. Stephen leads the editor and others to a pub, relating an anecdote on the way about "two Dublin vestals".

47. The episode is broken into short segments by newspaper-style headlines, and is characterised by an abundance of rhetorical figures and devices.

48. Bloom's thoughts are peppered with references to food as lunchtime approaches.

49. He meets an old flame, hears news of Mina Purefoy's labour, and helps a blind boy cross the street.

50. He enters the restaurant of the Burton Hotel, where he is revolted by the sight of men eating like animals.

51. He goes instead to Davy Byrne's pub, where he consumes a gorgonzola cheese sandwich and a glass of burgundy, and muses upon the early days of his relationship with Molly and how the marriage has declined: "Me.

52. And me now."

53. Bloom's thoughts touch on what goddesses and gods eat and drink.

54. He ponders whether the statues of Greek goddesses in the National Museum have anuses as do mortals.

55. On leaving the pub Bloom heads toward the museum, but spots Boylan across the street and, panicking, rushes into the gallery across the street from the museum.

56. At the National Library, Stephen explains to some scholars his biographical theory of the works of Shakespeare, especially Hamlet, which he argues are based largely on the posited adultery of Shakespeare's wife.

57. Buck Mulligan arrives and interrupts to read out the telegram that Stephen had sent him indicating that he would not make their planned rendezvous at The Ship.

58. Bloom enters the National Library to look up an old copy of the ad he has been trying to place.

59. He passes in between Stephen and Mulligan as they exit the library at the end of the episode.

60. In this episode, nineteen short vignettes depict the movements of various characters, major and minor, through the streets of Dublin.

61. The episode begins by following Father Conmee, a Jesuit priest, on his trip north, and ends with an account of the cavalcade of the Lord Lieutenant of Ireland, William Ward, Earl of Dudley, through the streets, which is encountered by several characters from the novel.

62. In this episode, dominated by motifs of music, Bloom has dinner with Stephen's uncle at the Ormond hotel, while Molly's lover, Blazes Boylan, proceeds to his rendezvous with her.

63. While dining, Bloom listens to the singing of Stephen's father and others, watches the seductive barmaids, and composes a reply to Martha Clifford's letter.

64. This episode is narrated by an unnamed denizen of Dublin who works as a debt collector.

65. The narrator goes to Barney Kiernan's pub where he meets a character referred to only as "The Citizen".

66. This character is believed to be a satirisation of Michael Cusack, a founder member of the Gaelic Athletic Association.

67. When Leopold Bloom enters the pub, he is berated by the Citizen, who is a fierce Fenian and anti-Semite.

68. The episode ends with Bloom reminding the Citizen that his Saviour was a Jew.

69. As Bloom leaves the pub, the Citizen throws a biscuit tin at Bloom's head, but misses.

70. The episode is marked by extended tangents made in voices other than that of the unnamed narrator; these include streams of legal jargon, a report of a boxing match, Biblical passages, and elements of Irish mythology.

71. All the action of the episode takes place on the rocks of Sandymount Strand, the shoreline that Stephen visited in Episode 3.

72. A young woman, Gerty MacDowell, is seated on the rocks with her two friends, Cissy Caffrey and Edy Boardman.

73. The girls are taking care of three children, a baby, and four-year-old twins named Tommy and Jacky.

74. Gerty contemplates love, marriage and femininity as night falls.

75. The reader is gradually made aware that Bloom is watching her from a distance.

76. Gerty teases the onlooker by exposing her legs and underwear, and Bloom, in turn, masturbates.

77. Bloom's masturbatory climax is echoed by the fireworks at the nearby bazaar.

78. As Gerty leaves, Bloom realises that she has a lame leg, and believes this is the reason she has been "left on the shelf".

79. After several mental digressions he decides to visit Mina Purefoy at the maternity hospital.

80. It is uncertain how much of the episode is Gerty's thoughts, and how much is Bloom's sexual fantasy.

81. Some believe that the episode is divided into two halves: the first half the highly romanticized viewpoint of Gerty, and the other half that of the older and more realistic Bloom.

82. Joyce himself said, however, that "nothing happened between [Gerty and Bloom].

83. It all took place in Bloom's imagination".

84. Nausicaa attracted immense notoriety while the book was being published in serial form.

85. It has also attracted great attention from scholars of disability in literature.

86. The style of the first half of the episode borrows from (and parodies) romance magazines and novelettes.

87. Bloom's contemplation of Gerty parodies Dedalus's vision of the wading girl at the seashore in A Portrait of the Artist as a Young Man.

88. Bloom visits the maternity hospital where Mina Purefoy is giving birth, and finally meets Stephen, who has been drinking with his medical student friends and is awaiting the promised arrival of Buck Mulligan.

89. As the only father in the group of men, Bloom is concerned about Mina Purefoy in her labour.

90. He starts thinking about his wife and the births of his two children.

91. He also thinks about the loss of his only 'heir', Rudy.

92. The young men become boisterous, and start discussing such topics as fertility, contraception and abortion.

93. There is also a suggestion that Milly, Bloom's daughter, is in a relationship with one of the young men, Bannon.

94. They continue on to a pub to continue drinking, following the successful birth of a son to Mina Purefoy.

95. This chapter is remarkable for Joyce's wordplay, which, among other things, recapitulates the entire history of the English language.

96. After a short incantation, the episode starts with latinate prose, Anglo-Saxon alliteration, and moves on through parodies of, among others, Malory, the King James Bible, Bunyan, Pepys, Defoe, Sterne, Walpole, Gibbon, Dickens, and Carlyle, before concluding in a Joycean version of contemporary slang.

97. The development of the English language in the episode is believed to be aligned with the nine-month gestation period of the foetus in the womb.

98. Episode 15 is written as a play script, complete with stage directions.

99. The plot is frequently interrupted by "hallucinations" experienced by Stephen and Bloom-fantastic manifestations of the fears and passions of the two characters.

100. Stephen and his friend Lynch walk into Nighttown, Dublin's red-light district.

101. Bloom pursues them and eventually finds them at Bella Cohen's brothel where, in the company of her workers including Zoe Higgins, Florry Talbot and Kitty Ricketts, he has a series of hallucinations regarding his sexual fetishes, fantasies and transgressions.

102. In one of these hallucinations, Bloom is put in the dock to answer charges by a variety of sadistic, accusing women including Mrs Yelverton Barry, Mrs Bellingham and the Hon Mrs Mervyn Talboys.

103. In another of Bloom's hallucinations, he is crowned king of his own city, which is called Bloomusalem-Bloom imagines himself being loved and admired by Bloomusalem's citizens, but then imagines himself being accused of various charges.

104. As a result, he is burnt at the stake and several citizens pay their respects to him as he dies.

105. Then the hallucination ends, Bloom finds himself next to Zoe, and the two talk.

106. After they talk, Bloom continues to encounter other miscellaneous hallucinations, including one in which he converses with his grandfather Lipoti Virag, who lectures him about sex, among other things.

107. At the end of the hallucination, Bloom is speaking with some prostitutes when he hears a sound coming from downstairs.

108. He hears heels clacking on the staircase, and he observes what appears to be a male form passing down the staircase.

109. He speaks with Zoe and Kitty for a moment, and then sees Bella Cohen come into the brothel.

110. He observes her appearance and talks with her for a little while.

111. But this conversation subsequently begins another hallucination, in which Bloom imagines Bella to be a man named Mr. Bello and Bloom imagines himself to be a woman.

112. In this fantasy, Bloom imagines himself (or "herself", in the hallucination) being dominated by Bello, who both sexually and verbally humiliates Bloom.

113. Bloom also interacts with other imaginary characters in this scene before the hallucination ends.

114. After the hallucination ends, Bloom sees Stephen overpay at the brothel, and decides to hold onto the rest of Stephen's money for safekeeping.

115. Stephen hallucinates that his mother's rotting cadaver has risen up from the floor to confront him.

116. He cries Non serviam!,

117. uses his walking stick to smash a chandelier, and flees the room.

118. Bloom quickly pays Bella for the damage, then runs after Stephen.

119. He finds Stephen engaged in an argument with an English soldier, Private Carr, who, after hearing Stephen utter a perceived insult to King Edward VII, punches him.

120. The police arrive and the crowd disperses.

121. As Bloom tends to Stephen, he has a hallucination of his deceased son, Rudy, as an 11-year-old.

122. Bloom takes Stephen to a cabman's shelter near Butt Bridge to restore him to his senses.

123. There, they encounter a drunken sailor, D. B. Murphy (W. B. Murphy in the 1922 text).

124. The episode is dominated by the motif of confusion and mistaken identity, with Bloom, Stephen and Murphy's identities being repeatedly called into question.

125. The narrative's rambling and laboured style in this episode reflects the protagonists'nervous exhaustion and confusion.

126. Bloom returns home with Stephen, makes him a cup of cocoa, discusses cultural and linguistic differences between them, considers the possibility of publishing Stephen's parable stories, and offers him a place to stay for the night.

127. Stephen refuses Bloom's offer and is ambiguous in response to Bloom's proposal of future meetings.

128. The two men urinate in the backyard, Stephen departs and wanders off into the night, and Bloom goes to bed, where Molly is sleeping.

129. She awakens and questions him about his day.

130. The episode is written in the form of a rigidly organised and "mathematical" catechism of 309 questions and answers, and was reportedly Joyce's favourite episode in the novel.

131. The deep descriptions range from questions of astronomy to the trajectory of urination and include a list of 25 men that purports to be the "preceding series" of Molly's suitors and Bloom's reflections on them.

132. While describing events apparently chosen randomly in ostensibly precise mathematical or scientific terms, the episode is rife with errors made by the undefined narrator, many or most of which are intentional by Joyce.

133. The final episode consists of Molly Bloom's thoughts as she lies in bed next to her husband.

134. The episode uses a stream-of-consciousness technique in eight paragraphs and lacks punctuation.

135. Molly thinks about Boylan and Bloom, her past admirers, including Lieutenant Stanley G. Gardner, the events of the day, her childhood in Gibraltar, and her curtailed singing career.

136. She also hints at a lesbian relationship in her youth, with a childhood friend, Hester Stanhope.

137. These thoughts are occasionally interrupted by distractions, such as a train whistle or the need to urinate.

138. Molly is surprised by the early arrival of her menstrual period, which she ascribes to her vigorous sex with Boylan.

139. The episode concludes with Molly's remembrance of Bloom's marriage proposal, and of her acceptance: "he asked me would I yes to say yes my mountain flower and first I put my arms around him yes and drew him down to me so he could feel my breasts all perfume yes and his heart was going like mad and yes I said yes I will Yes."

## D.28 Micromegas by Voltaire

1. The story is organized into seven brief chapters.

2. The first describes Micromegas, whose name literally means "small-large", an inhabitant of a planet orbiting the star Sirius.

3. Micromegas stands 120,000 royal feet (38.9 km) tall and his circumference at the waist is 50,000 royal feet (16.24 km).

4. The Sirian's home world is calculated to be 21.6 million times greater in circumference than Earth using mathematical ratios in a passage intended to relativize Man's home on a cosmic scale.

5. When he is almost 450 years old, approaching the end of what the inhabitants of the planet orbiting Sirius consider his childhood and having already solved over fifty of Euclid's problems (eighteen more than Blaise Pascal) before the age of 250 years while studying at his planet's Jesuit college, Micromegas writes a scientific book examining the insects on his planet, which at 100 royal feet (32.5 m) are too small to be detected by ordinary Sirian microscopes.

6. This book is considered heresy by his country's mufti, and after a 200-year trial, he is banished from the court for a term of 800 years.

7. Micromegas takes this as an opportunity to travel between the various planets in a quest to develop his heart and his mind.

8. Micromegas proceeds to begin his journey, traveling by taking advantage of gravity and "the forces of repulsion and attraction" (a reference endorsing the work of Sir Isaac Newton), and after extensive celestial travels he arrives on Saturn, where he befriends the native population and developed an intimate friendship with the secretary of the Academy of Saturn, a man less than a twentieth of his size (a "dwarf" standing only 6,000 royal feet or 1.95 km tall) and described as being clever but lacking the capacity for true genius.

9. In the second chapter, they discuss the differences between their planets.

10. The Saturnian has 72 senses while the Sirian has almost 1,000.

11. The Saturnian lives for 15,000 Saturnian years while the Sirian lives 700 times longer; Micromegas reports that he has visited worlds where people live much longer than this, but who still consider their lifespans too short.

12. All of this further relativizes the size of the Earth in relation to the extraterrestrials, but Micromegas also engages the Saturnian philosophically and found him disappointing.

13. At the end of their conversation, they decide to take a philosophical journey together, and, in a comedic passage that begins chapter three, the Saturnian's mistress arrives with the intent of preventing her lover's departure.

14. The Secretary woos her and she leaves to console herself with a local dandy.

15. The two aliens set off from Saturn in pursuit of knowledge, visiting Saturn's ring, its moons, Jupiter's moons, Jupiter itself (for one Earth-year), and Mars, which they find so small that they fear that they cannot even lay down.

16. Eventually, they arrive on Earth on July 5, 1737 at the end of the third chapter and pause only to eat some mountains for lunch at the start of chapter four before circumnavigating the globe in 36 hours with the Saturnian only getting his lower legs wet in the deepest ocean and the Sirian barely wetting his ankles.

17. The Saturnian decides that the planet must be devoid of life, since he had as of yet seen none but Micromegas chastises him, resisting the temptation to make hasty conclusions and using his reason to direct his search.

18. The Sirian fashions a magnifying glass from a diamond in his necklace measuring 160 royal feet in diameter and spots a tiny speck in the Baltic sea which he discovers is a whale.

19. The Saturnian proceeds to ask many questions, including how such a tiny "atom" could move, if it was sentient, and many others which embarrassed the Sirian.

20. As they examine it, Micromegas finds a boatful of philosophers on their return from the Arctic Circle and carefully picks their ship up.

21. In chapter five, the space travelers examine the boat and notice the men aboard only upon their driving a pole into his finger.

22. It is here that Voltaire breaks with the narrative to briefly relativize Man's diminutive size using the ratio of a man's height to the size of the Earth and uses the moment to perform the same calculus on the scale of human conflict.

23. Using their magnifying-glass, the travelers become able to see the humans.

24. In chapter six, the Secretary hastily concludes that the tiny beings are too small to be of any intelligence or spirit, and Micromegas reasons with him to convince his companion that what he sees is the humans speaking with each other.

25. Still, they cannot yet hear them and the travelers devise a hearing tube made with the clippings of Micromegas's fingernails in order to hear the tiny voices.

26. After listening for a while, they come to discern the words spoken and to understand French.

27. In order to establish communication while fearing that their full voices might deafen the humans, they devise a method in which they carry their suppressed voices through toothpicks to the men on the Sirian's finger.

28. They begin a conversation, wherein they are shocked to discover the breadth of the human intellect but also are exposed to human vanity and philosophy, which the travelers come to mock.

29. The travelers first are amazed at the humans' ability to measure their visitors, establishing an equality of the mind at all scales, and informs the travelers that such creatures as bees exist and that animals exist that are equally as small to bees as men are to the Micromegas.

30. The seventh and final chapter sees the humans testing the philosophies of Aristotle, Descartes, Malebranche, Leibniz and Locke against the travelers' wisdom.

31. Beginning the deeper conversation, one of the human philosophers explains to the extraterrestrial visitors that Mankind had not found lasting happiness and that, to the contrary, hundreds of thousands of men will go to war against each other for, in the novella's relativization, insignificant quarrels.

33. The conversation shifts upon the travelers'learning the occupation of their interlocutors towards the scientific prowess of Man, which ends when philosophical questions are asked.

34. Each philosopher espouses the teachings that he follows, and Micromegas finds fault in each theory save for that of the disciple of Locke, who exhibits philosophical modesty.

35. When the travelers hear the theory of Aquinas from his Summa Theologica that the universe was made uniquely for mankind, they fall into an enormous fit of laughter which causes the ship and its philosophers to fall in the Sirian's pocket.

36. Micromegas then is angry with the arrogance of Mankind and, taking pity on the humans, the Sirian decides to write them a book that will explain everything to them philosophically.

37. When the volume is presented to the French Academy of Sciences, the Academy's secretary opens the book only to find blank pages.

## E Per-Book Metric Values

<table><tr><td>Title</td><td>l</td><td> $p _ { c }$   $p _ { s }$ </td><td> $\mu _ { m }$ </td><td>μODD</td><td> $G _ { c }$ </td><td> $G _ { s }$ </td></tr><tr><td>A Christmas Carol</td><td>1.00</td><td>1.00 1.00</td><td>0.52</td><td>0.07</td><td>0.07</td><td>0.03</td></tr><tr><td>A Farewell to Arms</td><td>0.91</td><td>0.95 0.99</td><td>0.56</td><td>0.10</td><td>0.32</td><td>0.18</td></tr><tr><td>A High Wind in Jamaica</td><td>0.88</td><td>0.90 0.92</td><td>0.62</td><td>0.12</td><td>0.30</td><td>0.10</td></tr><tr><td>A Journey to the Centre of the Earth</td><td>1.00</td><td>0.89 1.00</td><td>0.50</td><td>0.09</td><td>0.17</td><td>0.38</td></tr><tr><td>A Little Princess</td><td>0.89</td><td>1.00 1.00</td><td>0.48</td><td>0.08</td><td>0.29</td><td>0.21</td></tr><tr><td>A Room with a View</td><td>0.98</td><td>0.90 0.98</td><td>0.44</td><td>0.07</td><td>0.36</td><td>0.07</td></tr><tr><td>A Study in Scarlet</td><td>1.00</td><td>0.93 0.98</td><td>0.56</td><td>0.06</td><td>0.36</td><td>0.02</td></tr><tr><td>A Tale of Two Cities</td><td>0.98</td><td>0.62 0.99</td><td>0.61</td><td>0.10</td><td>0.35</td><td>0.09</td></tr><tr><td>A Voyage to Arcturus</td><td>0.98</td><td>0.95 0.98</td><td>0.58</td><td>0.09</td><td>0.25</td><td>0.06</td></tr><tr><td>Adam Bede</td><td>1.00</td><td>0.41 0.80</td><td>0.55</td><td>0.16</td><td>0.04</td><td>0.49</td></tr><tr><td>Alice's Adventures in Wonderland</td><td>0.98</td><td>1.00 1.00</td><td>0.48</td><td>0.08</td><td>0.25</td><td>0.10</td></tr><tr><td>Anne of Green Gables</td><td>0.58</td><td>0.71 1.00</td><td>0.38</td><td>0.13</td><td>0.31</td><td>0.49</td></tr><tr><td>Anthem</td><td>0.95</td><td>0.92</td><td>1.00 0.38</td><td></td><td>0.13 0.44</td><td>0.05</td></tr><tr><td>Armadale</td><td>0.80</td><td>0.70 0.90</td><td>0.41</td><td></td><td>0.17 0.26</td><td>0.52</td></tr><tr><td>Around the World in Eighty Days</td><td>0.99</td><td>0.92</td><td>1.00</td><td>0.51</td><td>0.04 0.26</td><td>0.13</td></tr><tr><td>Arrowsmith</td><td>0.96</td><td>0.65</td><td>1.00</td><td>0.55</td><td>0.15 0.21</td><td>0.37</td></tr><tr><td>Babbitt</td><td>0.82</td><td>0.71</td><td>1.00 0.51</td><td></td><td>0.11 0.23</td><td>0.33</td></tr><tr><td>Bambi, a Life in the Woods</td><td>0.97</td><td>0.76</td><td>1.00 0.47</td><td></td><td>0.05</td><td>0.29 0.07</td></tr><tr><td>Bealby: A Holiday</td><td>0.96</td><td>1.00</td><td>0.92 0.59</td><td>0.13</td><td>0.20</td><td>0.20</td></tr><tr><td>Beauvallet</td><td>0.98</td><td>0.73</td><td>1.00 0.35</td><td>0.32</td><td>0.26</td><td>0.46</td></tr><tr><td>Candide</td><td>1.00</td><td>0.97</td><td>1.00 0.51</td><td>0.05</td><td>0.31</td><td>0.13</td></tr><tr><td>Captain Blood</td><td>0.85</td><td>0.45</td><td>1.00 0.33</td><td>0.27</td><td>0.21</td><td>0.21</td></tr><tr><td>Carmilla</td><td>0.90</td><td>0.94</td><td>1.00 0.53</td><td>0.08</td><td>0.32</td><td>0.06</td></tr><tr><td>Catriona</td><td>0.88</td><td>0.77</td><td>0.94 0.55</td><td>0.21</td><td>0.19</td><td>0.36</td></tr><tr><td>Clementina</td><td>0.77</td><td>0.81</td><td>0.86 0.57</td><td>0.24</td><td>0.21</td><td>0.57</td></tr><tr><td>David Copperfield</td><td>0.84</td><td>0.61 0.98</td><td>0.38</td><td>0.15</td><td>0.28</td><td>0.20</td></tr><tr><td>Deathworld</td><td>0.96</td><td>0.86 0.86</td><td>0.58</td><td>0.12</td><td>0.40</td><td>0.27</td></tr><tr><td>Demos</td><td>0.80</td><td>0.73 1.00</td><td>0.46</td><td>0.12</td><td>0.34</td><td>0.27</td></tr><tr><td>Dracula</td><td>0.88</td><td>0.93 0.97</td><td>0.54</td><td>0.08</td><td>0.32</td><td>0.28</td></tr><tr><td>Emma</td><td>0.88</td><td>0.67 1.00</td><td>0.55</td><td>0.10</td><td>0.25</td><td>0.18</td></tr><tr><td>Equality</td><td>0.72</td><td>0.71</td><td>1.00 0.49</td><td>0.16</td><td>0.24</td><td>0.49</td></tr><tr><td>Evelina</td><td>0.94</td><td>0.62 0.95</td><td>0.50</td><td>0.09</td><td>0.25</td><td>0.38</td></tr><tr><td>Fanshawe</td><td>0.98</td><td>0.90 0.96</td><td>0.62</td><td>0.14</td><td>0.32</td><td>0.11</td></tr><tr><td>Flatland: A Romance of Many Dimensions</td><td>0.99</td><td>0.59 1.08</td><td>0.71</td><td>0.19</td><td>0.24</td><td>0.13</td></tr><tr><td>Frankenstein; Or, The Modern Prometheus</td><td>0.74</td><td>0.71 1.00</td><td>0.66</td><td>0.21</td><td>0.32</td><td>0.10</td></tr><tr><td>Freckles</td><td>0.98</td><td>0.85</td><td>0.98 0.41</td><td>0.12</td><td>0.35</td><td>0.07</td></tr><tr><td>From the Earth to the Moon</td><td>0.96</td><td>0.71</td><td>1.00 0.50</td><td>0.07</td><td>0.21</td><td>0.31</td></tr><tr><td>Germinal</td><td>0.87</td><td>0.72 0.80</td><td>0.53</td><td>0.13</td><td>0.16</td><td>0.58</td></tr><tr><td>Green Mansions: A Romance of the Tropical Forest</td><td>0.97</td><td>0.96</td><td>1.00 0.59</td><td>0.10</td><td>0.34</td><td>0.15</td></tr><tr><td>Greenmantle</td><td>1.00</td><td>1.00</td><td>1.00 0.51</td><td>0.04</td><td>0.28</td><td>0.15</td></tr><tr><td>Gryll Grange</td><td>0.80</td><td>0.67</td><td>1.00 0.50</td><td>0.09</td><td>0.24</td><td>0.30</td></tr><tr><td>Heart of Darkness</td><td>1.00</td><td>1.00</td><td>1.00 0.62</td><td>0.22</td><td>0.31</td><td>0.00</td></tr><tr><td>Heidi</td><td>0.94</td><td>0.91</td><td>0.83 0.55</td><td>0.09</td><td>0.26</td><td>0.40</td></tr><tr><td>Herland</td><td>0.84</td><td>1.00 1.00</td><td>0.45</td><td>0.10</td><td>0.32</td><td>0.17</td></tr><tr><td>Howards End</td><td>0.98</td><td>0.73</td><td>1.00 0.50</td><td>0.07</td><td>0.23</td><td>0.24</td></tr><tr><td>Huntingtower</td><td>0.93</td><td>0.88</td><td>0.97 0.58</td><td>0.09</td><td>0.26</td><td>0.16</td></tr><tr><td>Indiana</td><td>0.89</td><td>0.71</td><td>1.00 0.55</td><td>0.13</td><td>0.25</td><td>0.24</td></tr><tr><td>Ivanhoe: A Romance</td><td>0.95</td><td>0.77</td><td>0.95 0.56</td><td>0.06</td><td>0.33</td><td>0.22</td></tr><tr><td>Jane Eyre</td><td>0.98</td><td>0.89</td><td>0.96 0.49</td><td>0.06</td><td>0.30</td><td>0.14</td></tr><tr><td>Jude the Obscure</td><td>0.92</td><td>0.77</td><td>1.00 0.55</td><td>0.12</td><td>0.23</td><td>0.28</td></tr><tr><td>Kazan</td><td>0.99</td><td>0.70</td><td>1.00 0.48</td><td>0.06</td><td>0.33</td><td>0.09</td></tr><tr><td>Kenilworth</td><td>1.00</td><td>0.73</td><td>0.94 0.56</td><td>0.12</td><td>0.22</td><td>0.26</td></tr><tr><td>Kidnapped</td><td>0.99</td><td>0.97</td><td>1.00 0.48</td><td>0.07</td><td>0.29</td><td>0.17</td></tr><tr><td>Kim</td><td>0.91</td><td>0.87</td><td>1.00 0.44</td><td>0.08</td><td>0.30</td><td>0.23</td></tr><tr><td>L'Assommoir</td><td>0.46</td><td>0.85</td><td>1.00 0.47</td><td>0.20</td><td>0.31</td><td>0.31</td></tr><tr><td>Lady Audley's Secret</td><td>0.92</td><td>0.73</td><td>0.97 0.45</td><td>0.06</td><td>0.26</td><td>0.17</td></tr><tr><td>Little Dorrit</td><td>0.84</td><td>0.63</td><td>0.99 0.55</td><td>0.15</td><td>0.40</td><td>0.22</td></tr><tr><td>Little Women</td><td>0.87</td><td>0.68</td><td>0.97 0.54</td><td>0.07</td><td>0.29</td><td>0.14</td></tr><tr><td>Madame Bovary</td><td>0.97</td><td>0.94</td><td>1.00 0.54</td><td>0.06</td><td>0.23</td><td>0.32</td></tr><tr><td>Maggie: A Girl of the Streets</td><td>0.98</td><td>0.89</td><td>1.00 0.44</td><td>0.06</td><td>0.20</td><td>0.20</td></tr><tr><td>Main Street</td><td>0.94</td><td>0.40</td><td>1.00 0.40</td><td>0.27</td><td>0.23</td><td>0.32</td></tr><tr><td>Mansfield Park</td><td>0.95</td><td>0.88</td><td>1.00 0.45</td><td>0.06</td><td>0.22</td><td>0.31</td></tr><tr><td>Marianela</td><td>0.91</td><td>0.64</td><td>0.95 0.57</td><td>0.11</td><td>0.24</td><td>0.23</td></tr></table>

<table><tr><td>Mary Barton</td><td>0.95</td><td>0.61</td><td>1.00 0.53</td><td></td><td>0.06</td><td>0.23</td><td>0.23</td></tr><tr><td>Metamorphosis</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.50</td><td>0.14</td><td>0.04</td><td>-0.00</td></tr><tr><td>Micromegas</td><td>0.98</td><td>1.00</td><td>0.97</td><td>0.53</td><td>0.10</td><td>0.15</td><td>0.10</td></tr><tr><td>Middlemarch</td><td>0.75</td><td>0.71</td><td>1.00</td><td>0.49</td><td>0.13</td><td>0.19</td><td>0.38</td></tr><tr><td>Mike</td><td>1.00</td><td>0.85</td><td>1.00</td><td>0.53</td><td>0.02</td><td>0.21</td><td>0.22</td></tr><tr><td>Moods</td><td>0.80</td><td>0.86</td><td>0.95</td><td>0.55</td><td>0.11</td><td>0.22</td><td>0.29</td></tr><tr><td>Moonfleet</td><td>1.00</td><td>0.89</td><td>1.00</td><td>0.51</td><td>0.13</td><td>0.39</td><td>0.11</td></tr><tr><td>My Ántonia</td><td>0.81</td><td>0.73</td><td>1.00</td><td>0.42</td><td>0.12</td><td>0.31</td><td>0.22</td></tr><tr><td>Ninety-Three</td><td>0.91</td><td>0.70</td><td>0.96</td><td>0.51</td><td>0.08</td><td>0.26</td><td>0.37</td></tr><tr><td>Northanger Abbey</td><td>0.98</td><td>0.74</td><td>1.00</td><td>0.61</td><td>0.13</td><td>0.30</td><td>0.15</td></tr><tr><td>Nostromo: A Tale of the Seaboard</td><td>0.72</td><td>0.45</td><td>0.85</td><td>0.67</td><td>0.15</td><td>0.35</td><td>0.34</td></tr><tr><td>O Pioneers!</td><td>0.92</td><td>0.83</td><td>0.98</td><td>0.48</td><td>0.08</td><td>0.43</td><td>0.11</td></tr><tr><td>Oblomov</td><td>0.75</td><td>0.61</td><td>0.83</td><td>0.55</td><td>0.12</td><td>0.18</td><td>0.25</td></tr><tr><td>Oil!</td><td>0.92</td><td>0.76</td><td>1.00</td><td>0.48</td><td>0.07</td><td>0.17</td><td>0.19</td></tr><tr><td>Oliver Twist</td><td>0.96</td><td>0.66</td><td>1.00</td><td>0.54</td><td>0.12</td><td>0.23</td><td>0.23</td></tr><tr><td>Persuasion</td><td>0.92</td><td>1.00</td><td>1.00</td><td>0.49</td><td>0.07</td><td>0.40</td><td>0.17</td></tr><tr><td>Pollyanna</td><td>0.83</td><td>0.69</td><td>0.79</td><td>0.46</td><td>0.11</td><td>0.23</td><td>0.54</td></tr><tr><td>Pride and Prejudice</td><td>0.98</td><td>0.61</td><td>1.00</td><td>0.52</td><td>0.06</td><td>0.29</td><td>0.13</td></tr><tr><td>Ramona</td><td>0.84</td><td>0.85</td><td>1.00</td><td>0.47</td><td>0.13</td><td>0.36</td><td>0.25</td></tr><tr><td>Rinkitink in Oz</td><td>0.97</td><td>0.96</td><td>1.00</td><td>0.50</td><td>0.05</td><td>0.28</td><td>0.15</td></tr><tr><td>Romola</td><td>0.82</td><td>0.70</td><td>0.98</td><td>0.53</td><td>0.07</td><td>0.26</td><td>0.28</td></tr><tr><td>Rookwood</td><td>0.71</td><td>0.57</td><td>1.00</td><td>0.46</td><td>0.17</td><td>0.20</td><td>0.33</td></tr><tr><td>Sandy</td><td>1.00</td><td>0.75</td><td>0.93</td><td>0.47</td><td>0.17</td><td>0.19</td><td>0.41</td></tr><tr><td>She</td><td>0.98</td><td>0.79</td><td>1.00</td><td>0.44</td><td>0.09</td><td>0.33</td><td>0.19</td></tr><tr><td>Shirley</td><td>0.91</td><td>0.78</td><td>0.97</td><td>0.53</td><td>0.06</td><td>0.39</td><td>0.22</td></tr><tr><td>Siddhartha</td><td>0.98</td><td>1.00</td><td>1.00</td><td>0.60</td><td>0.10</td><td>0.32</td><td>0.06</td></tr><tr><td>Sister Carrie: A Novel</td><td>0.95</td><td>0.77</td><td>0.98</td><td>0.54</td><td>0.03</td><td>0.25</td><td>0.22</td></tr><tr><td>Sons and Lovers</td><td>0.74</td><td>0.93</td><td>1.00</td><td>0.46</td><td>0.15</td><td>0.25</td><td>0.37</td></tr><tr><td>The Black Moth: A Romance of the XVIIIth Century</td><td>0.62</td><td>0.68</td><td>0.93</td><td>0.42</td><td>0.14</td><td>0.27</td><td>0.27</td></tr><tr><td>The Box-Car Children</td><td>0.92</td><td>0.94</td><td>0.86</td><td>0.54</td><td>0.05</td><td>0.28</td><td>0.22</td></tr><tr><td>The Cave Girl</td><td>0.98</td><td>1.00</td><td>0.97</td><td>0.51</td><td>0.03</td><td>0.24</td><td>0.09</td></tr><tr><td>The Deerslayer</td><td>1.00</td><td>0.31</td><td>1.00</td><td>0.57</td><td>0.13</td><td>0.17</td><td>0.17</td></tr><tr><td>The Documents in the Case</td><td>0.78</td><td>0.39</td><td>1.00</td><td>0.58</td><td>0.23</td><td>0.27</td><td>0.35</td></tr><tr><td>The Expedition of Humphry Clinker</td><td>0.10</td><td>0.79</td><td>1.00</td><td>0.49</td><td>0.35</td><td>0.41</td><td>0.51</td></tr><tr><td>The Further Adventures of Robinson Crusoe</td><td>0.97</td><td>0.88</td><td>0.97</td><td>0.53</td><td>0.07</td><td>0.31</td><td>0.11</td></tr><tr><td>The Garies and Their Friends</td><td>0.80</td><td>0.94</td><td>0.96</td><td>0.57</td><td>0.13</td><td>0.30</td><td>0.22</td></tr><tr><td>The Golovlyov Family The Good Soldier</td><td>0.95</td><td>0.69</td><td>0.98</td><td>0.35</td><td>0.19</td><td>0.36</td><td>0.17</td></tr><tr><td>The Hidden Staircase</td><td>0.68</td><td>0.84</td><td>0.97</td><td>0.61</td><td>0.20</td><td>0.36</td><td>0.27</td></tr><tr><td></td><td>0.94</td><td>0.92</td><td>1.00</td><td>0.44</td><td>0.08</td><td>0.28</td><td>0.23</td></tr><tr><td>The Hound of the Baskervilles</td><td>0.97</td><td>0.93</td><td>1.00</td><td>0.52</td><td>0.06</td><td>0.26</td><td>0.08</td></tr><tr><td>The House of Mirth</td><td>0.91</td><td>0.83</td><td>0.82</td><td>0.51</td><td>0.08</td><td>0.41</td><td>0.27</td></tr><tr><td>The House of the Seven Gables</td><td>0.95</td><td>0.86</td><td>1.00</td><td>0.53</td><td>0.13</td><td>0.35</td><td>0.16</td></tr><tr><td>The Invisible Man</td><td>0.96</td><td>0.86</td><td>1.00</td><td>0.54</td><td>0.10</td><td>0.28</td><td>0.18</td></tr><tr><td>The Jamesons</td><td>0.82</td><td>1.00</td><td>0.97</td><td>0.57</td><td>0.15</td><td>0.18</td><td>0.14</td></tr><tr><td>The Last of the Mohicans</td><td>0.96</td><td>0.85</td><td>1.00</td><td>0.59</td><td>0.09</td><td>0.29</td><td>0.13</td></tr><tr><td>The Maltese Falcon</td><td>0.95</td><td>0.75</td><td>0.95</td><td>0.58</td><td>0.11</td><td>0.28</td><td>0.11</td></tr><tr><td>The Marrow of Tradition</td><td>0.51</td><td>0.68</td><td>0.94</td><td>0.54</td><td>0.22</td><td>0.17</td><td>0.29</td></tr><tr><td>The Mill on the Floss</td><td>0.82</td><td>0.71</td><td>0.94</td><td>0.49</td><td>0.17</td><td>0.26</td><td>0.50</td></tr><tr><td>The Murder of Roger Ackroyd</td><td>0.89</td><td>0.56</td><td>1.00</td><td>0.42</td><td>0.15</td><td>0.36</td><td>0.10</td></tr><tr><td>The Mysteries of Udolpho</td><td>0.92</td><td>0.81</td><td>0.99</td><td>0.56</td><td>0.13</td><td>0.33</td><td>0.28</td></tr><tr><td>The Narrative of Arthur Gordon Pym of Nantucket</td><td>0.97</td><td>0.92</td><td>0.98</td><td>0.49</td><td>0.06</td><td>0.38</td><td>0.17</td></tr><tr><td>The Phantom of the Opera</td><td>0.92</td><td>0.71</td><td>1.00</td><td>0.60</td><td>0.13</td><td>0.42</td><td>0.22</td></tr><tr><td>The Red Badge of Courage: An Episode of the American Civil War</td><td>0.85</td><td>0.67</td><td>1.00</td><td>0.60</td><td>0.06</td><td>0.30</td><td>0.28</td></tr><tr><td>The Return of the Native</td><td>0.89</td><td>0.73</td><td>0.97</td><td>0.48</td><td>0.09</td><td>0.36</td><td>0.20</td></tr><tr><td>The Rise of Silas Lapham</td><td>0.71</td><td>0.85</td><td>0.92</td><td>0.47</td><td>0.15</td><td>0.29</td><td>0.38</td></tr><tr><td>The Scarlet Letter</td><td>0.93</td><td>0.92</td><td>0.95</td><td>0.46</td><td>0.15</td><td>0.29</td><td>0.27</td></tr><tr><td>The Secret Agent: A Simple Tale</td><td>0.92</td><td>1.00</td><td>0.98</td><td>0.46</td><td>0.07</td><td>0.34</td><td>0.07</td></tr><tr><td>The Seven Dials Mystery</td><td>0.87</td><td>0.59</td><td>0.79</td><td>0.52</td><td>0.12</td><td>0.26</td><td>0.40</td></tr><tr><td>The Sound and the Fury</td><td>0.52</td><td>1.00</td><td>0.59</td><td>0.53</td><td>0.23</td><td>0.17</td><td>0.47</td></tr><tr><td>The Three Musketeers</td><td>0.97</td><td>0.87</td><td>1.00</td><td>0.47</td><td>0.08</td><td>0.30</td><td>0.26</td></tr><tr><td>The Time Machine</td><td>0.83</td><td>0.94</td><td>0.97</td><td>0.48</td><td>0.07</td><td>0.22</td><td>0.12</td></tr><tr><td>The Triumph of the Scarlet Pimpernel</td><td>1.00</td><td>0.35</td><td>1.00</td><td>0.32</td><td>0.28</td><td>0.25</td><td>0.21</td></tr><tr><td>The Turn of the Screw</td><td>0.98</td><td>0.67</td><td>1.00</td><td>0.47</td><td>0.14</td><td>0.34</td><td>0.19</td></tr><tr><td>The Valley of Fear</td><td>0.88</td><td>0.93</td><td>1.00</td><td>0.49</td><td>0.11</td><td>0.35</td><td>0.24</td></tr><tr><td>The Valley of the Moon</td><td>0.70</td><td>0.52</td><td>0.95</td><td>0.57</td><td>0.17</td><td>0.21</td><td>0.40</td></tr><tr><td>The Vicar of Wakefield</td><td>0.77</td><td>0.66</td><td>1.00</td><td>0.47</td><td>0.17</td><td>0.27</td><td>0.29</td></tr><tr><td>The Voyages of Doctor Dolittle</td><td>0.99</td><td>0.69</td><td>1.00</td><td>0.53</td><td>0.07</td><td>0.18</td><td>0.23</td></tr><tr><td>The Way of All Flesh</td><td>0.92</td><td>0.63</td><td>1.00</td><td>0.59</td><td>0.18</td><td>0.17</td><td>0.45</td></tr><tr><td>The Wind in the Willows</td><td>0.83</td><td>0.92</td><td>0.97</td><td>0.39</td><td>0.16</td><td>0.31</td><td>0.09</td></tr><tr><td>The Wonderful Wizard of Oz</td><td>0.99</td><td>0.88</td><td>1.00</td><td>0.50</td><td>0.06</td><td>0.37</td><td>0.13</td></tr><tr><td>This Side of Paradise</td><td>0.99</td><td>0.90</td><td>0.97</td><td>0.67</td><td>0.17</td><td>0.36</td><td>0.06</td></tr><tr><td>Treasure Island</td><td>0.99</td><td>0.85</td><td>1.00</td><td>0.52</td><td>0.09</td><td>0.25</td><td>0.19</td></tr><tr><td>Trilby</td><td>0.89</td><td>0.88</td><td>0.88</td><td>0.60</td><td>0.13</td><td>0.32</td><td>0.16</td></tr><tr><td>Twenty Thousand Leagues under the Sea</td><td>0.87</td><td>0.74</td><td>0.97</td><td>0.54</td><td>0.16</td><td>0.27</td><td>0.32</td></tr><tr><td>Ulysses</td><td>0.97</td><td>1.00</td><td>0.99</td><td>0.56</td><td>0.06</td><td>0.30</td><td>0.01</td></tr><tr><td>Vanity Fair</td><td>0.85</td><td>0.81</td><td>1.00</td><td>0.55</td><td>0.12</td><td>0.31</td><td>0.30</td></tr><tr><td>Victory: An Island Tale</td><td>0.73</td><td>0.51</td><td>0.96</td><td>0.45</td><td>0.20</td><td>0.26</td><td>0.31</td></tr><tr><td>Vile Bodies</td><td>0.95</td><td>1.00</td><td>0.96</td><td>0.52</td><td>0.05</td><td>0.18</td><td>0.14</td></tr><tr><td>Villette</td><td>0.92</td><td>0.83</td><td>0.98</td><td>0.46</td><td>0.11</td><td>0.33</td><td>0.22</td></tr><tr><td>Virginia</td><td>0.91</td><td>0.76</td><td>0.96</td><td>0.58</td><td>0.12</td><td>0.22</td><td>0.22</td></tr><tr><td>We</td><td>0.75</td><td>0.75</td><td>1.00</td><td>0.48</td><td>0.13</td><td>0.32</td><td>0.31</td></tr><tr><td>What Maisie Knew</td><td>0.83</td><td>0.81</td><td>1.00</td><td>0.60</td><td>0.13</td><td>0.19</td><td>0.54</td></tr><tr><td>Where Angels Fear to Tread</td><td>0.95</td><td>1.00</td><td>1.00</td><td>0.49</td><td>0.07</td><td>0.21</td><td>0.13</td></tr><tr><td>White Fang</td><td>0.98</td><td>0.92</td><td>1.00</td><td>0.46</td><td>0.05</td><td>0.30</td><td>0.21</td></tr><tr><td>Wolfbane</td><td>0.91</td><td>0.80</td><td>0.81</td><td>0.55</td><td>0.14</td><td>0.40</td><td>0.29</td></tr><tr><td>Work: A Story of Experience</td><td>1.00</td><td>0.90</td><td>1.00</td><td>0.49</td><td>0.04</td><td>0.29</td><td>0.04</td></tr><tr><td>Wuthering Heights</td><td>0.97</td><td>0.88</td><td>0.98</td><td>0.45</td><td>0.12</td><td>0.31</td><td>0.22</td></tr></table>