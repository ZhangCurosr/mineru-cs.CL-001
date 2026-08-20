# Institutional Newspapers Pipeline: Deriving billions of high quality tokens from historical newspapers

Matteo Cargnelutti ᵃ, Catherine Brobston ᵃ, Eben English ᵇ, Jake Sadow ᵇ, Kacie Bailey ᵃ, Greg Leppert ✉ ᵃ, Amanda Watson ᶜ, Jessica Chapel ᵇ, Jonathan Zittrain ᵈ

ᵃ Institutional Data Initiative, Harvard Law School Library

ᵇ Boston Public Library

ᶜ Harvard Law School Library

ᵈ Harvard Law School, Harvard School of Engineering and Applied Sciences, Harvard Kennedy School

## Abstract

Historical newspapers are an abundant record of public life, but their dense, irregular and sometimes noisy layouts make computational access to these materials both challenging and limited. We present the Institutional Newspapers Pipeline, a modular system we jointly designed with Boston Public Library to extract high-quality, structured datasets from historical newspaper scans. It was architected so that each step remains interpretable and customizable, and so that the pipeline as a whole remains computationally frugal enough to run on workstation-level hardware. The pipeline runs each scan through a multi-step process: it segments scans into individual type-agnostic crops and performs OCR on each resulting segment before then performing text analysis, type classification, reading order detection, named entities recognition, subject classification, language detection, and pre-computed embeddings generation on every crop. We ran this pipeline against a portion of Boston Public Library's holdings and released the results as an open dataset. The optical character recognition (OCR) output represents 16.3 billion o200k\_base tokens across 83.1 million individual crops, extracted from 1,473,635 public domain newspaper scans published between 1795 and 1930. This report describes our methods for each processing step, the small models we trained, as well as the evaluation results and dataset-scale measurements we collected in the process. It accompanies the release of the pipeline, models, and dataset. We position this work as a substantial step towards unlocking high-quality data from tens of millions of newspaper scans.

1 Introduction..   
2 Contributions.   
3 Segmentation..   
3.1 Methodology.   
3.2 Results.   
4 Crop-level OCR..   
4.1 Methodology.   
4.2 Results. . 9   
5 Crop-level Language Detection and Text Analysis. 0   
5.1 Methodology. . 9   
5.2 Results. . 10   
6 Crop-type Classification.. ..11   
6.1 Methodology. ..11   
6.2 Results. . 13   
7 Scan-level Reading Order Detection. ..15   
7.1 Methodology. . 15   
7.2 Results.. . 16   
8 Crop-level Named Entity Recognition.. ... 17   
8.1 Methodology. . 17   
8.2 Results.. . 17   
9 Crop-level Subject Detection.. . 18   
9.1 Methodology. . 18   
9.2 Results.. . 18   
10 Crop-level Embeddings. ..20   
10.1 Methodology. . 20   
10.2 Results.. . 20   
11 Crop-level Detection of Chronicling America's Thesauri Terms. ..20   
12 Pipeline Footprint.. ..21   
12.1 Methodology. . 21   
12.2 Results. .. 21   
13 Dataset Preparation and Post-Processing.. . 22   
14 Discussion and Future Directions. . 23   
Rights determination.. .. 23   
Acknowledgements. . 24   
Disclaimers. ..24   
Reference list.. ..26   
Appendices.. .. 31   
Appendix A — Outer scan margins. . 31   
Appendix B — Crop-type auto-annotation prompt. ..31   
Appendix C — Crop-type classifier confidence. . 32   
Appendix D — Compute environment... .. 34   
Appendix E — Dataset field list.. . 34

## 1 Introduction

It is notoriously hard to extract data from historical newspapers, especially at scale (Clausner et al., 2015; Nikolaidou et al., 2022). Their layouts are dense and irregular, source scans vary widely in quality, and the text is set in columns, decorative headings, and tables that general-purpose tools do not handle well. At the same time, libraries across the world dedicate significant effort to preserve, digitize, and provide access to historical newspaper collections. The Library of Congress' Chronicling America program<sup>1</sup> is a prominent example, as are Europeana Newspapers<sup>2</sup> and the Bibliothèque nationale de France's Gallica press portal<sup>3</sup> in Europe.

Large language models (LLMs) require vast amounts of data in order to learn efficiently (Kaplan et al., 2020; Hoffmann et al., 2022). The nature and quality of what these models learn during pre-training influences their performance and behavior (Longpre et al., 2024). We posit that data derived from high-quality newspapers can significantly contribute to improving AI's “digital diet”<sup>4</sup>, and that this work can enable novel use cases beyond model training—from patron access to research. Open pre-training corpora such as The Pile (Gao et al., 2020) and Dolma (Soldaini et al., 2024) demonstrate the value of releasing diverse pre-training data and processing recipes to the research community. The Talkie-LM project (Levine et al., 2026), which trains LLMs on pre-1931 data to better understand the prediction capabilities of LLMs, is one example of the downstream impact such data can have. Talkie-LM was trained, in part, on Institutional Books: Harvard Library (Cargnelutti et al., 2025), which we released in 2025.

Entities, subjects, embeddings 173 person, 148 location, 47 organisation mentions · 8 of 12 leading subjects · 2 × 40 crop embeddings (text, image)

Reading order 40 crops sequenced 1 → 40

Crop-type classification

4 of 7 categories · image and text classifiers agree on 28 of 40

Crop-level OCR 1 Read twice: dots.mocr (Markdown), Tesseract (word level bboxes)

![](images/f30e6e75aedb308a74e82a2d5e988237aa34848314d8ca656c5de717d324be49.jpg)

Segmentation 40 crops· confidence 0.77-0.98

Newspaper scan 7,873 × 10,137 px

Figure 1. Output ofthe pipeline on thefrontpage of the Boston Evening Transcript of 28 August 1856. The pipeline's fifteen steps are grouped here into six layers, shown from the bottom up: the source scan; the 40 crops that the segmenter detects; the OCR text of each crop, shaded by token count; the type of each crop; the reading order as a path through the crops; and the entities, subjects, and embeddings that the enrichment steps add.

Furthermore, research into the historical events recorded in these materials has been hindered by the relative inaccessibility of their contents (Nguyen et al., 2021; Ehrmann et al., 2023). For many newspaper collections, the extracted text is of such low quality that it serves primarily as a loose and unreliable keyword search used to locate the original page scans from which a human can then read, rather than as an accurate representation of the underlying source which can be mined and analyzed using computational tools (Traub et al., 2015). Better extraction of content from newspapers has the potential to unlock new insights into our past at a scale only recently made possible by these emerging computational tools, including LLMs. And where LLMs have been trained on high-quality representations of historical materials, they show increased capabilities for analyzing them (Manjavacas and Fonteyn, 2022).

This newspaper processing pipeline was jointly designed by the Institutional Data Initiative and Boston Public Library. Our collaboration was guided throughout by Boston Public Library's focus on improving patron and researcher access through seeking an “atomic” understanding of individual items within its newspaper scans collection.

In designing this pipeline, we took inspiration from the work conducted by Lee et al. (2020) on the Newspaper Navigator dataset and Dell et al. (2023) with American Stories. Both explored using convolutional neural networks (CNNs) (LeCun et al., 1998) to segment and classify newspaper layout items.

While prior art proved how effective CNNs can be in this context, we chose to approach the problem from a different angle, informed by the unique challenges of the collection at hand and Boston Public Library's goals. We defined an atomic newspaper item as a “crop,” a standalone piece of content segmented from a newspaper scan. We separated segmentation from classification, both to better leverage our finite volume of training data and to treat classification as a distinct problem. We combined text and image classifiers for crop-type classification to disambiguate hard-to-classify items where text or visual signal alone is not sufficient. More generally, we kept each step of the processing pipeline separate, so that both the Institutional Data Initiative and Boston Public Library teams could individually evaluate, interpret, and steer them, and so that each step remained customizable for other collections. Throughout, we balanced the need for high-quality data from complex objects against computational frugality (Vanderbauwhede, 2023), favoring small, efficient methods and models that keep the pipeline within reach of workstation-grade hardware.

This technical report describes the pipeline we created, the methods we developed, and the models we trained, as well as the dataset we generated out of 1,473,635 newspaper scans from Boston Public Library's collection. This report's figures focus on the pre-1931 subset of Boston Public Library's collection, which we released as part of our dataset. We position this work as an initial step towards unlocking high-quality data from tens of millions of newspaper scans. This release is the first iteration in this process.

Figure 1 shows a simplified view of what the pipeline produces on a single scan from the collection. Each layer holds the data that one group of steps adds to that same page. The sections that follow describe these steps in the order the pipeline runs them.

## 2 Contributions

With this technical report, we introduce this initial set of contributions:

A dataset produced from our ongoing collaboration with Boston Public Library on its public domain newspapers collection. For each scan, we provide all of the data we collected and generated at crop level: bounding boxes, crop types, OCR outputs from both Tesseract and a specialized VLM, reading order indexes, NER entities, subject classes, as well as text and language analysis metrics. The VLM OCR output represents 16.3 billion o200k\_base (OpenAI, 2024) text tokens for 83.1 million individual crops, extracted from 1,473,635 individual scans.

● A production-ready pipeline that knowledge institutions can use on their own newspaper collections.

● A series of small and efficient detection and classification models for processing large-scale historical newspaper collections. This report describes the training and evaluation processes of each model. Additional information can be found in each model's Hugging Face repository.

All of these contributions are available:

On Hugging Face: https://huggingface.co/collections/institutional/institutional-newspapers

On GitHub: https://github.com/institutional/institutional-newspapers-pipeline

An Agent Skill file<sup>5</sup> is also available to enable agentic use of this dataset and the accompanying models.

## 3 Segmentation

## 3.1 Methodology

As a first step towards reaching an “atomic” understanding of newspaper scans, we set a goal of segmenting pages into individual crops. In the context of this project, a crop is a rectangular bounding box around an uninterrupted flow of text or visual elements. As a result, headlines are not separated from the article they anchor, and visual-heavy advertisements are generally self-contained within a single crop.

Taking inspiration from prior art, we hypothesized that separating crop segmentation from classification could help us better leverage a limited number of annotations. The segmentation model needed to learn from—and run inference against—full scans. As content and advertisement crops represent the vast majority of crops on each scan (see Section 6), training a combined detection-plus-classification model on full scans would have led to significant class imbalance, likely degrading both detection and classification accuracy. This concern comes in addition to a classification challenge we had previously identified in which certain crops could only be reliably classified using a combination of text and image signal (see Section 6). Separately, but for similar reasons, we assessed that an object detection model would likely perform better at that task than a segmentation model. The precision of the masks object detection models try to generate can conflict with the dense, noisy nature of newspaper scans.

We chose YOLO26x (Jocher et al., 2026) as our base model. While it is the largest model of the YOLO26 series at 55.7M parameters, it remains a fairly small CNN suitable for inference at scale.

In order to train and evaluate this model, we annotated 1,020 randomly sampled newspaper scans from Boston Public Library's collection using the VGG Image Annotator (VIA) (Dutta and Zisserman, 2019), for a total of 47,485 individual bounding box annotations. 210 of these scans were fully annotated manually, while 810 were pre-annotated using intermediary models trained on all available annotations. All machine-generated annotations were manually reviewed and edited. Of the 47,485 boxes in our annotations set, 9,028 were drawn manually and 38,457 were pre-annotated by our intermediary models. This process allowed us to iteratively estimate how many annotated scans we needed in order to train an effective object detection model, and to progressively focus our attention on patterns the model struggled with.

The final model was trained and evaluated on 1,020 scans, with 15% of them set aside for validation and another 15% for evaluation. To balance computational expense and accuracy, we chose to train our model with scans at a resolution of 960 pixels<sup>6</sup>, after iterating over resolutions ranging from 320px to 1280px. At inference time, we discard detections with a confidence score below 0.6 and apply non-maximum suppression with an intersection-over-union (IoU) threshold of 0.15. The confidence threshold removes low-certainty detections, while the deliberately low IoU threshold suppresses heavily overlapping boxes.

<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Test scans</td><td>153</td></tr><tr><td>Test instances (boxes)</td><td>2,991</td></tr><tr><td>Precision</td><td>0.927</td></tr><tr><td>Recall</td><td>0.910</td></tr><tr><td>F1</td><td>0.918</td></tr><tr><td>mAP@50</td><td>0.955</td></tr><tr><td>mAP@50-95</td><td>0.901</td></tr></table>

Table 1. Segmentation model evaluation on the held-out test set.

![](images/d7d53cf552149a0d868a8acacf48933927e2f05a0867709f10fb5ec80ff0ad51.jpg)  
Figure 2. Eigen-CAM heatmap for the segmentation model, overlaid on a samplefrontpage (North Brookfield Journal, May 22 1880). Computed at backbone layer index 8 (a C3k2 block) at an inference size of960 pixels. Activation concentrates on the masthead and headline banner and traces the column structure ofthe page.

Our evaluation set of 153 held-out scans (2,991 bounding-box instances) suggests that the model effectively segments this collection. Table 1 summarizes these results.

To help interpret the model's behavior, we generated Eigen-CAM heatmaps (Muhammad and Yeasin, 2020; rigvedrs, 2026) from the trained segmenter. Eigen-CAM projects the activations of a chosen layer onto their first principal component which highlights the regions that contribute most to the model's response.

Figure 2 overlays such a heatmap on a sample front page, computed at index 8 of the model's backbone (out of 11), the deepest block whose receptive field is still purely convolutional. The strongest activation follows the masthead and the headline banner across the top of the page. Secondary activation traces the vertical column structure and the denser advertisement block in the left margin. This suggests that the model learned to respond to the structural cues of a newspaper page (the layout boundaries between columns and blocks) rather than to incidental features of the scan.

## 3.2 Results

Out of 1,473,635 scans, the model identified 83,147,041 individual crops, for an average of 56.4 crops per scan. Model confidence is highly concentrated at the top of the range, around 0.95 (Figure 3). This is consistent with the conservative confidence threshold applied at inference time.

![](images/14b76999e2d64a54b286149375112857ccc6750d83f2937ead25a620f206b64e.jpg)  
Figure 3. Distribution of crop detection confidence scores across all detected crops.

![](images/82c806aabbb6a0a36f601a33afe893d5baf567871e888378336cd421ec1713c4.jpg)  
Figure 4. Average number of crops per scan, by decade.

In this collection, mid-19th-century newspaper scans appear to be the most dense in terms of average number of crops per scan. While it may be tempting to draw conclusions about the nature of these scans and the way newspaper layouts evolved, further analysis would be needed to clearly identify the cause (Figure 4). The data this pipeline produces may help digital humanities researchers investigate such questions.

Because our pipeline filters detections with low confidence scores and uses IoU to eliminate overlapping boxes, an average 89% of each scan is covered by crop detections (Figure 5). That figure suggests both that our model may have missed certain crops, and also that it did not generate crops for margins. Measuring the outer margins of a sample of scans supports the second explanation: the outermost crops leave an uncovered border of 2.6% to 3.6% of the scan on each side, which appears to account for most of the uncovered area (Appendix A).

![](images/faca93a75e39b2903378f58a9db70e05a27f4456c2bd26875e3cc7e4d4ba647c.jpg)  
Figure 5. Distribution of the fraction of scan area covered by detected crops.

## 4 Crop-level OCR

## 4.1 Methodology

OCR is one of the key challenges involved in extracting high-quality data from newspaper scans (Nguyen et al., 2021). During our initial tests, we observed that traditional OCR engines often failed to properly detect columns and parse large, decorative text elements. OCR-capable vision-language models (VLMs), which often produce higher-quality extractions than traditional open-source tools (Poznanski et al., 2025) yielded promising results out of the box when run against full scans, but accuracy and performance were inconsistent and seemed to drop significantly on complex or dense scans. This observation seems consistent with prior work. Layout complexity and line count are among the strongest predictors of VLM transcription error on historical pages, particularly for smaller and weaker models (Levchenko, 2025), page-level evaluation on historical corpora leaves even frontier models well short of production accuracy (Semnani et al., 2025), and on 19th-century newsprint, passing large text regions to a VLM without splitting them into smaller, rectangular crops induces degenerate repetition loops that push mean character error rates far above the median (Bourne, 2025). Reproducibility, operational control and cost were also a concern. Using frontier models, the cost of the 16 billion tokens we needed to generate would have ranged approximately from \$250,000 to \$800,000<sup>7</sup>, assuming a cost per million output tokens ranging from \$15 to \$50. Instead, operating at the crop level with local models and techniques let us assemble a reproducible system we could deploy and run at a fraction of this projected cost (see Section 12.2), with granular control over the system's input, output and behavior.

We therefore chose to focus our efforts on crop-level OCR, and to do so with both Tesseract and a specialized VLM.

We chose to use Tesseract 5 (Smith, 2007; Tesseract OCR contributors, 2024a) and the tessdata\_best models (Tesseract OCR contributors, 2024b). The failure modes of “traditional” OCR engines are well understood (Rice et al., 1999), and, contrary to VLMs, “hallucination” rarely occurs at the level of a full word or sentence (Yang et al., 2025). In addition, the word-level bounding boxes Tesseract produces can be used to compile OCR outputs compatible with existing library discovery systems (e.g, AltoXML<sup>8</sup>).

For the VLM portion of this apparatus, we tested multiple specialized VLMs for OCR and chose dots.mocr (Zheng et al., 2026) after reviewing samples. dots.mocr is only 3B parameters, offers multilingual support and, importantly, can parse tables. Our initial tests were run with dots.ocr (Li et al., 2025), which we replaced with dots.mocr shortly after it was released. This model is not immune to common pitfalls of VLM-based OCR including hallucinations (Ji et al., 2023) and token loops; in the latter, the model repeats the same token ad infinitum (Yang et al., 2025; Holtzman et al., 2020). It is worth noting that some of these “hallucinations” can be helpful, for example when transcribing damaged text (see Figure 6).

<table><tr><td colspan="2">TORAGESt age-Rooms may be tound on applicationat 1 Advertiser Office.</td></tr><tr><td>TOR \GE Ste age Revs s may he found on - applicationat I %e Advertiser</td><td>STORAGE Storage-Rooms may be found on application at The Advertiser</td></tr></table>

Figure 6. Example ofa helpful “hallucination”, in which the VLM was able to recover missing letters from a damaged scan. The same mechanism can lead to transcription errors.

We used the tesserocr Python library (sirfz, 2024) to run Tesseract against individual newspaper crops. This particular library allowed us to leverage multiprocessing to scale operations up to the entire collection. In order to balance accuracy with computational footprint, and after multiple rounds of quality testing, we chose to scale the image size of each crop down to a maximum of 1.5 megapixels before passing it to Tesseract.

Optimizing for inference with dots.mocr proved challenging and required making important decisions in advance. We ran inference using vLLM (Kwon et al., 2023) against 8 NVIDIA L40S GPUs, with each GPU running its own vLLM process.

We clamped image size for each crop to between 0.25 and 1 megapixel which appeared to offer an appropriate accuracy-versus-performance trade-off in that specific scenario. We also made aggressive use of vLLM's batching and caching capabilities and implemented a double-buffering mechanism in which the next batch is being prepared while vLLM runs inference on the current batch.

## 4.2 Results

We counted OCR tokens with the o200k\_base tokenizer to give a model-relevant measure of text volume. Across the collection, dots.mocr produced 16.3 billion tokens and Tesseract 14.7 billion. With few exceptions, dots.mocr consistently produced more tokens than Tesseract, both in total (Figure 7) and per scan (Figure 8), which is consistent with its ability to recover text from dense crops, decorative elements, and damaged scans that Tesseract tends to struggle with.

![](images/ef13f2b11c78299b36fe976ecf627a2de03d46d8b7bcba6862393cca1b09a387.jpg)  
Figure 7. Total o200k\_base tokens over time (by decade), comparing Tesseract and dots.mocr.

![](images/820f3809a3ab4b2cabf941d99b4b978a85c64e442371eb52cececec39c2af039.jpg)  
Figure 8. o200k\_base tokens per scan over time (by decade), comparing Tesseract and dots.mocr.

## 5 Crop-level Language Detection and Text Analysis

## 5.1 Methodology

## 5.1.1 Text analysis

In order to measure and compare the performance of the two OCR techniques we used, we collected the following metrics for the OCR text of every crop, for both Tesseract and dots.mocr:

● Character count.

● Word and sentence count, both total and unique, split with PyICU (PyICU contributors, 2024).

● A “tokenizability” score as a measure of how efficiently the extracted text is represented by the tokenizer.

● For dots.mocr output only: whether the text contains Markdown markers and/or a table.

We “flattened” the text of each crop before measuring it, so that the two OCR outputs could be compared on equal terms. Flattening strips HTML markup, rejoins words that were split by a hyphen at the end of a line, and replaces line breaks with spaces. We then removed zero-width spaces and split the result into words and sentences using PyICU, with the detected language of the crop as a processing hint a fallback to English when no language was available. The “tokenizability” score compares the number of words to the number of tokens those same words produce, on the basis that one word tokenizes into roughly 1.25 tokens for English text. This score can be used as a very coarse measure of text validity, though the presence of unusual vocabulary on which the tokenizer might not have been trained may negatively impact the score. Finally, we detect Markdown markers and tables on the raw VLM output rather than the flattened text, because flattening removes the very markers for which we look.

These metrics were also designed to offer continuity with the Institutional Books pipeline, with the goal of helping users make informed decisions when using the data.

## 5.1.2 Crop-level Language Detection

We chose to use Lingua (Stahl, 2024) to perform language detection at crop level, in order to help identify multilingual newspapers.

We made this choice based on the following criteria:

● Newspaper-derived text varies in length widely and Lingua is designed to remain accurate on short text.

We already had access to issue-level language metadata derived from Chronicling America (Library of Congress and National Endowment for the Humanities, 2007) for this collection. We could therefore substitute issue-level language metadata for low-confidence detections or detections performed on very short text (see Section 13).

● We wanted to limit computational expense for a data point for which we already had an issue-level reference, which is also why we chose to run language detection only on VLM-extracted text.

## 5.2 Results

A language code was assigned to 99.81% of all crops (82,988,421 crops). Lingua detected 96.96% of these codes directly. The remaining 3.04% were inherited from issue-level metadata, under the post-processing rules described in Section 13. English is by far the most prevalent language of the collection, accounting for 97.61% of crops with a language code.

The pipeline surfaces a meaningful multilingual tail led by Yiddish, German, Swedish, and French, reflecting the immigrant press held in Boston Public Library's collection (Table 2). The collection covers 73 distinct language codes in total, though the distribution drops off sharply: the ten most common account for 99.97% of crops with a code, and the remaining 63 for 0.03% between them.

<table><tr><td>Language</td><td>Crops</td><td>% of crops with a language code</td></tr><tr><td>English (ENG)</td><td>81,002,114</td><td>97.61%</td></tr><tr><td>Yiddish (YID)</td><td>983,375</td><td>1.18%</td></tr><tr><td>German (DEU)</td><td>467,929</td><td>0.56%</td></tr><tr><td>Swedish (SWE)</td><td>380,078</td><td>0.46%</td></tr><tr><td>French (FRA)</td><td>76,217</td><td>0.09%</td></tr></table>

Table 2. Top five language codes for dots.mocr (VLM) OCR text,  
as a share ofcrops with a language code.

Comparing the two OCR methods on the same crops (Table 3), dots.mocr recovers more characters, words, and sentences than Tesseract and reaches a slightly higher mean tokenizability score. Tesseract reports more unique words, which is consistent with its higher rate of character-level noise producing spurious tokens.

dots.mocr also emits a measurable amount of structured output: 0.78% of crops contain a Markdown table and 3.33% contain other Markdown markup. Based on our observations, both of these metrics suggest that dots.mocr and our post-processing (Section 13) failed to format a non-trivial but hard-to-measure amount of tables and headings.
<table><tr><td>Metric</td><td>Tesseract</td><td>dots.mocr (VLM)</td></tr><tr><td>o200k_base tokens</td><td>14,661,019,181</td><td>16,302,004,429</td></tr><tr><td>Characters</td><td>50,124,785,558</td><td>57,428,175,900</td></tr><tr><td>Mean words per crop</td><td>106.4</td><td>115.3</td></tr><tr><td>Mean unique words per crop</td><td>68.8</td><td>64.3</td></tr><tr><td>Mean sentences per crop</td><td>7.7</td><td>9.6</td></tr><tr><td>Mean unique sentences per crop</td><td>7.5</td><td>8.6</td></tr><tr><td>Mean “tokenizability&quot; score</td><td>86.85</td><td>87.35</td></tr><tr><td>Crops with a Markdown table</td><td>N/A</td><td>647,301 (0.78%)</td></tr><tr><td>Crops with Markdown markup</td><td>N/A</td><td>2,772,880 (3.33%)</td></tr></table>

Table 3. Text-metrics comparison between Tesseract and dots.mocr across all crops. Means are measured over the crops that carried textfor that engine, 81,133,206 for Tesseract and 82,985,118 for dots.mocr, out of 83,147,041 crops.

## 6 Crop-type Classification

## 6.1 Methodology

Because we separated segmentation from classification, we were able to devise a method of crop classification that leverages both image and text signals at crop-level in order to help make a more accurate and interpretable prediction.

We therefore chose to train two small classifiers:

● An image classifier based on YOLO26m-cls (11.6M parameters).

● A static embedding model fine-tuned as a classifier. Specifically, we used Model2Vec (Tulkens and van Dongen, 2024) to fine-tune potion-base-32M (MinishLab, 2024).

For both models, we focused on broad categories (for example “Content”, “Advertisement”, “Photograph or illustration”) to limit the effect of class imbalance.

To prepare the training set, we used the FP8 variant of Qwen/Qwen3-VL-30B-A3B-Thinking (Qwen Team, 2025b), a vision-language model from the Qwen3 series (Qwen Team, 2025a), to auto-annotate a large set of randomly sampled crops. We manually reviewed sampled annotations to refine our auto-annotation prompt (Appendix B) and to confirm that classifiers could be trained on this data. Our final manual check was performed on 600 annotations, yielding a 91% accuracy score for strict checks (fully correct only) and a 94.50% accuracy score for standard checks (fully correct plus ambiguous results combined).

We assembled training sets for our classification models by randomly sampling rows from our collection of auto-annotated crops. In total, our image classifier was trained on 188,477 annotations, and our text classifier on 185,900 annotations, both with a 70/15/15 split as a target. Although the annotations were pulled from the same pool, the text classifier received 2,577 fewer annotations than the image classifier. 1,740 of these were crops without text (automatically marked as “Empty” for text classification purposes), while the remaining 837 were “Empty” crops we manually augmented to train the image classifier by rotating them at 90°, −90° and 180° to help reduce the effect of extreme class imbalance for this category.

The image classifier was trained on crop images resized to 768 pixels, while the text classifier was trained on VLM-extracted OCR text (dots.ocr)<sup>9</sup>. The image classifier was trained on seven distinct classes, including an “Empty” class, while the text classifier was trained on six. The text classifier has no “Empty” class because a crop without OCR-able text is treated as “Empty” by convention rather than predicted as such. Table 4 compares the per-class evaluation results of the two classifiers.

The models have significant overlap and clear differences. They agree closely on text-heavy categories such as Advertisement, Content, and Section heading. The image classifier is far stronger on visual categories. For “Photograph or illustration” it reaches an F1 of 0.84 against the text classifier's 0.48, and for “Cartoon” 0.92 against 0.68. This behavior matches our initial expectations since these categories carry little OCR-able text.

<table><tr><td>Category</td><td>Image P</td><td>Image R</td><td>Image F1</td><td>Text P</td><td>Text R</td><td>Text F1</td></tr><tr><td>Advertisement</td><td>0.91</td><td>0.93</td><td>0.92</td><td>0.95</td><td>0.94</td><td>0.95</td></tr><tr><td>Content</td><td>0.93</td><td>0.90</td><td>0.92</td><td>0.90</td><td>0.96</td><td>0.93</td></tr><tr><td>Section heading</td><td>0.87</td><td>0.97</td><td>0.92</td><td>0.94</td><td>0.93</td><td>0.94</td></tr><tr><td>Masthead, nameplate or running head</td><td>0.98</td><td>0.79</td><td>0.87</td><td>0.93</td><td>0.86</td><td>0.89</td></tr><tr><td>Photograph or illustration</td><td>0.83</td><td>0.84</td><td>0.84</td><td>0.63</td><td>0.38</td><td>0.48</td></tr><tr><td>Cartoon</td><td>0.90</td><td>0.94</td><td>0.92</td><td>0.85</td><td>0.57</td><td>0.68</td></tr><tr><td>Empty</td><td>0.72</td><td>0.95</td><td>0.82</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Overall accuracy</td><td>0.913 (Top-1)</td><td></td><td></td><td>0.92</td><td></td><td></td></tr></table>

Table 4. Per-class evaluation ofthe image and text crop-type classifiers.  
“Empty” is marked N/A for the text classifier, which does not model that class.

In order to decide on the most likely category for any given crop, we combined both signals. The filtering mechanism we implemented makes decisions as follows:

● By default, priority is given to the classification with the highest confidence score.

● If the image classifier returns “Photograph or illustration”, priority is given to the image classifier.

● If the image classifier returns “Empty”, priority is given to the image classifier.

● If a crop’s text is flagged as “Empty”, priority is given to the image classifier.

We provide raw data from both the image and text classifiers as part of our dataset, so end users can apply a different filtering mechanism.

## 6.2 Results

Across the 83,147,041 crops we classified, the two classifiers agree on 90.55% of crops (Table 5). The combined decision follows the image classifier on 95.03% of crops and the text classifier on 95.51%. The two classifiers overlap heavily but neither reproduces the final decision alone, which confirms the relevance of combining both signals.

Both classifiers report high confidence on average: 0.953 for the image classifier and 0.967 for the text classifier, weighted across all crops. Confidence varies by category and is markedly lower for the visual categories both models handle least well. Appendix C reports the per-category figures.

<table><tr><td>Measure</td><td>Crops</td><td>% of classified</td></tr><tr><td>Total classified crops</td><td>83,147,041</td><td>N/A</td></tr><tr><td>Image and text agree</td><td>75,287,373</td><td>90.55%</td></tr><tr><td>Final category = image classifier</td><td>79,018,344</td><td>95.03%</td></tr><tr><td>Final category = text classifier</td><td>79,416,070</td><td>95.51%</td></tr></table>

Table 5. Crop-type classification agreement and the source of the final decision.

By final category, advertisements are by far the most common crop type, followed by content (Figure 9). Conversely, content crops carry the majority of the OCR tokens, well ahead of advertisements (Figure 10). This pattern holds over time. Although advertisement crops outnumber content crops in most decades, content still accounts for the majority of tokens in nearly all of them (Figures 11 and 12).

![](images/fbcd5b991f6e92d9492bdb9b8de978979807dec82ac618bf0c393a082cd54c78.jpg)  
Figure 9. Distribution of crops by final crop-type category.

![](images/d671f87fc9caaaef3610c3590d92ea7f0b94a18f19536f0011fb56fc20eccb59.jpg)  
Figure 10. Total dots.mocr o200k\_base tokens by final crop-type category.

![](images/699a6173a0efc3d4ded0ff2e4277ebc9bcc511f30304821a7f6608150b8062cc.jpg)  
Figure 11. Share of crops by final category over time (relative %, by decade).

![](images/93cad28e82cb725744a9a8aca5d353d9ae2070aa9af3cea15607c59c1f5352eb.jpg)  
Figure 12. Share ofdots.mocr tokens byfinal category over time (relative %, by decade).

Looking across decades at the total scan area each category covers (Figure 13) shows how the relationship between crop surface area and token density evolves in this collection. Until the 1860s, advertisements

took up roughly the same share of the page as they contributed in text tokens. From the 1870s onward, advertisements hold meaningfully more surface area than they contribute in text tokens, and the gap widens to about 15 percentage points by the 1920s. This is consistent with advertisements becoming progressively more visual over the period.

![](images/001a091366c55489502ad9b8c2dc0e9c0a8192b75e5918c3e9374af2c73f2ca3.jpg)  
Figure 13. Share of crop area by final category over time (relative %, by decade).

## 7 Scan-level Reading Order Detection

## 7.1 Methodology

Inferring the natural reading order of layout elements is known to be challenging (Wang et al., 2021). Our newspapers processing pipeline works at the level of crops. On any given page, only a subset of crops are directly connected to each other (for example, different parts of an article spread across different columns or pages), which arguably makes this problem more difficult to solve.

Newspaper layouts vary widely, not only over time but also from one title to the next. We identified early in our prototyping that applying a purely rule-based reading order detection method would not yield satisfactory results. Projects such as LayoutReader (Wang et al., 2021) explored using transformer-based models to perform reading order detection. While that approach yielded promising results, it presented two main issues for our use case. The first was that using a transformer-based model for this task at the scale of a library collection represents a significant computational expense. The second was that LayoutReader and similar models tend to operate at the level of spans, not crops.

We implemented a fast reading-order detection mechanism based on HDBSCAN (Campello et al., 2013; McInnes et al., 2017) combined with content-aware post-processing:

1. Classify crops as narrow or wide based on estimated column width relative to scan dimensions.

2. Cluster narrow “Content” crops into columns (HDBSCAN on x-centers).

3. Assign narrow non-content crops to the nearest column (by x-center distance).

4. Assign wide crops to the overlapping column with the most content above.

5. Order left-to-right by column, then top-to-bottom within columns.

6. Post-process via left-edge bucketing for final left-to-right, top-to-bottom “nudging”. Visual elements (photographs, cartoons) are sorted by their top edge rather than their y-center, so they appear before the content they illustrate.

To measure the accuracy of this reading-order detection method at page-level, we annotated 723 scans to establish an evaluation set. These reference scans were pre-annotated with an earlier version of this algorithm and corrected manually.

## 7.2 Results

Position accuracy is the share of crops the method places at exactly the right index in the sequence. We report it in two ways (Table 6). The macro figure scores each scan separately and then averages the scores without weighting them. The micro figure pools all crops and scores them together.

Measured against our evaluation set, the method reaches 72.1% macro and 80.8% micro position accuracy, with a Kendall's τ of 0.922 and 0.959 respectively.

<table><tr><td>Crop type</td><td>Crops</td><td>Position accuracy (macro)</td><td>Position accuracy (micro)</td><td>Kendall&#x27;s τ (macro)</td><td>Kendall&#x27;s τ (micro)</td></tr><tr><td>Content</td><td>14,859</td><td>0.716</td><td>0.744</td><td>0.970</td><td>0.974</td></tr><tr><td>Advertisement</td><td>21,630</td><td>0.671</td><td>0.849</td><td>0.820</td><td>0.955</td></tr><tr><td>Section heading</td><td>1,998</td><td>0.811</td><td>0.849</td><td>0.975</td><td>0.985</td></tr><tr><td>Masthead, nameplate 872 or running head</td><td></td><td>0.935</td><td>0.917</td><td>0.932</td><td>0.938</td></tr><tr><td>Photograph or illustration</td><td>142</td><td>0.494</td><td>0.486</td><td>0.769</td><td>0.803</td></tr><tr><td>Cartoon</td><td>117</td><td>0.558</td><td>0.650</td><td>0.984</td><td>0.977</td></tr><tr><td>Unknown</td><td>939</td><td>0.629</td><td>0.765</td><td>0.853</td><td>0.936</td></tr><tr><td>Empty</td><td>12</td><td>0.667</td><td>0.667</td><td>N/A</td><td>N/A</td></tr><tr><td>Overall</td><td>40,569</td><td>0.721</td><td>0.808</td><td>0.922</td><td>0.959</td></tr></table>

Table 6. Reading-order evaluation: position accuracy and Kendall's τ, overall and per crop type. Macro figures average per-scan scores; micro figures are crop-weighted across all crops, across scans.

We hypothesize that, in this context, the Kendall's τ metric is meaningful. While position accuracy assesses whether the predicted reading order matches our annotations exactly, the Kendall's τ figures suggest that the overall sequencing matches our annotations, even when exact positions differ.

While this method is overall effective and has a limited computational footprint, it cannot, by nature, properly handle deeply complex or unique layouts. We posit that these layouts would benefit from a transformer-based approach, and we envision this as a future direction for this project.

## 8 Crop-level Named Entity Recognition

## 8.1 Methodology

We chose to add a named entity recognition (NER) step to our pipeline in keeping with Boston Public Library's goal of achieving an “atomic” understanding of newspaper scans, as granular NER data can help enable new use cases for newspapers data (Ehrmann et al., 2023).

We consider this output to be experimental. Developing a NER method specific to this collection was out of scope for this initial pipeline development project and we therefore chose to use an “off-the-shelf” solution. We ran Flair’s base ner-fast model (Akbik et al., 2019) against the VLM-extracted text of each crop.

We applied the following filtering rules, designed through iterative testing:

● We discarded detections with a confidence score below 0.85.

● We only kept results for PER (persons), LOC (locations), and ORG (organizations) entities.

● We deduplicated detections and, for each group of likely duplicates, kept the detection with the highest confidence score.

## 8.2 Results

Across the collection, the pipeline detected 155.6 million location mentions, 142.2 million person mentions, and 49.1 million organization mentions (Table 7). Locations have the fewest unique surface forms relative to their volume, which is consistent with a small set of place names recurring across the collection.

The most frequent entities (Table 8) appear coherent with the nature of the collection: the locations are dominated by Boston and the surrounding region (“Boston”, “Mass”) alongside the major centers of national news (“New York”, “Washington”, “United States”), the organizations are dominated by institutions of government (“Congress”, “Senate”, “House”), and the persons are common family names of the period.

<table><tr><td>Entity type</td><td>Total detections</td><td>Unique surface forms</td></tr><tr><td>PER</td><td>142,245,435</td><td>381,205</td></tr><tr><td>LOC</td><td>155,570,972</td><td>298,058</td></tr><tr><td>ORG</td><td>49,059,445</td><td>380,115</td></tr></table>

<table><tr><td>Rank</td><td>PER</td><td>LOC</td><td>ORG</td></tr><tr><td>1</td><td>Smith (612,147)</td><td>Boston (8,661,331)</td><td>Congress (600,270)</td></tr><tr><td>2</td><td>Brown (478,864)</td><td>New York (4,159,692)</td><td>WM (455,265)</td></tr><tr><td>3</td><td>God</td><td>Washington</td><td>Senate</td></tr><tr><td></td><td>(389,782)</td><td>(2,966,122)</td><td>(427,325)</td></tr><tr><td>4</td><td>Clinton</td><td>Mass</td><td>House</td></tr><tr><td></td><td>(314,286)</td><td>(2,712,257)</td><td>(385,912)</td></tr><tr><td>5</td><td>Johnson</td><td>United States</td><td>Union</td></tr></table>

Table 7. Named-entity totals and unique surface forms.  
Table 8. Top five entities by frequency, per entity type.

This NER technique has limitations. Names may be out of distribution, in particular in historical contexts. OCR quality is another bottleneck. The model may also identify derogatory terms or other harmful language as names on occasion. For these reasons, while this data point can be used to assist with downstream research, it is experimental and should not be taken at face value.

## 9 Crop-level Subject Detection

## 9.1 Methodology

We decided to undertake crop-level subject detection as a way to provide additional signal on the nature of each crop. We consider this output experimental, and we focused mainly on establishing a process for performing such classification at this scale. The list of subjects we picked is based on our observations for this specific collection, and we believe it needs to be adjusted for specific research needs. We designed this section of the pipeline so it can be re-run as needed with a different set of subjects.

We ran MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (Laurer, 2024; Laurer et al., 2023), a fine-tune of ModernBERT (Warner et al., 2025), using Hugging Face's zero-shot classification pipeline (Wolf et al., 2020) on the VLM-extracted text of each crop, when available. We store the full “ranking” produced by the model, along with confidence scores.

## 9.2 Results

Across all crops, the most frequently top-ranked subjects are “Business, Finance & Market Reports” and “Commercial Advertisements & Classifieds”, followed by “Science, Technology, Health & Agriculture” and “Politics, International Relations & Government” (Figure 14). Two of these labels, “Commercial Advertisements & Classifieds” and “Mastheads, Page Headers & Printer Imprints”, overlap by design with the crop-type categories of Section 6. They were designed to give a secondary and independent signal on crop-type classification as well as to give the zero-shot classifier a broad category to fall back onto for ambiguous crops, rather than forcing a topical label onto a crop that carries no topic.

![](images/d844644b4caf69b4585469c02acb11f14206f336fb48c562d3b09f949f72c3a5.jpg)  
Figure 14. Top-ranked subject labels across all crops.

![](images/8548257c9d993d205ba4e3252ef14315cd1e1a8abd1b8fe883b4a678bc77ebca.jpg)  
Figure 15. Mean top-1 confidence score per subject label, with standard-deviation error bars.

Confidence scores are spread very widely within each subject (Figure 15). Across the 82,985,966 crops with a subject prediction, the mean top-1 confidence score is 0.65, but the standard deviation per label ranges from 0.08 to 0.23. Such a spread is expected in a zero-shot setting, where the classifier scores each label through textual entailment rather than against a distribution it was trained on (Yin et al., 2019). This is the main reason we treat this output as experimental. Mean confidence also varies by subject: the model is most confident on well-delimited topics such as “Sports, Racing & Games” and “Politics, International Relations & Government”, and least confident on broad or residual categories such as “Legal Notices, Auctions & Public Announcements” and mastheads.

The separation between the model's first and second choice is pronounced. Across crops with a subject prediction, the mean gap between the top-1 and top-2 confidence scores is 0.457, putting the mean second choice at 0.20. The runner-up is therefore well above chance, 1/12 across 12 labels. Nonetheless, because the spread is wide, the top labels are not equally reliable across crops, and users should work from the full distribution of scores. The dominant subject per crop type (Figure 16) behaves as expected: advertisement crops map overwhelmingly to commercial subjects, while content crops spread across news-oriented subjects, suggesting that the subject signal is overall consistent with the crop-type classification.

![](images/8dd696b9b05ab133d38551e86d5cf7e0f3aecb5d259e74598bdd926c8cdadc15.jpg)  
Figure 16. Dominant subject within each crop-type category (share of the leading subject, %).

## 10 Crop-level Embeddings

## 10.1 Methodology

We hypothesize that pre-computed embeddings can improve the out-of-the-box accessibility of a dataset. As such, we chose to generate embeddings for both the text and the image of each crop as a way to enable dataset-wide vector search and, more generally, to facilitate downstream use of this dataset.

For image embeddings, we chose facebook/dinov2-small (Oquab et al., 2024) for its small size (22M parameters) and its relative genericity. We resized each crop to 448×448 pixels before inference.

For text embeddings, we chose minishlab/potion-multilingual-128M run through Model2Vec for its light footprint (a 128M-parameter static model) and its overall genericity.

## 10.2 Results

Both embedding sets are stored as 32-bit floating-point vectors, one per crop. The DINOv2-small produces 384-dimensional vectors and the potion-multilingual 256-dimensional vectors. As such, a single crop's embeddings occupy roughly 384 × 4 + 256 × 4 = 2,560 bytes. Across all 83.1 million crops, this is approximately 128 GB for the image embeddings (about 1.54 GB per million crops) and approximately 85 GB for the text embeddings, for a combined total of roughly 213 GB.

While this type of pre-computed output is likely to be helpful, it also has known limitations. The static text embedding model we used has no attention mechanism and therefore relies heavily on token distribution, reducing its usability in semantically ambiguous contexts. Both models are generic and may not support all use cases. As such, specialized applications will likely benefit from embeddings generated with domain or task-specific models.

## 11 Crop-level Detection of Chronicling America's Thesauri Terms

In collaboration with Harvard Law School Library's Public Data Project<sup>10</sup>, we made use of thesauri on race, ethnicity, immigration, and citizenship (Gilmore, 2022), originally compiled by Library of Congress, National Endowment for the Humanities and state partners, and later preserved by the Public Data Project (Harvard Law School Library Innovation Lab, 2026). We ran detections for these terms against the VLM-extracted text of each crop.

The goal was to facilitate digital humanities research use cases, for example on contextualizing historical content and language. We surface these matches as a navigational and statistical aid rather than as an interpretation of the underlying text. These detections are both experimental and "naive"; they should not be used “as is” to infer the nature of any given piece of text.

## 12 Pipeline Footprint

## 12.1 Methodology

The pipeline was run locally on a GPU node we own (described in Appendix D). We designed the pipeline so it can be run on workstation-grade hardware, with throughput as the main trade-off. In any case, running this pipeline against Boston Public Library's collection represented a significant computational undertaking.

In some cases, we had to deploy optimization strategies to make the best of our hardware. Specifically, we identified a bottleneck early on when running inference with YOLO26 models through the ultralytics Python library. We noticed that, while the library allows for batched inference, image pre-processing appeared to happen sequentially rather than in parallel. We therefore re-implemented that part of the pipeline to pre-process images on our end, in parallel, before feeding them to the model as a pre-processed batch of matrices.

Retrieving scans and decoding them was also part of the challenge. To make this work, we processed the collection in batches of 200 issues and used a local disk cache with python-diskcache (Jenks, 2023), so that scan retrieval and decoding for one batch could overlap with the compute-heavy steps and would not have to be repeated.

## 12.2 Results

Processing time was dominated by the two OCR steps (Table 9). On a batch of 200 issues, dots.mocr OCR took 28.62 minutes on average and Tesseract OCR 16.48 minutes, together accounting for roughly half of the 86.69-minute average full batch. The remaining thirteen steps were comparatively inexpensive, most of them running in a few minutes or less.

While we deployed this pipeline on local hardware (Appendix D), we estimate the total cost of running it against this collection on rented hardware to be approximately \~\$25,000, assuming a total runtime of \~1650 hours and the cost of a rented 8xL40S GPU node to be \~\$15/hour on average (as estimated at the time of writing this technical report).

<table><tr><td>Pipeline step</td><td>Avg (min)</td><td>Min (min)</td><td>Max (min)</td></tr><tr><td>01 Scan retrieval &amp; caching</td><td>8.07</td><td>4.94</td><td>35.60</td></tr><tr><td>02 Crop detection (segmentation)</td><td>8.14</td><td>6.18</td><td>22.74</td></tr><tr><td>03 OCR (Tesseract)</td><td>16.48</td><td>12.28</td><td>22.79</td></tr><tr><td>04 OCR (dots.mocr VLM)</td><td>28.62</td><td>22.46</td><td>35.90</td></tr><tr><td>05 Classification (text)</td><td>1.01</td><td>0.79</td><td>1.30</td></tr><tr><td>06 Classification (image)</td><td>4.48</td><td>1.80</td><td>5.75</td></tr><tr><td>07 Classification (final)</td><td>0.09</td><td>0.06</td><td>0.27</td></tr><tr><td>08 Named entity recognition</td><td>4.55</td><td>3.44</td><td>7.20</td></tr><tr><td>09 Subject detection</td><td>5.04</td><td>4.18</td><td>6.19</td></tr><tr><td>10 Reading order</td><td>1.13</td><td>0.90</td><td>1.56</td></tr><tr><td>11 Token counting</td><td>1.28</td><td>0.72</td><td>2.42</td></tr><tr><td>12 Language detection</td><td>0.90</td><td>0.68</td><td>1.21</td></tr><tr><td>13 Text analysis</td><td>2.36</td><td>1.13</td><td>4.61</td></tr><tr><td>14 Chronicling America thesauri match</td><td>2.17</td><td>1.03</td><td>3.19</td></tr><tr><td>15 Embeddings</td><td>2.37</td><td>1.77</td><td>3.15</td></tr><tr><td>Full batch (200 issues)</td><td>86.69</td><td>68.20</td><td>111.49</td></tr></table>

Table 9. Average, minimum, and maximum processing time per pipeline step, per batch of 200 issues. Runs that crashed and/or ran faster because of pre-caching (e.g, after a crash) were not included. Durations for the “Full batch” row were measured directly from logs, not derived from per-step min/max/averages.

## 13 Dataset Preparation and Post-Processing

The pipeline was designed to record data points into a SQLite database as it runs. These records were then processed into a dataset, exported as Parquet<sup>11</sup> shards of 250 scans each (one row per scan), compressed with zstd (Collet and Kucherawy, 2021) and written with a row-group size of 10. The field list is provided in Appendix E.

As we compiled the dataset, we applied some light post-processing to enhance usability:

● Minor text-processing was applied to the VLM OCR text, including: regular-expression-based dehyphenation, Markdown heading normalization, and token-loop removal. The latter, in which a word or character is repeated many times over, is a known failure mode of VLM OCR (Holtzman et al., 2020).

● We corrected likely errors in the locality metadata derived from the Library of Congress’ API. For example, cases in which the city of Boston was associated with the state of New York.

● We applied filtering rules to language detection. We used the issue-level language code instead of our language detection for crops where the confidence score was below 0.50, or where the word count was below 30. In addition, we defaulted to issue-level language if said language was not supported by Lingua, which was the case for Yiddish.

The locality and issue-level language metadata that the last two rules rely on comes from the Library of Congress API (Library of Congress, 2026), which serves catalog records for loc.gov items as JSON.

These rules replaced the detected language code for 2,521,901 crops, or 3.04% of the crops that carry one. Replaced codes are stored without a confidence score so they remain distinguishable from direct detections. The data presented in this report reflects this post-processing.

## 14 Discussion and Future Directions

We see this work as a meaningful step towards unlocking high-quality data from historical newspapers, an important and abundant resource that remains underused in the age of computational access.

Our current set of models and methods was developed and tested against a subset of Boston Public Library's substantial newspaper collection. As a result, it is likely that our pipeline will struggle to process newspaper scans that are meaningfully different from what is present in this collection. This is a limitation we aim to address by continuously revising our models and methods, and by repeating this experiment on different collections with our library partners. Some of our methods have known, collection-independent limitations that we aim to soften over time. For example, our reading order detection mechanism (Section 7) makes assumptions about layout structure, and is therefore unlikely to be a good fit for newspaper scans that are less strictly column-based.

As a future direction, we also intend to further reduce the computational footprint of our pipeline. In particular, VLM-based OCR at this scale remains a significant bottleneck. We hypothesize that training a VLM specifically for performing OCR on historical newspaper crops could produce a model that is both smaller, more accurate and less sensitive to token looping issues than dots.mocr for this specific task.

We look forward to seeing what the community does with this data, these models, and this pipeline. We welcome suggestions and feedback, and hope to work together to further amplify the impact of this medium.

## Rights determination

We respect the intellectual property rights of authors, publishers, and other rights holders. We have taken deliberate steps to include only those issues for which there is no known copyright restriction. The source materials used in the preparation of this dataset were published before 1931, making them public domain in the United States. Issue-level metadata provided by Boston Public Library was used to make this assessment.

While this is relatively low risk, some materials in this dataset may be in the public domain in the United States but still subject to copyright or other rights protections in other jurisdictions. Additionally, the absence of an explicit copyright claim or rights status does not guarantee that a work is in the public domain, either in the U.S. or abroad. Information about the copyright status of individual issues is provided on a good-faith basis and reflects available data at the time of determination, but we cannot guarantee its completeness or accuracy.

Users of this dataset will be solely responsible for making independent legal assessments about how and where they use the materials. Some uses of materials may also be restricted by trademark, privacy, publicity rights, or other such rights or restrictions. It is the user's sole responsibility to consider the possibility that such rights or restrictions may be involved and to secure any needed permissions. If any rights holder believes that a work included in this release is misidentified or improperly included, we welcome contact and will promptly review any concerns. Our goal is to provide broad public access while maintaining respect for intellectual property rights and ensuring responsible data stewardship.

## Acknowledgements

The authors of this technical report would like to thank:

● The Digital & Online Services team at Boston Public Library for their expertise and continuous feedback, which shaped this work.

● Our team at the Institutional Data Initiative for their help with our initial rounds of annotations.

● Prof. Benjamin Charles Germain Lee and Dr. Molly Hardy for their advice and support.

This work was supported by unrestricted funding from Microsoft, OpenAI, Meta and Jane Street.

AI tools were used to assist in the preparation of this technical report.

## Disclaimers

## Harmful Language and Content in this Dataset

This dataset is a collection of historical works that reflect the language, culture, and perspectives of their time. Users should be aware that some materials may contain language or portrayals that are outdated, offensive, or harmful today, such as racism, sexism, colonial attitudes, and other forms of discrimination. Some content may include inaccurate information, providing insight into historical contexts that existed at the time of writing. The materials are maintained in their original form to retain contextual understanding and facilitate research efforts, but we encourage critical awareness and cultural sensitivity for the creators and/or subjects of the collection. These materials are offered as part of a historical perspective, but should not be considered a stand-alone research collection constructed to give a balanced perspective on any topic.

## Harmful Language in Bibliographic Description

Metadata for this collection may contain language that is overtly or implicitly harmful, outdated, or biased, or may by omission fail to represent important perspectives. Metadata may contain language created decades ago. It is common practice within the field of library science to reuse descriptions provided from the creator of the materials. While in some instances this allows communities and individuals to represent their materials in their own words, unexamined use of this practice may mean that racist or other offensive terminologies appear in our description. We also use national standardized terms in our work that can be outdated and harmful. Note that terminology in historical materials and in library descriptions does not always match the language we currently understand to be preferred by members of the communities depicted.

Furthermore, we acknowledge that the act of collecting materials is not always neutral, and the work of describing and classifying library materials is influenced by inherent personal, institutional, and societal biases. Outdated or offensive terminologies may be present in metadata such as subject headings, and harmful language or bias may be introduced by catalogers supplying titles and descriptions. In other cases, the source materials themselves present racist, offensive or otherwise harmful viewpoints in titles or descriptions that are routinely transcribed by catalogers.

Note: Some language in this statement was adopted from Harvard Library's statement on Harmful Language in Library collections<sup>12</sup>.

## Generated and Experimental Content

This dataset contains generated and/or experimental content. While reasonable care was taken to ensure its quality, it is provided “as is,” without warranties of any kind. It may contain errors or inaccuracies; users should verify the data independently and apply their own judgment.

## Reference list

Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S. and Vollgraf, R. (2019) 'FLAIR: An easy-to-use framework for state-of-the-art NLP', in Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics (Demonstrations). Minneapolis: Association for Computational Linguistics, pp. 54–59. https://doi.org/10.18653/v1/N19-4010

Bourne, J. (2025) 'Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models', arXiv preprint arXiv:2502.14901.   
https://doi.org/10.48550/arXiv.2502.14901

Campello, R.J.G.B., Moulavi, D. and Sander, J. (2013) 'Density-based clustering based on hierarchical density estimates', in Advances in Knowledge Discovery and Data Mining (PAKDD 2013), Lecture Notes in Computer Science 7819. Springer, pp. 160–172. https://doi.org/10.1007/978-3-642-37456-2\_14

Cargnelutti, M., Brobston, C., Hess, J., Cushman, J., Mukk, K., Scourtas, A., Courtney, K., Leppert, G., Watson, A., Whitehead, M. and Zittrain, J. (2025) 'Institutional Books 1.0: A 242B token dataset from Harvard Library's collections, refined for accuracy and usability', arXiv preprint arXiv:2506.08300. https://doi.org/10.48550/arXiv.2506.08300

Clausner, C., Papadopoulos, C., Pletschacher, S. and Antonacopoulos, A. (2015) 'The ENP image and ground truth dataset of historical newspapers', in Proceedings of the 13th International Conference on Document Analysis and Recognition (ICDAR 2015). IEEE, pp. 931–935.   
https://doi.org/10.1109/ICDAR.2015.7333898

Collet, Y. and Kucherawy, M.S. (ed.) (2021) Zstandard Compression and the 'application/zstd' Media Type, RFC 8878. Internet Engineering Task Force. https://doi.org/10.17487/RFC8878

Dell, M., Carlson, J., Bryan, T., Silcock, E., Arora, A., Shen, Z., D'Amico-Wong, L., Le, Q., Querubin, P. and Heldring, L. (2023) 'American Stories: A large-scale structured text dataset of historical U.S. newspapers', in Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track, pp. 80744–80772. https://doi.org/10.52202/075280-3540

Dutta, A. and Zisserman, A. (2019) 'The VIA annotation software for images, audio and video', in Proceedings ofthe 27th ACM International Conference on Multimedia (MM '19). ACM, pp. 2276–2279. https://doi.org/10.1145/3343031.3350535

Ehrmann, M., Hamdi, A., Linhares Pontes, E., Romanello, M. and Doucet, A. (2023) 'Named entity recognition and classification in historical documents: A survey', ACM Computing Surveys, 56(2), article 27, pp. 1–47. https://doi.org/10.1145/3604931

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S. and Leahy, C. (2020) 'The Pile: An 800GB dataset of diverse text for language modeling', arXiv preprint arXiv:2101.00027. https://doi.org/10.48550/arXiv.2101.00027

Gilmore, S. (2022) Race and Ethnicity Keyword Thesaurusfor Chronicling America: A New Tool on EDSITEment. National Endowment for the Humanities. Available at:   
https://www.neh.gov/blog/race-and-ethnicity-keyword-thesaurus-chronicling-america-new-tool-edsitemen t (Accessed: 29 July 2026).

Harvard Law School Library Innovation Lab (2026) Chronicling America Thesauri. Available at: https://harvard-lil.github.io/chronicling-america-thesauri/ (Accessed: 29 July 2026).

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T. et al. (2022) 'Training compute-optimal large language models', in Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 30016–30030. https://doi.org/10.52202/068431-2176

Holtzman, A., Buys, J., Du, L., Forbes, M. and Choi, Y. (2020) 'The curious case of neural text degeneration', in International Conference on Learning Representations (ICLR 2020). arXiv:1904.09751. https://doi.org/10.48550/arXiv.1904.09751

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A. and Fung, P. (2023) 'Survey of hallucination in natural language generation', ACM Computing Surveys, 55(12), article 248, pp. 1–38. https://doi.org/10.1145/3571730

Jenks, G. (2023) DiskCache: Disk andfile backedpersistent cache (version 5.6.3). Available at: https://github.com/grantjenks/python-diskcache (Accessed: 5 August 2026).

Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C. and Kalfaoglu, M.E. (2026) 'Ultralytics YOLO26: Unified real-time end-to-end vision models', arXiv preprint arXiv:2606.03748. https://doi.org/10.48550/arXiv.2606.03748. See also Jocher, G., Qiu, J. and Chaurasia, A. (2023) Ultralytics YOLO (version 8.0.0). Available at: https://github.com/ultralytics/ultralytics.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J. and Amodei, D. (2020) 'Scaling laws for neural language models', arXiv preprint arXiv:2001.08361. https://doi.org/10.48550/arXiv.2001.08361

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H. and Stoica, I. (2023) 'Efficient memory management for large language model serving with PagedAttention', in Proceedings ofthe 29th Symposium on Operating Systems Principles (SOSP '23). ACM, pp. 611–626. https://doi.org/10.1145/3600006.3613165

Laurer, M. (2024) ModernBERT-large-zeroshot-v2.0. Hugging Face. Available at: https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (Accessed: 31 July 2026).

Laurer, M., van Atteveldt, W., Casas, A. and Welbers, K. (2023) 'Building efficient universal classifiers with natural language inference', arXiv preprint arXiv:2312.17543.   
https://doi.org/10.48550/arXiv.2312.17543

LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998) 'Gradient-based learning applied to document recognition', Proceedings of the IEEE, 86(11), pp. 2278–2324. https://doi.org/10.1109/5.726791

Lee, B.C.G., Mears, J., Jakeway, E., Ferriter, M., Adams, C., Yarasavage, N., Thomas, D., Zwaard, K. and Weld, D.S. (2020) 'The Newspaper Navigator dataset: Extracting headlines and visual content from 16 million historic newspaper pages in Chronicling America', in Proceedings ofthe 29th ACM International Conference on Information and Knowledge Management (CIKM '20). ACM, pp. 3055–3062. https://doi.org/10.1145/3340531.3412767

Levchenko, M. (2025) 'Evaluating LLMs for historical document OCR: A methodological framework for digital humanities', arXiv preprint arXiv:2510.06743. https://doi.org/10.48550/arXiv.2510.06743

Levine, N., Duvenaud, D. and Radford, A. (2026) Introducing talkie: a 13B vintage language model from 1930. Available at: https://talkie-lm.com/introducing-talkie (Accessed: 29 July 2026).

Li, Y., Yang, G., Liu, H., Wang, B. and Zhang, C. (2025) 'dots.ocr: Multilingual document layout parsing in a single vision-language model', arXiv preprint arXiv:2512.02498.   
https://doi.org/10.48550/arXiv.2512.02498

Library of Congress (2026) APIs at the Library of Congress. Available at: https://www.loc.gov/apis/ (Accessed: 31 July 2026).

Library of Congress and National Endowment for the Humanities (2007) Chronicling America: Historic American Newspapers (National Digital Newspaper Program). Available at: https://www.loc.gov/collections/chronicling-america/ (Accessed: 29 July 2026).

Longpre, S., Yauney, G., Reif, E., Lee, K., Roberts, A., Zoph, B., Zhou, D., Wei, J., Robinson, K., Mimno, D. and Ippolito, D. (2024) 'A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, and toxicity', in Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, pp. 3245–3276. https://doi.org/10.18653/v1/2024.naacl-long.179

Manjavacas, E. and Fonteyn, L. (2022) 'Adapting vs. pre-training language models for historical languages', Journal ofData Mining & Digital Humanities, NLP4DH special issue. https://doi.org/10.46298/jdmdh.9152

McInnes, L., Healy, J. and Astels, S. (2017) 'hdbscan: Hierarchical density based clustering', Journal of Open Source Software, 2(11), p. 205. https://doi.org/10.21105/joss.00205

MinishLab (2024) Potion: Static embedding models (potion-base-32M, potion-multilingual-128M).   
Hugging Face. Available at: https://huggingface.co/minishlab (Accessed: 29 July 2026).

Muhammad, M.B. and Yeasin, M. (2020) 'Eigen-CAM: Class activation map using principal components', in 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, pp. 1–7. https://doi.org/10.1109/IJCNN48605.2020.9206626

Nguyen, T.T.H., Jatowt, A., Coustaty, M. and Doucet, A. (2021) 'Survey of post-OCR processing approaches', ACM Computing Surveys, 54(6), article 124, pp. 1–37. https://doi.org/10.1145/3453476

Nikolaidou, K., Seuret, M., Mokayed, H. and Liwicki, M. (2022) 'A survey of historical document image datasets', International Journal on Document Analysis and Recognition (IJDAR), 25(4), pp. 305–338. https://doi.org/10.1007/s10032-022-00405-8

OpenAI (2024) tiktoken: A fast BPE tokeniser for use with OpenAI's models. Available at: https://github.com/openai/tiktoken (Accessed: 29 July 2026).

Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A. et al. (2024) 'DINOv2: Learning robust visual features without supervision', Transactions on Machine Learning Research. Available at: https://openreview.net/forum?id=a68SUt6zFt.

Poznanski, J., Rangapur, A., Borchardt, J., Dunkelberger, J., Huff, R., Lin, D., Wilhelm, C., Lo, K. and Soldaini, L. (2025) 'olmOCR: Unlocking trillions of tokens in PDFs with vision language models', arXiv preprint arXiv:2502.18443. https://doi.org/10.48550/arXiv.2502.18443

PyICU contributors (2024) PyICU: Python extension wrapping the ICU C++ API. Available at: https://gitlab.pyicu.org/main/pyicu (Accessed: 29 July 2026).

Qwen Team (2025a) 'Qwen3 technical report', arXiv preprint arXiv:2505.09388. https://doi.org/10.48550/arXiv.2505.09388

Qwen Team (2025b) Qwen3-VL-30B-A3B-Thinking-FP8. Hugging Face. Available at: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 (Accessed: 29 July 2026).

Rice, S.V., Nagy, G. and Nartker, T.A. (1999) Optical Character Recognition: An Illustrated Guide to the Frontier. Boston, MA: Springer US. https://doi.org/10.1007/978-1-4615-5021-1

rigvedrs (2026) YOLO-26-CAM: EigenCAM for YOLO model interpretability. Available at: https://github.com/rigvedrs/YOLO-26-CAM (Accessed: 6 August 2026).

Semnani, S., Zhang, H., He, X., Tekgurler, M. and Lam, M. (2025) 'CHURRO: Making history readable with an open-weight large vision-language model for high-accuracy, low-cost historical text recognition', arXiv preprint arXiv:2509.19768. https://doi.org/10.48550/arXiv.2509.19768

sirfz (2024) tesserocr: A Python wrapperfor the tesseract-ocr API. Available at: https://github.com/sirfz/tesserocr (Accessed: 29 July 2026).

Smith, R. (2007) 'An overview of the Tesseract OCR engine', in Proceedings ofthe Ninth International Conference on Document Analysis and Recognition (ICDAR 2007). IEEE, pp. 629–633. https://doi.org/10.1109/ICDAR.2007.4376991

Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D. et al. (2024) 'Dolma: An open corpus of three trillion tokens for language model pretraining research', in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, pp. 15725–15788. https://doi.org/10.18653/v1/2024.acl-long.840

Stahl, P.M. (2024) Lingua: The most accurate natural language detection library for Python. Available at: https://github.com/pemistahl/lingua-py (Accessed: 29 July 2026).

Tesseract OCR contributors (2024a) Tesseract OCR (version 5). Available at: https://github.com/tesseract-ocr/tesseract (Accessed: 31 July 2026).

Tesseract OCR contributors (2024b) tessdata\_best: Best (most accurate) trained LSTM modelsfor Tesseract. Available at: https://github.com/tesseract-ocr/tessdata\_best (Accessed: 31 July 2026).

Traub, M.C., van Ossenbruggen, J. and Hardman, L. (2015) 'Impact analysis of OCR quality on research tasks in digital archives', in Research and Advanced Technology for Digital Libraries: 19th International Conference on Theory and Practice of Digital Libraries (TPDL 2015), Lecture Notes in Computer Science 9316. Cham: Springer, pp. 252–263. https://doi.org/10.1007/978-3-319-24592-8\_19

Tulkens, S. and van Dongen, T. (2024) Model2Vec: Fast state-of-the-art static embeddings. Zenodo. https://doi.org/10.5281/zenodo.17270888

Vanderbauwhede, W. (2023) 'Frugal Computing: On the need for low-carbon and sustainable computing and the path towards zero-carbon computing', arXiv preprint arXiv:2303.06642. https://doi.org/10.48550/arXiv.2303.06642

Wang, Z., Xu, Y., Cui, L., Shang, J. and Wei, F. (2021) 'LayoutReader: Pre-training of text and layout for reading order detection', in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021). Association for Computational Linguistics, pp. 4735–4744. https://doi.org/10.18653/v1/2021.emnlp-main.389

Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Adams, G.T., Howard, J. and Poli, I. (2025) 'Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference', in Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, pp. 2526–2547.   
https://doi.org/10.18653/v1/2025.acl-long.127

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q. and Rush, A.M. (2020) 'Transformers: State-of-the-art natural language processing', in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Association for Computational Linguistics, pp. 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6

Yang, Z., Tang, J., Li, Z., Wang, P., Wan, J., Zhong, H., Liu, X., Yang, M., Wang, P., Bai, S., Jin, L. and Lin, J. (2025) 'CC-OCR: A comprehensive and challenging OCR benchmark for evaluating large multimodal models in literacy', in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2025). IEEE, pp. 21744–21754. https://doi.org/10.1109/ICCV51701.2025.02019

Yin, W., Hay, J. and Roth, D. (2019) 'Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach', in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, pp. 3912–3921. https://doi.org/10.18653/v1/D19-1404

Zheng, H., Li, Y., Zhang, K., Xin, L., Zhao, G. et al. (2026) 'Multimodal OCR: Parse anything from documents', arXiv preprint arXiv:2603.13032. https://doi.org/10.48550/arXiv.2603.13032

## Appendices

## Appendix A — Outer scan margins

Section 3.2 reports that crop detections cover 89% of each scan on average. To determine how much of the remaining area is empty margin rather than missed content, we measured the distance between each edge of the scan and the outermost crop bounding box on a random sample of 100 scans (one scan drawn from each of 100 randomly selected metadata shards, seed 0). No sampled scan had to be discarded for missing dimensions or for having no crops. Bounding boxes are stored in source-scan pixels, so the left and top margins are the smallest x0 and y0 across all crops of the scan, and the right and bottom margins are the scan width and height minus the largest x1 and y1.

<table><tr><td>Margin</td><td>Mean (px)</td><td>Min (px)</td><td>Max (px)</td><td>Mean (% of scan)</td><td>Min (%)</td><td>Max (%)</td></tr><tr><td>Left</td><td>228</td><td>0</td><td>690</td><td>3.06%</td><td>0.00%</td><td>7.37%</td></tr><tr><td>Top</td><td>253</td><td>8</td><td>1,054</td><td>2.57%</td><td>0.12%</td><td>10.29%</td></tr><tr><td>Right</td><td>264</td><td>7</td><td>550</td><td>3.63%</td><td>0.10%</td><td>9.96%</td></tr><tr><td>Bottom</td><td>302</td><td>20</td><td>704</td><td>3.14%</td><td>0.21%</td><td>7.81%</td></tr></table>

Table A.1. Outer margins left uncovered by crop detections, across 100 randomly sampled scans.

The rectangle that encloses all crops of a scan covers 88.02% of the scan area on average, with a minimum of 75.84% and a maximum of 95.92%. That envelope is slightly smaller than the 89% average coverage reported in Section 3.2, which suggests the crops of a scan fill their own envelope almost completely. These results indicate that uncovered area therefore sits mainly in the outer margins, and the segmentation model leaves comparatively little content uncovered inside the printed area of the page.

## Appendix B — Crop-type auto-annotation prompt

The following prompt was used with Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 to auto-annotate the crop-type classification training set (see Section 6.1). The OCR text of the crop is appended after the Text: marker.

![](images/01460ab34bd633ae5de75fb6fe3158912c06a04f1b1ad53d6a28b14c709d49f3.jpg)  
Figure B.1. Prompt used to auto-annotate the crop-type classification training set (Section 6)

## Appendix C — Crop-type classifier confidence

Section 6.2 reports the overall confidence of each crop-type classifier (text and image). This appendix breaks those figures down by category.

Confidence is highest on the two categories that carry almost all of the volume, “Advertisement” and “Content”, and on the two structural categories, ‘Section heading” and “Masthead, nameplate or running head”. It drops sharply on “Photograph or illustration” and “Cartoon”, the two categories where Table 4 also reports the weakest precision and recall. Both classifiers therefore signal their own uncertainty on the same categories.

The text classifier does not model “Empty” as a learned label: as noted in Section 6.1, every crop with no OCR-able text is assigned “Empty” by rule. Its $1 . 0 0 0 \pm 0 . 0 0 0$ is therefore an artifact of that rule and not a measurement. The image classifier does predict “Empty” as a learned label, and its 0.679 mean is the lowest of the seven, which is consistent with the low precision reported for that label in Table 4.

![](images/04511cb464b3f3f4e87207b0108438152e6583c8a2d588ee78fd62f8c183db71.jpg)  
Figure C.1. Mean confidence score ofthe image and text crop-type classifiers, per predicted category.

<table><tr><td>Category</td><td>Image mean ± SD Image crops</td><td></td><td>Text mean ± SD</td><td>Text crops</td></tr><tr><td>Content</td><td> $0 . 9 5 0 \pm 0 . 1 0 7$ </td><td>26,973,099</td><td> $0 . 9 5 7 \pm 0 . 1 0 2$ </td><td>28,898,044</td></tr><tr><td>Advertisement</td><td> $0 . 9 6 3 \pm 0 . 0 8 9$ </td><td>49,025,100</td><td> $0 . 9 7 8 \pm 0 . 0 7 6$ </td><td>48,095,943</td></tr><tr><td>Section heading</td><td> $0 . 8 9 6 \pm 0 . 1 3 1$ </td><td>5,045,743</td><td> $0 . 9 3 8 \pm 0 . 1 2 5$ </td><td>4,091,158</td></tr><tr><td>Masthead, nameplate or running head</td><td> $0 . 9 4 8 \pm 0 . 1 3 0$ </td><td>1,414,387</td><td> $0 . 9 3 5 \pm 0 . 1 3 8$ </td><td>1,567,417</td></tr><tr><td>Photograph or illustration</td><td> $0 . 7 1 9 \pm 0 . 1 4 1$ </td><td>423,011</td><td> $0 . 7 0 5 \pm 0 . 1 8 4$ </td><td>227,434</td></tr><tr><td>Cartoon</td><td> $0 . 8 3 8 \pm 0 . 2 0 2$ </td><td>229,046</td><td> $0 . 7 4 2 \pm 0 . 1 9 7$ </td><td>105,970</td></tr><tr><td>Empty</td><td> $0 . 6 7 9 \pm 0 . 1 7 3$ </td><td>36,655</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td>161,075</td></tr><tr><td>All categories (weighted)</td><td>0.953</td><td>83,147,041</td><td>0.967</td><td>83,147,041</td></tr></table>

Table C.1. Mean confidence score and standard deviation of each crop-type classifier, per predicted category, across all classified crops.

## Appendix D — Compute environment

The pipeline was run on a single GPU node owned by the Institutional Data Initiative, with the following specification:

● 8 × NVIDIA L40S GPUs

● 256 CPU cores

● 768 GB ofRAM

## Appendix E — Dataset field list

Each row in the dataset represents a single newspaper scan (page). Most crop-level fields are list columns, with one entry per crop in the same order as crop\_bbox\_gen, which is sorted by reading order. NER, classification, subject, and language fields are lists of lists (one inner list per crop).

<table><tr><td>Suffix</td><td>Meaning</td></tr><tr><td>_src</td><td>&quot;From source&quot;. This field&#x27;s data comes from information we gathered from the collection itself.</td></tr><tr><td>_ext</td><td>&quot;External&quot;. This field&#x27;s data was pulled from an external source via a records matching mechanism.</td></tr><tr><td>_gen</td><td>&quot;Generated&quot;. This field&#x27;s data was generated as part of our analysis / processing pipeline.</td></tr><tr><td>_exp</td><td>&quot;Experimental&quot;. Similar to _gen, but was generated through more experimental or exploratory means.</td></tr></table>

Table E.1. Suf ix convention for column provenance.

<table><tr><td colspan="1" rowspan="1">Field</td><td colspan="1" rowspan="1">Type</td><td colspan="1" rowspan="1">Role</td><td colspan="1" rowspan="1">Section</td></tr><tr><td colspan="1" rowspan="1">scan_image</td><td colspan="1" rowspan="1">bytes</td><td colspan="1" rowspan="1">The source scan, stored as embeddedWEBP bytes at quality 70.</td><td colspan="1" rowspan="1">1</td></tr><tr><td colspan="1" rowspan="1">corpus</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Corpus identifier</td><td colspan="1" rowspan="1">13</td></tr><tr><td colspan="1" rowspan="1">issue_id_src</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Issue identifier</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">newspaper_id_src</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Newspaper (title) identifier</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">newspaper_id_type_gen</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Type of the newspaper identifier.</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">page_number_gen</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Page number within the issue. Inferredfrom filenames within source archive</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">scan_filename_src</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Source scan filename</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">scan_width_src</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Scan width in pixels</td><td colspan="1" rowspan="1">3</td></tr><tr><td colspan="1" rowspan="1">scan_height_src</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Scan height in pixels</td><td colspan="1" rowspan="1">3</td></tr><tr><td colspan="1" rowspan="1">year_ext</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Publication year</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">month_ext</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Publication month</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">day_ext</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Publication day</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">edition_ext</td><td colspan="1" rowspan="1">int</td><td colspan="1" rowspan="1">Edition</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">metadata_source_gen</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Source of the issue-level metadata</td><td colspan="1" rowspan="1">5,13</td></tr><tr><td colspan="1" rowspan="1">city_gen</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Locality city (post-processed, sourcedfrom metadata_source_gen)</td><td colspan="1" rowspan="1">13</td></tr><tr><td colspan="1" rowspan="1">state_gen</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Locality state (post-processed, sourcedfrom metadata_source_gen)</td><td colspan="1" rowspan="1">13</td></tr><tr><td colspan="1" rowspan="1">country_gen</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Locality country (post-processed, sourcedfrom metadata_source_gen)</td><td colspan="1" rowspan="1">13</td></tr><tr><td colspan="1" rowspan="1">language_ext</td><td colspan="1" rowspan="1">string</td><td colspan="1" rowspan="1">Issue-level language code (sourced frommetadata_source_gen)</td><td colspan="1" rowspan="1">5</td></tr><tr><td colspan="1" rowspan="1">crop_bbox_gen</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Crop bounding boxes</td><td colspan="1" rowspan="1">3</td></tr><tr><td colspan="1" rowspan="1">crop_bbox_conf_gen</td><td colspan="1" rowspan="1">list[float]</td><td colspan="1" rowspan="1">Crop detection confidence scores</td><td colspan="1" rowspan="1">3</td></tr><tr><td colspan="1" rowspan="1">crop_vlm_ocr_gen</td><td colspan="1" rowspan="1">list[string]</td><td colspan="1" rowspan="1">dots.mocr OCR text (post-processed)</td><td colspan="1" rowspan="1">4</td></tr><tr><td colspan="1" rowspan="1">crop_tesseract_ocr_gen</td><td colspan="1" rowspan="1">list[struct]</td><td colspan="1" rowspan="1">OCR text of each crop from Tesseract 5, as{text, metadata}. text is plain text.metadata is a list holding one record perrecognized word, each with text (thesurface form), conf (the Tesseractconfidence score), and bbox_xyxy (theword box as [x_min, y_min, x_max,y_max], in pixel coordinates relative to thecrop, rescaled back to the crop's fullresolution). Useful for building OCRoutputs compatible with existing librarydiscovery systems.</td><td colspan="1" rowspan="1">4</td></tr><tr><td colspan="1" rowspan="1">crop_vlm_ocr_token_count_gen</td><td colspan="1" rowspan="1">list[int]</td><td colspan="1" rowspan="1">dots.mocr o200k_base token count</td><td colspan="1" rowspan="1">4</td></tr><tr><td colspan="1" rowspan="1">crop_tesseract_ocr_token_count_gen</td><td colspan="1" rowspan="1">list[int]</td><td colspan="1" rowspan="1">Tesseract o200k base token count</td><td colspan="1" rowspan="1">4</td></tr><tr><td colspan="1" rowspan="1">crop_text_analysis_gen</td><td colspan="1" rowspan="1">list[list[string]]</td><td colspan="1" rowspan="1">Per-crop text metrics. Holdstokenizability_score, char_count,word_count, word_count_unique,word_type_token_ratio,sentence_count, andsentence_count_unique, each prefixedtesseract_ and vlm_, plusvlm_has_table and vlm_has_markdown.</td><td colspan="1" rowspan="1">5</td></tr><tr><td colspan="1" rowspan="1">crop_classification_gen</td><td colspan="1" rowspan="1">list[string]</td><td colspan="1" rowspan="1">Final crop-type category</td><td colspan="1" rowspan="1">6</td></tr><tr><td colspan="1" rowspan="1">crop_classification_image_only_gen</td><td colspan="1" rowspan="1">list[string]</td><td colspan="1" rowspan="1">Image-classifier category</td><td colspan="1" rowspan="1">6</td></tr><tr><td colspan="1" rowspan="1">crop_classification_image_only_conf_gen</td><td colspan="1" rowspan="1">list[float]</td><td colspan="1" rowspan="1">Image-classifier confidence</td><td colspan="1" rowspan="1">6</td></tr><tr><td colspan="1" rowspan="1">crop_classification_text_only_gen</td><td colspan="1" rowspan="1">list[string]</td><td colspan="1" rowspan="1">Text-classifier category</td><td colspan="1" rowspan="1">6</td></tr><tr><td colspan="1" rowspan="1">crop_classification_text_only_conf_gen</td><td colspan="1" rowspan="1">list[float]</td><td colspan="1" rowspan="1">Text-classifier confidence</td><td colspan="1" rowspan="1">6</td></tr><tr><td colspan="1" rowspan="1">crop_language_gen</td><td colspan="1" rowspan="1">list[string]</td><td colspan="1" rowspan="1">Detected language (post-processed,issue-level language used if confidence islow)</td><td colspan="1" rowspan="1">5</td></tr><tr><td colspan="1" rowspan="1">crop_language_conf_gen</td><td colspan="1" rowspan="1">list[float]</td><td colspan="1" rowspan="1">Detected language confidence(post-processed, NULL if issue-levellanguage was used)</td><td colspan="1" rowspan="1">5</td></tr><tr><td colspan="1" rowspan="1">crop_ner_per_gen</td><td colspan="1" rowspan="1">list[list[string]]</td><td colspan="1" rowspan="1">Person entities</td><td colspan="1" rowspan="1">8</td></tr><tr><td colspan="1" rowspan="1">crop_ner_per_conf_gen</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Person entity confidence</td><td colspan="1" rowspan="1">8</td></tr><tr><td colspan="1" rowspan="1">crop_ner_loc_gen</td><td colspan="1" rowspan="1">list[list[string]]</td><td colspan="1" rowspan="1">Location entities</td><td colspan="1" rowspan="1">8</td></tr><tr><td colspan="1" rowspan="1">crop_ner_loc_conf_gen</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Location entity confidence</td><td colspan="1" rowspan="1">8</td></tr><tr><td colspan="1" rowspan="1">crop_ner_org_gen</td><td colspan="1" rowspan="1">list[list[string]]</td><td colspan="1" rowspan="1">Organization entities</td><td colspan="1" rowspan="1">8</td></tr><tr><td colspan="1" rowspan="1">crop_ner_org_conf_gen</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Organization entity confidence</td><td colspan="1" rowspan="1">8</td></tr><tr><td colspan="1" rowspan="1">crop_subject_gen</td><td colspan="1" rowspan="1">list[list[string]]</td><td colspan="1" rowspan="1">Subject ranking</td><td colspan="1" rowspan="1">9</td></tr><tr><td colspan="1" rowspan="1">crop_subject_conf_gen</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Subject confidence</td><td colspan="1" rowspan="1">9</td></tr><tr><td colspan="1" rowspan="1">crop_chronam_thesauri_matches_exp</td><td colspan="1" rowspan="1">list[struct]</td><td colspan="1" rowspan="1">Keyword matches from the Race,Ethnicity, Citizenship and ImmigrationKeyword Thesauri for ChroniclingAmerica. Holds matches (a list of{category, terms}, terms being {term,count} pairs), match_count, andterm_count, each prefixed tesseract_and v1m_. Naive match: may be used fordownstream research, but not "as is". (Seesection 11)</td><td colspan="1" rowspan="1">11</td></tr><tr><td colspan="1" rowspan="1">crop_text_embeddings</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Text embeddings (potion-multilingual,256-d)</td><td colspan="1" rowspan="1">10</td></tr><tr><td colspan="1" rowspan="1">crop_image_embeddings</td><td colspan="1" rowspan="1">list[list[float]]</td><td colspan="1" rowspan="1">Image embeddings (DINOv2-small,384-d)</td><td colspan="1" rowspan="1">1</td></tr></table>

Table E.2. Dataset fields with their type, grouped by role, and the report section each relates to.