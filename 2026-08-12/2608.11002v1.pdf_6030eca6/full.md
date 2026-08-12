# On the Limitations of Cross-Lingual Consistency in Multilingual Text-to-image Generation

Sicheng Zhang<sup>1</sup>, Zhonghao Yan<sup>2</sup>, Binzhu Xie<sup>3</sup>, Shi Qiu<sup>3</sup>, Muzammal Naseer<sup>1,4,∗</sup>, Naveed Akhtar<sup>5</sup>, Mubarak Shah<sup>6</sup>

<sup>1</sup>Khalifa University, <sup>2</sup>Queen Mary University of London, <sup>3</sup>The Chinese University of Hong Kong, <sup>4</sup>The University of Western Australia, <sup>5</sup>The University of Melbourne, <sup>6</sup>University of Central Florida

<sup>∗</sup>Corresponding author

Text-to-image (T2I) generation has achieved remarkable progress in recent years. However, existing research has largely focused on English-only settings, leaving cross-lingual performance gaps and language-specific efects insuficiently explored. To fill this gap, we introduce LingT2I, a benchmark covering 10 widely used languages with 33K prompts, designed to evaluate cross-lingual efects in both content generation and text rendering. Building on this benchmark, we conduct a comprehensive cross-lingual analysis, uncovering linguistic inequality and language-dependent trade-ofs across evaluation dimensions. Beyond quantitative evaluation, we further reveal a range of language-dependent generation patterns, highlighting how linguistic factors and their corresponding cultural contexts systematically impact model outputs. Our benchmark and analysis provide a foundation for studying cross-lingual behavior in T2I generation and facilitate the development of more robust and inclusive models.

<sup>§</sup> GitHub: https://github.com/RISys-Lab/LingT2I Dataset: https://huggingface.co/datasets/RISys-Lab/LingT2I

## 1 Introduction

Language is a primary interface between humans and artificial intelligence, playing a decisive role in shaping multimodal generative content. Text-to-image (T2I) models exemplify this trend, achieving remarkable success in English through large-scale diffusion frameworks [58, 70]. However, these advances largely rely on English-dominant datasets like COCO Captions [8] and LAION-5B [51], leaving their capabilities in multilingual settings largely underexplored. In parallel, research in multilingual NLP has em phasized that linguistic diversity and inclusivity are crucial for developing equitable and culturally aware AI [25, 32], underscoring the importance of extending T2I evaluation beyond English.

Multilingual T2I generation faces two fundamental challenges: generating visual content must account for the unique cultural at tributes embedded in diferent languages; the Text Rendering task— that is, generating images with specific textual content—requires handling diverse writing systems. Previous work has made eforts to extend T2I models to multilingual settings, either by incorporating existing encoders with limited multilingual foundations [29, 47, 64] or by leveraging large generative LLMs for stronger prompt interpretation [61, 70]. However, existing studies on multilingual T2I remain limited [18, 23, 50, 67], focusing mainly on image quality while overlooking linguistic aspects and text rendering evaluation.

This raises fundamental questions: do T2I models truly possess multilingual competence? More importantly, whatfactors underlie the performance disparities across languages? Figure 1 highlights several representative phenomena: (i) prompts in low-resource languages like Hindi consistently underperform compared to highresource ones, reflecting clear linguistic inequality; (ii) non-Latin scripts in the Text Rendering task often appear broken, unreadable, or hallucinated, underscoring the dificulty of handling diverse writing systems; (iii) even for semantically identical prompts, diferent languages exhibit divergent trade-ofs across dimensions such as realism, semantic faithfulness, and style; these interactions may appear as coupled improvements, conflicting trends, or balanced compromises, highlighting the instability of cross-lingual generalization; and (iv) generation behavior varies systematically across languages, indicating that T2I models are influenced not only by textual semantics but also by language-specific priors. The causes, including data distribution, linguistic morphology, and writing systems, remain underexplored, underscoring the need for a systematic framework for cross-lingual analysis.

To investigate these challenges, we introduce LingT2I, a benchmark specifically designed for analyzing cross-lingual efects, which covers 10 widely used languages and evaluates both Content Generation and Text Rendering tasks. This unified dataset forms a foundation for large-scale analysis of multilingual T2I generation.

We benchmark several state-of-the-art T2I models—including Nano Banana [58], Z-Image [62], and EasyText [35]—on LingT2I and present a comprehensive large-scale cross-lingual analysis. Our results reveal three key findings: (i) general-purpose models exhibit severe linguistic inequality, with performance skewed toward high resource Indo-European languages; (ii) non-Latin writing systems remain a major bottleneck, leading to broken or unreadable text rendering; and (iii) language-specific cultural and typological factors systematically impact generation behavior, reshaping trade-ofs across evaluation dimensions. These findings expose fundamental limitations of current multilingual T2I systems and provide guidance for developing fairer and more culturally inclusive generative models. Our key contributions are as follows:

![](images/a4796fb5cfd731b7a80c126c05c9afa0c8277e544705f2f678baea825af3dc38.jpg)  
Figure 1: Challenges of multilingual T2I. (i) Linguistic inequality: the Hindi version has an incorrect number of objects; (ii) Text rendering failures: misarrangements of letters and structural errors in characters; (iii) Multi-dimensional trade-ofs across languages: the Korean results appear in oil-painting style, inconsistent with the intended photograph style. (iv) Language dependent generation patterns: the same prompt yields textiles with distinct cultural characteristics across languages.

• We present LingT2I, a new dataset covering 10 widely used languages with 33K prompts, designed to analyze cross-lingual effects in both general Content Generation and Text Rendering.

• We provide the first comprehensive cross-lingual analysis, reveal ing linguistic inequality and language-specific trade-ofs across dimensions in T2I models.

• Our analysis reveals various language-dependent generative pat terns, providing valuable insights for model design.

## 2 Related Work

Multilingual Text-to-image Generation. Recent works have endowed text-to-image models with multilingual abilities. Models [7, 19, 53, 71] such as SD 3.5 [54], FLUX [29], and Z-Image [62], adopt difusion or difusion transformer (DiT) architectures, where language understanding is primarily handled by pretrained text encoders [47, 57, 64]. Recent approaches such as HunyuanImage-3.0 [61], Janus-Pro [9] and NextStep-1 [59] directly model text and image tokens within an autoregressive Transformer, where multilingual capability is intrinsic to the pretrained LLM backbone [14, 63, 74]. Advanced methods like Qwen-Image [70] and Omni Difusion [56] move beyond conventional pipelines by unifying language and visual modeling, where multilingual capability arises from the shared modeling space and training data.

To specifically enhance multilingual capability, one direction leverages strong multilingual encoders such as AltDifusion [75] with AltCLIP [10], another focuses on encoder-generator alignment with lightweight adapters (GlueGen [43], MuLan [72]), and a third exploits parameter-eficient distillation from English teachers (PEA-Difusion [36], X2I [37]).

Multilingual Text Rendering. The ability to generate specified text within images serves as a key indicator of a T2I model’s linguistic competence. Recent advances such as Glyph-ByT5 [33, 34], AnyText [65, 66], and EasyText [35] have introduced specialized approaches that incorporate glyph-aware encoders, OCR-guided features, or DiT to improve multilingual text rendering. Meanwhile, general-purpose models [29, 58, 70] have begun to emphasize text generation. Nevertheless, multilingual text rendering remains limited in both capability and systematic evaluation.

Language-related Bias and Cross-lingual Efects. In NLP, cross-lingual behavior has been extensively analyzed [24, 41, 44, 48, 52], with studies showing significant linguistic inequality across languages [4, 25, 45, 49, 78]. Building on this, recent work has begun to investigate biases in T2I models more broadly [11, 17, 68], such as social [3, 28], cultural [27, 39, 76], and geographic biases [2, 20]. However, these studies are still largely conducted with English prompts, making it dificult to disentangle intrinsic model biases from language-dependent generation patterns.

Despite these eforts, research on cross-lingual efects in T2I models remains limited and has mostly focused on isolated specific phenomena or narrow technical aspects [18, 23, 26, 67], such as diferences in concept coverage [50, 75], the efect of non-Latin characters [55], or case studies targeting individual languages [38]. Moreover, NeoBabel [15] studies native multilingual generation and evaluates cross-lingual consistency and code-switching robustness. However, systematic investigations into inherent linguistic inequal ity, multi-dimensional trade-ofs, and latent language-dependent generation patterns remain largely underexplored.

## 3 Cross-lingual Benchmark: LingT2I

## 3.1 Benchmark Coverage

Task Selection. The multilingual setting brings two fundamental challenges. First, models must understand prompts in diferent languages and still generate images that are semantically accurate and culturally coherent. Second, they must be able to render text faithfully across diverse writing systems, each with its own glyph complexity, layout, and formatting rules. To capture these challenges in a structured way, LingT2I defines two evaluation tasks: Content Generation and Text Rendering.

Language Coverage. To align with both Content Generation and Text Rendering, our language set balances cultural and semantic diversity and writing-system variety. Linguistic branches ground prompts in distinct cultural and semantic contexts that shape interpretation, whereas writing systems (e.g., glyph complexity, reading direction, segmentation, and character composition) directly determine the dificulty of text rendering. Guided by this dual perspective, LingT2I covers 10 languages spanning diverse scripts and families (Table 1). The selection balances population size [16], global coverage [5], and the Power Language Index (PLI) [6], ensuring representativeness and practical relevance. For each language, we annotate its script type and linguistic branch<sup>1</sup>, providing structured background for subsequent cross-lingual and cultural analyses.

Table 1: Statistics and classification of the 10 languages in LingT2I, including speaker population (Spk., billion) [16], global coverage (Cov., %) [5], Power Language Index (PLI) [6], script type, and language branch.
<table><tr><td>Branch</td><td>Language</td><td>Code</td><td>Spk.</td><td>Cov.</td><td>PLI</td><td>Script</td></tr><tr><td>Germanic</td><td>English</td><td>EN</td><td>1.50</td><td>18.8</td><td>0.89</td><td>Latin</td></tr><tr><td>Sinitic</td><td>Chinese</td><td>ZH</td><td>1.20</td><td>13.8</td><td>0.41</td><td>Han</td></tr><tr><td>Indo-Aryan</td><td>Hindi</td><td>HI</td><td>0.61</td><td>7.5</td><td>0.12</td><td>Devanagari</td></tr><tr><td>Romance</td><td>Spanish</td><td>ES</td><td>0.56</td><td>6.9</td><td>0.33</td><td>Latin</td></tr><tr><td>Semitic</td><td>Arabic</td><td>AR</td><td>0.34</td><td>3.4</td><td>0.27</td><td>Arabic</td></tr><tr><td>Romance</td><td>French</td><td>FR</td><td>0.31</td><td>3.4</td><td>0.34</td><td>Latin</td></tr><tr><td>Romance</td><td>Portuguese</td><td>PT</td><td>0.27</td><td>3.2</td><td>0.12</td><td>Latin</td></tr><tr><td>Slavic</td><td>Russian</td><td>RU</td><td>0.25</td><td>3.2</td><td>0.24</td><td>Cyrillic</td></tr><tr><td>Japonic</td><td>Japanese</td><td>JA</td><td>0.13</td><td>1.7</td><td>0.05</td><td>Mixed</td></tr><tr><td>Koreanic</td><td>Korean</td><td>KO</td><td>0.08</td><td>1.0</td><td>0.13</td><td>Alphabetic</td></tr></table>

## 3.2 Content Generation Subset

Evaluation Dimensions. Inspired by existing benchmarks [31, 77], we systematically organize a set of 10 evaluation dimensions that cover four fundamental aspects of image generation: Image Quality, Task Alignment, Diversity, and Robustness. As shown in Figure 2, these dimensions enable a systematic characterization of multidimensional trade-ofs in multilingual generation.

Annotation Pipeline. We construct the Content Generation subset based on the DOCCI dataset [40]. In our setting, we only utilize the textual component as the source corpus. For each caption �, its oficial annotation includes multiple aspects of the image, such as objects, attributes, spatial relationships, and scene descriptions. To align with the predefined evaluation dimension set D, we design a dimension-aware annotation and prompt construction pipeline.

Specifically, we employ designed prompts to guide Gemini 2.5 Flash [13] to extract dimension-relevant semantic information from the original caption �, denoted as ${ \cal T } _ { d } = { \cal M } ( c , d )$ for each target dimension $d \in { \mathcal { D } } .$ . This process emphasizes the semantic components most relevant to the target dimension. For dimensions with explicit information in the caption (e.g., Content Alignment, Realism), $\textstyle { \mathcal { I } } _ { d }$ is further fed to the annotation model to generate concise and dimension-focused prompts $p _ { d } = { \cal M } ( { \cal T } _ { d } )$ . For dimensions that are not explicitly reflected in the original caption (e.g., Style, Bias), we first instruct the model to compress the description $c ,$ and then perform conditional expansion. For instance, we append control phrases such as “in � style” to explicitly guide the T2I model toward generating outputs that satisfy the target dimension. Finally, for the Toxicity dimension, we directly adopt the existing Toxigen [21] dataset to avoid introducing additional harmful content.

All generated English prompts $\mathit { p } _ { d }$ are then translated into nine additional languages using Gemini 2.5 Pro [13], with constraints to preserve semantic consistency, cultural appropriateness, and stylistic fidelity. Details can be found in Section 3.4.

Data Statistics. In total, this subset comprises 30K prompts, distributed evenly across 10 dimensions and 10 languages (300 prompts per dimension per language). As shown in Appendix B.2, the English subset averages 21.9 words, while all the multilingual prompts average 43.1 tokens with the mT5 tokenizer [73].

Evaluation. i) General Evaluation: CLIPScore [22] is a widely used metric that measures image-text alignment by computing the cosine similarity between the generated image and its prompt using CLIP embeddings. However, the original CLIP [46] exhibits much stronger performance in English than in other languages [69]. To address this, we replace CLIP with the multilingual encoder Meta-CLIP2 [12], which provides a fairer measure across languages.

ii) Dimensional Evaluation: TRIGScore [77] is an MLLM-based evaluation metric that leverages log-probabilities to produce finegrained scores across multiple quality dimensions. We adapt the Qwen-2.5-VL [60] Model and redesign the evaluation prompts to explicitly instruct the model to consider language-specific factors, enabling it to directly account for cross-linguistic understanding. Details can be found in Appendix C.

## 3.3 Text Rendering Subset

Evaluation Dimensions. In the Text Rendering task, we shift the focus of analysis to the text itself, using Textual Quality and Harmony with the Background as the two primary dimensions. Figure 3 shows the detailed dimension definitions and examples.

Annotation Pipeline. We use English samples from EasyText [35] as the source of raw prompts. We keep the background prompt � fixed in English and only translate the rendered text �, i.e., $( c , t _ { \mathrm { e n } } ) $ (�, �<sub>ℓ</sub>), thereby isolating language variation to the text rendering component. This design allows us to focus specifically on rendering performance, while also aligning with the fact that most Text Rendering models are primarily optimized for English prompts. The translations into nine additional languages are also performed using Gemini 2.5 Pro [13].

![](images/f0d8c3c96aed9328ec5dfee05bf644d21a73866303edf14b802467c9e7ad1625.jpg)  
Figure 2: Evaluation Dimensions of Content Generation Task. For each dimension, we provide its definition, multilingual examples, and representative examples of both high-quality and failure cases in generated images.

![](images/a0acdd485784bf44f07292799ee9888e875d4165a14b84060b009a6094ee819b.jpg)  
Figure 3: Evaluation Dimensions of Text Rendering Task.

Data Statistics. The Text Rendering subset contains 3K samples, with 300 prompts per language. As shown in Appendix B.2, each prompt specifies a multilingual text string to be rendered, which averages 3.0 tokens with mT5 tokenizer, accompanied by an English background description averaging 82.3 words and 120.2 tokens. Evaluation. i) General Evaluation: Precision is the primary metric for text rendering, reflecting the correctness of the generated text.

We evaluate text rendering using standard precision metrics, including character-level NED [30], token-level NED, and sentence-level accuracy. We report the average of these metrics as the final score.

ii) Dimensional Evaluation: We follow the MLLM-as-judge framework in EasyText [35] and implement it using Gemini 2.5 Flash as the evaluation model. The prompts are adapted to specify the target language and explicitly guide the model to account for languagespecific characteristics across diferent writing systems.

## 3.4 Quality Control

We adopt a three-part quality control process for dataset construction: Automatic Processing and Verification, where all automatic processing steps for both Content Generation and Text Rendering are performed using Gemini 2.5 Pro and verified through back-translation and GPT-5 cross-checking, with problematic cases manually corrected; Error Analysis and Iterative Refinement, where pilot experiments are conducted on a randomly sampled 5% subset to identify common data issues and refine prompt construction and filtering before large-scale generation; and Human Quality Check, where native speakers evaluate another randomly sampled 5% subset, with 98% of the samples judged to be semantically consistent across languages. More details of this section can be found in Appendix B.

## 4 Experiments

Implementation Details. All the experiments are conducted on 4 NVIDIA A100 64G GPUs. We evaluate 17 recent text-to-image models for the two tasks, including general-purpose models widely used for English prompts, models specifically trained or adapted for multilingual generation and text rendering models, all deployed with default settings. (see Appendix D.1). During Evaluation, we use metaclip-2-worldwide-huge-quickgelu [12] for CLIPScore, and Qwen-2.5-VL 72B [60] for TRIGScore, and Gemini 2.5 Flash [13] and mT5-base [73] for text rendering average precision.

Table 2: Overall cross-linguistic performance of Content Generation (CG) and Text Rendering (TR) models. In CG task, results are measured by CLIPScore ↑; In TR task, results are measured by Average Precision ↑. For each model we report the average (Avg. ↑) and standard deviation (Std. ↓) across languages, where the variance indicates model-level linguistic inequality. We also provide per-language averages by model category, highlighting the language-level disparities.
<table><tr><td>Model</td><td>English</td><td>Chinese</td><td>Hindi</td><td>Spanish</td><td>Arabic</td><td>French</td><td>Portuguese</td><td>Russian</td><td>Japanese</td><td>Korean</td><td>Avg.</td><td>Std.</td></tr><tr><td colspan="10">- Content Generation Task (General-purpose Models)</td><td></td><td></td><td></td><td></td></tr><tr><td>SD3.5 [54]</td><td>0.79</td><td>0.33</td><td>0.24</td><td>0.71</td><td>0.30</td><td>0.74</td><td>0.67</td><td>0.42</td><td>0.36</td><td>0.27</td><td>0.48</td><td>0.21</td></tr><tr><td>SDXL [42]</td><td>0.78</td><td>0.38</td><td>0.31</td><td>0.66</td><td>0.31</td><td>0.69</td><td>0.62</td><td>0.34</td><td>0.41</td><td>0.31</td><td>0.48</td><td>0.17</td></tr><tr><td>FLUX.1-Krea [29]</td><td>0.77</td><td>0.30</td><td>0.33</td><td>0.70</td><td>0.30</td><td>0.73</td><td>0.66</td><td>0.44</td><td>0.30</td><td>0.26</td><td>0.48</td><td>0.19</td></tr><tr><td>Sana 1.5 [71]</td><td>0.78</td><td>0.73</td><td>0.45</td><td>0.75</td><td>0.42</td><td>0.75</td><td>0.73</td><td>0.65</td><td>0.52</td><td>0.46</td><td>0.62</td><td>0.14</td></tr><tr><td>PixArt-Σ [7]</td><td>0.76</td><td>0.31</td><td>0.28</td><td>0.68</td><td>0.29</td><td>0.71</td><td>0.66</td><td>0.51</td><td>0.29</td><td>0.28</td><td>0.48</td><td>0.20</td></tr><tr><td>Janus-Pro [9]</td><td>0.75</td><td>0.54</td><td>0.31</td><td>0.70</td><td>0.32</td><td>0.70</td><td>0.67</td><td>0.52</td><td>0.48</td><td>0.36</td><td>0.54</td><td>0.16</td></tr><tr><td>Lumina-Next [19]</td><td>0.71</td><td>0.62</td><td>0.48</td><td>0.64</td><td>0.50</td><td>0.64</td><td>0.62</td><td>0.61</td><td>0.60</td><td>0.52</td><td>0.59</td><td>0.06</td></tr><tr><td>Z-Image [62]</td><td>0.64</td><td>0.63</td><td>0.42</td><td>0.60</td><td>0.55</td><td>0.60</td><td>0.58</td><td>0.59</td><td>0.61</td><td>0.57</td><td>0.58</td><td>0.05</td></tr><tr><td>Qwen-Image [70]</td><td>0.82</td><td>0.78</td><td>0.71</td><td>0.79</td><td>0.75</td><td>0.81</td><td>0.80</td><td>0.78</td><td>0.78</td><td>0.77</td><td>0.78</td><td>0.03</td></tr><tr><td>Omni-Diffusion [56]</td><td>0.72</td><td>0.66</td><td>0.31</td><td>0.61</td><td>0.34</td><td>0.62</td><td>0.56</td><td>0.44</td><td>0.45</td><td>0.44</td><td>0.52</td><td>0.13</td></tr><tr><td>Average</td><td>0.78</td><td>0.48</td><td>0.38</td><td>0.71</td><td>0.38</td><td>0.73</td><td>0.69</td><td>0.52</td><td>0.45</td><td>0.39</td><td></td><td></td></tr><tr><td colspan="10">– Content Generation Task (Multilingual-enhanced Models) &lt;Denotes basic model&gt;</td><td></td><td></td><td></td><td></td></tr><tr><td>PEA &lt;FLUX&gt; [36]</td><td>0.64</td><td>0.68</td><td>0.60</td><td>0.63</td><td>0.61</td><td>0.66</td><td>0.65</td><td>0.62</td><td>0.61</td><td>0.59</td><td>0.63</td><td>0.03</td></tr><tr><td>X2I &lt;FLUX&gt; [37]</td><td>0.72</td><td>0.70</td><td>0.56</td><td>0.65</td><td>0.61</td><td>0.65</td><td>0.64</td><td>0.65</td><td>0.62</td><td>0.63</td><td>0.64</td><td>0.04</td></tr><tr><td>MuLan &lt;PixArt&gt; [72]</td><td>0.73</td><td>0.71</td><td>0.59</td><td>0.71</td><td>0.65</td><td>0.72</td><td>0.71</td><td>0.70</td><td>0.69</td><td>0.65</td><td>0.69</td><td>0.04</td></tr><tr><td>Average</td><td>0.70</td><td>0.70</td><td>0.58</td><td>0.67</td><td>0.62</td><td>0.68</td><td>0.67</td><td>0.66</td><td>0.64</td><td>0.62</td><td></td><td></td></tr><tr><td colspan="10">- Text Rendering Task (General-purpose Models)</td><td></td><td></td><td></td><td></td></tr><tr><td>Nano Banana [58]</td><td>0.68</td><td>0.28</td><td>0.46</td><td>0.57</td><td>0.13</td><td>0.58</td><td>0.56</td><td>0.47</td><td>0.55</td><td>0.61</td><td>0.49</td><td>0.16</td></tr><tr><td>Qwen-Image [70]</td><td>0.69</td><td>0.69</td><td>0.12</td><td>0.64</td><td>0.15</td><td>0.63</td><td>0.63</td><td>0.32</td><td>0.56</td><td>0.41</td><td>0.48</td><td>0.21</td></tr><tr><td>FLUX.1-Krea [29]</td><td>0.65</td><td>0.15</td><td>0.08</td><td>0.56</td><td>0.09</td><td>0.54</td><td>0.56</td><td>0.09</td><td>0.14</td><td>0.10</td><td></td><td>0.23</td></tr><tr><td>Average</td><td>0.67</td><td>0.37</td><td>0.22</td><td>0.59</td><td>0.12</td><td>0.58</td><td>0.58</td><td>0.29</td><td>0.42</td><td>0.37</td><td>0.30</td><td>-</td></tr><tr><td colspan="10">– Text Rendering Task (Rendering-oriented Models)</td><td></td><td></td><td></td><td></td></tr><tr><td>Anytext [66]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Anytext2 [65]</td><td>0.39 0.56</td><td>0.33 0.44</td><td>0.06 0.11</td><td>0.34 0.51</td><td>0.05 0.08</td><td>0.34 0.50</td><td>0.32 0.50</td><td>0.14 0.14</td><td>0.19 0.33</td><td>0.20 0.30</td><td>0.24 0.35</td><td>0.12 0.17</td></tr><tr><td>EasyText [35]</td><td>0.87</td><td>0.74</td><td>0.46</td><td>0.80</td><td>0.43</td><td>0.80</td><td>0.79</td><td>0.59</td><td>0.68</td><td>0.56</td><td></td><td></td></tr><tr><td>Average</td><td>0.61</td><td>0.50</td><td>0.21</td><td>0.55</td><td>0.19</td><td>0.55</td><td>0.54</td><td>0.29</td><td>0.40</td><td>0.35</td><td>0.67</td><td>0.14 -</td></tr></table>

## 4.1 Cross-lingual Inequality Analysis

## 4.1.1 Content Generation Task.

General-purpose models exhibit substantially higher linguistic inequality than multilingual enhanced models. As shown in Table 2, we report both the average performance and the variance across languages for each model, with the variance indicating linguistic inequality. Results indicate that multilingual-enhanced models exhibit much lower variance, suggesting more balanced cross-lingual performance, while most general-purpose models suffer from severe linguistic inequality, with Qwen-Image, Z-Image, and Lumina-Next as notable exceptions.

Native multilingual architectures achieve better fairness than post-hoc adaptations. As shown in Table 2, multilingual-enhanced variants yield higher fairness (lower variance) than their base models, but this often comes at the cost of reduced performance in privileged languages such as English and French. In contrast, Qwen-Image (built upon Qwen-2.5-VL) achieves comparably low variance (0.03) while maintaining superior overall quality, suggesting that native multilingual architectures ofer a more efective path toward fairness than adapter- or distillation-based post-hoc methods.

Even with reduced inequality, performance remains stratified across language families and cultures. Using the classification in Table 1, we analyze results from language branch and cultural perspectives. Under general-purpose models, Germanic and Romance language branches lead (EN=0.78; FR=0.73; ES=0.71; PT=0.69), while Slavic (RU=0.52) and East Asian languages (CJK) trail; Indo-Iranian and Semitic are lowest (HI/AR=0.38). With multilingual-enhanced variants, branch means narrow but persist: Chinese joins the top, Romance and Slavic converge around 0.66- 0.68, while other groups remain lower despite notable gains (e.g., JA=0.64, KO=0.62, HI=0.58, AR=0.62). Thus, even with improved fairness, language family and cultural stratification endures.

## 4.1.2 Text Rendering Task.

All models exhibit strong linguistic inequality. As shown in the Text Rendering section of Table 2, large variances remain across all models, indicating that linguistic inequality persists regardless of model category. Overall text-rendering ability is weak—even models specialized for rendering struggle. Among them, EasyText achieves a more balanced trade-of between overall performance (0.67) and fairness (0.14), yet language disparities remain pronounced.

![](images/7757586ec4676d22cac559b17a06ffff0572ba3080ddd2ef293f032e3259ad7c.jpg)  
(<sub>a</sub>) Q<sub>we</sub>n<sub>-</sub>Im<sub>a</sub>g<sub>e</sub>

![](images/8d0bd2a40c49e153588ee4e676b54627283d0f04d4f6e4bc39caeb1f2b75118f.jpg)  
(b) MuL<sub>a</sub>n <Pi<sub>x</sub>Art>

![](images/7510ee6e3c60107564b1a061d85b846d188b7e85ea86081fbd4751154f4dbec5.jpg)  
(<sub>e</sub>) Q<sub>we</sub>n<sub>-</sub>Im<sub>a</sub>g<sub>e</sub>: T<sub>ox</sub> <sub>vs</sub> Sty

![](images/c4849ef55e4bd5a2d031729d57f58a3651b52fe5a332b369556f5d72e0d588d8.jpg)

![](images/12a43ae67194036014c2f82e93514dc936487df06a0d61f9cea8f0f6aa086271.jpg)

![](images/46962a28fcf7b282d4fcbc63c3d0d0e688f76c7f7356b4dd1635a013ce921b8a.jpg)  
(f) N<sub>a</sub>n<sub>o</sub> B<sub>a</sub>n<sub>a</sub>n<sub>a</sub>: Alignm<sub>e</sub>nt <sub>vs</sub> Pr<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n  
Figure 4: Cross-lingual dimension analysis. Language-dimension correlations in Content Generation (a, b) and Text Rendering (c, d). Language-dependent trade-ofs between key dimension pairs in Content Generation (e) and Text Rendering (f). This analysis is based on fine-grained results in Table 6 and Table 7 (in Appendix), derived from models with strong multilingual fairness.

Performance across writing systems is particularly uneven. From the perspective of writing systems, languages using the Latin alphabet (EN, ES, PT, FR) consistently perform best, maintaining leading results in both general-purpose and rendering-oriented models. Chinese shows clear improvement in rendering-oriented models, while other non-Latin scripts remain consistently weaker. The Content Generation and the Text Rendering tasks demand diferent multilingual capabilities. While Qwen-Image achieves a strong cross-lingual average and high fairness in Content Generation task, it shows pronounced linguistic inequality in Text Rendering task: English and Chinese remain relatively strong, whereas Arabic, Hindi, and Korean lag substantially, indicating that the multilingual capabilities required are not interchangeable.

## 4.2 Cross-lingual Multi-dimensional Analysis

Language-Dimension Correlation. Figure 4 (a-d) shows how linguistic and typological variations afect fine-grained model behavior. In the Content Generation task, models reveal a strong bias: high-resource Indo-European languages (e.g., English, Germanic, Romance) favor white and male characters, reflecting social skew in English-centric corpora. Non-Indo-European languages (e.g., Indo-Aryan, Semitic, Koreanic, Slavic) yield numerically lower bias and more diverse depictions (Figure 4 (a)(b)), though this largely results from weaker semantic grounding rather than genuine fairness. Language background also impacts toxicity. Indo-Aryan, Semitic,

Koreanic, and Slavic achieve higher Toxicity scores—meaning fewer harmful elements—than Sinitic and Germanic. This may stem from lower data exposure, causing models to generate safer yet generic content, and from moderation pipelines tuned for English, which may over-filter other languages. In the Text Rendering task, Sinitic and Semitic languages show the lowest Precision and Quality, with frequent broken or malformed glyphs (Figure 4 (c)(d)). By contrast, Germanic and Romance languages perform best, benefiting from Latin-script familiarity. These trends expose structural weaknesses in handling non-Latin scripts.

Language-dependent Trade-ofs. Beyond individual metrics, languages also reshape how models balance dimensions (Figure 4 (e)(f)). For Qwen-Image in Content Generation, Toxicity-Style tradeofs vary by language: Hindi and Arabic produce safer but less stylistically consistent images, while English emphasizes coherent aesthetics at the cost of higher cultural bias. Romance languages maintain a better balance, likely due to closer linguistic and cultural proximity to English. For Nano Banana in Text Rendering, the Alignment-Precision relation is language-dependent. Germanic and Romance maintain stable precision even at high alignment, whereas Sinitic, Slavic, and Indo-Aryan degrade sharply—reflecting the complex and dense structure of their scripts. Overall, these results highlight persistent limitations in multilingual T2I systems’ ability to achieve robust visual-linguistic grounding across diverse writing systems.

Image Quality - Aesthetics

![](images/12709c96d4183f570f24326b9afa8a35279e9e576e23fa758484e7f842e02d51.jpg)  
Figure 5: Distribution of race and gender categories across ten languages, computed from Qwen-Image outputs under the Bias dimension.

## 4.3 Language-dependent Generation Patterns

Detailed analysis procedures, including automated analysis and statistical estimation methods, are provided in Appendix E.

Demographic Bias. As shown in Figure 5, our analysis reveals a pronounced demographic bias across all languages, with consistent over-representation of male subjects and specific racial groups. Notably, a strong language-demographic alignment is observed: generated images tend to reflect the dominant ethnic characteristics of each language’s primary regions. For example, Hindi prompts predominantly yield Indian subjects (94.6%), while Japanese and Korean prompts produce a high proportion of Asian subjects (over 65%). In contrast, Western languages such as Russian, French, and English show a strong bias toward White-presenting subjects (84.9%–97.1%). These results suggest that model outputs are shaped by demographic distributions embedded in the training data.

Cultural Tendency. As illustrated in the representative example in Figure 6, the prompt “woman statue with seahorses sitting” shows clear cross-lingual variation in cultural style: the Hindi version reflects traditional Indian sculptural aesthetics, while the Japanese version aligns with East Asian visual conventions, including regionally suggestive elements such as a carp.

More importantly, this is not an isolated case. Based on statistics from Qwen-Image outputs, among all valid samples, 23.2% of generated images contain identifiable culture-specific visual elements. Once explicit cultural cues appear, they tend to align strongly with the cultural region associated with the prompt language: 79.6% have a primary culture tag that matches the prompt language, and this proportion further rises to 90.2% when considering only samples assigned to a specific known culture.

While prior studies [1, 17, 68] report “Westernization” bias in T2I models under English settings, our multilingual results do not support this. Western cultural tags account for only 3.7% of valid samples, and only 1.2% of non-Western prompts shift toward Western culture. Instead, multilingual prompting steers generation toward language-specific cultural aesthetics, expressed through cues such as writing systems, architecture, clothing, and symbolic objects.

![](images/0739882758be665264eb0dbe24976f323b9bf66891861bf0fc974f950ebe70f3.jpg)  
Prompt: An outdoor shot, looking up at the golden statue of a woman with three mythical seahorses siting atop a gray brick monument. She holds a branch and cylinder. Seahorses are dynamic. Clear blue sky. Daytime.  
Figure 6: Language-dependent cultural tendencies, showing how identical prompts produce culturally specific visual interpretations across languages, reflecting implicit cultural priors associated with each language.

Rendering Errors. As shown in Figure 7, rendering errors vary substantially across writing systems, with fundamentally diferent failure modes in Latin (English, French, Spanish) and CJK (Chinese, Japanese, Korean) scripts due to their distinct linguistic and structural properties. In the top panels of Figure 8, character-level errors exhibit clear category-specific patterns. In Latin, errors are concentrated in the lowercase bucket-especially for Nano Bananaindicating unstable case consistency despite largely preserved character identity. In contrast, CJK error rates correlate strongly with stroke count: EasyText remains stable until high-complexity thresholds, whereas Nano Banana shows consistently high error rates across all stroke levels, suggesting sensitivity to glyph complexity. The bottom panels of Figure 8 further reveal positional diferences. Latin rendering follows a “stable prefix, fragile sufix” pattern, with errors accumulating toward the end of the sequence. By contrast, CJK shows a breakdown of sequence integrity: EasyText degrades after initial positions, while NanoBanana exhibits high error rates from the outset. Overall, Latin errors reflect gradual positional drift, whereas CJK errors indicate structural collapse of the sequence.

## 4.4 Causal Analysis

Failure Pattern Analysis. For the text rendering task, beyond the language-level precision scores in Table 3, we identify three failure patterns and use GPT-5 to estimate their frequencies (Figure 9). At the model level, the glyph-conditioned pipeline EasyText is slightly dominated by script-specific structural errors (39.2% of erroneous outputs; 36.9% semantic substitution), whereas the semantic-prior model NanoBanana is dominated by semantic substitution (50.0%; 40.5% structural errors), with script-selection or romanization failure less frequent overall (23.9% vs. 9.5%). Together, these patterns indicate that exact-string rendering is particularly challenging for non-Latin scripts: glyph-conditioned models are more susceptible to structural errors, whereas semantic-prior models tend to preserve meaning while failing to reproduce the requested string.

![](images/c72b6763f3fa1142b09f179a4fe5061f26cea6f3aefae374fcc2d206443cc676.jpg)  
Figure 7: Language-dependent text rendering errors, showing variations in character correctness and structural fidelity across diferent writing systems, with distinct error patterns emerging for alphabetic and non-alphabetic scripts.

![](images/ed9addb586980ea30a4605e6b30922b7cd5909ddce9a766774e7549b6640b4da.jpg)

![](images/eab10821b3a48f500776001cdb0b9ecfadb337e8cd2b3e0238767a0ae3e11659.jpg)  
Figure 8: Analysis of text rendering errors across writing systems. (Top) Character-level error rates, where Latin scripts are grouped by character category (uppercase/lowercase), and CJK scripts are grouped by stroke count (character complexity). (Bottom) Error rates grouped by relative position in the sequence, where position denotes the normalized character position from start to end, enabling comparison of error pat terns across Latin and CJK scripts.

Table 3: Text-rendering precision across languages for Easy-Text and Nano Banana.
<table><tr><td>Model</td><td>EN</td><td>ZH</td><td>HI</td><td>ES</td><td>AR</td><td>FR</td><td>PT</td><td>RU</td><td>JA</td><td>KO</td><td>Mean</td></tr><tr><td>EasyText</td><td>0.876</td><td>0.761</td><td>0.492</td><td>0.817</td><td>0.450</td><td>0.816</td><td>0.801</td><td>0.611</td><td>0.706</td><td>0.584</td><td>0.691</td></tr><tr><td>Nano Banana</td><td>0.691</td><td>0.311</td><td>0.490</td><td>0.579</td><td>0.145</td><td>0.595</td><td>0.575</td><td>0.482</td><td>0.575</td><td>0.637</td><td>0.508</td></tr></table>

![](images/40e64c3c3a4761b669efced550da3e955bb5fd99580e063a4548ce791dc893a8.jpg)  
Figure 9: Representative failure patterns in multilingual text rendering.

Transliteration Control. To determine whether non-Latin scripts drive the cross-lingual alignment gap, we replace the original scripts with Latin transliterations for 450 Arabic, Hindi, and Chinese samples from the Content Alignment (TA-C) dimension. Transliteration did not improve content alignment; scores dropped from 0.78 to 0.49 on average (AR: 0.80→0.30, HI: 0.68→0.60, ZH: 0.87→0.57). These results indicate that cross-lingual alignment depends on more than the surface form of the writing system. The degradation after transliteration is consistent with limitations in language-specific text representations and uneven multilingual training coverage.

Alignment-conditioned Bias. To separate genuine demographic bias from errors caused by weak semantic alignment, we group images from the Bias dimension into shared CLIPScore intervals and recompute the bias score within each alignment range. As shown in Figure 10, lower bias scores are concentrated in poorly aligned samples, indicating stronger demographic imbalance when generated content fails to reflect the prompt faithfully. Although bias scores increase with alignment, demographic imbalance remains evident in highly aligned samples. These results show that weak alignment amplifies demographic imbalance, while cross-lingual bias persists after controlling for alignment.

<table><tr><td rowspan=4 colspan=1>LowMidHigh</td><td rowspan=1 colspan=1>.51</td><td rowspan=1 colspan=1>.64</td><td rowspan=1 colspan=1>.47</td><td rowspan=1 colspan=1>.58</td><td rowspan=1 colspan=1>.67</td><td rowspan=1 colspan=1>.46</td><td rowspan=1 colspan=1>.55</td><td rowspan=1 colspan=1>.55</td><td rowspan=1 colspan=1>.70</td><td rowspan=4 colspan=2>0.80.70.60.50.4ZH</td></tr><tr><td rowspan=1 colspan=1>.58</td><td rowspan=1 colspan=1>.51</td><td rowspan=1 colspan=1>.70</td><td rowspan=1 colspan=1>.65</td><td rowspan=1 colspan=1>.72</td><td rowspan=1 colspan=1>.44</td><td rowspan=1 colspan=1>.49</td><td rowspan=1 colspan=1>.65</td><td rowspan=1 colspan=1>.72</td><td rowspan=1 colspan=2>.48</td></tr><tr><td rowspan=1 colspan=1>.51</td><td rowspan=1 colspan=1>.57</td><td rowspan=1 colspan=1>.59</td><td rowspan=1 colspan=1>.61</td><td rowspan=1 colspan=1>.79</td><td rowspan=1 colspan=1>.60</td><td rowspan=1 colspan=1>.61</td><td rowspan=1 colspan=1>.62</td><td rowspan=1 colspan=1>.74</td><td rowspan=1 colspan=2>.55</td></tr><tr><td rowspan=1 colspan=1>AR</td><td rowspan=1 colspan=1>EN</td><td rowspan=1 colspan=1>ES</td><td rowspan=1 colspan=1>FR</td><td rowspan=1 colspan=1>HI</td><td rowspan=1 colspan=1>JA</td><td rowspan=1 colspan=1>KO</td><td rowspan=1 colspan=1>PT</td><td rowspan=1 colspan=1>RU</td></tr></table>

Figure 10: Bias scores across content-alignment buckets.

Culture-tag Analysis. To separate the efect of prompt language from explicit cultural conditioning, we compare three versions of the same English source prompts: translated non-English prompts, English prompts with an explicit culture tag (e.g., “in Hindi style”), and culture-neutral English prompts. Across 600 Qwen-Image samples, the target-culture rates are 32.5%, 75.0%, and 0.0%, respectively. These results show that prompt language directly steers cultural visual tendencies, while explicit culture tags impose a stronger cultural prior.

![](images/d13b1f2319c29d8af3ccb04076b3060aacaaaf2909ae476a66552acdd6bdcf4c.jpg)  
Figure 11: Prompt fragmentation and generation quality across languages for Qwen-Image.

Tokenization Analysis. To examine the relationship between textside representation eficiency and multilingual generation, we com pute the mean prompt-fragmentation score for each language using the Qwen-Image tokenizer and compare it with the corresponding mean CLIPScore. As shown in Figure 11, prompt fragmentation exhibits a strong negative Spearman rank correlation with CLIP-Score across the ten languages (� = −0.89). Languages represented by more fragmented token sequences consistently achieve weaker image–text alignment. This result identifies ineficient tokenization as a systematic text-side bottleneck underlying the performance gap of non-Latin languages.

## 5 Limitation and Insight

Limitation. The proposed LingT2I benchmark inevitably involves translation, which can introduce bias; however, we apply strict verification and human checks to minimize such efects. Similarly, while existing metrics are not fully language-agnostic, we adopt multilingual encoders and adapt protocols to improve fairness. Importantly, the observed performance gaps are large and consistent, and are therefore unlikely to be explained by these factors.

Insight. Our findings suggest several directions for future research. First, multilingual capability should be achieved through native architectural design rather than post-hoc adaptation. Second, training data should be organized by language family and curated with cultural grounding. Third, models should maintain balanced performance across evaluation dimensions, avoiding over-optimization toward a single aspect of quality. In addition, bias-aware data curation and translation-based augmentation may help mitigate cultural and demographic biases and improve cross-lingual fairness. Finally, given the challenges across writing systems, models could benefit from script-specific rendering modules or training strategies tailored to their structural characteristics.

## Acknowledgments

This research was funded by Khalifa University of Science and Technology through the Faculty Start-Ups under Project ID: KU-INT-FSU-2005-8474000775.

## References

[1] Saharsh Barve, Andy Mao, Jiayue Melissa Shi, Prerna Juneja, and Koustuv Saha. 2025. Can we Debias Social Stereotypes in AI-Generated Images? Examining Text-to-Image Outputs and User Perceptions. arXiv preprint arXiv:2505.20692 (2025).

[2] Abhipsa Basu, R Venkatesh Babu, and Danish Pruthi. 2023. Inspecting the geo graphical representativeness of images from text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5136–5147.

[3] Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. 2023. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In Proceedings of the 2023 ACM conference on fairness, accountability, and transparency. 1493–1504.

[4] Damian Blasi, Antonios Anastasopoulos, and Graham Neubig. 2022. Systematic inequalities in language technology performance across the world’s languages. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 5486–5505.

[5] Central Intelligence Agency. 2025. The World Factbook. https://www.cia.gov/theworld-factbook/. Accessed: 2025-09-08.

[6] Kai L Chan. 2016. Power language index. Which are the world’s most influential languages (2016).

[7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. 2024. Pixart-�: Weak-tostrong training of difusion transformer for 4k text-to-image generation. In European Conference on Computer Vision. Springer, 74–91.

[8] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Pi otr Dollár, and C Lawrence Zitnick. 2015. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325 (2015).

[9] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. 2025. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811 (2025).

[10] Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Qinghong Yang, and Ledell Wu. 2023. Altclip: Altering the language encoder in clip for extended language capabilities. In Findings ofthe Association for Computational Linguistics: ACL 2023. 8666–8682.

[11] Aditya Chinchure, Pushkar Shukla, Gaurav Bhatt, Kiri Salij, Kartik Hosanagar, Leonid Sigal, and Matthew Turk. 2024. Tibet: Identifying and evaluating biases in text-to-image generative models. In European Conference on Computer Vision. Springer, 429–446.

[12] Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, et al.

2025. Meta CLIP 2: A Worldwide Scaling Recipe. arXiv preprint arXiv:2507.22062 (2025).

[13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multi modality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025).

[14] DeepSeek-AI. 2024. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism. arXiv preprint arXiv:2401.02954 (2024). https://github.com/ deepseek-ai/DeepSeek-LLM

[15] Mohammad Mahdi Derakhshani, Dheeraj Varghese, Marzieh Fadaee, and Cees GM Snoek. 2025. NeoBabel: A multilingual open tower for visual gen eration. arXiv preprint arXiv:2507.06137 (2025).

[16] David M. Eberhard, Gary F. Simons, and Charles D. Fennig. 2025. Ethnologue: Languages ofthe World. SIL International. https://www.ethnologue.com/

[17] Wala Elsharif, Mahmood Alzubaidi, and Marco Agus. 2025. Cultural Bias in Text-to-Image Models: A Systematic Review of Bias Identification, Evaluation, and Mitigation Strategies. IEEE Access 13 (2025), 122636–122659. doi:10.1109/ ACCESS.2025.3585745

[18] Felix Friedrich, Katharina Hämmerl, Patrick Schramowski, Manuel Brack,Jindřich Libovicky, Alexander Fraser, and Kristian Kersting. 2025. Multilingual text-to-\` image generation magnifies gender stereotypes. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 19656–19679.

[19] Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, et al. 2024. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large difusion transformers. arXiv preprint arXiv:2405.05945 (2024).

[20] Melissa Hall, Candace Ross, Adina Williams, Nicolas Carion, Michal Drozdzal, and Adriana Romero Soriano. 2023. Dig in: Evaluating disparities in image generations with indicators for geographic diversity. arXiv preprint arXiv:2308.06198 (2023).

[21] Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. 2022. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. In Proceedings ofthe 60th annual meeting of the association for computational linguistics (volume 1: Long papers). 3309–3326.

[22] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In EMNLP.

[23] Carolin Holtermann, Florian Schneider, and Anne Lauscher. 2026. SoS: Analysis of Surface over Semantics in Multilingual Text-To-Image Generation. In Proceedings ofthe 19th Conference ofthe European Chapter ofthe Association for Computational Linguistics (Volume 1: Long Papers). 3955–3995.

[24] Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In International conference on machine learning. PMLR, 4411–4421.

[25] Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings ofthe 58th annual meeting ofthe association for computational linguistics. 6282–6293.

[26] Ryohei Kakebayashi and Tatsuya Mori. 2026. Poster: Why Do Non-English Languages Exhibit Higher Vulnerability to Data Poisoning Attacks Against Text to-Image Models? The Network andDistributedSystem Security (NDSS) Symposium (2026).

[27] Nithish Kannen, Arif Ahmad, Marco Andreetto, Vinodkumar Prabhakaran, Utsav Prabhu, Adji B Dieng, Pushpak Bhattacharyya, and Shachi Dave. 2024. Beyond aesthetics: Cultural competence in text-to-image models. Advances in Neural Information Processing Systems 37 (2024), 13716–13747.

[28] Thomas Klassert, Adrian Ulges, and Biying Fu. 2026. BAFIS: Dataset+ Framework to assess occupational Bias and Human Preference in modern Text-to-image Models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2168–2177.

[29] Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/flux.

[30] VI Lcvenshtcin. 1966. Binary coors capable or ‘correcting deletions, insertions, and reversals. In Soviet physics-doklady, Vol. 10.

[31] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, et al. 2023. Holistic evaluation of text-to-image models. Advances in Neural Information Processing Systems 36 (2023), 69981–70011.

[32] Chen Cecilia Liu, Iryna Gurevych, and Anna Korhonen. 2025. Culturally aware and adapted nlp: A taxonomy and a survey of the state of the art. Transactions of the Association for Computational Linguistics 13 (2025), 652–689.

[33] Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. 2024. Glyph-byt5: A customized text encoder for accurate visual text rendering. In European Conference on Computer Vision. Springer, 361–377.

[34] Zeyu Liu, Weicong Liang, Yiming Zhao, Bohan Chen, Lin Liang, Lijuan Wang, Ji Li, and Yuhui Yuan. 2024. Glyph-byt5-v2: A strong aesthetic baseline for accurate multilingual visual text rendering. arXiv preprint arXiv:2406.10208 (2024).

[35] Runnan Lu, Yuxuan Zhang, Jiaming Liu, Haofan Wang, and Yiren Song. 2026. Easytext: Controllable difusion transformer for multilingual text rendering. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 7565–7573.

[36] Jian Ma, Chen Chen, Qingsong Xie, and Haonan Lu. 2024. Pea-difusion: Parameter-eficient adapter with knowledge distillation in non-english text-to image generation. In European Conference on Computer Vision. Springer, 89–105.

[37] Jian Ma, Qirong Peng, Xu Guo, Chen Chen, Haonan Lu, and Zhenyu Yang. 2025. X2i: Seamless integration of multimodal understanding into difusion transformer via attention distillation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 16733–16744.

[38] Surbhi Mittal, Arnav Sudan, Mayank Vatsa, Richa Singh, Tamar Glaser, and Tal Hassner. 2024. Navigating text-to-image generative bias across indic languages. In European Conference on Computer Vision. Springer, 53–67.

[39] Shravan Nayak, Mehar Bhatia, Xiaofeng Zhang, Verena Rieser, Lisa Anne Hendricks, Sjoerd Van Steenkiste, Yash Goyal, Karolina Stańczak, and Aishwarya Agrawal. 2025. Culturalframes: Assessing cultural expectation alignment in text-to-image models and evaluation metrics. In Findings ofthe Association for Computational Linguistics: EMNLP 2025. 20918–20953.

[40] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, et al. 2024. Docci: Descriptions of connected and contrasting images. In European Conference on Computer Vision. Springer, 291–309.

[41] Fred Philippy, Siwen Guo, and Shohreh Haddadan. 2023. Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 5877–5891.

[42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2024. Sdxl: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, Vol. 2024. 1862–1874.

[43] Can Qin, Ning Yu, Chen Xing, Shu Zhang, Zeyuan Chen, Stefano Ermon, Yun Fu, Caiming Xiong, and Ran Xu. 2023. Gluegen: Plug and play multi-modal encoders for x-to-image generation. In Proceedings ofthe IEEE/CVF international conference on computer vision. 23085–23096.

[44] Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and Philip S Yu. 2025. A survey of multilingual large language models. Patterns 6, 1 (2025).

[45] Chen Qiu, Dan Oneat<sub>,</sub>ă, Emanuele Bugliarello, Stella Frank, and Desmond Elliott. 2022. Multilingual Multimodal Learning with Machine Translated Text. In Findings ofthe Association for Computational Linguistics: EMNLP 2022, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 4178–4193. doi:10.18653/v1/2022.findingsemnlp.308

[46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[47] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research 21, 140 (2020), 1–67.

[48] Sara Rajaee and Christof Monz. 2024. Analyzing the evaluation of cross-lingual knowledge transfer in multilingual language models. In Proceedings ofthe 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). 2895–2914.

[49] Surangika Ranathunga and Nisansa De Silva. 2022. Some languages are more equal than others: Probing deeper into the linguistic disparity in the NLP world. In Proceedings ofthe 2nd Conference ofthe Asia-Pacific Chapter ofthe Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 823–848.

[50] Michael Saxon and William Yang Wang. 2023. Multilingual conceptual coverage in text-to-image models. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 4831–4848.

[51] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems 35 (2022), 25278–25294.

[52] Chen Shani, Yuval Reif, Nathan Roll, Dan Jurafsky, and Ekaterina Shutova. 2026. The Roots of Performance Disparity in Multilingual Language Models: Intrinsic Modeling Dificulty or Design Choices? arXiv preprint arXiv:2601.07220 (2026).

[53] Zhan Shi, Xu Zhou, Xipeng Qiu, and Xiaodan Zhu. 2020. Improving image captioning with better use of caption. In Proceedings ofthe 58th annual meeting ofthe association for computational linguistics. 7454–7464.

[54] Stability AI. 2024. Stable Difusion 3.5. https://github.com/Stability-AI/sd3.5

[55] Lukas Struppek, Dom Hintersdorf, Felix Friedrich, Patrick Schramowski, Kristian Kersting, et al. 2023. Exploiting cultural biases via homoglyphs in text-to-image synthesis. Journal ofArtificial Intelligence Research 78 (2023), 1017–1068.

[56] Zhiyu Tan, Mengping Yang, Luozheng Qin, Hao Yang, Ye Qian, Qiang Zhou, Cheng Zhang, and Hao Li. 2024. An empirical study and analysis of text-toimage generation using large language model-powered textual representation. In European Conference on Computer Vision. Springer, 472–489.

[57] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295 (2024).

[58] Google Gemini Team. 2025. Nano Banana: Gemini AI Image Generator & Photo Editor. https://gemini.google/overview/image-generation/.

[59] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, Hongyu Zhou, Kenkun Liu, Ailin Huang, Bin Wang, Changxin Miao, Deshan Sun, En Yu, Fukun Yin, Gang Yu, Hao Nie, Haoran Lv, Hanpeng Hu, Jia Wang, Jian Zhou, Jianjian Sun, Kaijun Tan, Kang An, Kangheng Lin, Liang Zhao, Mei Chen, Peng Xing, Rui Wang, Shiyu Liu, Shutao Xia, Tianhao You, Wei Ji, Xianfang Zeng, Xin Han, Xuelin Zhang, Yana Wei, Yanming Xu, Yimin Jiang, Yingming Wang, Yu Zhou, Yucheng Han, Ziyang Meng, Binxing Jiao, Daxin Jiang, Xiangyu Zhang, and Yibo Zhu. 2025. NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale. arXiv preprint arXiv:2508.10711 (2025).

[60] Qwen Team. 2025. Qwen2.5-VL. https://qwenlm.github.io/blog/qwen2.5-vl/

[61] Tencent Hunyuan Team. 2025. HunyuanImage 3.0: Technical Report. https: //github.com/Tencent-Hunyuan/HunyuanImage-3.0.

[62] Z-Image Team. 2025. Z-Image: An Eficient Image Generation Foundation Model with Single-Stream Difusion Transformer. arXivpreprint arXiv:2511.22699 (2025).

[63] Tencent Hunyuan Team. 2024. Hunyuan-A13B. https://github.com/Tencent-Hunyuan/Hunyuan-A13B.

[64] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. 2025. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025).

[65] Yuxiang Tuo, Yifeng Geng, and Liefeng Bo. 2024. Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245 (2024).

[66] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. 2024. Anytext: Multilingual visual text generation and editing. In International Conference on Learning Representations, Vol. 2024. 56783–56799.

[67] Mor Ventura, Eyal Ben-David, Anna Korhonen, and Roi Reichart. 2025. Navigating cultural chasms: Exploring and unlocking the cultural pov of text-to-image models. Transactions of the Association for Computational Linguistics 13 (2025), 142–166.

[68] Yixin Wan, Arjun Subramonian, Anaelia Ovalle, Zongyu Lin, Ashima Suvarna, Christina Chance, Hritik Bansal, Rebecca Pattichis, and Kai-Wei Chang. 2024. Survey of bias in text-to-image generation: Definition, evaluation, and mitigation. arXiv preprint arXiv:2404.01030 (2024).

[69] Jialu Wang, Yang Liu, and Xin Wang. 2022. Assessing multilingual fairness in pre-trained multimodal representations. In Findings of the Association for Computational Linguistics: ACL 2022. 2681–2695.

[70] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei,Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. 2025. Qwen-Image Technical Report. arXiv:2508.02324 [cs.CV] https://arxiv.org/abs/2508.02324

[71] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng YU, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, Bingchen Liu, Daquan Zhou, and Song Han. 2025. SANA 1.5: Eficient Scaling of Training-Time and Inference Time Compute in Linear Difusion Transformer. In Forty-second International Conference on Machine Learning. https://openreview.net/forum?id=27hOkXzy9e

[72] Sen Xing, Muyan Zhong, Zeqiang Lai, Liangchen Li, Jiawen Liu, Yaohui Wang, Jifeng Dai, and Wenhai Wang. 2025. MuLan: Adapting Multilingual Difusion Models for Hundreds ofLanguages with Negligible Cost. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 267), Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu (Eds.). PMLR, 68953–68969.

[73] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Rafel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: Human language technologies. 483–498.

[74] Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia,

Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu, Shanghaoran Quan, and Zekun Wang. 2024. Qwen2.5 Technical Report. ArXiv abs/2412.15115 (2024). https://api.semanticscholar.org/CorpusID:274859421

[75] Fulong Ye, Guang Liu, Xinya Wu, and Ledell Wu. 2024. Altdifusion: A multilingual text-to-image difusion model. In Proceedings of the AAAI conference on artificial intelligence, Vol. 38. 6648–6656.

[76] Lili Zhang, Xi Liao, Zaijia Yang, Baihang Gao, Chunjie Wang, Qiuling Yang, and Deshun Li. 2024. Partiality and Misconception: Investigating Cultural Representativeness in Text-to-Image Models. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 620, 25 pages. doi:10.1145/3613904.3642877

[77] Sicheng Zhang, Binzhu Xie, Zhonghao Yan, Yuli Zhang, Donghao Zhou, Xiaofei Chen, Shi Qiu, Jiaqi Liu, Guoyang Xie, and Zhichao Lu. 2025. Trade-ofs in image generation: How do diferent dimensions interact?. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 17256–17267.

[78] Ej Zhou and Weiming Lu. 2025. Bias Beyond English: Evaluating Social Bias and Debiasing Methods in a Low-Resource Setting. In CCF International Conference on Natural Language Processing and Chinese Computing. Springer, 214–227.

## A Important Statements

## A.1 Social Impact

This work contributes positively to promoting fairness and inclusiveness in artificial intelligence. First, through a systematic evaluation of text-to-image models in multilingual and multicultural settings, we reveal existing linguistic and cultural biases in current generative models, and our framework provides a foundation for global language fairness assessment.

Second, the LingT2I benchmark and metric suite ofer meaningful directions for future work, helping drive the development of models that are more inclusive of linguistic and cultural diversity while improving the visibility and research value of underrepresented languages and cultures.

Third, our study enhances the transparency of multilingual generation evaluation, providing both academia and industry with a more measurable and interpretable framework, and advancing AI toward being more explainable, fair, and responsible.

Finally, our work will guide the generative AI research commu nity to better understand and respect linguistic, script, and cultural diversity, helping reduce technical bias and fostering a more inclusive global AI system.

## A.2 Ethical Statement

To avoid the potential social risks, we emphasize that all datasets used in this work comply with their oficial licenses and community standards, and we strictly adhere to ethical guidelines throughout data usage and research practices.

Although the evaluation encompasses dimensions of bias and toxicity, we have not introduced any new harmful data. We only utilized existing research-purpose datasets and will not directly disclose any toxicity-related data unless absolutely necessary, and the reproducibility of this evaluation is ensured by the complete scripts and prompts we provide.

The generation and evaluation in this paper are conducted only for academic research purposes. Throughout this process, assessments beneficial to enhancing fairness in human society and culture have been performed, yielding positive impact only.

## A.3 LLM Usage

All instances of LLM usage in the research are mentioned clearly in the main text and appendix, including specific models and their detailed usage.

Besides, LLMs are used to moderately polish the paper writing. Specifically, LLMs are employed for improving grammar and formatting consistency of LaTeX content.

## B Details of LingT2I Dataset

## B.1 Prompt for MLLM in Data Processing

Prompt for Translation. Figure 12 shows the prompt for Gemini-2.5-Pro in the translation progress. This is the final version, resulting from improvements.

Prompt for Content Generation Task Annotation. Figure 13 shows the prompt for Gemini-2.5-Flash in the data annotation process of Content Generation Task. We construct dimension-specific prompts using a unified template that guides the annotation model to extract and rewrite dimension-relevant information from the original caption. The template takes as input the caption, the target dimension, and its definition, and instructs the model to produce a concise prompt that preserves only the information relevant to the target dimension while removing irrelevant details.

For dimensions that are not explicitly described in the original caption (e.g., Style and Bias), we further employ a conditional expansion strategy. Specifically, the model is instructed to first generate a concise base description and then modify it by incorporating dimension-specific control signals (e.g., stylistic cues). The example template shown in Figure 13 illustrates this process, while the exact design of control phrases and dimension-specific elements follows prior benchmark practices, particularly HEIM [31] and TRIGScore [77].

## B.2 Dataset Examples and Statistics

Figure 15 and Figure 16 show the representative prompts across all 10 languages in LingT2I dataset’s Content Generation task and the corresponding output images from some example models. Figure 17 shows the representative prompts across all 10 languages in LingT2I dataset’s Text Rendering task and the corresponding output images from some example models.

The detailed dataset statistics of prompt length are shown in Figure 14.

## B.3 Quality Control

Automatic Processing and Verification. All automatic processing—including prompt shortening, filtering, augmentation, and translation for both the Content Generation and Text Rendering tasks—is conducted using Gemini 2.5 Pro. To ensure translation accuracy and linguistic consistency, we apply multi-round verification, including back-translation and cross-checking with GPT-5. During this process, we explicitly enforce constraints to preserve semantic meaning, maintain cultural nuance, and avoid introducing additional bias across languages. GPT-5 flags a small portion of samples (1.3%) as problematic, mainly due to minor semantic inconsistencies or cultural ambiguities. These cases are further reviewed and manually corrected to ensure final data quality.

Error Analysis and Iterative Refinement. Before large-scale data generation, we conduct pilot experiments on a randomly sampled 5% subset of the dataset. Based on this subset, we perform error analysis to identify common issues such as semantic drift, cultural misalignment, and translation inconsistency. Guided by these observations, we iteratively refine the prompt construction and filtering process for three rounds, until the data quality is considered stable. Human Quality Check. After finalizing the dataset, we further validate data quality through human evaluation. We randomly sample 5% of the full dataset and involve native speakers across all target languages, including university students and academic staf. Annotators are asked to assess semantic fidelity, cultural appropriateness, and fluency of the translated prompts. Overall, 98% of the samples are judged to be semantically consistent across languages.

Figure 18 shows the prompt for GPT5 to double check our translation. Table 4 shows the improvements made with each iteration and the resulting increase in the accuracy of the random samples.

Final Prompt for Translation  
![](images/77e0a785a82f4f763b6e54d1a54a19d141cc648aa49a4c41ef94f537fcfa8b20.jpg)  
Figure 12: Prompt template used for multilingual translation, ensuring semantic consistency, cultural appropriateness, and stylistic fidelity across languages.

Table 4: Iterative refinement of the translation prompt and its impact on translation quality. Accuracy is measured on a randomly sampled subset using GPT-5-based verification.
<table><tr><td>Iteration</td><td>Key Modifications</td><td>Issue Rate (%)</td><td>Pass Rate (%)</td></tr><tr><td>v1</td><td>Basic translation instructions with fluency and grammar constraints</td><td>8.7</td><td>91.3</td></tr><tr><td>v2</td><td>Added semantic consistency and tone preservation constraints</td><td>5.2</td><td>94.8</td></tr><tr><td>v3</td><td>Introduced cultural appropriateness and terminology fidelity checks</td><td>3.4</td><td>96.6</td></tr><tr><td>v4 (final)</td><td>Enforced prompt fidelity, bias control, and strict format preservation</td><td>1.3</td><td>98.7</td></tr></table>

## C Details of Metrics

## C.1 Content Generation Metric Settings

## C.1.1 CLIPScore.

For CLIPScore implementation, we use the Facebook metaclip-2- worldwide-huge-378 oficial checkpointon huggingface and inject it into the original CLIPScore github codebase.

## C.1.2 TRIGScore.

Computation (Dimensions except Robustness). The detailed TRIG Score computation method is as followed, from the original TRIG [77] paper:

For each sample from subset D we feed the task description, generated image, prompt, and specific dimensional evaluation criteria into the VLM, instructing it to evaluate the degree from a set of predefined rating tokens. Formally, let the token set be $\mathcal { T } ~ = ~ \{ t _ { 1 } , t _ { 2 } , . . . , t _ { n } \}$ where $t _ { i }$ represents a semantic rating (e.g., “Good”, “Medium”, “Bad”).

The model output is provided in the form of logits, which can be expressed as ${ \mathcal { L } } = \{ ( x , z ( x ) ) \mid x \in \mathcal { V } \}$ , where V denotes the set of all possible tokens and �(�) is the logit associated with token �. We select those rating tokens from L that satisfy $x \in \mathcal T$ , forming the candidate token set as $\mathcal { U } = \{ ( t , z ( t ) ) \in \mathcal { L } \mid t \in \mathcal { T } \}$

For each candidate token � in U (with corresponding logit �(�)), the softmax function is applied to convert the logits into normalized

![](images/10ddde3ded1cff0b46979b8d9cc8e1ecd97ca07cc1b3806138b7b11aa7a31a07.jpg)

## Dimension-aware Prompt

Figure 13: Prompt for Content Generation Task Annotation.

probabilities:

$$
\tilde { p } ( t ) = \frac { \exp ( z ( t ) ) } { \sum _ { t ^ { \prime } \in \mathcal { U } } \exp ( z ( t ^ { \prime } ) ) + \epsilon }\tag{1}
$$

Define a mapping function �(�) that assigns each rating token � a numerical weight, In a linear mapping case, we define:

$$
s _ { \mathrm { l i n e a r } } ( t _ { i } ) = \frac { i - 1 } { n - 1 } , \quad i = 1 , . . . , n
$$

Using this $s ( t ) ,$ , the weighted sum of the normalized probabilities is computed to obtain a preliminary score:

$$
S = \sum _ { t \in \mathcal { U } } s ( t ) \tilde { p } ( t )\tag{2}
$$

To account for model uncertainty, we use a confidence weight � and define the final score $S ^ { \prime }$ as:

$$
C = \operatorname* { m a x } _ { i } \tilde { p } ( t _ { i } ) , S ^ { \prime } = C \cdot S\tag{3}
$$

Computation (Robustness - Bias). In particular, the score of Bias dimension is computed using normalized entropy. For each dimension � (race, gender, age), we define:

$$
\mathrm { B i a s } ( d ) = \frac { H _ { d } } { H _ { d } ^ { \operatorname* { m a x } } } = \frac { - \sum _ { i = 1 } ^ { K } \hat { p _ { i } } \log \hat { p _ { i } } } { \log K } ,\tag{4}
$$

where $\mathscr { p } _ { i }$ is the proportion of category �, and � is the number of categories. The overall score is a weighted average across dimensions:

$$
\begin{array} { r l } { \mathrm { S c o r e } = 0 . 5 \cdot \mathrm { B i a s } _ { \mathrm { r a c e } } } & { } \\ { + \ : 0 . 3 \cdot \mathrm { B i a s } _ { \mathrm { g e n d e r } } } & { } \\ { + \ : 0 . 2 \cdot \mathrm { B i a s } _ { \mathrm { a g e } } . } \end{array}\tag{5}
$$

A score closer to 1 indicates a more balanced distribution (less bias), while values closer to 0 reflect stronger bias.

Computation (Robustness - Toxicity). The score of Toxicity dimension is defined as the rejection rate of NSFW content in generated images, i.e.,

$$
1 - P ( \mathrm { N S F W } ) .
$$

Prompts. Figure 19 shows the adapted prompt in the TRIGScore, we provide the specific prompts for the general, Bias, and Toxicity dimensions separately. The specific definitions of each dimension used in these prompts are presented separately in Table 5. For better multilingual understanding, we use Qwen-2.5-VL 72B instead of the 7B version in the original TRIG paper.

## C.2 Text Rendering Metric Settings

## C.2.1 Precision.

For Precision, we use Gemini 2.5 Flash as the OCR model for multilingual text recognition. The prompt for Gemini is shown in Figure 20.

Table 5: Detailed Dimension definitions used in Multilingual TRIGScore evaluation.
<table><tr><td>Dimension</td><td>Definition</td></tr><tr><td>Realism</td><td>Evaluate how realistic the image appears, including physical plausibility, natural textures, lighting, and absence of artificial distortions.</td></tr><tr><td>Originality</td><td>Evaluate the creativity and uniqueness of the image, including novel composition, style diversity, and avoidance of repetitive or clichéd patterns.</td></tr><tr><td>Aesthetics</td><td>Evaluate the overall visual appeal of the image, including color harmony, composition balance, contrast, and emotional impact.</td></tr><tr><td>Content Alignment</td><td>Evaluate whether the main objects, attributes, and scenes in the image accurately match the elements specified in the prompt.</td></tr><tr><td>Relation Alignment</td><td>Evaluate whether spatial and logical relationships between objects are correctly represented according to the prompt (e.g. position, scale, and arrangement).</td></tr><tr><td>Style Alignment</td><td>Evaluate whether the overall artistic and visual style of the image matches the style specified in the prompt without deviation.</td></tr><tr><td>Knowledge</td><td>Evaluate whether the image correctly reflects complex or specialized knowledge described in the prompt, avoiding factual errors or oversimplifications.</td></tr><tr><td>Ambiguous</td><td>Evaluate whether the image appropriately captures ambiguity, abstraction, or open-ended interpretation as described in the prompt, without oversimplifying it.</td></tr></table>

![](images/16e03401ccd6ab82d491a38d07795316ed07e004e00228e340b13073b2fc7f75.jpg)

![](images/ee0991a3ed19d1f2217bf6f4a9ef9570f4a0095ce7cf610b962e9175055e8ed4.jpg)  
(ii) Text Rendering Task (texts to be rendered)  
Figure 14: Dataset Statistics. Average token lengths computed using the mT5 tokenizer for Content Generation and Text Rendering tasks.

We use three complementary precision metrics: character-level NED, token-level NED, and sentence-level accuracy.

$$
\mathrm { N E D } _ { c h a r } = 1 - \frac { D _ { l e v } ( C _ { \rho r e d } , C _ { g t } ) } { \mathrm { m a x } ( | C _ { \rho r e d } | , | C _ { g t } | ) }
$$

where $D _ { l e v }$ denotes the Levenshtein distance between predicted and ground-truth character sequences.

$$
\mathrm { N E D } _ { t o k e n } = 1 - \frac { D _ { l e v } ( T _ { p r e d } , T _ { g t } ) } { \operatorname* { m a x } ( | T _ { p r e d } | , | T _ { g t } | ) }
$$

where tokens � are obtained using the mT5 tokenizer to ensure consistent multilingual segmentation.

$$
{ \mathrm { S e n t e n c e A c c } } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f } } \ S _ { \rlap / p r e d } = S _ { g t } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Finally, we compute the overall score as their average:

$$
\mathrm { P r e c i s i o n } = \frac { 1 } { 3 } \left[ \mathrm { N E D } _ { c h a r } + \mathrm { N E D } _ { t o k e n } + \mathrm { S e n t e n c e A c c } \right] .
$$

C.2.2 Text Quality, Text Aesthetics, and BG Fusion.

In these MLLM-as-judge metrics, we use Gemini-2.5-flash to give the three evaluation scores, and the prompts are shown in Figure 20.

## D Experiments

For reproducibility, we used 42 as the seed for all models, generating each prompt only once to produce a single image.

In terms of parameters, to align with the model’s structures and capabilities, we keep the oficial default recommended settings for parameters such as the number of generation steps, output resolution, and guidance scale.

We conducted all experiments using four NVIDIA A100 64GB GPUs. However, this configuration was chosen for experimental eficiency. Based on oficial instructions from all models, a single GPU with approximately 40GB of memory and CPU ofloading is suficient to complete all our generation experiments within an acceptable timeframe.

<table><tr><td rowspan=29 colspan=2>LanguagePrompts</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>Chinese</td><td rowspan=1 colspan=1>Hindi</td><td rowspan=1 colspan=1>Spanish</td><td rowspan=1 colspan=1>Arabic</td><td rowspan=1 colspan=1>French</td><td rowspan=1 colspan=1>Portuguese</td><td rowspan=1 colspan=3>Russian</td><td rowspan=1 colspan=1>Japanese</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Una ardilla</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>师萌</td><td rowspan=1 colspan=1>zorro está de</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Un écureuil-</td><td rowspan=1 colspan=1>Um esquilo-</td><td rowspan=1 colspan=1> </td><td rowspan=1 colspan=3>ベージュ色の</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>すす何</td><td rowspan=1 colspan=1>lado en un</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>renard se tient</td><td rowspan=1 colspan=1>raposa em uma</td><td rowspan=1 colspan=1>  </td><td rowspan=5 colspan=3>石造りのポーチの上、白い</td><td rowspan=4 colspan=1></td></tr><tr><td rowspan=1 colspan=1>A fox squirrel</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>市2収市</td><td rowspan=1 colspan=1>porche de</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>de côté sur un</td><td rowspan=1 colspan=1>varanda de</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>on a beige stone</td><td rowspan=3 colspan=1>一只狐松鼠侧</td><td rowspan=3 colspan=1>羽舷灭师</td><td rowspan=3 colspan=1>piedra beis,</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1>|</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>porche en pierre</td><td rowspan=1 colspan=1>pedra bege fica</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>柱のそば</td></tr><tr><td rowspan=1 colspan=1>porch stands</td><td rowspan=1 colspan=1>身站在米色的</td><td rowspan=1 colspan=1>城市</td><td rowspan=1 colspan=1>junto a una</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>beige, près</td><td rowspan=1 colspan=1>de lado junto a</td><td rowspan=1 colspan=1> </td><td rowspan=1 colspan=2>匹のフォック</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>叫</td></tr><tr><td rowspan=2 colspan=1>sideways by a</td><td rowspan=2 colspan=1>石头门廊上的</td><td rowspan=2 colspan=1>氢為</td><td rowspan=2 colspan=1>columna blanca.</td><td rowspan=2 colspan=1> </td><td rowspan=2 colspan=1>d&#x27;une colonne</td><td rowspan=2 colspan=1>uma coluna</td><td rowspan=2 colspan=1> .</td><td rowspan=2 colspan=3>スリスが横向</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>white column, It</td><td rowspan=1 colspan=1>一根白色柱子</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Está orientada</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>blanche. Il est</td><td rowspan=1 colspan=1>branca. Ele está</td><td rowspan=1 colspan=1> </td><td rowspan=1 colspan=3>きに立ってい</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.</td></tr><tr><td rowspan=1 colspan=1>faces left,</td><td rowspan=1 colspan=1>旁。它面朝左</td><td rowspan=1 colspan=1>朮孩燕</td><td rowspan=1 colspan=1>hacia la</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>tourné vers la</td><td rowspan=1 colspan=1>virado para a</td><td rowspan=1 colspan=1>, </td><td rowspan=1 colspan=3>ます、左を向</td><td></td></tr><tr><td rowspan=3 colspan=1>looking slightly</td><td rowspan=3 colspan=1>方，视线略微</td><td rowspan=3 colspan=1>彫す</td><td rowspan=3 colspan=1>izquierda,</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>gauche,</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=2 colspan=3>ます。左を向き、わずかに</td><td rowspan=2 colspan=1>丘豆是</td></tr><tr><td rowspan=1 colspan=1>esquerda,</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>forward, with a</td><td rowspan=1 colspan=1>向前，嘴里含</td><td rowspan=1 colspan=1>T 3</td><td rowspan=1 colspan=1>mirando</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>regardant</td><td rowspan=1 colspan=1>olhando</td><td rowspan=1 colspan=1>HeMHOrO</td><td rowspan=1 colspan=3>前を見つめ、</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>brown acorn in</td><td rowspan=1 colspan=1>着一颗棕色的</td><td rowspan=1 colspan=1>き3k</td><td rowspan=1 colspan=1>ligeramente</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>légèrement vers</td><td rowspan=1 colspan=1>ligeiramente</td><td rowspan=1 colspan=1>,  </td><td rowspan=1 colspan=3>ロには茶色の</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>its mouth.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3麻煎并</td><td rowspan=1 colspan=1>hacia adelante,</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>l&#x27;avant, avec un</td><td rowspan=1 colspan=1>para a frente,</td><td rowspan=1 colspan=1>y ee</td><td rowspan=1 colspan=3>ドングリをく</td><td rowspan=1 colspan=1>.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>师然河可</td><td rowspan=1 colspan=1>con una bellota</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>gland marron</td><td rowspan=1 colspan=1>com uma bolota</td><td rowspan=1 colspan=1></td><td rowspan=4 colspan=3>わえています。</td><td></td></tr><tr><td rowspan=4 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>哥叫</td><td rowspan=1 colspan=1>marrón en la</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>dans la bouche.</td><td rowspan=2 colspan=1>marrom na boca.</td><td rowspan=2 colspan=1>.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>boca.</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>V</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1><img src="images/4f4da6de84fa038d9081c8222dd7e01f8f9bf0a2ac33498d6f4a99fe461cc32b.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>灯</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Le</td><td rowspan=1 colspan=1>LA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>业</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>小</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr></table>

Figure 15: Examples for Content Generation task in Reality dimension.

## D.1 Model Settings

## D.1.1 Content Generation Models.

SD3.5 [54]. Stable Difusion 3.5 is an 8B parameter text-to-image model utilizing a multimodal difusion transformer architecture for high-quality image generation. We use the SD3.5-large model, with a resolution of 1024×1024.

SDXL [42]. SDXL is an improved latent difusion model for text-toimage generation, featuring an expanded UNet, dual text encoders, and a refinement stage for high-fidelity image synthesis. We use the stabilityai/stable-diffusion-xl-base-1.0 checkpoint, with a resolution of 1024×1024.

Sana [71]. Sana is an eficient framework for rapid, high-resolution text-to-image synthesis with strong text-image alignment, employing compression autoencoders and Linear DiT architecture. We use the SANA1.5\_4.8B\_1024px\_diffusers model, with a resolution of 1024×1024

Janus-Pro [9]. Janus-Pro is a novel autoregressive multimodal model generating images by tokenizing input images and processing via autoregressive transformers.We use the 7B model, with a resolution of 384×384

Qwen-Image [70]. Qwen-Image is a multimodal difusion–transformer model that unifies text-to-image generation and understanding, featuring scalable cross-modality alignment with a powerful visual–language joint backbone for high-quality and instructionfollowing image synthesis. We use the Qwen-Image model of T2I

Language

English

Chinese

An overhead cream labradoodle lies flat on dry grass, wearing a white collar. A red leash extends from its body; scattered leaves are visible.

Hindi

俯瞰视⻆下，⼀只奶油⾊的拉布拉多贵宾⽝平躺在⼲草上，戴着⼀个⽩⾊的项圈。⼀条红⾊的牵引绳从它⾝上延伸出来；可以看到散落的叶⼦。

Spanish

ऊपर से देखने पर, एक -.म रंग का ल5ै ाडूडल सखीू घास पर सपाट लेटा है, िजसने एक सफे द कॉलर पहना हुआ है। उसके शरEर से एक लाल पFटा फै ला हुआ है; Hबखरे हुए पJे Kदखाई दे रहे हM।

Visto desde arriba, un labradoodle de color crema está tumbado sobre la hierba seca, con un collar blanco. Una correa roja se extiende desde su cuerpo; se ven hojas esparcidas.

Arabic

French

Russian

Portuguese

بلك ،ىلعلأا نم يمیرك لدوداربلا اللونمستلقٍبشكل بشعلا ىلع حطسم يدتریو ،فاجلا ًاأبیض.یمتدمن طوق ؛رمحأ دوقم هدسج قاروأ رھظتو .ةرثانتم

Japanese

Vu d'en haut, un labradoodle de couleur crème est couché à plat sur de l'herbe sèche, portant un collier blanc. Une laisse rouge s'étend de son corps ; des feuilles éparses sont visibles.

Visto de cima, um labradoodle de cor creme está deitado na grama seca, usando uma coleira branca. Uma guia vermelha se estende de seu corpo; folhas espalhadas são visíveis.

Вид сверху:   
кремовый   
лабрадудль   
лежит плашмя на сухой траве в белом   
ошейнике. От его тела   
тянется   
красный   
поводок;   
видны   
разбросанные листья.

Korean

俯瞰で、クリーム⾊のラブラドゥードルが乾いた草の上に平らに寝 そべり、⽩い⾸輪をしています。体からは⾚いリード が伸びており、散らばった葉が⾒えます。

위에서  
내 려다본  
크 림 색  
래브라두들 한  
마리가 흰색줄 을하고  
<sup>마른</sup> 납작 <sub>있습니</sub> 잔디 위에엎드려合 다.  
몸 에 서빨간색  
리드 줄이 뻗어  
나와 있으며,  
흩 어 진  
나뭇 잎들이  
보입니다.

![](images/b655a4be3944628b3ff22abf0564318642ee60a1337f855e11e31dbec088a8a8.jpg)  
Figure 16: Examples for Content Generation task in Content Alignment dimension.

version, with a resolution of 1024×1024.

FLUX [29]. FLUX is an advanced text-to-image model employing a 12B parameter rectified flow transformer architecture for high fidelity image synthesis. We use the latest FLUX.1-Krea-dev model, with with a resolution of 1024×1024

PixArt-Σ [7]. PixArt-Σ is an improved Difusion Transformer model for high-resolution text-to-image, featuring weak-to-strong training and key-value token compression. In our experiment, we use the PixArt-Sigma-XL-2-1024-MS model, with a resolution of 1024×1024.

PEA [36]. PEA is a parameter-eficient adapter for non-English text-to-image generation that aligns multilingual CLIP encoders with pretrained difusion UNets via lightweight knowledge distillation. We use the MultilingualFLUX.1-adapter version with FLUX.1-schnell as the basic model, with a resolution of 1024×1024.

X2I [37]. X2I is a multimodal difusion–transformer framework that transfers the comprehension abilities of multimodal large language models to text-to-image generation via attention distil lation and AlignNet. We use the X2I-QwenVL2.5-7B framework with FLUX.1-schnell as the basic model, with a resolution of 1024×1024.

MuLan [72]. MuLan is a lightweight adapter that equips difusion models with multilingual generation via image-centered alignment between text encoders and difusion backbones. We use the mulan-pixart model finetuned based on PixArt-�, with a resolution of 1024×1024.

Lumina-T2X [19]. Lumina-T2X is a high-quality text-to-image framework that integrated with a LLaMA2-7B text encoder and a fine-tuned SDXL VAE. It achieves eficient training from scratch and supports flexible inference across various resolutions. We use the Lumina-T2I model with a resolution of 1024×1024.

Z-Image [62]. Z-Image is a highly eficient text-to-image model featuring a Scalable Single-Stream DiT (S3-DiT) architecture with 6B parameters. By concatenating text, visual semantic, and VAE tokens into a unified input stream, it achieves superior parameter eficiency and cross-modal interaction. We use the Z-Image model with a resolution of 1024×1024.

A decorative badge features the word <sks1> prominently in an elegant, cursive font, adorned with intricate diamond-like details; surrounding the text are several three-dimensional, rose gold flowers and sparkling diamond shapes, adding a luxurious touch; the background is a smooth, gradient blend of dark colors that highlights the central elements, creating a soft glow effect around the <sks1>, which is positioned centrally in the image; the rose gold flowers are partially overlapping the text, enhancing the visual depth.

![](images/870ddcfd4195ad91b414cd6302fc069a2bdd08372c81c53f152e0ccd7fa0077a.jpg)  
Figure 17: Examples for Text Rendering task.

OmniDifusion [56]. OmniDifusion is an LLM-powered text-toimage framework that integrates a frozen Baichuan2-7B model with a difusion UNet via a lightweight 4-layer transformer adapter. We use the OmniDiffusion-SDXL based model with a resolution of 1024×1024.

## D.1.2 Text Rendering Models.

Nano Banana [66]. We use Google Gemini oficial API with default settings to generate all images, with a resolution of 1024×1024.

Qwen-Image [70]. We use the same setting as in Content Generation task.

FLUX [29]. We use the same setting as in Content Generation task. AnyText [66]. AnyText is a difusion-based model for multilingual text generation and editing, integrating auxiliary latents and OCRguided embeddings to enhance text accuracy and visual coherence. We use the AnyText-v1.1 model, with a resolution of 512×512.

AnyText2 [65]. AnyText2 is a difusion-based multilingual text generation model featuring a WriteNet+AttnX architecture and a Text Embedding Module for controllable, high-fidelity text rendering. We use the AnyText2-v1.0 model, with a resolution of 512×512.

EasyText [35]. EasyText is a difusion-transformer model for multilingual text rendering, leveraging visual tokenization and implicit position alignment for controllable and layout-free generation. We use the EasyText-LoRA-ft model, with a resolution of 1024×1024.

## D.2 Cross-lingual efect across dimensions

We choose Qwen-Image, MuLan, Nano Banana and EasyText as four relatively fair model for further cross-lingual efect analysis. The Qwen-Image and MuLan models are for Content Generation task, the full results are shown in Table 6. The Nano Banana and EasyText models are for Text Rendering task, the full results are shown in Table 7.

Table 6: Cross-lingual Multi-dimensional Analysis on Qwen-Image and MuLan Model for Content Generation Task.
<table><tr><td rowspan="2">Language</td><td colspan="3">Image Quality</td><td colspan="3">Task Alignment</td><td colspan="2">Diversity</td><td colspan="2">Robustness</td></tr><tr><td>Realism</td><td>Originality</td><td>Aesthetics</td><td>Content</td><td>Relation</td><td>Style</td><td>Knowledge</td><td>Ambiguity</td><td>Toxicity</td><td>Bias</td></tr><tr><td colspan="9">- General-purpose Model: Qwen-Image</td><td></td></tr><tr><td>English</td><td>0.74</td><td>0.76</td><td>0.79</td><td>0.82</td><td>0.75</td><td>0.79</td><td>0.58</td><td>0.76</td><td>0.52</td><td>0.36</td></tr><tr><td>Chinese</td><td>0.72</td><td>0.78</td><td>0.77</td><td>0.79</td><td>0.73</td><td>0.82</td><td>0.63</td><td>0.77</td><td>0.65</td><td>0.44</td></tr><tr><td>Hindi</td><td>0.51</td><td>0.61</td><td>0.56</td><td>0.50</td><td>0.52</td><td>0.58</td><td>0.49</td><td>0.51</td><td>0.89</td><td>0.28</td></tr><tr><td>Spanish</td><td>0.70</td><td>0.76</td><td>0.76</td><td>0.79</td><td>0.72</td><td>0.76</td><td>0.60</td><td>0.74</td><td>0.69</td><td>0.36</td></tr><tr><td>Arabic</td><td>0.67</td><td>0.78</td><td>0.73</td><td>0.73</td><td>0.67</td><td>0.69</td><td>0.64</td><td>0.71</td><td>0.77</td><td>0.46</td></tr><tr><td>French</td><td>0.72</td><td>0.77</td><td>0.77</td><td>0.79</td><td>0.73</td><td>0.76</td><td>0.60</td><td>0.73</td><td>0.66</td><td>0.34</td></tr><tr><td>Portuguese</td><td>0.71</td><td>0.77</td><td>0.77</td><td>0.79</td><td>0.70</td><td>0.74</td><td>0.58</td><td>0.74</td><td>0.71</td><td>0.34</td></tr><tr><td>Russian</td><td>0.70</td><td>0.77</td><td>0.75</td><td>0.78</td><td>0.73</td><td>0.77</td><td>0.61</td><td>0.75</td><td>0.70</td><td>0.23</td></tr><tr><td>Japanese</td><td>0.69</td><td>0.74</td><td>0.74</td><td>0.75</td><td>0.70</td><td>0.75</td><td>0.62</td><td>0.72</td><td>0.76</td><td>0.52</td></tr><tr><td>Korea</td><td>0.65</td><td>0.74</td><td>0.71</td><td>0.69</td><td>0.68</td><td>0.74</td><td>0.60</td><td>0.69</td><td>0.76</td><td>0.43</td></tr><tr><td colspan="9">– Multilingual-enhanced Model: MuLan &lt;PixArt&gt;</td><td></td></tr><tr><td>English</td><td>0.49</td><td>0.79</td><td>0.55</td><td>0.47</td><td>0.46</td><td>0.66</td><td>0.56</td><td>0.83</td><td>0.56</td><td>0.45</td></tr><tr><td>Chinese</td><td>0.48</td><td>0.78</td><td>0.55</td><td>0.46</td><td>0.45</td><td>0.65</td><td>0.55</td><td>0.79</td><td>0.66</td><td>0.61</td></tr><tr><td>Hindi</td><td>0.35</td><td>0.68</td><td>0.39</td><td>0.34</td><td>0.36</td><td>0.47</td><td>0.48</td><td>0.70</td><td>0.82</td><td>0.66</td></tr><tr><td>Spanish</td><td>0.47</td><td>0.78</td><td>0.54</td><td>0.46</td><td>0.44</td><td>0.58</td><td>0.55</td><td>0.82</td><td>0.70</td><td>0.40</td></tr><tr><td>Arabic</td><td>0.42</td><td>0.78</td><td>0.47</td><td>0.40</td><td>0.42</td><td>0.50</td><td>0.55</td><td>0.79</td><td>0.80</td><td>0.47</td></tr><tr><td>French</td><td>0.47</td><td>0.79</td><td>0.54</td><td>0.46</td><td>0.44</td><td>0.61</td><td>0.56</td><td>0.83</td><td>0.66</td><td>0.40</td></tr><tr><td>Portuguese</td><td>0.47</td><td>0.79</td><td>0.53</td><td>0.45</td><td>0.44</td><td>0.57</td><td>0.55</td><td>0.83</td><td>0.71</td><td>0.43</td></tr><tr><td>Russian</td><td>0.45</td><td>0.77</td><td>0.51</td><td>0.43</td><td>0.44</td><td>0.59</td><td>0.54</td><td>0.81</td><td>0.68</td><td>0.40</td></tr><tr><td>Japanese</td><td>0.47</td><td>0.75</td><td>0.53</td><td>0.45</td><td>0.45</td><td>0.61</td><td>0.55</td><td>0.79</td><td>0.72</td><td>0.57</td></tr><tr><td>Korea</td><td>0.39</td><td>0.72</td><td>0.44</td><td>0.38</td><td>0.41</td><td>0.56</td><td>0.50</td><td>0.76</td><td>0.76</td><td>0.56</td></tr></table>

Table 7: Cross-lingual Multi-dimensional Analysis for Text Rendering Task.
<table><tr><td rowspan="2">Language</td><td colspan="3">Text</td><td colspan="2">Background</td></tr><tr><td>Precision</td><td>Quality</td><td>Aesthetics</td><td>Alignment</td><td>Fusion</td></tr><tr><td colspan="6">– General-purpose Model: Nano Banana</td></tr><tr><td>English</td><td>0.69</td><td>0.96</td><td>0.91</td><td>0.82</td><td>0.92</td></tr><tr><td>Chinese</td><td>0.31</td><td>0.71</td><td>0.87</td><td>0.78</td><td>0.87</td></tr><tr><td>Hindi</td><td>0.49</td><td>0.73</td><td>0.87</td><td>0.78</td><td>0.87</td></tr><tr><td>Spanish</td><td>0.58</td><td>0.97</td><td>0.90</td><td>0.80</td><td>0.91</td></tr><tr><td>Arabic</td><td>0.15</td><td>0.25</td><td>0.84</td><td>0.79</td><td>0.84</td></tr><tr><td>French</td><td>0.59</td><td>0.95</td><td>0.91</td><td>0.80</td><td>0.93</td></tr><tr><td>Portuguese</td><td>0.57</td><td>0.95</td><td>0.90</td><td>0.80</td><td>0.92</td></tr><tr><td>Russian</td><td>0.48</td><td>0.94</td><td>0.89</td><td>0.74</td><td>0.91</td></tr><tr><td>Japanese</td><td>0.58</td><td>0.82</td><td>0.88</td><td>0.78</td><td>0.90</td></tr><tr><td>Korea</td><td>0.64</td><td>0.86</td><td>0.89</td><td>0.77</td><td>0.91</td></tr><tr><td colspan="6">– Rendering-oriented Model: EasyText</td></tr><tr><td>English</td><td>0.88</td><td>0.86</td><td>0.90</td><td>0.82</td><td>0.91</td></tr><tr><td>Chinese</td><td>0.76</td><td>0.77</td><td>0.85</td><td>0.77</td><td>0.85</td></tr><tr><td>Hindi</td><td>0.49</td><td>0.61</td><td>0.83</td><td>0.78</td><td>0.85</td></tr><tr><td>Spanish</td><td>0.82</td><td>0.75</td><td>0.87</td><td>0.80</td><td>0.88</td></tr><tr><td>Arabic</td><td>0.45</td><td>0.60</td><td>0.77</td><td>0.80</td><td>0.81</td></tr><tr><td>French</td><td>0.82</td><td>0.74</td><td>0.87</td><td>0.80</td><td>0.89</td></tr><tr><td>Portuguese</td><td>0.80</td><td>0.74</td><td>0.87</td><td>0.79</td><td>0.90</td></tr><tr><td>Russian</td><td>0.61</td><td>0.61</td><td>0.84</td><td>0.75</td><td>0.87</td></tr><tr><td>Japanese</td><td>0.71</td><td>0.73</td><td>0.86</td><td>0.78</td><td>0.87</td></tr><tr><td>Korea</td><td>0.58</td><td>0.66</td><td>0.85</td><td>0.78</td><td>0.87</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/c9d3dbd55393184308ca5c8f0c156024e90f4732a8f2dc4b58cb2166ffca9262.jpg)  
Figure 18: Prompt template used for translation quality control.

## D.3 Complementary Result

The experiment results of all other models in Content Generation task could be found in Table 8 and Table 9.

The experiment results of all other models in Text Rendering task could be found in Table 10.

## E Language-dependent Generation Patterns

Demographic Bias. The demographic bias analysis is conducted based on the VLM-as-judge outputs for the Bias dimension, using the same prompt as defined in Figure 19.

Cultural Tendency. The cultural tendency analysis is conducted using GPT-5-mini as the evaluation model, with the prompt shown in Figure 21.

Specifically, we evaluate the images generated by Qwen-Image using GPT-based judgments, and aggregate the evaluation results to obtain the statistics reported in the main text.

Rendering Errors. Rendering errors are computed based on OCR outputs. The OCR prompting strategy has been introduced previously in Figure 20.

![](images/d52474cd035c5b017407da90e1a15cd8d44001f864efac036bfce5dd9da2001c.jpg)  
Figure 19: Evaluation Prompt for TRIGScore.

Table 8: The results for all models on all evaluation dimensions across ten languages in Content Generation Task - I.
<table><tr><td rowspan="2">Language</td><td colspan="3">Image Quality</td><td colspan="3">Task Alignment</td><td colspan="2">Diversity</td><td colspan="2">Robustness</td></tr><tr><td>Realism</td><td>Originality</td><td>Aesthetics</td><td>Content</td><td>Relation</td><td>Style</td><td>Knowledge</td><td>Ambiguity</td><td>Toxicity</td><td>Bias</td></tr><tr><td colspan="13">- SD3.5 [54]</td></tr><tr><td>English</td><td>0.69</td><td>0.76</td><td>0.73</td><td>0.72</td><td>0.66</td><td>0.70</td><td>0.69</td><td>0.78</td><td>0.58</td><td>0.50</td></tr><tr><td>Chinese</td><td>0.29</td><td>0.34</td><td>0.30</td><td>0.27</td><td>0.29</td><td>0.28</td><td>0.30</td><td>0.48</td><td>0.84</td><td>0.47</td></tr><tr><td>Hindi</td><td>0.27</td><td>0.29</td><td>0.27</td><td>0.25</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.33</td><td>0.96</td><td>0.43</td></tr><tr><td>Spanish</td><td>0.46</td><td>0.72</td><td>0.52</td><td>0.45</td><td>0.43</td><td>0.49</td><td>0.59</td><td>0.67</td><td>0.72</td><td>0.56</td></tr><tr><td>Arabic</td><td>0.27</td><td>0.29</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.26</td><td>0.26</td><td>0.34</td><td>0.93</td><td>0.41</td></tr><tr><td>French</td><td>0.50</td><td>0.74</td><td>0.56</td><td>0.50</td><td>0.47</td><td>0.56</td><td>0.63</td><td>0.69</td><td>0.69</td><td>0.44</td></tr><tr><td>Portuguese</td><td>0.40</td><td>0.71</td><td>0.45</td><td>0.39</td><td>0.38</td><td>0.43</td><td>0.54</td><td>0.65</td><td>0.75</td><td>0.63</td></tr><tr><td>Russian</td><td>0.29</td><td>0.47</td><td>0.30</td><td>0.28</td><td>0.28</td><td>0.27</td><td>0.31</td><td>0.39</td><td>0.82</td><td>0.48</td></tr><tr><td>Japanese</td><td>0.28</td><td>0.35</td><td>0.30</td><td>0.27</td><td>0.28</td><td>0.29</td><td>0.29</td><td>0.45</td><td>0.86</td><td>0.42</td></tr><tr><td>Korea</td><td>0.27</td><td>0.30</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.28</td><td>0.33</td><td>0.81</td><td>0.33</td></tr><tr><td colspan="9">- SDXL [42]</td><td></td></tr><tr><td>English</td><td>0.56</td><td>0.77</td><td>0.60</td><td>0.55</td><td>0.51</td><td>0.74</td><td>0.70</td><td>0.78</td><td>0.58</td><td>0.54</td></tr><tr><td>Chinese</td><td>0.29</td><td>0.37</td><td>0.30</td><td>0.27</td><td>0.29</td><td>0.29</td><td>0.29</td><td>0.47</td><td>0.78</td><td>0.65</td></tr><tr><td>Hindi</td><td>0.27</td><td>0.29</td><td>0.27</td><td>0.25</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.36</td><td>0.93</td><td>0.52</td></tr><tr><td>Spanish</td><td>0.40</td><td>0.71</td><td>0.45</td><td>0.38</td><td>0.37</td><td>0.53</td><td>0.52</td><td>0.68</td><td>0.75</td><td>0.57</td></tr><tr><td>Arabic</td><td>0.27</td><td>0.30</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.42</td><td>0.91</td><td>0.50</td></tr><tr><td>French</td><td>0.42</td><td>0.73</td><td>0.46</td><td>0.40</td><td>0.39</td><td>0.60</td><td>0.56</td><td>0.68</td><td>0.70</td><td>0.51</td></tr><tr><td>Portuguese</td><td>0.36</td><td>0.66</td><td>0.40</td><td>0.36</td><td>0.34</td><td>0.47</td><td>0.44</td><td>0.62</td><td>0.78</td><td>0.63</td></tr><tr><td>Russian</td><td>0.28</td><td>0.35</td><td>0.29</td><td>0.26</td><td>0.27</td><td>0.28</td><td>0.28</td><td>0.44</td><td>0.86</td><td>0.74</td></tr><tr><td>Japanese</td><td>0.30</td><td>0.39</td><td>0.31</td><td>0.28</td><td>0.29</td><td>0.31</td><td>0.30</td><td>0.49</td><td>0.85</td><td>0.69</td></tr><tr><td>Korea</td><td>0.28</td><td>0.32</td><td>0.28</td><td>0.26</td><td>0.28</td><td>0.28</td><td>0.27</td><td>0.40</td><td>0.86</td><td>0.44</td></tr><tr><td>– Sana 1.5 [71]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9"></td><td></td><td></td></tr><tr><td>English Chinese</td><td>0.62 0.49</td><td>0.79 0.76</td><td>0.73</td><td>0.69</td><td>0.62</td><td>0.81</td><td>0.62</td><td>0.83</td><td>0.57</td><td>0.54</td></tr><tr><td></td><td></td><td></td><td>0.61</td><td>0.52</td><td>0.51</td><td>0.66</td><td>0.53</td><td>0.82</td><td>0.65</td><td>0.66</td></tr><tr><td>Hindi</td><td>0.29</td><td>0.49</td><td>0.31</td><td>0.28</td><td>0.28</td><td>0.36</td><td>0.32</td><td>0.46</td><td>0.91</td><td>0.60</td></tr><tr><td>Spanish</td><td>0.55</td><td>0.78</td><td>0.63</td><td>0.58</td><td>0.55</td><td>0.69</td><td>0.57</td><td>0.78</td><td>0.71</td><td>0.47</td></tr><tr><td>Arabic</td><td>0.28</td><td>0.45</td><td>0.29</td><td>0.27</td><td>0.27</td><td>0.31</td><td>0.30</td><td>0.41</td><td>0.92</td><td>0.54</td></tr><tr><td>French</td><td>0.55</td><td>0.79</td><td>0.64</td><td>0.59</td><td>0.54</td><td>0.71</td><td>0.57</td><td>0.80</td><td>0.66</td><td>0.52</td></tr><tr><td>Portuguese</td><td>0.51</td><td>0.78</td><td>0.59</td><td>0.51</td><td>0.50</td><td>0.66</td><td>0.55</td><td>0.78</td><td>0.76</td><td>0.46</td></tr><tr><td>Russian</td><td>0.39</td><td>0.68</td><td>0.44</td><td>0.37</td><td>0.41</td><td>0.51</td><td>0.43</td><td>0.67</td><td>0.81</td><td>0.42</td></tr><tr><td>Japanese</td><td>0.32</td><td>0.57</td><td>0.35</td><td>0.31</td><td>0.31</td><td>0.41</td><td>0.39</td><td>0.70</td><td>0.80</td><td>0.62</td></tr><tr><td>Korea</td><td>0.30</td><td>0.52</td><td>0.31</td><td>0.28</td><td>0.29</td><td>0.36</td><td>0.35</td><td>0.52</td><td>0.86</td><td>0.63</td></tr><tr><td colspan="9">- Janus-Pro [9]</td><td></td><td></td></tr><tr><td>English</td><td>0.68</td><td>0.74</td><td>0.74</td><td>0.74</td><td>0.63</td><td>0.62</td><td>0.62</td><td>0.78</td><td>0.64</td><td>0.47</td></tr><tr><td>Chinese</td><td>0.53</td><td>0.46</td><td>0.57</td><td>0.52</td><td>0.37</td><td>0.38</td><td>0.29</td><td>0.45 0.47</td><td>0.78 0.91</td><td>0.76 0.31</td></tr><tr><td>Hindi Spanish</td><td>0.27 0.56</td><td>0.32 0.71</td><td>0.28 0.61</td><td>0.26 0.56</td><td>0.27 0.52</td><td>0.27 0.47</td><td>0.27 0.54</td></table>

Table 9: The results for all models on all evaluation dimensions across ten languages in Content Generation Task - II.
<table><tr><td rowspan="2">Language</td><td colspan="3">Image Quality</td><td colspan="3">Task Alignment</td><td colspan="2">Diversity</td><td colspan="2">Robustness</td></tr><tr><td>Realism</td><td>Originality</td><td>Aesthetics</td><td>Content</td><td>Relation</td><td>Style</td><td>Knowledge</td><td>Ambiguity</td><td>Toxicity</td><td>Bias</td></tr><tr><td colspan="13">– FLUX.1-Krea [29]</td></tr><tr><td>English</td><td>0.74</td><td>0.74</td><td>0.77</td><td>0.79</td><td>0.72</td><td>0.68</td><td>0.56</td><td>0.74</td><td>0.63</td><td>0.39</td></tr><tr><td>Chinese</td><td>0.27</td><td>0.30</td><td>0.29</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.39</td><td>0.80</td><td>0.30</td></tr><tr><td>Hindi</td><td>0.27</td><td>0.30</td><td>0.27</td><td>0.25</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.37</td><td>0.94</td><td>0.31</td></tr><tr><td>Spanish</td><td>0.54</td><td>0.68</td><td>0.59</td><td>0.55</td><td>0.49</td><td>0.44</td><td>0.50</td><td>0.66</td><td>0.76</td><td>0.49</td></tr><tr><td>Arabic</td><td>0.27</td><td>0.28</td><td>0.28</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.36</td><td>0.92</td><td>0.39</td></tr><tr><td>French</td><td>0.60</td><td>0.70</td><td>0.64</td><td>0.61</td><td>0.58</td><td>0.50</td><td>0.51</td><td>0.67</td><td>0.75</td><td>0.34</td></tr><tr><td>Portuguese</td><td>0.48</td><td>0.66</td><td>0.52</td><td>0.47</td><td>0.43</td><td>0.41</td><td>0.49</td><td>0.61</td><td>0.79</td><td>0.47</td></tr><tr><td>Russian</td><td>0.31</td><td>0.48</td><td>0.32</td><td>0.29</td><td>0.30</td><td>0.28</td><td>0.31</td><td>0.46</td><td>0.86</td><td>0.37</td></tr><tr><td>Japanese</td><td>0.27</td><td>0.31</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.37</td><td>0.88</td><td>0.36</td></tr><tr><td>Korea</td><td>0.27</td><td>0.29</td><td>0.27</td><td>0.25</td><td>0.27</td><td>0.26</td><td>0.27</td><td>0.32</td><td>0.89</td><td>0.38</td></tr><tr><td colspan="9">- PixArt-Σ [7]</td><td></td></tr><tr><td>English</td><td>0.61</td><td>0.79</td><td>0.69</td><td>0.64</td><td>0.60</td><td>0.76</td><td>0.60</td><td>0.82</td><td>0.57</td><td>0.46</td></tr><tr><td>Chinese</td><td>0.28</td><td>0.32</td><td>0.29</td><td>0.26</td><td>0.28</td><td>0.27</td><td>0.27</td><td>0.44</td><td>0.81</td><td>0.67</td></tr><tr><td>Hindi</td><td>0.27</td><td>0.34</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.55</td><td>0.93</td><td>0.64</td></tr><tr><td>Spanish</td><td>0.45</td><td>0.76</td><td>0.51</td><td>0.43</td><td>0.43</td><td>0.55</td><td>0.54</td><td>0.79</td><td>0.73</td><td>0.45</td></tr><tr><td>Arabic</td><td>0.27</td><td>0.34</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.57</td><td>0.93</td><td>0.71</td></tr><tr><td>French</td><td>0.48</td><td>0.76</td><td>0.53</td><td>0.48</td><td>0.45</td><td>0.61</td><td>0.56</td><td>0.79</td><td>0.70</td><td>0.45</td></tr><tr><td>Portuguese</td><td>0.41</td><td>0.75</td><td>0.47</td><td>0.40</td><td>0.40</td><td>0.52</td><td>0.51</td><td>0.76</td><td>0.74</td><td>0.51</td></tr><tr><td>Russian</td><td>0.31</td><td>0.59</td><td>0.33</td><td>0.29</td><td>0.31</td><td>0.33</td><td>0.36</td><td>0.68</td><td>0.82</td><td>0.50</td></tr><tr><td>Japanese</td><td>0.27</td><td>0.32</td><td>0.29</td><td>0.26</td><td>0.28</td><td>0.27</td><td>0.27</td><td>0.45</td><td>0.88</td><td>0.38</td></tr><tr><td>Korea</td><td>0.27</td><td>0.35</td><td>0.29</td><td>0.26</td><td>0.28</td><td>0.27</td><td>0.28</td><td>0.57</td><td>0.91</td><td>0.67</td></tr><tr><td colspan="9">- PEA &lt;FLUX&gt; [36]</td><td></td></tr><tr><td>English</td><td>0.51</td><td>0.68</td><td>0.57</td><td>0.50</td><td>0.53</td><td>0.44</td><td>0.42</td><td>0.66</td><td>0.66</td><td>0.38</td></tr><tr><td>Chinese</td><td>0.55</td><td>0.69</td><td>0.61</td><td>0.57</td><td>0.58</td><td>0.48</td><td>0.44</td><td>0.67</td><td>0.68</td><td>0.32</td></tr><tr><td>Hindi</td><td>0.38</td><td>0.68</td><td>0.42</td><td>0.37</td><td>0.43</td><td>0.44</td><td>0.42</td><td>0.64</td><td>0.83</td><td>0.50</td></tr><tr><td>Spanish</td><td>0.47</td><td>0.67</td><td>0.51</td><td>0.44</td><td>0.47</td><td>0.40</td><td>0.41</td><td>0.64</td><td>0.80</td><td>0.38</td></tr><tr><td>Arabic</td><td>0.43</td><td>0.68</td><td>0.47</td><td>0.42</td><td>0.46</td><td>0.41</td><td>0.43</td><td>0.65</td><td>0.84</td><td>0.50</td></tr><tr><td>French</td><td>0.47</td><td>0.69</td><td>0.50</td><td>0.46</td><td>0.48</td><td>0.42</td><td>0.44</td><td>0.66</td><td>0.72</td><td>0.48</td></tr><tr><td>Portuguese</td><td>0.47</td><td>0.69</td><td>0.52</td><td>0.46</td><td>0.49</td><td>0.40</td><td>0.43</td><td>0.66</td><td>0.79</td><td>0.45</td></tr><tr><td>Russian</td><td>0.42</td><td>0.67</td><td>0.46</td><td>0.40</td><td>0.47</td><td>0.39</td><td>0.41</td><td>0.65</td><td>0.77</td><td>0.45</td></tr><tr><td>Japanese</td><td>0.48</td><td>0.63</td><td>0.52</td><td>0.46</td><td>0.49</td><td>0.42</td><td>0.41</td><td>0.64</td><td>0.81</td><td>0.40</td></tr><tr><td>Korea</td><td>0.41</td><td>0.62</td><td>0.44</td><td>0.40</td><td>0.44</td><td>0.39</td><td>0.40</td><td>0.60</td><td>0.82</td><td>0.39</td></tr><tr><td colspan="9">− X2I &lt;FLUX&gt; [37]</td><td></td><td></td></tr><tr><td>English</td><td>0.61</td><td>0.72</td><td>0.68</td><td>0.64</td><td>0.62</td><td>0.55</td><td>0.42</td><td>0.71</td><td>0.65</td><td>0.46</td></tr><tr><td>Chinese</td><td>0.62</td><td>0.71</td><td>0.66</td><td>0.61</td><td>0.58</td><td>0.52</td><td>0.43</td><td>0.70</td><td>0.69</td><td>0.67</td></tr><tr><td>Hindi</td><td>0.39</td><td>0.56</td><td>0.42</td><td>0.37</td><td>0.38</td><td>0.44</td><td>0.35</td><td>0.59</td><td>0.87 0.76</td><td>0.64 0.45</td></tr><tr><td>Spanish Arabic</td><td>0.59 0.53</td><td>0.65 0.66</td><td>0.63 0.58</td><td>0.60 0.53</td><td>0.57 0.52</td><td>0.42 0.44</td><td>0.33 0.32</td><td>0.64 0.61</td></table>

![](images/41fcb6f5b7abd5cd369318224ab1b3905a29cf9895b5ef9cb9d602816a317a7a.jpg)  
Figure 20: Prompts used in Text Rendering Task.

Table 10: The results for all models on all evaluation dimensions across ten languages in Text Rendering Task.
<table><tr><td rowspan="2">Language</td><td colspan="3">Text</td><td colspan="2">Background</td></tr><tr><td>Precision</td><td>Quality</td><td>Aesthetics</td><td>Alignment</td><td>Fusion</td></tr><tr><td colspan="6">- Qwen-Image [70]</td></tr><tr><td>English</td><td>0.71</td><td>0.94</td><td>0.90</td><td>0.82</td><td>0.91</td></tr><tr><td>Chinese</td><td>0.70</td><td>0.86</td><td>0.87</td><td>0.77</td><td>0.88</td></tr><tr><td>Hindi</td><td>0.13</td><td>0.29</td><td>0.82</td><td>0.78</td><td>0.83</td></tr><tr><td>Spanish</td><td>0.66</td><td>0.92</td><td>0.90</td><td>0.80</td><td>0.90</td></tr><tr><td>Arabic</td><td>0.17</td><td>0.33</td><td>0.82</td><td>0.77</td><td>0.84</td></tr><tr><td>French</td><td>0.65</td><td>0.90</td><td>0.90</td><td>0.80</td><td>0.91</td></tr><tr><td>Portuguese</td><td>0.65</td><td>0.89</td><td>0.89</td><td>0.80</td><td>0.91</td></tr><tr><td>Russian</td><td>0.34</td><td>0.57</td><td>0.86</td><td>0.75</td><td>0.88</td></tr><tr><td>Japanese</td><td>0.60</td><td>0.80</td><td>0.86</td><td>0.77</td><td>0.88</td></tr><tr><td>Korea</td><td>0.43</td><td>0.65</td><td>0.85</td><td>0.76</td><td>0.86</td></tr><tr><td colspan="6">– FLUX.1-Krea [29]</td></tr><tr><td>English</td><td>0.67</td><td>0.94</td><td>0.92</td><td>0.81</td><td>0.94</td></tr><tr><td>Chinese</td><td>0.16</td><td>0.28</td><td>0.74</td><td>0.79</td><td>0.76</td></tr><tr><td>Hindi</td><td>0.09</td><td>0.21</td><td>0.63</td><td>0.79</td><td>0.66</td></tr><tr><td>Spanish</td><td>0.58</td><td>0.89</td><td>0.91</td><td>0.79</td><td>0.92</td></tr><tr><td>Arabic</td><td>0.09</td><td>0.19</td><td>0.60</td><td>0.80</td><td>0.63</td></tr><tr><td>French</td><td>0.56</td><td>0.85</td><td>0.90</td><td>0.80</td><td>0.93</td></tr><tr><td>Portuguese</td><td>0.58</td><td>0.88</td><td>0.90</td><td>0.79</td><td>0.92</td></tr><tr><td>Russian</td><td>0.10</td><td>0.20</td><td>0.77</td><td>0.79</td><td>0.82</td></tr><tr><td>Japanese</td><td>0.15</td><td>0.27</td><td>0.74</td><td>0.80</td><td>0.76</td></tr><tr><td>Korea</td><td>0.10</td><td>0.22</td><td>0.68</td><td>0.80</td><td>0.70</td></tr><tr><td colspan="6">– AnyText [66]</td></tr><tr><td>English</td><td>0.40</td><td>0.41</td><td>0.55</td><td>0.73</td><td>0.60</td></tr><tr><td>Chinese</td><td>0.34</td><td>0.43</td><td>0.51</td><td>0.70</td><td>0.54</td></tr><tr><td>Hindi</td><td>0.07</td><td>0.19</td><td>0.38</td><td>0.71</td><td>0.45</td></tr><tr><td>Spanish</td><td>0.35</td><td>0.37</td><td>0.52</td><td>0.72</td><td>0.57</td></tr><tr><td>Arabic</td><td>0.06</td><td>0.14</td><td>0.35</td><td>0.72</td><td>0.43</td></tr><tr><td>French</td><td>0.35</td><td>0.38</td><td>0.54</td><td>0.72</td><td>0.59</td></tr><tr><td>Portuguese</td><td>0.34</td><td>0.36</td><td>0.52</td><td>0.72</td><td>0.57</td></tr><tr><td>Russian</td><td>0.15</td><td>0.27</td><td>0.50</td><td>0.71</td><td>0.55</td></tr><tr><td>Japanese</td><td>0.21</td><td>0.31</td><td>0.48</td><td>0.71</td><td>0.52</td></tr><tr><td>Korea</td><td>0.21</td><td>0.31</td><td>0.45</td><td>0.71</td><td>0.50</td></tr><tr><td colspan="6">− AnyText2 [65]</td></tr><tr><td>English</td><td>0.58</td><td>0.57</td><td>0.62</td><td>0.77</td><td>0.68</td></tr><tr><td>Chinese</td><td>0.46</td><td>0.49</td><td>0.55</td><td>0.72</td><td>0.59</td></tr><tr><td>Hindi</td><td>0.12</td><td>0.24</td><td>0.43</td><td>0.75</td><td>0.53</td></tr><tr><td>Spanish</td><td>0.53</td><td>0.49</td><td>0.58</td><td>0.76</td><td>0.64</td></tr><tr><td>Arabic</td><td>0.09</td><td>0.16</td><td>0.34</td><td>0.75</td><td>0.45</td></tr><tr><td>French</td><td>0.52</td><td>0.47</td><td>0.60</td><td>0.76</td><td>0.65</td></tr><tr><td>Portuguese</td><td>0.51</td><td>0.48</td><td>0.58</td><td>0.75</td><td>0.64</td></tr><tr><td>Russian</td><td>0.15</td><td>0.25</td><td>0.49</td><td>0.75</td><td>0.57</td></tr><tr><td>Japanese</td><td>0.35</td><td>0.46</td><td>0.55</td><td>0.72</td><td>0.60</td></tr><tr><td>Korea</td><td>0.32</td><td>0.46</td><td>0.56</td><td>0.73</td><td>0.60</td></tr></table>

![](images/4b4952a52d4866c0edc4f34dd9fc88c338b0de7234cef70289d15c0188af7566.jpg)  
Figure 21: Prompt for Cultural Tendency.