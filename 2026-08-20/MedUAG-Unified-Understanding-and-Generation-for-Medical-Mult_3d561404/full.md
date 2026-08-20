# MedUAG: Unified Understanding and Generation for Medical Multimodal Models

Zijie Meng<sup>1,⋆</sup>, Yuncheng Zhang<sup>1,⋆</sup>, Hualiang Wang<sup>2,⋆</sup>, Yitian Tang<sup>1</sup>, Xiaotang Gai<sup>1</sup>,   
Chen Shen<sup>1</sup>, Songtao Jiang<sup>1</sup>, Shaosheng Cao<sup>3</sup>, Jian Wu<sup>1</sup>, Xian Wu<sup>4,†</sup>, Zuozhu Liu<sup>1,†</sup> <sup>1</sup>Zhejiang University, <sup>2</sup>Hong Kong University of Science and Technology, <sup>3</sup>Tsinghua University, <sup>4</sup>Tencent Jarvis Lab

Abstract—Recent Multimodal Large Language Models (MLLMs) are rapidly evolving into unified understanding and generation (UAG) frameworks. However, extending these unified paradigms to the medical domain is hindered by: the absence of comprehensive training and evaluation benchmarks, and the lack of broadly validated unified medical model. To address these gaps, we present a comprehensive foundation for medical UAG. First, we construct MedUAGCorpus, the largest unified medical understanding and generation dataset to date, comprising over 6 million instances across 14 imaging modalities. Second, we introduce MedUAGBench, a systematic benchmark that expands medical generation evaluation to 12 diverse tasks under standardized protocols. Finally, leveraging these resources, we develop MedUAG, an end-to-end trained unified medical model. Extensive experiments demonstrate that MedUAG achieves strong performance across a wide array of understanding and generation tasks, establishing a competitive baseline and paving the way for next-generation medical multimodal systems.

Index Terms—Medical Multimodal Learning; Unified Understanding and Generation; Benchmark and Dataset

## I. INTRODUCTION

Recent Multimodal Large Language Models (MLLMs) have evolved from isolated understanding or generation systems into unified understanding and generation (UAG) frameworks across heterogeneous modalities [1], as shown in Figure 1. Driven by the scaling and integration of data across diverse UAG tasks, these unified training paradigms have demonstrated the potential to learn universal representations within a single system, substantially reducing the reliance on taskspecific fine-tuning for downstream applications [2]–[6].

However, the few pioneering works in medical UAG [7], [8] are hindered by two significant limitations. (1) The absence of comprehensive UAG training corpora and evaluation benchmarks. While training and evaluating MLLMs necessitate large-scale data across diverse tasks, current studies often rely on a limited scope of medical imaging modalities and clinical applications. They resort to a narrow selection of common or readily accessible tasks, such as basic textconditioned image synthesis, CT-MRI translation or MRI super resolution. Consequently, some challenging generation tasks of profound clinical significance remain overlooked. (2) The lack of a broadly validated unified medical model. Beyond data and benchmarks, current medical UAG studies have yet to establish a unified model that is trained on largescale and diverse unified medical corpora and systematically validated across comprehensive understanding and generation benchmarks. Although recent approaches [7], [8] have shown promising initial results, their empirical validation remains limited in task breadth, modality coverage, and evaluation consistency. As a result, it remains unclear whether a single end-to-end model can robustly handle diverse medical multimodal demands, ranging from semantic understanding and clinical reasoning to structurally faithful and clinically meaningful image generation. This limitation not only leaves the generality and robustness of unified medical modeling insufficiently demonstrated, but also makes it difficult to assess its potential for practical application.

![](images/4bdffc4aa1119a0d7c183da23677f5d7def153425e6f3d12a1f9dab348ace1f2.jpg)  
Fig. 1. Comparison of different medical multimodal modeling paradigms. Our framework unifies understanding and generation within a single model.

To address these gaps, we introduce MedUAGBench, a comprehensive benchmark for unified medical generation. MedUAGBench extend the 5 tasks provided by UniMedVL [8] to 12 tasks in terms of the medical image generation, including various modalities under standardized prompts, metrics, and evaluation settings, enabling systematic and reproducible assessment of medical generative capability. In parallel, we construct MedUAGCorpus, the largest unified medical understanding and generation dataset to date, containing over 6M instances across 14 imaging modalities and diverse task types. Together, these resources provide the large-scale and diverse supervision foundation needed to support the comprehensive training and systematic validation of unified medical models, as summarized in Table I and Figure 2.

Building upon these resources, we develop MedUAG, an end-to-end model for unified medical understanding and gen-

TABLE I  
COMPARISON WITH EXISTING UNIFIED MEDICAL UNDERSTANDING AND GENERATION WORKS.
<table><tr><td rowspan="2"></td><td rowspan="2">Modalities in Generation</td><td colspan="2">Understanding</td><td colspan="4">Generation Tasks (#)</td><td rowspan="2">Data Scale</td></tr><tr><td>VQA</td><td>MRG</td><td>Synthesis</td><td>Translation</td><td>Reconstruction</td><td>Prediction</td></tr><tr><td>HealthGPT [7]</td><td>11</td><td>V</td><td></td><td>1</td><td>2</td><td>2</td><td>X</td><td>1.5M</td></tr><tr><td>UniMedVL [8]</td><td>10</td><td></td><td></td><td>2</td><td>1</td><td>1</td><td>1</td><td>5.6M</td></tr><tr><td>MedUAG (ours)</td><td>14</td><td></td><td>V</td><td>3</td><td>3</td><td>4</td><td>2</td><td>6.4M</td></tr></table>

## MedUAG: Unified Understanding and Generation for Medical Multimodal Models

![](images/e2c881ed2e956e6a0fa8ae96c1aa37ed22766d6c102868021401396d72480bf9.jpg)  
Fig. 2. Overview of MedUAG. Our unified framework supports both medical understanding and generation. On the understanding side, it covers VQA and MRG. On the generation side, it supports four task categories: synthesis, translation, reconstruction, and prediction across diverse medical modalities Representative demonstrations of each task type are provided.

eration. Extensive experiments show that MedUAG achieves strong performance across a broad range of tasks, establishing a competitive baseline for future research. Overall, our work provides a unified foundation spanning benchmark, data corpus, and model, which takes a step toward more general and practically useful next-generation medical multi-modal systems. Our main contributions are as follows:

• We introduce MedUAGBench, a comprehensive benchmark for unified medical generation, covering 12 tasks and various modalities under standardized evaluation protocols.

• We construct MedUAGCorpus, a large-scale unified understanding generation corpus with over 6M instances across 14 medical modalities and diverse task types.

• We develop MedUAG, an end-to-end unified medical model, and demonstrate strong performance across a wide range of understanding and generation tasks.

## II. METHODS

## A. Definition of Tasks

As illustrated in Figure 2, we study unified medical multimodal modeling through two primary task domains: understanding and generation.

1) Understanding Tasks: Understanding tasks evaluate the model’s ability to interpret medical images and extract clinically relevant semantics. We consider two primary tasks: (i) Visual Question Answering (VQA), which requires answering natural language questions grounded in specific medical images and evaluates localized diagnostic reasoning and response precision; and (ii) Medical Report Generation (MRG), which requires generating a complete clinical report from medical images. Unlike VQA, MRG requires the model to identify clinically significant findings and summarize them in a structured and coherent form.

![](images/4f2b4c1872cd69e1514c2ee00120e0cb0f81c18da94f481143228cf0a02ae569.jpg)  
Fig. 3. The construction process of datasets and model architecture of MedUAG. The dataset construction pipeline includes source data curation, task-specifi construction, and unified sample standardization. The model adopts dual image encoders and mixture of transformer experts to support both medical image understanding and generation within a single framework.

2) Generation Tasks: Generation tasks evaluate the model’s ability to produce medically meaningful images under diverse conditions. We categorize them into four paradigms. (i) Synthesis creates new medical images or edits existing ones under semantic guidance, including text-based, maskbased, and counterfactual settings. This requires the model to capture anatomical structure and pathological semantics, enabling plausible image generation or clinically specified image modification. Such capability is useful for data augmentation, medical education, and visualization of potential disease progression. (ii) Translation transforms images across domains while preserving the underlying anatomical content. It includes intra-modality translation, inter-modality translation, and structure-to-image generation, evaluating whether the model can preserve shared structures while adapting domainspecific appearance. Clinically, it can synthesize complementary views or modalities from existing scans, enriching diagnostic information without additional acquisition. (iii) Reconstruction recovers high-quality images from degraded inputs. In our dataset, this category includes low-dose CT denoising, low-dose PET denoising, low-field MRI enhancement, and undersampled MRI restoration. Successful reconstruction requires preserving diagnostically relevant details while reducing acquisition-induced distortions, supporting more efficient and lower-risk imaging protocols. (iv) Prediction generates clinically informative outputs beyond direct restoration or cross-domain translation. This category includes radiotherapy dose distribution prediction from CT images with annotated target volumes and organs at risk, as well as H&E-to-IHC pathological stain conversion. Compared with other generation tasks, prediction emphasizes outputs with more direct clinical relevance and application potential.

## B. Construction of Datasets

As illustrated in Figure 3, we construct a systematic and extensible pipeline to build the unified corpus and benchmark.

1) Source Data Curation: We first collect public datasets according to the task taxonomy, and then manually review and normalize them into a unified data pool for task-aware processing. For understanding tasks, we mainly use medical multimodal data from Hulu-Med [9]. For generation tasks, we collect public datasets covering synthesis, translation, reconstruction, and prediction [10]–[29]. After collection, all datasets are manually inspected to remove unusable samples and harmonize file structures and annotation formats, resulting in normalized raw data for subsequent construction.

2) Task-Specific Construction: Following curation, taskspecific construction further exploits the information available in each dataset and derives samples that match different downstream scenarios. This stage consists of three components: (i) Data alignment includes label restructuring and image registration, which adapt existing annotations to taskspecific requirements and, when necessary, align images of the same anatomical structures across different modalities. (ii) Source emulation simulates degraded acquisition conditions, such as noise injection and undersampling. (iii) Target generation performs task-dependent operations, including slicing and sampling volumetric data into 2D images, mask colorization for mask-based synthesis, and dose map rendering for radiotherapy dose prediction. Through these procedures, heterogeneous raw data are transformed into unified inputoutput pairs for different tasks.

3) Unified Sample Standardization: Finally, all processed samples are standardized into a unified format and split into the training corpus and benchmark at the patient level. Specifically, all images, slices, and derived samples from the same patient are assigned exclusively to either the training corpus or the benchmark, preventing data leakage between training and evaluation. Furthermore, to satisfy the fixed input size required by some generative models, images in generation tasks are resized and padded to 512 × 512 while preserving the original anatomical aspect ratio as much as possible. Each sample is then packaged with its image path, metadata, unique identifier, and task-specific text prompt. This step ensures format consistency across samples and supports unified multitask training and evaluation.

![](images/ac0bf6b5a8903a9988801888c123e1cbec3d86547aa2ef00888e17ed590f9335.jpg)

![](images/89e13de10cabf9cf3b4edf35ee5f83ff90c6f233043c176248f4f527293fa53b.jpg)  
(b) MedUAGBench  
Fig. 4. Statistical overview of dataset. (a) MedUAGCorpus: distribution of samples across anatomical systems and imaging modalities, together with the composition of the two training stages, including MedUAG-Corpus Align and MedUAG-Corpus SFT. “T2I”, “I2T”, “Recon”, “Und” and “Gen” indicate textto-image, image-to-text, reconstruction, understanding and generation, respectively. (b) MedUAGBench: distribution of benchmark samples across anatomical systems and imaging modalities, along with the hierarchical task composition covering reconstruction, translation, synthesis, and prediction.

4) Dataset Statistics: As shown in Figure 4, our dataset exhibits broad diversity in both the corpus and benchmark. MedUAGCorpus contains over 6M training instances, including a 1.76M domain-alignment set and a 4.61M instructiontuning set. It comprises more than 5.3M images and spans 14 anatomical systems and 14 imaging modalities. In contrast, MedUAGBench is more compact and evaluation-oriented. Since it focuses primarily on medical generation, its system and modality coverage is smaller, yet it still spans 10 anatomical systems and 11 imaging modalities. Moreover, the benchmark remains task-diverse, covering four core generation categories, thereby enabling comprehensive and fine-grained evaluation of medical image generation.

## C. Implementation of MedUAG

a) Model Architecure: As shown in Figure 3, MedUAG follows the design of Bagel [4] to balance the different granularity requirements of understanding and generation within a unified architecture. Specifically, it uses a ViT encoder [30] to extract high-level semantic visual tokens for understanding, and a VAE encoder-decoder [31] to model low-level latent representations for image generation. Built on a decoder-only transformer backbone [32], the model instantiates two taskspecific branches for understanding and generation, decoupling their optimization while preserving architectural consistency.

For understanding, the model takes text tokens and ViT visual tokens as input and performs autoregressive next-token prediction, following standard MLLMs [32], [33]. The objective is the cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { u n d } } = - \mathbb { E } _ { ( \mathbf { x } , \mathbf { v } , \mathbf { y } ) \sim \mathcal { D } } \left[ \sum _ { t = 1 } ^ { | \mathbf { y } | } \log p _ { \theta } ( y _ { t } \mid \mathbf { x } , \mathbf { v } , y _ { < t } ) \right] ,\tag{1}
$$

where x denotes the input text tokens, v the ViT visual tokens, and y the target output sequence.

For generation, the model operates in the VAE latent space and predicts the target velocity under the flow-matching objective [34], [35]:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e n } } = \mathbb { E } _ { ( \mathbf { z } _ { t } , \mathbf { u } _ { t } , \mathbf { c } ) \sim \mathcal { D } } \left[ \left\| f _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } ) - \mathbf { u } _ { t } \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{2}
$$

where $\mathbf { z } _ { t }$ is the noised latent feature at timestep t, c is the conditioning information, and $\mathbf { u } _ { t }$ is the target velocity.

The final text and image outputs are decoded by the LM head and VAE decoder, respectively. The overall objective is:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { u n d } } \mathcal { L } _ { \mathrm { u n d } } + \lambda _ { \mathrm { g e n } } \mathcal { L } _ { \mathrm { g e n } } , } \end{array}\tag{3}
$$

where $\lambda _ { \mathrm { u n d } }$ and $\lambda _ { \mathrm { g e n } }$ balance these two losses.

b) Domain Alignment: MedUAG is trained in two stages. The first stage adapts the pretrained backbone to the medical domain before unified instruction tuning. We construct the alignment corpus from three representative tasks: medical image reconstruction, text-to-image generation, and image captioning. These tasks provide complementary supervision: reconstruction offers dense pixel-level guidance for the generation branch, text-to-image generation strengthens medical textguided synthesis, and captioning improves image-language semantic alignment for the understanding branch. This stage establishes domain-aware representations and provides a stable initialization for subsequent instruction tuning.

c) Instruction Tuning: The second stage trains MedUAG with unified multimodal instructions covering both generation and understanding. The generation data include synthesis, translation, reconstruction, and prediction, while the understanding data include VQA and MRG. Compared with domain alignment, this stage emphasizes task diversity, compositional conditioning, and instruction responsiveness. Through joint instruction tuning, the understanding branch learns languageguided reasoning and reporting, while the generation branch learns diverse conditional image generation objectives, yielding a general-purpose medical multimodal model.

TABLE II  
AVERAGE RESULTS ACROSS THE FOUR GENERATION CATEGORIES. FOR SYNTHESIS, WE REPORT TASK-LEVEL MACRO AVERAGES OF FID, GFID, AND BIOCS. FOR TRANSLATION, RECONSTRUCTION, AND PREDICTION, WE REPORT TASK-LEVEL MACRO AVERAGES OF LPIPS, MSE, PSNR, AND SSIM.
<table><tr><td rowspan="3"></td><td rowspan="3">Medical</td><td colspan="3">Synthesis  $\operatorname { A v g } .$ </td><td colspan="4">Translation Avg.</td><td colspan="4">Reconstruction Avg.</td><td colspan="4">Prediction Avg.</td></tr><tr><td>FID↓</td><td>gFID↓</td><td>BioCS↑</td><td>LPIPS↓</td><td>MSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MSE↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>BLIP3-o [5]</td><td></td><td>320.35</td><td>316.74</td><td>0.349</td><td>0.625</td><td>0.064</td><td>12.536</td><td>0.348</td><td>0.513</td><td>0.056</td><td>13.177</td><td>0.373</td><td>0.713</td><td>0.091</td><td>10.596</td><td>0.268</td></tr><tr><td>UniWorld-V1 [36]</td><td>××</td><td>334.51</td><td>332.54</td><td>0.321</td><td>0.609</td><td>0.053</td><td>13.141</td><td>0.377</td><td>0.602</td><td>0.069</td><td>11.931</td><td>0.287</td><td>0.674</td><td>0.084</td><td>11.140</td><td>0.150</td></tr><tr><td>Bagel [4]</td><td>×</td><td>261.39</td><td>248.72</td><td>0.361</td><td>0.613</td><td>0.186</td><td>9.147</td><td>0.307</td><td>0.523</td><td>0.123</td><td>12.301</td><td>0.329</td><td>0.602</td><td>0.165</td><td>8.496</td><td>0.249</td></tr><tr><td>HealthGPT [7]</td><td>1</td><td>211.63</td><td>202.48</td><td>0.381</td><td>0.466</td><td>0.057</td><td>13.659</td><td>0.416</td><td>0.370</td><td>0.024</td><td>17.000</td><td>0.533</td><td>0.679</td><td>0.076</td><td>12.003</td><td>0.262</td></tr><tr><td>UniMedVL [8]</td><td></td><td>192.21</td><td>179.32</td><td>0.403</td><td>0.407</td><td>0.054</td><td>13.616</td><td>0.486</td><td>0.365</td><td>0.063</td><td>13.439</td><td>0.455</td><td>0.531</td><td>0.129</td><td>11.237</td><td>0.354</td></tr><tr><td>MedUAG (ours)</td><td></td><td>181.72</td><td>167.72</td><td>0.396</td><td>0.364</td><td>0.063</td><td>16.992</td><td>0.575</td><td>0.113</td><td>0.007</td><td>25.357</td><td>0.809</td><td>0.320</td><td>0.039</td><td>17.897</td><td>0.529</td></tr></table>

TABLE III

COMPARISON AMONG VARIOUS MODELS ON UNDERSTANDINGBENCHMARKS, WHERE OM.VQA INDICATES OMNIMEDVQA.
<table><tr><td>Model</td><td>Unified</td><td>VQA-RAD</td><td>SLAKE</td><td>PathVQA</td><td>OM.VQA</td><td> $\operatorname { A v g } .$ </td></tr><tr><td colspan="7">Proprietary Baselines</td></tr><tr><td>GPT-4.1</td><td>x</td><td>65.0</td><td>72.2</td><td>55.5</td><td>75.5</td><td>67.1</td></tr><tr><td>Claude Sonnet 4</td><td>x</td><td>67.6</td><td>70.6</td><td>54.2</td><td>65.5</td><td>64.5</td></tr><tr><td>Gemini-2.5-Flash</td><td>X</td><td>68.5</td><td>75.8</td><td>55.4</td><td>71.0</td><td>67.7</td></tr><tr><td colspan="7">General-purpose Baslines</td></tr><tr><td>Qwen2.5-VL-32B</td><td>X</td><td>71.8</td><td>71.2</td><td>41.9</td><td>68.2</td><td>63.3</td></tr><tr><td>InternVL3-38B</td><td>x</td><td>65.4</td><td>72.7</td><td>51.0</td><td>79.8</td><td>67.2</td></tr><tr><td>Llama3.2-11B</td><td>x</td><td>58.8</td><td>65.8</td><td>32.9</td><td>43.8</td><td>50.3</td></tr><tr><td>Qwen2.5VL-7B</td><td>x</td><td>63.2</td><td>66.8</td><td>44.1</td><td>63.6</td><td>59.4</td></tr><tr><td>InternVL3-8B</td><td>x</td><td>65.4</td><td>72.8</td><td>48.6</td><td>79.1</td><td>66.5</td></tr><tr><td>Janus-Pro-7B</td><td></td><td>49.7</td><td>55.2</td><td>35.4</td><td>59.6</td><td>50.0</td></tr><tr><td>Bagel</td><td></td><td>60.1</td><td>58.9</td><td>39.1</td><td>71.1</td><td>57.3</td></tr><tr><td colspan="7">Medical Baselines</td></tr><tr><td>LLaVA-Med-7B</td><td>x</td><td>46.6</td><td>51.9</td><td>35.2</td><td>34.8</td><td>42.1</td></tr><tr><td>RadFM</td><td>X</td><td>50.6</td><td>34.6</td><td>14.3</td><td>23.5</td><td>30.8</td></tr><tr><td>HuatuoGPT-V-7B</td><td>x</td><td>67.6</td><td>68.1</td><td>44.8</td><td>74.3</td><td>63.7</td></tr><tr><td>HealthGPT-M3</td><td>V</td><td>55.9</td><td>56.4</td><td>39.7</td><td>68.5</td><td>55.1</td></tr><tr><td>HealthGPT-L14</td><td></td><td>58.3</td><td>64.5</td><td>44.4</td><td>74.4</td><td>60.4</td></tr><tr><td>UniMedVL</td><td></td><td>61.9</td><td>75.4</td><td>53.5</td><td>85.8</td><td>69.2</td></tr><tr><td>MedUAG (ours)</td><td></td><td>75.6</td><td>78.0</td><td>54.5</td><td>77.0</td><td>71.3</td></tr></table>

## III. EXPERIMENTS

## A. Implementation Details

We train MedUAG in two stages initializing from the pretrained BAGEL-7B-MoT. In the alignment stage, the model is trained for 10k steps with a learning rate of $5 \times 1 0 ^ { - 5 }$ on 1.76M samples, including 1.14M reconstruction samples (64.6%), 191K text-to-image samples (10.8%), and 432K image-to-text samples (24.5%). In the subsequent SFT stage, training is resumed from the aligned checkpoint and continued for 20k steps with a learning rate of $2 \times 1 0 ^ { - 5 }$ on 4.61M instruction-tuning samples, consisting of 3.33M generation samples (72.3%) and 1.28M single-image understanding samples (27.7%). Training is conducted on 32 NVIDIA H800 80GB GPUs. In both stages, we set the maximum number of tokens per sample to 16,384, and adopt a constant learningrate schedule with 2,000 warmup steps, the AdamW optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 1 5 } ) .$ , EMA decay of 0.9999, and frozen VAE weights. The loss weights $\lambda _ { \mathrm { u n d } }$ and $\lambda _ { \mathrm { g e n } }$ are both set to 1.0.

## B. Benchmarks, Baselines and Metrics

a) Benchmarks: We evaluate MedUAG on both medical generation and understanding tasks to comprehensively assess its unified capability. For generation, we use MedUAGBench, a unified medical generation benchmark comprising 5,000 test instances over 12 tasks, which cover diverse medical imaging scenarios and require the model to handle heterogeneous generation objectives. For understanding, we evaluate on four widely used medical benchmarks, including VQA-RAD [37], SLAKE [38], PathVQA [39], and OmniMedVQA [40]. Together, these benchmarks provide a broad assessment of medical visual understanding ability, covering complementary reasoning demands across radiology, pathology, and general clinical question answering.

![](images/29867f50c11215ac6ab17ee7e51518ced26aaa839a0e75ba1935d227c13853dc.jpg)  
Fig. 5. Visualization of the MedUAG generation capability.

b) Baselines: We compare MedUAG with a diverse set of baselines covering both unified multimodal models and specialized medical MLLMs. For generation, we include representative unified multimodal models from the general domain, including BLIP3-o [5], UniWorld-V1 [36], and Bagel [4], as well as medical-domain unified models, including HealthGPT [7] and UniMedVL [8]. For understanding, we consider three groups of baselines: proprietary systems (GPT-4.1 [41], Claude Sonnet 4 [42], and Gemini-2.5-Flash [43]), general-purpose open-source models (e.g., Qwen2.5-VL [32], InternVL3 [44], Llama3.2 [45], Janus-Pro [3], and Bagel [4]), and specialized medical models (e.g., LLaVA-Med [46], RadFM [47], HuatuoGPT-V [48], HealthGPT [7], and UniMedVL [8]).

c) Metrics: For MedUAGBench, we follow the standard evaluation protocol for each generation category. Specifically, for synthesis, we report FID [49], generation FID (gFID), and BiomedCLIP Score (BioCS) [50] to evaluate image realism, generative fidelity, and medical semantic consistency, respectively. For translation, reconstruction, and prediction, we report LPIPS [51], MSE, PSNR, and SSIM [52] to measure perceptual similarity, pixel-wise error, reconstruction quality, and structural consistency. For medical understanding benchmarks, we use accuracy as the evaluation metric, where open-ended questions are assessed by Qwen3-VL-30B-A3B-Instruct [53].

TABLE IV  
ABLATION STUDY OF DOMAIN ALIGNMENT AND INITIALIZATION.
<table><tr><td>Align.</td><td>Init.</td><td>Synthesis (gFID↓)</td><td>Translation (LPIPS↓)</td><td>Reconstruction (PSNR↑)</td><td>Prediction (MSE↓)</td></tr><tr><td>X</td><td>Scratch</td><td>332.40</td><td>0.478</td><td>22.449</td><td>0.053</td></tr><tr><td>x</td><td>Bagel</td><td>128.28</td><td>0.396</td><td>24.476</td><td>0.040</td></tr><tr><td>√</td><td>Scratch</td><td>309.50</td><td>0.475</td><td>23.078</td><td>0.057</td></tr><tr><td></td><td></td><td>167.72</td><td>0.364</td><td>25.357</td><td>0.039</td></tr><tr><td>V</td><td>Bagel</td><td></td><td></td><td></td><td></td></tr></table>

TABLE V  
PSNR GAINS FROM DOMAIN ALIGNMENT ON RECONSTRUCTION TASKS.
<table><tr><td>Init.</td><td>Low-Dose CT</td><td>Low-Dose PET</td><td>Low-Field MRI</td><td>Undersampled MRI</td><td>Avg.</td></tr><tr><td>From Scratch</td><td>+1.13</td><td>+1.00</td><td>+0.33</td><td>+0.05</td><td>+0.63</td></tr><tr><td>From Bagel</td><td>+1.14</td><td>+0.05</td><td>+1.89</td><td>+0.44</td><td>+0.88</td></tr></table>

## C. Main Results

1) Comparison on MedUAGBench: Table II reports the quantitative results on MedUAGBench. MedUAG consistently outperforms both general-domain unified models and prior medical unified models across most generation settings, demonstrating the benefit of large-scale medical unified training. Its advantage is especially clear on reconstruction and prediction, where preserving anatomical structure and producing clinically grounded outputs are essential. These results suggest that MedUAG does not merely improve image realism, but also learns generation capabilities aligned with medical structure and task semantics. The qualitative examples in Figure 5 further show that MedUAG produces realistic and structurally faithful outputs across diverse imaging modalities.

2) Comparison on Understanding Benchmarks: Table III reports results on medical understanding benchmarks. Med-UAG achieves the best average accuracy of 71.3 against 16 baselines, leading on VQA-RAD and SLAKE with scores of 75.6 and 78.0, respectively, while remaining competitive on PathVQA and OmniMedVQA. Notably, although MedUAG is jointly trained for both generation and understanding, with understanding data comprising less than 30% of the corpus in both stages, it still demonstrates strong visual reasoning across radiology, pathology, and general medical QA. Its gap to UniMedVL on OmniMedVQA suggests that data composition, the understanding-generation mixing ratio, and task balancing remain important directions for future unified medical multimodal models.

## D. Ablation Studies

1) Effect of Domain Alignment: To assess domain alignment, we compare models with and without this stage under both from-scratch and Bagel initializations, as shown in Table IV. Domain alignment mainly benefits structure-preserving tasks: with Bagel initialization, it lowers translation LPIPS by 0.032 and improves reconstruction PSNR by 0.881, with consistent gains across reconstruction benchmarks in Table V. However, synthesis performance decreases after alignment, revealing a trade-off between anatomical fidelity and generative diversity. This suggests that alignment-induced structural priors should be retained, while synthesis-oriented objectives should be progressively strengthened in later training to better balance structural fidelity and generative realism.

![](images/20279605b3edf025e094beed1e8e0cb9860de5f6561923c660abcc212fa6830d.jpg)  
Fig. 6. Comparison of different SFT data ratios.

2) Effect of Base Model: Table IV further illustrates the efficacy of different initialization strategies. Compared to training from scratch, Bagel initialization consistently yields substantial gains across nearly all tasks, most notably in synthesis and reconstruction. Without domain alignment, Bagel drastically reduces gFID from 332.40 to 128.28 and boosts reconstruction PSNR by 2.027. This superiority persists even after domain alignment is introduced, where Bagel still provides the optimal foundation for translation, reconstruction, and prediction. These results underscore that general-domain unified pretraining establishes a robust representational prior, which significantly lowers the barrier for downstream medical adaptation and ensures a better optimization starting point.

3) Effect of Data Scale: We further analyze the effect of SFT data scale as shown in Figure 6. Increasing instructiontuning data improves performance across all tasks, but the gains gradually saturate at larger scales. Understanding and synthesis show more sustained improvements than reconstruction and prediction, likely because they occupy smaller portions of the SFT set and therefore benefit more from increased diversity and coverage as the total data grows. These results suggest that, beyond sufficient scale, further gains depend more on data diversity, task balance, and quality control than on simply adding more samples.

## IV. APPLICATION EXPLORATION

Beyond unified understanding and generation, MedUAG can also synthesize multimodal data for downstream training in low-resource settings, where paired clinical data are often scarce, costly, or privacy-restricted [54]. To evaluate this potential, we conduct an exploratory experiment on MRG task from CheXpert [55] using Qwen3-VL-4B as the backbone. We compare three training settings: 1K real image-report pairs, 1K real pairs augmented with 1K synthetic images generated by MedUAG from the corresponding reports, and a 2K-real upper bound. As shown in Table VI, synthetic augmentation improves the 1K-real baseline by 0.086 on average and narrows the gap to the 2K-real upper bound to only 0.009. These results suggest that MedUAG-generated images can provide effective supervision when real paired data are limited, highlighting the potential of unified medical multimodal models for scalable and cost-effective data augmentation.

TABLE VI  
EFFECT OF SYNTHETIC DATA ON MEDICAL REPORT GENERATION.
<table><tr><td>#Real</td><td>#Synthetic</td><td>#Total</td><td>METEOR</td><td>BERTScore</td><td>LLM Judge</td><td>Average</td></tr><tr><td>1K</td><td>0</td><td>1K</td><td>0.098</td><td>0.394</td><td>0.037</td><td>0.176</td></tr><tr><td>1K</td><td>1K</td><td>2K</td><td>0.191</td><td>0.495</td><td>0.100</td><td>0.262</td></tr><tr><td>2K</td><td>0</td><td>2K</td><td>0.196</td><td>0.491</td><td>0.125</td><td>0.271</td></tr></table>

## V. RELATED WORK

Recent medical vision-language models adapt generalpurpose MLLMs [32], [45], [53] to clinical scenarios [46], [47], [56]–[58]. These models have shown strong capabilities in medical VQA, report generation, and clinical reasoning by leveraging medical image-text alignment, instruction tuning, and large-scale domain-specific data. However, most Med-MLLMs are still centered on visual comprehension and text generation. They cannot directly synthesize, translate, reconstruct, or manipulate medical images at the pixel level, limiting their applicability to generative clinical tasks.

UAG has recently become an active direction in generaldomain multimodal learning [3]–[6], [36]. Existing methods explore different design choices, including coupling LLMs with generative modules, discretizing images into visual tokens, decoupling visual encoders, and adopting expert-based architectures to support both perception and generation. Despite their strong open-domain generalization, these models are primarily trained on natural images. As a result, they often struggle with clinical semantics, anatomical fidelity, and finegrained pathology in medical imaging.

Recent works have begun to explore unified multimodal models for medicine. HealthGPT [7] unifies comprehension and generation with heterogeneous knowledge adaptation, and UniMed-VL [8] models further mark important progress toward unified medical AI. Nevertheless, existing efforts remain constrained by limited data scale, narrow generation task coverage, and insufficiently standardized evaluation. In contrast, our work provides a more comprehensive foundation by constructing a large-scale corpus, a systematic generation benchmark, and an end-to-end unified medical model.

## VI. CONCLUSION

We present MedUAG, a unified medical multimodal model, together with MedUAGBench and MedUAGCorpus for systematic training and evaluation of medical understanding and generation. MedUAG achieves strong performance across both task families, with clear gains in reconstruction and prediction, demonstrating the promise of unified modeling for generalpurpose medical AI. We also identify key challenges, including limitations in some understanding settings and a trade-off between structural fidelity and synthesis diversity. Future work will further improve data composition, task balancing, and training strategies toward more robust and clinically useful medical multimodal systems.

## REFERENCES

[1] S. Zhao, X. Zhang, J. Guo, J. Hu, L. Duan, M. Fu, Y. X. Chng, G.- H. Wang, Q.-G. Chen, Z. Xu et al., “Unified multimodal understanding and generation models: Advances, challenges, and opportunities,” arXiv preprint arXiv:2505.02567, 2025.

[2] Y. Ge, S. Zhao, J. Zhu, Y. Ge, K. Yi, L. Song, C. Li, X. Ding, and Y. Shan, “Seed-x: Multimodal models with unified multigranularity comprehension and generation,” 2025. [Online]. Available: https://arxiv.org/abs/2404.14396

[3] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, “Janus-pro: Unified multimodal understanding and generation with data and model scaling,” arXiv preprint arXiv:2501.17811, 2025.

[4] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, G. Shi, and H. Fan, “Emerging properties in unified multimodal pretraining,” 2025. [Online]. Available: https://arxiv.org/abs/2505.14683

[5] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese et al., “Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset,” arXiv preprint arXiv:2505.09568, 2025.

[6] X. Wang, Y. Cui, J. Wang, F. Zhang, Y. Wang, X. Zhang, Z. Luo, Q. Sun, Z. Li, Y. Wang et al., “Multimodal learning with next-token prediction for large multimodal models,” Nature, pp. 1–7, 2026.

[7] T. Lin, W. Zhang, S. Li, Y. Yuan, B. Yu, H. Li, W. He, H. Jiang, M. Li, X. Song et al., “Healthgpt: A medical large vision-language model for unifying comprehension and generation via heterogeneous knowledge adaptation,” arXiv preprint arXiv:2502.09838, 2025.

[8] J. Ning, W. Li, C. Tang, J. Lin, C. Ma, C. Zhang, J. Liu, Y. Chen, S. Gao, L. Liu et al., “Unimedvl: Unifying medical multimodal understanding and generation through observation-knowledge-analysis,” arXiv preprint arXiv:2510.15710, 2025.

[9] S. Jiang, Y. Wang, S. Song, T. Hu, C. Zhou, B. Pu, Y. Zhang, Z. Yang, Y. Feng, J. T. Zhou, J. Hao, Z. Chen, R. Wu, T. Tang, J. Lv, H. Xu, H. Wang, J. Xiao, B. Feng, F. Zhu, K. Li, W. Xie, J. Sun, J. Wu, and Z. Liu, “Hulu-med: A transparent generalist model towards holistic medical vision-language understanding,” 2025. [Online]. Available: https://arxiv.org/abs/2510.08668

[10] P. Chambon, J.-B. Delbrouck, T. Sounack, S.-C. Huang, Z. Chen, M. Varma, S. Q. Truong, C. T. Chuong, and C. P. Langlotz, “Chexpert plus: Augmenting a large chest x-ray dataset with text radiology reports, patient demographics and additional image formats,” 2024. [Online]. Available: https://arxiv.org/abs/2405.19538

[11] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “Medmnist v2 - a large-scale lightweight benchmark for 2d and 3d biomedical image classification,” Scientific Data, vol. 10, no. 1, Jan. 2023. [Online]. Available: http://dx.doi.org/10.1038/ s41597-022-01721-8

[12] Y. Xie, C. Zhou, L. Gao, J. Wu, X. Li, H.-Y. Zhou, S. Liu, L. Xing, J. Zou, C. Xie, and Y. Zhou, “Medtrinity-25m: A large-scale multimodal dataset with multigranular annotations for medicine,” 2025. [Online]. Available: https://arxiv.org/abs/2408.02900

[13] J. Ruckert, L. Bloch, R. Br¨ ungel, A. Idrissi-Yaghir, H. Sch¨ afer, C. S.¨ Schmidt, S. Koitka, O. Pelka, A. B. Abacha, A. G. Seco de Herrera, H. Muller, P. A. Horn, F. Nensa, and C. M. Friedrich, “Rocov2:¨ Radiology objects in context version 2, an updated multimodal image dataset,” Scientific Data, vol. 11, no. 1, Jun. 2024. [Online]. Available: http://dx.doi.org/10.1038/s41597-024-03496-6

[14] J. Ma, Y. Zhang, S. Gu, C. Zhu, C. Ge, Y. Zhang, X. An, C. Wang, Q. Wang, X. Liu, S. Cao, Q. Zhang, S. Liu, Y. Wang, Y. Li, J. He, and X. Yang, “Abdomenct-1k: Is abdominal organ segmentation a solved problem?” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6695–6714, 2022.

[15] C. Ma, Y. Ji, J. Ye, L. Zhang, Y. Chen, T. Li, M. Li, J. He, and H. Shan, “Towards interpretable counterfactual generation via multimodal autoregression,” 2025. [Online]. Available: https: //arxiv.org/abs/2503.23149

[16] ULF-EnC Organizers, “Ulf-enc challenge: Ultra-low-field mri image enhancement challenge,” https://www.synapse.org/Synapse:syn65485242/ wiki/, 2025, accessed 2026-04-02.

[17] IXI Consortium, “Ixi dataset,” https://brain-development.org/ixi-dataset/, 2007, public MRI dataset with T1, T2 and PD-weighted images; accessed 2026-04-02.

[18] A. Thummerer, E. van der Bijl, A. J. Galapon, F. Kamp, M. Savenije, C. Muijs, S. Aluwini, R. J. H. M. Steenbakkers, S. Beuel, M. P. Intven, J. A. Langendijk, S. Both, S. Corradini, V. Rogowski, M. Terpstra, N. Wahl, C. Kurz, G. Landry, and M. Maspero, “Synthrad2025 grand challenge dataset: Generating synthetic cts for radiotherapy from head to abdomen,” Medical Physics, vol. 52, no. 7, Jul. 2025. [Online]. Available: http://dx.doi.org/10.1002/mp.17981

[19] K. Jin, X. Huang, J. Zhou, Y. Li, Y. Yan, Y. Sun, Q. Zhang, Y. Wang, and J. Ye, “Fives: A fundus image dataset for artificial intelligence based vessel segmentation,” Scientific Data, vol. 9, no. 1, p. 475, 2022. [Online]. Available: https://doi.org/10.1038/s41597-022-01564-3

[20] J. Staal, M. Abramoff, M. Niemeijer, M. Viergever, and B. van Ginneken, “Ridge-based vessel segmentation in color images of the retina,” IEEE Transactions on Medical Imaging, vol. 23, no. 4, pp. 501–509, 2004.

[21] T. R. Moen, B. Chen, D. R. Holmes, 3rd, X. Duan, Z. Yu, L. Yu, S. Leng, J. G. Fletcher, and C. H. McCollough, “Low-dose CT image and projection dataset,” Medical Physics, vol. 48, no. 2, pp. 902–911, 2021, pMID: 33202055; PMCID: PMC7985836.

[22] S. G. Armato, III, G. McLennan, L. Bidaut, M. F. McNitt-Gray, C. R. Meyer, A. P. Reeves, B. Zhao, D. R. Aberle, C. I. Henschke, E. A. Hoffman et al., “Data From LIDC-IDRI,” The Cancer Imaging Archive, 2015. [Online]. Available: https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

[23] S. Xue, H. Wang, Y. Chen, F. Liu, H. Zhu, M. Viscione, R. Guo, A. Rominger, B. Li, and K. Shi, “ UDPET: Ultra-low Dose PET Imaging Challenge Dataset ,” in proceedings of Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, vol. LNCS 15972. Springer Nature Switzerland, September 2025.

[24] J. Zbontar, F. Knoll, A. Sriram, T. Murrell, Z. Huang, M. J. Muckley, A. Defazio, R. Stern, P. Johnson, M. Bruno et al., “fastmri: An open dataset and benchmarks for accelerated mri,” arXiv preprint arXiv:1811.08839, 2018.

[25] S. Liu, C. Zhu, F. Xu, X. Jia, Z. Shi, and M. Jin, “Bci: Breast cancer immunohistochemical image generation through pyramid pix2pix,” 2022. [Online]. Available: https://arxiv.org/abs/2204.11425

[26] F. Li, Z. Hu, W. Chen, and A. Kak, “Adaptive supervised patchnce loss for learning h&e-to-ihc stain translation with inconsistent groundtruth image pairs,” 2023. [Online]. Available: https://arxiv.org/abs/2303.06193

[27] A. Babier, B. Zhang, R. Mahmood, K. L. Moore, T. G. Purdie, A. L. McNiven, and T. C. Y. Chan, “Openkbp: The open-access knowledge-based planning grand challenge and dataset,” Medical Physics, vol. 48, no. 9, p. 5549–5561, Jun. 2021. [Online]. Available: http://dx.doi.org/10.1002/mp.14845

[28] R. Gao, M. Diallo, H. Liu, A. Magliari, J. Sackett, W. Verbakel, S. Meyers, R. Mcbeth, M. Zarepisheh, S. Arberet, M. Kraus, F. C. Ghesu, and A. Kamen, “Automating rt planning at scale: High quality data for ai training,” 2025. [Online]. Available: https://arxiv.org/abs/2501.11803

[29] AAPM Grand Challenge Organizers, “Gdp-hmm challenge: Generalizable dose prediction for heterogenous multi-cohort and multi-site radiotherapy planning,” https://www.aapm.org/GrandChallenge/GDP-HMM/, 2025, associated with the HMM-RT dataset; accessed 2026-04-02

[30] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[31] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.

[32] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2.5-vl technical report,” eprint arXiv: 2502.13923, 2025.

[33] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892– 34 916, 2023.

[34] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint arXiv:2209.03003, 2022.

[35] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[36] B. Lin, Z. Li, X. Cheng, Y. Niu, Y. Ye, X. He, S. Yuan, W. Yu, S. Wang, Y. Ge, Y. Pang, and L. Yuan, “Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation,” 2025. [Online]. Available: https://arxiv.org/abs/2506.03147

[37] J. J. Lau, S. Gayen, A. Ben Abacha, and D. Demner-Fushman, “A dataset of clinically generated visual questions and answers about radiology images,” Scientific data, vol. 5, no. 1, p. 180251, 2018.

[38] B. Liu, L.-M. Zhan, L. Xu, L. Ma, Y. Yang, and X.-M. Wu, “Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering,” in 2021 IEEE 18th international symposium on biomedical imaging (ISBI). IEEE, 2021, pp. 1650–1654.

[39] X. He, Y. Zhang, L. Mou, E. Xing, and P. Xie, “Pathvqa: 30000+ questions for medical visual question answering,” arXiv preprint arXiv:2003.10286, 2020.

[40] Y. Hu, T. Li, Q. Lu, W. Shao, J. He, Y. Qiao, and P. Luo, “Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 170–22 183.

[41] OpenAI. (2025, Apr.) Introducing gpt-4.1 in the api. [Online]. Available: https://openai.com/index/gpt-4-1

[42] Anthropic. (2025, May) Introducing claude 4. [Online]. Available: https://www.anthropic.com/news/claude-4

[43] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[44] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[45] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[46] C. Li, C. Wong, S. Zhang, N. Usuyama, H. Liu, J. Yang, T. Naumann, H. Poon, and J. Gao, “Llava-med: Training a large language-and-vision assistant for biomedicine in one day,” Advances in Neural Information Processing Systems, vol. 36, pp. 28 541–28 564, 2023.

[47] C. Wu, X. Zhang, Y. Zhang, H. Hui, Y. Wang, and W. Xie, “Towards generalist foundation model for radiology by leveraging web-scale 2d&3d medical data,” Nature Communications, vol. 16, no. 1, p. 7866, 2025.

[48] J. Chen, C. Gui, R. Ouyang, A. Gao, S. Chen, G. H. Chen, X. Wang, Z. Cai, K. Ji, X. Wan et al., “Towards injecting medical visual knowledge into multimodal llms at scale,” in Proceedings of the 2024 conference on empirical methods in natural language processing, 2024, pp. 7346– 7370.

[49] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.

[50] S. Zhang, Y. Xu, N. Usuyama, H. Xu, J. Bagga, R. Tinn, S. Preston, R. Rao, M. Wei, N. Valluri et al., “Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs,” arXiv preprint arXiv:2303.00915, 2023.

[51] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[52] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[53] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[54] J. Wang, K. Wang, Y. Yu, Y. Lu, W. Xiao, Z. Sun, F. Liu, Z. Zou, Y. Gao, L. Yang et al., “Self-improving generative foundation model for synthetic medical image generation and clinical applications,” Nature Medicine, vol. 31, no. 2, pp. 609–617, 2025.

[55] J. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute, H. Marklund, B. Haghgoo, R. Ball, K. Shpanskaya et al., “Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison,” in Proceedings of the AAAI conference on artificial intelligence, vol. 33, no. 01, 2019, pp. 590–597.

[56] J. Chen, C. Gui, R. Ouyang, A. Gao, S. Chen, G. H. Chen, X. Wang, Z. Cai, K. Ji, X. Wan et al., “Towards injecting medical visual knowledge into multimodal llms at scale,” in Proceedings of the 2024 conference on empirical methods in natural language processing, 2024, pp. 7346– 7370.

[57] S. Jiang, Y. Wang, S. Song, T. Hu, C. Zhou, B. Pu, Y. Zhang, Z. Yang, Y. Feng, J. T. Zhou et al., “Hulu-med: A transparent generalist model towards holistic medical vision-language understanding,” arXiv preprint arXiv:2510.08668, 2025.

[58] K. Zhang, R. Zhou, E. Adhikarla, Z. Yan, Y. Liu, J. Yu, Z. Liu, X. Chen, B. D. Davison, H. Ren et al., “A generalist vision–language foundation model for diverse biomedical tasks,” Nature medicine, vol. 30, no. 11, pp. 3129–3141, 2024.