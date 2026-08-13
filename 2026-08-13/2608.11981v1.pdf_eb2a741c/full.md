# Benchmarking Trustworthiness of SLMs: Pre-trained vs. Compressed

1<sup>st</sup> Haokun Lin

2<sup>nd</sup> Kaijie Zhu

3<sup>rd</sup> Haobo Xu

4<sup>th</sup> Yichen Wu

Institute of Automation, CAS

Institute of Automation, CAS

Tsinghua University

Harvard Medical School

Beijing, China

Beijing, China

Beijing, China

Boston, U.S.

ORCID 0009-0000-1084-7115 ORCID 0009-0002-6220-1476 ORCID 0009-0007-8311-7958 ORCID 0000-0003-2859-3285

5<sup>th</sup> Zhichao Lu

6<sup>th</sup> Qingfu Zhang

City University of Hong Kong

City University of Hong Kong

7<sup>th</sup> Zhenan Sun

Hong Kong, China

Institute of Automation, CAS

ORCID 0000-0002-4618-3573

Hong Kong, China

ORCID 0000-0003-0786-0671

Beijing, China

ORCID 0000-0003-4029-9935

Abstract—Small Language Models (SLMs) have emerged as a more efficient alternative to traditional Large Language Models (LLMs), offering promising potential in resource-constrained scenarios. Existing approaches to building SLMs typically follow two paths: training compact models from scratch, or compressing larger pre-trained models using methods such as pruning, quantization, or distillation. As language models become increasingly integrated into real-world applications, ensuring their trustworthiness has become a critical concern. However, how to build trustworthy SLMs remains an underexplored question. In this work, we present a comprehensive evaluation of SLM trustworthiness across multiple dimensions, including fairness, robustness, privacy, and ethics. We first examine the effects of pruning and quantization, and find that quantization is significantly more effective in preserving trustworthiness compared to pruning. More importantly, we demonstrate that compressing a reliable large model via quantization can produce SLMs with superior trustworthiness and adaptability compared to using small models trained from scratch. Furthermore, knowledge distillation from trustworthy teacher models can further enhance the reliability of SLMs. We hope our findings provide practical guidance and a foundation for future research into the development and deployment of trustworthy small language models.

Index Terms—Small language models, Pruning, Quantization, Knowledge distillation, Trustworthiness

## I. INTRODUCTION

Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks [1]–[3]. This success is largely attributed to their massive parameter scales—models with over 7 billion parameters have become a popular baseline. However, deploying such large models incurs significant computational and memory costs, making them impractical for resourceconstrained environments such as edge devices. As a result, Small Language Models (SLMs), typically with fewer than 1–2 billion parameters, have attracted increasing attention for their efficiency during inference. There are two primary approaches to obtaining SLMs: (1) designing and training compact models from scratch using curated datasets and optimized architectures [4], [5], and (2) compressing larger LLMs via techniques like pruning, quantization, and distillation [6]–[10].

The growing deployment of LLMs in real-world applications brings trustworthiness to the forefront, involving aspects such as safety, fairness, robustness, privacy, and ethical alignment [11]. Numerous studies [12]–[14] have investigated trustworthiness in LLMs, particularly for models exceeding 7B parameters or proprietary LLMs. However, how to ensure the trustworthiness of SLMs remains an open question. Prior works [15]–[17] have explored the effect of compression on trustworthiness, but few directly compare pre-trained SLMs with compressed larger models under a unified framework.

In this work, we conduct a comprehensive analysis to understand how to build more trustworthy SLMs, using fairness, robustness, privacy, and ethics as key evaluation dimensions. First, we evaluate the impact of pruning and quantization on model trustworthiness. Our findings across different model families and sizes suggest that pruning can impair reliability, whereas quantization largely preserves the trustworthiness of the original full-precision models. Therefore, we advocate using quantization as a more reliable path toward building SLMs. Second, we directly compare several pre-trained SLMs (under 1B parameters) with quantized larger models. The results consistently show that quantized LLMs outperform pre-trained SLMs across all trustworthiness dimensions, indicating that compressing a reliable large model is more effective than training a small model from scratch. Lastly, we explore the effect of knowledge distillation and find that distilling from a more trustworthy teacher can further enhance the reliability of SLMs. Our main contributions are summarized as follows:

• We conduct a thorough evaluation of compressed SLMs and recommend quantization over pruning as a more effective and reliable technique for preserving trustworthiness.

• We highlight a key insight: quantizing a larger, more trustworthy model yields more robust and flexible SLMs compared to directly using pre-trained small models.

• We discover that knowledge distillation effectively improves SLM trustworthiness by leveraging the guidance of stronger teacher models.

![](images/024f5233d215f62cbc01a76fa5b296165aa1fbe71ed6be44f98a84d611327386.jpg)  
Fig. 1: Our evaluation framework for assessing the trustworthiness of SLMs, including state-of-the-art pruning and quantization methods, a comparison between pre-trained SLMs and compressed larger models, and the impact of distillation. Our result suggest quantization as the preferred compression technique, while surpassing pre-trained SLMs. Distillation could provide additional improvements.

## II. RELATED WORK

## A. Pre-Trained Small Language Models

Recently, small language models (SLMs) have shown strong potential in resource-constrained scenarios compared to their large-scale counterparts. Several works focus on designing SLMs from scratch. For example, MobiLlama [4] improves efficiency by reducing redundancy in transformer blocks, while SmolLM2 [5] maximizes model performance through a multi stage rebalancing of diverse training data sources. In addition, the Qwen 2.5 [2] and Llama 3.2 [3] series have released pretrained SLMs with parameter scales ranging from 0.5B to 1B. In this research, we explore the trustworthiness differences between such pre-trained SLMs and compressed LLMs.

## B. Model Compression

Network Pruning is a widely used compression technique that reduces model size by eliminating redundant or unimportant weights [18]–[20]. For large language models, unstructured pruning sets individual unimportant weights to zero, while N:M semi-structured pruning enforces a constraint that at least N out of every contiguous M weights must be zero. This semistructured format is particularly advantageous on NVIDIA GPUs, as it enables acceleration of matrix multiply-accumulate operations. Among representative methods, SparseGPT [7] improves the efficiency of the traditional Optimal Brain Surgeon (OBS) algorithm by adjusting the remaining weights to mini mize the loss change caused by pruning. Wanda [21] proposes to leverage input activations as the importance indicator, while achieving performance comparable to SparseGPT. Structured pruning [22] often causes notable performance degradation for LLMs without additional retraining; therefore, we do not include it in our exploration.

Network Quantization reduces the memory footprint of models by converting weight matrices into low-bit representations [23]–[29]. Post-training quantization has gained popularity for LLMs due to its low computational cost and the absence of retraining requirements. GPTQ [6] leverages second-order information to perform error compensation during quantization. AWQ [8] identifies and protects salient weights with the assistance of the activation matrix, preserving model accuracy. DuQuant [9] introduces rotation and permutation transformations to enhance low-bit weight-activation quantization.

Knowledge Distillation (KD) is a key technique for both compressing LLMs and improving their downstream performance. The central idea is to transfer the knowledge encoded in a high-capacity teacher model into a smaller student model, typically by training the student to mimic the teacher’s output distributions [30]–[35]. This paradigm enables compact models to retain much of the performance of their larger counterparts while significantly reducing computational overhead. Moreover, KD provides an effective means to extract task-specific or domain-specific knowledge from proprietary or closed-source models, serving as a practical alternative to direct access [36]. In this work, we provide empirical insights that inform the trustworthy use of compressed models with these three techniques.

## C. Model Trustworthiness

TrustLLM [12] evaluates the reliability of language models across six key dimensions. Truthfulness measures whether a model conveys accurate information. Safety focuses on preventing harmful, unsafe, or unlawful outputs. Fairness ensures that models do not introduce bias or discrimination across different demographics. Robustness captures a model’s ability to maintain stable performance under diverse inputs or conditions. Privacy emphasizes protecting individual autonomy and sensitive data. Together, these dimensions provide a comprehensive framework for assessing the trustworthiness of LLMs. For SLMs, prior research [15] has mainly examined the safety of quantized LLMs, while [16] focuses on the trustworthiness of compressed large models. In contrast, our work investigates the trustworthiness of SLMs by directly comparing pre-trained SLMs with compressed larger models.

<table><tr><td>Model</td><td>Ethics</td><td>Privacy</td><td>Robustness</td><td>Fairness</td><td>Overall</td></tr><tr><td>Gemma-1.1-7B</td><td>82.19%</td><td>60.13%</td><td>68.90%</td><td>44.73%</td><td>63.99%</td></tr><tr><td>Unstructured-Sparsegpt Unstructured-Wanda</td><td>81.12% 80.90%</td><td>58.40% 56.86%</td><td>63.84% 64.41%</td><td>38.43% 29.82%</td><td>60.45% 58.00%</td></tr><tr><td>4:8-Sparsegpt 4:8-Wanda</td><td>77.36% 74.87%</td><td>59.24% 52.70%</td><td>65.20% 63.57%</td><td>32.66% 54.50%</td><td>58.61% 61.41%</td></tr><tr><td>2:4-Sparsegpt 2:4-wanda</td><td>72.92% 73.08%</td><td>56.37% 55.35%</td><td>61.82% 59.27%</td><td>24.43% 36.23%</td><td>53.89% 55.98%</td></tr><tr><td>Llama-3.1-8B</td><td>81.29%</td><td>49.16%</td><td>71.62%</td><td>36.03%</td><td>59.52%</td></tr><tr><td>Unstructure-Sparsegpt Unstructured-Wanda</td><td>80.05% 77.18%</td><td>46.08% 51.56%</td><td>60.57% 61.04%</td><td>44.08% 31.76%</td><td>57.69% 55.38%</td></tr><tr><td>4:8-Sparsegpt 4:8-Wanda</td><td>76.74% 74.98%</td><td>53.40% 48.67%</td><td>54.87% 57.29%</td><td>44.10%</td><td>57.28%</td></tr><tr><td>2:4-Sparsegpt 2:4-Wanda</td><td>66.02%</td><td>47.75%</td><td>56.76%</td><td>46.19% 44.65%</td><td>56.78% 53.79%</td></tr><tr><td>Qwen2.5-7B</td><td>59.61% 83.78%</td><td>42.22%</td><td>47.27%</td><td>72.04%</td><td>55.29%</td></tr><tr><td>Unstructured-Sparsegpt</td><td>84.29%</td><td>41.53% 44.22%</td><td>67.05%</td><td>28.33%</td><td>55.17%</td></tr><tr><td>Unstructured-Wanda</td><td>84.42%</td><td>38.38%</td><td>58.64% 61.01%</td><td>29.51% 28.79%</td><td>54.17% 53.15%</td></tr><tr><td>4:8-Sparsegpt</td><td>83.99%</td><td></td><td></td><td></td><td></td></tr><tr><td>4:8-Wanda</td><td></td><td>40.87%</td><td>60.27%</td><td>30.42%</td><td>53.89%</td></tr><tr><td></td><td>83.86%</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>38.41%</td><td>56.35%</td><td>36.14%</td><td>53.69%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2:4-Sparsegpt 2:4-Wanda</td><td>82.12% 82.20%</td><td>40.36% 39.48%</td><td>53.37% 56.48%</td><td>30.65% 29.89%</td><td>51.62% 52.01%</td></tr></table>

TABLE I: Trustworthiness assessment for pruned LLMs.

## III. PRELIMINARY

Quantization. The general b-bit uniform quantization $\mathcal { Q } _ { b } ( \cdot )$ can be represented as:

$$
\begin{array} { r } { \hat { \mathbf { x } } = \mathcal { Q } _ { b } ( \mathbf { x } ) = s \cdot \Pi _ { \Omega ( b ) } ( \mathbf { x } / s ) , } \end{array}\tag{1}
$$

where s is the quantization step size, and $\Pi _ { \Omega ( b ) }$ is the projection function onto the set of b-bit integers $\Omega ( b ) \stackrel { \cdot \cdot } { = } \left\{ 0 , 1 , . . . , 2 ^ { b } - 1 \right\}$ Knowledge Distillation. Given an input x, the teacher model produces logits $z ^ { ( T ) }$ and the student model produces $z ^ { ( S ) }$ . With temperature T, the softened probabilities are

$$
p _ { i } ^ { ( T ) } = \frac { \exp ( z _ { i } ^ { ( T ) } / T ) } { \sum _ { j } \exp ( z _ { j } ^ { ( T ) } / T ) } , \quad q _ { i } = \frac { \exp ( z _ { i } ^ { ( S ) } / T ) } { \sum _ { j } \exp ( z _ { j } ^ { ( S ) } / T ) } .\tag{2}
$$

The supervised distillation [30] minimizes the KL divergence:

$$
\mathcal { L } _ { \mathrm { K D } } = \sum _ { i } p _ { i } ^ { ( T ) } \log \frac { p _ { i } ^ { ( T ) } } { q _ { i } } .\tag{3}
$$

## IV. SAFETY EVALUATION FOR SMALL LLMS

In this section, we present a series of experiments to address our core research question: How can we obtain more trustworthy small language models? To systematically explore this, we further investigate the following sub-questions:

• RQ1: Are pruned large language models trustworthy?

• RQ2: Are quantized large language models trustworthy?

• RQ3: Are compressed LLMs more trustworthy than pretrained SLMs?

• RQ4: Can distillation improve trustworthiness?

## A. Setup

Language Models and Compression Methods. We conduct comprehensive evaluations across a range of instruction-tuned language models. For pre-trained small language models, we select the following models: h2o-danube3-500m-Chat [37], MobiLlama-500m-Chat [4], SmolLM2-360M-Instruct [5], and Qwen2.5-0.5B-Instruct [2]. For compression methods, we adopt widely used techniques, including SparseGPT [7] and Wanda [21] for pruning, as well as GPTQ [6] and AWQ [8] for quantization. All compression experiments are calibrated using the WikiText-v2 dataset. To obtain compressed SLMs, we apply these compression methods to commonly used LLMs, such as Gemma [1], Llama [3], and Qwen [2]. For distillation, we conduct supervised knowledge distillation [30] on Qwen2.5 models using Alpaca dataset [38].

Trustworthiness Measurement. We adopt the TrustLLM [12] benchmark as the primary framework for assessing trustworthiness. Specifically, we evaluate models across four dimensions: machine ethics, privacy, robustness, and fairness. For ethics, we use the ETHICS [39] and Social-Chem-101 [40] datasets to assess implicit ethics, and the MoralChoice dataset for explicit ethics evaluation. For privacy, we adopt agreement tests on private information usage and privacy scenario tasks. For robustness, we use AdvGLUE [41] and AdvInstruction to measure resistance to natural noise, and further examine performance on out-of-distribution (OOD) detection and generalization tasks. For fairness, we evaluate from three perspectives: disparagement, representation bias, and preference bias in subjective choices. We report the average accuracy on sub-tasks under each dimension, and the overall average score is used as the indicator of model trustworthiness.

<table><tr><td>Model</td><td>Ethics</td><td>Privacy</td><td>Robustness</td><td>Fairness</td><td>Overall</td></tr><tr><td>Llama-3.2-1B 1B-INT4-AWQ</td><td>73.98% 69.71%</td><td>60.21% 59.71%</td><td>59.43% 54.17%</td><td>70.81% 79.11%</td><td>66.11% 65.67%</td></tr><tr><td>1B-INT4-GPTQ Llama-3.2-3B</td><td>72.32% 82.88%</td><td>57.74% 60.15%</td><td>56.19% 66.42%</td><td>68.45% 43.30%</td><td>63.67% 63.19%</td></tr><tr><td>3B-INT4-AWQ 3B-INT4-GPTQ</td><td>81.27% 81.93%</td><td>56.58% 57.53%</td><td>58.80% 67.21%</td><td>56.51% 59.52%</td><td>63.29% 66.55%</td></tr><tr><td>Qwen2.5-0.5B 0.5B-INT4-AWQ</td><td>65.71% 57.56%</td><td>27.56% 27.77%</td><td>62.42% 53.34%</td><td>48.99% 48.67%</td><td>51.17% 46.83%</td></tr><tr><td>0.5B-INT4-GPTQ</td><td>69.26%</td><td>27.91%</td><td>62.17%</td><td>52.37%</td><td>52.93%</td></tr><tr><td>Qwen2.5-1.5B 1.5B-INT4-AWQ</td><td>77.52% 73.83%</td><td>40.89% 38.97%</td><td>69.38% 68.64%</td><td>71.82% 71.89%</td><td>64.90% 63.33%</td></tr><tr><td>1.5B-INT4-GPTQ</td><td>78.63%</td><td>40.38%</td><td>67.34%</td><td>69.07%</td><td>63.86%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-3B</td><td>80.13%</td><td>40.48%</td><td>68.79%</td><td>29.99%</td><td>54.85%</td></tr><tr><td>3B-INT4-AWQ</td><td>79.00%</td><td></td><td></td><td></td><td></td></tr><tr><td>3B-INT4-GPTQ</td><td></td><td>40.93%</td><td>68.49%</td><td>32.10%</td><td>55.13%</td></tr><tr><td></td><td>80.43%</td><td>44.98%</td><td>68.75%</td><td>29.82%</td><td>56.00%</td></tr><tr><td>Qwen2.5-7B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>83.78%</td><td>41.53%</td><td>67.05%</td><td>28.33%</td><td>55.17%</td></tr><tr><td>7B-INT4-AWQ</td><td>83.79%</td><td>39.02%</td><td>67.68%</td><td>28.71%</td><td>54.80%</td></tr><tr><td>7B-INT4-GPTQ</td><td>83.87%</td><td>40.15%</td><td>68.09%</td><td>29.18%</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>55.32%</td></tr></table>

TABLE II: Trustworthiness assessment for quantized LLMs.

## B. Pruning Influence

Pruning generally harms trustworthiness. We conduct comprehensive evaluations of both unstructured and semi-structured pruning on various large language models, particularly those at the 7B scale. The results, summarized in Table I, pruned LLMs show an obvious degradation in trustworthiness after pruning. For example, Gemma-1.1-7B and LLaMA-3.1-8B exhibit noticeable declines, with LLaMA-3.1-8B dropping by nearly 10% in robustness. These findings suggest that pruning substantially reduces model trustworthiness across multiple dimensions.

Semi-structured pruning increases degradation. As shown in Table I, applying semi-structured pruning further reduces trustworthiness compared to unstructured pruning. In particular, the performance under 2:4 sparsity is worse than under 4:8 sparsity, indicating that more restrictive sparsity patterns introduce greater degradation. This suggests that rigid pruning constraints can distort important representation subspaces, thereby amplifying the loss of trustworthiness across multiple dimensions. Considering that unstructured pruning offers little inference speedup and semi-structured pruning leads to low trustworthiness, we do not recommend using pruned models as trustworthy SLMs.

## C. Quantization Influence

Quantization has relatively minor influence, especially for larger models. As shown in Table II, quantization maintains the trustworthiness of Qwen2.5-7B with minimal degradation. We further extend our evaluation to 4-bit quantization on smaller

<table><tr><td>Model</td><td>Ethics</td><td>Privacy</td><td>Robustness</td><td>Fairness</td><td>Overall</td></tr><tr><td>Qwen2.5-1.5B</td><td>77.52%</td><td>40.89%</td><td>69.38%</td><td>71.82%</td><td>64.90%</td></tr><tr><td>1.5B-INT8-GPTQ</td><td>77.68%</td><td>41.08%</td><td>69.13%</td><td>71.85%</td><td>64.93%</td></tr><tr><td>1.5B-INT4-GPTQ</td><td>78.63%</td><td>40.38%</td><td>67.34%</td><td>69.07%</td><td>63.86%</td></tr><tr><td>1.5B-INT3-GPTQ</td><td>77.31%</td><td>40.72%</td><td>67.01%</td><td>68.23%</td><td>63.32%</td></tr><tr><td>Qwen2.5-3B</td><td>80.13%</td><td>40.48%</td><td>68.79%</td><td>29.99%</td><td>54.85%</td></tr><tr><td>3B-INT8-GPTQ</td><td>80.26%</td><td>40.35%</td><td>68.70%</td><td>29.54%</td><td>54.71%</td></tr><tr><td>3B-INT4-GPTQ</td><td>80.43%</td><td>44.98%</td><td>68.75%</td><td>29.82%</td><td>56.00%</td></tr><tr><td>3B-INT3-GPTQ</td><td>80.02%</td><td>41.32%</td><td>68.65%</td><td>29.13%</td><td>54.78%</td></tr></table>

TABLE III: Trustworthiness assessment under different quantized bits for Qwen2.5 models.

Qwen models and the LLaMA-3.2 series. According to Table II, the quantized models generally exhibit comparable performance to their full-precision counterparts, especially in the case of larger models. For instance, the Qwen2.5-0.5B model experiences a moderate 4% drop under AWQ quantization, whereas GPTQ yields more stable results, even improving the accuracy. For models larger than 1.5B, the decrease in trustworthiness remains minimal (less than 2%). These findings highlight quantization as a viable strategy for building trustworthy SLMs, with larger-scale models demonstrating greater resilience to low-bit compression.

GPTQ offers more reliable trustworthiness than AWQ. As presented in Table II, models quantized using GPTQ typically achieve higher accuracy in trustworthiness evaluations compared to those quantized with AWQ. This trend holds across most of the settings in our experiments. Interestingly, we observe that for certain models, such as LLaMA-3.2-3B and Qwen2.5-3B, GPTQ quantized variants even outperform their FP16 counterparts. These results further demonstrate that quantized models, particularly when using GPTQ, can not only reduce memory consumption but also maintain or even enhance trustworthiness, highlighting the potential of low-bit quantization in building efficient and reliable SLMs.

<table><tr><td>Model</td><td>Ethics</td><td>Privacy</td><td>Robustness</td><td>Fairness</td><td>Overall</td></tr><tr><td>h2o-danube3-500m-Chat</td><td>61.92%</td><td>29.02%</td><td>56.85%</td><td>43.27%</td><td>47.76%</td></tr><tr><td>MobiLlama-500m-Chat</td><td>55.79%</td><td>32.33%</td><td>48.02%</td><td>45.69%</td><td>45.46%</td></tr><tr><td>SmolLM2-360M-Instruct</td><td>51.47%</td><td>48.59%</td><td>60.16%</td><td>47.45%</td><td>51.92%</td></tr><tr><td>Qwen2.5-0.5B-Instruct</td><td>65.71%</td><td>27.56%</td><td>62.42%</td><td>48.99%</td><td>51.17%</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>77.52%</td><td>40.89%</td><td>69.38%</td><td>71.82%</td><td>64.90%</td></tr><tr><td>Qwen2.5-1.5B-INT4-AWQ</td><td>73.83%</td><td>38.97%</td><td>68.64%</td><td>71.89%</td><td>63.33%</td></tr><tr><td>Qwen2.5-1.5B-INT4-GPTQ</td><td>78.63%</td><td>40.38%</td><td>67.34%</td><td>69.07%</td><td>63.86%</td></tr><tr><td>Qwen2.5-1.5B-INT8-GPTQ</td><td>77.68%</td><td>41.08%</td><td>69.13%</td><td>71.85%</td><td>64.93%</td></tr></table>

TABLE IV: Trustworthiness comparison among pre-trained SLMs and compressed LLMs.
<table><tr><td>Model</td><td>Ethics</td><td>Privacy</td><td>Robustness</td><td>Fairness</td><td>Overall</td></tr><tr><td>Qwen2.5-7B</td><td>83.78%</td><td>41.53%</td><td>67.05%</td><td>28.33%</td><td>55.17%</td></tr><tr><td>Qwen2.5-3B</td><td>80.13%</td><td>40.48%</td><td>68.79%</td><td>29.99%</td><td>54.85%</td></tr><tr><td>KD-3B</td><td>82.12%</td><td>41.36%</td><td>69.88%</td><td>32.45%</td><td>56.45%</td></tr></table>

TABLE V: Exploration of distillation impact.

Quantization showcases robustness towards different compression ratios. We further analyze the effect of different quantization levels on trustworthiness. As shown in Table III, GPTQ exhibits stable performance across 3–8 bits on Qwen2.5, with no systematic degradation as the bit-width decreases. This trend suggests that low-bit quantization largely preserves the reliability-related behaviors learned by the full-precision model, and the trustworthiness scores are relatively insensitive to the compression ratio within this range. Overall, these results support our conclusion that quantization (especially GPTQ) is a robust and practical strategy for obtaining trustworthy SLMs under varying efficiency constraints.

## D. Directly Using SLMs or Compressing Larger LLMs?

Based on our analysis of pruning and quantization effects on LLMs, we focus on quantization as the primary technique to compare compressed LLMs against directly pre-trained SLMs. In this subsection, we evaluate several pre-trained SLMs and quantized versions of larger models. Specifically, we examine four pre-trained SLMs with fewer than 1B parameters to provide a comprehensive perspective.

Performance Analysis. As shown in Table IV, these pre-trained SLMs generally achieve lower trustworthiness scores, typically around 50In contrast, we apply 4-bit quantization to Qwen2.5- 1.5B, which can reduce weight memory by approximately 4× and potentially deliver up to 3× speedup with optimized kernels [8]. This makes the quantized 1.5B model comparable in practical deployment efficiency to smaller SLMs (e.g., 0.5B models), while retaining stronger representations. Consequently, the quantized Qwen2.5-1.5B model achieves substantially higher trustworthiness (around 63%) than the pre-trained SLMs. This gain is driven by the minimal degradation introduced by quantization and the stronger base capability of the 1.5B model compared to smaller counterparts (e.g., Qwen2.5-0.5B).

Memory and Latency Discussion. On RTX-4090 with batch=1 decoding, generation latency is often memory-bound. Although the 1.5B model has \~3× more parameters, INT4 quantization reduces weight memory traffic by \~4×, which can largely offset the increased parameter count in practice. As a result, Qwen2.5- 1.5B-INT4-GPTQ typically exhibits comparable ms/token to Qwen2.5-0.5B FP16 (often within \~1.2–1.5×), rather than being strictly slower, while achieving markedly better trustworthiness. Moreover, quantization offers a flexible knob to produce SLMs of different sizes and efficiency levels by adjusting bit-width, enabling smooth trade-offs between memory, speed, and trustworthiness.

From this analysis, we derive an important insight: instead of relying solely on small models trained from scratch, leveraging quantization on larger pre-trained LLMs can yield compact, efficient, and more trustworthy SLMs.

## E. Distillation Enhancement

We explore whether knowledge distillation can improve SLM trustworthiness. Specifically, Qwen2.5-3B is distilled from Qwen2.5-7B using the Alpaca dataset, with results listed in Table V. The distilled model shows improvements across all four trustworthiness categories, demonstrating the benefit of transferring knowledge from a more reliable model. We expect that distilling from larger and more trustworthy teacher models could yield even greater improvements in the reliability of small language models.

## V. CONCLUSION

This work explores effective strategies for developing trustworthy Small Language Models (SLMs) through a systematic empirical study. Our findings reveal three important observations: (1) Quantization offers a more reliable means than pruning for preserving model trustworthiness; (2) Compressing a well-aligned large model via quantization results in more robust and adaptable SLMs than directly using pre-trained small models; (3) Knowledge distillation serves as a complementary approach to further improve the reliability of compact models. Overall, our study sheds light on practical and scalable pathways to build efficient SLMs without compromising their trustworthiness, and lays the foundation for future efforts in this direction.

## REFERENCES

[1] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Juliette Love, et al., “Gemma: Open models based on gemini research and technology,” arXiv preprint arXiv:2403.08295, 2024.

[2] Qwen Team, “Qwen2.5: A party of foundation models,” September 2024.

[3] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[4] Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M Anwer, Michael Felsberg, Tim Baldwin, Eric P Xing, and Fahad Shahbaz Khan, “Mobillama: Towards accurate and lightweight fully transparent gpt,” arXiv preprint arXiv:2402.16840, 2024.

[5] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, et al., “Smollm2: When smol goes big–data-centric training of a small language model,” arXiv preprint arXiv:2502.02737, 2025.

[6] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh, “Gptq: Accurate post-training quantization for generative pre-trained transformers,” arXiv preprint arXiv:2210.17323, 2022.

[7] Elias Frantar and Dan Alistarh, “Sparsegpt: Massive language models can be accurately pruned in one-shot,” in International conference on machine learning. PMLR, 2023, pp. 10323–10337.

[8] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han, “Awq: Activation-aware weight quantization for llm compression and acceleration,” arXiv preprint arXiv:2306.00978, 2023.

[9] Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei, “Duquant: Distributing outliers via dual transformation makes stronger quantized llms,” Advances in Neural Information Processing Systems, vol. 37, pp. 87766–87800, 2024.

[10] Haokun Lin, Xinle Jia, Shaozhen Liu, Shujun Xia, Weitao Huang, Haobo Xu, Junyang Li, Yicheng Xiao, Xingrun Xing, Ziyu Guo, et al., “Efficient diffusion language models: A comprehensive survey,” Authorea Preprints, 2026.

[11] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al., “A survey on evaluation of large language models,” ACM transactions on intelligent systems and technology, vol. 15, no. 3, pp. 1–45, 2024.

[12] Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, et al., “Trustllm: Trustworthiness in large language models,” arXiv preprint arXiv:2401.05561, 2024.

[13] Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, et al., “Promptbench: Towards evaluating the robustness of large language models on adversarial prompts,” arXiv e-prints, pp. arXiv–2306, 2023.

[14] Badhan Chandra Das, M Hadi Amini, and Yanzhao Wu, “Security and privacy challenges of large language models: A survey,” ACM Computing Surveys, vol. 57, no. 6, pp. 1–39, 2025.

[15] Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, and Martin Vechev, “Exploiting llm quantization,” Advances in Neural Information Processing Systems, vol. 37, pp. 41709–41732, 2024.

[16] Junyuan Hong, Jinhao Duan, Chenhui Zhang, Zhangheng Li, Chulin Xie, Kelsey Lieberman, James Diffenderfer, Brian Bartoldson, Ajay Jaiswal, Kaidi Xu, et al., “Decoding compressed trust: Scrutinizing the trustworthiness of efficient llms under compression,” arXiv preprint arXiv:2403.15447, 2024.

[17] Kejia Chen, Jiawen Zhang, Jiacong Hu, Yu Wang, Jian Lou, Zunlei Feng, and Mingli Song, “Assessing safety risks and quantization-aware safety patching for quantized large language models,” in Forty-second International Conference on Machine Learning.

[18] Song Han, Huizi Mao, and William J Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,” arXiv preprint arXiv:1510.00149, 2015.

[19] Xingrun Xing, Zheng Liu, Shitao Xiao, Boyan Gao, Yiming Liang, Wanpeng Zhang, Haokun Lin, Guoqi Li, and Jiajun Zhang, “Efficientllm: Scalable pruning-aware pretraining for architecture-agnostic edge language models,” arXiv preprint arXiv:2502.06663, 2025.

[20] Haobo Xu, Sirui Chen, Ruizhong Qiu, Yuchen Yan, Chen Luo, Monica Cheng, Jingrui He, and Hanghang Tong, “Prune as you generate: Online rollout pruning for faster and better rlvr,” arXiv preprint arXiv:2603.24840, 2026.

[21] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter, “A simple and effective pruning approach for large language models,” arXiv preprint arXiv:2306.11695, 2023.

[22] Xinyin Ma, Gongfan Fang, and Xinchao Wang, “Llm-pruner: On the structural pruning of large language models,” Advances in neural information processing systems, vol. 36, pp. 21702–21720, 2023.

[23] Lianwei Yang, Haisong Gong, Haokun Lin, Yichen Wu, Zhenan Sun, and Qingyi Gu, “Dopq-vit: Towards distribution-friendly and outlier aware post-training quantization for vision transformers,” arXiv preprint arXiv:2408.03291, 2024.

[24] Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, and Zhenan Sun, “Quantization meets dllms: A systematic study of post-training quantization for diffusion llms,” arXiv preprint arXiv:2508.14896, 2025.

[25] Lianwei Yang, Haokun Lin, Tianchen Zhao, Yichen Wu, Hongyu Zhu, Ruiqi Xie, Zhenan Sun, Yu Wang, and Qingyi Gu, “Lrq-dit: Log-rotation post-training quantization of diffusion transformers for image and video generation,” arXiv preprint arXiv:2508.03485, 2025.

[26] Jingxuan Zhang, Yunta Hsieh, Zhongwei Wang, Haokun Lin, Xin Wang, Ziqi Wang, Yingtie Lei, and Mi Zhang, “Quantvla: Scale-calibrated post-training quantization for vision-language-action models,” arXiv preprint arXiv:2602.20309, 2026.

[27] Haokun Lin, Xinle Jia, Haobo Xu, Bingchen Yao, Xianglong Guo, Yichen Wu, Zhichao Lu, Ying Wei, Qingfu Zhang, and Zhenan Sun, “Duquant++: Fine-grained rotation enhances microscaling fp4 quantization,” arXiv preprint arXiv:2604.17789, 2026.

[28] Lianwei Yang, Haokun Lin, Yichen Wu, Zhenan Sun, and Qingyi Gu, “Dapq-dit: Distribution-aware post-training quantization for efficient generative tasks in diffusion transformers,” in Proceedings of the 2026 International Conference on Multimedia Retrieval, 2026, pp. 2371–2380.

[29] Lianwei Yang, Haokun Lin, Yichen Wu, Caifeng Shan, Zhenan Sun, and Qingyi Gu, “Reshape and rotate: Adaptive weight reshaping and fine-grained rotation for ultra-low-bit diffusion transformers quantization,” Neurocomputing, p. 133830, 2026.

[30] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[31] Haobo Xu, Yuchen Yan, Dingsu Wang, Zhe Xu, Zhichen Zeng, Tarek F Abdelzaher, Jiawei Han, and Hanghang Tong, “Slog: An inductive spectral graph neural network beyond polynomial filter,” in Forty-first International Conference on Machine Learning, 2024.

[32] Yue Jiang, Haokun Lin, Yang Bai, Bo Peng, Zhili Liu, Yueming Lyu, Yong Yang, and Jing Dong, “Image-level memorization detection via inversion-based inference perturbation,” in International Conference on Learning Representations, 2025, vol. 2025, pp. 47960–47979.

[33] Shujun Xia, Haokun Lin, Yichen Wu, Yinan Zhou, Zixuan Li, Zhongwei Wan, Xingrun Xing, Yefeng Zheng, Xiang Li, Caifeng Shan, et al., “Medrek: Retrieval-based editing for medical llms with key-aware prompts,” arXiv preprint arXiv:2510.13500, 2025.

[34] Zixuan Li, Haokun Lin, Yicheng Xiao, Zhiwei Li, Xinyang Song, Zelong Zheng, Yong He, Heng Yao, Ke Ding, Chao Yu, et al., “Iv-cot: Implicit visual chain-of-thought for structure-aware text-to-image generation,” arXiv preprint arXiv:2606.24849, 2026.

[35] Jinqian Yang, Yichen Wu, Wanhua Li, Haokun Lin, Renzhen Wang, Xiangchu Feng, and Xixi Jia, “Mac-splat: Multi-attribute consistency for high-fidelity sparse-view reconstruction,” arXiv preprint arXiv:2607.10792, 2026.

[36] Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Can Xu, Dacheng Tao, and Tianyi Zhou, “A survey on knowledge distillation of large language models,” arXiv preprint arXiv:2402.13116, 2024.

[37] Pascal Pfeiffer, Philipp Singer, Yauhen Babakhin, Gabor Fodor, Nischay Dhankhar, and Sri Satish Ambati, “H2o-danube3 technical report,” arXiv preprint arXiv:2407.09276, 2024.

[38] Rishi Bommasani, “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021.

[39] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt, “Aligning ai with shared human values,” arXiv preprint arXiv:2008.02275, 2020.

[40] Maxwell Forbes, Jena D Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi, “Social chemistry 101: Learning to reason about social and moral norms,” arXiv preprint arXiv:2011.00620, 2020.

[41] Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, and Bo Li, “Adversarial glue: A multi task benchmark for robustness evaluation of language models,” arXiv preprint arXiv:2111.02840, 2021.