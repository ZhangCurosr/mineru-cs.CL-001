# Gradient Mirage: Trainable yet Label-Unidentifiable Gradients in Large Language Model Split Learning

Shiyu Miao<sup>1</sup>, Yunlong Mao<sup>1</sup>\*, Zirui Huang<sup>1</sup>, Liang Yao<sup>2</sup>, Tianshuo Zheng<sup>3</sup>, Yanhui Gu<sup>4</sup>, Fan Liu<sup>2</sup>, Sheng Zhong<sup>1</sup>,

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University <sup>2</sup>College of Computer Science and Software Engineering, Hohai University <sup>3</sup>School of Mathematics, Nanjing University <sup>4</sup> Nanjing Normal University maoyl@nju.edu.cn, zhongsheng@nju.edu.cn

## Abstract

Gradient matching attacks (GMAs) in LLM split learning (SL) rely on a critical yet underexplored assumption: the gradient exposed at the split interface is a faithful derivative of the client’s full-label training objective. This gradient–objective consistency allows a curious server to recover private labels by searching for a sequence whose induced gradient explains the observation. We propose Gradient Mirage, a defense that breaks this consistency without discarding the optimization utility of the backward signal. Our key idea is to induce the adversary to solve a misspecified inverse problem, in which no plausible label sequence in the sequence space can explain the observed gradients. Concretely, Gradient Mirage achieves this by inducing inconsistency across three dimensions: objective, direction, and scale. Selective Autoregressive Supervision derives the exposed gradient from a masked surrogate loss rather than the full-label objective assumed by the attacker; Scale Blinding then applies randomized multiplicative rescaling, obscuring the gradient’s natural magnitude; and Directional Privatization further randomizes the gradient direction while preserving its magnitude through the von Mises–Fisher (vMF) mechanism under a directional metric differential privacy guarantee. Crucially, utility is preserved: the Top segment still learns from all target tokens via Dual-Track Backpropagation, the exposed gradient remains informative since each supervised token retains its complete autoregressive context, and Bottom-Gradient Recovery restores the effective gradient for Bottom-segment optimization. Extensive experiments show that Gradient Mirage provides substantially stronger protection than existing defenses under comparable fine-tuning performance, achieving a better privacy-utility trade-off.

Code — https://github.com/StevenMsy/GMA-SL

## 1 Introduction

Machine Learning-as-a-Service (MLaaS) has become a dominant paradigm for deploying large-scale models (Cai et al. 2024; Lin et al. 2026), allowing data owners to outsource computation and reduce infrastructure overhead. In its training setting, commonly known as Training-asa-Service (TaaS), Split Learning (SL) (Gupta and Raskar

![](images/f3aaaf9245f7fb5c7cf3b618cde3b6f3064e73bb369c5a830e6d1ad0524cfb70.jpg)  
Figure 1: Illustration of label leakage in label-shielded split learning. Given the gradient returned from the client’s Top segment to the Trunk segment, a curious-but-honest server can recover the label sequence by optimization.

2018; Vepakomma et al. 2018; Poirot et al. 2019; Chen et al. 2024; Shen et al. 2025; Zhang et al. 2026) mitigates direct data exposure by partitioning model training between the client and server. We focus on LLM supervised finetuning under Label-Shielded Split Learning (Pasquini, Ateniese, and Bernaschi 2021), where the compute-intensive Trunk segment is outsourced to the server, while the Bottom and Top segments remain on the client to keep private inputs, labels, and supervised loss local.

However, prior research (Chen et al. 2024) shows that this LLM-SL system fails to provide effective privacy protection against a curious-but-honest server. In particular, the client’s private data becomes highly vulnerable to GMAs (Zhu, Liu, and Han 2019; Zhao, Mopuri, and Bilen 2020; Deng et al. 2021; Balunovic et al. 2022; Gupta et al. 2022) at the split interface between the Trunk and Top segment. As illustrated in Fig. 1, the adversary’s guiding question is essentially: ”What label sequence would yield such gradients?” By optimizing for a label sequence whose interface gradients match the observed ones, it can recover sensitive training content even without explicit access to labels.

We term this attack, launched at the split interface between the Trunk and Top segment, GMA-SL. GMA-SL constitutes a particularly compelling variant of classical GMA: We delineate its relationship to, and key distinctions from classical GMA in Section 2.2, and provide the concrete attack instantiation and implementation details in Section 2.3.

GMA-SL poses a particularly severe gradient leakage threat in LLM-SL. Unlike classical GMA, it matches activation gradients at the split interface, which retain explicit batch structure and are closely tied to local supervision. Moreover, because the cut-layer representation is already available, the adversary primarily optimizes the label sequence rather than jointly reconstructing inputs and labels, making the attack both simpler and harder to defend against.

Motivated by this challenge, we propose Gradient Mirage, a defense method tailored to GMA-SL. Its key insight is that what the model learns and what the exposed gradient reveals need not coincide. Gradient Mirage deliberately decouples these two objectives. Under the constraint of minimally affecting optimization, we seek an exposed signal residing in a “gray zone” of the gradient space: to the server-side optimizer, it remains a serviceable backward signal for trunk updates; to the adversary, it looks like a gradient, yet no suitable label sequence in the sequence space can explain it under the attacker’s assumed objective. Consequently, an adversary performing gradient matching against this signal is in effect solving a misspecified inverse problem—one whose solution set is not anchored to the true labels.

Gradient Mirage realizes this design by inducing inconsistency across three dimensions. Dual-Track Backpropagation confines the full-label loss $\mathcal { L } _ { \mathrm { F } }$ to local learning at the Top segment, while the transmitted gradient is generated by a masked surrogate loss ${ \mathcal { L } } _ { \mathrm { M } }$ via Selective Autoregressive Supervision, yielding Objective Inconsistency $( { \mathcal { L } } _ { \mathrm { F } } \neq { \mathcal { L } } _ { \mathrm { M } } ) { : }$ making the attacker’s assumed objective misspecified at its root. Directional Privatization then randomizes the gradient direction via vMF mechanism (Weggenmann and Kerschbaum $2 0 2 1 ;$ von Mises 1918; Fisher 1953) with a directional metric differential privacy guarantee, inducing Directional Inconsistency $\begin{array} { r l r } { ( \frac { \bar { g _ { \mathrm { r e l e a s e } } } } { \| g _ { \mathrm { r e l e a s e } } \| } ) } & { { } \ne } & { \frac { \nabla _ { x } \mathcal { L } _ { \mathrm { M } } } { \| \nabla _ { x } \mathcal { L } _ { \mathrm { M } } \| } ) } \end{array}$ . Scale Blinding applies randomized multiplicative rescaling, introducing Scale Inconsistency $( \| g _ { \mathrm { r e l e a s e } } \| \neq \| \nabla _ { x } \mathcal { L } _ { \mathrm { M } } \| )$ : even when the direction is approximately matched, the gradient’s natural magnitude is no longer trustworthy. Meanwhile, optimization compatibility is supported by design: the Top segment still learns under full supervision, the vMF mechanism retains the gradient magnitude with a provable lower bound on the amplitude signal-to-noise ratio, and Bottom-Gradient Recovery restores the effective gradient for bottom optimization. Gradient Mirage thereby disrupts the labelidentifiability of the exposed gradient while retaining stable and effective SL.

We present, to the best of our knowledge, the first defense shown to be effective against GMA-SL, a particularly severe gradient leakage threat in LLM-SL. As GMA-SL constitutes a specialized form of GMA in the SL setting, we adapt representative GMA defenses and a stringent sequence-level differential privacy (DP) mechanism as practical baselines, demonstrating that our method consistently achieves superior privacy-utility trade-offs.

The contributions are summarized as follows: (i) We identify gradient–objective consistency as the key vulnerability underlying GMA-SL and provide a systematic characterization of this attack in label-shielded LLM-SL, clarifying its attack surface and key distinctions from classical GMA. (ii) We propose Gradient Mirage, a defense that decouples what the model learns from what the gradient reveals, injecting objective, directional, and scale inconsistencies into the released gradient while preserving optimization utility. (iii) We conduct extensive experiments across multiple LLMs and datasets, adapting representative GMA defenses and a rigorous DP method as baselines. Gradient Mirage consistently achieves a superior privacy-utility trade-off.

![](images/2e2bc45bb062baac6d9b9781c6b510f8f29cf2d6658ccee52dad7aad081f3c58.jpg)  
Figure 2: Two classical split learning paradigms: labelexposed split learning and label-shielded split learning.

## 2 Preliminary

## 2.1 Split Learning

SL partitions a neural network across clients and a central server, so that clients hold only a small portion of the model while the majority of parameters—and thus the computational burden—reside on the server. As illustrated in Fig. 2, typical SL can be categorized into two paradigms: (I) Label-Exposed $\mathbf { S L } ,$ , where the network is split into two parts and clients disclose labels to the server; (II) Label-Shielded $\mathbf { S L } ,$ where labels remain on-device and the network is partitioned into three segments, with the front and tail placed on the client and the middle trained on the server.

In LLM supervised fine-tuning, next-token labels are obtained by shifting the input sequence by one position, so exposing labels would effectively reveal the private text. We therefore consider Label-Shielded SL. Given a client’s private dataset $D _ { \mathrm { p r i } } ,$ the overall network $f _ { \theta }$ is partitioned into a Bottom $f _ { \mathrm { { b t m } } } ,$ a Trunk $f _ { \mathrm { t r k } } ,$ , and a Top $f _ { \mathrm { t o p } }$ . The server hosts $f _ { \mathrm { t r k } }$ , while the client retains $f _ { \mathrm { { b t m } } }$ and $f _ { \mathrm { t o p } }$ . For a mini-batch $x \in D _ { \mathrm { p r i } }$ , the client transmits $x _ { \mathrm { { b t m } } } = f _ { \mathrm { { b t m } } } ( x )$ to the server, and receives $x _ { \mathrm { t r k } } = f _ { \mathrm { t r k } } ( x _ { \mathrm { b t m } } )$ from the server. The client then obtains the prediction $\hat { y } ~ = ~ f _ { \mathrm { t o p } } ( x _ { \mathrm { t r k } } )$ and computes the training objective $\mathcal { L } ( \hat { y } , y )$ with label y kept local. During backpropagation, the exchanged gradients $\partial \mathcal { L } / \partial x _ { \mathrm { t r k } }$ allow the server to update the Trunk, while the client updates the Bottom and Top using ${ \partial \mathcal { L } } / { \partial x _ { \mathrm { b t m } } }$ and $\partial \mathcal { L } / \partial x _ { \mathrm { t o p } }$

## 2.2 Classical Gradient Matching Attack

Classical GMA arises in distributed training, where multiple participants collaboratively optimize a shared model while keeping their data local. During training, they exchange model updates—typically gradients—rather than raw data. This gradient exchange creates the primary attack surface exploited by gradient-matching methods. $\mathbf { A } s$ illustrated in Fig. 3, classical GMA assumes access to the model snapshot ${ \theta } ^ { ( \overline { { t } } ) }$ used to generate the observed gradient; reconstruction is then performed with respect to this fixed snapshot.

![](images/b4cde5e1eca2be5819d1c18eac37ee80cfcc9124252a656adae79745f98d0624.jpg)  
Figure 3: Classical gradient-matching attack with a frozen parameter snapshot. At training step t, the adversary fixes the model parameters $\pmb \theta ^ { ( t ) }$ (the snapshot) and optimizes dummy inputs $x _ { d u m }$ and labels $y _ { d u m }$

Let $\bar { F } ( \cdot ; \theta )$ be the model and $\mathcal L ( \cdot , \cdot )$ the training loss. At step t, the shared gradient is

$$
\mathbf { g } ^ { ( t ) } = \nabla _ { \pmb { \theta } } \mathcal { L } \Big ( F \Big ( \boldsymbol { x } ^ { ( t ) } ; \pmb { \theta } ^ { ( t ) } \Big ) , \boldsymbol { y } ^ { ( t ) } \Big ) ,\tag{1}
$$

where $\pmb \theta ^ { ( t ) }$ denotes the model-parameter snapshot at step t, and $( x ^ { ( t ) } , y ^ { ( t ) } )$ is the private minibatch that generated $\mathbf { g } ^ { ( t ) }$ The adversary introduces dummy variables $\left( x _ { d u m } , y _ { d u m } \right)$ and solves:

$$
( x ^ { * } , y ^ { * } ) \ = \ \arg \operatorname* { m i n } _ { \tilde { x } , \tilde { y } } \ D \Big ( \nabla _ { \theta } \mathcal { L } \Big ( F ( \tilde { x } ; \theta ^ { ( t ) } ) , \tilde { y } \Big ) , \mathbf { g } ^ { ( t ) } \Big ) ,\tag{2}
$$

where $D ( \cdot , \cdot )$ is a gradient-discrepancy measure (e.g., $D ( \mathbf { a } , \mathbf { b } ) = \| \mathbf { a } - \mathbf { b } \| _ { 2 } ^ { 2 }$ , cosine distance, or a weighted combination). The optimization is carried out by gradient descent on the dummy variables:

$$
\begin{array} { r } { ( \tilde { x } , \tilde { y } ) \gets ( \tilde { x } , \tilde { y } ) - \eta \nabla _ { ( \tilde { x } , \tilde { y } ) } D \Big ( \nabla _ { \theta } \mathcal { L } \Big ( F ( \tilde { x } ; \theta ^ { ( t ) } ) , \tilde { y } \Big ) , \mathbf { g } ^ { ( t ) } \Big ) , \mathbf { } } \end{array}\tag{3}
$$

while keeping $\pmb \theta ^ { ( t ) }$ unchanged throughout the optimization.

## 2.3 Gradient Matching Attack in SL

The GMA in the SL setting can be viewed as a variant of the classical GMA. Unlike classical GMA, matching is performed in the activation space at the split interface rather than directly on $\nabla _ { \theta }$ . The schematic overview of the entire pipeline is shown in Fig. 4.

Let the Top segment be $F ( \cdot ; \theta _ { t o p } )$ and the top proxy held by the adversary $\bar { \boldsymbol { F } } ( \cdot ; \theta _ { p r o } )$ . Given label y, the client locally computes prediction $\widehat { y } ^ { } = F ( x _ { t r k } ; \theta _ { t o p } )$ and loss $\mathcal { L } ( \hat { \mathbf { y } } , \mathbf { y } )$ The gradient propagated back to the cut layer is given by:

$$
\mathbf { g } _ { \mathrm { t r k } } ~ = ~ \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } ( F ( x _ { \mathrm { t r k } } ; \pmb { \theta } _ { \mathrm { t o p } } ) , y ) .\tag{4}
$$

The attacker is assumed to observe $\mathbf { g } _ { \mathrm { t r k } }$ . The goal is to recover a plausible label y˜ that yields a matching gradient signal. The attacker uses a differentiable proxy Top segment $\breve { F ( \cdot ; \theta _ { \mathrm { p r o } } ) }$ and introduces dummy variables $\tilde { \mathbf { x } } _ { \mathrm { t r k } }$ and $\tilde { \mathbf { y } } .$ . The proxy-induced gradient is

$$
\tilde { \mathbf { g } } _ { \mathrm { t r k } } ( \tilde { y } ) = \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } ( F ( x _ { \mathrm { t r k } } ; \theta _ { \mathrm { p r o } } ) , \tilde { y } ) ,\tag{5}
$$

The attack solves the following matching problem:

$$
\tilde { y } ^ { * } = \arg \operatorname* { m i n } _ { \tilde { y } } D ( \mathbf { g } _ { \mathrm { t r k } } , \tilde { \mathbf { g } } _ { \mathrm { t r k } } ( x _ { \mathrm { t r k } } , \tilde { y } ) ) ,\tag{6}
$$

where $D ( \cdot , \cdot )$ is a discrepancy measure.

The optimization is carried out by gradient descent on the dummy variables:

$$
\tilde { y } \gets \tilde { y } - \eta \nabla _ { \tilde { y } } D \big ( \mathbf { g } _ { \mathrm { t r k } } , \tilde { \mathbf { g } } _ { \mathrm { t r k } } \big ( x _ { \mathrm { t r k } } , \tilde { y } \big ) \big ) ,\tag{7}
$$

while $\theta _ { \mathrm { p r o } }$ is held fixed during the inner-loop optimization. Compared with classical GMA, GMA-SL yields the following advantages. (i) Batch-separable matching. In SL, the observed gradient $\mathbf { \sigma } _ { \nabla _ { x _ { \mathrm { t r k } } } } \mathcal { L } \in \mathbf { \sigma } \mathbb { R } ^ { B \times L \times D }$ retains explicit batch structure, allowing samples to be matched independently except for the scalar coupling introduced by batchaveraged loss. The only coupling arises from the scalar coefficient induced by averaging the loss over the batch (discussed in Section 2.4). This behavior contrasts with classical GMA, where the attacker matches gradients with respect to model parameters $( \mathrm { e . g . , \nabla _ { W } } )$ , which implicitly aggregates information across the batch and can entangle samples. (ii) A simpler optimization objective. With $x _ { \mathrm { t r k } }$ fixed, the attacker mainly needs to optimize a dummy label y˜. Compared to classical GMA-where both the input x˜ and label y˜ are typically optimized-this reduces the number of free variables and mitigates optimization interference. At the same time, SL-based GMA also introduces new challenges. (i) Limited gradient observability. The attacker can only match gradients defined at the split interface (i.e., gradients with respect to activations), rather than the full gradient over all model parameters. In classical GMA, gradients from deeper layers can sometimes be weighted or exploited to improve reconstruction, whereas GMA-SL is restricted to $\nabla _ { \mathbf x _ { \mathrm { t r k } } } .$ (ii) Semi-white-box scenario limitation. The attacker does not have access to an exact snapshot of the victim’s top model parameters, making the setting non–white-box. Instead, we consider the semi–white-box assumption (Chen et al. 2024) in which the attacker only knows the initialization (pretrained weight) used for LLM fine-tuning.

## 2.4 Loss in Autoregressive Decoder Structure

In this section, we formalize the autoregressive training loss used throughout the paper and clarify how the loss is computed by the client and the adversary under our threat model introduced in section 3.

For a mini-batch of size B and sequence length $L ,$ let $\textbf { H } \in \mathbb { R } ^ { B \times L \times D }$ denote the input feature tensor (tokenwise hidden states) at the input of the language-modeling head, and let $\mathbf { M } \in \left\{ 0 , 1 \right\} ^ { B \times \hat { L } }$ be the attention mask, where $\mathbf { M } _ { b , \ell } = 1$ indicates that position ℓ in example b is a valid token contributing to training, and $\mathbf { M } _ { b , \ell } = 0$ indicates padding or an ignored position. The model produces next-token logits

![](images/e14a8bdf6ff8ffd45bf175940d7f0ccc576cee705bacdf3ca6428127e2b2aaee.jpg)  
Figure 4: Gradient matching attack in split learning (GMA-SL). The adversary observes the gradient with respect to the cutlayer representation $\nabla x _ { t r k }$ , produced by the victim’s Top segment, then optimizes a dummy label $y _ { d u m }$ using a differentiable proxy top model $F ( \cdot ; \pmb \theta _ { \mathrm { p r o } } )$ , such that the proxy-induced gradient $\nabla _ { x _ { \mathrm { t r k } } } \dot { \mathcal { L } } \big ( F ( x _ { \mathrm { t r k } } ; \pmb \theta _ { \mathrm { p r o } } ) , \dot { \tilde { y } } \big )$ matches the observed one. The schematic computation graph is shown on the right side.

$$
\mathbf { L } = f _ { \mathrm { L M } } ( \mathbf { H } ; \pmb { \theta } ) \in \mathbb { R } ^ { B \times L \times V } ,\tag{8}
$$

where V is the vocabulary size and θ are model parameters. Let $\mathbf { Y } \in \{ 1 , \dots , V \} ^ { \bar { B } \times L }$ be the token id matrix for the same sequences. With the standard one-step shift, the training target for position ℓ is $\mathbf { Y } _ { b , \ell + 1 }$ . Define the per-token cross-entropy

$$
\ell _ { b , \ell } ( \pmb \theta ) = - \log \mathrm { S o f t m a x } ( \mathbf { L } _ { b , \ell , : } ) _ { \mathbf { Y } _ { b , \ell + 1 } } .\tag{9}
$$

We mask out invalid positions and (optionally) the final position, yielding the masked autoregressive loss

$$
\mathcal { L } _ { \mathrm { A R } } ( \pmb { \theta } ) = \frac { 1 } { \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } } \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } \ell _ { b , \ell } ( \pmb { \theta } ) .\tag{10}
$$

For the clients, they perform the computation in Eq. 10 locally at the end of the Top segment. For the adversary, they execute:

$$
\begin{array} { c } { \tilde { \mathbf { Z } } = F ( x _ { \mathrm { t r k } } ; \theta _ { \mathrm { p r o } } ) \in \mathbb { R } ^ { B \times L \times V } , } \\ { \ell _ { b , \ell } ^ { \sim } ( \pmb { \theta } ) = - \log \mathrm { S o f t m a x } \Big ( \tilde { \mathbf { L } } _ { b , \ell , : } \Big ) _ { \tilde { \mathbf { Y } } _ { b , \ell + 1 } } , } \\ { \mathcal { L } _ { \mathrm { A R } } ( \pmb { \theta } ) = \frac { 1 } { \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } } \displaystyle \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } \ell _ { b , \ell } ^ { \sim } ( \pmb { \theta } ) . } \end{array}\tag{11}
$$

The adversary fully knows the normalization factor $N =$ $\begin{array} { r } { \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { \bar { M } } _ { b , \ell + 1 } } \end{array}$ . This is because, in addition to the cutlayer representation $\mathbf { x } _ { \mathrm { b t m } } .$ , the attention mask M must also be transmitted to the server: the server-side trunk requires M to construct attention score matrices and to carry out the forward pass. Since this information is a functional prerequisite for executing the protocol, even the weakest adversarial assumptions in the LLM-SL setting typically grant the server access to M, and hence to N.

## 2.5 Batch-Separable Matching

In this section, we elaborate on advantage (i) of the adversary in GMA-SL, as introduced in Section 2.3, namely why samples within the same mini-batch do not interfere with one another during gradient matching. We begin with the computational details of gradient matching in SL.

Let B be the batch size, L the sequence length, and V the vocabulary size. The server observes (i) the smashed data $x _ { \mathrm { t r k } } \in \mathbf { \bar { \Gamma } } \mathbb { R } ^ { B \times L \times D }$ , (ii) the attention mask M ∈ $\{ 0 , 1 \} ^ { \bar { B } \times \bar { L } }$ , and (iii) the back-propagated gradient w.r.t. the smashed data $\mathbf { G } ^ { ' } = \mathbf { g } _ { \mathrm { t r k } } \in \dot { \mathbb { R } } ^ { B ^ { \times } L \times D }$ , where $\begin{array} { r l } { \mathbf { g } _ { \mathrm { t r k } } } & { { } = } \end{array}$ $\nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } _ { \mathrm { A R } } ( x _ { \mathrm { t r k } } ; \theta _ { \mathrm { t o p } } )$ . The attacker optimizes a dummy soft label tensor $\tilde { \textbf { Z } } \in \mathbb { R } ^ { B \times L \times V }$ (logits in label space), which is transformed into a distribution via temperaturescaled softmax:

$$
\mathbf { P } \ = \ \mathrm { S o f t m a x } \Big ( \tilde { \mathbf { Z } } / \tau \Big ) \in [ 0 , 1 ] ^ { B \times L \times V } ,\tag{12}
$$

where $\tau > 0$ is a temperature hyperparameter. For the nexttoken soft target distribution, the per-token cross-entropy is

$$
\ell _ { b , \ell } = \mathrm { \bf ~ - \sum _ { v = 1 } ^ { V } } { \bf P } _ { b , \ell + 1 , v } \log \mathrm { S o f t m a x } \Big ( \tilde { \bf L } _ { b , \ell , : } \Big ) _ { v } .\tag{13}
$$

Masking is applied to ignore padding positions (and effectively excludes the final token by using the ℓ + 1 target):

$$
\mathcal { L } _ { \mathrm { s o f t } } ( x _ { \mathrm { t r k } } , \mathbf { \tilde { Z } } ; \theta _ { \mathrm { p r o } } ) = \frac { 1 } { N } \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } { \bf M } _ { b , \ell + 1 } \ell _ { b , \ell } ,\tag{14}
$$

where $\begin{array} { r } { N = \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } } \end{array}$ . From $\mathcal { L } _ { \mathrm { { s o f t } } }$ , the attacker computes the gradient with respect to the observed smashed data $x _ { \mathrm { t r k } } \mathbf { \cdot }$

$$
\begin{array} { r } { \tilde { \bf g } _ { \mathrm { t r k } } ( \tilde { \bf Z } ) = \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } _ { \mathrm { s o f t } } ( x _ { \mathrm { t r k } } , \tilde { \bf Z } ; \theta _ { \mathrm { p r o } } ) \in \mathbb { R } ^ { B \times L \times D } . } \end{array}\tag{15}
$$

The attack then minimizes a discrepancy between the induced gradient $\tilde { \bf G } ( \tilde { \bf Z } )$ and the observed gradient G, where $\widetilde { \bf G } ( \widetilde { \bf Z } ) \ \triangleq \ \widetilde { \bf g } _ { \mathrm { t r k } } ( \widetilde { \bf Z } )$ is exactly the gradient defined in $\mathbf { E q } .$

(14); we adopt the uppercase notation hereafter to emphasize its role as a tensor being matched against G. Taking TAG as an illustrative example, we define the gradientmatching objective as

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { T A G } } ( \tilde { \mathbf { Z } } ) = \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L } \mathbf { M } _ { b , \ell } \Big ( \beta \big \| \tilde { \mathbf { G } } _ { b , \ell , : } ( \tilde { \mathbf { Z } } ) - \mathbf { G } _ { b , \ell , : } \big \| _ { 2 } ^ { 2 } } } \\ & { } & { + ( 1 - \beta ) \big \| \tilde { \mathbf { G } } _ { b , \ell , : } ( \tilde { \mathbf { Z } } ) - \mathbf { G } _ { b , \ell , : } \big \| _ { 1 } \Big ) , } \end{array}\tag{16}
$$

where $\beta \in [ 0 , 1 ]$ controls the trade-off between the $\ell _ { 2 }$ and $\ell _ { 1 }$ discrepancies. The key optimization variable is $\tilde { \mathbf { Z } } ,$ but the matching loss depends on $\tilde { \mathbf { Z } }$ only through the intermediate gradient $\mathbf G ( \tilde { \mathbf Z } ) = \nabla _ { x _ { \mathrm { t r k } } } \mathcal L _ { \mathrm { s o f t } }$ . Therefore, updating $\tilde { \mathbf { Z } }$ requires differentiating through a gradient, i.e., computing a mixed second-order derivative:

$$
\begin{array} { r l } & { \nabla _ { \tilde { \mathbf { Z } } } \mathcal { L } _ { \mathrm { T A G } } ( \tilde { \mathbf { Z } } ) = \frac { \partial \mathcal { L } _ { \mathrm { T A G } } } { \partial \tilde { \mathbf { G } } } \cdot \underbrace { \frac { \partial \tilde { \mathbf { G } } } { \partial \tilde { \mathbf { Z } } } } _ { \mathrm { r e q u i r e s 2 n d o r t e r m s } } } \\ & { ~ = \frac { \partial \mathcal { L } _ { \mathrm { T A G } } } { \partial \tilde { \mathbf { G } } } \cdot \frac { \partial } { \partial \tilde { \mathbf { Z } } } \Big ( \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } _ { \mathrm { s o f t } } ( x _ { \mathrm { t r k } } , \tilde { \mathbf { Z } } ; \theta _ { \mathrm { p r o } } ) \Big ) } \\ & { ~ = \frac { \partial \mathcal { L } _ { \mathrm { T A G } } } { \partial \tilde { \mathbf { G } } } \cdot \nabla _ { \tilde { \mathbf { Z } } , x _ { \mathrm { t r k } } } ^ { 2 } \mathcal { L } _ { \mathrm { s o f t } } ( x _ { \mathrm { t r k } } , \tilde { \mathbf { Z } } ; \theta _ { \mathrm { p r o } } ) . } \end{array}\tag{17}
$$

In the implementation, we obtain $\mathbf { G } ( \tilde { \mathbf { Z } } )$ via automatic differentiation while retaining the computational graph $( \mathrm { c r e a t e { \mathrm { - } } g r a p h { \mathrm { = } } T r u e } ) ^ { 1 }$ , which enables backpropagation from $\mathcal { L } _ { \mathrm { T A G } }$ to $\tilde { \mathbf { Z } } .$

Finally, the adversary decodes discrete token predictions by $\begin{array} { r l } { \tilde { \mathrm { Y } } _ { b , \ell _ { - } } = } & { { } \underset { \mathrm { ~ \tiny ~ \cdot ~ } } { \arg \operatorname* { m a x } } _ { v \in [ V ] } \tilde { \mathbf { Z } } _ { b , \ell , v } , } \end{array}$ and ignores positions where $\mathbf { M } _ { b , \ell } = 0$

We now return to the main theme of this section: when optimizing the dummy label logits Z<sup>˜</sup>, samples within the same mini-batch are batch-separable and hence do not affect each other, except through the global normalization constant $N$ which is known to the adversary.

The key observation comes directly from the computational graph. For each sample $b \in [ B ]$ , the server-side proxy Top segment and the shifted soft-target loss induce a persample subgraph

$$
\begin{array} { r l r } & { } & { x _ { \mathrm { t r k } } ^ { ( b ) } , \ \mathbf { M } ^ { ( b ) } \  \tilde { \mathbf { L } } ^ { ( b ) } = \ F ( x _ { \mathrm { t r k } } ^ { ( b ) } , \ \mathbf { M } ^ { ( b ) } ; \ \theta _ { \mathrm { p r o } } ) , } \\ & { } & { \tilde { \mathbf { Z } } ^ { ( b ) } \  \mathbf { P } ^ { ( b ) } = \ \mathrm { S o f t m a x } \ ( \tilde { \mathbf { Z } } ^ { ( b ) } / \tau ) , ~ } \\ & { } & { \big ( \tilde { \mathbf { L } } ^ { ( b ) } , \mathbf { P } ^ { ( b ) } , \mathbf { M } ^ { ( b ) } \big ) \  \ \mathcal { L } _ { \mathrm { s o f t } } ^ { ( b ) } \  \ \tilde { \mathbf { g } } _ { \mathrm { t r k } } ^ { ( b ) } = \ \nabla _ { x _ { \mathrm { t r k } } ^ { ( b ) } } \mathcal { L } _ { \mathrm { s o f t } } . } \end{array}\tag{18}
$$

and these subgraphs are disjoint across different $b \mathbf { \hat { s } }$ because modern LLM implementations execute the forward/backward pass independently per example and only stack tensors along the batch dimension for efficiency. Concretely, the masked shifted loss decomposes as

$$
\begin{array} { l } { { \displaystyle \mathcal { L } _ { \mathrm { s o f t } } ( x _ { \mathrm { t r k } } , \tilde { \mathbf { Z } } ; \theta _ { \mathrm { p r o } } ) = \frac { 1 } { N } \sum _ { b = 1 } ^ { B } \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } \ell _ { b , \ell } } } \\ { ~ = \displaystyle \frac { 1 } { N } \sum _ { b = 1 } ^ { B } \mathcal { L } _ { \mathrm { s o f t } } ^ { ( b ) } , } \end{array}\tag{19}
$$

where $\begin{array} { r l r } { \mathcal { L } _ { \mathrm { s o f t } } ^ { ( b ) } } & { { } \triangleq } & { \sum _ { \ell = 1 } ^ { L - 1 } \mathbf { M } _ { b , \ell + 1 } \ell _ { b , \ell } } \end{array}$ depends only on $( \mathbf { x } _ { \mathrm { t r k } } ^ { ( b ) } , \mathbf { M } ^ { ( b ) } , \tilde { \mathbf { Z } } ^ { ( b ) } )$ . As a result, the dummy gradient tensor factorizes across the batch:

$$
\begin{array} { r l } & { \tilde { \mathbf { g } } _ { \mathrm { t r k } } ( \tilde { \mathbf { Z } } ) = \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } _ { \mathrm { s o f t } } \left( x _ { \mathrm { t r k } } , \tilde { \mathbf { Z } } ; \theta _ { \mathrm { p r o } } \right) } \\ & { ~ { = } \displaystyle \frac { 1 } { N } \Big [ \nabla _ { x _ { \mathrm { t r k } } ^ { ( 1 ) } } \mathcal { L } _ { \mathrm { s o f t } } ^ { ( 1 ) } , \dots , \nabla _ { x _ { \mathrm { t r k } } ^ { ( B ) } } \mathcal { L } _ { \mathrm { s o f t } } ^ { ( B ) } \Big ] , } \end{array}\tag{20}
$$

and there are no cross-sample Jacobian terms. More explicitly, for any $b \neq b ^ { \prime }$

$$
{ \frac { \partial \tilde { \mathbf { g } } _ { \mathrm { t r k } } ^ { ( b ) } ( \tilde { \mathbf { Z } } ) } { \partial \tilde { \mathbf { Z } } ^ { ( b ^ { \prime } ) } } } = \mathbf { 0 } , \ \mathrm { e q u i v a l e n t l y } \nabla _ { \tilde { \mathbf { Z } } ^ { ( b ^ { \prime } ) } , x _ { \mathrm { t r k } } ^ { ( b ) } } ^ { 2 } \mathcal { L } _ { \mathrm { s o f t } } = \mathbf { 0 } .
$$

Therefore, the gradient-matching objective $( \mathrm { e } . \mathrm { g } . , \mathcal { L } _ { \mathrm { T A G } } )$ decomposes into a sum of per-sample objectives (up to the shared scalar $1 / N )$

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathrm { T A G } } ( \tilde { \mathbf { Z } } ) = \sum _ { b = 1 } ^ { B } \mathcal { L } _ { \mathrm { T A G } } ^ { ( b ) } ( \tilde { \mathbf { Z } } ^ { ( b ) } ; \mathbf { G } ^ { ( b ) } ) } \\ & { \displaystyle \Rightarrow \nabla _ { \tilde { \mathbf { Z } } ^ { ( b ) } } \mathcal { L } _ { \mathrm { T A G } } ( \tilde { \mathbf { Z } } ) = \nabla _ { \tilde { \mathbf { Z } } ^ { ( b ) } } \mathcal { L } _ { \mathrm { T A G } } ^ { ( b ) } ( \tilde { \mathbf { Z } } ^ { ( b ) } ; \mathbf { G } ^ { ( b ) } ) , } \end{array}\tag{21}
$$

meaning that optimizing $\tilde { \mathbf { Z } } ^ { ( b ) }$ for one sample cannot change the matching loss terms of any other sample $b ^ { \prime } \neq { \bar { b } } .$ The only batch-level coupling is the normalization $N =$ $\begin{array} { r } { \sum _ { b = 1 } ^ { B } \dot { \sum _ { \ell = 1 } ^ { L - 1 } } \mathbf { M } _ { b , \ell + 1 } } \end{array}$ , which is fully known to the adversary because M must be provided to the server to execute attention in the trunk/top forward pass. Consequently, GMA-SL admits batch-separable gradient matching: the attacker may conceptually solve B independent subproblems (one per sample) within each mini-batch, without cross-sample interference.

## 3 Threat Model

We consider a curious-but-honest adversary. Specifically, the server is assumed to faithfully execute the split learning protocol (e.g., forward/backward computations and parameter updates) without deviating from it. Meanwhile, the server can log any legitimately available information and perform offline analysis or computation to reconstruct private sequences $x \in D _ { \mathrm { p r i } }$ , i.e., a passive inference adversary. Adversary’s Capacities. We endow the curious server with the following capabilities: (i) Interface observability. The curious server can monitor all cut-layer activations (i.e., clients’ smashed representations) $x _ { \mathrm { b t m } }$ and the corresponding back-propagated gradients $\nabla x _ { \mathrm { t r k } }$ exchanged during the fine-tuning procedure. (ii) White-box access to the trunk. The adversary has full white-box knowledge of the serverside Trunk segment. (iii) Semi-white-box access to clientside segments, which will be explained below. (iv) Flexible gradient-matching objective. The adversary may launch

![](images/167c4fb8b883e423aecf10e72e95f4dc86e80d77c6aef8655391be471eb30a9c.jpg)  
Figure 5: Illustration ofGradient Mirage against GMA-SL. The private track (red) remains local, while the exposed track (blue to-cyan) is protected before transmission. The numbered markers indicate the corresponding processing stages.

GMA-SL using any gradient-discrepancy measure as the matching objective, including cosine distance, $\ell _ { 1 }$ distance, $\ell _ { 2 }$ distance, or their weighted combinations (e.g., TAG).

Semi-White-Box Access. Following prior work on LLM-SL (Chen et al. 2024), we adopt a semi-white-box threat model. Concretely, the adversary does not have access to the victim’s exact per-step fine-tuned parameters or optimizer state, but is aware of the model architecture and pre-trained checkpoint. This access level—situated between the whitebox and black-box extremes—is both realistic and widely adopted in LLM-SL, since pre-trained weights are often public or available to the server as a model provider. Consequently, the attacker can mount near-white-box attacks but still faces a fine-tuning gap between the pre-trained proxy and the continually updated victim model.

## 4 Gradient Mirage

Gradient Mirage aims to induce three layers of inconsistency in GMA-SL while minimizing their impact on optimization (Fig. 5). Objective Inconsistency arises because the adversary assumes a full-label loss, whereas the exposed gradient is generated from a masked loss through Selective Autoregressive Supervision. Directional Inconsistency remains because Directional Privatization randomizes the exposed gradient direction using the von Mises–Fisher mechanism. Scale Inconsistency is introduced by Scale Blinding, making the gradient’s natural magnitude unreliable even when its direction is approximately matched. To further improve optimization compatibility, Gradient Mirage employs Dual-Track Backpropagation to preserve full-label learning at the Top segment and Bottom-Gradient Recovery to restore the effective gradient used for bottom optimization.

## 4.1 Dual-Track Backpropagation

In SL for autoregressive language modeling, the Top segment has access to full target labels, whereas lower segments receive only split-interface gradient, which may be exploited by GMA. This asymmetry suggests that full supervision should be confined to local learning within the Top segment, rather than directly reflected in the gradient

<sup>�</sup>�<sup>ℒ</sup>�exposed at the split interface.

To this end, we introduce two backward tracks with distinct roles: a Private Learning Track and an Exposed $O p t i -$ mization Track. For an input sequence, let $x _ { \mathrm { t r k } }$ denote the hidden representation sent from the trunk to the Top segment. We define a standard full-label autoregressive loss $\mathcal { L } _ { F }$ over all target tokens and a masked loss ${ \mathcal { L } } _ { M }$ over a selected subset of tokens. After a single forward pass produces logits $Z _ { p r e } ,$ , the two tracks perform separate backward passes. In the Private Learning Track, $\mathcal { L } _ { F }$ is used solely to optimize the Top-segment parameters, while its gradient with respect to $x _ { \mathrm { t r k } }$ is discarded. In the Exposed Optimization Track, ${ \mathcal { L } } _ { M }$ produces the interface gradient transmitted to the lower segments. This dual-track design preserves full-label learning at the Top segment while exposing only a controlled optimization signal across the split interface.

## 4.2 Scale Blinding

Building on the masked loss ${ \mathcal { L } } _ { M }$ defined above, we introduce Scale Blinding to induce scale inconsistency by randomizing the magnitudes of token-level gradient contributions. Specifically, we replace the default unit weight of each supervised token with an independently sampled coefficient $\alpha _ { t } > 0 ;$ , yielding the reweighted masked loss

$$
\mathcal { L } _ { S } = \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t = 1 } ^ { T } m _ { t } \alpha _ { t } \ell _ { t } , \bar { \mathbf { g } } = \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } _ { S } .\tag{22}
$$

The released gradient therefore no longer reflects the natural token-wise scale induced by ${ \mathcal { L } } _ { M }$ , hindering gradient matching and reconstruction. We parameterize the blinding distribution by its mean scale and relative token-wise variation, and select Unif[1500, 2000] from a broad effective range; sensitivity analyses and scale-aware adaptive-attack results are provided in Appendix D. This randomized reweighting avoids deterministic scale patterns, incurs negligible computational overhead, and preserves supervision at all selected positions while disrupting magnitude-sensitive matching objectives such as $\ell _ { 1 } , \ell _ { 2 } .$ , and their weighted combination.

## 4.3 Selective Autoregressive Supervision

In LLM-SL, the exposed split-interface gradient depends critically on the token positions included in the loss. To this end, we propose Selective Autoregressive Supervision (SAS), summarized in Algorithm 1, which supervises only a structured subset of token positions in each exposed backward pass. SAS preserves a useful optimization signal at the Top segment while inducing a mismatch between the true training gradient and the gradient assumed by an attacker.

For each sample, we construct the autoregressive target sequence by left-shifting the input label sequence, yielding $\mathbf { y } = ( y _ { 1 } , y _ { 2 } , \dots , y _ { T } )$ , where $T$ is the number of supervised target tokens. During the first forward pass, the Top segment produces token-wise predictive distributions $\{ p _ { t } \} _ { t = 1 } ^ { T } ,$ from which we compute a token-level difficulty score based on predictive entropy $\begin{array} { r } { H _ { t } = - \sum _ { v \in \mathcal { V } } p _ { t } ( v ) \log p _ { t } ( v ) } \end{array}$ , where $\nu$ denotes the vocabulary. Intuitively, a larger entropy means greater uncertainty about token $y _ { t } .$ , indicating that it is harder under the current model state.

We sort the sample’s token positions by entropy and partition them into three strata: high-entropy, medium-entropy, and low-entropy. Each stratum is then randomly and evenly divided into k disjoint subgroups, yielding 3k entropy-aware subgroups in total, where k is a user-specified hyperparameter. We next construct k token groups by pairing one subgroup from each entropy stratum, so that each group contains a balanced mix of high-, medium-, and low-entropy tokens. Denoting these token groups by $\mathcal { G } _ { 1 } , \mathcal { G } _ { 2 } , \ldots , \mathcal { G } _ { k }$ , we have $\mathcal { G } _ { i } \cap \mathcal { G } _ { j } ~ = ~ \emptyset ~ ( i ~ \neq ~ j ) , ~ \bigcup _ { i = 1 } ^ { k } \mathcal { G } _ { i } ~ = ~ \{ 1 , 2 , \ldots , T \}$ Hence, the k token groups are disjoint and jointly cover the entire label sequence.

At each training step, one token group is selected as the full-supervision group, and all its token positions are included in the masked loss. For the remaining $k - 1$ token groups, we randomly sample a fixed proportion of token positions and include only the sampled subset in the loss computation. Thus, the exposed loss is neither standard full-label supervision nor naive random masking, but a structured, entropy-aware supervision pattern that varies across steps while maintaining coverage over the full label sequence across training.

Formally, let $m _ { t } \in \{ 0 , 1 \}$ denote the supervision mask for token position t. For a selected full-supervision group $\mathcal { G } _ { r }$ , we set $m _ { t } = 1 , \forall t \in \mathcal { G } _ { r }$ . For each remaining group $\mathcal { G } _ { j }$ with $j \neq$ $r ,$ we independently sample a subset $\bar { S _ { j } } \ \bar { \subseteq } \ \bar { \mathcal { G } _ { j } }$ according to a total token sampling ratio $\rho \in [ 0 , 1 ]$ , and set $\dot { m } _ { t } = 1 , \forall t \in$ $S _ { j } , \ m _ { t } \ = \ 0 , \ \forall t \in \mathcal { G } _ { j } \ \backslash \ \mathcal { S } _ { j }$ . The masked autoregressive loss is then defined as $\begin{array} { r } { \mathcal { L } _ { M } \stackrel {  } { = } \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t = 1 } ^ { T } m _ { t } \ell _ { t } , \ell _ { t } \stackrel { _ { \ast } } { = } \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t = 1 } ^ { T } m _ { t } \ell _ { t } \mathrm { ~ , ~ } \ell _ { t } \stackrel { _ { \ast } } { = } \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t = 1 } ^ { T } m _ { t } \ell _ { t } \mathrm { ~ , ~ } \ell _ { t } \stackrel { _ { \ast } } { = } \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } } \end{array}$ $- \log p _ { t } ( y _ { t } )$

The design of SAS serves two complementary purposes. First, by ensuring that one token group is fully supervised at each step, the returned gradient still carries a sufficiently strong and coherent optimization signal. Second, by masking the remaining groups through entropy-aware partial sampling, the actual supervision pattern deviates from the attacker’s default assumption that the observed interface gradient corresponds to a standard full-label autoregressive loss over the complete target sequence.

Algorithm 1: Selective Autoregressive Supervision   
Input: Label sequence $\mathbf { y } = ( y _ { 1 } , \dots , y _ { T } ) ;$ predictive   
distributions $\{ p _ { t } \} _ { t = 1 } ^ { T }$ from the first forward   
pass; group number k; sampling ratio ρ   
Output: Supervision mask $\mathbf { n } = ( m _ { 1 } , \dots , m _ { T } )$   
1 Construct autoregressive targets by left-shifting ${ \bf y } ;$   
2 for $t = 1$ to T do   
// Compute token entropy   
3 $\begin{array} { r } { H _ { t } \gets - \sum _ { v \in \mathcal { V } } p _ { t } ( v ) \operatorname* { l o g } p _ { t } ( v ) ; } \end{array}$   
4 Sort token positions $\{ 1 , \ldots , T \}$ by $H _ { t }$ in descending   
order;   
5 Partition the sorted positions into three strata:   
$\mathcal { H } ^ { \mathrm { h i g h } } , \mathcal { H } ^ { \mathrm { m i d } } , \mathcal { H } ^ { \mathrm { l o w } } ;$   
6 Randomly and evenly split each stratum into k   
disjoint subgroups: $\{ \mathcal { H } _ { i } ^ { \mathrm { h i g h } } \} _ { i = 1 } ^ { k } , \{ \mathcal { H } _ { i } ^ { \mathrm { m i d } } \} _ { i = 1 } ^ { k } ,$   
$\{ \mathcal { H } _ { i } ^ { \mathrm { l o w } } \} _ { i = 1 } ^ { k } ;$   
7 for $i = 1$ to k do   
8 Form token group $\mathcal { G } _ { i }  \mathcal { H } _ { i } ^ { \mathrm { h i g h } } \cup \mathcal { H } _ { i } ^ { \mathrm { m i d } } \cup \mathcal { H } _ { i } ^ { \mathrm { l o w } }$   
9 Select one token group $\mathcal { G } _ { r }$ as the full-supervision   
group;   
// Full supervision   
10 Initialize $m _ { t } \gets 0$ for all $t = 1 , \dots , T ;$   
11 foreach $t \in \mathcal G _ { r }$ do   
12 $m _ { t } \gets 1$   
13 for $j = 1$ to k do   
14 if ${ \bf \dot { \boldsymbol { j } } } \neq { \boldsymbol { r } }$ then   
15 Randomly sample ${ \mathcal { S } } _ { j } \subseteq { \mathcal { G } } _ { j }$ with total   
sampling ratio $\rho ;$   
// Partial supervision   
16 foreach $t \in S _ { j }$ do   
17 $\perp m _ { t } \gets 1$   
18 return m;

The hyperparameter k controls the granularity of the entropy-aware supervision and can be chosen according to the training schedule. Notably, k is not required to equal the total number of training epochs. For example, even when fine-tuning lasts for $E ~ = ~ 6$ epochs, one may set $k = 3$ , use the resulting three token groups for 3 epochs, and then recompute the token-wise predictive distributions and entropy-based grouping under the updated model state for the remaining 3 epochs. In our LLM fine-tuning experiments, we use $E = 3$ epochs and accordingly set $k = 3$

## 4.4 Directional Privatization

To introduce directional inconsistency, we adopt the vMF mechanism for directional perturbation. Rather than adding isotropic noise to the gradient, we perturb only its direction while preserving its magnitude, thereby obscuring direction while retaining optimization stability.

Given a nonzero interface gradient $\textbf { g } \in \ \mathbb { R } ^ { d } ,$ , let $\mu ~ =$ $\frac { \mathbf { g } } { \| \mathbf { g } \| _ { 2 } } ~ \in ~ \mathbb { S } ^ { d - 1 }$ . We then sample a perturbed direction $\tilde { \mu } \sim$ $\mathrm { V M F } ( \mu , \kappa )$ , and release $\tilde { \textbf { g } } = \| \mathbf { g } \| _ { 2 } \tilde { \mu }$ , where the concentration parameter $\kappa \geq 0$ controls the perturbation strength.

Larger κ concentrates $\tilde { \mu }$ around $\mu ,$ whereas $\kappa = 0$ yields the uniform distribution on the hypersphere. The vMF mechanism satisfies Directional Metric Differential Privacy defined in Definition 1 under the angular distance $d _ { \angle } ( { \bf u } , { \bf v } ) =$ arccos $( \mathbf { u } ^ { \top } \mathbf { v } )$ . Here, the concentration parameter κ determines the corresponding privacy budget ε.

Definition 1 (Directional Metric Differential Privacy). Let $( \mathbb { S } ^ { d - 1 } , d )$ be the unit hypersphere equipped with angular metric $d _ { \angle } ,$ , and let $\varepsilon > 0 .$ . A randomized mechanism $\mathbf { \bar { \mathcal { M } } } : \mathbb { S } ^ { d - 1 } \to \mathbf { \bar { \mathcal { P } } } ( \mathbb { S } ^ { d - 1 } )$ satisfies εd-metric differential privacy if, for any two input directions u, $\textbf { v } \in \tilde { \mathbb { S } } ^ { d - 1 }$ and any measurable output set $\hat { S } \subseteq \mathbb { S } ^ { d - 1 }$ , it holds that

$$
\mathrm { P r } [ \mathcal { M } ( \mathbf { u } ) \in S ] \leq \exp \left( \varepsilon d _ { \mathcal { L } } ( \mathbf { u } , \mathbf { v } ) \right) \mathrm { P r } [ \mathcal { M } ( \mathbf { v } ) \in S ] .
$$

Compared with additive-noise DP mechanisms (Dwork et al. 2006), the vMF mechanism avoids the need for a userspecified clipping threshold $C ,$ , which is often difficult to estimate and tune in practice. The vMF mechanism also preserves the gradient magnitude, which uniformly bounds amplitude distortion. Specifically, for $\textbf { n } = \tilde { \textbf { g } } - \textbf { g }$ , we have $\mathbf { \dot { \| } g \| _ { 2 } } / \| \mathbf { n } \| _ { 2 } \geq 1 / 2$ for every realization, with equality when ${ \tilde { \mu } } = - \mu .$ . Hence, it provides directional privacy while preserving sufficient optimization signal for stable training.

Theorem 1 (Lower Bound on the Amplitude Signal– to-Noise Ratio). Let $\textbf { g } \in \mathbb { R } ^ { d }$ be a nonzero gradient and let $\begin{array} { r } { \mu = \frac { \mathbf { g } } { \| \mathbf { g } \| _ { \mathbf { \bar { s } } } } } \end{array}$ . Suppose that the vMF mechanism samples a perturbed direction $\tilde { \mu } \in \mathbb { S } ^ { d - 1 }$ and releases $\tilde { \mathbf { g } } = \| \mathbf { g } \| _ { 2 } \tilde { \mu }$

Define theperturbation noise as $\mathbf { n } = \tilde { \mathbf { g } } - \mathbf { g } .$ , and amplitude signal-to-noise ratio as $\begin{array} { r } { \mathrm { A S N R } = \frac { \| \mathbf { g } \| _ { 2 } } { \| \mathbf { n } \| _ { 2 } } } \end{array}$ . For any realization $o f \tilde { \mu } ,$ the amplitude signal-to-noise ratio satisfies ASNR ≥ $^ { \frac { 1 } { 2 } , }$ where the case $\tilde { \mu } = \mu$ is interpreted as $\mathrm { A S N R } = + \infty$ The lower bound is tight and is attained when ${ \tilde { \mu } } = - \mu .$

## 4.5 Bottom-Gradient Recovery

Scale Blinding deliberately amplifies token-level gradient contributions, which may distort the gradient scale used for Bottom optimization. We therefore normalize the gradient reaching the Bottom segment by the expected blinding factor. Let $\bar { \boldsymbol { \alpha } } = \mathbb { E } [ \boldsymbol { \alpha } _ { t } ]$ , and let g˜ denote the protected interface gradient. For $x _ { \mathrm { { t r k } } } = f ( x _ { \mathrm { { b t m } } } )$ , the gradient propagated through the Trunk is $\mathbf { \tilde { g } } _ { \mathrm { b t m } } = \mathbf { \tilde { \Lambda } } \mathbf { \tilde { J } } _ { f } ( x _ { \mathrm { b t m } } ) ^ { \top } \tilde { \mathbf { g } }$ . Before updating the Bottom segment, we recover its effective scale as $\begin{array} { r } { \mathbf { g } _ { \mathrm { b t m } } ^ { \mathrm { r e c } } = \frac { 1 } { \bar { \alpha } } \tilde { \mathbf { g } } _ { \mathrm { b t m } } } \end{array}$

Since α¯ is the expected token weight, this normalization removes the average scale amplification introduced by Scale Blinding while retaining its token-wise randomness.

## 5 Experiments

## 5.1 Experimental Configuration

Models and $\mathbf { D a t a s e t s } . ^ { 2 }$ We evaluate GMA-SL and its defenses on three LLMs: Llama-2-7B (Touvron et al. 2023), Llama-3-8B (Grattafiori et al. 2024), and DeepSeek-LLM-7B (Bi et al. 2024), using CodeAlpaca (Chaudhary 2023),

GSM8K (Cobbe et al. 2021), and PIQA (Bisk et al. 2020). All models and attack tensors use bfloat16. We assign six decoder layers to the Bottom segment and five to the Top segment, with the remaining layers hosted in the Trunk.

Evaluation Metrics. We evaluate the data reconstruction performance of GMA-SL using the following metrics: (i) ROUGE-based F1 scores (Lin 2004), primarily ROUGE-L F1; (ii) Meteor (Banerjee and Lavie 2005), as a complementary semantically aware metric. We use mean perplexity (PPL) to evaluate the fine-tuning performance of the LLMs. Attack Setting. We perform attacks periodically during fine-tuning. Specifically, over a total of 600 fine-tuning steps, we launch one GMA-SL attack every 50 steps and evaluate the corresponding reconstruction metrics. We adopt BiSR(b) (Chen et al. 2024), a state-of-the-art GMA-SL paradigm, with the optimization objective:

$$
\begin{array} { r l } & { L ( \tilde { y } ) = \beta \left. \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } ( x _ { \mathrm { t r k } } , \tilde { y } ) - \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } ( x _ { \mathrm { t r k } } , y ) \right. _ { 2 } } \\ & { \quad + ( 1 - \beta ) \left. \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } ( x _ { \mathrm { t r k } } , \tilde { y } ) - \nabla _ { x _ { \mathrm { t r k } } } \mathcal { L } ( x _ { \mathrm { t r k } } , y ) \right. _ { 1 } . } \end{array}\tag{23}
$$

In our experiments, $\beta$ is set to 0.65. In the ablation study, we also use cosine distance as the matching objective.

## 5.2 Existing Privacy-Preserving Defenses

Gradient Pruning. An intuitive defense against GMA-SL is Gradient Pruning (GP) (Vadera and Ameen 2022), which forces the adversary to match against an incomplete gradient. Fig. 6 presents the effect of GP against GMA-SL under different pruning rates (PRs), evaluated on Llama-3-8B. It can be seen that GP does not effectively defend against GMA-SL, and training becomes unstable when PR exceeds 90%.

Gradient Dropout. Gradient Dropout (GD) (Sotthiwat et al. 2026) transforms the shared gradient by randomly retaining a subset of coordinates and perturbing the rest. Each gradient component is kept with probability $p$ and rescaled by $1 / p ;$ otherwise, it is replaced with Gaussian noise sampled from ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ . The key idea is to preserve partial training signal while disrupting coordinate-wise gradient information, thereby hindering GMAs. Fig. 7 presents the effect of GD against GMA-SL under different drop rates and Gaussian noise standard deviations, evaluated on Llama-3-8B.

Gradient Sequence Local DP. Unlike DP-SGD (Abadi et al. 2016), the object to be released and protected in our setting is the gradient sequence $G \in \mathbb { R } ^ { L \times D }$ . Inspired by DP-Forward (Du et al. 2023), which enforces privacy in the forward pass by clipping and perturbing client-side intermediate activations before transmission, we extend Sequence Local DP (SeqLDP) (Du et al. 2023) to Gradient Sequence Local DP (GradSeq-LDP).

Definition 2 (GradSeq-LDP). For $\varepsilon \ge 0 , 0 \le \delta \le 1$ , M satisfies $( \varepsilon , \delta )$ -GradSeq-LDP, $i f \forall G , G ^ { \prime } \in \mathbb { R } ^ { L \times D }$ , and any possible output subset O,

$$
\operatorname* { P r } [ \mathcal { M } ( G ) \in \mathcal { O } ] \leq e ^ { \varepsilon } \operatorname* { P r } [ \mathcal { M } ( G ^ { \prime } ) \in \mathcal { O } ] + \delta .
$$

We implement GradSeq-LDP using the analytic Gaussian mechanism (aGM) (Balle and Wang 2018). GradSeq-LDP enforces privacy protection at an earlier stage than DP-SGD. Consequently, applying such a strong privacy mechanism at this location may substantially affect training. As shown in Tab. 4, a small ε indeed provides strong defense effectiveness, but leads to highly unstable training. Fig. 8 shows that fine-tuning under small privacy budgets is highly unstable and underperforms Top-Only Training. Even when a particular run appears stable, the training can become unstable after changing the random seed, as illustrated in Fig. 10.

<table><tr><td colspan="2">Dataset</td><td colspan="3">CodeAlpaca</td><td colspan="3">PIQA</td><td colspan="3">GSM8K</td></tr><tr><td>Model</td><td>Method</td><td>|RougeL-F Meteor</td><td></td><td>PPL</td><td>|RougeL-F Meteor</td><td></td><td>PPL</td><td>|RougeL-F Meteor</td><td></td><td>PPL</td></tr><tr><td rowspan="9"> $\mathrm { L l a m a } 3$  -8B</td><td>GP</td><td> $1 . 0 0 \pm . 0 0$ </td><td></td><td> $9 9 { \scriptstyle \pm . 0 0 } 5 . 2 9 { \scriptstyle \pm . 5 4 }$ </td><td> $. 9 6 \pm . 0 1$ </td><td> $. 9 4 \pm . 0 2$ </td><td> $1 1 . 5 0 { \scriptstyle \pm 2 . 1 6 }$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 1$ </td><td> $1 2 . 2 3 { \pm } 1 . 1 3$ </td></tr><tr><td>GD</td><td> $. 9 4 \pm . 0 1$ </td><td></td><td> $9 4 { \scriptstyle \pm 0 1 } 4 . 7 8 { \scriptstyle \pm . 1 0 }$ </td><td> $. 9 5 \pm . 0 1$ </td><td> $. 9 1 \pm . 0 0$ </td><td> $7 . 7 6 \pm . 0 3$ </td><td> $. 9 5 \pm . 0 1$ </td><td> $. 9 4 \pm . 0 1$ </td><td> $1 2 . 4 8 \pm . 9 7$ </td></tr><tr><td> $\mathrm { L D P } ( 1 0 2 4 0 )$ </td><td> $. 5 6 { \pm } . 0 3$ </td><td></td><td> $. 5 7 { \scriptstyle \pm . 0 2 \quad 4 . 6 7 \pm . 0 0 }$ </td><td> $. 8 0 \pm . 0 3$ </td><td> $. 7 6 \pm . 0 5$ </td><td> $7 . 7 4 \pm . 0 6$ </td><td> $. 9 0 \pm . 0 1$ </td><td> $. 9 2 \pm . 0 1$ </td><td> $1 1 . 0 4 \pm . 0 3$ </td></tr><tr><td> $\mathrm { L D P } ( 1 2 2 8 8 )$ </td><td> $. 6 6 \pm . 0 3$ </td><td></td><td> $. 6 6 { \scriptstyle \pm . 0 4 ~ 4 . 6 1 { \scriptstyle \pm . 0 4 } }$ </td><td> $. 8 1 \pm . 0 3$ </td><td> $. 7 7 \pm . 0 5$ </td><td> $\underline { { 7 . 2 0 \pm . 0 2 } }$ </td><td> $. 8 9 { \pm } . 0 2$ </td><td> $\underline { { . 8 9 \pm . 0 2 } }$ </td><td> $1 1 . 1 2 { \pm } . 0 4$ </td></tr><tr><td> $\mathrm { T o p \mathrm { - } O n l y }$ </td><td></td><td></td><td> $4 . 8 3 { \pm } . 0 1$ </td><td></td><td></td><td> $8 . 8 7 \pm . 0 2$ </td><td></td><td></td><td> $1 2 . 0 0 { \pm } . 0 1$ </td></tr><tr><td>Standard</td><td> $1 . 0 0 \pm . 0 0$ </td><td></td><td> $. 9 9 { \scriptstyle \pm . 0 0 } 4 . 4 4 { \scriptstyle \pm . 0 0 }$ </td><td> $. 9 9 \pm . 0 0$ </td><td> $. 9 7 \pm . 0 1$ </td><td> $7 . 4 0 \pm . 0 1$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $\underline { { 1 0 . 5 9 \pm . 0 0 } }$ </td></tr><tr><td>Ours(512)</td><td> $\mathbf { \nabla } _ { \mathbf { \cdot } } 2 \mathbf { 0 } \pm . 0 0$ </td><td></td><td> $. 2 3 { \scriptstyle \pm . 0 0 } 4 . 4 5 { \scriptstyle \pm . 0 0 }$ </td><td> $\mathbf { \Omega } \cdot 2 \mathbf { 1 } \pm . 0 2$ </td><td> ${ \bf 1 4 } \pm . 0 1$ </td><td> $7 . 4 0 \pm . 0 2$ </td><td> ${ \bf 1 1 } \pm . 0 1$ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } 7 \pm . 0 1$ </td><td> $1 0 . 6 4 \pm . 0 3$ </td></tr><tr><td>Ours(1024)</td><td> $. 3 5 \pm . 0 2$ </td><td></td><td> $\pm 0 2 \ \pm 4 . 4 2 \substack { \pm . 0 1 }$ </td><td> $. 4 5 \pm . 0 2$ </td><td> $\mathbf { . 3 1 \pm . 0 3 }$ </td><td> ${ \bf 7 . 1 9 \pm . 0 3 }$ </td><td> $. 3 6 \pm . 0 1$ </td><td> $\mathbf { \Omega } _ { 2 1 \pm . 0 1 }$ </td><td> $\mathbf { 1 0 . 5 0 } \pm . 0 1$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Llama2 -7B</td><td>GP</td><td> $. 9 9 \pm . 0 0$ </td><td></td><td> $9 9 { \scriptstyle \pm . 0 1 \quad 5 . 0 0 \pm . 0 9 }$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 1$ </td><td> $6 . 5 0 { \scriptstyle \pm . 6 2 }$ </td><td> $. 9 8 \pm . 0 1$ </td><td> $. 9 8 \pm . 0 0$ </td><td> $1 0 . 6 6 \pm . 2 3 $ </td></tr><tr><td>GD LDP(10240)</td><td> $. 9 0 \pm . 0 2$ </td><td></td><td> $. 8 9 { \pm } . 0 2 6 . 9 8 { \pm } 1 . 5 8$ </td><td> $. 8 1 \pm . 0 2$ </td><td> $. 8 0 \pm . 0 2$ </td><td> $7 . 7 6 { \pm } 1 . 7 7$ </td><td> $. 9 0 \pm . 0 1$ </td><td> $. 9 2 \pm . 0 0$ </td><td> $1 0 7 . 0 8 { \scriptstyle \pm 1 6 3 . 0 0 }$ </td></tr><tr><td>LDP(12288)</td><td> $. 7 8 { \pm } . 0 3$   $. 8 2 \pm . 0 4$ </td><td></td><td> $. 7 4 { \scriptstyle \pm . 0 4 \ } 4 . 8 0 { \scriptstyle \pm . 0 8 }$ </td><td> $. 7 9 \pm . 0 4$ </td><td> $. 7 4 \pm . 0 4$ </td><td> $6 . 3 0 \pm . 6 6$ </td><td> $. 3 8 \pm . 0 2$ </td><td> $. 3 3 { \pm } . 0 1$ </td><td> $1 0 . 4 9 \pm . 1 5$ </td></tr><tr><td> $\mathrm { T o p \mathrm { - } O n l y }$ </td><td></td><td> $. 8 0 \pm . 0 4$ </td><td> $4 . 8 5 { \pm } . 1 3$ </td><td> $. 8 3 \pm . 0 2$ </td><td> $. 7 9 \pm . 0 2$ </td><td> $5 . 8 5 \pm . 0 5$   $6 . 2 7 { \scriptstyle \pm . 0 6 }$ </td><td> $. 4 4 \pm . 0 3$ </td><td> $. 4 0 \pm . 0 3$ </td><td> $1 0 . 4 8 { \pm } . 1 4$ </td></tr><tr><td>Standard</td><td> $. 9 9 \pm . 0 0$ </td><td> $. 9 8 \pm . 0 1$ </td><td> $4 . 9 6 \pm . 0 4$   $4 . 7 3 { \scriptstyle \pm . 0 2 }$ </td><td> $. 9 9 \pm . 0 1$ </td><td> $. 9 9 \pm . 0 1$ </td><td> ${ \pm . 2 9 \pm . 1 0 }$ </td><td> $. 9 8 \pm . 0 0$ </td><td> $. 9 8 \pm . 0 0$ </td><td> $1 1 . 2 0 { \scriptstyle \pm . 2 8 }$   $\mathbf { 9 . 7 1 } { \pm . 1 4 }$ </td></tr><tr><td>Ours(512)</td><td> $\mathbf { 1 0 } \pm . 0 1$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours(1024)</td><td> $. 1 7 \pm . 0 2$ </td><td></td><td> $\mathbf { 1 1 } { \pm } . 0 1 \quad 4 . 8 2 { \pm } . 1 0$   $\begin{array} { l l } { { \pm } 1 5 { \pm } . 0 1 } & { { \underline { { 4 . 7 7 } } } { \pm } . 0 5 } \end{array}$ </td><td> $\mathbf { 0 5 } { \pm } . 0 2$   $\mathbf { 1 0 } \pm . 0 1$ </td><td> $\mathbf { \delta } \mathbf { . 0 5 } \pm . 0 0$   $\mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf \delta \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf \delta \mathbf { \delta } \mathbf { \delta } \mathbf \delta \mathbf { \delta } \delta \mathbf { \delta } \mathbf { \delta \delta } \mathbf \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta \delta } \mathbf \delta \delta \mathbf { \delta } \delta \mathbf \delta \delta \mathbf { \delta } \delta \delta \mathbf \delta \delta \delta \mathbf \delta \delta \delta \mathbf \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta $ </td><td> $5 . 6 3 \pm . 1 3$   $5 . 5 6 \pm . 2 3 $ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } 3 \pm . 0 1$   $\mathbf { 0 8 } { \pm } . 0 1$ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } 4 \pm . 0 1$   $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } 7 \pm . 0 1$ </td><td> $1 0 . 4 3 { \pm } . 1 6$   $\underline { { 1 0 . 1 4 \pm . 1 7 } }$ </td></tr><tr><td rowspan="9">DeepSeek -LLM-7B</td><td>GP</td><td> $. 9 9 \pm . 0 0$ </td><td></td><td> $9 7 { \scriptstyle \pm . 0 0 } 5 . 3 9 { \scriptstyle \pm . 0 5 }$ </td><td> $. 9 4 \pm . 0 2$ </td><td> $. 9 3 \pm . 0 2$ </td><td> $1 4 . 3 9 { \scriptstyle \pm 2 . 0 9 }$ </td><td> $. 7 9 \pm . 0 1$ </td><td> $. 7 5 \pm . 0 3$ </td><td> $3 9 . 1 1 \pm 3 5 . 0 0$ </td></tr><tr><td>GD</td><td> $. 6 8 \pm . 0 4$ </td><td></td><td>.68±.04 6.33±.33</td><td> $. 8 4 \pm . 0 1$ </td><td> $. 8 2 \pm . 0 3$ </td><td> $3 0 1 . 7 8 { \scriptstyle \pm 4 1 5 . 9 7 }$ </td><td> $. 8 4 \pm . 0 2$ </td><td> $. 8 5 \pm . 0 3$ </td><td> $1 6 . 1 0 { \scriptstyle \pm 3 . 4 6 }$ </td></tr><tr><td>LDP(8192)</td><td> $\underline { { 6 0 \pm . 0 2 } }$ </td><td></td><td> $. 5 4 { \scriptstyle \pm . 0 3 } \quad 5 . 0 5 { \scriptstyle \pm . 0 5 }$ </td><td> $. 8 2 { \pm } . 0 2$ </td><td> $. 7 6 \pm . 0 2$ </td><td> $6 . 7 1 \pm . 2 4$ </td><td> $. 3 1 \pm . 0 2$ </td><td> $. 2 1 { \pm } . 0 3$ </td><td> $1 2 . 1 3 { \pm } . 1 2$ </td></tr><tr><td> $\mathrm { L D P } ( 1 0 2 4 0 )$ </td><td> $. 7 4 \pm . 0 2$ </td><td></td><td> $. 7 1 { \scriptstyle \pm . 0 3 } \quad { \underline { { 5 . 0 0 } } } { \scriptstyle \pm . 1 0 }$ </td><td> $. 8 5 \pm . 0 1$ </td><td> $. 7 1 \pm . 0 2$ </td><td> $6 . 7 2 \pm . 2 7$ </td><td> $. 3 4 \pm . 0 1$ </td><td> $. 2 4 \pm . 0 3$ </td><td> $1 2 . 1 3 { \pm } . 1 2$ </td></tr><tr><td>Top-Only</td><td></td><td></td><td> $5 . 4 4 \pm . 1 2$ </td><td></td><td></td><td> $7 . 4 4 \pm . 1 3$ </td><td></td><td></td><td> $1 3 . 7 2 \pm . 1 1$ </td></tr><tr><td>Standard</td><td> $1 . 0 0 \pm . 0 0$ </td><td></td><td> $. 9 8 { \scriptstyle \pm . 0 0 } 4 . 3 8 { \scriptstyle \pm . 0 0 }$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $. 9 7 \pm . 0 1$ </td><td> ${ \bf 5 . 7 9 } \pm . 0 1$ </td><td> $. 9 8 \pm . 0 0$ </td><td> $. 9 8 \pm . 0 0$ </td><td> ${ \bf 1 0 . 0 1 } { \pm } . 0 0 $ </td></tr><tr><td>Ours(512)</td><td> $\mathbf { 0 7 \pm } . 0 2$ </td><td> $\mathbf { 0 5 } { \pm } . 0 2$ </td><td> $5 . 3 0 \pm . 2 2$ </td><td> $\mathbf { 0 7 \pm . 0 1 }$ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } \mathbf { 4 } \pm . 0 0$ </td><td> $6 . 6 7 \pm . 1 6$ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } 3 \pm . 0 0$ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } \mathbf { 1 } \pm . 0 0$ </td><td></td></tr><tr><td>Ours(1024)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $1 1 . 9 9 2 . 1 1 $ </td></tr><tr><td></td><td> ${ \bf 1 1 } \pm . 0 1$ </td><td> $\mathbf { 0 6 } \pm . 0 1$ </td><td> $5 . 1 5 { \pm } . 1 4$ </td><td> $. 1 4 \pm . 0 1$ </td><td> $\mathbf { 0 8 } { \pm } . 0 1$ </td><td> $6 . 4 5 { \scriptstyle \pm . 0 2 }$ </td><td> $\mathbf { 0 7 \pm . 0 1 }$ </td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 0 } 3 \pm . 0 0$ </td><td> $1 2 . 0 4 \pm . 2 9$ </td></tr></table>

Table 1: Comparison between Gradient Mirage and existing privacy-preserving methods across three models.

![](images/36e46249973042974519ef34d691cb0367e989e133f669c0ac3edb0dbc43f33d.jpg)  
Figure 6: Effect of GP against GMA-SL under different PRs.

![](images/a7b3c81b468e1ecde6a82cc76a29e05f2a96d2e47b67d142deab30990eb6ba37.jpg)  
Figure 7: Effect of GD against GMA-SL under different drop rates and Gaussian noise standard deviations.

remain heavily dominated by injected noise, exhibiting limited preservation of the original gradient signal. Despite such severe gradient perturbation, GradSeq-LDP still provides insufficient protection against GMA-SL, highlighting its unfavorable privacy–utility trade-off in this setting.

## 5.3 Benchmarking Gradient Mirage

To quantify the perturbation imposed by GradSeq-LDP on the exposed gradients, we employ four complementary gradient utility metrics: Amplitude Signal-to-Noise Ratio (ASNR), ASNR@10%, Recall@10% (R@10%), and Jaccard@10% (J@10%), which characterize the preservation of gradient magnitude and salient coordinates from complementary perspectives. Detailed definitions are provided in Appendix C.1. As shown in Fig. 9, even with an extremely large privacy budget of $\varepsilon = 1 2 2 8 8$ , the privatized gradients

To fairly compare defense effectiveness against BiSR(b), we tune all privacy-preserving methods to achieve comparable fine-tuning performance. Results are reported in Tab. 1. For GP, we use a PR of 0.7 for the Llama-series models and 0.8 for DeepSeek to obtain fine-tuning performance comparable to that of Gradient Mirage. For GD, we set the retention probability to $p = 0 . 6$ and the Gaussian noise standard deviation to $5 \times 1 0 ^ { - 3 }$ . For GradSeq-LDP, the clipping coefficients C are set to 0.12, 0.02, and 0.01 for Llama-3, Llama-2, and DeepSeek, respectively, while the corresponding privacy budgets ε are specified in the tables. For Gradient Mirage, we set the SAS sampling ratio to $\rho = 0 . 6$ and use the directional metric privacy budget $\varepsilon \in \{ 5 1 2 , 1 0 2 4 \}$ induced by the vMF mechanism for all models. All experiments are conducted with a batch size of 8 and repeated using three random seeds.

![](images/469e2a7215ca0b545549192a903a512ce8cd4e05ded2ed37f3693ff6710b2c0b.jpg)

![](images/ed55ba2d2ce8a17606dd339df40109396b51dd1082dea5999a5593a20c30fb34.jpg)

![](images/422ada1768ab7efae0dc3d5f5505613fbdf357fba8200f624f25666e7d0623ee.jpg)

![](images/3a99dfc38674a89d381a1a17d550ea7adca0114bc801888dd05deca2f147749b.jpg)  
Figure 8: Comparison between Top-only FT and SL FT under GradSeq-LDP. The left two panels correspond to 1-decoder and the right two to 5-decoder. Experiments were conducted on Llama3-8B using the CodeAlpaca dataset.

![](images/d432412e129d15bcbbfd9839a47fc3dce592f0d357486ecd41362d1569757949.jpg)

![](images/2fe6e1c1ce65055a3f93a8716b680f50ed00980e6215a597e9b165dfd2117812.jpg)

![](images/16f2bab89c07056b2a2e3745c68ba2d562e60b3e0190ba29dc549672c8bbb23f.jpg)

![](images/88c5fca430f502d267d99710b55796619da422b6d98b0d4f296743f0be069f14.jpg)  
Figure 9: Gradient utility of GradSeq-LDP under different privacy budgets across three datasets. Even at large ε, the privatized gradients retain limited signal strength and salient-coordinate information.

![](images/4eb80f5fb5e890c77a167ad32a1278a7872abdf65b50e97621aaa133417630fc.jpg)

![](images/81c6e7d3cbef93a1f163199c6811773b95086fa79bc34d75247a076baee6cb87.jpg)  
Figure 10: Seed sensitivity ofSL FT across random seeds.

## 5.4 Analysis of Scale Blinding Parameterization

Parameterization. To disentangle the effect of the absolute scaling magnitude from token-wise randomness, we parameterize the blinding coefficient as

$$
\alpha _ { t } = m ( 1 + u _ { t } ) , u _ { t } \sim \mathrm { U n i f } [ - r , r ] ,\tag{24}
$$

where $m = \mathbb { E } [ \alpha _ { t } ]$ controls the mean amplitude shift and $r$ controls the relative token-wise variation. Our default choice $\alpha _ { t } \sim \mathrm { U n i f } [ 1 5 0 0 , 2 0 0 0 ]$ corresponds to $m = 1 7 5 0$ and $r =$ $1 / 7$ , with a coefficient of variation of approximately 0.082. Scale-Aware Adaptive Attack. We additionally consider an attacker that jointly estimates an optimal nuisance scale during reconstruction. Specifically, the attacker optimizes both the dummy label $y _ { \mathrm { d u m } }$ and a learnable raw-scale parameter $s _ { \mathrm { r a w } }$ using the following scale-aware gradient-matching objective: min $\begin{array} { r l r } { \mathrm { i } _ { y _ { \mathrm { d u m } } , s _ { \mathrm { r a w } } } \left\| \mathbf { g } _ { \mathrm { r e l e a s e } } - s \mathbf { g } ( y _ { \mathrm { d u m } } ) \right\| _ { 2 } ^ { 2 } , } & { { } s } & { { } = } \end{array}$ $\mathrm { s o f t p l u s } ( s _ { \mathrm { r a w } } ) + \epsilon$ , where $\scriptstyle { \mathrm { g r e l e a s e } }$ denotes the released gradient, $\mathbf { g } ( y _ { \mathrm { d u m } } )$ denotes the dummy gradient generated from $y _ { \mathrm { { d u m } } } .$ , and $\epsilon _ { \mathrm { n u m } } = 1 0 ^ { - 8 }$ is a small constant introduced for numerical stability. The softplus parameterization ensures that the estimated scale remains strictly positive throughout optimization. Both $y _ { \mathrm { d u m } }$ and $s _ { \mathrm { r a w } }$ are jointly updated during reconstruction.

Effect of the Mean Scale $m _ { \bullet }$ . To study the impact of the mean amplitude shift, we fix the relative token-wise variation as $r = 1 / 7$ and evaluate different mean scales: $m \in$ {100, 500, 1000, 1500, 1750, 2500, 4000}. The corresponding defense effectiveness and fine-tuning performance are reported in Tab. 2 and Fig. 11, respectively. Tab. 2 reports ROUGE-L-F as the reconstruction metric. Throughout the Scale Blinding experiments in section 5.4, the directional DP budget is fixed at $\varepsilon = 5 1 2$

As the mean scale increases, the reconstruction quality decreases rapidly in the low-scale regime and gradually stabilizes, while the fine-tuning utility remains largely unaffected across different settings. This indicates that increasing the mean scale effectively disrupts magnitude-sensitive gradient matching, but the privacy gain becomes less significant once the scale reaches a sufficiently large range.

Based on these observations, we choose $m = 1 7 5 0$ (corresponding to $\alpha _ { t } \sim \mathrm { U n i f } [ 1 5 0 0 , 2 0 0 0 ] )$ as our default setting. Although this value is not necessarily the globally optimal choice, it provides a favorable privacy-utility trade-off while avoiding unnecessarily large gradient scaling.

Effect of Relative Token-wise Variation r. We further investigate the impact of the relative token-wise variation $r ,$ which controls the randomness introduced into the tokenlevel scaling coefficients. We evaluate different values of r under multiple mean scales m, and the reconstruction results are summarized in Tab. 3. The experiments are conducted on the CodeAlpaca dataset using the Llama-3-8B model. ROUGE-L-F is reported as the reconstruction metric.

When r is close to zero, the scaling coefficients become concentrated around a similar value, causing Scale Blinding to degenerate toward deterministic global rescaling. Such a transformation can be partially compensated by an adaptive attacker and provides limited disruption to the relative magnitude pattern among token-level gradients.

<table><tr><td rowspan="2">Matching Objective</td><td colspan="7">m</td></tr><tr><td>100</td><td>500</td><td>1000</td><td>1500</td><td>1750</td><td>2500</td><td>4000</td></tr><tr><td>TAG</td><td> $. 3 8 2 { \pm } . 0 2 6 $ </td><td>.246±.018</td><td> $. 2 1 1 { \pm } . 0 1 2 $ </td><td>.195±.016</td><td> $. 2 0 2 { \pm } . 0 0 2$ </td><td>.170±.002</td><td>.168±.010</td></tr><tr><td>Cosine Distance</td><td>.089±.008</td><td>.088±.006</td><td> $. 0 8 7 { \pm } . 0 0 8 $ </td><td>.090±.003</td><td> $. 0 9 4 { \pm } . 0 0 5$ </td><td>.100±.004</td><td>.088±.007</td></tr><tr><td>Adaptive TAG</td><td>.360±.024</td><td>.248±.018</td><td> $. 2 1 3 { \pm } . 0 1 0 $ </td><td>.197±.021</td><td>.208±.016</td><td>.187±.005</td><td>.184±.013</td></tr></table>

Table 2: Effect ofthe mean scale m on reconstruction quality under different matching objectives.
<table><tr><td rowspan="2">r</td><td colspan="7">m</td></tr><tr><td>100</td><td>500</td><td>1000</td><td>1500</td><td>1750</td><td>2500</td><td>4000</td></tr><tr><td>1/5</td><td> $. 3 4 1 { \pm } . 0 2 1$ </td><td>.212±.014</td><td>.191±.012</td><td>.187±.012</td><td> $. 1 8 3 { \pm } . 0 1 1$ </td><td>.143±.013</td><td>.133±.014</td></tr><tr><td>1/7</td><td> $. 3 6 0 { \pm } . 0 2 4$ </td><td>.248±.018</td><td>.213±.010</td><td>.197±.021</td><td>.208±.016</td><td>.187±.005</td><td>.184±.013</td></tr><tr><td>1/10</td><td> $. 3 9 1 { \pm } . 0 1 7$ </td><td>.250±.011</td><td>.219±.013</td><td>.199±.023</td><td>.208±.007</td><td>.186±.011</td><td>.189±.009</td></tr><tr><td>1/25</td><td> $. 3 9 7 { \pm } . 0 2 4 $ </td><td>.258±.017</td><td>.221±.023</td><td>.237±.015</td><td> $. 2 1 9 { \pm } . 0 1 0 $ </td><td>.194±.010</td><td>.193±.010</td></tr><tr><td>1/50</td><td> $. 4 0 4 { \pm } . 0 2 4 $ </td><td>.263±.017</td><td>.227±.016</td><td>.232±.022</td><td> $. 2 3 9 { \pm } . 0 0 8 $ </td><td>.198±.005</td><td>.204±.009</td></tr></table>

Table 3: Effect of the relative token-wise variation r on reconstruction quality under different mean scales m.

<table><tr><td rowspan="2">BiSR(b)</td><td colspan="3">CodeAlpaca</td></tr><tr><td>RougeL-F</td><td>Rouge2-F</td><td>Meteor</td></tr><tr><td>ε=128</td><td>.02±.01</td><td>.00±.00</td><td>.01±.00</td></tr><tr><td>ε=256</td><td>.03±.01</td><td>.00±.00</td><td>.02±.01</td></tr><tr><td>ε=1024</td><td>.12±.05</td><td>.05±.03</td><td>.10±.05</td></tr><tr><td>ε=8192</td><td>.62±.07</td><td>.48±.09</td><td>.62±.08</td></tr><tr><td>ε=10240</td><td>.56±.03</td><td> $. 4 1 { \pm } . 0 2 $ </td><td>.57±.02</td></tr><tr><td>ε=12288</td><td>.66±.03</td><td>.52±.04</td><td>.66±.04</td></tr></table>

Table 4: Defense performance of GradSeq-LDP against GMA-SL under different privacy budgets.

<table><tr><td>Setting</td><td>RougeL-F</td><td>Rouge2-F</td><td>Meteor</td></tr><tr><td>w/o Scale Blinding</td><td>.71±.01</td><td>.51±.02</td><td>.69±.01</td></tr><tr><td>w/ Scale Blinding</td><td>.35±.02</td><td>.18±.02</td><td>.34±.02</td></tr></table>

Table 5: Impact of Scale Blinding.

As shown in Table 3, introducing token-wise variation consistently improves the effectiveness of Scale Blinding compared with small variation settings. However, the benefit does not increase monotonically with r. Excessively large variation may introduce overly heterogeneous gradient contributions across tokens, which can increase optimization disturbance without providing proportional privacy gains.

Considering both reconstruction resistance and optimization stability, we choose $r = 1 / 7$ as the default setting. This value introduces sufficient token-wise randomness while avoiding excessive perturbation of the gradient structure.

<table><tr><td>Setting</td><td>|RougeL-F Rouge2-F</td><td></td><td>Meteor</td></tr><tr><td>Scale Blinding-Only</td><td>.55±.11</td><td>.25±.04</td><td>.43±.11</td></tr><tr><td>ρ=0.8</td><td>.42±.09</td><td>.11±.04</td><td>.28±.09</td></tr><tr><td>ρ=0.7</td><td>.41±.06</td><td>.11±.03</td><td>.28±.04</td></tr><tr><td>ρ=0.6</td><td>.37±.04</td><td>.09±.02</td><td>.25±.03</td></tr><tr><td> $\scriptstyle \rho = 0 . 6 , \varepsilon = 8 1 9 2$ </td><td>.29±.02</td><td>.05±.01</td><td>.19±.02</td></tr><tr><td> $\rho { = } 0 . 6 , \varepsilon { = } 1 0 2 4$ </td><td>.17±.01</td><td>.01±.00</td><td>.11±.01</td></tr><tr><td> $\rho { = } 0 . 6 , \varepsilon { = } 5 1 2$ </td><td>.09±.01</td><td>.00±.00</td><td>.07±.01</td></tr></table>

Table 6: Impact of SAS and Directional Privatization.

## 5.5 Ablation Study

Impact of SAS and Directional Privatization. We use cosine distance as the matching objective to isolate SAS and Directional Privatization. As shown in Tab. 6, both SAS and Directional Privatization substantially reduce reconstruction performance by perturbing the positional and directional information of the exposed gradient, respectively. The configuration of Scale Blinding is kept fixed throughout this experiment.

Impact of Scale Blinding. We evaluate Scale Blinding against BiSR(b), which adopts the magnitude-sensitive objective used by TAG. As shown in Tab. 5, removing Scale Blinding increases ROUGE-L F1 from 0.35 to 0.71, indicating that SAS and Directional Privatization are insufficient. The vMF mechanism uses a privacy budget of $\varepsilon = 1 0 2 4$ while SAS is configured with a total token ratio of ρ = 0.6.

Impact of Dual-Track Backpropagation. As shown in Tab. 7, removing Dual-Track Backpropagation consistently increases PPL across all datasets. This confirms that retaining full-label supervision in the private learning track is important for preserving fine-tuning utility.

Impact of Bottom-Gradient Recovery. Disabling Bottom-Gradient Recovery likewise degrades PPL across all datasets, as the blinded gradient scale directly affects Bottom optimization. These results validate its role in preventing Scale Blinding from impairing training.

![](images/7d2731ecbf4e7643f59759824bb6622883cce791980492bfafb7dc590f3bf42a.jpg)

Figure 11: Fine-tuning performance under different mean scales m with fixed relative token-wise variation r = 1/7. The results are obtained on Llama-3-8B with the CodeAlpaca dataset.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Standard PPL</td><td colspan="2">Dual-Track BP (X)</td><td colspan="2">Bottom-Gradient Rec (X)</td><td colspan="2">Trunk Training (X)</td></tr><tr><td>PPL</td><td>∆(%)</td><td>PPL</td><td>∆(%)</td><td>PPL</td><td>∆(%)</td></tr><tr><td>CodeAlpaca</td><td>4.42±.01</td><td>4.51±.05</td><td>2.04↑</td><td>4.51±.01</td><td>2.04↑</td><td>4.48±.05</td><td>1.36↑</td></tr><tr><td>PIQÀ</td><td>7.19±.03</td><td>7.30±.02</td><td>1.53↑</td><td>7.43±.03</td><td>3.34↑</td><td>7.23±.02</td><td>0.56↑</td></tr><tr><td>GSM8K</td><td>10.50±.01</td><td>10.61±.05</td><td>1.05↑</td><td>10.69±.03</td><td>1.82↑</td><td>10.59±.03</td><td>0.86↑</td></tr></table>

Table 7: Impact of Dual-Track Backpropagation, Bottom-Gradient Recovery, and Trunk Training on SL utility.

Impact of Trunk Training. The gradients in the Trunk segment are subject to a certain degree of perturbation. Therefore, whether these gradients should be used to update the model is itself an important factor that warrants ablation. As shown in Tab. 7, we find that updating the trunk parameters does not compromise training stability; instead, it slightly improves training performance.

For the ablation studies of Dual-Track Backpropagation, Bottom-Gradient Recovery, and Trunk Training, the vMF mechanism uses a privacy budget of ε = 1024, while SAS is configured with a total token ratio of ρ = 0.6. The parameters of Scale Blinding are set to m = 1750 and r = 1/7.

## 6 Conclusion

We identify gradient–objective consistency as the root vulnerability enabling GMA-SL in LLM-SL, and propose Gradient Mirage, a defense that turns gradient matching into a misspecified inverse problem. By injecting objective, directional, and scale inconsistencies into the exposed gradient while preserving full-label learning and recovering the effective bottom gradient, Gradient Mirage decouples what the model learns from what the gradient reveals. Experiments across multiple LLMs and datasets show that it substantially suppresses label reconstruction under comparable fine-tuning performance, achieving a superior privacy-utility trade-off. This work focuses on autoregressive training scenarios of large language models in split learning. Exploring gradient privacy attacks and defenses under other training paradigms and different model architectures, such as multimodal and vision models, remains a challenging direction.

## Acknowledgments

This work was partially funded by NSFC Grants No.62272222 and 62272215, Jiangsu Province Outstanding Youth Fund Project (No.BK20230080), the Nanjing University-China Mobile Communications Group Co.,Ltd. Joint Institute (NJ20250025).

## References

Abadi, M.; Chu, A.; Goodfellow, I.; McMahan, H. B.; Mironov, I.; Talwar, K.; and Zhang, L. 2016. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, 308–318.

Balle, B.; and Wang, Y.-X. 2018. Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International conference on machine learning, 394–403. PMLR.

Balunovic, M.; Dimitrov, D.; Jovanovic, N.; and Vechev, M.´ 2022. Lamp: Extracting text from gradients with language model priors. Advances in Neural Information Processing Systems, 35: 7641–7654.

Banerjee, S.; and Lavie, A. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 65–72.

Bi, X.; Chen, D.; Chen, G.; Chen, S.; Dai, D.; Deng, C.; Ding, H.; Dong, K.; Du, Q.; Fu, Z.; et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954.

Bisk, Y.; Zellers, R.; Gao, J.; Choi, Y.; et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 7432–7439.

Cai, Z.; Ma, R.; Fu, Y.; Zhang, W.; Ma, R.; and Guan, H. 2024. LLMaaS: Serving Large-Language Models on Trusted Serverless Computing Platforms. IEEE Transactions on Artificial Intelligence, 6(2): 405–415.

Chaudhary, S. 2023. Code alpaca: An instruction-following llama model for code generation.

Chen, G.; Qin, Z.; Yang, M.; Zhou, Y.; Fan, T.; Du, T.; and Xu, Z. 2024. Unveiling the vulnerability of private finetuning in split-based frameworks for large language models: A bidirectionally enhanced attack. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, 2904–2918.

Cho, K.; Van Merrienboer, B.; Bahdanau, D.; and Bengio,¨ Y. 2014. On the properties of neural machine translation: Encoder–decoder approaches. In Proceedings of SSST-8, eighth workshop on syntax, semantics and structure in statistical translation, 103–111.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; et al. 2021. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168, 9.

Deng, J.; Wang, Y.; Li, J.; Wang, C.; Shang, C.; Liu, H.; Rajasekaran, S.; and Ding, C. 2021. Tag: Gradient attack on transformer-based language models. In Findings ofthe association for computational linguistics: EMNLP 2021, 3600– 3610.

Du, M.; Yue, X.; Chow, S. S.; Wang, T.; Huang, C.; and Sun, H. 2023. Dp-forward: Fine-tuning and inference on language models with differential privacy in forward pass.

In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, 2665–2679.

Dwork, C.; McSherry, F.; Nissim, K.; and Smith, A. 2006. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, 265–284. Springer.

Fisher, R. A. 1953. Dispersion on a sphere. Proceedings of the royal society of London. Series A. Mathematical and physical sciences, 217(1130): 295–305.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Gupta, O.; and Raskar, R. 2018. Distributed learning of deep neural network over multiple agents. Journal of Network and Computer Applications, 116: 1–8.

Gupta, S.; Huang, Y.; Zhong, Z.; Gao, T.; Li, K.; and Chen, D. 2022. Recovering private text in federated learning of language models. Advances in neural information processing systems, 35: 8130–8143.

Lin, C.-Y. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 74–81.

Lin, Y.; Zhang, Q.; Ruan, W.; Zhang, D.; Hong, J.; Wu, Y.; Xia, H.; Mao, Y.; and Zhong, S. 2026. Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report). arXiv preprint arXiv:2603.01499.

Pasquini, D.; Ateniese, G.; and Bernaschi, M. 2021. Unleashing the tiger: Inference attacks on split learning. In Proceedings of the 2021 ACM SIGSAC conference on computer and communications security, 2113–2129.

Poirot, M. G.; Vepakomma, P.; Chang, K.; Kalpathy-Cramer, J.; Gupta, R.; and Raskar, R. 2019. Split learning for collaborative deep learning in healthcare. arXiv preprint arXiv:1912.12115.

See, A.; Liu, P. J.; and Manning, C. D. 2017. Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1073– 1083.

Shen, X.; Liu, Y.; Liu, Y.; Wang, P.; Liu, H.; Hong, J.; Duan, B.; Huang, Z.; Mao, Y.; Wu, Y.; et al. 2025. SAP: Privacy-Preserving Fine-Tuning on Language Models with Splitand-Privatize Framework. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 502–510.

Sotthiwat, E.; Zhang, C.; Xiao, X.; and Zhen, L. 2026. Safeguarding Federated Learning From Data Reconstruction Attacks via Gradient Dropout. TIFS.

Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Vadera, S.; and Ameen, S. 2022. Methods for pruning deep neural networks. Ieee Access, 10: 63280–63300.

Vepakomma, P.; Gupta, O.; Swedish, T.; and Raskar, R. 2018. Split learning for health: Distributed deep learning without sharing raw patient data. arXiv preprint arXiv:1812.00564.

von Mises, R. 1918. Uber die “Ganzzahligkeit” der Atom- <sup>¨</sup> gewichte und verwandete Fragen. Physikalische Zeitschrift, 19: 490.

Weggenmann, B.; and Kerschbaum, F. 2021. Differential privacy for directional data. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, 1205–1222.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; et al. 2024. Qwen2. 5 Technical Report. arXiv e-prints, arXiv–2412.

Zhang, Q.; Shi, Y.; Zhang, Z.; Wang, H.; Zhang, S. Q.; and Li, J. 2026. Stealing Split Learning Bottom Models by Recovering Embedding Geometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20660–20669.

Zhao, B.; Mopuri, K. R.; and Bilen, H. 2020. idlg: Improved deep leakage from gradients. arXiv preprint arXiv:2001.02610.

Zhu, L.; Liu, Z.; and Han, S. 2019. Deep leakage from gradients. Advances in neural information processing systems, 32.

## SUPPLEMENTARY MATERIALCE<sup>L</sup>

The supplementary material provides additional details <sub>about</sub> <sub>our</sub> <sub>work.</sub> <sub>Specifically,</sub> <sub>we</sub> <sub>provide:</sub>�<sub>top</sub>

• More details of the vMF Mechanism.

• More details of Selective Autoregressive Supervision.

• Additional experimental details and analyses.

• Visualization of the effectiveness of Gradient Mirage.

• Future Work.

## A Details of the vMF Mechanism

This section provides additional details on the vMF mechanism. Sections A.1 and A.2 describe our implementation of vMF sampling, including the tangent-normal decomposition and the rejection sampling procedure, respectively. Section A.3 provides the proof of the theoretical result stated in the main text.

## A.1 Tangent-Normal Decomposition in vMF Sampling

The von Mises-Fisher privacy mechanism adopts the von Mises-Fisher (vMF) distribution, named after von Mises and Fisher. This distribution is defined over the unit hypersphere $S ^ { n - 1 }$

Definition 3 (The vMF Distribution). The vMF distribution on $\mathbb { S } ^ { n - 1 }$ with mean direction $\mu \in \mathbb { S } ^ { n - 1 }$ and concentration parameter $\kappa \geq 0$ is defined by thefollowing density:

$$
\operatorname { v M F } ( \boldsymbol { \mu } , \kappa ) [ \mathbf { x } ] = C _ { \mathrm { v M F } } ( \boldsymbol { n } , \kappa ) \cdot \exp \left( \boldsymbol { \kappa } \cdot \boldsymbol { \mu } ^ { \top } \mathbf { x } \right) ,
$$

where $\mathbf { x } \in \mathbb { S } ^ { n - 1 }$ and $C _ { \mathrm { v M F } } ( n , \kappa )$ is the normalization constant.

To sample a new direction vector $\tilde { \mu } \in \mathbb S ^ { n - 1 }$ under the vMF mechanism centered at $\mu \in \mathbb { S } ^ { n - 1 }$ , we use the tangentnormal decomposition. As illustrated in Fig. 12, the sampled vector $\tilde { \mu }$ is decomposed into a component along µ and a component in the subspace orthogonal to $\mu .$ . Let $t = \tilde { \mu } ^ { \top } \mu =$ cos $\theta ,$ , where $\theta$ is the angle between $\tilde { \mu }$ and $\mu .$ . Then the orthogonal component has magnitude $h = { \sqrt { 1 - t ^ { 2 } } } = \sin \theta$ If $\bar { \xi } \sim \mathrm { U n i f } ( \bar { \mathbb { S } } ^ { n - 2 } \ \bot \ \mu )$ is a unit vector uniformly sampled from the subsphere orthogonal to $\mu ,$ the sampled direction can be written as $\tilde { \mu } = t \mu + \sqrt { 1 - t ^ { 2 } } \xi$ or equivalently $\tilde { \mu } = \cos \theta \mu + \sin \theta \xi .$ Therefore, vMF sampling is separated into two parts: sampling the axial projection t (or equivalently the angle θ) and sampling the tangential direction $\xi$ uniformly on the orthogonal subsphere.

## A.2 Rejection Sampling for t

After the tangent-normal decomposition, vMF sampling reduces to drawing the axial variable $t \in [ - 1 , 1 ]$ . In our implementation, t is generated by a rejection sampling scheme with a Beta proposal. Specifically, we first sample

$$
y \sim \mathrm { B e t a } \left( { \frac { p - 1 } { 2 } } , { \frac { p - 1 } { 2 } } \right) ,\tag{25}
$$

![](images/ca65c5b1c50a696742284642ac478e14e3b6b60c2b258f07f1fc659ae918c60f.jpg)  
Figure 12: Tangent-normal decomposition in vMF sampling. The sampled direction µ˜ is decomposed into an axial component $t = \tilde { \mu } ^ { \top } \mu =$ cos θ along µ and a tangential component of magnitude ${ \sqrt { 1 - t ^ { 2 } } } = \sin \theta .$ , where $\xi \sim \mathrm { U n i f } ( \mathbb { S } ^ { n - 2 } \perp $ $\mu )$

and transform it into a candidate $\begin{array} { r } { t = \frac { 1 - ( 1 + b _ { 0 } ) y } { 1 - ( 1 - b _ { 0 } ) y } } \end{array}$ , where $b _ { 0 } =$ $\frac { - 2 \kappa + \sqrt { 4 \kappa ^ { 2 } + ( p - 1 ) ^ { 2 } } } { p - 1 }$ . Following the standard construction, we

$$
a = \frac { p - 1 + 2 \kappa + \sqrt { 4 \kappa ^ { 2 } + ( p - 1 ) ^ { 2 } } } { 4 } ,\tag{26}
$$

and

$$
d = { \frac { 4 a b _ { 0 } } { 1 + b _ { 0 } } } - ( p - 1 ) \log ( p - 1 ) .\tag{27}
$$

For each proposed sample, let

$$
s = { \frac { 2 a b _ { 0 } } { 1 - ( 1 - b _ { 0 } ) y } } .\tag{28}
$$

The candidate is accepted if

$$
\log u \leq ( p - 1 ) \log s - s + d ,\tag{29}
$$

where $u \sim \mathrm { U n i f } ( 0 , 1 )$ . Otherwise, the sample is rejected and the procedure is repeated. This yields a valid sample for $t ,$ which is then combined with the tangential direction to form the final vMF sample.

## A.3 Proofs related to the vMF Mechanism

Theorem 2 (The vMF Mechanism Satisfies $\varepsilon d _ { 2 } { \bf - P r i v a c y ) }$ Let $\mathcal { M } _ { \mathrm { v M F } }$ denote the vMF mechanism. Then $\mathcal { M } _ { \mathrm { v M F } }$ satisfies εd -metric differential privacy, where

$$
d _ { 2 } ( \mathbf { u } , \mathbf { v } ) = \| \mathbf { u } - \mathbf { v } \| _ { 2 } .
$$

For any u, $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ and any measurable set $S \subseteq \mathbb { S } ^ { d - 1 }$

$$
\operatorname* { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { u } ) \in S ] \leq \exp \bigl ( \varepsilon d _ { 2 } ( \mathbf { u } , \mathbf { v } ) \bigr ) \ \operatorname* { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { v } ) \in S ] .
$$

ProofofTheorem 2. Let $\mathbf { u } , \mathbf { v } \in \mathbb { S } ^ { d - 1 }$ be arbitrary unit vectors, and let $S \subseteq \mathbb { S } ^ { d - 1 }$ be any measurable set. For any $\mathbf { z } \in S .$ the vMF density satisfies

$$
\frac { p _ { \bf u } ( { \bf z } ) } { p _ { \bf v } ( { \bf z } ) } = \frac { C _ { d } ( \varepsilon ) \exp ( \epsilon { \bf u } ^ { \top } { \bf z } ) } { C _ { d } ( \epsilon ) \exp ( \epsilon { \bf v } ^ { \top } { \bf z } ) } = \exp \bigl ( \epsilon ( { \bf u } - { \bf v } ) ^ { \top } { \bf z } \bigr ) .
$$

By the Cauchy–Schwarz inequality,

$$
( \mathbf { u } - \mathbf { v } ) ^ { \top } \mathbf { z } \leq \| \mathbf { u } - \mathbf { v } \| _ { 2 } \| \mathbf { z } \| _ { 2 } .
$$

Since $\mathbf { z } \in \mathbb { S } ^ { d - 1 }$ , we have $\| \mathbf { z } \| _ { 2 } = 1$ . Hence,

$$
\frac { p _ { \mathbf { u } } ( \mathbf { z } ) } { p _ { \mathbf { v } } ( \mathbf { z } ) } \leq \exp \bigl ( \varepsilon \| \mathbf { u } - \mathbf { v } \| _ { 2 } \bigr ) = \exp \bigl ( \varepsilon d _ { 2 } ( \mathbf { u } , \mathbf { v } ) \bigr ) .
$$

Integrating both sides over $\mathbf { z } \in S .$ , we obtain

$$
\begin{array} { r l } { \displaystyle \operatorname* { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { u } ) \in S ] = \int _ { S } p _ { \mathbf { u } } ( \mathbf { z } ) d \mathbf { z } } \\ { \displaystyle } & { \leq \exp \bigl ( \varepsilon d _ { 2 } ( \mathbf { u } , \mathbf { v } ) \bigr ) \int _ { S } p _ { \mathbf { v } } ( \mathbf { z } ) d \mathbf { z } } \\ { \displaystyle } & { = \exp \bigl ( \varepsilon d _ { 2 } ( \mathbf { u } , \mathbf { v } ) \bigr ) \operatorname* { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { v } ) \in S ] . } \end{array}
$$

Therefore, $\mathcal { M } _ { \mathrm { v M F } }$ satisfies $\varepsilon d _ { 2 } { \mathrm { - m e t r i c } }$ differential privacy. □

Theorem 3 (The vMF Mechanism Satisfies $\varepsilon d _ { \angle } { \bf - P r i v a c y } )$ $\mathcal { M } _ { \mathrm { v M F } }$ also satisfies $\varepsilon d _ { \angle }$ -metric differential privacy, where

$$
d _ { \angle } ( \mathbf { u } , \mathbf { v } ) = \operatorname { a r c c o s } ( \mathbf { u } ^ { \top } \mathbf { v } ) .
$$

For any u, $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ and any measurable se $S \subseteq \mathbb { S } ^ { d - 1 }$

$$
\operatorname* { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { u } ) \in S ] \leq \exp \left( \varepsilon d _ { \mathcal { L } } ( \mathbf { u } , \mathbf { v } ) \right) \operatorname* { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { v } ) \in S ] .
$$

Proof of Theorem 3. From Theorem 2, the vMF mechanism satisfies

$$
\mathrm { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { u } ) \in S ] \leq \exp \left( \varepsilon \| \mathbf { u } - \mathbf { v } \| _ { 2 } \right) \mathrm { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { v } ) \in S ] .
$$

It remains to show that $\| \mathbf { u } - \mathbf { v } \| _ { 2 } \leq d _ { \angle } ( \mathbf { u } , \mathbf { v } )$ . Let $\theta \ : = \ :$ $d _ { \angle } ( { \bf u } , { \bf v } ) \in [ 0 , \pi ]$ . Since u and v are unit vectors,

$$
\| \mathbf { u } - \mathbf { v } \| _ { 2 } = { \sqrt { 2 - 2 \mathbf { u } ^ { \top } \mathbf { v } } } = { \sqrt { 2 - 2 \cos \theta } } = 2 \sin { \frac { \theta } { 2 } } .
$$

For any $\theta \in [ 0 , \pi ]$ , we have 2 sin ${ \frac { \theta } { 2 } } ~ \leq ~ \theta$ . Therefore, $\lVert \mathbf { u } - \mathbf { \rho } _ { \mathrm { ~ } }$ $\mathbf { v } \| _ { 2 } \leq d _ { \angle } ( \mathbf { u } , \mathbf { \bar { v } } )$ . Substituting this inequality into the result of Theorem 2 yields

$$
\exp \left( \varepsilon \| \mathbf { u } - \mathbf { v } \| _ { 2 } \right) \leq \exp \left( \varepsilon d _ { \angle } ( \mathbf { u } , \mathbf { v } ) \right) .
$$

Hence,

$$
\begin{array} { r } { \mathrm { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { u } ) \in S ] \leq \exp ( \varepsilon d _ { \angle } ( \mathbf { u } , \mathbf { v } ) ) \qquad } \\ { \mathbf { \nabla } \cdot \mathrm { P r } [ \mathcal { M } _ { \mathrm { v M F } } ( \mathbf { v } ) \in S ] . } \end{array}
$$

Thus, the mechanism satisfies εd<sub>∠</sub>-metric differential privacy. □

Proof of Theorem 1. Since $\mathbf { g } = \lVert \mathbf { g } \rVert _ { 2 } \mathbf { u }$ and $\tilde { \mathbf { g } } = \lVert \mathbf { g } \rVert _ { 2 } \tilde { \mathbf { u } }$ the perturbation noise can be written as

$$
\begin{array} { r } { \mathbf { n } = \tilde { \mathbf { g } } - \mathbf { g } = \| \mathbf { g } \| _ { 2 } ( \tilde { \mathbf { u } } - \mathbf { u } ) . } \end{array}
$$

Therefore, $\lVert \mathbf { n } \rVert _ { 2 } = \lVert \mathbf { g } \rVert _ { 2 } \lVert \tilde { \mathbf { u } } - \mathbf { u } \rVert _ { 2 }$ . Because both u and u˜ are unit vectors, their Euclidean distance is at most 2:

$$
\begin{array} { r } { \| \tilde { \mathbf { u } } - \mathbf { u } \| _ { 2 } \leq \| \tilde { \mathbf { u } } \| _ { 2 } + \| \mathbf { u } \| _ { 2 } = 2 . } \end{array}
$$

Thus, $\| \mathbf { n } \| _ { 2 } \leq 2 \| \mathbf { g } \| _ { 2 }$ . It follows that

$$
\mathrm { A S N R } = { \frac { \Vert \mathbf { g } \Vert _ { 2 } } { \Vert \mathbf { n } \Vert _ { 2 } } } \geq { \frac { \Vert \mathbf { g } \Vert _ { 2 } } { 2 \Vert \mathbf { g } \Vert _ { 2 } } } = { \frac { 1 } { 2 } } .
$$

The equality holds when $\| \tilde { \mathbf { u } } - \mathbf { u } \| _ { 2 } = 2$ , which occurs if and only if $\tilde { \mathbf { u } } = - \mathbf { u }$ . Hence, the bound is tight. □

![](images/8138c751ce47c9fbdf8bf0b17cdc44023efe1e0493c50aca8acd43c3efa03bf1.jpg)  
Figure 13: Illustration of the zero-gradient suffix induced by ∇� [�masking the last k target tokens in selective autoregressive supervision. The arrows indicate the direction of gradient propagation.

## B Details of Selective Autoregressive Supervision

## B.1 Last-k Tokens in Selective Autoregressive Supervision

We next discuss a practical implementation detail of SAS from the client’s perspective.

While Selective Autoregressive Supervision is designed to perturb the supervision structure exposed through the split-interface gradient, its implementation should also preserve the stealthiness of the defense.

In particular, if the last k consecutive target tokens are all excluded from the loss computation, then the corresponding gradient signals on a contiguous block of k token representations in $x _ { \mathrm { t r k } }$ become identically zero. Such a structured zero-gradient suffix can serve as an explicit artifact of the defense and may reveal the existence of selective masking to an adversarial server.

To see this more formally, consider the autoregressive loss with a binary supervision mask $m _ { t } \in \{ 0 , 1 \}$ :

$$
\mathcal { L } _ { M } = \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t = 1 } ^ { T } m _ { t } \ell _ { t } , \ell _ { t } = - \log p _ { t } ( y _ { t } ) ,\tag{30}
$$

where $p _ { t }$ denotes the predictive distribution for the target token $y _ { t } .$ . Let $h _ { i }$ denote the split-interface representation at position i, i.e., the i-th token representation in $x _ { \mathrm { t r k } }$ . Due to the causal structure of autoregressive decoding, the representation $h _ { i }$ can only influence the losses of future or aligned target positions. Abstractly, we may write

$$
\frac { \partial \mathcal { L } _ { M } } { \partial h _ { i } } = \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t \in \mathcal { D } ( i ) } m _ { t } \frac { \partial \ell _ { t } } { \partial h _ { i } } ,\tag{31}
$$

where $\mathcal { D } ( i )$ denotes the set of supervised prediction positions whose computation depends on $h _ { i } .$ . For the last few token positions, this dependency set contains only a small number of subsequent losses. Therefore, if the last k target positions are all masked out, namely $m _ { T - k + 1 } =$ $m _ { T - k + 2 } = \cdot \cdot \cdot = m _ { T } = 0$ , then for the corresponding suffix positions whose only downstream supervised losses lie in this masked region, we obtain $\begin{array} { r } { \frac { \partial \mathcal { L } _ { M } } { \partial h _ { i } } = \bar { 0 } } \end{array}$ . Consequently, the gradient returned to the server-side Trunk segment contains a contiguous zero-gradient block over the suffix of $x _ { \mathrm { t r k } }$ . This phenomenon is illustrated in Fig. 13.

![](images/099d45aef76d131d8f143315eb543d2ebb7cba13ec565ac4c90ee14e2c07bff6.jpg)  
Figure 14: Training dynamics ofthe learning-based inversion decoder acrossfive $L L M s .$ Shown are the training loss, evaluation loss, and evaluation RougeL-F over training steps under three random seeds.

This artifact is undesirable because it weakens the concealment of the defense. From the client’s perspective, the goal of SAS is not only to introduce a supervision mismatch against gradient-matching attacks, but also to avoid leaving obvious structural signatures in the exposed gradient. A consecutive zero-gradient suffix is particularly problematic because it is independent of token content and can be detected directly from the norm pattern of the interface gradient. An adversary may observe that $\begin{array} { r } { \left\| \frac { \partial \mathcal { L } _ { M } } { \partial h _ { i } } \right\| _ { 2 } = 0 } \end{array}$ for several consecutive positions near the end of the sequence, and infer that these positions were systematically removed from supervision.

To avoid this issue, SAS should ensure that the last k token positions are not simultaneously excluded from the masked loss. Equivalently, the supervision mask should satisfy $\begin{array} { r } { \sum _ { t = T - k + 1 } ^ { T } m _ { t } \geq 1 } \end{array}$ . This constraint guarantees that at least one suffix target token contributes to the loss, thereby preventing a fully zero-gradient suffix from appearing in $x _ { \mathrm { t r k } }$ . To prevent such a zero-gradient suffix, we enforce the last k tokens to be valid supervised positions in our implementation. We set k = 5 throughout our experiments.

## C More Experimental Details

## C.1 Definitions of Gradient Utility Metrics

To quantify the utility of privatized gradients, we use four complementary metrics: the Amplitude Signal-to-Noise Ratio (ASNR), ASNR@10%, Recall@10% (R@10%), and Jaccard@10% (J@10%). Let $g \in \mathbb { R } ^ { L D }$ denote the original gradient and $\tilde { g } \in \mathbb { R } ^ { L D }$ the privatized gradient. The injected noise is defined as $n = \tilde { g } - g$

ASNR. The Amplitude Signal-to-Noise Ratio measures the overall gradient magnitude relative to the perturbation magnitude:

$$
\mathrm { A S N R } ( g , \tilde { g } ) = \frac { \| g \| _ { 2 } } { \| n \| _ { 2 } } .
$$

ASNR@10%. Let $S _ { 1 0 } ( g )$ denote the index set of the top 10% entries of $g$ ranked by absolute value. ASNR@10% is computed on this subset:

$$
\mathrm { A S N R @ 1 0 \% } ( g , \tilde { g } ) = \frac { \| g _ { S _ { 1 0 } ( g ) } \| _ { 2 } } { \| n _ { S _ { 1 0 } ( g ) } \| _ { 2 } } .
$$

Recall@10%. Let $S _ { 1 0 } ( g )$ and $S _ { 1 0 } ( \tilde { g } )$ be the top 10% index sets of $g$ and ${ \tilde { g } } ,$ respectively. Recall@10% measures the fraction of salient coordinates in the original gradient that are retained after privatization:

$$
\mathrm { R @ 1 0 9 7 } _ { 0 } ( g , \tilde { g } ) = \frac { | S _ { 1 0 } ( g ) \cap S _ { 1 0 } ( \tilde { g } ) | } { | S _ { 1 0 } ( g ) | } .
$$

Jaccard@10%. Jaccard@10% measures the overlap between the two top-10% index sets:

$$
\mathrm { J @ 1 0 } \% ( g , \tilde { g } ) = \frac { \vert S _ { 1 0 } ( g ) \cap S _ { 1 0 } ( \tilde { g } ) \vert } { \vert S _ { 1 0 } ( g ) \cup S _ { 1 0 } ( \tilde { g } ) \vert } .
$$

ASNR and ASNR@10% evaluate magnitude preservation, while R@10% and J@10% measure support preservation.

## C.2 Training Details of the Learning-based Inversion Decoder

Here we present the training details of the learning-based inversion decoder used in the semi-white-box Forward Inversion Paradigm (SIP) (Chen et al. 2024), which corresponds to the “SIP-only” setting in our experiments. The goal of this decoder is to directly reconstruct the original input token sequence from the smashed data produced by the client-side Bottom segment.

![](images/77493784322a06d366d9cd9e4229e4d81282a5106e3da5c2544336590be904ad.jpg)  
Figure 15: Training Curves of Gradient Pruning and Gradient Dropout.

<table><tr><td>Dataset</td><td>Method</td><td>Final PPL</td><td>PPL@500steps</td><td>PPL@1000steps</td><td>RougeL-F</td><td>Rouge1-F</td><td>Rouge2-F</td><td>Meteor</td><td>TRR</td></tr><tr><td rowspan="9">Code- Alpaca</td><td>Grad Pruning</td><td>5.292±.538</td><td>4.595±.001</td><td>4.934±.017</td><td>.995±.001</td><td>.995±.001</td><td>.981±.001</td><td>.987±.001</td><td>.990±.000</td></tr><tr><td>GradSeq-LDP(10240)</td><td>4.670±.002</td><td>4.962±.055</td><td>4.782±.043</td><td>.564±.032</td><td>.567±.037</td><td>.413±.020</td><td>.570±.016</td><td>.634±.007</td></tr><tr><td>GradSeq-LDP(12288)</td><td>4.611±.037</td><td>4.915±.022</td><td>4.804±.050</td><td>.657±.029</td><td>.661±.030</td><td>.517±.042</td><td>.664±.042</td><td>.717±.041</td></tr><tr><td>Grad Dropout</td><td>4.782±.104</td><td>5.341±.158</td><td>4.954±.041</td><td>.942±.008</td><td>.942±.008</td><td>.915±.012</td><td>.940±.007</td><td>.943±.013</td></tr><tr><td>Gradient Mirage(512)</td><td>4.451±.001</td><td>4.671±.022</td><td>4.610±.008</td><td>.202±.002</td><td>.208±.003</td><td>.108±.001</td><td>.226±.000</td><td>.255±.002</td></tr><tr><td>Gradient Mirage(1024)</td><td>4.421±.008</td><td>4.603±.008</td><td>4.552±.013</td><td>.351±.022</td><td>.361±.022</td><td>.177±.020</td><td>.341±.017</td><td>.371±.009</td></tr><tr><td>Top-Only Training</td><td>4.831±.005</td><td>5.190±0.006</td><td>4.970±.004</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td>4.444±.004</td><td>4.504±.012</td><td>4.447±.005</td><td>1.000±.000</td><td>1.000±.000</td><td>.989±.002</td><td>.991±.002</td><td>.992±.001</td></tr><tr><td rowspan="8">GradSeq-LDP(10240)</td><td>Grad Pruning</td><td>11.500±2.161</td><td>9.229±.598</td><td>13.624±.095</td><td>.959±.011</td><td>.959±.011</td><td>.908±.023</td><td>.935±.017</td><td>.938±.013</td></tr><tr><td></td><td>7.738±.056</td><td>7.443±.025</td><td>7.212±.017</td><td>.799±.034</td><td>.804±.033</td><td>.613±.039</td><td>.760±.048</td><td>.758±.028</td></tr><tr><td>GradSeq-LDP(12288)</td><td>7.202±.019</td><td>7.726±.040</td><td>7.416±.021</td><td>.805±.033</td><td>.810±.032</td><td>.626±.056</td><td>.769±.049</td><td>.762±.039</td></tr><tr><td>Grad Dropout</td><td>7.758±.029</td><td>9.111±.265</td><td>8.316±.145</td><td>.950±.006</td><td>.950±.006</td><td>.867±.002</td><td>.912±.003</td><td>.906±.008</td></tr><tr><td>Gradient Mirage(512)</td><td>7.404±.020</td><td>8.120±.042</td><td>7.894±.100</td><td>.209±.017</td><td>.215±.017</td><td>.040±.008</td><td>.137±.013</td><td>.183±.013</td></tr><tr><td>Gradient Mirage(1024) Top-Only Training</td><td>7.191±.033 8.874±.018</td><td>7.824±.043 9.613±.027</td><td>7.456±.035</td><td>.448±.020</td><td>.455±.019</td><td>.145±.014</td><td>.306±.025</td><td>.374±.007</td></tr><tr><td>Standard Training</td><td>7.396±.013</td><td>7.338±.036</td><td>9.269±.029 7.106±.017</td><td>.993±.004</td><td>.993±.004</td><td>.969±.010</td><td>.975±.007</td><td>.978±.005</td></tr><tr><td rowspan="8">GSM8K</td><td></td><td>12.227±1.131</td><td>10.769±.059</td><td>10.973±.301</td><td>.997±.003</td><td></td><td></td><td></td><td></td></tr><tr><td>Grad Pruning GradSeq-LDP(10240)</td><td>11.038±.025</td><td>11.712±.152</td><td></td><td>.904±.014</td><td>.997±.006</td><td>.994±.003</td><td>.996±.012</td><td>.996±.002</td></tr><tr><td></td><td>11.122±.039</td><td>11.669±.114</td><td>11.431±.005 11.423±.058</td><td>.885±.019</td><td>.904±.014 .886±.018</td><td>.843±.013</td><td>.923±.007 .910±.013</td><td>.914±.007 .901±.013</td></tr><tr><td>GradSeq-LDP(12288)</td><td>12.477±.972</td><td>12.556±.153</td><td>12.365±.172</td><td>.949±.010</td><td>.949±.011</td><td>.821±.024 .904±.015</td><td>.940±.010</td><td>.927±.011</td></tr><tr><td>Grad Dropout Gradient Mirage(512)</td><td>10.643±.031</td><td>10.958±.051</td><td>10.806±.055</td><td>.112±.008</td><td>.119±.007</td><td>.011±.005</td><td>.070±.006</td><td>.079±.008</td></tr><tr><td>Gradient Mirage(1024)</td><td>10.500±.011</td><td>10.710±.042</td><td>10.587±.034</td><td>.363±.012</td><td>.387±.009</td><td>.087±.007</td><td>.213±.013</td><td>.243±.017</td></tr><tr><td>Top-Only Training</td><td>11.999±.007</td><td>12.283±.005</td><td>12.075±.004</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td>10.588±.003</td><td>10.471±.059</td><td>10.416±.016</td><td>1.000±.000</td><td>1.000±.000</td><td>.993±.001</td><td>.995±.001</td><td>.996±.000</td></tr></table>

Table 8: Comparison between Gradient Mirage and existing privacy-preserving methods with Llama3-8B.

Following the setting of the original work, we use the open-source CNN-DailyMail News Text Summarization dataset as the auxiliary training data (See, Liu, and Manning 2017). Specifically, the training data is derived from the CNN-DailyMail news summarization corpus.

The inversion decoder takes the smashed data generated by the Bottom segment as input. In our setting, the Bottom segment contains 6 decoder layers. Given the hidden representations with embedding dimension D, the inverter uses a single-layer unidirectional GRU (Cho et al. 2014) with hidden size 4096 to model the sequential dependency of the intermediate activations. The GRU output at each token position is then passed through a linear projection layer, which maps the hidden state to the vocabulary space and produces token-level logits. A dropout rate of 0.1 is applied after the GRU layer during training.

The inverter is trained in an auto-encoder-style manner. The Bottom segment of the target language model is used as a frozen encoder to generate intermediate representations from the auxiliary text data, while only the inversion decoder is optimized. The training objective is token-level crossentropy loss between the decoder output and the original input token sequence. In this way, the decoder learns to map the smashed data back to the corresponding textual tokens.

We train the inversion decoder for 3 epochs with a batch size of 32. The model is optimized using AdamW with a learning rate of 1e-3 and a weight decay of 1e-5. After training, the learned inverter can be directly applied to the smashed data observed during split fine-tuning to produce an initial reconstruction of the client’s private input sequence. Fig. 14 illustrates the training dynamics of the decoder on five LLMs (Touvron et al. 2023; Grattafiori et al. 2024; Yang et al. 2024, 2025; Bi et al. 2024).

![](images/7c9cbc97f537aaae662d9672fbbc4ebcf1d4a817bbde366eb580c54ffa2634d0.jpg)  
Figure 16: Comparison between the training dynamics with and without Trunk segment updates on Llama3-8B across three datasets. Updating the Trunk segment generally leads to lower validation perplexity across random seeds.

<table><tr><td>Dataset</td><td>Method</td><td>Final PPL</td><td>PPL@500steps</td><td>PPL@1000steps</td><td>RougeL-F</td><td>Rouge1-F</td><td>Rouge2-F</td><td>Meteor</td><td>TRR</td></tr><tr><td rowspan="9">Code- Alpaca</td><td>Grad Pruning</td><td>5.000±.089</td><td>4.864±.088</td><td>4.765±.044</td><td>.993±.003</td><td>.993±.003</td><td>.988±.006</td><td>.989±.005</td><td>.995±.003</td></tr><tr><td>GradSeq-LDP(10240)</td><td>4.804±.083</td><td>5.141±.149</td><td>4.991±.155</td><td>.775±.033</td><td>.776±.034</td><td>.656±.048</td><td>.741±.043</td><td>.886±.018</td></tr><tr><td>GradSeq-LDP(12288)</td><td>4.853±.131</td><td>5.132±.143</td><td>5.024±.172</td><td>.820±.037</td><td>.821±.037</td><td>.725±.052</td><td>.795±.043</td><td>.909±.018</td></tr><tr><td>Grad Dropout</td><td>6.979±1.575</td><td>7.329±.929</td><td>82.223±129.115</td><td>.902±.017</td><td>.902±.016</td><td>.859±.026</td><td>.890±.016</td><td>.932±.008</td></tr><tr><td>Gradient Mirage(512)</td><td>4.817±.098</td><td>5.186±.124</td><td>5.066±.068</td><td>.095±.012</td><td>.097±.012</td><td>.016±.002</td><td>.112±.005</td><td>.199±.010</td></tr><tr><td>Gradient Mirage(1024)</td><td>4.772±.049</td><td>5.031±.136</td><td>5.061±.124</td><td>.174±.020</td><td>.177±.019</td><td>.042±.007</td><td>.146±.008</td><td>.297±.004</td></tr><tr><td>Top-Only Training</td><td>4.961±.040</td><td>5.231±.158</td><td>5.025±.083</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td>4.731±.018</td><td>4.776±.108</td><td>4.669±.035</td><td>.990±.003</td><td>.990±.003</td><td>.981±.006</td><td>.982±.005</td><td>.992±.003</td></tr><tr><td>Grad Pruning</td><td>6.501±.615</td><td>5.849±.055</td><td>5.576±.188</td><td>.996±.004</td><td>.996±.004</td><td>.991±.009</td><td>.995±.005</td><td>.997±.003</td></tr><tr><td rowspan="8">PIQA</td><td>GradSeq-LDP(10240)</td><td>6.302±.656</td><td>6.325±.248</td><td>6.280±.208</td><td>.792±.038</td><td>.793±.038</td><td>.635±.057</td><td>.744±.043</td><td>.861±.008</td></tr><tr><td>GradSeq-LDP(12288)</td><td>5.851±.049</td><td>6.147±.067</td><td>6.172±.256</td><td>.826±.017</td><td>.826±.017</td><td>.691±.026</td><td>.786±.023</td><td>.883±.005</td></tr><tr><td>Grad Dropout</td><td>7.757±1.772</td><td>11.069±.524</td><td>8.452±1.022</td><td>.813±.020</td><td>.815±.017</td><td>.743±.026</td><td>.798±.019</td><td>.859±.008</td></tr><tr><td>Gradient Mirage(512)</td><td>5.631±.130</td><td>5.770±.095</td><td>5.767±.052</td><td>.047±.015</td><td>.047±.016</td><td>.001±.002</td><td>.046±.001</td><td>.115±.007</td></tr><tr><td>Gradient Mirage(1024)</td><td>5.561±.232</td><td>5.748±.061</td><td>5.607±.065</td><td>.102±.012</td><td>.103±.011</td><td>.016±.003</td><td>.088±.013</td><td>.203±.016</td></tr><tr><td>Top-Only Training</td><td>6.272±.059</td><td>6.661±.129</td><td>6.465±.208</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td>5.285±.095</td><td>5.380±.113</td><td>5.339±.060</td><td>.994±.006</td><td>.994±.006</td><td>.988±.012</td><td>.989±.011</td><td>.994±.006</td></tr><tr><td>Grad Pruning</td><td>10.659±.230</td><td>10.520±.089</td><td>10.498±.164</td><td>.978±.005</td><td>.978±.005</td><td>.970±.005</td><td></td><td></td></tr><tr><td rowspan="8">GSM8K</td><td>GradSeq-LDP(10240)</td><td>10.487±.152</td><td>10.621±.052</td><td>10.337±.134</td><td>.375±.021</td><td>.390±.019</td><td>.203±.019</td><td>.982±.002</td><td>.991±.001</td></tr><tr><td>GradSeq-LDP(12288)</td><td>10.483±.143</td><td>10.608±.062</td><td>10.338±.134</td><td></td><td></td><td></td><td>.333±.013</td><td>.629±.014</td></tr><tr><td></td><td>107.082±162.999</td><td></td><td>73.177±103.329</td><td>.440±.028</td><td>.456±.027</td><td>.264±.028</td><td>.404±.029</td><td>.680±.016</td></tr><tr><td>Grad Dropout</td><td>10.430±.159</td><td>48.571±61.245 10.760±.207</td><td>10.302±.055</td><td>.897±.011</td><td>.897±.011 .031±.009</td><td>.880±.004</td><td>.921±.003</td><td>.947±.005</td></tr><tr><td>Gradient Mirage(512)</td><td>10.142±.169</td><td>10.387±.257</td><td>10.237±.300</td><td>.031±.009 .075±.012</td><td>.078±.012</td><td>.002±.001 .005±.002</td><td>.042±.006 .071±.011</td><td>.082±.012 .154±.024</td></tr><tr><td>Gradient Mirage(1024) Top-Only Training</td><td>11.198±.279</td><td>11.644±.090</td><td>11.295±.172</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td>9.714±.138</td><td>9.934±.289</td><td>9.786±.168</td><td>.977±.001</td><td>.977±.001</td><td>.968±.001</td><td>.981±.000</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.991±.000</td></tr></table>

Table 9: Comparison between Gradient Mirage and existing privacy-preserving methods with Llama2-7B.

## C.3 Training Curves of GP and GD

Fig. 15 illustrates the training curves of GP and GD. It can be observed that, to achieve strong privacy protection with GP or GD, the training process becomes extremely unstable, making it challenging to achieve a favorable privacy-utility trade-off.

## C.4 Ablation Details of Trunk Training

Fig. 16 illustrates the differences in training dynamics with and without Trunk segment updates, evaluated on Llama3- 8B across three datasets. For each dataset, the upper panel corresponds to training with Trunk segment updates, which consistently yields better performance.

## C.5 Extended Defense Experiments

In the main text, we evaluate defense performance using only ROUGE-L-F and METEOR. Here, we provide additional results for ROUGE-1-F, ROUGE-2-F, and Token Recovery Rate (TRR), as well as results over more training steps. Tab. 8, 9, and 10 report the results for Llama-3-8B, Llama-2-7B, and DeepSeek-LLM-7B, respectively. We further report the defense performance against the BiSR(b) paradigm when cosine distance is used as the matching objective, as shown in Tab. 11.

## C.6 Robust Training under Extreme Scale Blinding

While the default mean scale m = 1750 provides a favorable privacy-utility trade-off, we further investigate whether Gradient Mirage remains trainable under extremely large scaling factors. As m becomes excessively large, the amplified gradients introduced by Scale Blinding may cause optimization instability, particularly for the Trunk segment whose gradients are directly affected by the perturbed objective.

To address this issue, we adopt a simple Trunk-Frozen Training strategy: the gradients of the Trunk segment are discarded, while the Top segment continues to be optimized normally and the Bottom segment is updated using the recovered gradients produced by Bottom-Gradient Recovery. Since the recovered Bottom gradients remove the expected amplification introduced by Scale Blinding, they remain relatively stable and informative for optimization.

We evaluate this strategy under significantly larger mean scales m $\in \{ 4 \times 1 0 ^ { 3 } , 1 0 ^ { 4 } , \overset { \smile } { 2 } . 5 \times 1 0 ^ { 4 } , \overset { \smile } { 5 } \times 1 0 ^ { 4 } , 1 \overset { \smile } { 0 } ^ { 5 } \}$ . As shown in Fig. 17, the proposed training strategy maintains stable convergence across all tested scales and consistently outperforms Top-Only Training.

<table><tr><td>Dataset</td><td>Method</td><td>Final PPL</td><td>PPL@500steps</td><td>PPL@1000steps</td><td>RougeL-F</td><td>Rouge1-F</td><td>Rouge2-F</td><td>Meteor</td><td>TRR</td></tr><tr><td rowspan="7">Code- Alpaca</td><td>Grad Pruning</td><td>5.389±.050</td><td>5.305±.027</td><td>5.475±.094</td><td>.991±.001</td><td>.991±.001</td><td>.968±.004</td><td>.967±.001</td><td>.976±.002</td></tr><tr><td>GradSeq-LDP(8192)</td><td>5.045±.049</td><td>5.319±.125</td><td>5.358±.099</td><td>.598±.015</td><td>.605±.015</td><td>.376±.028</td><td>.540±.026</td><td>.738±.015</td></tr><tr><td>GradSeq-LDP(10240)</td><td>5.000±.098</td><td>5.327±.100</td><td>5.252±.160</td><td>.741±.020</td><td>.746±.018</td><td>.637±.024</td><td>.709±.030</td><td>.844±.014</td></tr><tr><td>Grad Dropout</td><td>6.328±.333</td><td>6.371±.575</td><td>7.167±1.046</td><td>.681±.040</td><td>.682±.040</td><td>.584±.041</td><td>.682±.041</td><td>.718±.044</td></tr><tr><td>Gradient Mirage(512)</td><td>5.304±.223</td><td>5.397±.158</td><td>5.404±.028</td><td>.070±.023</td><td>.071±.024</td><td>.030±.013</td><td>.045±.015</td><td>.163±.030</td></tr><tr><td>Gradient Mirage(1024)</td><td>5.149±.135</td><td>5.260±.160</td><td>5.292±.024</td><td>.105±.007</td><td>.107±.008</td><td>.028±.008</td><td>.062±.009</td><td>.204±.011</td></tr><tr><td>Top-Only Training</td><td>5.438±.121</td><td>5.801±.042</td><td>5.636±.045</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="10">PIQA</td><td>Standard Training</td><td>4.378±.001</td><td>4.495±.001</td><td>4.418±.001</td><td>.998±.000</td><td>.998±.000</td><td>.984±.001</td><td>.977±.001</td><td>.983±.001</td></tr><tr><td>Grad Pruning</td><td>14.387±2.094</td><td>13.592±.867</td><td>14.741±1.939</td><td>.942±.016</td><td>.944±.015</td><td>.882±.026</td><td>.928±.024</td><td>.931±.019</td></tr><tr><td>GradSeq-LDP(8192)</td><td>6.710±.235</td><td>6.921±.032</td><td>6.798±.105</td><td>.819±.020</td><td>.822±.019</td><td>.659±.039</td><td>.763±.027</td><td>.831±.005</td></tr><tr><td>GradSeq-LDP(10240)</td><td>6.717±.274</td><td>6.870±.022</td><td>6.749±.073</td><td>.845±.010</td><td>.848±.009</td><td>.694±.008</td><td>.712±.019</td><td>.827±.014</td></tr><tr><td>Grad Dropout</td><td>301.776±415.971</td><td>11.223±2.122</td><td>118.565±154.226</td><td>.844±.013</td><td>.854±.024</td><td>.773±.017</td><td>.820±.025</td><td>.848±.008</td></tr><tr><td>Gradient Mirage(512)</td><td>6.673±.164</td><td>6.917±.162</td><td>6.923±.295</td><td>.071±.005</td><td>.072±.005</td><td>.008±.005</td><td>.036±.004</td><td>.135±.009</td></tr><tr><td>Gradient Mirage(1024)</td><td>6.451±.015</td><td>7.089±.081</td><td>6.940±.195</td><td>.142±.009</td><td>.148±.010</td><td>.021±.008</td><td>.081±.014</td><td>.221±.012</td></tr><tr><td>Top-Only Training</td><td>7.444±.130 5.794±.008</td><td>7.935±.064</td><td>7.688±.008</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td></td><td>6.022±.002</td><td>5.870±.016</td><td>.999±.002</td><td>.999±.002</td><td>.979±.009</td><td>.968±.012</td><td>.966±.007</td></tr><tr><td rowspan="8">GSM8K</td><td>Grad Pruning</td><td>39.113±35.004</td><td>7.430±.183</td><td>7.766±.136</td><td>.792±.013</td><td>.796±.015</td><td>.695±.024</td><td>.753±.029</td><td>.792±.024</td></tr><tr><td>GradSeq-LDP(8192)</td><td>12.126±.119</td><td>13.091±.384</td><td>12.667±.296</td><td>.312±.017</td><td>.332±.015</td><td>.131±.023</td><td>.214±.027</td><td>.475±.009</td></tr><tr><td>GradSeq-LDP(10240)</td><td>12.130±.115</td><td>12.981±.326</td><td>12.507±.313</td><td>.344±.014</td><td>.364±.018</td><td>.156±.021</td><td>.241±.025</td><td>.513±.008</td></tr><tr><td>Grad Dropout</td><td>16.104±3.461</td><td>17.533±3.572</td><td>17.125±3.352</td><td>.843±.024</td><td>.852±.025</td><td>.784±.040</td><td>.848±.030</td><td>.909±.011</td></tr><tr><td>Gradient Mirage(512)</td><td>11.989±.105</td><td>12.796±.149</td><td>12.327±.165</td><td>.028±.002</td><td>.030±.001</td><td>.000±.000</td><td>.012±.001</td><td>.078±.013</td></tr><tr><td>Gradient Mirage(1024)</td><td>12.037±.294</td><td>13.006±.181</td><td>12.659±.100</td><td>.074±.005</td><td>.077±.006</td><td>.004±.000</td><td>.032±.003</td><td>.163±.012</td></tr><tr><td>Top-Only Training</td><td>13.719±.105</td><td>14.407±.478</td><td>13.873±.390</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Training</td><td>10.007±.002</td><td>10.173±.021</td><td>10.094±.008</td><td>.978±.002</td><td>.978±.002</td><td>.962±.003</td><td>.976±.001</td><td>.986±.001</td></tr></table>

Table 10: Comparison between Gradient Mirage and existing privacy-preserving methods with DeepSeek-LLM-7B.

![](images/6b14f673763404cc4c2e5bb777cfef81d4068aa9b09ba8c826d5c19f7c8f53b2.jpg)

![](images/b69cb48cd3d0b12300caf02b1d7d45d6a12db9798a73fdc599edec00e3c9a5d6.jpg)

![](images/e43b1d152188b20e55f4e0a3ef9c22cb5400d064533c9258cc810c51e58a7719.jpg)

![](images/f914df82c9fcc3c9de466ec8fc9ac34541d69fb95ed05bd8eccda533bc54b598.jpg)

![](images/87114a2e0e9ecf6bbf133d19ce9c6d2689a26c27d2baa37ba0f3dffd70ae8119.jpg)

Figure 17: Fine-tuning performance under extremely large mean scales m with the Trunk-Frozen Training strategy, evaluated on Llama-3-8B with the CodeAlpaca dataset. The proposed strategy maintains stable convergence across all tested scales and random seeds.
<table><tr><td colspan="2">Dataset</td><td colspan="3">CodeAlpaca</td><td colspan="3">PIQA</td><td colspan="3">GSM8K</td></tr><tr><td>Model</td><td>Method</td><td>RougeL-F</td><td>Rouge2-F</td><td>Meteor</td><td>RougeL-F</td><td>Rouge2-F</td><td>Meteor</td><td>RougeL-F</td><td>Rouge2-F</td><td>Meteor</td></tr><tr><td rowspan="4">Llama3 -8B</td><td>GP(0.7) GD(0.6, 5e-3)</td><td>.488±.006 .089±.016</td><td>.181±.002 .009±.009</td><td>.347±.002 .061±.010</td><td>.691±.003 .237±.010</td><td>.444±.007 .088±.012</td><td>.601±.001 .186±.007</td><td>.473±.008 .066±.012</td><td>.179±.012 .000±.000</td><td>.307±.011 .039±.003</td></tr><tr><td>LDP(10240) LDP(12288)</td><td>.103±.012 .132±.005</td><td>.010±.005 .018±.007</td><td>.071±.010 .093±.004</td><td>.309±.007 .322±.012</td><td>.106±.003 .114±.005</td><td>.234±.008 .247±.011</td><td>.073±.006 .080±.002</td><td>.003±.002 .003±.001</td><td>.046±.003 .046±.003</td></tr><tr><td>Standard</td><td>.557±.012</td><td>.254±.015</td><td>.416±.013</td><td>.716±.007</td><td>.448±.009</td><td>.625±.009</td><td>.469±.016</td><td>.177±.016</td><td>.317±.016</td></tr><tr><td>Ours(512)</td><td>.094±.005</td><td>.002±.001</td><td>.068±.003</td><td>.121±.017</td><td>.013±.002</td><td>.082±.004</td><td>.103±.004</td><td>.003±.001</td><td>.066±.005</td></tr><tr><td rowspan="4">Llama2</td><td>Ours(1024) GP(0.7)</td><td>.161±.011</td><td>.008±.002</td><td>.106±.003</td><td>.246±.019</td><td>.040±.009</td><td>.156±.012</td><td>.136±.015</td><td>.012±.002</td><td>.105±.010</td></tr><tr><td>GD(0.6, 5e-3)</td><td>.478±.017 .016±.007</td><td>.285±.029 .001±.001</td><td>.416±.027 .008±.003</td><td>.629±.017 .037±.014</td><td>.464±.021 .003±.002</td><td>.573±.022 .021±.008</td><td>.393±.015 .012±.001</td><td>.202±.015 .000±.000</td><td>.334±.016</td></tr><tr><td>LDP(10240)</td><td>.092±.017</td><td>.016±.003</td><td>.054±.007</td><td>.225±.010</td><td>.069±.011</td><td>.152±.016</td><td>.026±.002</td><td>.002±.000</td><td>.008±.001 .015±.001</td></tr><tr><td rowspan="2">LDP(12288)</td><td>.103±.011</td><td>.019±.004</td><td>.061±.007</td><td>.235±.016</td><td>.077±.014</td><td>.159±.019</td><td>.033±.002</td><td>.003±.001</td><td>.019±.003</td></tr><tr><td>Standard Ours(512)</td><td>.471±.008</td><td>.287±.012 .414±.016</td><td>.617±.022</td><td>.444±.021</td><td>.570±.012</td><td></td><td>.420±.004 .226±.005</td><td>.371±.008</td></tr><tr><td rowspan="3">DeepSeek</td><td>Ours(1024)</td><td>.066±.009 .110±.017</td><td>.006±.006 .018±.007</td><td>.041±.006 .066±.003</td><td>.085±.007 .131±.005</td><td>.007±.005 .014±.005</td><td>.061±.004 .082±.011</td><td>.046±.006 .072±.007</td><td>.001±.001 .003±.002</td><td>.030±.003 .040±.001</td></tr><tr><td>GP(0.8)</td><td>.381±.018</td><td>.192±.014</td><td>.289±.022</td><td>.549±.021</td><td>.347±.024</td><td>.446±.023</td><td>.361±.009</td><td>.175±.007</td><td></td></tr><tr><td>GD(0.6, 5e-3) LDP(8192)</td><td>.002±.001</td><td>.000±.000</td><td>.002±.001</td><td>.001±.001</td><td>.000±.000</td><td>.002±.001</td><td>.003±.001</td><td>.000±.000</td><td>.249±.008 .002±.000</td></tr><tr><td rowspan="4">-LLM-7B</td><td>LDP(10240)</td><td>.033±.002</td><td>.003±.001</td><td>.016±.001</td><td>.180±.011</td><td>.051±.011</td><td>.109±.005</td><td>.020±.003</td><td>.001±.000</td><td>.009±.002</td></tr><tr><td>Standard</td><td>.032±.002 .434±.017</td><td>.003±.001</td><td>.015±.001</td><td>.187±.010</td><td>.057±.007</td><td>.113±.005</td><td>.025±.005</td><td>.000±.000</td><td>.011±.002</td></tr><tr><td>Ours(512)</td><td></td><td>.250±.015</td><td>.358±.013</td><td>.644±.006</td><td>.473±.023</td><td>.576±.022</td><td>.383±.022</td><td>.211±.015</td><td>.307±.016</td></tr><tr><td>Ours(1024)</td><td>.015±.002</td><td>.001±.001</td><td>.007±.000</td><td>.072±.016</td><td>.005±.003</td><td>.034±.007</td><td>.018±.004</td><td>.000±.000</td><td>.007±.002</td></tr><tr><td rowspan="4"></td><td></td><td>.057±.008</td><td>.005±.001</td><td>.028±.003</td><td>.166±.020</td><td>.021±.007</td><td>.078±.012</td><td>.066±.005</td><td>.002±.001</td><td>.030±.004</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Comparison of defense performance against the BiSR(b) attack paradigm using cosine distance as the matching objective across three models.

![](images/ced5a501b14ed7a761baf8d114d10dd648970f6439ea29699af4a5f149dc2e9b.jpg)  
Figure 18: Qualitative comparison of reconstructed inputs under different privacy-preserving methods across Llama-3-8B, Llama-2-7B, and DeepSeek-LLM-7B. Compared with existing defenses, Gradient Mirage yields substantially more distorted and incoherent reconstructions.

![](images/860709361d72085e16b6877ac2b544cf52020c447690bfca0e13c12a753f3273.jpg)  
Figure 19: Qualitative results ofGMA-SL under instruction fine-tuning across CodeAlpaca, PIQA, and GSM8K. Although the query tokens are excluded from supervision during user-side training, GMA-SL successfully reconstructs the supervised answer portion.

## D Visualization of Gradient Mirage Performance

Fig. 18 presents representative defense performance across three LLMs. Compared with existing defense methods, Gradient Mirage produces substantially more distorted and incoherent reconstructions, revealing considerably less information about the ground-truth input. These examples visually demonstrate the effectiveness of Gradient Mirage in defending against GMA-SL.

## E Future Work

Beyond the settings evaluated in our main experiments, we additionally find that GMA-SL remains effective in the instruction fine-tuning setting. Under all attack settings considered for GMA-SL, the labels used in user-side training are constructed by left-shifting the entire sequence by one token, meaning that supervision is applied to all tokens. However, we further find that under instruction fine-tuning, where the query part is not supervised, GMA-SL still poses a substantial threat to the answer part.

GMA-SL matches the interface gradients computed under full-sequence supervision. However, we apply the same attack assumption to the instruction fine-tuning setting, where only the answer tokens are supervised. Let the sequence be decomposed as $x = ( x ^ { q } , x ^ { a } )$ , where $x ^ { q }$ and $x ^ { a }$ denote the query and answer parts, respectively. The optimization objective can be written as $\begin{array} { r } { \mathcal { L } _ { \mathrm { a n s } } = \sum _ { t \in \mathcal { T } _ { \mathfrak { a } } } \ell ( f _ { \theta } ( x ) _ { t } , y _ { t } ) } \end{array}$ , where $\mathcal { T } _ { a }$ is the set of positions belonging to the answer part. Even under this mismatch, namely, attacking the instruction finetuning scenario with the full-supervision assumption, we find that the reconstruction quality for the answer part remains remarkably high, whereas the reconstruction quality for the query part is relatively poor. Representative examples on CodeAlpaca, GSM8K, and PIQA are shown in Fig. 19.

Looking forward, we plan to extend Gradient Mirage to the study of gradient privacy attacks and defenses in multimodal large language models. Several deeper questions also warrant further investigation, including the underlying mechanisms of GMA-SL and, in particular, why it remains effective under instruction fine-tuning despite supervision being restricted to the answer portion. Moreover, unlike supervised learning, reinforcement learning does not rely on explicit token-level labels. In RL-based $\mathrm { S L }$ , it remains an open question how to formally define the privacy leakage of reward models, as well as how an adversary could effectively extract private information from reward models.