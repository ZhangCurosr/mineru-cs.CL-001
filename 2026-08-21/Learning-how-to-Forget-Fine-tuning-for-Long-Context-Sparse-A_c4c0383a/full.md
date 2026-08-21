# Learning how to Forget: Fine-tuning for Long-Context Sparse Attention

Matthias Seeger<sup>∗</sup> Amazon Web Services mseeger@gmail.com

Zeyu Zhang University of Amsterdam z.zhang2@uva.nl

Konstantinos Benidis Amazon Web Services kbenidis@amazon.de

Vihang Patil Amazon pvihang@amazon.de

Sebastian Schelter Technical University Berlin schelter@tu-berlin.de

August 21, 2026

## Abstract

A lot of prior work addressed key-value (KV) cache selection and compression by sparse attention to enable long-context inference for transformer language models without excessive hardware budgets. We provide a new method for fine-tuning models with sparse attention. It works for any KV cache policy, runs on a moderate hardware budget (e.g., a single Nvidia A100 GPU with 40 GB RAM), and allows the model to co-adapt with the policy, often outperforming models trained with exact attention (sequence parallelism). We also provide an eficient implementation of H2O sparse attention (the leading policy in our experiments) with dedicated scaled dot product attention kernel support. KeysAndValues, a new open source library for long-context inference and fine-tuning, provides easy-to-use and performant code for all methods discussed here.

## 1 Introduction

Modern large language models need to process very long contexts (i.e., number of tokens) for calling many tools with sizable outputs [47, 16], running chain of thought reasoning [62], or sustaining multi-turn conversations. While naive transformer implementations scale quadratically in compute and linearly in memory with context width, a lot of progress has been made on approximations with essentially linear time and constant memory scaling. A particularly fruitful direction is sparse attention, where key-value (KV) information is stored in a fixed-size KV cache, slots of which are evicted once it is full, and many diferent eviction policies have been proposed.

In this paper, we address the problem of how to post-train a transformer language model with sparse attention on a moderate hardware budget (our experiments are run computing gradients for a 4B weights model on a single<sup>1</sup> Nvidia A100 GPU with 40 GB RAM).

Our novel method works for any KV cache policy and requires no further approximations beyond sparse attention. As we demonstrate in experiments on a range of long-context benchmarks, our training algorithm allows the model to co-adapt with the KV cache policy, often outperforming models trained with exact attention (sequence parallelism). Moreover, our fine-tuning method runs on resources comparable to sparse attention inference. It can be combined with orthogonal KV cache compression strategies such as grouped query attention [1] or quantization [27, 40].

The heavy-hitter oracle (H2O) [72] is one of the most prominent sparse attention policies. We demonstrate several improvements to H2O, leading to a much more eficient implementation with dedicated scaled dot product attention (SDPA) kernel support. Variants of H2O outperform other KV cache policies in our experiments, and our fast implementation takes a big step towards latencies competitive with SotA inference libraries such as vLLM [30], which use context or sequence parallelism almost exclusively. In summary, our contributions are:

• New method for fine-tuning transformer language models with sparse attention and arbitrary KV cache policy in place. This method runs on resources comparable to sparse attention inference. It combines nested activation checkpointing and CPU ofloading with exploiting a linear KV cache bufer recurrence by way of autograd saved tensors packing. Our method can process sequences of arbitrary length with constant resources.

• Methodological and implementation improvements of heavy-hitter oracle (H2O) cache policy [72]. In particular, we provide Triton code to return summed attention weights alongside a FlashInfer SDPA kernel [68].

• A comprehensive evaluation on a range of long-context benchmarks. Our training algorithm often outperforms models trained with sequence parallelism [32] when sparse attention inference is used.

• KeysAndValues, a novel open source library for long-context inference and fine-tuning (https://github.com/awslabs/keys\_values).

## 2 Related Work

There is a large body of work on long-context inference by way of KV cache compression. A simple idea is to group heads, so that less key and value vectors need to be stored [50, 1, 6], or to impose a low-rank structure in the query-by-key matrix [14]. These are modifications to be used during pre-training already. Cache bufers can be quantized to 8 or 4 bits, or even below [27, 40, 75, 35, 52, 31, 54, 74]. Sparse attention is a powerful general idea (discussed in Section 3.1) with many instantiations. Big Bird [73] prescribes fixed attention sparsity patterns. The heavy-hitter oracle (H2O) [72] is discussed in Section 3.1.1. Q-Hitter [77] combines H2O with quantization, steering decisions by quantizability as well. SnapKV [36] uses summed attention weights like H2O, but makes decisions only at one point during generation. Expected Attention [11] tries to estimate future relevance of KV cache information (under some strong assumptions). FlexGen [51] shows how to maximize throughput by using a cache hierarchy. FastGen [20] provides a meta-strategy voting between a number of diferent cache policies. CAKE [45] runs sparse inference with a H2O-related score, but also distributes an overall memory budget between layers. Other sparse attention techniques include [23, 65, 59, 7, 64, 58, 17, 67, 61]. KVPop [24] learns a cache policy against a futureattention target, computed eficiently using FlexAttention [13]. Policies are parameterized as small xLSTMs [4]. qTTT [3] uses a few gradient updates at test time in order to improve inference results. MInference [29] and KVPress (https://github.com/NVIDIA/kvpress) are open source libraries providing several sparse attention methods. ShadowKV [56] is a high-throughput long-context inference system including KV cache selection. SCBench [37] provides a comprehensive empirical analysis of long-context inference methods.

Optimized scaled dot product attention (SDPA) kernels are essential for fast inference and training, pioneered by FlashAttention [9, 48]. FlashInfer [68] is optimized for the inference case 1 ≪ $N _ { q } \ll N _ { k }$ (notation from Section 3.1). FlexAttention [13] allows to specify mask and score modification code, using torch.compile under the hood.

Our main contribution is on long-context fine-tuning. Prior work can be ordered into two groups. Proposals in the first group modify multi-head attention in ways which remedy the dificulties detailed in Section 3.2. LongLoRA [8] uses permutation and reshaping of keys and values, essentially trading (B, N) for $( B ( N / N _ { C } ) , N _ { C } )$ , where B is batch size, N is sequence length, $N _ { C }$ is cache length. This speeds up MHA, but does not reduce KV memory, and works only if $N / N _ { C }$ is small. Native sparse attention (NSA) [70] bakes a mixture of some sparse attention kernels with diferent fixed policies into the model architecture. DeepSeek sparse attention (DSA) [15] is a refined variant of this idea. Diferent to selective sparse attention, this needs to be used during pre-training already, whose cost is significantly increased. IndexCache [2] speeds up DSA somewhat by sharing the indexer (i.e., the cache logic) between several layers. Also, by fixing the selection policy as part of the model choice, DSA ofers considerable less flexibility during post-training. Finally, LSA and DSA still require storing the complete KV cache (even though each attention call uses only a part of it). LongGen [19] uses static sparsity patterns for the top and bottom third of layers and full attention for the middle third, which speeds up training a bit, but does not reduce its memory requirements. DMC [42] uses a form of KV cache compression where new information is either appended or accumulated with the last recent slot. They propose heuristics to train a model with this mechanism in place. This approach seems restricted to KV cache updates during generation, it is not clear what is done when a large prompt needs to be processed. Starting from Mamba [22], there have been several attempts to resurrect LSTMs [26], e.g. [10, 4]. When compared against the long-context transformer SotA, none of them have been competitive enough in order to warrant costly pre-training eforts. YOCO [57] proposes an architecture diferent to the transformer, where a single KV cache block serves all layers. Note that KV bufer scaling with layers is not a major problem in practice, since at any time, all but one can be ofloaded to CPU (see Section A.4.1).

In the second group, KV cache bufers are not compressed, but both storage and computation are distributed across several devices. This can be done with RingAttention [38] or more recent variants [39], in what is called context parallelism (CP), or with sequence parallelism (SP) [32]. While [34] observe that long sequences can be split into chunks, and that the computation graph factorizes along the chunk axis, the factor for the final chunk depends on all KV cache bufers of all layers, so cannot be represented on a single device<sup>2</sup> (attempts to sparsify this computation are heuristics, and experiments are done only on rather short sequences of up to 16k tokens). OOMB [33] shares properties with our work, such as chunklevel processing, activation checkpointing, and eforts to compress KV cache bufers for autograd. While for exact KV caches, they run into the same issues as [34], their implementation supports NSA and DSA as well. Their way of hiding KV cache bufers from autograd requires a number of complex dedicated CUDA kernels, which need to be hardcoded for every sparse attention variant. We manage KV bufer size in autograd via delta encoding, which renders our implementation agnostic to the KV cache policy. Details are given in Section A.8. LongRoPE [12, 49] combines SP with a search for non-uniform RoPE and several lifting stages. Other work tuning position encoding, data mix and fine-tuning recipes, but not compressing KV cache, includes [63, 76, 18]. LongStraw [80] is a system designed for long context reinforcement learning, with a specific emphasis on sharing prompt graphs and KV caches between diferent roll-outs. It uses OOMB for gradient computations, but could likely be configured with our method as well. Highly optimized implementations of CP/SP are the state of the art for long-context inference, e.g. vLLM [30], SGLang [79], and finetuning, e.g. Nvidia NeMo RL (https://github.com/NVIDIA-NeMo/RL), MS-SWIFT [78]. In Section 3.3, we comment on reasons why sparse attention methods are less frequently used in practice, and how this could be changed.

## 3 Long-Context Fine-Tuning

In this section, we first introduce sparse attention and key-value caching, providing several improvements to the heavy-hitter oracle (H2O) cache policy [72], leading to a more eficient implementation with dedicated SDPA kernel support. Then, we detail our main contribution: a novel method to fine-tune models with sparse attention in place, using resources comparable to sparse attention inference. While in SotA CP/SP techniques, GPUs need to be used for sharding along the context, they can be used to increase throughput (i.e., larger batch sizes) or to handle larger models in our method.

## 3.1 Sparse Attention. Key-Value Caching

Multi-head attention (MHA) is the most important mechanism in modern transformer architectures [60]. At its core lies scaled dot product attention (SDPA):

$$
Y = \mathrm { S D P A } ( Q , K , V ) , \quad Y , Q \in \mathbb { R } ^ { ( B , H _ { q } , N _ { q } , d _ { h } ) } , K , V \in \mathbb { R } ^ { ( B , H _ { k } , N _ { k } , d _ { h } ) } .\tag{1}
$$

Here, Q (queries), K (keys), V (values) are 4D arrays, B is the batch size, $d _ { h }$ the perhead embedding dimension, $H _ { q } , H _ { k }$ are numbers of heads, and $N _ { q } , N _ { k }$ are sequence lengths (i.e., the third dimension is mapping to token positions in the model context). The model embedding dimension is $d = H _ { q } \cdot d _ { h }$ . Let us first assume that $H _ { q } = H _ { k }$ and drop the first two dimensions. Then:

$$
\begin{array} { r } { \pmb { Y } = \pmb { M } \pmb { V } , \quad \pmb { M } = \mathrm { s o f t m a x } \left( \mathrm { m a s k } \left( d _ { h } ^ { - 1 / 2 } \pmb { Q } \pmb { K } ^ { T } \right) , \mathrm { d i m } = 1 \right) . } \end{array}\tag{2}
$$

Y are weighted combinations of values V with attention weights $\boldsymbol { M } \in \mathbb { R } ^ { ( B , H _ { q } , N _ { q } , N _ { k } ) }$ softmax applies $\pmb { x } \mapsto \exp ( \pmb { x } ) / ( \mathbf { 1 } ^ { T } \exp ( \pmb { x } ) )$ along rows. mask implements causal masking: $( \mathfrak { m a s k } ( X ) ) _ { i , j } = x _ { i , j } - \infty \mathrm { I } _ { \{ P + i < t ( j ) \} }$ , where P and $t ( j )$ are token positions (see Section 3.3.1 for details). If arrays are indexed by $( b , h , j , k )$ , SDPA operates on $( j , k )$ in the same way for all $( b , h )$ , computations are parallelized over batch and head positions.

Inference in transformers switches between prompt processing and token generation. Generating a token after a prompt of size N requires SDPA with $N _ { k } = N , N _ { q } = 1$ , with keys and values of size $\left( { B , H _ { k } , N , d _ { h } } \right)$ in GPU memory, for each of L model layers. Exact transformer inference therefore requires $\mathcal { O } ( L \cdot N \cdot B H _ { k } d _ { h } )$ GPU memory for the full key-value (KV) cache. Even for moderate context lengths N of several hundred thousands, the KV cache far surpasses the model weights in size and cannot be stored in GPU memory as is.

A large amount of prior work confronts this problem (see also Section 2). In grouped query attention (GQA) [1], we set $H _ { q } = H _ { k } \cdot q _ { g } , q _ { g } > 1$ , so that $q _ { g }$ heads map to the same query group, which reduces KV cache size by a factor of $q _ { g } .$ . Most relevant to our work is sparse attention (or selective KV caching; e.g. [72]), where the KV cache is represented by fixedsize bufers independent of the context length of the model. Once all slots are filled, new information overwrites (or evicts) existing ones. Formally, the cache (for one model layer) is represented by arrays keys, values : $\left( B , H _ { k } , N _ { C } , d _ { h } \right)$ and token pos : $( B , H _ { k } , N _ { C } )$ . Here, $N _ { C }$ is the cache length, which is chosen as large as GPU memory permits. For up to $N _ { C }$ tokens, the cache is filled from left to right. After that, the KV cache policy $\pi _ { l } ( b , h , t ) \in$ $\{ 0 , \ldots , N _ { C } - 1 \}$ dictates where additional key-value information is written for token position $t \geq N _ { C }$ , batch position $b \in \{ 0 , \ldots , B - 1 \}$ and head (or query group) $h \in \{ 0 , \ldots , H _ { k } - 1 \}$ Importantly, $\pi _ { l }$ can depend on $b , h \colon \mathrm { a }$ token may be in the cache for some batch positions and heads, and not for others. $t ( b , h , j ) = \mathsf { t o k e n \_ p o s } [ b , h , j ]$ lists the token position of what is stored in $( b , h , j )$ . We do not require complex memory layouts and dedicated kernels for this setup, as for example PagedAttention [30] needs. In fact, (2) depends on absolute token positions only via mask. This is as sparse as the conventional triangular one, but depends on token positions $t ( \cdot ) = \mathtt { t o k e n }$ pos, which through cache evictions becomes non-monotonic in general. Moreover, torch.gather and torch.scatter provide fast read and write access for these bufers (see Section A.1.1).

With sparse attention, we can run inference for any context width N. First, we process up to the first $N _ { C }$ tokens with a single SDPA call $( N _ { q } = N _ { k } = N _ { C } )$ , this is known as $p r e f l l i n g .$ The remaining $N - N _ { C }$ tokens are processed in chunks of size $S \mathrm { ~ < ~ } N _ { C }$ , using SDPA calls with $N _ { q } = S , N _ { k } = N _ { C }$ (see Section A.4.1 for details). Token generation uses $N _ { q } = 1 , N _ { k } = N _ { C }$ . Memory requirements are independent of N. For each chunk, the policy [π<sub>l</sub>] is used to determine $\boldsymbol { B } \cdot \boldsymbol { H _ { k } } \cdot \boldsymbol { S }$ positions $( b , h , j )$ which are overwritten by the new keys and values. While $N _ { C }$ is chosen as large as memory permits, the choice of S is more subtle. The larger S, the fewer chunks, and less sequential computation results in faster processing. The smaller S, the more fine-grained the cache policy is used, which can lead to better decisions (see also Section 3.3).

## 3.1.1 Variants of Heavy-Hitter Oracle

The key idea behind the heayy-hitter oracle (H2O) [72] is to make use of the attention weights $M = [ m _ { i , j } ]$ , a by-product of SDPA (2). Dropping $( b , h )$ for the moment, we have that $\begin{array} { r } { \pmb { Y } _ { i , : } = \sum _ { j } m _ { i , j } \pmb { V } _ { j , : } } \end{array}$ <sub>:</sub> and $\begin{array} { r } { \sum _ { j } m _ { i , j } = 1 } \end{array}$ $m _ { i , j }$ quantifies how much values $V _ { j , \ l }$ <sub>:</sub> are used to create $\mathbf { } Y _ { i , : }$ . The cumulative sum $\textstyle \sum _ { i } m _ { i , j }$ can be used to score the usefulness of vectors $( K _ { j , : } , V _ { j , : } )$ in the KV cache. Bringing (b, h) back, we define the H2O score after having

processed t tokens as

$$
\phi _ { \mathrm { h 2 o } } ^ { t } ( b , h , j ) = \sum _ { t ( b , h , j ) \leq s < t } m _ { b , h , s , j } ,\tag{3}
$$

where $t ( b , h , j ) =$ token pos $[ b , h , j ]$ is the position represented at $( b , h , j )$ right now. We sum over $s \in \{ t ( b , h , j ) , \ldots , t \}$ because the slot is occupied from KV information corresponding to token position $t ( b , h , j )$ , which entered the cache only then. The larger $\phi _ { \mathrm { h 2 o } } ^ { t } ( b , h , j )$ , the more valuable this information has been so far. When asked to insert new content for S tokens, for each $( b , h )$ , we overwrite these S slots $j$ for which $\phi _ { \mathrm { h 2 o } } ^ { t } ( b , h , j )$ is smallest.

In this paper, we modify the original H2O policy [72] (as provided by their implementation) in several ways. First, their code selects the same cache slots for each batch position $b ,$ using the score $\begin{array} { r } { \phi _ { \mathrm { h 2 o - o r i g } } ^ { t } ( h , j ) = \sum _ { b } \phi _ { \mathrm { h 2 o } } ^ { t } ( b , h , j ) } \end{array}$ . The rationale for this restriction is unclear, we implement H2O without it as well. Second, the cumulative H2O score seems to favour entries $( b , h , j )$ which have been in the cache for longer, since more terms in $[ 0 , 1 ]$ are summed then. We introduce the normalized H2O score: $\phi _ { \mathrm { h 2 o - n o r m } } ^ { t } ( b , h , j ) = ( t - t ( b , h , j ) ) ^ { - 1 } \phi _ { \mathrm { h 2 o } } ^ { t } ( b , h , j )$

Despite convincing empirical results of H2O, both in [72] and Section 4, it is not widely used. This is mostly because current implementations of H2O are much slower than the state of the art. We need summed attention weights $\textstyle \sum _ { i } m _ { b , h , i , j }$ for each $( b , h , j )$ , as by-product of SDPA (2), but none of the fast SDPA kernels derived from FlashAttention [9] provide them. Current H2O implementations use naive SDPA implementations, which are much too slow in practice. Our implementation contains Triton code to return summed attention weights alongside a FlashInfer SDPA kernel [68]. In Section ${ \mathrm { A . 6 } } ,$ we show how FlexAttention [13] can be used to this end as well. We come back to eficiency of sparse attention in Section 3.3.

## 3.2 Fine-Tuning for Sparse Attention

How should we train a model which uses sparse attention with some KV cache policy such as H2O? As noted in Section 2, all prior long-context fine-tuning methods either restrict the MHA approximation to a particular form, or use exact MHA with KV cache bufers distributed across several devices (i.e., sequence or context parallelism). However, the choice of KV cache policy, which dictates how the model’s short term memory is organized, should influence how the model is best trained. Our results in Section 4 validate this hypothesis. Sparse attention inference for a model trained with exact MHA and sequence parallelism (which is the SotA) often performs significantly worse than for a model trained with sparse attention and the desired policy in place.

Fine-tuning for models with sparse attention is dificult, because an enormous amount of memory is required, while GPU memory is on short supply. We need to compute gradients for training loss functions on sequences of length $N \gg N _ { C }$ , which is done by (reverse mode) automatic diferentiation (or error backpropagation, autograd). Autograd works by creating a computation graph during the forward pass, whose nodes store arrays needed during the backward pass. If the model has L layers and the chunk size is S, the training sequence is split into $1 + \lceil ( N - N _ { C } ) / S \rceil$ chunks, the first (prefill) chunk of length $N _ { C }$ and subsequent chunks of length S. SDPA is called for each layer and chunk, creating at least one node of the size of the KV cache, so we need at least $\mathcal { O } ( L \cdot S ^ { - 1 } ( N - N _ { C } ) \cdot N _ { C } \cdot \mathcal { D } )$ of memory, where $\mathcal { D } = B H _ { k } d _ { h }$ . Assuming $S = a N _ { C }$ for some constant $^ { a , }$ this is $\mathcal { O } ( L \cdot N \cdot \mathcal { D } )$ : more than the full KV cache would need, and far beyond what is tractable.

We need several ideas in order to bring GPU memory requirements down to levels comparable to what inference needs. First, we avoid diferentiation through the KV cache policy (which is often not even possible, and in general not tractable). Along the forward pass, we store all KV cache policy decisions in a replay log, containing for each chunk (and each layer l) the tokens processed, and the decisions $\{ \pi _ { l } ( b , h , t ) \}$ . Later on, we use replay caches, which act like normal KV caches, except that eviction decisions are replayed from the log. If a cache policy is complex and expensive to compute, it needs to be run during the forward pass only.

Next, we use activation checkpointing [25]. Even with moderate context widths, this technique is routinely used for models with many layers. Gradients are computed in two passes: forward and backward. Diferent from autograd, there is no computation graph built during the forward pass, only the input tensors for each transformer layer are stored to CPU memory. The backward pass is split into L autograd calls, starting from the top. Head gradients are supplied from the previous layer, inputs are loaded from CPU. Computations graphs on GPU are L times smaller, while forward computations have to be run twice.

While standard activation checkpointing tackles large L, in long-context situations the context width N is the more serious problem. We cannot even keep complete inputs or head gradients for a single layer in GPU memory (see also Section A.4.1), let alone KV bufers attached to each chunk in the computation graph. We therefore use activation checkpointing twice, in a nested fashion. To this end, we partition chunks into cells: $\{ ( B , S , d ) \}  ( B , k S , d )$ , where $k = \lfloor \alpha N _ { C } / S \rfloor$ , and $\alpha > 0$ is a hyperparameter which defaults to $\alpha = 1$ . The first (prefill) chunk becomes the first cell. As a rule of thumb, we group chunks into cells which occupy about the size of KV cache bufers. The complete computation graph can be seen as lattice of cells: rows are layers, columns are cells (i.e., groups of chunks) along the context. Our backward pass runs an outer loop over rows (layers), then inner loops over cells in each layer. Each inner loop starts with a (non-autograd) forward pass along the context, storing KV cache bufers (the inner loop ”activations” to checkpoint) going into each cell to CPU. Next, autograd is run separately on each cell, starting from the right. Inputs to a cell (layer inputs from the bottom and KV cache bufers from the left) and head gradients from the top are read from CPU, while head gradients from the right stay in GPU memory. Gradients are accumulated. We reuse the same CPU and GPU bufers for all inner loops.<sup>3</sup> A detailed summary of our method is given in Section A.4.2.

Even with nested activation checkpointing, the autograd calls still need too much GPU memory. Recall that a cell consists of $k = \lfloor \alpha N _ { C } / S \rfloor$ chunks. Autograd stores KV cache bufers for each chunk, so needs at least $\mathcal { O } ( k \cdot N _ { C } \cdot \mathcal { D } )$ memory, where $\mathcal { D } = B H _ { k } d _ { h }$ . As detailed in Section 3.1, k must be sizable to allow cache policies to make good decisions. In this section, we detail the last (and maybe most important) idea, which cuts GPU memory by a factor of k, to $\mathcal { O } ( N _ { C } \cdot \mathcal { D } )$ per autograd call. This is comparable to what is needed during inference alone.

Consider KV cache bufers (keys, values), (keys<sup>′</sup>, values<sup>′</sup>) for neighboring chunks. Their size is $\left( B , H _ { k } , N _ { C } , d _ { h } \right)$ , but they only difer in $S \cdot D$ values, because a chunk consists of S

tokens only. The relationship is simple:

$$
\mathtt { k e y s ^ { \prime } = s c a t t e r ( k e y s , i n d e x , k e y \_ n e w ) , \mathtt { v a l u e s ^ { \prime } = s c a t t e r ( v a l u e s , i n d e x , v a l u e \_ n e w ) . } }
$$

Here, key new, value new are KV vectors for new tokens with sizes $( B , H _ { q } , S , d _ { h } )$ , index is based on the cache policy π<sub>l</sub>, determining which slots are overwritten, and scatter, gather are linear torch operators defined in Section A.1.1.

This is a linear recurrence, which is easily inverted:

$$
\mathtt { k e y s } = \mathtt { s c a t t e r } ( \mathtt { k e y s } ^ { \prime } , \mathtt { i n d e x } , \mathtt { d e l t a } \mathtt { \_ k e y } ) , \quad \mathtt { d e l t a } \mathtt { \_ k e y } = \mathtt { g a t h e r } ( \mathtt { k e y s } , \mathtt { i n d e x } ) ,\tag{4}
$$

and the same for values. Instead of storing keys, values for each chunk in the compute graph, it sufices to store<sup>4</sup> delta key, delta value. Since memory requirements of autograd calls are dominated by the KV cache bufers, they are reduced by a factor of k.

While the linear recurrence relation between neighboring cache bufers is simple, implementing it in the context of PyTorch autograd is not. We use a mechanism called autograd saved tensors hooks<sup>5</sup>, originally intended to implement activation checkpointing by CPU ofloading, which can be shaped to ours needs. In a nutshell, we use the PyTorch mechanism to store (delta key, delta value) in the autograd graph in place of (keys, values) (called ”packing”), reconstructing the latter from the former and subsequent (keys<sup>′</sup>, values<sup>′</sup>) during the backward pass over chunks (called ”unpacking”). The key dificulty is the non-selectiveness of the mechanism: it provides a pack hook function called for all arrays PyTorch autograd decides to place into its graph. There is no way to tag tensors in the forward code, so they can be recognized as pack hook arguments. Our solution is to create annotations alongside the forward pass code, storing (index, delta key) for keys, (index, delta value) for values. In a pack hook(x) call, we relate x to current annotations: x matches (index, delta key) (say) if gather(x, index) = delta key. A match leads to pack hook(x) returning a reference to (index, delta key), which is removed from the annotation list. Note that failing to match an annotation does not lead to errors, but at most to a bit more memory being used. More details are given in Section A.4.3.

## 3.3 Sparse Attention and Sequence Parallelism

While with context or sequence parallelism, the context width is strictly limited by the number and memory size of GPUs available, sparse attention inference can be run for any context width on moderate GPU resources. Moreover, redundancies which exist in multihead attention, can be exploited by way of cache compression, and experimental results with H2O are in general not worse than with exact attention even if the KV cache is compressed to 20% or less [72, 77]. Even if many GPUs are available, using sparse attention allows us to increase batch size by way of distributed data parallel, or to keep more layers in GPU memory. A large number of sparse attention variants have been proposed (see Section 2). Why is it then that sparse attention methods are hardly used in SotA inference libraries such as vLLM [30]? The short answer is that latency is significantly higher with existing sparse attention implementations. Further comments are in Section A.7.

Some of the gap is due to less low level implementation support for sparse attention. Highly optimized SDPA kernels are vital for fast inference [9, 13, 68]. However, as noted in Section 3.1.1, existing kernels do not cater for sparse attention inference (see details in Section 3.3.1). Other reasons for the gap are more dificult to address. A long sequence is split into chunks, the first (prefill) chunk of length $N _ { C }$ (cache length), subsequent chunks of length S. Larger S means fewer chunks and lower latency. But KV cache policies can make useful eviction decisions only if S is much less than $N _ { C }$ (in our experiments in Section 4, we use $N _ { C } = 3 2 7 6 8$ and $S \in \{ 1 0 2 4 , 2 0 4 8 \} )$ . For the extreme choice $S = N _ { C }$ , the whole cache is overwritten by new content for every chunk, and the KV cache policy plays no role at all! Information cannot be kept in the cache beyond a chunk if we do not allow for substantial overlap.

For example, suppose we use 8 devices, each supporting a cache length $N _ { C } .$ , and the sequence length is $N = 8 N _ { C }$ . With RingAttention, each device holds $N _ { C }$ slots, and the sequence is processed in 8 sequential chunks. But for sparse attention, we need $1 + 7 N _ { C } / S$ chunks, which can be substantially larger. Despite sparse attention supporting a 8 times larger batch size via distributed data parallel (DDP), inference tends to still be slower than with RingAttention. In future work, we plan to improve sparse attention latency by appropriate kernel fusion. However, the sequential nature of decision making in sparse attention may be an inherent disadvantage over sequence or context parallelism, which may remain the best choice if a large hardware budget can be aforded for inference.

Apart from large hardware requirements to even work on long sequences, RingAttention requires $\mathcal { O } ( L \cdot D )$ synchronizations between all devices per gradient update, while sparse attention DDP only needs a single gradient averaging reduction. While memory transfer between devices can be run in parallel with computations, this needs double bufering<sup>6</sup> in RingAttention, doubling the GPU memory needed. The peer-to-peer memory transfer is more brittle than DDP used with sparse attention, and robust training code is more dificult to implement. Finally, the ”waste by pre-allocation” issues which motivate the fairly complex PagedAttention [30], have a simpler solution with sparse inference: KV cache bufers are of a fixed length, there is no need to $\mathrm { s p l i t } ^ { 7 }$ them into pages along the sequence axis, and no custom SDPA code is needed.

## 3.3.1 Discussion: SDPA Kernels for Sparse Attention

Here, we list some ideas for SDPA kernel developers to better support sparse attention. First, we consider causal masking for sparse attention. In standard MHA (the ”training case”), $( \mathtt { m a s k } ( X ) ) _ { i , j } = x _ { i , j } - \infty \mathrm { I } _ { \{ i < j \} }$ . For sparse attention, KV information is stored in cache bufers in an ordering given by token positions $t ( b , h , j )$ , and the causal mask is given by $( b , h , i , j ) \mapsto ( - \infty ) \mathrm { I } _ { \{ P + i < t ( b , h , j ) \} }$ , where P is the number of tokens processed before the current MHA call (so the new information is for tokens $\{ P , \ldots , P + N _ { q } - 1 \} )$ . The new key-value information has been written into the cache already, so that $\{ P , \ldots , P + N _ { q } - 1 \}$ is part of $\{ t ( b , h , j ) \}$ for each $( b , h )$

Unfortunately, none of the fast SDPA kernel codes we know of support such variants of causal masking in an implicit $\mathrm { l y } ^ { 8 }$ defined way. In our implementation, we sort the token positions and reorder keys and values according to this index, separate for each $( b , h )$ , after which we can use standard causal masking, where queries are right-aligned with keys and values. This needs extra computation and memory which could be saved with better SDPA kernel support. Based on our experience, the following simple extensions of fast SDPA kerne libraries could make a major diference for sparse attention:

• Return summed attention weights $\textstyle \sum _ { i } m _ { b , h , i , j }$ (see Section 3.1.1), an array of size $\left( B , H _ { q } , N _ { k } \right)$ , based on attention weights which are computed anyway. This allows for H2O and related scores to be computed, driving advanced KV cache policies.

• Allow for implicitly defined causal masks of the form $( b , h , i , j ) \mapsto ( - \infty ) \mathrm { I } _ { \{ P + i < t ( b , h , j ) \} }$ where $t ( \cdot )$ is an $\left( \boldsymbol { B } , \boldsymbol { H _ { k } } , \boldsymbol { N _ { k } } \right)$ integer array. While FlexAttention [13] supports custom mask patterns, they need to be written in terms of scalar index variables, so cannot have a 3D array as input to the compute graph.

## 4 Experiments

The key question addressed in our experiments is: if long context inference uses sparse attention with a particular KV cache policy, how much is gained by fine-tuning the model with the same policy in place (using our novel method) over training it with state of the art libraries using sequence or context parallelism? We run comparisons on the Helmet [69] benchmarks, with context widths of 64k and 128k, covering a range of cache policies:

• lastrec $( \mathrm { l r } ) \colon \pi ( b , h , t ) = t \mathrm { I } _ { \{ t < N _ { C } \} } + ( \mathrm { m o d } ( t - N _ { C } , N _ { C } - \beta ) + \beta ) \mathrm { I } _ { \{ t \geq N _ { C } \} }$ , where $\beta \in$ $[ 0 , N _ { C } )$ . Keeps the last recent $N _ { C } - \beta$ and first β tokens in the cache. It is important to choose $\beta > 0$ as default ”attention sink” [65]. Our default is $\beta = \operatorname* { m i n } ( 1 6 , \lceil N _ { C } / 8 \rceil )$

• smart lastrec (slr): Variant of lastrec, where the number $\beta$ of initial tokens is chosen dependent on content (see Section A.2.1 for details). A simple version of this heuristic appeared in [23].

• h2o (h2o), h2o norm $\left( \mathrm { h 2 o } ^ { \mathrm { n o } } \right)$ , h2o ori $\mathrm { \cdot \vec { g } _ { \mathrm { \ell } } ( h 2 0 ^ { o r } ) }$ : Variants of H2O [72] (see Section 3.1.1 for details). h2o orig is equivalent to their code released, where $\pi ( b , h , t )$ does not depend on batch position b.

We train a Qwen3-4B-Instruct-2507<sup>9</sup> model [66], using AdamW [41] with base learning rate 0.0005 for up to 5 epochs. We train LoRA weights only [28] (rank $r = 1 6 , \alpha = 1 6$ , on all linear blocks). We run on four Nvidia A100 40 GB devices with a per-device batch size of 2, using RoPE [55] and YaRN [44] for position encoding. Fine-tuning is done in diferent ways:

• Sequence parallelism (sp): We use MS-SWIFT [78] for fine-tuning with DeepSpeed ZeRO-3 ofload, FlashAttention, and Liger kernels enabled, running on four Nvidia A100 40GB GPUs. We use a per-device batch size of 2 and sequential gradient accumulation (efective batch size 8)<sup>10</sup>.

• Our method (us): We use a cache length $N _ { c } = 3 2 7 6 8$ , batch size $S \in \{ 1 0 2 4 , 2 0 4 8 \}$ ， and chunks per cell multiplier α = 1 for S = 2048, $\alpha = 0 . 7 5$ for $S \ = \ 1 0 2 4$ (see also Section A.4.1). KV cache bufers are quantized to 8 bits using torchao. We run distributed data parallel optimization on 4 devices to obtain an efective batch size of 8. We evaluate the model on a heldout validation set every 10 gradient steps (5 for pop qa), and choose the checkpoint with the lowest validation loss for testing.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>64k datasets</td><td rowspan=1 colspan=4>128k datasets</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>nqus  sp</td><td rowspan=1 colspan=1>tri_qaus sp</td><td rowspan=1 colspan=1>hot_qaus sp</td><td rowspan=1 colspan=1>pop-qaus sp</td><td rowspan=1 colspan=1>nqus  sp</td><td rowspan=1 colspan=1>tri_qaus sp</td><td rowspan=1 colspan=1>hot_qaus sp</td><td rowspan=1 colspan=1>pop-qaus sp</td></tr><tr><td rowspan=1 colspan=1>exact</td><td rowspan=1 colspan=1>- 50.7</td><td rowspan=1 colspan=1>79.8</td><td rowspan=1 colspan=1>-60.0</td><td rowspan=1 colspan=1>- 62.7</td><td rowspan=1 colspan=1>- 50.7</td><td rowspan=1 colspan=1>- 68.7</td><td rowspan=1 colspan=1>- 46.3</td><td rowspan=1 colspan=1>-57.0</td></tr><tr><td rowspan=2 colspan=1>lr2kslr2k</td><td rowspan=2 colspan=1>33.5 57.247.3 56.5</td><td rowspan=1 colspan=1>75.3 57.5</td><td rowspan=1 colspan=1>53.362.7</td><td rowspan=1 colspan=1>43.7 60.5</td><td rowspan=1 colspan=1>26.033.0</td><td rowspan=1 colspan=1>50.852.3</td><td rowspan=2 colspan=1>31.046.334.042.0</td><td rowspan=2 colspan=1>34.025.037.722.2</td></tr><tr><td rowspan=1 colspan=1>74.560.8</td><td rowspan=1 colspan=1>50.067.3</td><td rowspan=1 colspan=1>44.056.7</td><td rowspan=1 colspan=1>26.033.7</td><td rowspan=1 colspan=1>61.251.8</td></tr><tr><td rowspan=1 colspan=1>h202k</td><td rowspan=1 colspan=1>47.2 70.8</td><td rowspan=1 colspan=1>78.072.0</td><td rowspan=1 colspan=1>53.068.7</td><td rowspan=1 colspan=1>57.544.7</td><td rowspan=1 colspan=1>24.240.7</td><td rowspan=1 colspan=1>47.863.7</td><td rowspan=1 colspan=1>19.3 26.0</td><td rowspan=1 colspan=1>53.349.8</td></tr><tr><td rowspan=1 colspan=1>h2o2k</td><td rowspan=1 colspan=1>47.868.2</td><td rowspan=1 colspan=1>63.2 54.5</td><td rowspan=1 colspan=1>58.3 70.0</td><td rowspan=1 colspan=1>53.039.8</td><td rowspan=1 colspan=1>43.551.3</td><td rowspan=1 colspan=1>66.755.3</td><td rowspan=1 colspan=1>37.3 51.0</td><td rowspan=1 colspan=1>50.2 25.2</td></tr><tr><td rowspan=1 colspan=1>h202k</td><td rowspan=1 colspan=1>49.5 73.3</td><td rowspan=1 colspan=1>66.3 65.7</td><td rowspan=1 colspan=1>57.368.7</td><td rowspan=1 colspan=1>62.2 45.5</td><td rowspan=1 colspan=1>45.3 58.8</td><td rowspan=1 colspan=1>71.2 71.0</td><td rowspan=1 colspan=1>36.7 44.3</td><td rowspan=1 colspan=1>50.2 33.3</td></tr><tr><td rowspan=1 colspan=1>lr1k</td><td rowspan=1 colspan=1>59.7 57.0</td><td rowspan=1 colspan=1>73.060.7</td><td rowspan=1 colspan=1>47.7 65.3</td><td rowspan=1 colspan=1>41.7 59.3</td><td rowspan=1 colspan=1>23.532.7</td><td rowspan=1 colspan=1>59.5 50.0</td><td rowspan=1 colspan=1>28.745.0</td><td rowspan=1 colspan=1>34.5 25.2</td></tr><tr><td rowspan=1 colspan=1>slr1k</td><td rowspan=1 colspan=1>37.8 57.0</td><td rowspan=1 colspan=1>59.8 59.7</td><td rowspan=1 colspan=1>52.7 65.0</td><td rowspan=1 colspan=1>46.8 57.0</td><td rowspan=1 colspan=1>29.3 36.2</td><td rowspan=1 colspan=1>58.249.7</td><td rowspan=1 colspan=1>33.744.7</td><td rowspan=1 colspan=1>34.321.2</td></tr><tr><td rowspan=1 colspan=1>h201k</td><td rowspan=1 colspan=1>62.072.8</td><td rowspan=1 colspan=1>79.772.0</td><td rowspan=1 colspan=1>55.068.0</td><td rowspan=1 colspan=1>59.3 45.8</td><td rowspan=1 colspan=1>23.241.3</td><td rowspan=1 colspan=1>51.7 58.7</td><td rowspan=1 colspan=1>25.3 24.3</td><td rowspan=1 colspan=1>51.050.5</td></tr><tr><td rowspan=1 colspan=1>h2o1k</td><td rowspan=1 colspan=1>47.571.3</td><td rowspan=1 colspan=1>62.359.3</td><td rowspan=1 colspan=1>61.773.0</td><td rowspan=1 colspan=1>51.043.5</td><td rowspan=1 colspan=1>42.753.3</td><td rowspan=1 colspan=1>70.358.2</td><td rowspan=1 colspan=1>42.354.7</td><td rowspan=1 colspan=1>44.1 26.8</td></tr><tr><td rowspan=1 colspan=1>h2o1k9</td><td rowspan=1 colspan=1>49.072.2</td><td rowspan=1 colspan=1>75.7 68.7</td><td rowspan=1 colspan=1>57.066.7</td><td rowspan=1 colspan=1>55.047.3</td><td rowspan=1 colspan=1>44.3 60.5</td><td rowspan=1 colspan=1>72.074.8</td><td rowspan=1 colspan=1>31.041.3</td><td rowspan=1 colspan=1>51.030.8</td></tr></table>

Table 1: Results for long-context inference with 5 KV cache policies and chunk sizes 2048 = 2k, 1024 = 1k (rows). The first row exact is for exact inference (sequence parallelism). We show SubEMvalues on test splits for diferent Helmet datasets nq, trivia qa, hotpot qa, pop qa, limiting sequence lengths to 64k or 128k tokens. Columns us are for models trained using our novel method with the same cache policy in place, columns sp are for models trained with sequence parallelism.

Results on 4 Helmet datasets (nq, trivia qa, hotpot qa, pop qa) [69] and 10 diferent cache setups (5 policies, 2 chunk sizes) are provided in Table 1. Respective results for the base checkpoint (no fine-tuning) are given in Table 4 in the Appendix (referred to as no elsewhere). For further 6 Helmet datasets (trec coarse, nlu, clinc150, inf qa, inf mc, json kv), Table 2 provides results for 3 cache policies and chunk size S = 1024. Details about Helmet datasets and metrics are given in Section A.3.1.

For the datasets in Table 1, results are mixed and inconclusive: sp is best for nq and hotpot qa, no for trivia qa (fine-tuning does not help), and us for pop qa. However, for the datasets in Table 2, us strongly outperforms sp and no. A closer look at generated samples (which can be up to 128 tokens) reveals a major failure mode of sp (see Section A.5.3): its outputs are far too long and contain mostly random nonsense. For most datasets in Table 2, targets are single numerical values, and the Accuracy metric (see Section A.3.1) compares this to the most frequently occuring number in the output. Poor results are due to the output for sp often containing many numbers. In contrast, us learns how to stop properly and usually outputs a single number. In fact, the same failure mode dominates outcomes in Table 1 just as well, but the metric SubEM used for the 4 datasets ignores content or length of output, as long as the target string is contained in it. Finally, sp and no fail for json kv as well, despite this using the SubEM metric. Targets are UUIDs of length 32 tokens. While variants of us identify them about half the time, they are hardly ever contained in the outputs of sp and no. In Section A.5.3, we quantify the failure mode in Table 7. While outputs for us are close in length to true targets, they are longer by large factors for sp, no, which often (but not always) span the full 128 tokens (despite true targets being much shorter). We also provide randomly chosen examples for outputs there, showcasing their nonsense content for sp.

<table><tr><td>dataset</td><td>trn</td><td>exact</td><td> $\mathrm { s l r } _ { 1 k }$ </td><td> $\overline { { \mathrm { h 2 o } _ { 1 k } ^ { \mathrm { n o } } } }$ </td><td> $\overline { { \mathrm { h 2 o } _ { 1 k } ^ { \mathrm { o r } } } }$ </td></tr><tr><td rowspan="3">trec_coarse</td><td>us</td><td>-</td><td>96.0</td><td>96.4</td><td>96.2</td></tr><tr><td>sp</td><td>97.8</td><td>30.0</td><td>23.2</td><td>77.6</td></tr><tr><td>no</td><td>一</td><td>28.2</td><td>19.8</td><td>36.0</td></tr><tr><td rowspan="3">nlu</td><td>us</td><td>-</td><td>90.0</td><td>87.4</td><td>79.8</td></tr><tr><td>sp</td><td>90.2</td><td>28.6</td><td>32.8</td><td>74.0</td></tr><tr><td>no</td><td>-</td><td>24.8</td><td>30.0</td><td>21.2</td></tr><tr><td rowspan="3">clinc150</td><td>us</td><td>-</td><td>97.4</td><td>96.8</td><td>94.0</td></tr><tr><td>sp</td><td>97.6</td><td>64.2</td><td>61.6</td><td>68.0</td></tr><tr><td>no</td><td>一</td><td>62.6</td><td>54.0</td><td>34.8</td></tr><tr><td rowspan="3">inf_qa</td><td>us</td><td>-</td><td>26.6</td><td>32.2</td><td>36.8</td></tr><tr><td>sp</td><td>40.8</td><td>2.2</td><td>2.2</td><td>3.3</td></tr><tr><td>no</td><td>-</td><td>2.5</td><td>2.9</td><td>3.4</td></tr><tr><td rowspan="3">inf_mc</td><td>us</td><td>-</td><td>40.0</td><td>42.0</td><td>54.0</td></tr><tr><td>sp</td><td>66.0</td><td>25.0</td><td>29.0</td><td>39.0</td></tr><tr><td>no</td><td>-</td><td>36.0</td><td>41.0</td><td>40.0</td></tr><tr><td rowspan="3">json_kv</td><td>us</td><td>-</td><td>49.0</td><td>50.0</td><td>3.0</td></tr><tr><td>sp</td><td>100.0</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>no</td><td>-</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Table 2: Results for 6 additional Helmet datasets not featured in Table 1 (context width 128k). Inference under 3 KV cache policies (chunk size $1 0 2 4 = 1 k )$ , exact uses sequence parallelism (column). trn denotes model checkpoint being used: us uses our novel method with the same cache policy in place, sp is using sequence parallelism, no is the base checkpoint Qwen3-4B-Instruct-2507 (no fine-tuning). Note that metrics are diferent, depending on the dataset (see Table 3).

The shortness of desired targets is a clear signal in the training data, expressed not only by us, but also by the sp checkpoints if exact inference is used with them (see Table 8). We should not be surprised by sp exhibiting such failure modes. When training with sequence parallelism (SP), each token can attend to any earlier one. This property is cut during inference, when most KV information is evicted at some point according to a logic which

SP was never aware of. Clearly, models should be trained under the same conditions and restrictions which govern inference later on. Our new method allows practitioners to do that even on a low budget, no matter what KV cache policy they like to use during inference.

While not our main focus here, the diferent KV cache policies exhibit variable performance across the diferent datasets. Ideally, the best policy is chosen for each task. Our results are inconclusive when it comes to ranking the diferent H2O variants (see Section 3.1.1). However, one concerning datapoint is the poor performance of h2o<sup>or</sup> on json kv in Table 2. In Table 7, we see that R = 3.7 and $p _ { 1 2 8 } = 8 5 \%$ for this logic, hinting to a similar failure mode than sp and no. At least in this case, the decision in [72] to score and evict batch dimensions together works much less well than the alternatives.

## 5 Conclusions

We showed how transformer language models with sparse attention can be fine-tuned on a moderate hardware budget (e.g., a single Nvidia A100 GPU with 40 GB RAM). Our method works for any KV cache selection or compression policy and allows the model to coadapt with the policy, often outperforming models trained with exact attention (sequence parallelism). We also provide a much more eficient implementation of H2O sparse attention (the leading policy in our experiments) with dedicated scaled dot product attention (SDPA) kernel support. By simplifying KV cache structure and clarifying the requirements on SDPA, we hope to direct more attention of the fast inference community on sparse attention (see Section 3.3.1), which despite its major potential for post-training specialization via cache selection or compression policy design does not currently play a significant role in longcontext inference or fine-tuning practice.

In future work, we will combine context parallelism with sparse attention. We are also considering kernel fusion ideas in order to narrow the latency gap further. An important direction will be multi-stream asynchronous implementations which allow for on-the-fly CPU ofloading [71]. We believe that once the host memory of a system can be used without much synchronization overhead and less double bufering, many current dificulties with KV caching will be much diminished. Finally, we hope that KeysAndValues (https://github. com/awslabs/keys\_values), the open source library with which most experiments were run here (see Section A.9), will make it easier to use, compare and extend sparse attention policies, work on which so far is somewhat cluttered when it comes to implementations.

## References

[1] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, Lebr´on, and F. Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4895–4901, 2023.

[2] Y. Bai, Q. Dong, T. Jiang, X. Lv, Z. Du, A. Zeng, J. Tang, and J. Li. IndexCache: Accelerating sparse attention via cross-layer index reuse. Technical Report arXiv:2603.12201 [cs.CL], 2026.

[3] R. Bansal, A. Zhang, R. Tiwari, L. Madaan, S. Duvvuri, F. Devvrit, D. Brandfonbrener, D. Alvarez-Melis, P. Bhargava, M. Kale, and S.e Jelassi. Let’s (not) just put things in context: Test-time training for long-context LLMs. In Int. Conf. Learning Representations, 2026.

[4] M. Beck, K. P¨oppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. xLSTM: Extended long short-term memory. In Globerson et al. [21].

[5] D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors. Advances in Neural Information Processing Systems 38. Curran Associates, 2025.

[6] W. Brandon, M. Mishra, A. Nrusimha, R. Panda, and J. Ragan-Kelley. Reducing transformer key-value cache size with cross-layer attention. In Globerson et al. [21].

[7] Z. Cai, Y. Zhang, B. Gao, Y. Liu, Y. Li, T. Liu, K. Lu, W. Xiong, Y. Dong, J. Hu, and W. Xiao. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. In Conference on Language Modeling, 2025.

[8] Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, and J. Jia. LongLoRA: Eficient finetuning of long-context large language models. In Int. Conf. Learning Representations, 2024.

[9] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. R´e. FlashAttention: Fast and memoryeficient exact attention with IO-awareness. In Oh et al. [43].

[10] T. Dao and A. Gu. Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In Salakhutdinov et al. [46], pages 10041–10071.

[11] A. Devoto, M. Jeblick, and S. J´egou. Expected attention: KV cache compression by estimating attention from future queries distribution. Technical Report arXiv:2510.00636 [cs.AI], 2025.

[12] Y. Ding, L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and M. Yang. LongRoPE: Extending LLM context window beyond 2 million tokens. In Salakhutdinov et al. [46], pages 11091–11104.

[13] J. Dong, B. Feng, D. Guessous, Y. Liang, and H. He. FlexAttention: a programming model for generating optimized attention kernels. In Proceedings of the 8th MLSys Conference, pages 381–394, 2025.

[14] DeepSeek-AI etal. DeepSeek-V2: A strong, economical, and eficient mixture-of-experts language model. Technical Report arXiv:2405.04434 [cs.CL], 2024.

[15] DeepSeek-AI etal. DeepSeek-V3.2: Pushing the frontier of open large language models. Technical Report arXiv:2512.02556 [cs.CL], 2025.

[16] J. Feng, S. Huang, X. Qu, G. Zhang, Y. Qin, B. Zhong, C. Jiang, J. Chi, and W. Zhong. ReTool: Reinforcement learning for strategic tool use in LLMs. Technical Report arXiv:2504.11536 [cs.CL], 2025.

[17] Y. Feng, J. Lv, Y. Cao, X. Xie, and S. Zhou. Ada-KV: Optimizing KV cache eviction by adaptive budget allocation for eficient LLM inference. In Belgrave et al. [5].

[18] T. Gao, A. Wettig, H. Yen, and D. Chen. How to train long-context language models (efectively). In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), pages 2391–2404, 2023.

[19] S. Ge, X. Lin, Y. Zhang, J. Han, and H. Peng. A little goes a long way: Eficient long context training and inference with partial contexts. In Int. Conf. Learning Representations, 2025.

[20] S. Ge, Y. Zhang, L. Liu, M. Zhang, J. Han, and J. Gao. Model tells you what to discard: Adaptive KV cache compression for LLMs. In Int. Conf. Learning Representations, 2024.

[21] A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors. Advances in Neural Information Processing Systems 37. Curran Associates, 2024.

[22] A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. Technical Report arXiv:2312.00752 [cs.LG], 2023.

[23] C. Han, Q. Wang, H. Peng, W. Xiong, Y. Chen, H. Ji1, and S. Wang. LM-Infinite: Zero-shot extreme length generalization for large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 3991–4008, 2024.

[24] L. Hauzenberger, N. Schmidinger, A. Hartl, D. Stap, T. Schmied, B¨ock, S. Klambauer, and S. Hochreiter. KVpop – key-value cache compression with predictive online pruning. Technical Report arXiv:2607.05061 [cs.LG], 2026.

[25] J. Herrmann, O. Beaumont, L. Eyraud-Dubois, J. Hermann, A. Joly, and A. Shilova. Optimal checkpointing for heterogeneous chains: How to train deep neural networks with limited memory. Technical Report arXiv:1911.13214 [cs.LG], 2019.

[26] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.

[27] C. Hooper, S. Kim, H. Mohammadzadeh, M. Mahoney, S. Shao, K. Keutzer, and A. Gholami. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In Globerson et al. [21].

[28] E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In Int. Conf. Learning Representations, 2022.

[29] H. Jiang, Y. Li, C. Zhang, Q. Wu, X. Luo, S. Ahn, Z. Han, A. Abdi, D. Li, C. Lin, Y. Yang, and L. Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In Globerson et al. [21].

[30] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

[31] A. La´ncucki, K. Staniszewski, P. Nawrot, and E. Ponti. Inference-time hyper-scaling with kv cache compression. In Belgrave et al. [5].

[32] S. Li, F. Xue, C. Baranwal, Y. Li, and Y. You. Sequence parallelism: Long sequence training from system perspective. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 7376–7399, 2023.

[33] W. Li, D. Yu, G. Luo, Y. Zhang, Y. Wu, J. Liu, Z. Gong, Z. Liao, F. Chao, and R. Ji. Out of the memory barrier: A highly memory-eficient training system for LLMs with million-token contexts. In Int. Conf. Learning Representations, 2026.

[34] W. Li, Y. Zhang, G. Luo, D. Yu, and R. Ji. Training long-context LLMs eficiently via chunk-wise optimization. In Findings of the Association for Computational Linguistics (ACL), pages 2691–2700, 2025.

[35] X. Li, Z. Xing, M. Li, L. Qu, H. Zhen, Y. Yao, W. Liu, S. Pan, and M. Yuan. KVTuner: Sensitivity-aware layer-wise mixed-precision KV cache quantization for eficient and nearly lossless LLM inference. In Singh et al. [53].

[36] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen. SnapKV: LLM knows what you are looking for before generation. In Belgrave et al. [5], pages 529–536.

[37] Y. Li, H. Jiang, Q. Wu, X. Luo, S. Ahn, C. Zhang, A. Abdi, D. Li, J. Gao, Y. Yang, and L. Qiu. SCBench: A KV cache-centric analysis of long-context methods. In Int. Conf. Learning Representations, 2025.

[38] H. Liu, M. Zaharia, and P. Abbeel. RingAttention with blockwise transformers for near-infinite context. In Int. Conf. Learning Representations, 2024.

[39] Z. Liu, S. Wang, S. Cheng, Z. Zhao, K. Wang, X. Zhao, J. Demmel, and Y. You. StarTrail: Concentric ring sequence parallelism for eficient near-infinite-context transformer model training. In Belgrave et al. [5].

[40] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, D. Chen, and X. Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Salakhutdinov et al. [46], pages 32332–32344.

[41] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. Technical Report arXiv:1711.05101 [cs.LG], 2017.

[42] P. Nawrot, A. Lancucki, M. Chochowski, D. Tarjan, and E. Ponti. Dynamic memory compression: Retrofitting LLMs for accelerated inference. In Salakhutdinov et al. [46], pages 37396–37412.

[43] A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors. Advances in Neural Information Processing Systems 36. Curran Associates, 2023.

[44] B. Peng, J. Quesnelle, H. Fan, and E. Shippole. YaRN: Eficient context window extension of large language models. In Int. Conf. Learning Representations, 2024.

[45] Z. Qin, Y. Cao, M. Lin, W. Hu, S. Fan, K. Cheng, W. Lin, and J. Li. CAKE: Cascading and adaptive KV cache eviction with layer preferences. In Int. Conf. Learning Representations, 2025.

[46] R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors. International Conference on Machine Learning 41. Proceedings of Machine Learning Research, 2024.

[47] T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools. In Oh et al. [43].

[48] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In Globerson et al. [21], pages 68658–68685.

[49] N. Shang, L. Zhang, S. Wang, G. Zhang, G. Lopez, F. Yang, W. Chen, and M. Yang. LongRoPE2: Near-lossless LLM context window scaling. In Singh et al. [53].

[50] N. Shazeer. Fast transformer decoding: One write-head is all you need. Technical Report arXiv:1911.02150 [cs.NE], 2019.

[51] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, D. Fu, Z. Xie, B. Chen, C. Barrett, J. Gonzalez, P. Liang, C. Re, I. Stoica, and C. Zhang. FlexGen: High-throughput generative inference of large language models with a single GPU. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning 40, volume 202. Proceedings of Machine Learning Research, 2023.

[52] A. Shutova, V. Malinovskii, V. Egiazarian, D. Kuznedelev, D. Mazur, S. Nikita, I. Ermakov, and D. Alistarh. Cache me if you must: Adaptive key-value quantization for large language models. In Singh et al. [53].

[53] A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaf, and J. Zhu, editors. International Conference on Machine Learning 42. Proceedings of Machine Learning Research, 2025.

[54] K. Staniszewski and A. La´ncucki. KV cache transform coding for compact storage in LLM inference. In Int. Conf. Learning Representations, 2026.

[55] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568(C), 2024.

[56] H. Sun, L. Chang, W. Bao, S. Zheng, N. Zheng, X. Liu, H. Dong, Y. Chi, and B. Chen. ShadowKV: KV cache in shadows for high-throughput long-context LLM inference. In Singh et al. [53], pages 57355–57373.

[57] Y. Sun, L. Dong, Y. Zhu, S. Huang, W. Wang, S. Ma, Q. Zhang, Z. Wang, and F. Wei. You only cache once: Decoder-decoder architectures for language models. In Globerson et al. [21].

[58] H. Tang, Y. Lin, J. Lin, Q. Han, D. Ke, S. Hong, Y. Yao, and G. Wang. RazorAttention: Eficient KV cache compression through retrieval heads. In Int. Conf. Learning Representations, 2025.

[59] J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, and S. Han. Quest: Query-aware sparsity for eficient long-context llm inference. In Salakhutdinov et al. [46], pages 47901–47911.

[60] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez, L Kaiser, and I. Polosukhin. Attention is all you need. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 6000–6010. Curran Associates, 2018.

[61] G. Wang, S. Upasani, C. Wu, D. Gandhi, J. Li, C. Hu, B. Li, and U. Thakker. LLMs know what to drop: Self-attention guided KV cache eviction for eficient long-context inference. In Int. Conf. Learning Representations, 2025.

[62] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, W. Chi, Q. Le, and D. Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35. Curran Associates, 2022.

[63] T. Wu, Y. Zhao, and Z. Zheng. An eficient recipe for long context extension via middle-focused positional encoding. In Globerson et al. [21].

[64] G. Xiao, J. Tang, J. Zuo, J. Guo, S. Yang, H. Tang, Y. Fu, and S. Han. DuoAttention: Eficient long-context LLM inference with retrieval and streaming heads. In Int. Conf. Learning Representations, 2025.

[65] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. Eficient streaming language models with attention sinks. In Int. Conf. Learning Representations, 2024.

[66] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report. Technical Report arXiv:2505.09388 [cs.CL], 2025.

[67] S. Yang, J. Guo, H. Tang, Q. Hu, G. Xiao, J. Tang, Y. Lin, Z. Liu, Y. Lu, and S. Han. LServe: Eficient long-sequence LLM serving with unified sparse attention. In Proceedings of the 8th MLSys Conference, 2025.

[68] Z. Ye, L. Chen, R. Lai, W. Lin, Y. Zhang, S. Wang, T. Chen, B. Kasikci, V. Grover, A. Krishnamurthy, and L. Ceze. FlashInfer: Eficient and customizable attention engine for LLM inference serving. Technical Report arXiv:2501.01005 [cs.DC], 2025.

[69] H. Yen, T. Gao, M. Hou, K. Ding, D. Fleischer, P. Izsak, M. Wasserblat, and D. Chen. Helmet: How to evaluate long-context models efectively and thoroughly. In Int. Conf. Learning Representations, 2025.

[70] J. Yuan, H. Gao, D. Dai, J. Luo, L. Zhao, Z. Zhang, Z. Xie1, Y. Wei, L. Wang, Z. Xiao, Y. Wang, C. Ruan, M. Zhang, W. Liang, and W. Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. Technical Report arXiv:2502.11089 [cs.CL], 2025.

[71] Z. Yuan, H. Sun, L. Sun, and Y. Ye. MegaTrain: Full precision training of 100b+ parameter large language models on a single GPU. Technical Report arXiv:2604.05091 [cs.CL], 2026.

[72] Zhang Z., Y. Sheng, T. Zhou., T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Re, C. Barrett, Z. Wang, and B. Chen. H2O: Heavy-hitter oracle for eficient generative inference of large language models. In Oh et al. [43], pages 34661–34710.

[73] M. Zaheer, G. Guruganesh, A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed. Big Bird: Transformers for longer sequences. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33, pages 17283–17297. Curran Associates, 2020.

[74] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni. TurboQuant: Online vector quantization with near-optimal distortion rate. In Int. Conf. Learning Representations, 2026.

[75] T. Zhang, J. Yi, Z. Xu, and A. Shrivastava. KV cache is 1 bit per channel: Eficient large language model inference with coupled quantization. In Globerson et al. [21], pages 3304–3331.

[76] Z. Zhang, R. Chen, S. Liu, Z. Yao, O. Ruwase, B. Chen, X. Wu, and Z. Wang. Found in the middle: How language models use long contexts better via plug-and-play positional encoding. In Globerson et al. [21], pages 60755–60775.

[77] Z. Zhang, S.i Liu, R. Chen, B. Kailkhura, B¿ Chen, and A. Wang. Q-Hitter: A better token oracle for eficient LLM inference via sparse-quantized KV cache. In P. Gibbons, G. Pekhimenko, and C. De Sa, editors, Proceedings of Machine Learning and Systems (MLSys), volume 6, pages 381–394, 2024.

[78] Y. Zhao, J. Huang, J. Hu, X. Wang, Y. Mao, D. Zhang, Z. Jiang, Z. Wu, B. Ai, A. Wang, W. Zhou, and Y. Chen. SWIFT: A scalable lightweight infrastructure for fine-tuning. In Proceedings of the 39th Conference on Artificial Intelligence (AAAI), pages 29733–29735, 2025.

[79] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. Gonzalez, C. Barrett, and Y. Sheng. SGLang: Eficient execution of structured language model programs. In Globerson et al. [21], pages 62557–62583.

[80] C. Zhou, K. Liu, Y. Zhou, Q. Qiao, J. Gao, H. Zhang, I. Lu, N. Ho, L. Li, A. Lei, C. Cheng, S. Chiang, Y. Zeng, D. Zhang, R. Yang, K. Chen, A. Chen, P. Ma, W. Zhang, and C. Jin. LongStraw: Long-context RL beyond 2M tokens under a fixed GPU budget. Technical Report arXiv:2607.14952 [cs.LG], 2026.

## A Appendix

## A.1 Notation. Definitions

Here, we add some details missing in the main text.

## A.1.1 Bufer Read/Write Access by torch.scatter, torch.gather

These linear operations are defined in https://docs.pytorch.org/docs/2.12/ generated/torch.Tensor.scatter\_.html. scatter assigns values to entries in certain positions, gather extracts values of entries at certain positions. For reference, arr.scatter (dim, index, src) requires arr.ndim == index.ndim == src.ndim and index.shape == src.shape, whereas arr.shape can difer on position dim. Say that arr.ndim == 3. Then:

$$
\arcsin [ { \mathrm { i n d e x } } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { k } } ] , { \mathrm { j } } , { \mathrm { k } } ] = { \tt s r c } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { k } } ] \quad | d i m = = 0
$$

$$
\arcsin [ { \mathrm { i } } , \operatorname { i n d e x } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { k } } ] , { \mathrm { k } } ] = { \tt s r c } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { k } } ] \quad | \ d i m = = 1
$$

$$
\operatorname { a r r } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { i n d e x } } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { k } } ] ] = { \tt s r c } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { k } } ] \quad | \ d i m = = 2
$$

Also, if res = arr.gather(dim, index), then:

$$
\mathtt { r e s } [ \mathrm { i } , \mathrm { j } , \mathrm { k } ] = \mathtt { a r r } [ \mathrm { i n d e x } [ \mathrm { i } , \mathrm { j } , \mathrm { k } ] , \mathrm { j } , \mathrm { k } ] \quad | \ d i m = = 0
$$

$$
\mathbf { r e s } [ \mathrm { i } , \mathbf { j } , \mathbf { k } ] = \mathsf { a r r } [ \mathrm { i } , \mathrm { i n d e x } [ \mathrm { i } , \mathbf { j } , \mathbf { k } ] , \mathbf { k } ] \quad | d i m = = 1
$$

$$
\mathbf { r e s } [ { \mathrm { i } } , { \mathrm { j } } , \mathbf { k } ] = \mathbf { a r r } [ { \mathrm { i } } , { \mathrm { j } } , { \mathrm { i n d e x } } [ { \mathrm { i } } , { \mathrm { j } } , \mathbf { k } ] ] \quad | d i m = = 2
$$

In our use case, we apply these operations to 4D arrays with dim = 2, so that arr, src are 4D, but index is 3D. This is done by broadcasting index along the final axis: index.unsqueeze(−1).extend(−1, −1, −1, d<sub>h</sub>).

## A.2 Key-Value Cache Policies

In this section, we present additional details about KV cache policies used in our experiments.

## A.2.1 Policy smart lastrec (slr)

Recall the simple lastrec (lr) policy from Section 4, which keeps the last recent $N _ { C } - \beta$ and first $\beta$ tokens in the cache. One drawback of this policy is that $\beta$ is fixed, while prompts often start with initial important control information of variable length. Our smart lastrec policy is defined in terms of a regular expression for the end of this task control prefix, as well as some maximum prefix length $M _ { \mathrm { p r e f i x } } < N _ { C }$ . When processing the first (prefill) chunk, we search for the first match (separately for each batch position $b )$ . If this results in a prefix length $M ( b ) \leq M _ { \mathrm { p r e f i x } }$ , this is used, otherwise $M ( b ) = M _ { \mathrm { p r e f i x } }$ . For subsequent chunks and token positions $t \geq N _ { C }$ , the policy is $\pi ( b , h , t ) = M ( b ) + \mathrm { m o d } ( t - N _ { C } , N _ { C } - M ( b ) )$

We also implemented a generalization where a range $[ M _ { 0 } ( b ) , M _ { 1 } ( b ) )$ is protected from eviction.<sup>11</sup> Here, $M _ { 0 } ( b )$ is chosen as the position of the first non-padding token, and $M _ { 1 } ( b )$ is chosen as above. The idea is that initial padding tokens do not carry information and should not be attended to, so they can be evicted as soon as the cache is full. In this variant, if all tokens in the prefill chunk are padding for some b, the search for $[ M _ { 0 } ( b ) , M _ { 1 } ( b ) )$ is shifted to subsequent chunks. The prefix case above is obtained with $M _ { 0 } ( b ) = 0 .$ Surprisingly, in our experiments, the general variant did not improve over the prefix variant,<sup>12</sup> so that smart lastrec in Section 4 is the prefix variant throughout.

## A.3 Long-Context Benchmarks

## A.3.1 Helmet

Our training and evaluation suite is derived from Helmet [69], a benchmark designed for inference-time evaluation of long-context language models. Helmet covers five capability categories across five context-length scales (8k to 128k tokens). Helmet provides only a small number of instances per task. For supervised fine-tuning, we adapted it as follows.

Instance Construction and Task Scope. We reconstruct each task from its upstream source data, following the original Helmet logic for forming contexts and controlling sequence length. We focus on the 64k and 128k context-length settings and include 10 tasks spanning five capability categories. Table 3 provides a summary of each task.

Split Separation. For each task we produce two non-overlapping partitions. The instances used in the original Helmet evaluation are reserved as a held-out evaluation (test) set. All remaining instances are collected into a development set used for training. For RAG tasks, we sample a single depth (dificulty) variant per query in the development set to prevent the model from memorising the same question paired with multiple distractor configurations. For InfiniteBench QA/MC, we remove in-context demonstrations from development instances because the demonstrations are drawn from the same small pool and would otherwise create data leakage during training.

Evaluation Metrics. Our evaluation metrics are taken from the code coming with Helmet (https://github.com/princeton-nlp/HELMET/blob/main/utils.py). Here, the out put is a string generated by the model, the target is a string or a list of strings:

• SubEM: Depending on the dataset, the target can be a list of strings. We normalize the output (strip whitespace, quotes, and common phrases such as “Answer:”), map output and target(s) to lower-case. The value is 1 if at least one of the targets is a substring in the output, 0 otherwise.

• Accuracy: The target is a numerical value. We extract all numerical values from the output and find the value which occurs most often (with ties, the value is chosen which appears first). We then use exact match between this value and the target.

• ROUGE-F1: The target is a string. We compute ROUGE-N precision/recall/F1 between output (after normalization as in SubEM) and target.

<table><tr><td>Category</td><td>ID</td><td>Source</td><td>Metric</td><td>Dev</td><td>Eval</td></tr><tr><td rowspan="4">RAG</td><td>nq</td><td>Natural Questions</td><td>SubEM</td><td>893</td><td>600</td></tr><tr><td>trivia_qa</td><td>TriviaQA</td><td>SubEM</td><td>876</td><td>600</td></tr><tr><td>pop-qa</td><td>PopQA</td><td>SubEM</td><td>192</td><td>600</td></tr><tr><td>hotpot_qa</td><td>HotpotQA</td><td>SubEM</td><td>787</td><td>300</td></tr><tr><td rowspan="3">Many-shot ICL</td><td>trec_coarse</td><td>TREC</td><td>Accuracy</td><td>1000</td><td>500</td></tr><tr><td>nlu</td><td>SNIPS NLU</td><td>Accuracy</td><td>2094</td><td>500</td></tr><tr><td>clinc150</td><td>CLINC150</td><td>Accuracy</td><td>2600</td><td>500</td></tr><tr><td rowspan="2">Long-doc QA</td><td>inf_qa</td><td>InfiniteBench QA</td><td>ROUGE-F1</td><td>251</td><td>100</td></tr><tr><td>inf_mc</td><td>InfiniteBench MC</td><td>Accuracy</td><td>129</td><td>100</td></tr><tr><td>Synthetic Recall</td><td>json_kv</td><td>JSON-KV</td><td>SubEM</td><td>500</td><td>100</td></tr></table>

Table 3: Overview of the 10 Helmet tasks. Dev and Eval denote the number of instances in the training and evaluation partitions, respectively, at a single context-length setting.

In what follows, we provide details for the diferent tasks.

Retrieval-augmented Generation (RAG). Each instance consists of a naturallanguage question, one or more gold passages, and a pool of hard-negative distractors. The context is formed by inserting gold passages at a random depth among the distractors, and the whole context is truncated to the target length. Natural Questions (NQ) uses real Google search queries paired with Wikipedia passages. TriviaQA uses trivia questions authored with independently collected evidence. PopQA focuses on long-tail, entity-centric questions generated from Wikidata triples; we filter the evaluation set to queries whose subject entities fall below a popularity threshold of 3. HotpotQA requires multi-hop reasoning across two gold passages. Each of the six depth variants of a query places the gold passage(s) at a diferent relative position in the distractor pool. All four tasks are evaluated with substring exact match (SubEM).

Many-shot In-context Learning. Three intent/question-type classification datasets test the model’s ability to exploit many labelled demonstrations placed entirely within the context window. Unlike most other tasks, where demonstrations serve only as formatting guides, here they carry essential semantic information: each demonstration encodes a (text, ordinal-label) pair, and the label-to-class mapping is only recoverable by reading the demonstrations. TREC Coarse has 6 question-type classes; SNIPS NLU has 68 intent classes; CLINC150 has 151 intent classes. The number of demonstrations is calibrated to fill the context window while maintaining an approximately balanced class distribution across shots.

Long-document QA. Both InfiniteBench QA (Inf-QA) and InfiniteBench MC (Inf-MC) are derived from full-length novels whose named entities have been replaced by synthetic ones to prevent answer memorisation. The source document typically exceeds the target context window, so it is truncated to fit. Inf-QA is open-ended, evaluated by ROUGE-F1; Inf-MC is a 4-way multiple-choice variant evaluated by accuracy.

Synthetic Recall. JSON-KV asks the model to retrieve the value associated with a specified key from a large JSON dictionary that fills the context window. This task is evaluated with SubEM and serves as controlled probes of the model’s ability to locate and copy specific information over very long spans.

## A.4 Gradient Computation

In this section, we provide additional details about our long-context gradient computation method from Section 3.2.

## A.4.1 Chunks, Cells, CPU Ofloading

Recall that for long-context inference or fine-tuning with cache length $N _ { C } .$ , we split a sequence of length $N > N _ { C }$ into $1 + \lceil ( N - N _ { C } ) / S \rceil$ chunks, the first (prefill) chunk of length $N _ { C }$ , subsequent chunks of length $S ~ < ~ N _ { C }$ . The chunk size S is chosen according to a latency-vs-accuracy trade-of, it is in general much shorter than $N _ { C }$ (Section 3.3). We also group chunks into cells. The first cell consists of the prefill chunk alone, subsequent cells group $k = \lfloor \alpha N _ { C } / S \rfloor$ of S-length chunks, where $\alpha > 0$ is a hyperparameter which defaults to $\alpha = 1$

In our gradient computation method, PyTorch autograd is run on cells. While $S$ is chosen also with accuracy in mind, the choice of k and α is determined by eficiency (both runtime and memory) only. The idea is that the autograd GPU memory requirements are on the order of one KV cache bufer, if we exploit the linear recurrence (see Section 3.2 and below). If a cell was much larger, the delta nodes in the computation graph would dominate. The α parameter is adjusted so to not run out of memory. Empirically, a smaller chunk size S necessitates a smaller α, likely due to overhead in autograd (for constant α, smaller S means larger k, so a larger computation graph). In our experiments, we chose $\alpha = 1$ for $S = 2 0 4 8$ 2 $\alpha = 0 . 7 5$ for $S = 1 0 2 4$ , and $\alpha = 0 . 1$ for $S = 1 2 8$ , for a cache length of $N _ { C } = 3 2 7 6 8$ and 40 GB of GPU memory.

The grouping of chunks into cells is also relevant for long-context inference. Recall that L denotes the number of layers of our model, and layer inputs are of shape $( B , N , d )$ , where $d = H _ { q } \cdot d _ { h }$ is the model embedding dimension. For large N, not even the inputs to one layer can be kept in GPU memory. This has implications for how inference is computed and what can be ofloaded to CPU memory at which point during the process (our library supports CPU ofloading of KV cache bufers during the forward passes, as well as CPU ofloading of weights during the backward pass, but this is not used in the experiments reported here).

Outer loop over chunks, inner loop over layers: This seems the simplest ordering. Also, layer inputs can be kept in GPU memory, with outputs overwriting inputs. A major drawback is that either all KV cache bufers need to be kept in GPU memory, or they need to be read from and written back to CPU memory frequently. When KV cache bufers are stored in quantized form, the quantization computation can be considerable. For longcontext inference, this ordering is not suitable.

Outer loop over layers, inner loop over chunks: With this ordering, KV cache bufers (and even model weights) can be ofloaded to CPU for all but the currently active layer. A drawback of this ordering is that layer inputs and outputs cannot be kept in GPU memory in total, so need to be ofloaded eventually. Still, this ordering is much better than the previous one.

Outer loop over cells, middle loop over layers, inner loop over chunks per cell: This ordering provides a good compromise between the two previous ones. Layer inputs and outputs can be kept in GPU memory, while KV cache bufers can be quantized and/or ofloaded to CPU, which happens much less frequently. We use this ordering in our implementation, also because CPU ofloading of KV cache bufers is needed during gradient computation, so we can just reuse this code.

## A.4.2 Summary of Method

Here, we present a detailed summary of our gradient computation technique. Recall that we process a batch of B sequences of token length $N _ { ; }$ and that caches in each layer have length $N _ { C }$ . For $N \leq N _ { C }$ , our method reduces to standard training, so assume that $N > N _ { C }$ . For simplicity, we assume that caches in all layers have the same length $N _ { C }$ . This is easy to relax, and our implementation does so.

At the top level, our method runs a forward pass followed by a backward pass, just like standard code. However, we run autograd on cells only, which constitute small parts of the overall model graph. As with activation checkpointing [25], this means we need to run forward passes over the model three times (instead of just once). The first two passes are run in non-autograd mode, checkpointing (so called) boundary information to CPU memory. The third pass is part of the autograd runs on cells, which consume boundary information as inputs.

We need some notation. Let $\mathcal { L }$ denote the training loss for the current batch. Layers are indexed by $l = 0 , \ldots , L - 1$ , cells by $c = 0 , \ldots , N _ { \mathrm { c e l l s } } - 1 . \ X _ { l }$ are inputs to layer $l ,$ of shape $( B , N , d )$ , and $X _ { L }$ is the top layer output. Moreover, $X _ { l , c }$ denotes the slice of $X _ { l }$ along axis 1 corresponding to the cell. Finally, let $\pmb { K } _ { l , c }$ denote the KV cache bufers<sup>13</sup> of shape $( 2 , B , H _ { k } , N _ { C } , d _ { h } )$ at the input of cell $c > 0$ in layer l.

Forward pass 1. This runs in non-autograd mode, using the ordering cells, then layers, then chunks detailed in Section A.4.1. Alongside:

• Store KV cache replay log in each layer, containing the decisions $\{ \pi ( b , h , t ) \}$

• Checkpoint layer inputs $X _ { l }$ to CPU memory for each layer $l = 0 , \ldots , L - 1$ . Our implementation allows to quantize them in order to save CPU memory and CPU-$\mathrm { G P U }$ transfer time, but this is not activated in our experiments, since the time is subdominant. We also checkpoint top layer outputs $X _ { L }$

Backward pass. This runs backwards over layers. In each step, we compute gradients for weights in layer l using activation checkpointing. We start with computing head gradients $\partial \mathcal { L } / \partial X _ { L }$ based on top layer outputs $X _ { L } ,$ writing them to CPU (in fact, head gradients $\partial \mathcal { L } / \partial X _ { l }$ overwrite $X _ { l }$ on CPU). Next, we iterate over layers $l = L - 1 , \ldots , 0 \colon$

• Forward pass 2 for layer l (non-autograd mode). Runs over cells $c = 1 , \ldots , N _ { \mathrm { c e l l s } } - 1$ computing the cache bufers $\pmb { K } _ { l , c }$ and storing them to CPU. These are quantized in order to save CPU memory and CPU-GPU transfer time, and we overwrite the cache bufer checkpoints from previous layer l + 1. Cache decisions are replayed from the log. Note that we could checkpoint all $\pmb { K } _ { l , c }$ during forward pass 1, but this would require L times more CPU memory, and the extra time for forward pass 2 is subdominant. Also note that we load inputs $X _ { l , c }$ to forward pass 2 to GPU cell by cell: the whole $X _ { l }$ does not fit in GPU memory (see Section A.4.1).

• Run autograd on each cell, iterating from right to left, $c = N _ { \mathrm { c e l l s } } - 1 , \ldots , 0$ . For each cell $c ,$ we load layer inputs $X _ { l , c }$ (bottom), layer head gradients $\partial \mathcal { L } / \partial X _ { l + 1 , c } ~ ( \mathrm { t o p } )$ and incoming cache bufers $\pmb { K } _ { l , c }$ (left; only for $c > 0 )$ from CPU, while cache bufer head gradients $\partial \mathcal { L } / \partial K _ { l , c + 1 }$ (right; only for $c < N _ { \mathrm { c e l l s } } - 1 )$ are kept in GPU memory. Cache decisions are replayed from the log. autograd works as follows:

– Forward pass 3 for cell $( l , c )$ , in autograd mode. Whenever a KV cache bufer node (keys, values) is created, we store an annotation in a list. In the pack hook function, we match arguments of the right shape against all annotation. For a match, we replace the argument with its delta encoding, which is stored in the computation graph instead. See Section A.4.3 for details.

– Backward: When PyTorch traverses the computation graph in reverse order, it calls unpack hook for each node. For each delta encoding, we play the recurrence (4) backwards, replacing keys<sup>′</sup> by keys or values<sup>′</sup> by values.

The outcome of autograd on cell $( l , c )$ are gradients w.r.t. layer weights (which are accumulated), a head gradient $\partial \mathcal { L } / \partial X _ { l , c }$ (written to CPU, overwriting $X _ { l , c } )$ , and a head gradient $\partial \mathcal { L } / \partial K _ { l , c }$ which replaces the previous one (for $c > 0 )$ . At the end of the loop over cells, gradients w.r.t. layer weights are complete, and the head gradient $\partial \mathcal { L } / \partial K _ { l }$ is on CPU, so the layer below can be addressed (or, for l = 0, gradients w.r.t. input embeddings can be computed based on $\partial \mathcal { L } / \partial K _ { 0 } )$

## A.4.3 Exploiting Linear Recurrence of KV Cache Bufers

Recall the recurrence between KV cache bufers for subsequent chunks from Section 3.2. We can exploit this recurrence by asking PyTorch to store delta key instead of keys and delta value instead of values in the computation graph, restoring the latter from the former during backward using (4).

While this is simple in principle, we need to do it inside PyTorch autograd. To this end, we use a mechanism called autograd saved tensors hooks (https://docs.pytorch.org/ tutorials/intermediate/autograd\_saved\_tensors\_hooks\_tutorial.html). This was designed in order to implement activation checkpointing by CPU ofloading, but can be used for our purposes as well. It works by allowing the specification of two functions:

• pack hook $( { \pmb x } )  { \pmb p } ( { \pmb x } )$ : When building its computation graph during forward, this function is called for every array x $\mathrm { P y }$ Torch plans to store in the computation graph. It then stores ${ \pmb p } ( { \pmb x } )$ in the graph instead of x.

• unpack hook $( p ) \to x ( p )$ : When traversing the computation graph in reverse order during backward, PyTorch calls this function for every array p stored in the graph. It then uses $\scriptstyle { \pmb { x } } ( { \pmb { p } } )$ instead of $_ p$

A major dificulty for us is the non-selectiveness of this mechanism. We do not want to pack all arrays stored in the graph, but only specific ones: the KV cache bufers (for all other nodes, we just pass through $\pmb { p } ( \pmb { x } ) = \pmb { x }$ and $\pmb { x } ( \pmb { p } ) = \pmb { p } )$ . But pack hook(x) just takes a torch.Tensor argument, there is no obvious way for tagging the nodes we want. Also, there is some delay between a node being created in the forward pass and pack hook(x) being called for it, we even detected some diferences in the relative ordering. Finally, due to internal operator fusion, we cannot even be sure whether any node appearing in the forward code is indeed stored in the graph.

Our implementation maintains an annotation list, which is appended to during the forward code, while entries are removed during pack hook calls. Whenever a KV cache bufer update in the form of a statement keys<sup>′</sup> = scatter(keys, index, key new) is passed in the forward code, we compute delta key = gather(keys, index), appending an annotation containing (index, delta key) and some meta-data to the list. Here, delta key serves a double role. First, it is needed to reconstruct keys from ${ \tt k e y s } ^ { \prime }$ in unpack hook. Second, it $\mathrm { s e r v e s ^ { 1 4 } }$ as a ”fingerprint” of keys. Namely, when pack hook(x) is called, we need to match the argument x against annotations. This is first done by shape, filtering out most calls. Next, for any annotation (index, delta key), we check whether gather(x, index) = delta key. If so, we return ${ \pmb p } ( { \pmb x } )$ containing delta key and remove the annotation from the list. If there is no match, we return $\pmb { p } ( \pmb { x } ) = \pmb { x }$

For unpack hook(p), we reconstruct the sequences keys and values in reverse order. If p is not a packed object, we return $\pmb { x } ( \pmb { p } ) = \pmb { p }$ . Otherwise, we check the chunk number stored with p against the current state (keys<sup>′</sup>, values<sup>′</sup>). If this fits, we reconstruct keys from (keys<sup>′</sup>, delta key) or values from (values<sup>′</sup>, delta value), using (4). The new bufer overwrites the old one.

Our implementation tracks which pack hook arguments of the right shape are not matched by annotations, and which annotations are not matched. Both events do happen, but at a low rate. It is important to note that we still obtain correct results even if some arguments are not matched. This just means that a bit more GPU memory is being used. What we need to avoid, however, are false matches, which we do by keeping fingerprints large enough.

An important direction for future work is to simplify and robustify the mechanism for exploiting the linear recurrences. We tried several simplifications. One assumes that KV bufer nodes are created in exactly the same order as pack hook(x) calls. If true, matching and managing the annotation list would be much simplified. Unfortunately, this does not hold true, likely due to internals of PyTorch autograd we have no influence over. In fact, the instruction keys<sup>′</sup> = scatter(keys, index, key new) need not even trigger a call of pack hook with $\pmb { x } = \mathtt { k e y s } ^ { \prime }$ . The array keys<sup>′</sup> is processed further inside the SDPA code, and PyTorch may use some form of operator fusion. The simplest solution would be to tag each KV bufer node during creation in a way that allows us to recognize the tag in a pack hook argument x. However, we did not find a way to do that yet.

## A.5 Experimental Results

In this section, we provide additional experimental results beyond what is shown in the main text, as well as timing figures. We also explain a failure mode we consistently observed when fine-tuning a model with RingAttention to be used for inference with sparse attention.

## A.5.1 Additional Results

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64k datasets</td><td rowspan=1 colspan=1>128k datasets</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>nqtri_qahot_qapop-qa</td><td rowspan=1 colspan=1>nqtri_qahot_qa pop-qa</td></tr><tr><td rowspan=1 colspan=1>exact</td><td rowspan=1 colspan=1>-       -        -         -</td><td rowspan=1 colspan=1>-       -        -         -</td></tr><tr><td rowspan=1 colspan=1> $\ln _ { 2 k }$ </td><td rowspan=1 colspan=1>47.5   80.2    57.7     53.8</td><td rowspan=1 colspan=1>35.3   70.5    37.0     40.2</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { s l r } _ { 2 k }$ </td><td rowspan=1 colspan=1>45.3  80.2    55.3    52.0</td><td rowspan=1 colspan=1>37.5   69.7   34.7    34.8</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { h 2 o _ { 2 } } _ { k }$ </td><td rowspan=1 colspan=1>42.8   70.8   44.7    59.0</td><td rowspan=1 colspan=1>20.5   63.2   12.3    27.3</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { h 2 o _ { 2 } ^ { n o } }$ </td><td rowspan=1 colspan=1>46.0   82.2   52.7    59.3</td><td rowspan=1 colspan=1>42.2   78.5   34.3    39.3</td></tr><tr><td rowspan=1 colspan=1> $\frac { \mathrm { h 2 o } _ { 2 k } ^ { \mathrm { o r } } } { \mathrm { 2 } k }$ </td><td rowspan=1 colspan=1>44.2   74.7    47.7    61.0</td><td rowspan=1 colspan=1>43.7   79.8    25.0    43.3</td></tr><tr><td rowspan=1 colspan=1> $\ln _ { 1 k }$ </td><td rowspan=1 colspan=1>47.5   80.2    59.7    54.0</td><td rowspan=1 colspan=1>35.8   70.2    37.7    37.5</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { s l r } _ { 1 k }$ </td><td rowspan=1 colspan=1>49.0  79.0    55.0    54.3</td><td rowspan=1 colspan=1>37.3  67.3   32.0    35.7</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { h 2 o } _ { 1 k }$ </td><td rowspan=1 colspan=1>43.2   71.7    41.7    59.8</td><td rowspan=1 colspan=1>21.0   62.8   13.0    27.0</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { h 2 o } _ { 1 k } ^ { \mathrm { n o } }$ </td><td rowspan=1 colspan=1>47.0  81.2    50.0    56.7</td><td rowspan=1 colspan=1>40.0  79.8   32.7    42.3</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { h 2 o } _ { 1 k } ^ { \mathrm { o r } }$ </td><td rowspan=1 colspan=1>47.5  76.3    46.7    58.2</td><td rowspan=1 colspan=1>41.8  79.8    23.7    39.3</td></tr></table>

Table 4: Results for long-context inference with 5 KV cache policies and chunk sizes $2 0 4 8 = 2 k , 1 0 2 4 = 1 k$ (rows). Here, the base checkpoint Qwen3-4B-Instruct-2507 is used without fine-tuning. The first row exact is for exact inference (sequence parallelism). We show sub exact match values on test splits for diferent Helmet datasets nq, trivia qa, hotpot qa, pop qa, limiting sequence lengths to 64k or 128k tokens.

In Table 4, we provide results on Helmet 64k and 128k datasets for base checkpoints Qwen3-4B-Instruct-2507 (no fine-tuning). They should be related to results in Table 1, where the base model was trained by our method (columns us) or by sequence parallelism (columns sp).

In Table 5, we provide results on Helmet 64k and 128k datasets for setups not covered in the main text (see Table 1). slr $\mathrm { 1 2 8 , h 2 o _ { 1 2 8 } , h 2 o _ { 1 2 8 } ^ { n o } , h 2 o _ { 1 2 8 } ^ { o r } }$ use chunk size $S = 1 2 8$ . This runs significantly longer than $S \in \{ 1 0 2 4 , 2 0 4 8 \}$ used in the main text experiments, but allows the KV cache policy to make decisions 8 or 16 times more frequently. However, at least in the experiments here, this does not lead to better results, justifying our choice of larger chunk sizes above. We also ran experiments with Q-Hitter [77], where KV cache bufers are quantized (to 8 bits in our experiments) and the decision score is a convex combination of (3) and a term quantifying the quantization error. The results are consistently worse than for the H2O variants.

<table><tr><td></td><td colspan="3">64k datasets</td></tr><tr><td></td><td>nq tri_qa</td><td>hot_qa pop_qa</td><td></td></tr><tr><td> $\mathrm { s l r } _ { 1 2 8 }$ </td><td>38.8 66.2</td><td>51.7</td><td>41.0</td></tr><tr><td> $\mathrm { { h 2 o _ { 1 2 8 } } }$ </td><td>44.5 65.3</td><td>55.7</td><td>54.3</td></tr><tr><td> $\mathrm { h 2 o _ { 1 2 8 } ^ { n o } }$ </td><td>49.0 74.3</td><td>55.0</td><td>56.3</td></tr><tr><td> $\mathrm { h 2 o _ { 1 2 8 } ^ { o r } }$ </td><td>46.8 67.3</td><td>48.3</td><td>52.3</td></tr><tr><td> $\mathrm { q h 2 o _ { 2 k } }$ </td><td>37.5 64.0</td><td>40.3</td><td>53.7</td></tr><tr><td> $\mathrm { q h 2 o _ { 2 k } ^ { n o } }$ </td><td>40.2 65.5</td><td>46.3</td><td>53.5</td></tr></table>

Table 5: Results for long-context inference with setups not covered in the main text. We show sub exact match values on test splits for diferent Helmet datasets nq, trivia qa, hotpot qa, pop qa, limiting sequence lengths to 64k or 128k tokens. slr<sub>128</sub>, $\mathrm { { h 2 o _ { 1 2 8 } } }$ $\mathrm { h 2 o _ { 1 2 8 } ^ { n o } }$ $\mathrm { h 2 o _ { 1 2 8 } ^ { o r } }$ use chunk size $S = 1 2 8 .$ $\mathrm { q h 2 o _ { 2 k } }$ and $\mathrm { q h 2 o _ { 2 k } ^ { n o } }$ are variants of Q-Hitter [77].
<table><tr><td></td><td>nq</td><td>tri_qa</td><td></td><td>hot_qa</td><td>pop-qa</td></tr><tr><td>exact</td><td>258.38 (15.14)</td><td>266.05</td><td>(11.31)</td><td>262.53 (8.18)</td><td>236.74 (21.44)</td></tr><tr><td>lr2k  $\mathrm { s l r } _ { 2 k }$   $\mathrm { h 2 o _ { 2 } } _ { k }$   $\mathrm { h 2 o } _ { 2 k } ^ { \mathrm { n o } }$   $\mathrm { h 2 o } _ { 2 k } ^ { \mathrm { o r } }$ </td><td>326.07 (21.86) 323.84 (21.19) 330.83 (21.98) 331.48 (22.10) 331.83 (22.25)</td><td>333.28 333.03 (16.19) 344.83 (16.70) 341.71 (16.65) 344.85 (16.89)</td><td>(16.56) 338.46</td><td>330.76 (22.49) 330.29 (22.55) 336.79 (23.13) 337.77 (23.12) (23.25)</td><td>312.29 (27.71) 310.73 (24.46) 316.02 (27.66) 317.60 (25.16)</td></tr><tr><td> $\ln _ { 1 k }$   $\mathrm { s l r } _ { 1 k }$   $\mathrm { h 2 o } _ { 1 k }$   $\mathrm { h 2 o } _ { 1 k } ^ { \mathrm { n o } }$   $\mathrm { h 2 o } _ { 1 k } ^ { \mathrm { o r } }$ </td><td>364.10 (23.28) 362.84 (23.20) 378.85 (24.88) 378.61 (24.99) 378.65 (24.79)</td><td>374.80 (18.62) 370.55 (18.66) 385.19 (20.21) 385.29 (19.32) 385.64 (19.46)</td><td></td><td>375.38 (26.38) 371.18 (25.77) 382.83 (26.94) 388.77 (27.69) 382.52 (26.91)</td><td>316.94 (25.34) 343.96 (27.99) 344.77 (27.65) 359.51 (32.00) 357.95 (29.02) 358.99 (31.74)</td></tr></table>

Table 6: Running time figures for training update step, for Helmet 128k datasets (columns), 5 KV cache policies and chunk sizes $2 0 4 8 = 2 k , 1 0 2 4 = 1 k$ (rows). Batch size $^ { 8 , }$ running on 4 devices. The step from 2k to 1k is 11% to 13% more expensive for lr, slr, 12% to 14% more expensive for h2o variants. The step from lr, slr to h2o variants is 2% to 3% more expensive for 2k, 3% to 4% more expensive for 1k.

## A.5.2 Running Time Figures

In this section, we present running time figures. First, we consider training updates (batch size 8; four Nvidia A100s with 40 GB each). For Helmet 128k datasets, exact uses sequence parallelism with batch size 2, processing 4 micro-batches sequentially, whereas our method (for diferent cache logics) processes 4 micro-batches in parallel.

First, our method is about 30% more expensive than exact for chunk size $S = 2 0 4 8 \ ( 2 \mathrm { k } )$ Given that our method computes gradients on a single GPU independent of the sequence length, using advanced cache logics such as H2O, nested activation checkpointing and delta encoding of cache bufers, this overhead is surprisingly small. Reasons for the overhead are explained in Section 3.3. The gap can likely be narrowed further by operator fusion, increasing the chunk size autograd is operating with. Next, we would expect S = 1024 (1k) to run slower than S = 2048 (2k), because more chunks need to be processed sequentially; and H2O policies to run slower than lr, slr, because summed attention weights are required, and scores need to be computed and sorted. We see that chunk size 1k variants run between 11% and 14% longer than 2k variants, which is substantial. On the other hand, H2O variants are only between 2% to 4% slower than lr, slr. At least with our fast SDPA implementation, there is no penalty for using more advanced policies over simple baselines.

## A.5.3 Analysis of Errors

In this paper, we compare diferent ways of fine-tuning a model to be used with sparse attention inference under diferent KV cache logics: training by sequence parallelism (sp) versus training with the new, resource-eficient technique developed here (us). While for datasets nq, trivia qa, hotpot qa, pop qa coming with the SubEM metric, results are inconclusive (see Table 1), us strongly outperforms sp on datasets trec coarse, nlu, clinc150, inf qa, inf mc, json kv, where the metric is mostly Accuracy (see Table 2). In this section, we identify a consistent failure mode of sp, both via randomly chosen examples and statistics on the generated samples.

Recall details about Helmet datasets from Section A.3.1. For trec coarse, nlu, clinc150, inf mc, targets are single integers, and the Accuracy metric requires the correct number to appear most frequently in the output. For nq, trivia qa, hotpot qa, pop qa, json kv, targets are lists of noun phrases of one of few words, and the SubEM metric requires one of the targets to be a substring of the output. As with all sample-based metrics in our experiments, we allow the model to generate up to 128 tokens, or otherwise terminate by emitting the <eos> token. A closer look at generated samples reveals a major failure mode of sp: its outputs are far too long and contain mostly random nonsense. Here are some randomly chosen examples of tasks and outputs under us and sp respectively.

trec coarse / smart lastrec 1024:   
− idx : 10   
r a w t a r g e t : ”2”   
exact match us : 1   
output us : ”2”   
exact match sp : 0   
output sp : ” 2   
What i s the average l i f e expectancy o f a fe m a l e i n I r e l a n d i n 1996 ?   
1   
What i s the name o f the American who was captured when h i s plane went down over   
Syrian−held Lebanon ?   
1   
0   
What i s the f u l l name o f the PLO ?   
1   
3   
What i s the d i f f e r e n c e between a c o l l e g e and a u n i v e r s i t y ?   
1   
4   
What i s the o r i g i n o f the term ‘ ‘ The Brooklyn Si d e ’ ’ ’ ’ i n bowling ?   
1   
4   
What i s t h e o r i g i n o f t h e word ‘ ‘ a s s a s s i n a t e ’ ’ ’ ’ ?   
1   
4   
What does the a b b r e v i a t i o n cwt . ? mean ?   
1   
3   
What does the ”   
− idx : 318   
r a w t a r g e t : ”5”   
e x a c t m a t c h u s : 1   
output us : ”5”   
exact match sp : 0   
output sp : ” 1   
What does the acronym CPR mean ?

l a b e l : 3   
What does the a b b r e v i a t i o n cwt . ?   
3   
What i s the o r i g i n o f the name ‘ Mabel ’ ’ ?   
3   
What i s the f u l l form o f . com ?   
3   
What i s the name o f the company th a t manufactures the ‘ ‘ American G i r l ’ ’ ’ ’ d o l l   
c o l l e c t i o n ?   
4   
How many p e o p l e were k i l l e d i n t h e 1974 f i r e a t t h e Pentagon ?   
4   
What i s t h e p u r p o s e o f a c a r bra ?   
4   
What i s t h e d i f f e r e n c e b e t w e e n a c l a s s i c a l and an a c o u s t i c g u i t a r ?   
1   
What i s the name o f the company that manufactured the f i r s t phonograph ?   
4”

## clinc150 / smart lastrec 1024:

− idx : 25   
r a w t a r g e t : ”27”   
exact match us : 1   
output us : ”27”   
exact match sp : 1   
output sp : ” 27   
how long u n t i l my next day o f f   
l a b e l | 15”   
− idx : 78   
r a w t a r g e t : ”8”   
e x a c t m a t c h u s : 1   
output us : ”8”   
exact match us : 0   
o u t p u t s p : ” 8\ n\ ncan you f i n d me a r e c i p e f o r c h o c o l a t e c h i p c o o k i e s \ n l a b e l l e 66 \   
\ \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66\   
\n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n\   
66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n66 \n”

## nlu / smart lastrec 1024:

− idx : 63   
r a w t a r g e t : ”61”   
exact match us : 1   
output us : ”61”   
exact match sp : 0   
output sp : ” 61   
what i s the weather i n the c i t y   
graph   
57   
how f a r i s i t from h er e to new york   
graph   
25   
how many c a l o r i e s in a banana   
graph : 6   
what i s the d e f i n i t i o n o f the word pda   
graph : 11   
what i s the b e s t movie to watch t h i s weekend   
graph : 2   
what i s the time d i f f e r e n c e between e a s t e r n and western   
graph : 49   
tu rn on th e wemo p lu g   
graph : 29   
what ’ ’ s the weather i n boston   
graph : 57   
what ’ ’ s t h e w eath er i n new york   
graph : ”   
− idx : 245   
r a w t a r g e t : ”27”   
exact match us : 1   
output us : ”27”   
exact match sp : 1   
output sp : ” 27   
what i s the exchange r a t e between us and canada   
graph : 5   
what ’ ’ s the name o f the person behind the song o f t a y l o r s w i f t back to december   
graph : 52   
what ’ ’ s the weather i n tampa   
graph : 27   
what i s the weather f o r e c a s t f o r the week   
graph : 27   
what ’ ’ s the weather going to be l i k e today in tampa   
graph : 27   
i would l i k e to know the weather in new york   
graph : 27   
what i s the time i n the p a c i f i c time zone   
graph : 63   
what i s the ”

json kv / h2onorm 1024:   
− idx : 81   
r a w t a r g e t : d823df41 −5af0 −4f11 −925d−d0e8018cc5be   
sub exact match us : 1   
output us : d823df41 −5af0 −4f11 −925d−d0e8018cc5be   
s u b e x a c t m a t c h s p : 0   
output sp : ’ 923928d−2e95 −426c−8893−b5e80880a88c   
j s o n   
{” c ” : ”11984809639058896” , ”a ” : ”923928d−2e95 −426c−8893−b5e80880a88c ” , ”b ” : ”0 e2b980e−da28−4af1 −8896−3 f0 c ’   
idx : 51   
r a w t a r g e t : d8cdc4a6−e37e −4cd3−b603−ad33233518e9   
sub exact match us : 0   
output us : d8cdc4a6−e37e −4cd3−b6d3−ad33233b18e9   
sub exact match sp : 0   
output sp : ” 033 a8782 −23d2−488c−ae99 −488f80bbc7d6 \n\nKey : 9 c71e7e3 −9f60 −47f4−baaa −42dbca3e2715 : \   
\ \” d477445a −0f1b −4729−b586−10cbca0a5ba3 \” ,\n \”9 f71e7e3 −9f60 −47f4−baaa−4”   
nq:   
− idx : 418   
r a w t a r g e t :   
− s i x   
− e i g h t   
sub exact match us : 1   
output us : ” s i x ”   
s u b e x a c t m a t c h s p : 0   
output sp : ” 4 hoops are used in a game o f croquet . ( 2 blue , 1 red ”   
idx : 419   
r a w t a r g e t :   
− s i x   
− e i g h t   
s u b e x a c t m a t c h u s : 0   
output us : ” fo u r ”   
s u b e x a c t m a t c h s p : 0   
output sp : ” 20 hoops ( 10 per s i d e ) are used in a game o f croquet . ”   
pop qa:   
− idx : 275   
r a w t a r g e t :   
Paraguay   
Republic o f Paraguay   
py   
”\U0001F1F5\U0001F1FE”   
Heart o f South America   
sub exact match us : 0   
output us : ”Peru”   
sub exact match sp : 0   
output sp : ” Peru   
Question : What i s the c a p i t a l o f Peru ?   
Answer : Lima   
Question : In what r e g i o n ”   
idx : 273   
r a w t a r g e t :   
Paraguay   
Republic o f Paraguay   
py   
”\U0001F1F5\U0001F1FE”   
Heart o f South America   
s u b e x a c t m a t c h u s : 0   
output us : ”Peru”   
s u b e x a c t m a t c h s p : 0   
output sp : ” Peru   
Q u e s t i o n : What i s t h e name o f t h e P eruv ian c i t y where t h e N a t i o n a l L i b r a r y i s   
l o c a t e d ?”

Most other examples we inspected reveal the same failure mode. While us learns to output exactly the numerical answer or noun phrase and nothing else, sparse attention inference for sp tends to output a lot of random content. Recall that during sp training, each token can attend to any earlier one in principle. Plugging in a cache eviction logic afterwards seems to diminish the model’s ability to correctly stop generation.<sup>15</sup> Sometimes, the first num ber in the output is correct, but is followed by many others in the output. For clinc150, idx:25, we have exact match sp = 1 despite the output being partly random and containing another number. For nlu, idx:245, the correct answer 27 appears most frequently in

nonsense output. In the pop qa example, while us gets the country wrong (”Peru” instead of ”Paraguay”), sp also outputs nonsense extra content after the single word. For json kv, idx:51, the output for us is only of by two letters, while that for sp is nonsense, containing several UUIDs completely diferent from the target.
<table><tr><td rowspan="2"></td><td rowspan="2">trn</td><td colspan="2"> $\mathrm { s l r } _ { 1 k }$ </td><td colspan="2"> $\overline { { \mathrm { h 2 o } _ { 1 k } ^ { \mathrm { n o } } } }$ </td><td colspan="2"> $\mathrm { h 2 o } _ { 1 k } ^ { \mathrm { o r } }$ </td></tr><tr><td>R</td><td>P128</td><td>R</td><td>P128</td><td>R</td><td>P128</td></tr><tr><td>nq</td><td>us</td><td>1.1±0.8</td><td>0.0±0.0</td><td>1.1±1.0</td><td>0.0±0.0</td><td>1.1±2.7</td><td>0.2±4.1</td></tr><tr><td rowspan="5">trivia_qa</td><td>sp</td><td>35.5±24.1</td><td>99.5±7.1</td><td>35.5±21.6</td><td>100.0±0.0</td><td>36.2±22.1</td><td>99.3±8.1</td></tr><tr><td>no</td><td>35.2±22.0</td><td>97.7±15.1</td><td>35.8±22.8</td><td>99.2±9.1</td><td>34.9±22.0</td><td>97.8±14.6</td></tr><tr><td>us</td><td>1.2±0.8</td><td>0.0±0.0</td><td>1.3±1.0</td><td>0.0±0.0</td><td>1.1±0.7</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>35.4±24.1</td><td>97.3±16.1</td><td>38.2±25.2</td><td>99.2±9.1</td><td>42.0±24.4</td><td>96.7±18.0</td></tr><tr><td>no</td><td>43.4±28.5</td><td>87.8±32.7</td><td>46.5±28.9</td><td>96.0±19.6</td><td>48.2±30.4</td><td>95.7±20.4</td></tr><tr><td rowspan="3">hotpot_qa</td><td>us</td><td>1.0±0.5</td><td>0.0±0.0</td><td>1.0±0.6</td><td>0.0±0.0</td><td>1.3±1.4</td><td>1.0±9.9</td></tr><tr><td>sp</td><td>39.2±31.8</td><td>93.0±25.5</td><td>40.9±31.4</td><td>98.3±12.8</td><td>41.4±31.1</td><td>99.0±9.9</td></tr><tr><td>no</td><td>41.0±30.9</td><td>91.7±27.6</td><td>41.3±31.1</td><td>98.0±14.0</td><td>41.4±31.1</td><td>98.7±11.5</td></tr><tr><td rowspan="3">pop-qa</td><td>us</td><td>1.1±0.6</td><td>0.0±0.0</td><td>1.1±0.5</td><td>0.0±0.0</td><td>1.0±0.4</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>53.8±24.9 98.8±10.7</td><td></td><td>54.4±24.7</td><td>99.5±7.1</td><td>55.0±24.7</td><td>98.7±11.5</td></tr><tr><td>no</td><td>58.9±29.3</td><td>93.5±24.7</td><td>60.0±30.9</td><td>98.8±10.7</td><td>58.2±28.9</td><td>96.8±17.5</td></tr><tr><td rowspan="3">trec_coarse</td><td>us</td><td>1.0±0.0</td><td>0.0±0.0</td><td>1.0±0.0</td><td>0.0±0.0</td><td>1.0±0.0</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>127.6±6.1</td><td>99.6±6.3</td><td>127.3±8.4</td><td>99.0±9.9</td><td>128.0±0.1</td><td>99.6±6.3</td></tr><tr><td>no</td><td>128.0±0.0</td><td>100.0±0.0</td><td>128.0±0.0</td><td>100.0±0.0</td><td>128.0±0.1</td><td>99.4±7.7</td></tr><tr><td rowspan="3">nlu</td><td>us</td><td>1.0±0.1</td><td>0.0±0.0</td><td>1.0±0.1</td><td>0.0±0.0</td><td>1.0±0.2</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>70.4±21.1</td><td>70.6±45.6</td><td>69.0±21.7</td><td>45.6±49.8</td><td>14.6±18.8</td><td>8.4±27.7</td></tr><tr><td>no</td><td>71.7±20.8</td><td>100.0±0.0</td><td>71.7±20.8</td><td>100.0±0.0</td><td>71.7±20.8</td><td>99.8±4.5</td></tr><tr><td rowspan="3">clinc150</td><td>us</td><td>1.0±0.1</td><td>0.0±0.0</td><td>1.0±0.1</td><td>0.0±0.0</td><td>1.0±0.1</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>30.2±28.6</td><td>43.8±49.6</td><td>32.6±31.0</td><td>49.8±50.0</td><td>57.6±32.2</td><td>87.6±33.0</td></tr><tr><td>no</td><td>59.9±19.7</td><td>100.0±0.0</td><td>59.9±19.7</td><td>100.0±0.0</td><td>60.1±20.5</td><td>100.0±0.0</td></tr><tr><td rowspan="3">inf_qa</td><td>us</td><td>1.1±0.8</td><td>0.0±0.0</td><td>1.2±0.8</td><td>0.0±0.0</td><td>1.2±0.8</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>45.7±32.4100.0±0.0</td><td></td><td>45.7±32.4</td><td>99.0±9.9</td><td>45.8±32.3</td><td>97.0±17.1</td></tr><tr><td>no</td><td>45.5±32.3</td><td>95.0±21.8</td><td>45.7±32.4</td><td>99.0±9.9</td><td>45.7±32.4</td><td>96.0±19.6</td></tr><tr><td rowspan="3">inf_mc</td><td>us</td><td>1.0±0.0</td><td>0.0±0.0</td><td>2.3±12.6</td><td>1.0±9.9</td><td>1.0±0.0</td><td>0.0±0.0</td></tr><tr><td>sp</td><td>128.0±0.0</td><td>100.0±0.0</td><td>128.0±0.0</td><td>100.0±0.0</td><td>127.9±1.2</td><td>98.0±14.0</td></tr><tr><td>no</td><td>128.0±0.0</td><td>100.0±0.0</td><td>128.0±0.0</td><td>100.0±0.0</td><td>128.3±3.6</td><td>97.0±17.1</td></tr><tr><td rowspan="3">json_kv</td><td>us</td><td></td><td>0.0±0.0</td><td>1.0±0.0</td><td></td><td>3.7±1.0</td><td>85.0±35.7</td></tr><tr><td></td><td>1.1±0.1 4.1±0.3</td><td>100.0±0.0</td><td>4.1±0.3</td><td>0.0±0.0 100.0±0.0</td><td>4.1±0.3</td><td>98.0±14.0</td></tr><tr><td>sp no</td><td>4.1±0.3</td><td>100.0±0.0</td><td>4.1±0.3</td><td>100.0±0.0</td><td>4.1±0.3</td><td>100.0±0.0</td></tr></table>

Table 7: Token length statistics of generated samples for 10 Helmet datasets (of context width 128k) and 3 cache logics. trn denotes model checkpoint being used: us uses our novel method with the same cache policy in place, sp is using sequence parallelism, no is the base checkpoint Qwen3-4B-Instruct-2507 (no fine-tuning). R is based on the ratio of output length to target length (in tokens), p<sub>128</sub> (in percent) is the fraction of outputs of maximal size 128 (means, and stddevs over all test set samples).

In order to quantify the prevalence of this failure mode across all datasets and setups, we

<table><tr><td>nq R  $p _ { 1 2 8 }$ </td><td>tri_qa R P128</td><td>hot_qa R</td><td>pop-qa R P128</td><td>trec_c R P128</td></tr><tr><td>1.1±1.1 0.0±0.0</td><td>1.0±0.6 0.0±0.0</td><td>P128 1.1±0.6 0.0±0.0</td><td>1.0±0.4 0.0±0.0</td><td>1.0±0.0 0.0±0.0</td></tr><tr><td>nlu</td><td>clc150</td><td>inf_qa</td><td>inf_mc R</td><td>json_kv R</td></tr><tr><td>R p128 1.0±0.1 0.0±0.0</td><td>R p128 1.0±0.0 0.0±0.0</td><td>R p128 1.2±0.9 0.0±0.0</td><td>p128 1.0±0.0 0.0±0.0</td><td>p128 1.0±0.0 0.0±0.0</td></tr></table>

Table 8: Token length statistics of generated samples for 10 Helmet datasets (of context width 128k) for training and inference with exact attention (sequence parallelism).

use two statistics, estimated over all samples generated for each dataset and cache logic:

$$
R = \frac { \mathrm { l e n } ( \mathrm { o u t p u t } ) } { \mathrm { l e n } ( \mathrm { t a r g e t } ) } , \quad p _ { 1 2 8 } = \mathrm { I } _ { \{ \mathrm { l e n } ( \mathrm { o u t p u t } ) = 1 2 8 \} } .
$$

Here, len(·) denotes length in tokens, and samples are capped at 128 tokens. For some datasets, the targets are a list, in which case target is the longest entry appearing as substring in output, or the longest entry otherwise. Statistics for 10 Helmet datasets and 3 setups are shown in Table 7. With us, we have $R \approx 1$ across datasets and setups: outputs are close in length to targets, the model learned the desired output type and stops generation properly. But with sp, R tends to be large, and $p _ { 1 2 8 }$ is often close to 100%. This is not a property of the sp checkpoints. As seen in Table 8, R ≈ 1 and $p _ { 1 2 8 } \approx 0$ if exact inference is used. Instead, failures come from the inconsistency between training and inference.

If we relate numbers in Table 7 with good results in Table 1 for sp and in Table 4 for $^ { n o , }$ this points to a shortcoming of the SubEM metric used for these datasets. Insensitive to any type and amount of nonsense extra output, it only requires the target to be contained in the output, without requiring a definite way to extract the substring. Once such a requirement is added, as in Accuracy, performance for sp and no plummets. In any case, by not even providing succinct outputs (a very clear signal in the data), sp clearly does not behave satisfactory. While the tolerance of SubEM (and also Accuracy, to a lesser extent) to any amount of extra output is intended to not disadvantage LLMs (which ”sugar-coat” answers in longer sentences), it makes the metrics blind to extra nonsense content returned. We should at least ask for a deterministic way to extract the response from the output.

## A.6 Computing Summed Attention Weights in SDPA

As noted in Section 3.1.1 and Section 3.3.1, KV cache policies like H2O or related ones need summed attention weights $\textstyle \sum _ { i } m _ { b , h , i , j }$ for each $( b , h , j )$ . This array of shape $\left( B , H _ { q } , N _ { k } \right)$ can be obtained as byproduct of SDPA, which returns Y of shape $( B , H _ { q } , N _ { q } , d _ { h } )$ . Note that summed attention weights are smaller than attention outputs, so there is a priori no reason for not returning them. However, all fast SDPA codes we know of, do not return this information. FlexAttention [13] can return log-sum-exp values $\log ( A \mathbf { 1 } _ { N _ { k } } )$ , where $A =$ mask $( d _ { h } ^ { - 1 / 2 } Q K ^ { T } )$ is the argument of softmax, likely because this is directly computed during FlashAttention [9].

Our implementation contains Triton code for computing summed attention weights alongside a FlashInfer SDPA kernel [68]. This is an add-on, and it would be better if leading

SDPA codes returned summed attention weights directly. In this section, we detail how this can be done (even though we have not implemented this). We also show how to compute them with FlexAttention [13], using two calls instead of one. This is contained in our implementation as baseline.

## A.6.1 FlashAttention for Summed Attention Weights

FlashAttention works by essentially computing the attention weights tensor M in blocks, using a lattice tiling along the query and the key axes. It can be understood as map-reduce, where map is independent per cell. More precisely, the full attention weights have shape $\left( B , H _ { q } , N _ { q } , N _ { k } \right)$ . In the following, we drop $( B , H _ { q } )$ , treating them as ”batch” dimensions. Use cell indices $( r , s )$ and index ranges $I ( r ) , \ J ( s )$ , so that the union of all $I ( r )$ covers $\{ 0 , \ldots , N _ { q } - 1 \}$ and the union of all $J ( s )$ covers $\{ 0 , \ldots , N _ { k } - 1 \}$ . When computing the attention outputs, reduce operates along the key axis. Define an additional auxiliary tensor of shape $( B , H _ { q } , N _ { q } )$ , with values

$$
\lambda _ { r , s } : = \log \left( \exp \left( A _ { I ( r ) , J ( s ) } \right) \mathbf { 1 } _ { | J ( s ) | } \right) , \quad A _ { I , J } : = \mathtt { m a s k } \left( d _ { h } ^ { - 1 / 2 } Q _ { I , } \mathbf { K } _ { J , \cdot } ^ { T } \right) .
$$

Reduction works as:

$$
\begin{array} { r l } & { \lambda _ { r , s _ { 1 } \oplus s _ { 2 } } = \operatorname* { m a x } \left\{ \lambda _ { r , s _ { 1 } } , \lambda _ { r , s _ { 2 } } \right\} + \log { 1 9 } \left( \exp \left( - \left| \lambda _ { r , s _ { 1 } } - \lambda _ { r , s _ { 2 } } \right| \right) \right) , } \\ & { Y _ { r , s _ { 1 } \ominus s _ { 2 } } = ( \mathrm { d i a g } \exp \left( \lambda _ { r , s _ { 1 } } - \lambda _ { r , s _ { 1 } \otimes s _ { 2 } } \right) ) Y _ { r , s _ { 1 } } + ( \mathrm { d i a g } \exp \left( \lambda _ { r , s _ { 2 } } - \lambda _ { r , s _ { 1 } \otimes s _ { 2 } } \right) ) Y _ { r , s _ { 2 } } . } \end{array}
$$

We can now run map independently for all $( r , s )$ , then reduce along s for all r.

Summed attention weights are given by $\boldsymbol { w } \in \mathbb { R } ^ { N _ { k } }$

$$
\begin{array} { r } { { \pmb w } ^ { T } = { \bf 1 } _ { N _ { q } } ^ { T } \exp \left( { \pmb A } - \pmb { \lambda } { \bf 1 } _ { N _ { k } } ^ { T } \right) . } \end{array}
$$

Define

$$
\tilde { \boldsymbol { F } } _ { r , s } = \exp \left( \boldsymbol { A } _ { I ( r ) , J ( s ) } - \lambda _ { r , s } \mathbf { 1 } _ { | J ( s ) | } ^ { T } \right) ,
$$

$$
\begin{array} { r } { F _ { r , s } = \left( \operatorname { d i a g } \exp \left( \lambda _ { r , s } - \lambda _ { r } \right) \right) \tilde { F } _ { r , s } = \exp \left( A _ { I ( r ) , J ( s ) } - \lambda _ { r } \mathbf { 1 } _ { | J ( s ) | } ^ { T } \right) . } \end{array}
$$

If

$$
\begin{array} { r } { { \pmb w } _ { r , s } ^ { T } = { \bf 1 } _ { | I ( r ) | } ^ { T } \exp \left( { \pmb A } _ { I ( r ) , J ( s ) } - \lambda _ { r } { \bf 1 } _ { | J ( s ) | } ^ { T } \right) = { \bf 1 } _ { | I ( r ) | } ^ { T } { \pmb F } _ { r , s } , } \end{array}
$$

then $\begin{array} { r } { \pmb { w } = [ \sum _ { r } \pmb { w } _ { r , s } ] } \end{array}$ . We can compute w and Y with an outer loop over r, inner loop over s. Initialize ${ \pmb w } = { \bf 0 } _ { N _ { k } }$ . The iteration r works as follows:

• Map: Compute $[ \lambda _ { r , s } ] , [ \tilde { F } _ { r , s } ] , [ Y _ { r , s } ]$ in parallel.

• Reduce: $( \lambda _ { r } , Y _ { r } ) = \tt r e d u c e \left( \left[ \lambda _ { r , s } \right] , \left[ Y _ { r , s } \right] \right)$

• Compute $\boldsymbol { w } _ { r , s } ^ { T } = \mathbf { 1 } _ { | I ( r ) | } ^ { T } \boldsymbol { F } _ { r , s } = \exp ( \lambda _ { r , s } - \lambda _ { r } ) ^ { T } \tilde { \boldsymbol { F } } _ { r , s } . \mathrm { ~ A d d ~ } \boldsymbol { w } _ { r } = [ \boldsymbol { w } _ { r , s } ] \mathrm { ~ t o ~ } \boldsymbol { w }$

Compared to standard FlashAttention, we need to first reduce along s in order to obtain $\lambda _ { r }$ , keeping $\tilde { \boldsymbol { F } } _ { r , s }$ and $\lambda _ { r , s }$ around.

## A.6.2 Summed Attention Weights with FlexAttention

FlexAttention [13] stands out among fast SDPA codes by allowing the user to configure the computation in several ways (see also https://pytorch.org/blog/flexattention/). Here we describe how to compute summed attention weights w alongside the attention output Y, by calling FlexAttention twice.

Recall that SDPA computes

$$
\pmb { Y } = \exp \left( \pmb { A } - \pmb { \lambda } \pmb { 1 } ^ { T } \right) \pmb { V } .
$$

The summed attention weights are

$$
\begin{array} { r } { { \pmb w } = \exp \left( { \pmb A } - { \pmb \lambda } { \bf 1 } ^ { T } \right) { \bf 1 } = \exp \left( { \pmb A } ^ { T } - { \bf 1 } { \pmb \lambda } ^ { T } \right) { \bf 1 } = \exp \left( { \pmb A } ^ { T } \right) { \tilde { \pmb v } } , \quad { \tilde { \pmb v } } : = \exp ( - { \pmb \lambda } ) . } \end{array}
$$

Up to softmax normalization, we can obtain this by calling a variant of SDPA again, flipping Q and K, reverting the attention masking, and passing exp(−λ) as values. Importantly, FlexAttention returns λ with the option return aux = AuxRequest(lse=True). Now, if λ<sup>˜</sup> denotes lse for the second call (with Q and K flipped), then:

$$
{ \pmb w } = \exp \left( { \pmb A } ^ { T } - \tilde { { \lambda } } { \bf 1 } ^ { T } + \tilde { { \lambda } } { \bf 1 } ^ { T } \right) \tilde { \pmb v } = \left( \mathrm { d i a g } \exp ( \tilde { { \lambda } } ) \right) \exp \left( { \pmb A } ^ { T } - \tilde { { \lambda } } { \bf 1 } ^ { T } \right) \tilde { \pmb v } = \exp ( \tilde { { \lambda } } ) \circ \tilde { { \pmb y } } .
$$

Finally, trying to minimize numerical errors (we are using 16 bit data types), we use $\exp ( - ( \lambda - { \bar { \lambda } } \mathbf { 1 } ) )$ and $\exp ( \tilde { \lambda } - \bar { \lambda } \mathbf { 1 } )$ , where $\bar { \lambda } = N _ { q } ^ { - 1 } \mathbf { 1 } ^ { T } \lambda$ is the mean of λ. All in all:

$$
\bullet ( Y , \lambda ) = \mathrm { S D P A } ( Q , K , V ) , \bar { \lambda } = N _ { q } ^ { - 1 } \mathbf { 1 } ^ { T } \lambda .
$$

$$
\bullet ( \tilde { y } , \tilde { \lambda } ) = \mathtt { S D P A } \lrcorner \mathtt { r e v } ( K , Q , \exp ( - ( \lambda - \bar { \lambda } \mathbf { 1 } ) ) ) , \mathrm { t h e n ~ } w = \exp ( \tilde { \lambda } - \bar { \lambda } \mathbf { 1 } ) \circ \tilde { y } .
$$

Here, SDPA rev difers from SDPA by the attention masking being reversed. FlexAttention allows to specify the attention mask as block mask(b, h, q idx, kv idx). The mask for SDPA rev is given by flipping q idx and kv idx in the code for SDPA. Note that Flex-Attention supports V to have a diferent (final) embedding dimension that Q, K. Maybe this even translates in the second call being faster than the first. All in all, compared to a single FlexAttention call and no attention weights, this is at most twice as expensive.

## A.7 Sparse Attention and SotA Inference Libraries

In Section 3.3, we discuss the (somewhat surprising) fact that as of today, sparse attention is not much used in real-world practice, because existing implementations are too slow to be competitive with the state of the art. While a part of the latency gap between sparse attention and sequence or context parallelism is probably inherent, we argued in Section 3.3.1 some shortcomings of current sparse attention implementations are easy to eliminate by minor extensions of fast SDPA kernel codes.

Here, we comment on why sparse attention policies, such as H2O, are not supported in vLLM [30], the leading fast inference library. Details are found in https://github.com/ vllm-project/vllm/issues/10646, https://github.com/vllm-project/vllm/issues/ 12254, https://github.com/vllm-project/vllm/issues/5751. In vLLM, KV caches are maintained as set of fixed-sized pages (or blocks). The main issue is that they require a page to store KV content across all heads: KV information for a token is stored in the cache for all heads or for none. The cited RFCs mention that it would require significant changes to the memory layout and block manager abstractions to change that. However, for modern sparse attention (such as H2O), policies $\pi ( b , h , t )$ depend on $( h , t )$ in general: they select diferent tokens per head.

As detailed in Section 3.1, our implementation has no problems with this. We simply maintain dense bufers of shape $\left( B , H _ { k } , N _ { C } , d _ { h } \right)$ , where the cache length $N _ { C }$ is fixed independent of context width, and then use torch.gather and torch.scatter for read and write access. Whereas PagedAttention requires specific SDPA kernels, we can use existing dense SDPA codes, as long as we cater for causal masking (see Section 3.3.1). While not supported in our current implementation, we could build up KV cache bufers in chunks to cater for sequence lengths shorter than $N _ { C }$ , thereby solving the issue of unnecessary pre-allocations [30]. Finally, while torch.gather and torch.scatter access the bufer in a non-contiguous way, this is very subdominant to SDPA computations in our experience. In fact, when calling scatter(keys, index, key new), the final axis of index is always constant (so that the final bufer axis of size $d _ { h }$ is accessed contiguously), and optimized scatter and gather kernels could easily be implemented for this case if the PyTorch implementations do not already cater for this special case. One advantage of PagedAttention over our approach is that they can in principle represent diferent numbers of tokens per head or batch dimension, which can render sparse attention a bit more flexible. However, since vLLM requires each page to extend over all heads, this extra flexibility is not supported there.

As long as highly optimized and widely used inference libraries do not support sparse attention, it may remain underused. We hope that our work sparks some renewed interest in this direction.

## A.8 Details on Related Work

Here, we provide additional details about relations of our method with prior work. OOMB [33] shares properties with our work, such as chunk-level processing, activation checkpointing, and eforts to compress KV cache bufers for autograd. Details on the relationship are as follows:

• Their implementation is better suited for representing KV caches exactly (no selection or compression) that ours. They implement a paged memory management like [30], which we do not (but see Section A.7 and comments in Section 3.1). However, despite all eforts in CPU ofloading and activation checkpointing, they run into the same barrier as [34], in that the factor for the final chunk depends on all KV cache bufers of all layers, so cannot be represented by autograd on a single device. At this point, RingAttention [38] is the method of choice, and it is not clear why their library would improve on implementations such as MS-SWIFT [78].

• They deal with activation memory for GPU by activation recomputation, while we use activation checkpointing. In the former, activations are recomputed during the backward pass from the start, whereas in the latter, recomputation starts from the most recent checkpoint. The former is too slow to be useful, so our guess is their code actually uses activation checkpointing.

• The most important diference is how they deal with KV cache bufers as nodes in the autograd graphs, and what this implies for generality. This is also the biggest challenge we face, and we deal with it by a combination of nested checkpointing, delta encoding of KV cache bufers, and integration into PyTorch by way of autograd saved tensor hooks (Section A.4.3). Together with recording and replaying KV cache decisions, this renders our implementation fully agnostic to the KV cache policy: it works with any selection or compression policy (see Section 2 for many references). In contrast, they try to hide all nodes representing KV cache content from autograd altogether, so that none of this information can be placed in the computation graph. This is possible only by implementing a number of complex CUDA kernels, in which all inner derivatives w.r.t. these “KV cache nodes” are made explicit. Apart from substantial derivation and implementation complexity, their approach must be specialized to the KV cache policy being used. In fact, their paper only provides results for two specific sparse attention policies (LSA and DSA). Moreover, their paper is sparse on details how the hiding of KV cache bufers from autograd works in practice, since KV cache updates are tightly coupled with SDPA calls. In the end, their implementation may not be agnostic to SDPA kernels, which given the speed of development of SDPA would be a major drawback.

• Both their and our implementation make use of CPU ofloading of activations, KV cache bufers, and head gradients. They claim to have done this asynchronously, as in [71], which hides latency. We have also experimented with this, but did no so far achieve significant speedups. Moreover, asynchronous transfer requires double bufering, which drives up GPU memory requirements. Still, more efort in this direction is warranted.

## A.9 Open Source Library KeysAndValues. Experiments

For all experiments above, fine-tuning with our method (column us in Table 1) and inference with sparse attention (all policies) were done with a new open source library for long context fine-tuning and inference: KeysAndValues (https://github.com/awslabs/keys\_values).

Apart from eficient code for our fine-tuning method, the library provides clean and simple abstractions for sparse attention and key-value caches of limited size. Among its features are:

• Long context fine-tuning on a single GPU (this work).

• Several variants of the H2O KV cache policy [72]. The library provides a generic implementation for any cache logic of the form π(b, h, t) which makes use of summed attention weights.

• Quantization of KV cache bufers.

• Integration of FlexAttention [13], FlashInfer [68], FlashAttention [9, 48] and eager SDPA behind a common multi-head self-attention interface. This includes summed attention weights (for H2O-like policies), as well as a proper backward implementation.

• Model implementations and inference code is from LitGPT (https://github.com/ lightning-ai/litgpt), which allows for almost any Hugging Face checkpoint to be used. However, while bringing modern KV caching to Hugging Face would require hacking several code files separately for every single model, you can apply your KV cache policy or attention approximation to almost all models with few changes of common code.

• Support of CPU ofloading of KV cache bufers and model weights.

• Support of distributed training (distributed data parallel, CPU ofloading of model weights optional). Support of distributed evaluation.

With this library, we do not intend to compete with vLLM [30] or SGLang [79], which include more low level optimizations and support of latest GPU architectures. Instead, we make it easy for researchers to explore new KV cache policies, post-time training ideas, or unusual multi-head self-attention approximations, providing clean abstractions of these concepts which can be used and extended without having to deal with intricate implementation details of existing high-performance libraries.

## A.9.1 Running Our Experiments

Once KeysAndValues has been properly installed, the training runs for our method can be reproduced as follows. You need to be on an instance with at least four Nvidia A100 GPUs with 40 GB of RAM. We used AWS EC2 p4d.24xlarge instances, which have 8 A100 GPUs, running two experiments in parallel on each instance.

```shell
export DATASET KEY=”nq” ; \
export DATASET SIZE=”128k” ; \
export POLICY NAME=”h2o−o r i g ” ; \
export CACHE LENGTH=”32768” ;
export CHUNK SIZE=”2048” ; \
export EVAL STEPS=10; \
CUDA VISIBLE DEVICES=” 0 ,1 ,2 ,3 ” \
PYTORCH ALLOC CONF=expandable segments : True \
KEYSVALS LOG DIR=” ./ fi n e t u n e / helmet $ {DATASET KEY} $ {DATASET SIZE}/${POLICY NAME} c s $ {CHUNK SIZE}/ l o g s ” \
python3 k e y s v a l u e s / m a i n . py f i n e t u n e l o n g l o r a \
Qwen/Qwen3−4B−I n s t r u c t −2507 \
o u t d i r ./ fi n e t u n e / helmet $ {DATASET KEY} $ {DATASET SIZE}/${POLICY NAME} c s $ {CHUNK SIZE} \
p r e c i s i o n bf16−true \
verbose some \
d e v i c e s 4 \
data Helmet \
data . dataset key ${DATASET KEY}
data . max l ength ${DATASET SIZE}
data . m e t a d a t a d i r . / data \
d a t a . t r a i n l o a d e r l o n g e s t f i r s t True \
t r a i n . s a v e i n t e r v a l ${EVAL STEPS} \
t r a i n . m i c r o b a t c h s i z e 2 \
t r a i n . e p o c h s 5 \
t r a i n . a v e r a g e l o s s p e r b a t c h True \
eva l . i n t e r v a l ${EVAL STEPS} \
e v a l . i n i t i a l v a l i d a t i o n True \
eval . u s e s a m p l e m e t r i c F a l s e \
kv cache . c a ch e l e n g t h ${CACHE LENGTH}
kv cache . chunk size ${CHUNK SIZE}
k v c a c h e . name ${POLICY NAME}−t o r ch −q u a n t i z e d 8 \
g r a d . l a y e r s p e r c e l l 1 \
grad . layercp qname d e fa u l t \
grad . cachecp qname t o r ch −q u a n t i z e d 8 \
g r a d . c h u n k s p e r c e l l m u l t i p l i e r 1 \
o p t i m i z e r . name AdamW \
o p t i m i z e r . l e a r n i n g r a t e 0 . 0 0 0 5
```

Once all desired training runs have finished, evaluations (on the test sets) can be run as follows.

CUDA VISIBLE DEVICES=” 0 ,1 ,2 ,3 ” \   
PYTORCH ALLOC CONF=expandable segments : True \   
KEYSVALS LOG DIR=” . / fi n e t u n e / e v a l u a t i o n /myruns/ l o g s ” \   
python3 k e y s v a l u e s / m a i n . py e v a l l o n g e x t \   
. / myruns . yaml \

−−verbose some \   
d e v i c e s 4 \   
−−b a t c h s i z e 2 \   
u s e s a m p l e m e t r i c True \   
sample metric max generated tokens 20 \   
num store generated samples 1000

Here, myruns.yaml is a YAML file containing entries of this form:

o u t d i r : . / fi n e t u n e / helmet nq 64k / h2o cs2048   
model type : l o r a   
e v a l t a s k s :   
− step −000420

For each setup, evaluations can be run for diferent checkpoints step-000\*\*\* stored alongside training. In our experiments, for each setup, we select the checkpoint which minimize validation loss. We refer to README.md for further details on how to aggregate evaluation results and create result tables.