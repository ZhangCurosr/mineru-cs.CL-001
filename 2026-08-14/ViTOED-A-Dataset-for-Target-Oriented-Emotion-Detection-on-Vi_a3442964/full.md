# ViTOED: A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts

Chanh Vo<sup>1,2</sup>, Son T. Luu<sup>1,2</sup>, and Ngan Luu-Thuy Nguyen<sup>1,2,\*</sup>

<sup>1</sup>Faculty of Information Science and Engineering, University of Information Technology, Ho Chi Minh City

<sup>2</sup>Vietnam National University, Ho Chi Minh City, Vietnam

Email: 20521122@gm.uit.edu.vn, sonlt@uit.edu.vn, ngannlt@uit.edu.vn

Abstract—This paper introduces ViTOED, a novel dataset for target-oriented emotion detection in Vietnamese social media texts. The ViTOED comprises 10,985 user comments and 21,244 manually annotated opinion quadruples (source, target, expression, polarity) that follow strict guidelines. The dataset reveals Vietnamese-specific phenomena, such as implicit sources and targets and vocabulary ambiguities, enabling deeper analysis of user emotions toward entities. We propose a baseline using structured sentiment graphs and evaluate various Vietnamese pre-trained language models. The empirical results highlight challenges in span detection and relation extraction and indicate substantial room for model improvement in Vietnamese Target-Oriented Emotion Detection tasks.

Index Terms—targeted sentiment analysis, dataset, social media texts

## I. INTRODUCTION

Sentiment Analysis (SA) is a task in Natural Language Processing (NLP) that analyzes users’ opinions and emotions toward specific entities [1]. In sentiment analysis, detecting Entity and Aspect levels in users’ input text is a challenging but potentially useful task for performing qualitative and quantitative analyses of users’ emotions toward specific targets. According to the definition from [1], "an actual word or phrase indicating an entity category is called the expression". One of the challenges for sentiment analysis is to detect the "implicit expressions", which contain no polarity markers but still carry the human-aware polarity in context for expressing the emotions [2]. Moreover, a user’s text can contain multiple entities, and each entity can have its own expression to a specific target entity with corresponding polarity. Exploiting the emotion associated with a target helps understand the rationale for the polarity carried by individuals. It improves the accuracy of sentiment analysis of users’ social media text.

To be clear, the task of target-oriented emotion detection is denoted as giving a sentence s in human language, the goal is detecting a list of opinion tuples $O = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { n } \}$ , in which each o ∈ O is a quadruple of <source, target, expression, polarity>. The source indicates the holder of expression in the sentence, and the target is the subject to which the source refers. The expression refers to the emotions or behaviors expressed by users. Finally, the polarity indicates the sentiment or opinion of the users. Figure 1 illustrates an example for detecting the emotion by the target in the user’s sentence. At first, this sentence is categorized as "Sadness" (as reported in the UIT-VSMEC dataset [3]), but it actually expresses two opinions. One represents the "negative" emotion between the source "con" (in the Vietnamese language, this is the pronunciation between son and daughter with his/her parent) toward the target "cha" (English: father) through the expression "đã sai vì chưa làm được gì cho" (English: fault because he/she disobeys his/her father). The remaining emotion expresses the emotion of the source "con" with the target "cha mẹ " (English: parents) as positive through the expression "thật sự xin lỗi" (English: hereby apologize). This is positive because it shows that the subject (the source) in the sentence recognizes their mistake and apologizes to their parents, indicating empathy toward the target.

![](images/f98e51d64b1605b779ed47340f08bdd07fd1f6fe6f13ea9fde667db7ab9cd451.jpg)  
Fig. 1: A sample of target-oriented emotion detection from the sentence (English: I was wrong for not being able to do anything for my father. I am truly sorry to my parents.)

To address the problem of target-oriented emotion detection in Vietnamese, we proposed the ViTOED dataset, which contains about 10,985 sentences collected from Vietnamese social networking sites. There are 21,244 manually annotated opinion quadruples in which each sentence can have multiple opinions. Besides, we construct a baseline for target-oriented sentiment analysis based on [4] and evaluate various Vietnamese pretrained language models on our dataset to assess the ability of computers to understand the emotions expressed toward a specific target by the source. To sum up, our paper has three main contributions: (1) We introduce ViTOED, a new human-annotated Vietnamese dataset for emotion detection by target. The dataset is constructed under a strict annotation process, (2) We analyze the dataset to identify a phenomenon in Vietnamese regarding the difference between the target, source, and expression in expressing emotion in a sentence, and (3) We evaluate various Vietnamese pre-trained language models using the Structured Sentiment Analysis Graph proposed by [4] on our dataset to investigate the performance of machine learning models for target-oriented emotion tasks in Vietnamese.

## II. RELATED WORKS

Although Sentiment analysis (SA) plays a vital role in many real-world applications, this task has several critical challenges. According to [5], biases and harms in sentiment analysis models make SA models unable to represent all users’ perspectives in the real world. This bias issue is due to the lack of explanation in predicting the sentiment through "an interdisciplinary lens" [5]. To reduce bias, SA models need to capture external factors that contribute to sentiment rather than focusing solely on lexical text categorized as positive or negative. One potential approach to reducing bias in the SA is targeted sentiment analysis [4], in which SA models exploit targets, holders, expressions, and their polarity to form an argumentative opinion about specific objects in the sentence. There are several datasets employed for this task, such as the $D S _ { U n i s }$ [6] in English and MultiBooked [7] in Spanish. These available datasets are valuable for building and evaluating the target-oriented sentiment models.

In Vietnamese, most large-scale datasets currently available focus on the Aspect-based Sentiment Analysis (ABSA) task [8]. The high-quality, human-annotated datasets available in ABSA provide a valuable resource for constructing and evaluating ABSA models. Some of the ABSA datasets are the VLSP2018 [9] and UIT-ViSFD [10]. Those ABSA datasets are constructed at the sentence level, making it difficult to exploit the polarity represented by a set of tokens or words within a sentence. Additionally, the UIT-ViSD4SA dataset [11] enables computers to perform ABSA at the token level, which exploits aspect and corresponding polarity at the span level. However, this dataset has not mined the rationale of the holder and target of the polarity in the sentence toward the aspects. Hence, our motivation is to introduce ViTOED, which exploits emotion and expression by targeting tokens at the input sentence level.

## III. DATASET

## A. Data Creation Process

We use data from social networks, including 6,000 sentences from the UIT-VSMEC [3], and 5,010 comments were collected raw from our social media platforms. At first, we give the annotation guidelines to the annotators. Then, we let the annotators annotate a set of 210 sample sentences to evaluate the agreement among annotators. If the agreement between annotators is adequate, we let them annotate the entire dataset. After that, we perform cross-checking among annotators to ensure accurate labeling. To make labeling more convenient for annotators, we use Doccano<sup>1</sup> to annotate the data with a friendly user interface and easy setup on individual devices.

## B. Annotation Guidelines

For each sentence, the annotator must annotate the quadruple: source, target, expression, and polarity. There may be more than one quadruple per sentence. The meanings of source, target, expression, and polarity are described below:

TABLE I: Examples from the ViTOED dataset
<table><tr><td rowspan=1 colspan=1>No.</td><td rowspan=1 colspan=1>Comment</td><td rowspan=1 colspan=1>Label</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>tao thích bài này lm luôn ý(English: I like this song very much)</td><td rowspan=1 colspan=1>Source: &quot;tao&quot;,Target: “bài này&quot;,Expression: &quot;thích&quot;,Polarity: Positive</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7 thàng này không phi con ngưichúng mày (English: These 7 guys are not human)</td><td rowspan=1 colspan=1>Source: “&quot;,Target: “7 thng này&quot;,Expression: “không phåi con ngưichúng mày &quot;,Polarity: Positive</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>tao ho rng ht tóc rồi(English: I coughed so much that myhair fell out)</td><td rowspan=1 colspan=1>Source: &quot;tao&quot;,Target: “&quot;,Expression: &quot;ho rng ht tóc ri&quot;,Polarity: Negative</td></tr></table>

Source: The origin of the comment comes from the sender, including an individual or organization. The source of comments expresses an individual’s views, feelings, and actions towards the object, thing, or phenomenon mentioned. The source of sentiment is subjective to the commentator and is revealed through many different emotional forms. Aspects often originate from nouns and pronouns and are usually in the first person (i.e., "tao" in Vietnamese).

Target: The target refers to an individual, group, object, or phenomenon, which is directed by the "speaker" (the source) to express emotions, states, thoughts, etc. that affect the "object" (the target). The subjects are derived from nouns, pronouns, and characters. They are usually in the second person (For example, "mày", "bài này", "7 thằng này" as shown in Table I). In addition, the target can contain additional noun phrases of ownership relations.

Expression: Expressing the source’s emotions, opinion, feelings, and actions toward a particular object or event. Emotions include adjectives that describe emotions, or words that have an exclamative meaning (exclamation words), i.e., "đáng sợ lắm", or action (verb) accompanied by interjections and adjectives describing emotions, i.e., "ho rụng hết tóc rồi", as shown in Table I.

Polarity: Describe the negativity, positivity, and neutrality in the expression, which comprises three categories: Positive expresses the optimistic, encouraging, motivating, and empathetic emotion, Negative expresses offense, incite, hate, and scare contents, and Neutral indicates normal expressions that are not positive or negative. In addition, humor expressions that do not contain hateful or offensive content can be classified as Positive while the expressions that are untrue and misleading, as well as expressions that contain profanity, rudeness, and sensitive aspects, are also classified as Negative.

## C. Inter Annotator Agreement Evaluation

To measure Inter-Annotator Agreement (IAA) among annotators, we need to calculate the span-based intersection of the Source, Target, and Expression entities. Following the work by [11], we employ the F1-score for measuring the agreement among annotators. The F1-Score is computed as shown in Equation 1, where a and b are the first and second annotators, A and B are the set of tokens annotated by the first and second annotators:

TABLE II: Overview statistic about the ViTOED dataset.
<table><tr><td rowspan="2"></td><td colspan="2">Texts</td><td colspan="3">Source</td><td colspan="3">Target</td><td colspan="3">Expression</td><td colspan="3">Polarity</td></tr><tr><td>#</td><td>avg.</td><td>#</td><td>avg.</td><td>max</td><td>#</td><td>avg.</td><td>max</td><td>#</td><td>avg.</td><td>max</td><td>negative</td><td>neutral</td><td>positive</td></tr><tr><td>train</td><td>7,706</td><td>16.9</td><td>2,359</td><td>1.4</td><td>8</td><td>8,632</td><td>3.0</td><td>20</td><td>15,024</td><td>6.5</td><td>29</td><td>6,571</td><td>3,936</td><td>4,517</td></tr><tr><td>dev</td><td>1,085</td><td>16.0</td><td>385</td><td>1.7</td><td>12</td><td>1,160</td><td>3.1</td><td>16</td><td>2,087</td><td>7.0</td><td>21</td><td>962</td><td>505</td><td>620</td></tr><tr><td>test</td><td>2,194</td><td>16.8</td><td>572</td><td>1.2</td><td>6</td><td>2,127</td><td>2.8</td><td>20</td><td>4,133</td><td>6.9</td><td>24</td><td>1,681</td><td>1,366</td><td>1,086</td></tr></table>

$$
F _ { 1 } ( a , b ) = { \frac { 2 \cdot | A \cap B | } { | A | + | B | } }\tag{1}
$$

For Polarity, we must retrieve the opinion quadruples and then use pairwise to calculate the number of overlapping labels for the two annotators, dividing by the corresponding number of opinion quadruples for each annotator. Finally, we take the average overlapping pairwise between two annotators for the final agreement in Polarity, as shown in Equation 2, where a and b are the first and second annotators, A and B are the set (S, T, E, P) of matched labels between two annotators.

$$
P = { \frac { P ( a , b ) + P ( b , a ) } { 2 } } = { \frac { { \frac { | A \cap B | } { | A | } } + { \frac { | B \cap A | } { | B | } } } { 2 } }\tag{2}
$$

TABLE III: Inter-rater agreement degree.
<table><tr><td>Round</td><td>Source</td><td>Target</td><td>Expression</td><td>Polarity</td></tr><tr><td>1</td><td>0.17</td><td>0.31</td><td>0.34</td><td>0.15</td></tr><tr><td>2</td><td>0.26</td><td>0.48</td><td>0.54</td><td>0.35</td></tr><tr><td>3</td><td>0.27</td><td>0.51</td><td>0.64</td><td>0.40</td></tr><tr><td>4</td><td>0.32</td><td>0.61</td><td>0.77</td><td>0.47</td></tr></table>

Since we have a group of 3 annotators, the final IAA is computed by averaging across the pairs. To ensure that annotators understand the guidelines and make accurate annotations, we train them through multiple rounds. Table III describes the inter-annotator agreement result between annotators after four rounds. It can be seen that the agreement level improves after each round, indicating the efficiency of the annotation process and the guidelines. The Target and Expression show high IAA after four rounds, indicating strong consensus among annotators. On the other hand, as described in Table III, the low IAA for the Source is because many sentences are difficult to detect as containing a source (Examples #3 and #4 in Table I), which leads to annotators being confused, resulting in omissions or errors, making it challenging for the annotators to identify and label them accurately. For "Polarity," it depends on the consistency across (Source, Target, Expression, Polarity) tuples, so the agreement rate of approximately 0.5 is moderate. The level of agreement indicates that the annotators understood the labeling guidelines and followed the instructions accurately.

## D. Dataset Analysis

The ViTOED datasets consist of 21,244 annotated opinion quadruples within 10,985 sentences from social media networks. We divided the dataset into three subsets: Train, Dev, and Test, with a ratio of 7:1:2. This was done to facilitate data analysis, observation, as well as model training and parameter tuning. Table II provides an overview of the ViTOED dataset. The max and avg are computed at the word level. In general, the distributions of source, target, and expression across the training, development, and test sets are similar. The sentiment tends to lean slightly toward the negative class. However, the differences among the three polarities are not significant.

Although the number of sources and targets is relatively small compared to the number of opinions, their distribution is appropriately aligned with the proportions in the train, dev, and test data splits. We can also observe that the number of targets is only half the number of opinions, indicating that targets are generally mentioned at an average frequency in social media comments. As for the sources, given the nature of social media comments and the structure of Vietnamese, sources are often implicit or assumed to be the commenters, resulting in a relatively low number of explicit sources. In Table IV, we define four types of opinions in the comments of the dataset. The T-E demonstrated that an opinion missing the Source accounts for 52.10% of the total opinions. This type of opinion is the most common in social media comments today.

TABLE IV: Dataset Distribution by Opinion Category. S is the Source, T is the Target, and E is the Expression
<table><tr><td>Entity</td><td>S-T-E</td><td>S-E</td><td>T-E</td><td>E</td></tr><tr><td>Train</td><td>1,667</td><td>1,064</td><td>8,054</td><td>4,239</td></tr><tr><td>Dev</td><td>299</td><td>329</td><td>1,922</td><td>1583</td></tr><tr><td>Test</td><td>277</td><td>134</td><td>1,093</td><td>583</td></tr><tr><td>Total</td><td>2,243</td><td>1,527</td><td>11,069</td><td>6,405</td></tr></table>

Additionally, we investigate the frequency of words in each component —Source, Target, and Expression. We select the top-5 most frequently appearing words, as shown in Table V. For the source, most of the words are Vietnamese personal pronouns, i.e., "tao", "em", "mình", "tui", and "ta" (in English, these personal pronouns are I). In Vietnamese, the personal pronouns "tao", "em", "mình", "tui", and "ta" are commonly used to refer to individuals in communication. Moreover, users tend to use the abbreviated forms of personal pronouns on social media, such as "t" for "tao" and "e" for "em". Overall, the "tao" (in English, similar to "I") and its abbreviated form "t" are mainly used to make the source. For the target, the popular personal pronouns are "mày", "thằng", "bọn", and "đứa" (in English, these personal pronouns are "You"). In addition, the word "mấy" also appeared frequently in the target. This word serves as an adjunct that takes a plural form of personal pronouns in Vietnamese. For example, the word "mấy thằng" means a group of people. Especially, according to Table V, the words "mày" (English: you) and "tao" (English: I) are used frequently to express both source and target in the sentence, although the "tao" mostly expresses the source, while "mày" focuses on describing the target. In Vietnamese, these words are commonly used in communication between friends or groups of people who are acquainted.

TABLE V: Top words for SOURCE, TARGET, and EXPRESSION
<table><tr><td></td><td colspan="2">SOURCE</td><td colspan="2">TARGET</td><td colspan="6">EXPRESSION</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="2">POSITIVE</td><td colspan="2">NEUTRAL</td><td colspan="2">NEGATIVE</td></tr><tr><td>TOP</td><td>Word</td><td>Freq</td><td>Word</td><td>Freq</td><td>Word</td><td>Freq</td><td>Word</td><td>Freq</td><td>Word</td><td>Freq</td></tr><tr><td>1</td><td>tao (I)</td><td>881</td><td>mày (You)</td><td>396</td><td>yêu (Love)</td><td>76</td><td>đi (Go)</td><td>39</td><td>s (Scare)</td><td>80</td></tr><tr><td>2</td><td>t (I)</td><td>174</td><td>my (any)</td><td>374</td><td>thương (Love)</td><td>66</td><td>ko (Not)</td><td>19</td><td>đéo (f*ck)</td><td>52</td></tr><tr><td>3</td><td>ta (I)</td><td>90</td><td>thàng (he)</td><td>349</td><td>đp (beauty)</td><td>47</td><td>mua (buy)</td><td>14</td><td>buồn (sad)</td><td>41</td></tr><tr><td>4</td><td>e (younger)</td><td>81</td><td>tao (me)</td><td>230</td><td>vui (happy)</td><td>38</td><td>bão (said)</td><td>14</td><td>chi (scold)</td><td>40</td></tr><tr><td>5</td><td>Em (younger)</td><td>66</td><td>ta (me)</td><td>204</td><td>cm on (thanks)</td><td>35</td><td>coi (see)</td><td>13</td><td>ghét (hat)</td><td>35</td></tr></table>

Finally, we investigate the word frequency for the expression according to the three polarities as shown in Table V. Based on each polarity, there are several characteristics of words such as "yêu", "thuơng" (English: love), and "đẹp" (English: beautiful) that express the positive polarity, "sợ" (English: scare), "đéo" (English: f\*ck), and "buồn" (English: sad) express for the negative emotion, and "đi" (English: let’s or go), "ko" (English: no, don’t) shows the neutral expression.

## IV. METHODOLOGY

We employ the Structured Sentiment Graph proposed by [4] as our baseline model. Figure 2 illustrates the overview of the baseline model. First, we tokenized the input sentences into token level using the spaCy tools for Vietnamese<sup>2</sup>. Then, the sequence of tokenized tokens is passed through the word, pos-tag, and character embedding layers, and then through an LSTM layer to construct character representation vectors. These vectors are concatenated to generate a contextualized embedding that captures the sentence’s semantic meaning at the token level.

Additionally, the input sentence is passed through a BERT model to generate a BERT-contextualized vector, enhancing token representations. Then, the character-based LSTM and ERT-contextualized vectors are passed into a BiLSTM to create the contextualized representation. Finally, we create a sentiment graph by fitting the contextualized embedding into two Feed-forward networks (FNN), including the HeadFNN and DependentFNN.

To represent the sentiment graph, we use the two parsing graph representations by [4], including the head-first and head-final nodes. The head-first sets the first token in the span as the root and other tokens within the span as the dependents (as illustrated in Figure 3a). On the other hand, the head-final sets the final token in the span as the root, and other tokens in the spans as dependents (as shown in Figure 3b).

![](images/bf166f0f85f810c5573ebe5c940a35f30eebb8ab6a4bd76d31df47201e133048.jpg)  
Fig. 2: The sentiment graph baseline model.

Finally, we employ two types of BERTology models for Vietnamese —multilingual and monolingual —to represent contextual embeddings. The multilingual models comprise m-BERT [12] and XLM-R [13], while the monolingual models in the Vietnamese language consist of PhoBERT [14], Vi-SoBERT [15], and CafeBERT [16]. Additionally, we also employ two Vietnamese encoder-decoder models, including ViT5 [17] and BARTPho [18], and multilingual models supporting Vietnamese, including mT5 [19] and mBART [20].

## V. EXPERIMENTAL RESULTS

## A. Experimental results

To evaluate the performance of the baseline model, we follow four metrics as introduced in [4], including Span F1,

![](images/eba630861b6ae9f8238e72a98e7f31ec5b73f1007b85412ba7825717f99cce31.jpg)  
(a) head-first

![](images/b5ab19c952528372de5f6f7002e80216522a6a204c1cb8f2282d6c84e91a92a5.jpg)  
(b) head-final  
Fig. 3: The example of the parsing graph representation. (Vietnamese: Tao thấy bộ phim đó có cái gì hay ho đâu mà xem. English: I think that movie isn’t worth watching.)

Targeted F1, Parsing Graph F1(UF1 and LF1 regarding unlabeled and labeled arcs of the parsing graph), and Sentiment Graph F1 (NSF1 and SF1 regarding Non-polar Sentiment Graph and Sentiment Graph). Table VI illustrates the results of the performance of baseline models employed by different Vietnamese pre-trained language models.

TABLE VI: Performance comparison of different models across various metrics on the test set. The bold indicates the best on head-first, and the underline demonstrates the best on head-final.
<table><tr><td colspan="2"></td><td colspan="3">Spans</td><td>Targeted</td><td colspan="2">Parsing Graphs</td><td colspan="2">Sent. Graphs</td></tr><tr><td colspan="2"></td><td>Source</td><td>Target</td><td>Exp</td><td>F1</td><td>UF1</td><td>LF1</td><td>NSF1</td><td>SF1</td></tr><tr><td rowspan="2">mBERT</td><td>head-first</td><td>57.07</td><td>56.30</td><td>74.57</td><td>28.08</td><td>55.46</td><td>35.73</td><td>50.46</td><td>31.20</td></tr><tr><td>head-final</td><td>53.87</td><td>54.82</td><td>77.21</td><td>22.68</td><td>62.73</td><td>37.97</td><td>49.23</td><td>29.34</td></tr><tr><td rowspan="2">XLM-R</td><td>head-first</td><td>57.50</td><td>56.04</td><td>69.43</td><td>27.50</td><td>55.00</td><td>34.15</td><td>51.09</td><td>29.89</td></tr><tr><td>head-final</td><td>52.73</td><td>54.45</td><td>77.49</td><td>23.82</td><td>64.15</td><td>40.05</td><td>50.75</td><td>30.63</td></tr><tr><td rowspan="2">mBART</td><td>head-first</td><td>55.46</td><td>54.46</td><td>65.36</td><td>26.46</td><td>52.97</td><td>33.01</td><td>48.07</td><td>28.17</td></tr><tr><td>head-final</td><td>55.44</td><td>51.93</td><td>74.63</td><td>21.15</td><td>62.73</td><td>37.21</td><td>49.81</td><td>29.02</td></tr><tr><td rowspan="2">mT5</td><td>head-first</td><td>60.00</td><td>56.53</td><td>71.14</td><td>28.90</td><td>55.96</td><td>35.38</td><td>50.42</td><td>30.64</td></tr><tr><td>head-final</td><td>53.65</td><td>52.20</td><td>80.78</td><td>22.11</td><td>66.52</td><td>39.04</td><td>52.32</td><td>29.41</td></tr><tr><td rowspan="2">ViSoBERT</td><td>head-first</td><td>59.14</td><td>56.88</td><td>72.70</td><td>30.80</td><td>56.39</td><td>36.85</td><td>52.66</td><td>33.34</td></tr><tr><td>head-final</td><td>54.18</td><td>55.62</td><td>77.39</td><td>25.82</td><td>63.72</td><td>40.55</td><td>51.20</td><td>33.07</td></tr><tr><td rowspan="2">PhoBERT</td><td>head-first</td><td>59.03</td><td>56.94</td><td>74.13</td><td>27.04</td><td>55.21</td><td>33.54</td><td>52.88</td><td>29.02</td></tr><tr><td>head-final</td><td>54.80</td><td>55.12</td><td>78.71</td><td>23.39</td><td>65.15</td><td>39.88</td><td>51.01</td><td>30.88</td></tr><tr><td rowspan="2">CafeBERT</td><td>head-first head-final</td><td>58.12</td><td>56.96</td><td>74.56</td><td>29.24</td><td>57.05</td><td>36.00</td><td>51.20</td><td>31.84</td></tr><tr><td></td><td>52.99</td><td>53.69</td><td>73.34</td><td>24.14</td><td>61.52</td><td>38.69</td><td>49.68</td><td>31.03</td></tr><tr><td rowspan="2">ViT5</td><td>head-first</td><td>59.62</td><td>55.44</td><td>75.61</td><td>29.91</td><td>57.61</td><td>38.41</td><td>53.86</td><td>33.97</td></tr><tr><td>head-final</td><td>54.65</td><td>54.96</td><td>80.56</td><td>24.69</td><td>65.84</td><td>42.40</td><td>51.67</td><td>33.40</td></tr><tr><td rowspan="2">BARTPho</td><td>head-first</td><td>52.18</td><td>52.90</td><td>52.96</td><td>18.56</td><td>48.22 60.89</td><td>26.92</td><td>44.04</td><td>19.97</td></tr><tr><td>head-final</td><td>41.33</td><td>45.07</td><td>68.10</td><td>11.34</td><td></td><td>29.97</td><td>45.87</td><td>20.43</td></tr></table>

According to Table VI, the monolingual pre-trained language models, such as ViSoBERT and PhoBERT achieve better performance than multilingual models, such as mBERT and XLM-R, on this task. Besides, the head-first approach is efficient for extracting entities in a sentence, such as the source and target, whereas the head-final approach is suitable for detecting expressions. For the parsing graph, the head-final approach also performs better than the head-first approach. In contrast, the head-first and head-final approaches obtain similar performance in representing the sentiment graph. Additionally, mT5 and CafeBERT achieved the highest results for the target-oriented emotion detection task, in which they are good at extracting the source and target entities, respectively, while ViT5 is robust at exploiting the expression, parsing, and sentiment graphs, and ViSoBERT shows high performance in detecting the targeted expression.

## B. Error Analysis

The cause of label misalignment is the model’s failure to predict missing entities or to correctly align entities. The model has not yet handled spans well and is prone to errors across spans, resulting in lower performance. As shown in Table VII, the target entity is misaligned across two spans, and the model also failed to predict the Expression, resulting in a decrease in accuracy. The inaccurate predictions are caused by the model confusing the Source and Target, as both labels are associated with person-related or possessive nouns referring to people. This confusion between the two labels is unavoidable, and the language model has not yet handled this type of noun effectively

TABLE VII: Results of graph annotation by opinion components by F1(%).
<table><tr><td rowspan="2"></td><td rowspan="2">Source</td><td rowspan="2">Target</td><td colspan="3">Expression</td></tr><tr><td>negative</td><td>neutral</td><td>positive</td></tr><tr><td>head-first</td><td>29.95</td><td>44.92</td><td>43.77</td><td>19.87</td><td>43.77</td></tr><tr><td>head-first+inlabel</td><td>40.94</td><td>39.69</td><td>48.90</td><td>33.08</td><td>46.35</td></tr><tr><td>head-final</td><td>24.35</td><td>40.89</td><td>55.74</td><td>22.41</td><td>34.24</td></tr><tr><td>head-final+inlabel</td><td>32.51</td><td>36.35</td><td>62.01</td><td>25.48</td><td>36.77</td></tr><tr><td>Dep. edge</td><td>26.14</td><td>16.23</td><td>32.54</td><td>14.73</td><td>25.42</td></tr><tr><td>Dep. label</td><td>25.65</td><td>14.53</td><td>28.88</td><td>11.66</td><td>27.36</td></tr></table>

![](images/17ce46086491c8737bcb4ea8ea025dd6df422deaf1f4065748e8dc54e69bb931.jpg)

(a) Ground truth  
![](images/143e0bea056c690cc5b26032d3a780931747665a28adc0b14249c0d668d37959.jpg)  
(b) Prediction  
Fig. 4: Error sample: the edges are misaligned between tokens. (English: Please support me by liking this. Thank you, and wishing everyone good health.)

In addition, we also test the +inlabel structure that adds information to nodes that are not the main head in a graph. The Dep. edges uses syntactic information to identify the main head of each span while Dep. labels is a filtered version of Dep. edges to remove unreasonable choices, such as punctuation or peripheral words, when selecting the head. According to Table VII, adding the +inlabel structure improved the accuracy in identifying and extracting these spans. (In Table VII, the results are recorded on the development set by the ViT5 model). Besides, the incorrect predictions are due to the model’s weak handling of edges and roots. When using graphs with the head-first or head-final approach, the root is either the starting or ending point of the token, and the remaining tokens in the same set should be placed along the edges. However, the model mistakenly places the root, leading to misaligned token edges and label prediction errors (see the error example in Figure 4).

To sum up, based on the error analysis, we identified three main challenges for target-oriented emotion detection, as reflected in the baseline’s performance on the ViTOED. First, there is difficulty in recognizing source, target, and expression entity spans in the text due to the complexity of the sentence vocabulary. Second, the misprediction between the root and edges makes it challenging to detect the correct target emotion expressed by the subject from the sentence. Finally, confusion between source and target due to Vietnamese morphology can affect model performance in correctly identifying the source and target in a sentence.

## VI. CONCLUSION

This paper presents ViTOED, a human-annotated dataset for target-oriented emotion detection in Vietnamese social media texts. ViTOED consists of 21,244 annotated opinions from 10,985 comments of users on social networks. Besides, we construct a baseline model based on the structured sentiment graph architecture [4] and evaluate various Vietnamese Pre-trained language models on the dataset. From the error analysis, the model still exhibits prediction errors, including span misalignment between source and target, misprediction of relations, and confusion between target and source entities, indicating a challenge for further improvement in language understanding and for correctly mining sentiments. The source code and dataset are available at: https://github. com/sonlam1102/vitoed.

## ACKNOWLEDGMENTS

This research is funded by Vietnam National University HoChiMinh City (VNU-HCM) under grant number NCM2025-26-02

## REFERENCES

[1] B. Liu, Sentiment analysis and opinion mining. Springer Nature, 2022.

[2] Z. Li, Y. Zou, C. Zhang, Q. Zhang, and Z. Wei, “Learning implicit sentiment in aspect-based sentiment analysis with supervised contrastive pre-training,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Online and Punta Cana, Dominican Republic, Nov. 2021, pp. 246–256.

[3] V. A. Ho, D. H.-C. Nguyen, D. H. Nguyen, L. T.-V. Pham, D.-V. Nguyen, K. V. Nguyen, and N. L.-T. Nguyen, “Emotion recognition for vietnamese social media text,” in Computational Linguistics: 16th International Conference of the Pacific Association for Computational Linguistics, PACLING 2019, Hanoi, Vietnam, October 11–13, 2019. Springer, 2020, pp. 319–333.

[4] J. Barnes, R. Kurtz, S. Oepen, L. Øvrelid, and E. Velldal, “Structured sentiment analysis as dependency graph parsing,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Online, 2021, pp. 3387–3402.

[5] P. Venkit, M. Srinath, S. Gautam, S. Venkatraman, V. Gupta, R. Passonneau, and S. Wilson, “The sentiment problem: A critical survey towards deconstructing sentiment analysis,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, Dec. 2023, pp. 13 743–13 763.

[6] C. Toprak, N. Jakob, and I. Gurevych, “Sentence and expression level annotation of opinions in user-generated discourse,” in Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, Uppsala, Sweden, Jul. 2010, pp. 575–584.

[7] J. Barnes, T. Badia, and P. Lambert, “MultiBooked: A corpus of Basque and Catalan hotel reviews annotated for aspect-level sentiment classification,” in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan, May 2018.

[8] D. Van Thin, D. N. Hao, and N. L.-T. Nguyen, “A systematic literature review on vietnamese aspect-based sentiment analysis,” ACM Trans Asian Low-Resour. Lang. Inf. Process., vol. 22, no. 8, aug 2023.

[9] H. T. Nguyen, H. V. Nguyen, Q. T. Ngo, L. X. Vu, V. M. Tran, B. X. Ngo, and C. A. Le, “Vlsp shared task: sentiment analysis,” Journal of Computer Science and Cybernetics, vol. 34, no. 4, pp. 295–310, 2018.

[10] L. Luc Phan, P. Huynh Pham, K. Thi-Thanh Nguyen, S. Khai Huynh, T. Thi Nguyen, L. Thanh Nguyen, T. Van Huynh, and K. Van Nguyen, “Sa2sl: From aspect-based sentiment analysis to social listening system for business intelligence,” in Knowledge Science, Engineering and Management: 14th International Conference, KSEM 2021, Tokyo, Japan, August 14–16, 2021, Proceedings. Springer, 2021, pp. 647–658.

[11] K. N. T. Thanh, S. H. Khai, P. P. Huynh, L. P. Luc, D.-V. Nguyen, and K. N. Van, “Span detection for aspect-based sentiment analysis in Vietnamese,” in Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation, Shanghai, China, 11 2021, pp. 318–328.

[12] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota, Jun. 2019, pp. 4171–4186.

[13] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Online, Jul. 2020, pp. 8440–8451.

[14] D. Q. Nguyen and A. Tuan Nguyen, “PhoBERT: Pre-trained language models for Vietnamese,” in Findings of the Association for Computational Linguistics: EMNLP 2020, Online, Nov. 2020, pp. 1037–1042.

[15] N. Nguyen, T. Phan, D.-V. Nguyen, and K. Nguyen, “ViSoBERT: A pretrained language model for Vietnamese social media text processing,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, Dec. 2023, pp. 5191–5207.

[16] P. Do, S. Tran, P. Hoang, K. Nguyen, and N. Nguyen, “VLUE: A new benchmark and multi-task knowledge transfer learning for Vietnamese natural language understanding,” in Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, Jun. 2024, pp. 211–222.

[17] L. Phan, H. Tran, H. Nguyen, and T. H. Trinh, “ViT5: Pretrained text-totext transformer for Vietnamese language generation,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Student Research Workshop, Hybrid: Seattle, Washington + Online, Jul. 2022, pp. 136–142.

[18] N. L. Tran, D. M. Le, and D. Q. Nguyen, “Bartpho: pre-trained sequenceto-sequence models for vietnamese,” arXiv preprint arXiv:2109.09701, 2021.

[19] L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, “mT5: A massively multilingual pre-trained text-to-text transformer,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Online, 2021, pp. 483–498.

[20] Y. Liu, J. Gu, N. Goyal, X. Li, S. Edunov, M. Ghazvininejad, M. Lewis, and L. Zettlemoyer, “Multilingual denoising pre-training for neural machine translation,” Transactions of the Association for Computational Linguistics, vol. 8, pp. 726–742, 2020.