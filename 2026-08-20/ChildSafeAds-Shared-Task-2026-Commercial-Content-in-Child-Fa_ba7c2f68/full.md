# ChildSafeAds Shared Task 2026: Commercial Content in Child-Facing YouTube Videos

Thales Bertaglia<sup>1</sup>\* Catalina Goanta<sup>1</sup> Gerasimos Spanakis<sup>2</sup> Gunes Acar<sup>3</sup>

<sup>1</sup>Utrecht University, Netherlands

<sup>2</sup>Maastricht University, Netherlands

<sup>3</sup>Radboud University, Netherlands

## Abstract

CHILDSAFEADS<sup>1</sup> is a shared task on commercial content in YouTube videos likely to reach children and teenagers. It contains 3,360 videos from 939 channels. Each instance begins with a segment submitted to SponsorBlock, an opensource crowdsourced browser extension whose users mark sponsor segments so that others can skip them. We pair the segment with its available transcript, video and channel information, and a sales or service page linked from the video description. Systems determine what kind of offer is being promoted (ST1), assign product categories (ST2), and identify legal risk flags (ST3). The evidence is divided into four cumulative access levels, from the transcript to the linked page, so results can be compared against the cost of collecting the data. 45.5% of videos in our data failed to properly use the in-platform ad disclosure method (the “Includes paid promotion” label). GPT-5.4 pro duced the labels after the expert organiser team reviewed samples and iterated on the taxonomy, prompts and model choices. GPT-5.6-luna independently labelled the development set. This report describes the task, data and evaluation. An updated version will add participating systems and shared-task results.

## 1 Introduction

Advertising on YouTube often appears inside the video itself, delivered by the same creator and in the same style as the surrounding content. Children and teenagers can struggle to recognise this form of commercial persuasion (De Veirman et al., 2019; Loose et al., 2023). Disclosure is also inconsistent across countries and platforms (Bertaglia et al., 2025; Gui et al., 2025b). An audit of affiliatesponsored YouTube videos found a disclosure in about 10% of cases (Mathur et al., 2018). Authorities seeking to monitor this content at scale cannot begin with disclosures alone.

SponsorBlock provides candidates independent of platform disclosure. It is an open-source crowdsourced browser extension and public API for skipping sponsor segments in YouTube videos.<sup>2</sup> A contributor marks where a sponsorship starts and ends; other users can vote on the submission, and the extension skips accepted segments automatically. The resulting public database provides an independent signal that a video may contain commercial content. 45.5% of videos in our data failed to properly use the in-platform ad disclosure method (the “Includes paid promotion” label). Therefore, monitoring systems relying only on YouTube’s label would miss many relevant candidates.

Once a candidate segment has been found, the practical questions are straightforward. What is being promoted? What type of product is it? What aspects of the promotion need closer legal review? CHILDSAFEADS turns these questions into three linked classification tasks. Systems can work from the transcript alone or add video metadata, channel information and the linked promotional page. Every submission records which evidence it used. The shared task can therefore compare accuracy at different levels of data access and cost.

This report introduces the 3,360-instance dataset, its splits, the three tasks, the legal taxonomy and the evaluation setup. The updated version will describe the submitted systems and final results.

## 2 Legal Context

The task draws inspiration from parts of EU consumer and audiovisual advertising law to develop labels that can be predicted from the released data. ST1 describes what the consumer receives, using distinctions from the Consumer Rights Directive (CRD) (European Parliament and Council of the

European Union, 2011). ST3 draws mainly on the Unfair Commercial Practices Directive (UCPD), Audiovisual Media Services Directive (AVMSD) and Digital Services Act (DSA) (European Parliament and Council of the European Union, 2005, 2010, 2022).

The ST3 flags fall into three groups. Disclosure flags ask whether the audience can recognise the commercial nature of the content. UCPD Article 5(3) requires the assessment to account for an identifiable vulnerable group, so disclosure clear to an adult may still be inadequate for children. The main legal references are UCPD Articles 6–7 and Annex I point 11, AVMSD Articles 10–11 and 28b, and DSA Articles 26 and 28. We label a promotion as undisclosed advertising when the available transcript, description and platform field contain no disclosure. Inadequate disclosure applies when a disclosure is present but may not be clear enough for the intended audience.

Content flags cover misleading claims and direct exhortation. UCPD Articles 6–7 address misleading actions and omissions. Annex I point 28 prohibits directly urging children to buy a product or persuade an adult to buy it for them. We use a narrow test: language that pressures a child to make the purchase happen qualifies, while a generic instruction to click a link, download an app or use a code does not. Product flags cover agerestricted or prohibited products and the marketing of foods high in fat, salt or sugar, grounded mainly in AVMSD Article 9 and DSA Article 28.

The released data includes files that map each flag to the relevant provisions and explain the connection. It provides citations and notes rather than reproducing the law verbatim. The flags indicate cases for further review. They do not capture every fact needed for a legal assessment, and a flag is not a finding of infringement.

## 3 Related Work

Research on influencer advertising has measured disclosure practices across platforms and countries and studied how children and teenagers recognise commercial persuasion (Mathur et al., 2018; Bertaglia et al., 2025; Gui et al., 2025b; De Veirman et al., 2019; Loose et al., 2023). Related work examines children’s economic exploitation, adolescent advertising literacy and food marketing (van der Hof et al., 2020; Zarouali et al., 2020; Martínez-Pastor et al., 2021). These studies establish why disclosure and product type matter for child-facing content, but they do not provide a benchmark for classifying both from the same videos.

Computational work has detected sponsored YouTube segments and used model explanations to support human labelling and assess influencermarketing compliance (Rodrigues et al., 2021; Bertaglia et al., 2023; Gui et al., 2025a). CHILD-SAFEADS uses SponsorBlock’s crowd-submitted segment boundaries for discovery and focuses on classifying the resulting cases. In legal NLP, LexGLUE benchmarks the analysis of legal texts and legal knowledge (Chalkidis et al., 2022), while consumer-law research connects legal rules to datadriven analysis (Lippi et al., 2020). Recent work also combines legal analysis with platform data to audit DSA transparency (Kaushal et al., 2024) and youth safety (Xue et al., 2025).

The closest shared-task precedent is LegalLens, which links the detection of legal violations in ordinary text to supporting legal provisions (Hagag et al., 2024). Inspired by that structure, CHILD-SAFEADS connects evidence from a video, the platform and the promoted product to a set of legally grounded labels.

## 4 What is ChildSafeAds?

Each CHILDSAFEADS instance starts with a crowdsubmitted SponsorBlock interval: a start and end time within a YouTube video where a contributor identified a sponsorship. The dataset also includes the words spoken during that interval, information about the video and channel, and one likely sales or service page linked from the video description. Systems do not decide whether the segment is commercial or whether the channel is child-facing. These are assumptions made when constructing the dataset. Systems perform one or more of the three tasks below.

Given: marked sponsorship segment → ST1: offer type → ST2: product category → ST3: compliance-risk flags

The three tasks share the same instances and evidence. Teams may enter any subset. Because every instance begins with a SponsorBlock mark, the benchmark evaluates classification after a likely commercial segment has been found. It does not evaluate commercial-content detection and contains no non-commercial examples.

## 4.1 ST1: Commercial Type

ST1 asks what the consumer receives through the promoted transaction. It assigns one of five labels:

physical goods, digital content or services, physical services, no identifiable offer, or other.

## 4.2 ST2: Product Category

ST2 asks what kind of product is being promoted. It assigns one or more of twelve categories covering apps, hardware and electronics, food, fashion, health, education, financial products, gambling, gambling-adjacent mechanics, toys, creator communities and other products.

## 4.3 ST3: Compliance-Risk Indicators

ST3 asks which aspects of the promotion potentially require a closer legal review. It assigns any of six flags: misleading claim, inadequate disclosure, undisclosed advertising, direct exhortation, age-restricted or prohibited product, and marketing for foods high in fat, salt or sugar (HFSS). The labels no flag and insufficient context stand alone. Systems predict the flags, not their severity or the relevant legal provisions. The labelled data include short supporting quotes when the quoted text matches the released evidence.

The task also includes an unscored bonus track for risks that require participants to collect further information about a shop, account or product.

Figure 1 shows how the fields work together. The linked page identifies an online betting service, which determines the product category. The transcript says that a bet cannot lose, supporting the misleading-claim flag. The video did not include a paid-promotion label, but the speaker names the sponsor in the segment, so the instance receives no disclosure flag.

## 5 Dataset Construction

## 5.1 Source Data and Sampling

On 5 May 2026, we downloaded a copy of SponsorBlock’s public database. We selected intervals submitted in the sponsorship category when they had at least one more upvote than downvote. When a video contained several eligible intervals, we kept the highest-voted one. Contributors make these submissions to improve their viewing experience, not to make a legal determination. We use them only to build the initial pool of likely commercial segments.

We then selected channels likely to reach a substantial teenage audience. The screening combined keywords and other public information about each channel with a GPT-5.4 classifier. It was designed to remove clearly adult-oriented channels, not to estimate each viewer’s age. In the released task, “child-facing” means that a channel passed this screening process. Systems receive that designation and do not predict it.

<table><tr><td>Split</td><td>Instances</td><td>Channels</td><td>Videos</td></tr><tr><td>Train</td><td>2,353</td><td>632</td><td>2,353</td></tr><tr><td>Dev</td><td>504</td><td>154</td><td>504</td></tr><tr><td>Test</td><td>503</td><td>153</td><td>503</td></tr><tr><td>Total</td><td>3,360</td><td>939</td><td>3,360</td></tr></table>

Table 1: The v1.0 split is channel-disjoint, so recurring channel style and disclosure habits cannot cross split boundaries. Brands and destinations are not held out.

## 5.2 Evidence Collection

For each selected video, we collected the transcript around the marked interval, the video title and description, and basic channel information. We also followed links in the description, resolved redirects and extracted text from the page most likely to describe the promotion. We added a video to the dataset only when at least one link produced a usable page. This requirement removed 1,116 of 4,985 candidates (22.4%), so the dataset underrepresents expired links and pages that block automated access.

Video descriptions often contain several links. GPT-5.4 compared the marked segment with the candidate links and selected a likely matching page for 2,092 instances. For the other 1,208, no link could be matched confidently, so we kept the first usable page in the description. Some of these fallback pages may belong to a different promotion in the same video. Another 60 instances have no transcript, but still include the available video, channel and page evidence.

We also recorded whether YouTube reported that the video “Includes paid promotion”. On 17 July 2026, this field was present for 1,791 videos, absent for 1,529 and unavailable for 40. The 45.5% absence rate is tied to that collection date because YouTube can change both the label and the interface that exposes it.

## 5.3 Splits

The 3,360 instances cover 3,360 videos and 939 channels (Table 1). No channel appears in more than one split. A system therefore cannot learn a channel’s presentation style or disclosure habits from training data and encounter the same channel at test time. Brands and destination pages can recur, so the benchmark tests generalisation to new channels rather than new advertisers.

<table><tr><td>Record stage</td><td>Anonymised content</td></tr><tr><td>1. Commercial segment</td><td>The speaker names [brand] as the sponsor, promotes a sports bet and says that the bet cannot lose.</td></tr><tr><td>2. Description link 3. Resolved page</td><td>[URL] [page title]; online sports betting and gambling service.</td></tr><tr><td>Platform disclosure</td><td>false</td></tr><tr><td>Task output</td><td>Released label</td></tr><tr><td>ST1</td><td>digital_content_or_services</td></tr><tr><td>ST2</td><td>gambling</td></tr><tr><td>ST3</td><td>age_restricted_or_prohibited_product;misleading_claim</td></tr><tr><td></td><td>4. Released evidence age_restricted_or_prohibited_product: “This episode of [programme] is</td></tr><tr><td></td><td>sponsored by [brand].&quot; The resolved page identifies an online betting service. misleading_claim: “[brand] is giving you a bet you can&#x27;t lose.&quot;</td></tr></table>

Figure 1: A real training instance with identifiers removed. Source summaries are paraphrased; evidence quotes reproduce the release after redaction and punctuation adjustments. Labels are unchanged.

## 5.4 Data Access Levels

Table 2 groups the released fields by the work needed to collect them. Level 1 requires only the marked transcript. Level 2 adds video metadata, level 3 adds channel information, and level 4 adds the crawled promotional page. The levels are cumulative. Teams report the highest level they use, allowing fair comparison between systems with different data requirements.

Tables 3 and 4 show strong label imbalance. The most common ST3 flag is the misleading claim, while each product-risk flag has fewer than 60 training examples. The rarest labels are ST1 other (2), ST2 gambling (12) and ST3 insufficient context (15). Macro-F1 gives each label equal weight, but scores for these rare labels will remain sensitive to a small number of predictions.

Table 5 shows two examples of the released record structure. The release also includes the label taxonomy and legal-reference file. SponsorBlockderived data retain the CC BY-NC-SA 4.0 licence. The data-use agreement covers video-derived fields, requires compliance with YouTube’s Terms of Service, and prohibits redistribution, re-identification, contact and publications that single out a creator.

## 6 Annotation and Reliability

## 6.1 LLM Annotation

We used GPT-5.4 as a judge: the model received the evidence and label definitions for an instance and returned the applicable labels. The process was iterative. The organiser team, including a legal expert, reviewed samples, discussed errors and revised the taxonomy, prompts and model choices. This review led, for example, to a narrower definition of direct exhortation and a new annotation pass for that flag. The sample review was not a systematic double-annotation study, so we do not report human inter-annotator agreement.

We used GPT-5.4 as the primary judge. For ST1 and ST2 it received the extracted brand, the title and text of the linked page, the transcript and the video description. The main ST3 pass received the same evidence without the page title, together with YouTube’s paid-promotion field and summaries of the relevant legal provisions. The revised directexhortation pass used only the brand and transcript.

To measure label stability across models, GPT-5.6-luna independently labelled the 504 development instances. The comparison below is crossmodel agreement after expert-guided task development, not agreement between human annotators.

## 6.2 Cross-Model Agreement

The two models assign the same ST1 label in 94.4% of cases. For multi-label ST2, they assign exactly the same label set in 66.3% of cases; their average set overlap, measured by Jaccard similarity, is 0.74. They assign exactly the same full ST3 flag set in 44.4% of cases (Table 6). The largest differences concern inadequate disclosure and misleading claims: the primary judge assigns inadequate disclosure in 98 cases where the second does not, while the second assigns misleading claim in 128 cases where the primary does not. We report both fine-grained and family-level ST3 scores so that results can also be read at the broader disclosure, content and product levels.

<table><tr><td>Level</td><td>Field group</td><td>Contents</td><td>Collection cost</td></tr><tr><td>1</td><td>transcript</td><td>Segment text, start and end time</td><td>Lowest; platform-scale text</td></tr><tr><td>2</td><td>video_context</td><td>Video ID, title, description, plat- form disclosure</td><td>One metadata call per video</td></tr><tr><td>3</td><td>channel_context</td><td>Channel ID and channel name</td><td>One channel lookup</td></tr><tr><td>4</td><td>product_page</td><td>Raw and resolved URL, page ti- tle and page text</td><td>Outbound crawl; pages may be dead or block automation</td></tr></table>

Table 2: Access levels are ordered by the cost of obtaining fields at platform scale. Teams report the highest level used, so interpret accuracy alongside collection cost.

ST1: commercial type
<table><tr><td>Label</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>Physical goods</td><td>1,125</td><td>218</td><td>235</td></tr><tr><td>Digital content or services</td><td>1,069</td><td>248</td><td>225</td></tr><tr><td>Physical services</td><td>122</td><td>24</td><td>33</td></tr><tr><td>No identifiable offer</td><td>35</td><td>14</td><td>9</td></tr><tr><td>Other</td><td>2</td><td>0</td><td>1</td></tr></table>

ST2: product category
<table><tr><td>Label</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>Apps</td><td>654</td><td>171</td><td>124</td></tr><tr><td>Hardware and electronics</td><td>516</td><td>120</td><td>99</td></tr><tr><td>Other</td><td>412</td><td>87</td><td>97</td></tr><tr><td>Food</td><td>293</td><td>49</td><td>52</td></tr><tr><td>Creator community</td><td>269</td><td>47</td><td>70</td></tr><tr><td>Fashion</td><td>267</td><td>50</td><td>78</td></tr><tr><td>Health</td><td>264</td><td>55</td><td>50</td></tr><tr><td>Education</td><td>140</td><td>12</td><td>18</td></tr><tr><td>Financial products</td><td>135</td><td>20</td><td>56</td></tr><tr><td>Gambling-adjacent</td><td>92</td><td>17</td><td>25</td></tr><tr><td>Toys</td><td>77</td><td>8</td><td>29</td></tr><tr><td>Gambling</td><td>12</td><td>5</td><td>3</td></tr></table>

Table 3: ST1 and ST2 label counts by split. ST1 assigns one commercial type to each instance; ST2 may assign multiple product categories.

## 7 Evaluation Setup

We hosted the competition on CodaBench, an online platform for running reproducible benchmarks (Xu et al., 2022). Teams could enter any subset of the three tasks. A submission contains one machine-readable prediction record for each instance; a missing prediction receives zero credit. The development phase ran from 20 July to 10 August 2026. The evaluation phase ran from 11 to 18 August and closed at 00:00 UTC on 19 August. The final export contains submissions from 21 teams, excluding the organiser baseline.

For each task, we compute F1 for every label and average the labels equally (macro-F1). The overall score is the mean of the three task scores. We also report prediction coverage, the share of instances for which a system returned a prediction, and a family-level ST3 score that groups the six flags into disclosure, content and product risks. The supplied majority-class baseline scores 0.093 on the development set, with 0.151 for ST1, 0.042 for ST2 and 0.085 for ST3.

Each submission states the highest data-access level used, which CodaBench displays beside the score. Teams can also report their method, estimated cost and whether they retrieved legal material beyond the references supplied with the task. Legal grounding is not scored because the submission format does not support a reliable evaluation of retrieved provisions. The system reports will instead show how teams used legal material and what each additional data source contributed.

## 8 Discussion, Limitations, and Ethics

The access levels turn a practical constraint into part of the evaluation. A transcript can be processed cheaply across many videos. A promotional page can make the offer much easier to identify, but every page must be found, resolved and crawled.

<table><tr><td>ST3 flag</td><td>Legal family</td><td>Severity</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>Misleading claim</td><td>Content</td><td>Conditional</td><td>1,277</td><td>260</td><td>259</td></tr><tr><td>Inadequate disclosure</td><td>Disclosure</td><td>Conditional</td><td>611</td><td>118</td><td>139</td></tr><tr><td>No flag</td><td>Housekeeping</td><td>None</td><td>529</td><td>127</td><td>112</td></tr><tr><td>Undisclosed advertising</td><td>Disclosure</td><td>Per se</td><td>352</td><td>74</td><td>89</td></tr><tr><td>Direct exhortation</td><td>Content</td><td>Per se</td><td>304</td><td>77</td><td>71</td></tr><tr><td>Age-restricted or prohibited product</td><td>Product</td><td>Conditional</td><td>59</td><td>16</td><td>17</td></tr><tr><td>HFSS food marketing</td><td>Product</td><td>Soft law</td><td>40</td><td>5</td><td>5</td></tr><tr><td>Insufficient context</td><td>Housekeeping</td><td>None</td><td>15</td><td>7</td><td>6</td></tr></table>

Table 4: ST3 compliance-risk counts by split. ST3 is multi-label; family-level evaluation groups the six substantive flags as disclosure, content or product risks. “Per se” marks a practice prohibited in itself, “conditional” requires contextual assessment, and “soft law” refers to non-binding guidance. The two housekeeping labels carry no severity. Severity is not predicted.
<table><tr><td>Record state</td><td>Available evidence</td><td>Released labels</td></tr><tr><td>Complete</td><td>Transcript present; description link [URL]; resolved [page title] with extracted product text; YouTube&#x27;s “Includes paid promotion&quot;label absent.</td><td>ST1 digital content or services; ST2 other; ST3 direct exhortation and misleading claim.</td></tr><tr><td>Transcript missing</td><td>Transcript empty; description link [URL]; resolved [page title] with extracted product text; YouTube&#x27;s “Includes paid promotion&quot; label present.</td><td>ST1 physical goods; ST2 hardware and electronics; ST3 inadequate disclosure.</td></tr></table>

Table 5: Two real training records with identifiers removed. The second retains description, page and platform evidence despite a missing transcript. Labels are unchanged.

<table><tr><td>Subtask or label</td><td>Measure</td><td>Value</td></tr><tr><td>ST1</td><td>Exact label</td><td>476/504 (94.4%)</td></tr><tr><td>ST2</td><td>Mean set Jaccard</td><td>0.74</td></tr><tr><td>ST2</td><td>Exact set</td><td>334/504 (66.3%)</td></tr><tr><td>ST3</td><td>Exact set</td><td>224/504 (44.4%)</td></tr><tr><td>Direct exhortation</td><td>Per flag</td><td>439/504 (87.1%)</td></tr><tr><td>No flag</td><td>Per flag</td><td>389/504 (77.2%)</td></tr></table>

Table 6: Cross-model agreement on 504 development instances. No human inter-annotator agreement figure is available.

The shared-task results will show whether that extra evidence improves performance enough to justify the cost.

The dataset does not represent all advertising on YouTube. It contains only videos with a Sponsor-Block sponsorship mark and a promotional page that we could crawl. Both requirements introduce selection bias, and we have no non-commercial examples to measure discovery at a realistic base rate. The channel screen estimates which channels are likely to reach teenagers from public metadata; it does not measure the actual audience. The selected page could not be aligned to the marked segment in 1,208 instances, and 60 instances have no transcript. Pages can also change after collection.

The benchmark is predominantly English and text-based. It does not include video frames, visual disclosure timing, non-verbal audio or all transaction details that could matter in a legal assessment. These omissions are especially relevant to ST3, where exact cross-model agreement is 44.4%. We used GPT-5.4 as a judge to produce the final labels, informed by expert review and validation of a sample that shaped the task definitions and prompts. Systems are evaluated against the released taxonomy.

The release contains no viewer records or measured audience demographics, but it includes channel names, video identifiers, titles, and descriptions. The data-use agreement prohibits redistribution, re-identification, contact, harassment and publications that single out a creator. We redact identifiers in worked examples. The dataset is intended for research and human review, not profiling or automatic enforcement.

Acknowledgements. We thank the SponsorBlock contributors and the participating teams. This research has been supported by funding from the ERC Starting Grant HUMANads (ERC-2021-StG No 101041824).

## References

Thales Bertaglia, Catalina Goanta, Gerasimos Spanakis, and Adriana Iamnitchi. 2025. Influencer selfdisclosure practices on Instagram: A multi-country longitudinal study. Online Social Networks and Media, 45:100298.

Thales Bertaglia, Stefan Huber, Catalina Goanta, Gerasimos Spanakis, and Adriana Iamnitchi. 2023. Closing the loop: Testing ChatGPT to generate model explanations to improve human labelling of sponsored content on social media. In Explainable Artificial Intelligence, pages 198–213. Springer Nature Switzerland.

Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Katz, and Nikolaos Aletras. 2022. LexGLUE: A benchmark dataset for legal language understanding in English. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4310–4330, Dublin, Ireland. Association for Computational Linguistics.

Marijke De Veirman, Liselot Hudders, and Michelle R. Nelson. 2019. What is influencer marketing and how does it target children? a review and direction for future research. Frontiers in Psychology, 10:2685.

European Parliament and Council of the European Union. 2005. Directive 2005/29/ec concerning unfair business-to-consumer commercial practices in the internal market. Accessed 18 August 2026.

European Parliament and Council of the European Union. 2010. Directive 2010/13/eu on the coordination of certain provisions laid down by law, regulation or administrative action in member states concerning the provision of audiovisual media services. Accessed 18 August 2026.

European Parliament and Council of the European Union. 2011. Directive 2011/83/eu on consumer rights. Accessed 18 August 2026.

European Parliament and Council of the European Union. 2022. Regulation (eu) 2022/2065 on a single market for digital services. Accessed 18 August 2026.

Haoyang Gui, Thales Bertaglia, Taylor Annabell, Catalina Goanta, Tjomme Dooper, and Gerasimos Spanakis. 2025a. Evaluating LLM-generated legal explanations for regulatory compliance in social media influencer marketing. In Proceedings of the Natural Legal Language Processing Workshop 2025, pages 157–171, Suzhou, China. Association for Computational Linguistics.

Haoyang Gui, Thales Bertaglia, Catalina Goanta, Sybe de Vries, and Gerasimos Spanakis. 2025b. Across platforms and languages: Dutch influencers and legal disclosures on Instagram, YouTube and TikTok. In Social Networks Analysis and Mining, pages 3–12. Springer Nature Switzerland.

Ben Hagag, Gil Gil Semo, Dor Bernsohn, Liav Harpaz, Pashootan Vaezipoor, Rohit Saha, Kyryl Truskovskyi, and Gerasimos Spanakis. 2024. LegalLens shared task 2024: Legal violation identification in unstructured text. In Proceedings ofthe Natural Legal Language Processing Workshop 2024, pages 361–370, Miami, FL, USA. Association for Computational Linguistics.

Rishabh Kaushal, Jacob van de Kerkhof, Catalina Goanta, Gerasimos Spanakis, and Adriana Iamnitchi. 2024. Automated transparency: A legal and empirical analysis of the digital services act transparency database. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, pages 1121–1132. Association for Computing Machinery.

Marco Lippi, Giuseppe Contissa, Agnieszka Jabłonowska, Francesca Lagioia, Hans-Wolfgang Micklitz, Przemysław Palka, Giovanni Sartor, and Paolo Torroni. 2020. The force awakens: Artificial intelligence for consumer law. Journal ofArtificial Intelligence Research, 67:169–190.

Femke Loose, Liselot Hudders, Steffi De Jans, and Ini Vanwesenbeeck. 2023. A qualitative approach to unravel young children’s advertising literacy for youtube advertising: In-depth interviews with children and their parents. Young Consumers, 24(1):74– 94.

Esther Martínez-Pastor, Ricardo Vizcaíno-Laorga, and David Atauri-Mezquida. 2021. Health-related food advertising on kid youtuber vlogger channels. Heliyon, 7(10):e08178.

Arunesh Mathur, Arvind Narayanan, and Marshini Chetty. 2018. Endorsements on social media: An empirical study of affiliate marketing disclosures on youtube and pinterest. Proceedings ofthe ACM on Human-Computer Interaction, 2(CSCW):1–26.

Joao P. Santos Rodrigues, Ana C. Munaro, and Emerson Cabrera Paraiso. 2021. Identifying sponsored content in youtube using information extraction. In 2021 IEEE International Conference on Systems, Man, and Cybernetics, pages 3075–3080. IEEE.

Simone van der Hof, Eva Lievens, Ingrida Milkaite, Valerie Verdoodt, Tineke Hannema, and Ton Liefaard. 2020. The child’s right to protection against economic exploitation in the digital world. The International Journal ofChildren’s Rights, 28(4):833–859.

Zhen Xu, Sergio Escalera, Adrien Pavão, Magali Richard, Wei-Wei Tu, Quanming Yao, Huan Zhao, and Isabelle Guyon. 2022. Codabench: Flexible, easy-to-use, and reproducible meta-benchmark platform. Patterns, 3(7):100543.

Linda Xue, Francesco Corso, Nicolo Fontana, Geng Liu, Stefano Ceri, and Francesco Pierri. 2025. Towards an automated framework to audit youth safety on TikTok. In Proceedings of the Fourth Workshop

on Bridging Human-Computer Interaction and Natural Language Processing, pages 113–119, Suzhou, China. Association for Computational Linguistics.

Brahim Zarouali, Valerie Verdoodt, Michel Walrave, Karolien Poels, Koen Ponnet, and Eva Lievens. 2020. Adolescents’ advertising literacy and privacy protection strategies in the context of targeted advertising on social networking sites: Implications for regulation. Young Consumers, 21(3):351–367.