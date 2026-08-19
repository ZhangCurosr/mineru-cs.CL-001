# Domain-Adapted Molecular Language Models for Efficient Search of Make-on-Demand Libraries

Henrik Wille,<sup>a</sup> Luis-Finley Schütz,<sup>a</sup> and Felix Strieth-Kalthoff \*<sup>,a,b</sup>

a University of Wuppertal, School of Mathematics and Natural Sciences, Gaußstr. 20, 42119 Wuppertal, Germany

b University of Wuppertal, Interdisciplinary Center of Machine Learning and Data Analytics, Gaußstr. 20, 42119 Wuppertal, Germany

水 Corresponding author: strieth-kalthoff@uni-wuppertal.de

Pretrained molecular language models are increasingly used as molecular encoders for learning structure–property relationships. However, their practical suitability for molecular discovery within and beyond their pretraining domain remains unclear. Herein, we systematically benchmark four molecular language models across six virtual molecular libraries spanning drug discovery, organic materials, and catalysis. Native molecular language model embeddings show substantial variation in discovery performance across libraries, whereas molecular fingerprints provide a consistently strong and robust baseline. Consistent with a potential domain–representation mismatch, we show that explicit domain adaptation substantially improves representation performance. Fine-tuning molecular language model encoders on structures from the target virtual library consistently improves sample efficiency, with several adapted encoders emerging as the top-performing representations across the benchmark tasks. These results show that molecular representation quality depends strongly on the target domain and that explicit adaptation can improve the practical utility of molecular foundation models. More broadly, our findings establish domainadapted molecular representations as a promising strategy for sample-efficient adaptive decision making in virtual screening and self-driving laboratories.

## Introduction

The rapid discovery of functional molecules is central to many areas of chemistry, including medicinal chemistry, agricultural chemistry, and materials science.[1–3] In recent years, data-driven decision-making has become increasingly important for closing the loop between molecular property measurements and the selection of subsequent candidate molecules, particularly via the use of machine learning methods.[4–9] In many practical settings, these candidates are selected from a predefined, make-ondemand virtual library, whose members are enumerated from available molecular building blocks and validated synthetic reactions. Such libraries constrain the search to candidate molecules that are typically accessible at a feasible experimental cost.[10,11]

Despite recent advances in generative molecular design,[12,13] virtual library search remains widely used in real-world discovery scenarios.[14] Two recent developments have reinforced the importance of this approach. First, commercially available virtual libraries, most prominently the Enamine REAL space, have grown to the billion-molecule scale, covering an increasingly broad range of molecular structures.[15] Although exhaustive virtual screening of such libraries is infeasible, datadriven decision-making can identify promising candidates by computationally evaluating only a small, adaptively selected fraction of the library.[16,17] Second, automated laboratories are well suited for accessing molecules from combinatorial libraries, if candidate synthesis relies on a small number of well-defined synthetic procedures. When coupled with automated property measurements, data-driven decision-making enables closed design–make–test–analyze loops (Fig. 1a).[18,19] Although the specific costs and evaluation budgets differ between virtual screening and experimental discovery settings, both scenarios require models that effectively learn molecular structure– property relationships from limited property evaluations.

Molecular representation is central to learning and generalizing such structure–property relationships within a virtual library. Over the last decade, traditional cheminformatic representations,[20] such as physicochemical descriptors and structural fingerprints,[21] have been complemented by learned embeddings from deep neural networks, including graph neural networks[22,23] and molecular language models (Fig. 1b).[24,25] Such models, often discussed in the context of molecular foundation models,[26] learn representations through selfsupervised pretraining on large collections of molecular structures. Particularly transformer models, that operate on SMILES notation and are trained with established objectives from the field of language modeling, have gained widespread attention as transferable representations for molecular property prediction and adaptive decision making.[27–32]

In molecular discovery scenarios, however, pretrained molecular encoders may be poorly matched to the target virtual library (Fig. 1c). Common pretraining datasets typically originate from widely used molecular databases and catalogs, including PubChem, ChEMBL and ZINC, which are dominated by druglike molecules. As a result, the learned embedding spaces are primarily shaped by this region of chemical space. A focused drug-discovery library may occupy only a small region of this space, whereas libraries designed for materials chemistry or catalysis may occupy different regions altogether. As a result, a virtual library for a given molecular discovery problem can therefore be shifted relative to the pretraining distribution, or compressed within the model’s embedding space. Such observations motivate a representation–domain mismatch hypothesis: candidate molecules may be insufficiently differentiated by the representation, which can limit the learning of structure–property relationships, especially in the presence of property cliffs.[17,33] This raises the broader question of how well pretrained molecular representations are suited for iterative discovery within and beyond their pre-training domain. Importantly, however, a distribution shift or embedding contraction alone does not establish that a representation is inadequate. For molecular discovery, representation quality ultimately depends on how effectively the representation supports the downstream surrogate model and decision-making procedure.

![](images/4dfc51ca7c2e142c24b73ba2dfed3c112f6d475b3652800e25ff6d6b70bdfd2c.jpg)

![](images/4e4ca98df331a7437b59d12180eaf64eb82673cee95f244a3a6adee5ef4ff5e3.jpg)

![](images/eb638fed0b360eae32b387fcf589a897f34307b72f7ba64155d39968e32a8d80.jpg)  
Figure 1: a) Design–Make–Test–Analyze cycles for discovering functional molecules from virtual make-on-demand libraries. b) Molecular representations as central factor governing sample-efficient discovery. c) Schematic two-dimensional projections of a pretrained molecular representation for a drug discovery library (circles, light blue) and a virtual library for materials discovery (squares, dark blue). Domain adaptation may improve the ability to differentiate molecules within a virtual library.

Herein, we examine the association between representation– domain fit and the efficiency of iterative molecular discovery. Inspired by recent demonstrations of task- or domain-specific fine-tuning for molecular property prediction,[34–38] we investigate whether potential mismatches can be mitigated by domain adaptation to the distribution of a given virtual library (Fig. 1c). Importantly, for practical use in iterative discovery, such adaptation should rely only on access to molecular structures from the candidate library and should not require costly property labels. To this end, we perform systematic benchmarks of molecular discovery efficiency under both virtual screening and experimental discovery budgets across six virtual libraries spanning drug discovery, materials chemistry, and catalysis. Overall, we aim to identify principles that govern efficient molecular discovery and provide practical guidance for selecting and adapting molecular representations in new discovery settings. We foresee that such insights could enable more efficient use of computational and experimental resources in real-world molecular design campaigns, including virtual screening for drug discovery, or self-driving laboratories for the discovery of functional molecular materials.

## Results and Discussion

Benchmark experiments were performed on six virtual libraries covering drug discovery, materials chemistry, and catalysis. The drug discovery tasks comprised a library of \~1.9 million molecules from ZINC15 with docking scores against an AmpC β-lactamase (Zinc-AmpC);[39] a focused subset of \~1.5 million ZINC15 molecules satisfying the rule-of-four criteria, with docking scores against 8-oxoguanine DNA glycosylase (RO4- OGG1);[40] and \~2.1 million molecules from the Enamine highthroughput screening collection, with docking scores against a thymidylate kinase (Enamine-TMPK).[17] The remaining three libraries represent applications outside drug discovery: a set of \~2.3 million molecules from the Harvard Clean Energy Project,[41] labeled with density functional theory-derived power conversion efficiencies (HCEP-PCE);[42] a combinatorial library of \~180,000 potential organic laser emitters with computed fluorescence oscillator strengths (Laser-OSC);[43] and \~300,000 phosphine ligands from the Kraken database[44] with predicted enantioselectivities in a Pd-catalyzed cross-coupling (Kraken-ES).[45] Further information on library construction, together with detailed descriptions of the molecular and target property distributions, is provided in the Supplementary Information (SI).

![](images/8f5a9459d71b5ad81e5799ab9fff541b36a06e3f38c6d70dbe16237a80b92c29.jpg)

![](images/b0627feda65a7593236314c9ece818eb03668ee1eed7154ab17922b66c98b775.jpg)  
Figure 2: Distribution of candidate molecules in the virtual libraries relative to a reference library of 25,000 randomly selected compounds from the ZINC15 library. Top: A t-distributed stochastic neighbor embedding (t-SNE) model was fitted to Morgan fingerprints of all compounds in the reference library, and molecules from the target libraries were projected into the same t-SNE embedding space. Bottom: Distributions of pairwise molecular distances for 5 million molecule pairs sampled from within each library, computed using Tanimoto similarity on molecular fingerprints (light blue, left) and cosine distance on MolFormer embeddings (dark blue, right).

Virtual molecular libraries occupy distinct regions of chemical and representation space.

Chemical space analysis confirms that these libraries show systematic differences in molecular distribution (Fig. 2). To investigate distribution shifts and possible representation– domain mismatches, we collected a reference set of 25,000 randomly selected ZINC15 molecules as a proxy for the druglike chemical space represented in common pretraining datasets.[46] Qualitative projection of the target relative to this reference set (Fig. 2a) suggests a substantial overlap for the three drug-discovery libraries, whereas the materials and catalysisfocused libraries show the expected domain shift. Pairwise molecular distance distributions between the reference and each target library (see SI) support this conclusion, indicating that the libraries outside drug discovery could be well suited for investigating potential representation–domain mismatches.

We therefore investigated the distribution of these libraries within the embedding spaces of several pretrained molecular language models with different transformer architectures:

ChemBERTa, a RoBERTa-type encoder (12M parameters), trained through masked-token prediction on 100k molecules from ZINC15 and ChEMBL;[47,48]

MolFormer, an encoder-only transformer with linear attention and rotary positional embeddings (45M parameters), pretrained on approximately 1.1B molecules from ZINC and PubChem;[49]

\- T5Chem, a T5 encoder–decoder model (110M encoder parameters) pretrained on 33.5M training samples;[50] and

SmiTed, a transformer-based encoder–decoder model with a learned, autoencoder-based pooling and reconstruction framework (55M encoder parameters), pretrained on \~91M curated SMILES strings from PubChem.[51]

Exact checkpoints, embedding dimensions, and pooling procedures are provided in the SI.

Analysis of pairwise distance distributions in embedding space shows a substantial contraction for the non-drug libraries (Fig. 2b), indicating that the molecules occupy a more restricted region of embedding space. Similar dispersion was observed across all molecular language models, with SmiTed producing particularly narrow embedding distributions (see SI for further details). While these findings may provide an initial indication of a potential representation–domain mismatch, it should be noted that a library occupying a narrow region in embedding space does not necessarily imply the loss of information for learning structure–property relationships. We therefore treat this embedding dispersion as a candidate diagnostic, and evaluate its relationship to discovery efficiency in the following sections.

![](images/37ea7952f731659a976ba72ee38e2ae079b6e0fae1f09a3788e06c130bcc9384.jpg)

B Relative Representation Effects per Library  
![](images/4f314141a8819cc5172fb3032c93d3e3cb5036f0f6a8c421b9685b1a8f9c3f7f.jpg)  
D Domain Indicators of Encoder Embedding Performance

C Global Representation Effect Ratios  
![](images/a4e8b9a0c71a51d0cd69e03b1dc4f9f744038bd5cf1fb08908823ac827901b83.jpg)

![](images/57740596514fd3b3bac58eabd53c0a2919a92bc30047ce8d8c455dae61c11f8a.jpg)  
Figure 3: Evaluation of molecular representations for iterative discovery in virtual molecular libraries. a) Top-1% recall trajectories averaged over 20 independent runs. All experiments used a Laplace-approximated neural network surrogate and an upper confidence bound candidate ranking with independent top-k batch selection. b) Libraryspecific representation effects, and c) global representation effects from the hierarchical Bayesian analysis of run-level AUOC values, reported relative to Morgan fingerprints. d) Exploratory correlations between library distribution descriptors and benchmark optimization performance. Further experimental details are provided in the Supplementary Information. AUOC: area under the optimization curve.

## Native molecular language model representations show variable performance across tasks.

Motivated by these observations, we evaluated the molecular representations in benchmark tasks of iterative, model-based discovery across two complementary acquisition-budget regimes. For the virtual screening scenario, we used a total budget of 20,000 property evaluations in batches of 1,000, representing feasible budgets for computational campaigns in which e.g. molecular docking is used for property evaluation. The experimental discovery regime was limited to batches of 50 evaluations up to a total budget of 1,000 evaluations, approximating the budget constraints of a self-driving laboratory scenario in which each evaluation requires molecular synthesis and experimental property measurement. After each acquisition round, a neural network surrogate model with a Laplace approximation for uncertainty quantification was trained on all previously evaluated candidates. An upper confidence bound acquisition function then was used to rank unevaluated candidates, and select the top-scoring candidates as a batch. Comparisons with alternative surrogate models and acquisition strategies are provided in the SI.

For each experiment, we recorded optimization trajectories for the recall of the top 1% library candidates as a function of the cumulative evaluation budget. All experiments were repeated 20 times using independently sampled seed populations. From each trajectory, we computed the area under the optimization curve (AUOC) and fit a hierarchical Bayesian model to the run-level AUOC values, in order to separate global representation effects from library-specific contributions and variation between seed populations.

Morgan fingerprints were consistently among the strongest native representations across the six discovery tasks. A quantitative comparison of traditional cheminformatic representations and molecular language model embeddings shows that, while significant library-by-library differences in top-1% candidate recall trajectories are observed (Fig. 3a), Morgan fingerprints consistently rank in the best-performing representations. Statistical analysis reveals that all other tested representations, including descriptors and language model embeddings, showed significant negative global posterior effects relative to this baseline. In line with recent literature precedents, these findings confirm the utility of Morgan fingerprints as a robust baseline for molecular discovery.51 Descriptor-based representations show greater variation among libraries. For certain libraries, especially those from drug discovery, the posterior effects relative to fingerprints are close to zero, whereas performance is lower for other libraries (see Fig. 3b and SI).

Native molecular-language-model embeddings varied substantially in discovery performance. MolFormer and T5Chem were the strongest native molecular language model representations, and approached the performance of Morgan fingerprints in several libraries (Fig. 3b). However, neither model consistently outperforms fingerprints across the full set of benchmark tasks, as evidenced by negative global posterior effects (Fig. 3c). ChemBERTa embeddings show a significantly more negative effect, which may reflect the smaller model size and the potentially lower expressivity of the resulting embeddings. Native SmiTed embeddings show the lowest global effect across all molecular representations tested. This behavior is consistent with the extremely narrow within-library embedding distances observed during the geometric analysis (Fig. S1–S24), indicating that the frozen SmiTed embeddings may not be suitable for downstream property prediction without simultaneous adaptation of the encoder or the learned pooling mechanism.

The observed ordering of representation was largely consistent across the virtual screening and experimental discovery regimes (Fig. 3c), although differences were less pronounced in the experimental discovery scenario. Under the lower experimental budget, Morgan fingerprints and the embeddings from MolFormer and T5Chem show overlapping global posterior effects. One possible explanation for this observation is a stronger contribution from initial-sample variability and exploration. Especially during the first iterations of an experimental discovery campaign, the surrogate model has very limited information about the target property landscape, and candidate selection may be strongly influenced by the initial population and predictive uncertainty. In contrast, once the surrogate is provided with sufficient molecule–property pairs to learn advanced structure–activity relationships, expressive representations may become more important for refining these relationships. While a formal investigation of this hypothesis is beyond the scope of this study, our findings confirm that the general trends across representations remain consistent under both budget regimes investigated.

Overall, these findings identify only a weak relation between representation–domain fit and experimental optimization efficiency. On the one hand, it was notable that both physicochemical descriptors and molecular language model embeddings consistently underperformed molecular fingerprints on the HCEP-PCE library, which also showed the largest domain shift from the ZINC15 reference (mean Tanimoto distance of 0.95). This pattern would be compatible with the representation– domain mismatch hypothesis. On the other hand, language model embeddings supported relatively efficient discovery in the Laser-OSC (T5Chem) and Kraken-ES (MolFormer) libraries, despite significant domain shifts (mean Tanimoto distances of 0.92 and 0.93, respectively). Indeed, exploratory performance correlations identify only a weak negative correlation between pairwise Tanimoto distances to the reference library and the relative posterior effects for different molecular language model embeddings (Fig. 3d). Likewise, within-library embedding dispersion, which we identified above as a diagnostic of potential representation–domain mismatches, shows only a weak positive correlation with relative representation effects. Further correlations between optimization behavior with readily accessible descriptors of the molecular distribution did not reveal stable, interpretable trends.

## Domain adaptation improves discovery efficiency.

Given these weak indications of possible representation–domain mismatches, we evaluated whether the observed underperformance of molecular language models could be mitigated by explicit domain adaptation. We envisioned that, by fine-tuning on molecular structures from the target library, the representation could become more sensitive to structural variation within the virtual library, thereby improving learned structure–property relationships and, in turn, increasing discovery efficiency. For this purpose, we evaluated three conceptually distinct strategies: (i) continued language-modelstyle pretraining with the model’s original masked-token prediction or span corruption objective; (ii) contrastive representation adaption, using triplets of similar and dissimilar molecules from the library in combination with a temperaturescaled multiple-negatives ranking loss;[52,53] and (iii) fingerprintsupervised auxiliary-label domain adaptation,[38] in which a prediction head was trained to reconstruct Morgan fingerprint bits from the encoder representation. All encoder updates were performed using parameter-efficient fine-tuning through lowrank adaptation (LoRA).[54] Full data sampling procedures, losses, and hyperparameters are described in the SI.

![](images/245a6c94a82cfd57b8035796d2152986c08c4a41423f1ee3bd926ebe512da5eb.jpg)

![](images/689057df4aa9790d1cc0a9ad363761d188b77291c4a22f34885a5e49f2a32851.jpg)  
Figure 4: Evaluation of domain-adapted transformer embeddings for iterative discovery in virtual molecular libraries. a) Global representation effects, as obtained from a hierarchical Bayesian analysis of raw trajectories. b) Relative effects of supervised domain adaptation, as obtained from a hierarchical Bayesian analysis of raw trajectories.

Our findings show that optimization efficiency increased significantly when domain-adapted molecular language model embeddings are used. Across all tested libraries and molecular language models, auxiliary-label adaptation produced the largest average gains (Fig. 4a). Notably, the adapted ChemBERTa, MolFormer, and T5Chem embeddings show positive global effects relative to Morgan fingerprints, with broadly similar trends across several libraries and acquisition-budget regimes (see SI for further details).

ChemBERTa benefited most strongly from domain adaptation. Although native embeddings showed significantly negative average effects relative to Morgan fingerprints, the adapted substantially reduced or reversed these effects. This result shows that even a comparatively compact encoder can yield a practical embedding space for efficient optimization, even when the native embedding space is not well-suited to the discovery task. Similarly, we find that MolFormer and T5Chem improve upon their comparatively strong native performance. SmiTed embeddings follow the same directional trend, but the adapted model retained a clear negative posterior effect relative to Morgan fingerprints.

These findings indicate that domain adaptation combines complementary information from pretraining and finetuning,[36] rather than merely reproducing Morgan fingerprints. If the auxiliary-label fine-tuning strategy transformed the encoders into approximate fingerprint calculators, their behavior in molecular discovery scenarios would be expected to approach, but not systematically exceed, that of fingerprints. We hypothesize that the use of fingerprints as auxiliary objectives introduces a bias towards library-specific molecular substructures and local chemical environments, while the pretrained encoder retains more global structural similarity concepts from its initial pretraining corpus. The auxiliary-label strategy therefore provides a simple route for incorporating general chemical expertise and domain-specific information into the molecular language model representation.

Contrastive domain adaptation also improves discovery performance relative to the native embeddings, albeit with smaller gains than those obtained using auxiliary-label finetuning. Nonetheless, the use of a contrastive training objective was beneficial. Continued pretraining with only the original language model objective provided significantly smaller changes (see SI for further details), confirming that the contrastive component contributes directly to the change in representation performance. The observed differences between these three adaptation strategies highlight the value of chemically informed training objectives, and indicate possible directions for further improvement. We foresee that combining complementary domain adaptation objectives, e.g. by combining contrastive learning with auxiliary-label strategies based on fingerprints and physicochemical descriptors, will lead to further improvements in sample-efficient optimization. Similarly, a systematic investigation of fine-tuning hyperparameters is expected to be beneficial, but was beyond the scope of this study.

## Conclusions

Overall, this study shows that molecular discovery from virtual libraries can involve substantial discrepancies between the geometry of pretrained molecular representations and the structure of the target library. Molecular language models are commonly pretrained on corpora dominated by drug-like molecules, whereas virtual libraries for materials discovery and catalysis can occupy distinct regions of chemical space. These differences are reflected in shifted and comparatively narrow distributions within the corresponding embedding spaces, and may influence discovery efficiency. However, these differences explained only little of the observed performance differences in benchmark experiments of molecular discovery. The performance of molecular language model embeddings varies considerably across virtual libraries, and this variation can only partly be attributed to the representation–domain fit. By contrast, molecular fingerprints perform robustly across chemical domains, target properties, and acquisition budgets and outperform the native molecular language model encoders on average. These findings reinforce molecular fingerprints as a strong default representation for iterative molecular discovery.

The main practical result is that shortcomings of the native molecular language model representations can be effectively mitigated through library-specific domain adaptation. Finetuning the molecular language model encoders using molecular fingerprints as auxiliary labels provides consistent performance gains and allows adapted ChemBERTa, MolFormer, and T5Chem embeddings to exceed the Morgan fingerprint baseline. Because this adaptation requires only molecular structures sampled from the target virtual library, it can be performed at the outset of a discovery campaign, and could be applied to billionscale virtual libraries.

The scope of these conclusions is naturally limited by the breadth of the virtual libraries included in our benchmarks, and confirmation across a broader range of molecular distributions and target properties would therefore be valuable. However, such validation is particularly difficult outside drug discovery, where few sufficiently large and consistently labeled molecular libraries are available for systematic benchmarking. Representation performance may also depend on the surrogate model, acquisition function, and batch-selection strategy, although the additional benchmarks indicate that the main trends are robust across several methodological choices. Updating the molecular encoder directly during surrogate model training may further improve its expressivity, but this strategy substantially increases the computational cost of each iteration and may become prohibitive for large virtual libraries.

Overall, our findings provide a basis for developing more effective domain adaptation objectives that incorporate additional forms of chemical information. Together, these developments could extend the utility of molecular foundation models beyond their use as general-purpose feature extractors and enable representations tailored to the molecular domain and decision problem of a specific discovery campaign. In both computational virtual screening and self-driving laboratories, we foresee that domain-adapted representations can improve sample efficiency and accelerate discovery.

## Author contributions

Conceptualization: F. S.-K.; Data curation: H. W., L.-F. S., F. S.-K.; Formal analysis: H.W., L.-F. S., F. S.-K.; Funding acquisition: F .S.-K.; Investigation: H. W., L.-F. S., F. S.-K.; Methodology: F. S.-K.; Project administration: F. S.-K.; Resources: F. S.-K.; Software: F. S.-K.; Supervision: F. S.-K.; Validation: H. W., L.-F. S., F. S.-K.; Visualization: F. S.-K.; Writing – original draft: F. S.-K.; Writing – review & editing: H. W.; L.-F. S.; F. S.-K.

## Conflicts of interest

There are no conflicts to declare.

## Data availability

All experiments reported in this paper were performed using the BAYLEYS (Bayesian Library Exploration and Virtual Screening) Python package, which is available on GitHub at https://github.com/fsk-lab/bayleys. Python scripts for reproducing the described experiments and performing the associated data analyses are provided in the same repository. A snapshot of this repository, together with the complete molecular libraries, experiment configurations, and experimental results is archived on Zenodo (https://doi.org/10.5281/zenodo.21907928).

## Acknowledgements

Computations were performed on the Pleiades HPC center at the University of Wuppertal. Financial support from the Deutsche Forschungsgemeinschaft (SPP2363, "Molecular Machine Learning“, Grant No. 497260357) is gratefully acknowledged. The authors thank Mohammad Haddadnia (Harvard University) for helpful discussions.

## References

[1] J.-L. Reymond, Acc. Chem. Res. 2015, 48, 722–730. https://doi.org/10.1021/ar500432k

[2] C. Lipinski, A. Hopkins, Nature 2004, 432, 855–861. https://doi.org/10.1038/nature03193

[3] H. Bronstein, C. B. Nielsen, B. C. Schroeder, I. McCulloch, Nat. Rev. Chem. 2020, 4, 66–77. https://doi.org/10.1038/s41570-019-0152-9

[4] B. Sanchez-Lengeling, A. Aspuru-Guzik, Science 2018, 361, 360–365. https://doi.org/10.1126/science.aat2663

[5] K. T. Butler, D. W. Davies, H. Cartwright, O. Isayev, A. Walsh, Nature 2018, 559, 547–555. https://doi.org/10.1038/s41586-018-0337-2

[6] B. A. Koscher, R. B. Canty, M. A. McDonald, K. P. Greenman, C. J. McGill, C. L. Bilodeau, W. Jin, H. Wu, F. H. Vermeire, B. Jin, et al., Science 2023, 382, eadi1407. https://doi.org/10.1126/science.adi1407

[7] T. Lookman, P. V. Balachandran, D. Xue, R. Yuan, npj Comput. Mater. 2019, 5, 21. https://doi.org/10.1038/s41524- 019-0153-8

[8] D. Reker, G. Schneider, Drug Discovery Today 2015, 20, 458–465. https://doi.org/10.1016/j.drudis.2014.12.004

[9] J. Vamathevan, D. Clark, P. Czodrowski, I. Dunham, E. Ferran, G. Lee, B. Li, A. Madabhushi, P. Shah, M. Spitzer, et al., Nat. Rev. Drug Discov. 2019, 18, 463–477. https://doi.org/10.1038/s41573-019-0024-5

[10] J. Kuan, M. Radaeva, A. Avenido, A. Cherkasov, F. Gentile, Wiley Interdiscip. Rev. Comput. Mol. Sci. 2023, 13, e1678. https://doi.org/10.1002/wcms.1678

[11] A. V. Sadybekov, V. Katritch, Nature 2023, 616, 673–685. https://doi.org/10.1038/s41586-023-05905-z

[12] Y. Du, A. R. Jamasb, J. Guo, T. Fu, C. Harris, Y. Wang, C. Duan, P. Liò, P. Schwaller, T. L. Blundell, Nat. Mach. Intell. 2024, 6, 589–604. https://doi.org/10.1038/s42256-024-00843- 5

[13] M. Sumita, S. Ishida, K. Yoshizoe, R. Tamura, K. Terayama, K. Tsuda, Chem. Rev. 2026, 126, 3007–3054. https://doi.org/10.1021/acs.chemrev.5c00689

[14] S. M. Papidocha, A. Burger, V. Bernales, A. Aspuru-Guzik, Curr. Opin. Chem. Eng. 2026, 51, 101217. https://doi.org/10.1016/j.coche.2025.101217

[15] O. O. Grygorenko, D. S. Radchenko, I. Dziuba, A. Chuprina, K. E. Gubina, Y. S. Moroz, iScience 2020, 23, 101681. https://doi.org/10.1016/j.isci.2020.101681

[16] F. Gentile, V. Agrawal, M. Hsing, A.-T. Ton, F. Ban, U. Norinder, M. E. Gleave, A. Cherkasov, ACS Cent. Sci. 2020, 6, 939–949. https://doi.org/10.1021/acscentsci.0c00229

[17] D. E. Graff, E. I. Shakhnovich, C. W. Coley, Chem. Sci. 2021, 12, 7866–7881. https://doi.org/10.1039/D0SC06805E

[18] M. Abolhasani, E. Kumacheva, Nat. Synth. 2023, 2, 483–492. https://doi.org/10.1038/s44160-022-00231-0

[19] G. Tom, S. P. Schmid, S. G. Baird, Y. Cao, K. Darvish, H. Hao, S. Lo, S. Pablo-García, E. M. Rajaonson, M. Skreta, et al., Chem. Rev. 2024, 124, 9633–9732. https://doi.org/10.1021/acs.chemrev.4c00055

[20] L. David, A. Thakkar, R. Mercado, O. Engkvist, J. Cheminform. 2020, 12, 56. https://doi.org/10.1186/s13321- 020-00460-5

[21] H. L. Morgan, J. Chem. Doc. 1965, 5, 107–113. https://doi.org/10.1021/c160017a018

[22] D. Duvenaud, D. Maclaurin, J. Aguilera-Iparraguirre, R. Gómez-Bombarelli, T. Hirzel, A. Aspuru-Guzik, R. P. Adams, Adv. Neur. Inf. Proc. Sys. 2015, 29, 2224–2232. https://doi.org/10.48550/arXiv.1509.09292

[23] P. Reiser, M. Neubert, A. Eberhard, L. Torresi, C. Zhou, C. Shao, H. Metni, C. van Hoesel, H. Schopmans, T. Sommer, et al., Commun. Mater. 2022, 3, 93. https://doi.org/10.1038/s43246-022-00315-6

[24] K.-D. Luong, A. Singh, J. Chem. Inf. Model. 2024, 64, 4392– 4409. https://doi.org/10.1021/acs.jcim.3c02070

[25] M. Caldas Ramos, C. J. Collison, A. D. White, Chem. Sci. 2025, 16, 2514–2572. https://doi.org/10.1039/D4SC03921A

[26] J. Choi, G. Nam, J. Choi, Y. Jung, JACS Au 2025, 5, 1499– 1518. https://doi.org/10.1021/jacsau.4c01160

[27] A. Kristiadi, F. Strieth-Kalthoff, M. Skreta, P. Poupart, A. Aspuru-Guzik, G. Pleiss, Proc. Int. Conf. Mach. Learn. 2024, 41, 25603–25622. https://doi.org/10.48550/arXiv.2402.05015

[28] Z. Cao, S. Sciabola, Y. Wang, J. Chem. Inf. Model. 2024, 64, 1882–1891. https://doi.org/10.1021/acs.jcim.3c01938

[29] M. A. Masood, S. Kaski, T. Cui, J. Cheminform. 2025, 17, 58. https://doi.org/10.1186/s13321-025-00986-6

[30] L. Andersen, M. Rausch-Dupont, A. Martínez León, A. Volkamer, J. S. Hub, D. Klakow, Digit. Discov. 2026, 5, 3082–3095. https://doi.org/10.1039/d5dd00522a

[31] N. Evbarunegbe, S. Wa, L. Taylor, A. G. Green, ChemRxiv 2026, https://doi.org/10.26434/chemrxiv.15003911/v2.

[32] M. Haddadnia, Y. Chali, A. Jayaraj, C. Kraay, J. Reis, F. Strieth-Kalthoff, H. Arthanari, Proc. Int. Conf. Mach. Learn. 2026, 43, https://doi.org/10.48550/arXiv.2606.26657.

[33] Z. Zhang, Y. Bian, A. Xie, P. Han, S. Zhou, J. Chem. Inf. Model. 2023, 64, 2921–2930. https://doi.org/10.1021/acs.jcim.3c01707

[34] H. Abdel-Aty, I. R. Gould, J. Chem. Inf. Model. 2022, 62, 4852–4862. https://doi.org/10.1021/acs.jcim.2c00715

[35] P. Zhang, L. Kearney, D. Bhowmik, Z. Fox, A. K. Naskar, J. Gounley, J. Chem. Inf. Model. 2023, 63, 7689–7698. https://doi.org/10.1021/acs.jcim.3c01650

[36] Y. Duan, X. Yang, X. Zeng, W. Wang, Y. Deng, D. Cao, J. Med. Chem. 2024, 67, 9575–9586. https://doi.org/10.1021/acs.jmedchem.4c00692

[37] G. Kallergis, E. Asgari, M. Empting, A. K. H. Hirsch, F. Klawonn, A. C. McHardy, Commun Chem 2025, 8, 114. https://doi.org/10.1038/s42004-025-01484-4

[38] A. Sultan, M. Rausch-Dupont, S. Khan, O. Kalinina, D. Klakow, A. Volkamer, arXiv 2025, https://doi.org/10.48550/arXiv.2503.03360.

[39] J. Lyu, S. Wang, T. E. Balius, I. Singh, A. Levit, Y. S. Moroz, M. J. O’Meara, T. Che, E. Algaa, K. Tolmachova, et al., Nature 2019, 566, 224–229. https://doi.org/10.1038/s41586- 019-0917-9

[40] A. Luttens, I. Cabeza de Vaca, L. Sparring, J. Brea, A. L. Martínez, N. A. Kahlous, D. S. Radchenko, Y. S. Moroz, M. I. Loza, U. Norinder, et al., Nat. Comput. Sci. 2025, 5, 301– 312. https://doi.org/10.1038/s43588-025-00777-x

[41] J. Hachmann, R. Olivares-Amaya, S. Atahan-Evrenk, C. Amador-Bedolla, R. S. Sánchez-Carrera, A. Gold-Parker, L. Vogt, A. M. Brockway, A. Aspuru-Guzik, J. Phys. Chem. Lett. 2011, 2, 2241–2251. https://doi.org/10.1021/jz200866s

[42] J. Hachmann, R. Olivares-Amaya, A. Jinich, A. L. Appleton, M. A. Blood-Forsythe, L. R. Seress, C. Román-Salgado, K. Trepte, S. Atahan-Evrenk, S. Er, et al., Energy Environ. Sci. 2014, 7, 698–704. https://doi.org/10.1039/c3ee42756k

[43] F. Strieth-Kalthoff, H. Hao, V. Rathore, J. Derasp, T. Gaudin, N. H. Angello, M. Seifrid, E. Trushina, M. Guy, J. Liu, et al., Science 2024, 384, eadk9227. https://doi.org/10.1126/science.adk9227

[44] T. Gensch, G. dos Passos Gomes, P. Friederich, E. Peters, T. Gaudin, R. Pollice, K. Jorner, A. Nigam, M. Lindner-D’Addario, M. S. Sigman, et al., J. Am. Chem. Soc. 2022, 144, 1205–1217. https://doi.org/10.1021/jacs.1c09718

[45] S. Zhao, T. Gensch, B. Murray, Z. L. Niemeyer, M. S. Sigman, M. R. Biscoe, Science 2018, 362, 670–674. https://doi.org/10.1126/science.aat2299

[46] T. Sterling, J. J. Irwin, J. Chem. Inf. Model. 2015, 55, 2324– 2337. https://doi.org/10.1021/acs.jcim.5b00559

[47] S. Chithrananda, G. Grand, B. Ramsundar, arXiv 2020, https://doi.org/10.48550/arXiv.2010.09885.

[48] W. Ahmad, E. Simon, S. Chithrananda, G. Grand, B. Ramsundar, arXiv 2022, https://doi.org/10.48550/arXiv.2209.01712.

[49] J. Ross, B. Belgodere, V. Chenthamarakshan, I. Padhi, Y. Mroueh, P. Das, Nat. Mach. Intell. 2022, 4, 1256–1264. https://doi.org/10.1038/s42256-022-00580-7

[50] J. Lu, Y. Zhang, J. Chem. Inf. Model. 2022, 62, 1376–1387. https://doi.org/10.1021/acs.jcim.1c01467

[51] E. Soares, E. Vital Brazil, V. Shirasuna, D. Zubarev, R. Cerqueira, K. Schmidt, Commun. Chem. 2025, 8, 193. https://doi.org/10.1038/s42004-025-01585-0

[52] M. Henderson, R. Al-Rfou, B. Strope, Y. Sung, L. Lukacs, R. Guo, S. Kumar, B. Miklos, R. Kurzweil, arXiv 2017, https://doi.org/10.48550/arXiv.1705.00652.

[53] A. van den Oord, Y. Li, O. Vinyals, arXiv 2019, https://doi.org/10.48550/arXiv.1807.03748.

[54] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, Int. Conf. Learn. Rep. 2021, https://doi.org/10.48550/arXiv.2106.09685.

# Domain-Adapted Molecular Language Models for Efficient Search of Make-on-Demand Libraries

Henrik Wille,<sup>a,†</sup> Luis-Finley Schütz,<sup>a,†</sup> and Felix Strieth-Kalthoff <sup>a,b,</sup>\*

a University of Wuppertal, School of Mathematics and Natural Sciences, Gaußstr. 20, 42119 Wuppertal, Germany

b University of Wuppertal, Interdisciplinary Center of Machine Learning and Data Analytics, Gaußstr. 20, 42119 Wuppertal, Germany

## 1 Experimental Setup

All experiments were performed using our Python library BAYLEYS (Bayesian Library Exploration and Virtual Screening). The code base for reproducing all experiments in this paper are provided on GitHub (https://github.com/fsk-lab/bayleys). The molecular libraries, as well as specific implementation details and experimental procedures, are described below.

## 1.1 General Procedure: Iterative Optimization

To benchmark the efficiency of iterative optimization algorithms, we used virtual molecular libraries for which computed target properties were available for all molecules in the library. The libraries used in our benchmark experiments are described in detail in Section 2. In the context of our benchmark experiments, “observing” the target property corresponds to a lookup in the full library. Before optimization, molecular representation vectors (see Sections 1.3 and 1.4) for all molecules in the library were pre-computed and cached.

Iterative optimization experiments were performed in a batch-wise manner. For initialization, a batch of b molecules were drawn randomly from the full library, and the corresponding target properties were “observed”. These molecules formed the seed dataset of observations. The surrogate model (see Section 1.5) was then trained on this dataset using the precomputed representation vectors as inputs. From the trained model, the predictive mean and uncertainty for all remaining molecules in the virtual library were inferred. These predictions were used to compute an acquisition function value for each unevaluated molecule. The next batch of b molecules was selected for evaluation according to a batch acquisition policy (see Section 1.6). The target properties of these molecules were then “observed”, and the corresponding data points were added to the dataset of observations. This procedure was repeated until a given budget B of observations was exhausted. Each experiment was repeated 20 times with different seed datasets of observations. To enable paired comparisons, the same initial datasets were used across different methods for each library.

All experiments were performed in two settings:

\- Virtual Screening Setting: $b = 1 , 0 0 0 ~ B = 2 0 , 0 0 0$

\- Experimental Discovery Setting: $b = 5 0 B = 1 , 0 0 0$

For quantitative evaluation, we extract optimization trajectories that record the recall of top-k candidates (top-1000 or top-1%) as a function of the number of experimental observations. Averaged optimization trajectories over the 20 repeats are shown for visual comparison of optimizer performance.

For a quantitative statistical comparison between optimizers, we compute the normalized area under the optimization curve (AUOC) through trapezoidal integration, and then separate library, model, and seed effects using a hierarchical Bayesian regression approach.

For that purpose, we analyzed the normalized AUOC values for each library (indexed i), optimizer (indexed j) and seed (index k) using a hierarchical beta regression model

$$
y _ { i j k } \sim \mathrm { B e t a } \big ( \mu _ { i j k } \cdot \phi , \big ( 1 - \mu _ { i j k } \big ) \cdot \phi \big )
$$

where $\mu _ { i j k }$ is the expected normalized AUC, and $\phi$ is the concentration parameter of the beta distribution.

The expected AUC was modeled on the logit scale, representing the paired design described above.

$$
\mathrm { l o g i t } \big ( \mu _ { i j k } \big ) = \alpha + a _ { i } + s _ { i k } + \beta _ { j } + \gamma _ { i j }
$$

The individual terms reflect the sources of variability in our experimental design.

\- 5 is the global intercept.

$a _ { i }$ is the library-specific baseline effect of library i.

$\beta _ { j }$ is the average effect of optimizer j relative to a chosen reference optimizer.

$s _ { i k }$ is the library-specific effect of seed k in library i, accounting for the paired experimental design. Seed populations from different libraries were treated as unrelated.

$\gamma _ { i j }$ is the library-specific deviation of optimizer j from its average effect.

We model $\beta _ { j } , s _ { i k }$ , and $\gamma _ { i j }$ as centered varying effects from a normal distribution prior, controlled by shared scale parameters from a half-normal distribution. 5, $a _ { i }$ and $\beta _ { j }$ were assigned normal priors. For $\phi ,$ we used a log-normal prior.

The model was built using Monte-Carlo integration using the pymc library.<sup>1</sup> Posterior sampling was performed using a No-U-Turn-Sampler with four independent chains. Each chain comprised 2000 tuning iterations and 2000 retained posterior draws. Sampling used diagonal mass-matrix adaptation and a target acceptance probability of 0.9.

From the obtained posterior samples, we calculate relative optimizer effects on the logit scale. For optimizer j, the relative effect compared to the reference optimizer is

$$
\theta _ { j } = \beta _ { j } \ .
$$

In addition, we compute library-specific effects as

$$
\theta _ { i j } = \beta _ { j } + \gamma _ { i j }
$$

Both values are independent of library and seed effects, and we report statistics of $\theta _ { j }$ and $\theta _ { i j }$ over 8,000 independent posterior samples.

From these logit-scale effects, effect ratios were calculated as

$$
\Theta = e ^ { \theta } \ .
$$

An effect ration of 1 indicates no difference from the reference optimizer, values above indicate larger expected AUC values compared to the optimizer, i.e., improved performance, whereas values below 1 indicate lower expected AUC values.

## 1.2 Heuristic Molecular Representations

The following three heuristic molecular representations were evaluated in our benchmark experiments. Morgan Fingerprints<sup>2</sup> were used as bit vectors with a radius of 2 and a vector size of 2048. Fingerprints were generated using the RDKit (version 2026.3.1).<sup>3</sup>

Mordred Descriptors were generated with the Mordred library.<sup>4</sup> Only descriptors derived from the twodimensional structure were included. Descriptors with zero variance across the full virtual library were removed.

RDKit Descriptors were generated by calculating all descriptors implemented in RDKit.Chem.Descriptors (version 2026.3.1).<sup>3</sup>

## 1.3 Molecular Language Models

We evaluated four molecular language models pre-trained on large corpora of SMILES strings. For each model, we used the indicated Hugging Face checkpoint, and obtained molecular embeddings from the model encoder.

ChemBERTa (checkpoint: seyonec/ChemBERTa-zinc-base-v1) is a RoBERTa-style encoder-only transformer model, using 5 layers and 12 attention heads. ChemBERTa was pre-trained with masked language modeling (15% masked input tokens) pre-trained on 100,000 SMILES strings from the ZINC library. Molecular embeddings were obtained from canonicalized SMILES by pooling encoder outputs.

MolFormer<sup>6</sup> (checkpoint: ibm/MoLFormer-XL-both-10pct) is an encoder-only transformer that uses linear attention and rotary positional embeddings. MolFormer was pre-trained with masked language modeling on about 1.1B unlabeled molecules from the PubChem and ZINC databases. Molecular embeddings were obtained from canonicalized SMILES by pooling encoder outputs.

T5Chem<sup>7</sup> (checkpoint: GT4SD/multitask-text-and-chemistry-t5-base-augm) is a T5-base encoder– decoder transformer trained as a multi-task model for chemistry and natural language. It was trained with text-to-text prompting on five tasks: forward reaction prediction, retrosynthesis, molecular captioning, text-conditional molecule generation, and paragraph-to-action extraction. The model was trained on Pistachio-derived reaction pairs, paragraph-to-action procedure data, and ChEBI-20 molecule description pairs (33.5M training samples across tasks). Molecular embeddings were obtained from canonicalized SMILES by pooling encoder outputs.

SmiTed<sup>8</sup> (checkpoint: bisectgroup/materials-smi-ted-fork) is an encoder-decoder model with a BERTlike token encoder and an autoencoder model for SMILES reconstruction. Pre-training combined masked token prediction for the encoder with a reconstruction loss for SMILES reconstruction from the autoencoder. The model was pre-trained on 91M curated SMILES from the PubChem library. Molecular embeddings were obtained from canonicalized SMILES, using the latent space of the autoencoder.

For domain adaptation of these models to the individual virtual libraries, we added a projection head, implemented as a fully connected neural network (one hidden layer of 1024 neurons), on the molecular embeddings. The model was then fine-tuned by training this projection head and a low-rank adaptation (LoRA) module applied to the weights of the original language model. LoRA was used as implemented in the peft-finetuning library.<sup>9</sup> Unless otherwise noted, finetuning was performed using the Adam optimizer with a learning rate of $5 . 0 \cdot 1 0 ^ { - 4 }$ for 25 epochs. LoRA was applied to the transformer key and query modules, with LoRA parameters of r = 8 and α = 16. Fine-tuned embeddings were obtained by pooling embeddings from the fine-tuned encoder.

Here, we considered two different fine-tuning tasks:

\- Language model finetuning was tested on the MolFormer encoder, by pre-training with a BERT-style masked-token prediction loss (15%).

Contrastive finetuning used a weighted sum of a language modeling loss and a contrastive loss. The language modeling task followed the objective used during the original model training, using the original language modeling head. For contrastive learning, positive and negative pairs were generated through a fingerprint-similarity-based strategy. For each example, a positive pair was sampled from the library as a molecule with a Tanimoto similarity of > 0.7. In case no positive example could be sampled, a randomized version of the respective SMILES string was sampled as a fallback. Negative pairs were generated by sampling a molecule with a Tanimoto similarity $\mathrm { o f } < 0 . 1$ . We then computed a batch-based triplet contrastive loss, as implemented by Henderson et al.<sup>10</sup> Here, we computed the loss as a weighted sum between the original language modeling loss (0.99) and the contrastive learning loss (0.01).

\- Supervised finetuning was performed by reconstructing molecular fingerprints from the projection head outputs. Morgan fingerprints with radius 2 and length 1024 were used as targets. The model was trained using a mean squared error (MSE) loss for quantifying reproduction quality.

## 1.4 Surrogate Models

Surrogate models were trained on target values that were standardized to zero mean and unit variance. Gaussian Process models were implemented using GPyTorch.<sup>11</sup> All Gaussian Process models used a Matern 5/2 Kernel with automatic relevance detection, except for models trained on molecular fingerprints, which used a Tanimoto Kernel from the gauche library.<sup>12</sup> Following the recommendations by Hvarfner et al.,<sup>13</sup> Kernel lengthscale was initialized as ${ \sqrt { d } } ,$ where d is the dimensionality of the molecular representation. Models were trained by minimizing the negative marginal log-likelihood using the Adam optimizer with a learning rate of 0.05.

Feedforward Neural Network Models with Laplace Approximation: Feedforward neural networks were implemented in Pytorch. The networks were trained for up to 1,500 epochs using the Adam optimizer with a batch size of 128, a learning rate of $1 . 0 \cdot 1 0 ^ { - 3 }$ , a weight decay of $1 . 0 \cdot 1 0 ^ { - 4 }$ , and a dropout rate of 0.1. Early stopping on a held-out validation dataset was used with a patience of 50 epochs. Predictive posterior distributions were obtained by fitting a last-layer Laplace approximation after model training, using the Laplace-Torch library. 14

Random Forest models were implemented using scikit-learn<sup>15</sup> with 200 estimators and otherwise default hyperparameters. Predictive variances were estimated from the variance of predictions across trees.

## 1.5 Acquisition Functions and Batch Acquisition Policies

After surrogate model training, predictive posteriors were obtained by computing the predictive mean $\mu ( x )$ and variance $\sigma ^ { 2 } ( x )$ for each unevaluated molecule in the virtual library. These values were used to calculate acquisition function values for each molecule.

Optimization was treated as a maximization problem for all virtual libraries. In this context, we used the following two acquisition functions.

The upper confidence bound (UCB) acquisition function is defined as

$$
U C B ( x ) = \mu ( x ) + \beta \cdot \sigma ( x )
$$

where the parameter $\beta$ controls the exploration–exploitation trade-off.

The log expected improvement (LogEI) acquisition function is defined as

$$
\begin{array} { r } { \log E I ( x ) = \log \bigl ( z \cdot \phi ( z ) + \sigma ( x ) \cdot \varphi ( z ) \bigr ) } \end{array}
$$

where $z = \frac { \mu ( x ) - f ( x ^ { * } ) - \beta } { \sigma } .$ . $\varphi$ and Φ are the probability density function and the cumulative density function of the normal distribution, respectively. Again, the parameter $\beta$ controls the exploration– exploitation trade-off.

With these acquisition function values for each molecule, the following policies for batch acquisition were used.

Top-K Acquisition selects a batch of candidates by sorting all candidates by their acquisition function value, and selecting the K candidates with the largest values.<sup>16</sup>

Ensemble Acquisition generates an ensemble of K acquisition functions using K different values of the exploration parameter $\beta ,$ equally spaced between 0 and $\beta _ { \mathrm { m a x } }$ . For each value of $\beta _ { : }$ the candidate that maximizes the corresponding acquisition function is selected for the next batch.<sup>17</sup> After acquisition of a candidate molecule, the selected molecule was removed from the pool of unevaluated molecules before proceeding to the next acquisition function.

## 2 Molecular Library Distributions

In this work, we evaluated a total of six virtual libraries from the literature that contained ground-truth target property labels for all candidates. For all molecular libraries, we performed a systematic analysis of the underlying molecule distribution using Morgan Fingerprints (radius 2, length 2048, details see above) and all transformer encoders described in section 1.3. Specifically, we quantify the dispersion of within-library similarities based on a sample of 5M random pairs, using Tanimoto similarities for molecular fingerprints and cosine similarities for transformer embeddings.

To investigate the relation to established libraries of drug-like molecules, we sampled a reference set of 25,000 molecules from the ZINC15 library. Similarity of the target virtual library to this reference library was then investigated in three complementary ways:

A Uniform Manifold Approximation (UMAP) model<sup>18</sup> was fitted on the molecular fingerprints (Morgan Fingerprints, radius 2, length 2048, details see above) for all molecules in the reference library. The molecular fingerprints for all molecules in the library of interest were then projected into the same UMAP embedding space. All UMAP models were built using the umap package.<sup>19</sup>

A T-Distributed Stochastic Neighbor Embedding (t-SNE) model<sup>20</sup> was fitted on the molecular fingerprints (Morgan fingerprints, radius 2, length 2048, details see above) for all molecules in the reference library. The molecular fingerprints for all molecules in the library of interest were then projected into the same t-SNE embedding space. All t-SNE models were built using the openTSNE library.<sup>21</sup>

\- We sampled 5M random pairs of molecules from the target library and the reference library, respectively, and quantified molecular similarity as described above.

For all libraries, the histograms of within-library and to-reference similarity, the dimensionality-reduced projections, and the histogram of target properties, are shown in Figures S1–S24. Characteristics of the respective distributions are given in Tables S1–S6.

## 2.1 1.9M Molecules from the Zinc Library Docked against AmpC (Zinc-AmpC)

From the original dataset by Shoichet and co-workers, who docked a sample from the Zinc15 virtual library (\~96M molecules) against AmpC β-lactamase,<sup>22</sup> we selected a random subsample of 1,914,566 molecules, which corresponds to approx. 2% of the full library.

![](images/53e856bc82c21b7e58e4213677cbbdadcff1efec6cf56e3099a3b673aeacfa82.jpg)  
Figure S 1: Histogram of target properties (docking scores) for the Zinc-AmpC library.

Table S 1: Descriptive statistics of the similarity distributions of the Zinc-AmpC library for different molecular embeddings
<table><tr><td colspan="4">Within Library Similarities</td><td colspan="3">Similarities To Reference</td></tr><tr><td></td><td>Mean</td><td>SD Median</td><td>IQR</td><td>Mean</td><td>SD Median</td><td>IQR</td></tr><tr><td>Morgan Fingerprint</td><td>0.867</td><td>0.042 0.870</td><td>0.054</td><td>0.869</td><td>0.040 0.871</td><td>0.052</td></tr><tr><td>ChemBERTa</td><td>0.275</td><td>0.098 0.263</td><td>0.130</td><td>0.274</td><td>0.100 0.260</td><td>0.133</td></tr><tr><td>MolFormer</td><td>0.368</td><td>0.051 0.367</td><td>0.068</td><td>0.393</td><td>0.056 0.392</td><td>0.074</td></tr><tr><td>SmiTed</td><td>0.006</td><td>0.002 0.006</td><td>0.002</td><td>0.042</td><td>0.007 0.041</td><td>0.009</td></tr><tr><td>T5Chem</td><td>0.333</td><td>0.095</td><td>0.325 0.129</td><td>0.338</td><td>0.099 0.328</td><td>0.135</td></tr></table>

![](images/2573f000caf22e8646c293a9fbdb49ac34fd814816440141bda95b4693736abc.jpg)

![](images/6066077310275559df399fc1b103fa52669078b97e394200985ba5133befa1a5.jpg)

![](images/765cabde1bbbe5592bfc8e32fcde2f160edca1eecd5270ab1412182c5706ae85.jpg)

![](images/c01a8d68e90b84eb4b75b95c4524e8aedfcf5c5478b1d954ea4da4c7b373df97.jpg)

![](images/869c0f2f352209240fcaa0fef2a6dd1fc38009a121691ca3d1cdf90105389d43.jpg)

![](images/f40f438dea501a50f087419c73b0091afc521d63ce594ea603fd2c5f1b052161.jpg)

![](images/93b26bcaf9dec20e1f6a39e62659588fa6586a3552a02956c960e948b1a7fdd2.jpg)

![](images/24fa0ea4aee8197df4682bddaf5273c47dc235778651462074cddb958dd84e83.jpg)

![](images/55440c75c9dfc9dca7369b7092416774aea52f8543b062a5699d449be93e7dc0.jpg)

![](images/f5420a60b808886d97a9adb2292466f08937415535232507a1c1321c2e8ee8cc.jpg)  
Figure S 2: Histograms of embedding similarities for the Zinc-AmpC library. Left column: within-library similarities. Right column: to-reference similarities.

![](images/89511de5f2b9697d25f46eb6bc6c4ecd8e22a3028212c19df09378f8d5c82c08.jpg)  
Figure S 3: Projection of molecular fingerprints of all candidate molecules from the Zinc-AmpC library into the UMAP coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

![](images/1fb146032d431cf19ae263ccdbc0d28b33eaec1178511ef51a9e7d61c45bef5a.jpg)  
Figure S 4: Projection of molecular fingerprints of all candidate molecules from the Zinc-AmpC library into the t-SNE coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

## 2.2 1.5M Molecules from the RO4 Library Docked against OGG1 (RO4-OGG1)

From the original dataset by Luttens et al.,<sup>23</sup> who docked a sample from the Zinc15 virtual library that was filtered by “Rule-of-Four” criteria (\~14,7M molecules) against oxoguanine glycosylase (OGG1), we selected a random subset of 1,470,000 molecules, which corresponds to approx. 10% of the full library.

![](images/a4eaf9e6164edb5455292458cad6c1d95bef9b36664f8f3d499e908433f818f6.jpg)  
Figure S 5: Histogram of target properties (docking scores) for the RO4-OGG1 library.

Table S 2: Descriptive statistics of the similarity distributions of the RO4-OGG1 library for different molecular embeddings
<table><tr><td colspan="3">Within Library Similarities</td><td colspan="3">Similarities To Reference</td></tr><tr><td></td><td>Mean SD</td><td>Median IQR</td><td>Mean SD</td><td>Median</td><td>IQR</td></tr><tr><td>Morgan Fingerprint</td><td>0.853 0.039</td><td>0.856 0.049</td><td>0.862</td><td>0.039 0.864</td><td>0.049</td></tr><tr><td>ChemBERTa</td><td>0.228 0.078</td><td>0.217 0.100</td><td>0.270</td><td>0.090 0.258</td><td>0.117</td></tr><tr><td>MolFormer</td><td>0.339 0.071</td><td>0.335</td><td>0.095 0.402</td><td>0.065 0.401</td><td>0.085</td></tr><tr><td>SmiTed</td><td>0.005 0.001</td><td>0.005</td><td>0.002 0.042</td><td>0.007 0.042</td><td>0.009</td></tr><tr><td>T5Chem</td><td>0.263 0.082</td><td>0.253</td><td>0.109</td><td>0.324 0.094</td><td>0.315 0.127</td></tr></table>

![](images/2e09372ce7152b8f38e26dca4d1e055b97f2d8c18795a715f49faba2bf3aa9ab.jpg)

![](images/d5440bb6586728b1c150b3dfce7f548147a9ed27766ed354865288b7b71f6d21.jpg)

![](images/d6e1ceece9429ca6ad00e7fea05f5719c6020904c1dfd77e8c2ddf0575c1d172.jpg)

![](images/3fa2313e6cbc055779445b297344adf6ab08c6a56095057a2cf4c20d034f0552.jpg)

![](images/6b4fdd909a5cf762672a98bdbb62c9716397a299a1ed0f05dfb3d34ca5974903.jpg)

![](images/586bbf7e58e6186bb2da163140c83222bc180947a943599b5edc279b02ed85b1.jpg)

![](images/10509d0c679c3407b617a59818f1411fa6d27b0367cd68cce5f50ab41d41c988.jpg)

![](images/d4633b5f0c0ebfcd3682de5f3d1efc8f0c464cda5ad8d87e329a1dba7b1534a6.jpg)

![](images/158d0e7c81e21dd031e4e2fc1c246c33177cd20e183dc91b5f9812932fde8a2d.jpg)

![](images/2d7046f876da318e0e4d2cb6f1b0b043ede44b503c1a0b78c2626cefe68796ba.jpg)  
Figure S 6: Histograms of embedding similarities for the RO4-OGG1 library. Left column: within-library similarities. Right column: to-reference similarities.

![](images/63c4b2ce24ca750d5caa55ccebdbd455d9715e6a4480033a779bf5598fcceea6.jpg)  
Figure S 7: Projection of molecular fingerprints of all candidate molecules from the RO4-OGG1 library into the UMAP coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

![](images/58ff3eb8b4fbfb516925310a0e1dda8e12fe05edf3184d384588d5029cdf2990.jpg)  
Figure S 8: Projection of molecular fingerprints of all candidate molecules from the RO4-OGG1 library into the t-SNE coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

## 2.3 2.1M Molecules from the Enamine-HTS Collection Docked against TMPK (Enamine-TMPK)

The Enamine-HTS library, containing a total of 2,104,318 molecules, docked against a thymidylate kinase (TMPK) by Coley and co-workers.<sup>16</sup>

![](images/5a6fe49bf65608baba934725624c3dc3213243b9d1112f07416cbffa4ec7c06f.jpg)  
Figure S 9: Histogram of target properties (docking scores) for the Enamine-TMPK library.

Table S 3: Descriptive statistics of the similarity distributions of the Enamine-TMPK library for different molecular embeddings
<table><tr><td colspan="4">Within Library Similarities</td></tr><tr><td>Mean</td><td>SD Median</td><td>IQR Mean</td><td>SD Median IQR</td></tr><tr><td>Morgan Fingerprint 0.867</td><td>0.043 0.871</td><td>0.053 0.878</td><td>0.039 0.880 0.051</td></tr><tr><td>ChemBERTa 0.315</td><td>0.106 0.301</td><td>0.138 0.334</td><td>0.100 0.323 0.132</td></tr><tr><td>MolFormer 0.333</td><td>0.063 0.332</td><td>0.084 0.418</td><td>0.070 0.415 0.092</td></tr><tr><td>SmiTed 0.006</td><td>0.002 0.006</td><td>0.002 0.042</td><td>0.008 0.041 0.010</td></tr><tr><td>T5Chem 0.309</td><td>0.103 0.296</td><td>0.132</td><td>0.383 0.106 0.373 0.144</td></tr></table>

![](images/db4a357c2757936b5f5c395ef8d4bfd2f4f813addc093fca30e97609420427e8.jpg)

![](images/f867acebab78dd50b8446efd4c2778d239c894d08f17f55b25b8af0899282e7e.jpg)

![](images/0d334751b0dd263afb95f6d504d2f39f4107154e2fbd61fcae11e04adf769cf5.jpg)

![](images/ef79e6b690d2b8b4e90166d5e5b64e99ec442b5b155d004a5d918fdb3fe7b279.jpg)

![](images/4255064a4185564f55b8dba59376d4b0bf68522463f89bfb28e28c9011d923f2.jpg)

![](images/1fa9754c7a203b1c978ca3bdc8dce1ad99acaebc8b7c25bff8b35ab8f7fe6141.jpg)

![](images/01aa87a10af75cf12b2105d9a91924631d41f31f783f3c992a4469856b7662f9.jpg)

![](images/c4f0f9a70bb6b37d383b647347d46d1dccd4152ba051fbe50c87a651e4eea5a4.jpg)

![](images/7d70fb22acd897c7176f61c4ffde5a7d8260d8ed2f504c4b2832e3327885bcf9.jpg)

![](images/58f67d2cec4b886257161fd6542572736fae1489c16088427a47dcd1fbe55357.jpg)  
Figure S 10: Histograms of embedding similarities for the Enamine-TMPK library. Left column: within-library similarities,. Right column: to-reference similarities.

![](images/83e3fb9e98a34d1bbffe0703a4157319c8a9fc3ff9f7f93fe40b824b21152b02.jpg)  
Figure S 11: Projection of molecular fingerprints of all candidate molecules from the Enamine-TMPK library into the UMAP coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

![](images/490e4e90f213ff92e229bb0b737b365325943c47dd59166b7d5af829bc9c1b9c.jpg)  
Figure S 12: Projection of molecular fingerprints of all candidate molecules from the Enamine-TMPK library into the t-SNE coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

## 2.4 2.3M Molecules from the Harvard Clean Energy Project (HCEP-PCE)

The dataset of 2,320,648 molecules from the Harvard Clean Energy Project,<sup>24</sup> along with DFT-computed power conversion efficiencies (PCE) for use in organic solar cells.

![](images/a2c95332127d74dc2fd99c474f0407757f22293a97b0a0d82bf013e302290e6c.jpg)  
Figure S 13: Histogram of target properties (docking scores) for the HCEP-PCE library.

Table S 4: Descriptive statistics of the similarity distributions of HCEP-PCE library for different molecular embeddings
<table><tr><td colspan="3">Within Library Similarities</td><td colspan="3">Similarities To Reference</td></tr><tr><td></td><td>Mean SD</td><td>Median IQR</td><td>Mean</td><td>SD Median</td><td>IQR</td></tr><tr><td>Morgan Fingerprint</td><td>0.837 0.064</td><td>0.848 0.081</td><td>0.946</td><td>0.027 0.947</td><td>0.036</td></tr><tr><td>ChemBERTa</td><td>0.232 0.096</td><td>0.220 0.131</td><td>0.566</td><td>0.109</td><td>0.570 0.150</td></tr><tr><td>MolFormer</td><td>0.136 0.041</td><td>0.132 0.055</td><td>0.509</td><td>0.090</td><td>0.517 0.130</td></tr><tr><td>SmiTed</td><td>0.008 0.003</td><td>0.008 0.004</td><td>0.036</td><td>0.006</td><td>0.035 0.008</td></tr><tr><td>T5Chem</td><td>0.139 0.060</td><td>0.129</td><td>0.073 0.589</td><td>0.115</td><td>0.590 0.157</td></tr></table>

Morgan Fingerprint  
![](images/1230e5abd170daa5456bad0f141d0b7a46eb8777ca6498c1b4f3c575d5e22e5b.jpg)

![](images/6e8728e4b5b914c76c29fdd05050edcc2fde9430886805984fb1060dec9711db.jpg)

![](images/7834f3f2b47d5c213afc6fe448966320036edecefb882fca094a69063e52b462.jpg)

![](images/112ac2dd060f4aafaa90f021cbb4bb32755647bdd54d0c322c520205bf6c43ed.jpg)

![](images/fb4e60523fd658eddb58db0a30790ab581da4007067739e860faaa9d439fd47e.jpg)

![](images/7d4ab954b5b9a20cdb42ba0cf160db3c4a9efc4ae0d8f438d6debaf3faef35f2.jpg)

![](images/0c84e74ff26f032c1137dd06e822c8b7376751d7249a5ec4bd2f14bfa64ac9a2.jpg)

![](images/a9451304458927a425f55ea344e2c95c23ffc6a98ec6c9250e28078b096db808.jpg)

![](images/d02a5660f32670c343b35ee4e6ca836a32b4e7510670c73aeaa55d571aa92d20.jpg)

![](images/a8fbffaf04a5fbc3d3c75b4b3acd710336bbd1bff022aa61442eeb72542907b7.jpg)  
Figure S 14: Histograms of embedding similarities for the HCEP-PCE library. Left column: within-library similarities. Right column: to-reference similarities.

![](images/87c6f2e78c7bb51149404fd5fb08c60b120884674362ba4c1fe8eb6452a5e47b.jpg)  
Figure S 15: Projection of molecular fingerprints of all candidate molecules from the HCEP-PCE library into the UMAP coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

![](images/266a9ae7212df5dd6f959f490b2de65d18ae881d025060664c41f0914d0e612e.jpg)  
Figure S 16: Projection of molecular fingerprints of all candidate molecules from the HCEP-PCE library into the t-SNE coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

## 2.5 180k Organic Laser Candidates with Computed Oscillator Strengths (Laser-OSC)

The library of 182,888 conjugated organic molecules for use as thin-film organic laser emitters,<sup>25</sup> along with DFT-computed emission oscillator strengths (OSC).

![](images/21f4b8738d9f31b98d250de27d46cb189cd66f1a3a988a8620749e6fe006c77a.jpg)  
Figure S 17: Histogram of target properties (docking scores) for the Laser-OSC library.

Table S 5: Descriptive statistics of the similarity distributions of the Laser-OSC library for different molecular embeddings
<table><tr><td colspan="4">Within Library Similarities</td><td colspan="3">Similarities To Reference</td></tr><tr><td></td><td>Mean SD</td><td>Median</td><td>IQR</td><td>Mean</td><td>SD Median</td><td>IQR</td></tr><tr><td>Morgan Fingerprint</td><td>0.803 0.083</td><td>0.825</td><td>0.093</td><td>0.923</td><td>0.034 0.926</td><td>0.046</td></tr><tr><td>ChemBERTa</td><td>0.151 0.067</td><td>0.140</td><td>0.084</td><td>0.493</td><td>0.095 0.492</td><td>0.129</td></tr><tr><td>MolFormer</td><td>0.181 0.057</td><td>0.176</td><td>0.074</td><td>0.482</td><td>0.066 0.486</td><td>0.089</td></tr><tr><td>SmiTed</td><td>0.036 0.053</td><td>0.021</td><td>0.021</td><td>0.041</td><td>0.038 0.033</td><td>0.018</td></tr><tr><td>T5Chem</td><td>0.109 0.047</td><td>0.104</td><td>0.063</td><td>0.511</td><td>0.104 0.508</td><td>0.144</td></tr></table>

![](images/f068e5299d47b69c78848c1a107f011009b145035714173f00789392b3bfb601.jpg)

![](images/5c773cc25793e77bf21c9f4fec383d4456d61ee7c3d70d1719ee85b566c8bc37.jpg)

![](images/7b81c937b9de8b4894d775e621d6f08340e12328d756147410110cd7b8e5b7bc.jpg)

![](images/69053de8d3fe0e809c6dd2db1682a2f21ee27eb404dae496163bcdef0aab461c.jpg)

![](images/8e715e93e885b4e2cffa8ad085228131c9dfff46c4cc874a94d2a42c9b88c272.jpg)

![](images/3b85e21d5dbef4d42c43f53887061de58eed3f7fad3dd1c88342a6d11070a093.jpg)

![](images/37295cb4605cea86597629c4587b7fa588d269a6c9d3db40869e8a820fe56952.jpg)

![](images/2000f513ee9c7832933756e63fca4708f33cd831654c066fb4c028780214f2a6.jpg)

![](images/d8dca7c55ac44806145e6c9b29aabb34586eccfd1421413547c87d5bcbe8a8e9.jpg)

![](images/d7f8b7d81bf0b7d0a062b73905272550adb25224cbef9888e2c79b35739c2e83.jpg)  
Figure S 18: Histograms of embedding similarities for the Laser-OSC library. Left column: within-library similarities. Right column: to-reference similarities.

![](images/daeb1c03be27dc1a4f25b832ea2ccce2930dd6237d83efb2a2cf6d88a7ff2b0a.jpg)  
Figure S 19: Projection of molecular fingerprints of all candidate molecules from the Laser-OSC library into the UMAP coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

![](images/2ebcf84abc8de3d4519273febf8f43211af7643784f80c40811501aad061ae85.jpg)  
Figure S 20: Projection of molecular fingerprints of all candidate molecules from the Laser-OSC library into the t-SNE coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

## 2.6 300k Phosphine Ligands with Predicted Enantioselectivities (Kraken-ES)

The database of 300,000 phosphine ligands<sup>26</sup> with predicted enantioselectivity values $( \Delta \Delta \mathbf { G } ^ { \ddagger } )$ for an enantioselective C(sp3)–C(sp2) cross coupling reactions. The $\Delta \Delta \mathrm { G } ^ { \ddagger }$ values were obtained by fitting a random forest model on the experimental data reported by Biscoe and co-workers,<sup>27</sup> using the MLextrapolated phosphine descriptors from the Kraken database. The model was then applied to all posphines in the Kraken library to obtain $\Delta \Delta \mathrm { G } ^ { \ddagger }$ values which were used as optimization targets.

![](images/e46e7269257fde63f8699ba41aa88e419e4fa4113f98438cf6b645dbd92eacc8.jpg)  
Figure S 21: Histogram of target properties (docking scores) for the Kraken-ES library.

Table S 6: Descriptive statistics of the similarity distributions of the Kraken-ES library for different molecular embeddings
<table><tr><td colspan="4">Within Library Similarities</td><td colspan="4">Similarities To Reference</td></tr><tr><td></td><td>Mean SD</td><td>Median</td><td>IQR</td><td>Mean</td><td>SD</td><td>Median</td><td>IQR</td></tr><tr><td>Morgan Fingerprint</td><td>0.873 0.058</td><td>0.879</td><td>0.064</td><td>0.925</td><td>0.035</td><td>0.928</td><td>0.046</td></tr><tr><td>ChemBERTa</td><td>0.358 0.138</td><td>0.345</td><td>0.187</td><td>0.424</td><td>0.114</td><td>0.417</td><td>0.156</td></tr><tr><td>MolFormer</td><td>0.254 0.084</td><td>0.246</td><td>0.112</td><td>0.494</td><td>0.084</td><td>0.500</td><td>0.116</td></tr><tr><td>SmiTed</td><td>0.019</td><td>0.010 0.018</td><td>0.012</td><td>0.020</td><td>0.008</td><td>0.018</td><td>0.009</td></tr><tr><td>T5Chem</td><td>0.352</td><td>0.149</td><td>0.334</td><td>0.211</td><td>0.592 0.111</td><td>0.593</td><td>0.155</td></tr></table>

Morgan Fingerprint  
![](images/716d8b6e3475a643420ecb90e1a0d6506716920e84670b4f8f3494e751fe3de9.jpg)

![](images/12da023fbd61b93c431f529a1220582d23fe043d393d7abaf9f7cf1e14b9b4c4.jpg)

![](images/43c2ec52d96fa2e1e9cdfab602dec860cdc7ee4b38bbb4ecff9d4c1eee96080e.jpg)

![](images/9c3e0e80168438b86d45b0fa315beead3920b9c35aa7443c954fffa5ca925829.jpg)

![](images/a773157a9853341abc0c85813b0bebd3f4ea013dcf907c49946f3162593e3f55.jpg)

![](images/b939bac1606ac04a5e9ba471d954d4689f29d520551ed6944b6057d7cf412008.jpg)

![](images/4bc53a50d26d1f516b9166df5039341157291b60e61ab06aeb464ec98cad35b1.jpg)

![](images/aaa02dd9e74e99996b59c2f96a355cc820f4edbc1be6e3c847763f48cf14225c.jpg)

![](images/2a306981c2243bbaebfe507c4536c20cc03ad8bd02e319979d13e3100a4e57de.jpg)

![](images/9ed966f12cfabf3355eb695414f278beb8947ec06e1b124c71c1c6f546057c8c.jpg)  
Figure S 22: Histograms of embedding similarities for the Kraken-ES library. Left column: within-library similarities. Right column: to-reference similarities.

![](images/f09fde53a8a2aa01399a989cc490ccaa8f2f35ca60845fecc6a97cbb83948dd7.jpg)  
Figure S 23: Projection of molecular fingerprints of all candidate molecules from the Kraken-ES library into the UMAP coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

![](images/ecae8c9aa3e03ce92713e4dd852c6d8be88aa2495584b19b244e7516dd98e4c5.jpg)  
Figure S 24: Projection of molecular fingerprints of all candidate molecules from the Kraken-ES library into the t-SNE coordinates trained on the fingerprints from the reference sample of 25k molecules from the ZINC15 library.

## 3 Benchmarking Surrogate Models and Acquisition Policies

## 3.1 Surrogate Model Benchmarks

![](images/9125961482560b79e90d9a0e3e14fd5b15a359aefd0fcc1fde102b82d0943cef.jpg)  
Figure S 25: Optimization trajectories for different surrogate models on MolFormer embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/d81418682deaf240ae21d6486cd318a345aadc0e5555e4e428dd494f203665ad.jpg)  
Figure S 26: Logit-scale effect of the surrogate model relative to a Laplace Neural network model, differentiated by molecular libraries. Experiments were performed using MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples

![](images/b48b6580f3f1f1af9b873a4d6b8f3779ee9eebe6eb14351fd744332315e228a3.jpg)  
Figure S 27: Effect ratios of the surrogate model relative to a Laplace Neural network model, averaged over all libraries. Experiments were performed using MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/54637eee9d24d40be32e1acad54c9c6c12688de0ecce90f91e50d5735514fcb6.jpg)  
Figure S 28: Optimization trajectories for different surrogate models on MolFormer embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/e58562c5f3e49932c2d8beea34335db0e4896d3b70326591421c21218c21100e.jpg)  
Figure S 29: Logit-scale effect of the surrogate model relative to a Laplace Neural network model, differentiated by molecular libraries. Experiments were performed using MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/f9b50216ea6b5ea4e8f7889f57d479d156d88c63adddc99cc5903e23750ae05d.jpg)  
Figure S 30: Effect ratios of the surrogate model relative to a Laplace Neural network model, averaged over all libraries. Experiments were performed using MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/81730acb10e8f34e71b66d2e52d1483234d718c85fd3ea5518e227a43cd31f4e.jpg)  
Figure S 31: Optimization trajectories for different surrogate models on MolFormer embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/91f6f02672ae0cffee22e16199102232d0ef57cdc9a53de23dcb0c1ac28c740f.jpg)  
Figure S 32: Logit-scale effect of the surrogate model relative to a Laplace Neural network model, differentiated by molecular libraries. Experiments were performed using MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/4b36d4bbeaf08d4f48eebd4b719cfda1f506657e3a40f448a53c38f5d0a4f144.jpg)  
Figure S 33: Effect ratios of the surrogate model relative to a Laplace Neural network model, averaged over all libraries. Experiments were performed using MolFormer embeddings in the in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals..

![](images/cb561978b70e6616be00e058956fde759c18a843e307b1ade24fa02508d7d8ab.jpg)  
Figure S 34: Optimization trajectories for different surrogate models on MolFormer embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/5ae20a50ae0d780c97c10488d79c6547d8883ff7ed99508924602627fd2ad75d.jpg)  
Figure S 35: Logit-scale effect of the surrogate model relative to a Laplace Neural network model, differentiated by molecular libraries. Experiments were performed using MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/59144f5b29adb37fe53a4bb4ea3aaf1807cc2667edbd1b69258b8d7a508076a1.jpg)  
Figure S 36: Effect ratios of the surrogate model relative to a Laplace Neural network model, averaged over all libraries. Experiments were performed using MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

## 3.2 Acquisition Strategy Benchmarks

![](images/1a7fcac12356164d6f40a38b20da2eb7119ed4e602c2bc5ca52851e6845337c6.jpg)  
Figure S 37: Optimization trajectories for different acquisition strategies, using a Laplace Neural Network surrogate model on MolFormer embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/014c521c8aac8c4357dc7f323ab44a976c20a725be657bbb6bce90a5d0fba896.jpg)  
Figure S 38: Logit-scale effect of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/08a21918a4aa341a693f4e512d2ccd299c3d24b3357bf9cb0a8208f9e8fb10b6.jpg)  
Figure S 39: Effect ratios of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/bff413484f329f5e5934f8e50f8a4997bd1e88112d6fe7c052ed5ed79665e484.jpg)  
Figure S 40: Optimization trajectories for different acquisition strategies, using a Laplace Neural Network surrogate model on MolFormer embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/216e194c0f53450a78b56ff3daf7a1a344caf312a6855cafcdf03bc5da0336e2.jpg)  
Figure S 41: Logit-scale effect of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/4456ccd2fa3eea01d62579563e5d40caa973ef56236b4ced9a7a18041ad0c595.jpg)  
Figure S 42: Effect ratios of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/7f9f321eba8be2b7e3d9ce4bf3b80f7bb6b68de03806daa2d71df0a80e8d77f5.jpg)  
Figure S 43: Optimization trajectories for different acquisition strategies, using a Laplace Neural Network surrogate model on MolFormer embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/5a3afb5ccb5301d135a7b3cf67f1f23246a7db8290a35f15879113341c553ed6.jpg)  
Figure S 44: Logit-scale effect of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/9b26e55e281ebfe583177394b4cd26090d7295cb95684d79e1624262b692a45a.jpg)  
Figure S 45: Effect ratios of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the experimental discovery setting $( 1 , 0 0 0$ experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/1ba49cc0304700ca9d28907a94aa32fe92a3fcbed74c86dfd4528ca086c988fe.jpg)  
Figure S 46: Optimization trajectories for different acquisition strategies, using a Laplace Neural Network surrogate model on MolFormer embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/0a099c1ff47da017fba16cd6bd05e647bd178bccfd52af397f8395746c6fb478.jpg)  
Figure S 47: Logit-scale effect of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/1c335ffac726c68be701a35f7d3e1c34f7251739dc92ada5518e2b5031231382.jpg)  
Figure S 48: Effect ratios of the acquisition strategy relative to Top-K acquisition with a UCB acquisition function, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the experimental discovery setting $( 1 , 0 0 0$ experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

## 4 Evaluating Molecular Representations

## 4.1 Heuristic Representations and Pre-Trained Transformer Embeddings

![](images/81c2d65cb1f3668bc6206fecea55b14c28d07be3f461114dd4e8faf8a499ce92.jpg)  
Figure S 49: Optimization trajectories for different domain-specific encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/2a7cd050ba674792cb13ca82fb8e7c0266d154fe9c9df906c55637208d332acb.jpg)  
Figure S 50: Optimization trajectories for different molecular language model encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean

![](images/d2637e14fdcbbcf9dcd271af96c753114bbab2ecd6998db8a58c808647940238.jpg)  
Figure S 51: Logit-scale effect of the molecular encoder relative to Morgan fingerprints, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/cb6a16e0cfb7da63b820ca228cc5c487cfe7334fded4a9bc6888252546799852.jpg)  
Figure S 52: Effect ratios of the molecular encoder relative to Morgan fingerprints, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/e381336ac927f3371ad4801fd1f2bc2a65df8a1503f2ecc6957ff2ff1b9c3b42.jpg)  
Figure S 53: Optimization trajectories for different domain-specific encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/04f55f4a51a7d972b11557ba73aab9919ca52f2f487089ab5ef102f1ed666837.jpg)  
Figure S 54: Optimization trajectories for different molecular language model encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/b50bd0233db78df1c15bd4a2431fd0bc3f06c31312541fd0d405ab5aa1605006.jpg)  
Figure S 55: Logit-scale effect of the molecular encoder relative to Morgan fingerprints, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/3d1acc404b5e2fda67a96cf1245390b0a436783a9215a522dd2d5139dc588e27.jpg)  
Figure S 56: Effect ratios of the molecular encoder relative to Morgan fingerprints, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/71c0115f769def6cad6420fcf01fe2021fe0bb33df3f99613ee88d31fd2e68d2.jpg)  
Figure S 57: Optimization trajectories for different domain-specific encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/98ba8a83da4dcfac27c297ff769eacfa00d14e0633391c172c39618c6c3040c2.jpg)  
Figure S 58: Optimization trajectories for different molecular language model encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/e79811a2fc045324ea141fa1f01925c144b406dc76f4378dd4e261af926e2308.jpg)  
Figure S 59: Logit-scale effect of the molecular encoder relative to Morgan fingerprints, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/125fb3adb2ef514d964dccead1ef6e86be892db206c5339c42a0e0ad6316d2d3.jpg)  
Figure S 60: Effect ratios of the molecular encoder relative to Morgan fingerprints, averaged over all libraries. Experiments were performed using a Laplace Neural network model in experimental discovery setting (1,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/02b1b571e5d8c458f8a345cf8cb726772dcf93f556f3e2387b0723fd9d1fd1b6.jpg)  
Figure S 61: Optimization trajectories for different domain-specific encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/f074f45f81ac016553e459ea20aaef7980592192297122e895fb3f9bf8accea4.jpg)  
Figure S 62: Optimization trajectories for different molecular language model encoders using a Laplace Neural Network surrogate model, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/4a1ff12762b86d76f17a55979608ab96a021909354b277c08f692f828ba9807b.jpg)  
Figure S 63: Logit-scale effect of the molecular encoder relative to Morgan fingerprints, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/898eb337349106f86ab76628a5c94f1d4a3652f1cda8f1ab6abe6577839f7d47.jpg)  
Figure S 64: Effect ratios of the molecular encoder relative to Morgan fingerprints, averaged over all libraries. Experiments were performed using a Laplace Neural network model in experimental discovery setting (1,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

## 4.2 Transformer Pooling Strategies

![](images/1659e70f7e6d781b2dbcbcaa53d58739adb9ab51aa31f2225cc387f151cf1253.jpg)  
Figure S 65: Optimization trajectories for different pooling strategies on a MolFormer encoder using a Laplace Neura Network surrogate model, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/3960e3a4bc9bc7e361511b4a96124ddb12c00098bb0f2a1fd38b36667f800a5a.jpg)

Figure S 66: Logit-scale effect of the transformer pooling strategy relative to mean pooling, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.  
![](images/88b1260a4eb559cf7b5147bb5fb6b2ea91ee078df2fa5827de987a56b8cd4beb.jpg)  
Figure S 67: Effect ratios of the transformer pooling strategy relative to mean pooling, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/760628557f4164398b21195c1c54762440efc92959f404a1be7137725a3af0c6.jpg)  
Figure S 68: Optimization trajectories for different pooling strategies on a MolFormer encoder using a Laplace Neural Network surrogate model, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/2d2af1cb2b5ac95cbc90e95a162c10a0c660ab58f90648b51cd1cab473645c99.jpg)  
Figure S 69: Logit-scale effect of the transformer pooling strategy relative to mean pooling, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/932c8597a29586f0f1232a266f510184e15a99883563347a481b986f26adfb1a.jpg)  
Figure S 70: Effect ratios of the transformer pooling strategy relative to mean pooling, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/b6359c7a370573c734601c4be5e682cd5daec4ca4b6a8af36e94e85c49f5754c.jpg)  
Figure S 71: Optimization trajectories for different pooling strategies on a MolFormer encoder using a Laplace Neural Network surrogate model, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/1684214ce9e4716920289af6aa9deaac1bd9fadcab49bad3a5236cba0b271b39.jpg)  
Figure S 72: Logit-scale effect of the transformer pooling strategy relative to mean pooling, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/7e86f354697afe1ecf60dd6afa831a84013963627771c42d2a16f39bb8cdcde1.jpg)  
Figure S 73: Effect ratios of the transformer pooling strategy relative to mean pooling, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/803194c9ddbf9bcd701efa07bfc477b06db8a65cc530d2d06c81764c3450fce7.jpg)  
Figure S 74: Optimization trajectories for different pooling strategies on a MolFormer encoder using a Laplace Neural Network surrogate model, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/b628bff59d4d4959f3520cde9ca49d127759f14e077e0a00fa75e24f4bd461a3.jpg)  
Figure S 75: Logit-scale effect of the transformer pooling strategy relative to mean pooling, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/b8a1002876cef959738883f3fe0648ce8d2cd4eba069829461814c8431ea2854.jpg)  
Figure S 76: Effect ratios of the transformer pooling strategy relative to mean pooling, averaged over all libraries. Experiments were performed using a Laplace Neural network model on MolFormer embeddings in experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

## 4.3 Molecular Transformer Fine-Tuning

## ChemBERTa

![](images/c8f6ed83a818c8d6dee351a8f26bb2ff9c17a924342681f1bf6efc783286c7ac.jpg)  
Figure S 77: Optimization trajectories after using different strategies to fine-tune ChemBERTa embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/d214ffd529cca6b9a86feae301a3e58f07e8045b0f3742f91c638117fbd2bf1c.jpg)  
Figure S 78: Logit-scale effect of the fine-tuning strategy on ChemBERTa embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/c0194402dd717df96fefb06b39682572c1ff33fb6c360d02a72f00cbeb327fad.jpg)  
Figure S 79: Effect ratios of the fine-tuning strategy on ChemBERTa embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/3afcd19299b0e727c32862e90f8592d47ed8f0009a94a6411e6cf1c30e4303b8.jpg)  
Figure S 80: Optimization trajectories after using different strategies to fine-tune ChemBERTa embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/ddddba2a3b1d11402ecf76e67cd78f6c196d7a8f941dc651caf6c1243f15952f.jpg)  
Figure S 81: Logit-scale effect of the fine-tuning strategy on ChemBERTa embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/16ac490af63a58f36806cf6bc4c187f7a8b2785437b1c5e06ea0bc58c593832f.jpg)  
Figure S 82: Effect ratios of the fine-tuning strategy on ChemBERTa embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/d3fb7ca25251f65362236936d407c36deeb79fbd84f993e8e20f7e0c7ab5056f.jpg)  
Figure S 83: Optimization trajectories after using different strategies to fine-tune ChemBERTa embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/135a6c3b66c20fd91bb46f2a1ef13c1b908e99289992ef6a346de88ec03fb332.jpg)  
Figure S 84: Logit-scale effect of the fine-tuning strategy on ChemBERTa embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/950ba6cb21f46573252ada7ec2d70bea9c387492e5edafc35302fdb07a8fc755.jpg)  
Figure S 85: Effect ratios of the fine-tuning strategy on ChemBERTa embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/b9725fac44ebbb7e538e4a01a60004c59e1399b54d4002034e552aba7021af18.jpg)  
Figure S 86: Optimization trajectories after using different strategies to fine-tune ChemBERTa embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/658dc7858553eff5deb5a41e442d745d4794ebc5876afb884fbf20580417ca7c.jpg)  
Figure S 87: Logit-scale effect of the fine-tuning strategy on ChemBERTa embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/6ce78b327b5699995d8a68cc2662c1ae6631bf9988807881e809cf5a05775b2c.jpg)  
Figure S 88: Effect ratios of the fine-tuning strategy on ChemBERTa embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/9b406e12987aebb3f6ee307cd6ba1b8de93c263dcfedc8f5e68d3c99e95deec0.jpg)  
Figure S 89: Optimization trajectories after using different strategies to fine-tune MolFormer-XL embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/14b3dcb5a48d475f27c2bdd0d53290782877edfb9db1acb023cea6ed037a6af4.jpg)  
Figure S 90: Logit-scale effect of the fine-tuning strategy on MolFormer-XL embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/b3f49f87c7223c1d1b55fdea53ada6a075f87915cd5bb45ea027985240e123ba.jpg)  
Figure S 91: Effect ratios of the fine-tuning strategy on MolFormer-XL embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/00f5366601e22be89e476d13461ed3f122ec60c6084fb30ca6a5e3c6c1cbc3f4.jpg)  
Figure S 92: Optimization trajectories after using different strategies to fine-tune MolFormer-XL embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/aa28f4e5bc00b00023adfca4841bd849c6fd004a6760023146764dfce3a9dbb3.jpg)  
Figure S 93: Logit-scale effect of the fine-tuning strategy on MolFormer-XL embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/3b13ccd7ded466b1750825fbc95f63b8ff90b40e0b802ee0f0f140ab49616cd6.jpg)  
Figure S 94: Effect ratios of the fine-tuning strategy on MolFormer-XL embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/8e28311fcaf4306acbdb3faec03043bd8cb27de57c13be0d4f2ff5262e6ee55e.jpg)  
Figure S 95: Optimization trajectories after using different strategies to fine-tune MolFormer-XL embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/65c452d29175c31aec5ffdea6324853830f92c41665c9e182d437df9e52b05a9.jpg)  
Figure S 96: Logit-scale effect of the fine-tuning strategy on MolFormer-XL embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/88f19a31620a12a8ef235332f5f496a658fbde91de05ac58217c0b0292a2f053.jpg)  
Figure S 97: Effect ratios of the fine-tuning strategy on MolFormer-XL embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/1532b05717b39874a74dc0de62b72031863b2b5fcc65634fc4df5aca381f0281.jpg)  
Figure S 98: Optimization trajectories after using different strategies to fine-tune MolFormer-XL embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/133819da7ca96b6c0dc50e2a1d908e4ad040122586973c2a3d42527e5d6ecaab.jpg)  
Figure S 99: Logit-scale effect of the fine-tuning strategy on MolFormer-XL embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/98faf55deabd2f5277bf8a0c38348587970b7f26d6d0b2f85a8ed85f3bf910d1.jpg)  
Figure S 100: Effect ratios of the fine-tuning strategy on MolFormer-XL embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/974a51de019a23d9c78b14b4ea82d99556f0ced5d1c1448824f4d67cfa2b6d6e.jpg)  
Figure S 101: Optimization trajectories after using different strategies to fine-tune SmiTed embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/439a68e567be9409d0dc99bac9a7bcdcd7df8a04a2b160b9f5923ae6110fd519.jpg)  
Figure S 102: Logit-scale effect of the fine-tuning strategy on SmiTed embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/4696ad78e48b8f7e3868601e92ab0d73730073774a2cc56eb9d2c40141ae730c.jpg)  
Figure S 103: Effect ratios of the fine-tuning strategy on SmiTed embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/c561ad5e6dade410c311101501414a0993e1097b48b3fe2dd1d62ec40db17cfa.jpg)  
Figure S 104: Optimization trajectories after using different strategies to fine-tune SmiTed embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/af4a5c96a8cfa772f3f6a24d6fe3cb661d6af387114c77c2d2ddec6d5b1fbf4c.jpg)  
Figure S 105: Logit-scale effect of the fine-tuning strategy on SmiTed embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/3f7b47fc2944db4d49099ac0834155d22b83c5147fcd8c8d9626da276b04a304.jpg)  
Figure S 106: Effect ratios of the fine-tuning strategy on SmiTed embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/455da3809f096ccafc7f6e0836331dd117236cee9b6681e1eb979c5c8259b905.jpg)  
Figure S 107: Optimization trajectories after using different strategies to fine-tune SmiTed embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/74bca2fa565a00aeb43b1a41f502f65c6f5fa5570bdb4d8e61f0b4bd76572d33.jpg)  
Figure S 108: Logit-scale effect of the fine-tuning strategy on SmiTed embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/d4e7a1aafedf9353a9f4a55ba7793db3001cbce0f6a04f4af2a76676cb70b21a.jpg)  
Figure S 109: Effect ratios of the fine-tuning strategy on SmiTed embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/53c246dca27b08eec1272bd8ff28848c9cf8f79e01c83313ab979f40709872a2.jpg)  
Figure S 110: Optimization trajectories after using different strategies to fine-tune SmiTed embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/27650122fa1f58d30d161763dae70e588626ea27285820c6d8a2f236de93da80.jpg)  
Figure S 111: Logit-scale effect of the fine-tuning strategy on SmiTed embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/4ff59a728c4469ad9307a63123921b9b248232b38f101da941de8a7b63f8aa36.jpg)  
Figure S 112: Effect ratios of the fine-tuning strategy on SmiTed embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/5ccc2d09c829cbc7e8f3715984e9708931ffcd0f8343cb5db514d8b6d605a074.jpg)  
Figure S 113: Optimization trajectories after using different strategies to fine-tune T5Chem embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/c85eeb306c5ce48b9d2c2c21dc58dfe495fa476134e532aa3a8aecbd4764827f.jpg)  
Figure S 114: Logit-scale effect of the fine-tuning strategy on T5Chem embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/40241b3cfd2407f03a5bc55ea2dd4217554cf901abc69ac45dee8ef4adb8ba18.jpg)  
Figure S 115: Effect ratios of the fine-tuning strategy on T5Chem embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/ceb8c18a6f52057925d57ec63a61836a32c0cc914448f2696d9cd3c3491c8806.jpg)  
Figure S 116: Optimization trajectories after using different strategies to fine-tune T5Chem embeddings, evaluated over a budget of 20,000 experiments (batch size of 1,000 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/e0d75d20061bfec54424462ff0a014fb8e161011bab18bf1e4e03ff7608eec62.jpg)  
Figure S 117: Logit-scale effect of the fine-tuning strategy on T5Chem embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/4e36de1666f1ca9d7c9e6028a023524d62a411f1c113cd36872d574951e3eda6.jpg)  
Figure S 118: Effect ratios of the fine-tuning strategy on T5Chem embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the virtual screening setting (20,000 experiments, batch size 1,000). Bars within the violins indicate the 95% highest-density intervals.

![](images/46e5b5309a813ab30727c0798f735f3e98c2dafed86fe6f4d828a5b6feb1c240.jpg)  
Figure S 119: Optimization trajectories after using different strategies to fine-tune T5Chem embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/9601fe28b585bcc47fc6b52311a3a991809294268c73fc1b4c3dbcf68c05342e.jpg)  
Figure S 120: Logit-scale effect of the fine-tuning strategy on T5Chem embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/6aeb3fbe21170ad5e8291c3be1cdb2633efa0c932a8ed34000dd5a7058f24228.jpg)  
Figure S 121: Effect ratios of the fine-tuning strategy on T5Chem embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

![](images/1f83b385becaf0be4adb71775bf73d5c32b916a89d2aabcdf6f6b758e3203aaf.jpg)  
Figure S 122: Optimization trajectories after using different strategies to fine-tune T5Chem embeddings, evaluated over a budget of 1,000 experiments (batch size of 50 experiments). Trajectories are shown as the mean over 20 independent campaigns from different random seed populations. The shaded areas indicate the standard error of the mean.

![](images/b9247825d7e46a3a6c7979d5280c3e6f2dabca2c00db12138e783233f479955b.jpg)  
Figure S 123: Logit-scale effect of the fine-tuning strategy on T5Chem embeddings, differentiated by molecular libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Means and standard deviations were obtained from 8,000 posterior samples.

![](images/74f6233bee97132b15ac3145d0a9ee49f69c314145f725495393f5731395a526.jpg)  
Figure S 124: Effect ratios of the fine-tuning strategy on T5Chem embeddings, averaged over all libraries. Experiments were performed using a Laplace Neural network model in the experimental discovery setting (1,000 experiments, batch size 50). Bars within the violins indicate the 95% highest-density intervals.

## 5 Supplementary References

O. Abril-Pla, V. Andreani, C. Carroll, L. Dong, C. J. Fonnesbeck, M. Kochurov, R. Kumar, J. Lao, C. C. Luhmann, O. A. Martin, M. Osthege, R. Vieira, T. Wiecki and R. Zinkov, Peer J. Comput. Sci., 2023, 9, e1516.

2 D. Rogers and M. Hahn, J. Chem. Inf. Model., 2010, 50, 742–754.

3 RDKit: Open-source cheminformatics. https://www.rdkit.org.

4 H. Moriwaki, Y.-S. Tian, N. Kawashita and T. Takagi, J. Cheminformatics, 2018, 10, 4.

5 S. Chithrananda, G. Grand and B. Ramsundar, arXiv, 2020, DOI:10.48550/arXiv.2010.09885.

6 J. Ross, B. Belgodere, V. Chenthamarakshan, I. Padhi, Y. Mroueh and P. Das, Nat. Mach. Intell., 2022, 4, 1256–1264.

7 J. Lu and Y. Zhang, J. Chem. Inf. Model., 2022, 62, 1376–1387.

8 E. Soares, E. Vital Brazil, V. Shirasuna, D. Zubarev, R. Cerqueira and K. Schmidt, Commun. Chem., 2025, 8, 193.

9 E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang and W. Chen, Int. Conf. Learn. Rep., 2021, DOI:10.48550/arXiv.2106.09685.

10 M. Henderson, R. Al-Rfou, B. Strope, Y. Sung, L. Lukacs, R. Guo, S. Kumar, B. Miklos and R. Kurzweil, arXiv, 2017, DOI:10.48550/arXiv.1705.00652.

11 J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger and A. G. Wilson, Adv. Neur. Inf. Proc. Syst., DOI:10.48550/arXiv.1809.11165.

12 R.-R. Griffiths, L. Klarner, H. B. Moss, A. Ravuri, S. Truong, S. Stanton, G. Tom, B. Rankovic, Y. Du, A. Jamasb, A. Deshwal, J. Schwartz, A. Tripp, G. Kell, S. Frieder, A. Bourached, A. Chan, J. Moss, C. Guo, J. Durholt, S. Chaurasia, F. Strieth-Kalthoff, A. A. Lee, B. Cheng, A. Aspuru-Guzik, P. Schwaller and J. Tang, Adv. Neur. Inf. Proc. Sys., DOI:10.48550/arXiv.2212.04450.

13 C. Hvarfner, E. O. Hellsten and L. Nardi, Proc. Int. Conf. Mach. Learn., 2024, 41, 20793–20817.

14 E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer and P. Hennig, in Proceedings of the 35th International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA, 2021, pp. 20089–20103.

15 F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot and É. Duchesnay, J. Mach. Learn. Res., 2011, 12, 2825–2830.

16 D. E. Graff, E. I. Shakhnovich and C. W. Coley, Chem. Sci., 2021, 12, 7866–7881.

17 F. Häse, L. M. Roch, C. Kreisbeck and A. Aspuru-Guzik, ACS Cent. Sci., 2018, 4, 1134–1145.

18 L. McInnes, J. Healy and J. Melville, arXiv, 2020, DOI:10.48550/arXiv.1802.03426.

19 https://github.com/lmcinnes/umap.

20 L. van der Maaten and G. Hinton, J. Mach. Learn. Res., 2008, 9, 2579–2605.

21 https://github.com/pavlin-policar/openTSNE.

22 J. Lyu, S. Wang, T. E. Balius, I. Singh, A. Levit, Y. S. Moroz, M. J. O’Meara, T. Che, E. Algaa, K. Tolmachova, A. A. Tolmachev, B. K. Shoichet, B. L. Roth and J. J. Irwin, Nature, 2019, 566, 224– 229.

23 A. Luttens, I. Cabeza de Vaca, L. Sparring, J. Brea, A. L. Martínez, N. A. Kahlous, D. S. Radchenko, Y. S. Moroz, M. I. Loza, U. Norinder and J. Carlsson, Nat. Comput. Sci., 2025, 5, 301–312.

24 J. Hachmann, R. Olivares-Amaya, S. Atahan-Evrenk, C. Amador-Bedolla, R. S. Sánchez-Carrera, A. Gold-Parker, L. Vogt, A. M. Brockway and A. Aspuru-Guzik, J. Phys. Chem. Lett., 2011, 2, 2241–2251.

25 F. Strieth-Kalthoff, H. Hao, V. Rathore, J. Derasp, T. Gaudin, N. H. Angello, M. Seifrid, E. Trushina, M. Guy, J. Liu, X. Tang, M. Mamada, W. Wang, T. Tsagaantsooj, C. Lavigne, R. Pollice, T. C. Wu, K. Hotta, L. Bodo, S. Li, M. Haddadnia, A. Wołos, R. Roszak, C. T. Ser, C. Bozal-Ginesta, R. J. Hickman, J. Vestfrid, A. Aguilar-Granda, E. L. Klimareva, R. C. Sigerson, W. Hou, D. Gahler, S. Lach, A. Warzybok, O. Borodin, S. Rohrbach, B. Sanchez-Lengeling, C. Adachi, B. A. Grzybowski, L. Cronin, J. E. Hein, M. D. Burke and A. Aspuru-Guzik, Science, 2024, 384, eadk9227.

26 T. Gensch, G. dos Passos Gomes, P. Friederich, E. Peters, T. Gaudin, R. Pollice, K. Jorner, A. Nigam, M. Lindner-D’Addario, M. S. Sigman and A. Aspuru-Guzik, J. Am. Chem. Soc., 2022, 144, 1205–1217.

27 S. Zhao, T. Gensch, B. Murray, Z. L. Niemeyer, M. S. Sigman and M. R. Biscoe, Science, 2018, 362, 670–674.