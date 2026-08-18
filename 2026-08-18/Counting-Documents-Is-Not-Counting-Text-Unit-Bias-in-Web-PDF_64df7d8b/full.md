# Counting Documents Is Not Counting Text: Unit Bias in Web-PDF Corpus Statistics

Luca Foppiano Common Crawl Foundation luca@commoncrawl.org

## Abstract

PDF corpora advertise their size in tokens but compute every rate they publish (coverage, OCR routing, re-fetch recovery, language mix) per document, and none decomposes its token total. The two units diverge sharply. On CC-MAIN-2021-31-PDF-UNTRUNCATED (7.9M web PDFs, 32.6B tokens), 3.02% of textbearing documents hold half the tokens (Gini 0.807); documents over 50 pages are 5.00% of the corpus but 53.53% of its text. The PDFs produced by a T X toolchain are 1.66% of documents and 4.05% of the text. The clearest casualty is Common Crawl’s truncation cap: it affected 23.06% of documents and 63.08% of the text. Reconstructing the truncated files and extracting both versions, two widely used libraries recover 11.4% and 1.4% of that text; between 72% and 97% of affected documents yield nothing; roughly 55–62% of the corpus’s text is lost. Under the 5 MiB cap adopted in March 2025, 30.19% of tokens would still be truncated, and recovery on those documents rises only from 3.3% to 13.2%. We recommend that corpus statistics be reported in both units: documents and tokens.

## 1 Introduction

PDFs have become a first-class source of pretraining text (Poznanski et al., 2025; Montalvo and Wightman, 2024). In one current open data pool, PDF-derived text amounts to roughly 4.3T tokens.<sup>1</sup> FinePDFs alone is described as “about 3 trillion tokens across 475 million documents” (Kydlícekˇ et al., 2025).

Those corpora are not naive about tokens: they headline them. What none of them does is compute a rate in tokens, or break a token total down. FinePDFs routes 368.8M of 1.29Bfiles to OCR and recovers 53.5% of its truncated files by re-fetching; CCpdf reports per-document success rates (Turski et al., 2023); PDFA gives three totals (2,159,432 documents, 18M pages, 9.7B tokens) but no drop rates in any unit (Montalvo and Wightman, 2024). No PDF corpus we are aware of decomposes its token total by length or quality.

This would be harmless if documents were interchangeable. They are not. The web-text community already recognises this: Nemotron-CC reports a token-weighted yield, “+57.4% more highquality tokens” (Su et al., 2025), and PDFs have far more extreme length skew than HTML. FinePDFs itself observes that its documents have a “∼5.3k character median (about 2× other corpora)” with a “95th percentile ∼68k” against “∼11–13k elsewhere,” and then reports every rate per document anyway.

We measure what those rates become when weighted by text. Our contributions:

1. the first token-weighted characterisation of a web-PDF corpus, with a direct documentversus-token comparison of every headline statistic (§4.1–4.3);

2. a concentration result: text mass in web PDFs is far more skewed than document counts suggest (Gini 0.807);

3. the first measurement of what Common Crawl’s truncation cap costs in text: 23.06% of documents but 63.08% of tokens, at most 11.4% of it recoverable, and 30.19% still lost at the new 5 MiB limit (§4.4–4.6);

4. released code.

## 2 Related Work

Totals in tokens, rates in documents. FinePDFs (Kydlícek et al. ˇ , 2025) routes documents between a text path (Docling) and a GPU OCR path with a learned classifier reporting F1 0.71 on the OCR class, and quantifies every stage in files. CCpdf (Turski et al., 2023) tabulates “number of documents per processing step and language” and reports no token count for its own corpus. PDFA (Montalvo and Wightman, 2024), derived from the same corpus we study, filters per document, discarding files over 100 MB or slower than 500 ms to render, i.e. selecting on size, and publishes no drop rates. GovScape (Huang et al., 2026) notes that “for some pages, no corresponding text representation is found embedded within the PDF. . . these pages are currently excluded” without quantifying the exclusion in any unit. Extraction benchmarks (Ouyang et al., 2025; Poznanski et al., 2025) measure fidelity on documents that parsed and report no coverage at all.

Truncation is named but never quantified. Common Crawl caps the payload it stores per record. PDFs are the format worst affected: for the crawl CC-MAIN-2023-06 they were 0.8% of successfully fetched records but 11.85% of 88 TiB of WARC storage (PDF Association, 2023). CCpdf stated the consequence plainly in 2023: the crawler’s 1 MB cap is “quite high for HTML pages, but unfortunately rather low for PDF files” (Turski et al., 2023). Their workaround was to re-download the truncated documents from origin; to our knowledge nobody since has quantified the effect.

From CC-MAIN-2025-13 (March 2025) the cap rose from 1 MiB to 5 MiB, reported as +13% fetched content (403 TiB → 455 TiB) (Common Crawl Foundation, 2025).

The Common Crawl Foundation reported the effect of the change on truncation by MIME type: PDFs fell from 25.7% to 6.8%, against 2.25% to 0.14% for all types and 2.2% to 0.04% for HTML. Even after the change PDFs are truncated at roughly 49× the rate of content generally. What is not reported, before or after, is what that costs in text.

## 3 Data and Method

Corpus. CC-MAIN-2021-31-PDF-UNTRUNCATED (Digital Corpora et al., 2021) contains every PDF found in Common Crawl CC-MAIN-2021-31, with payloads that Common Crawl truncated refetched whole from origin. It ships five metadata tables covering 8.3M URLs. Because file\_name is the post-SHA-256 identity, we deduplicate to one row per file, giving 7,932,654 unique documents and 32,570,135,761 tokens.

<table><tr><td>Shape</td><td>doc %</td><td>tok %</td><td>ratio</td></tr><tr><td>report/thesis (&gt;50 pp)</td><td>4.42</td><td>49.71</td><td>11.24×</td></tr><tr><td>article-shaped (4–30 pp)</td><td>31.64</td><td>26.87</td><td>0.85×</td></tr><tr><td>long (31–50 pp)</td><td>3.21</td><td>9.08</td><td>2.83×</td></tr><tr><td>landscape (slides)</td><td>14.08</td><td>7.76</td><td>0.55×</td></tr><tr><td>2-3pp</td><td>21.20</td><td>3.76</td><td>0.18×</td></tr><tr><td>1 page (flyer/form)</td><td>24.96</td><td>2.70</td><td>0.11×</td></tr></table>

Table 1: The same corpus counted two ways. Categories are mutually exclusive and orientation takes precedence over page count, so a long landscape document is counted as slides; the row therefore covers portrait documents only. Counting purely by page count, documents over 50 pages are 5.00% of the corpus and 53.53% of its text; 0.49% of documents with unparseable page metadata are omitted.

The unit. Text mass is taken from tika\_eval\_num\_tokens, a token count already published as part of the corpus metadata (Digital Corpora et al., 2021) (Apache Tika 2.8.0, tesseract disabled), a whitespace/ICU count rather than an LLM tokenizer’s. We claim only proportions, which is all the argument requires. The field is right-censored at 10,000,000, but only two documents in the corpus reach that cap.

Truncation. The provenance table records cc\_truncated, fetched\_status, fetched\_length and the WARC byte range per file (cc\_truncated=‘length’ coincides exactly with fetched\_status=‘REFETCHED\_SUCCESS’). For truncated records the WARC record length pins to ∼1.049 MB, so the cap is directly observed, and the re-fetched originals give the true size of every document Common Crawl stored only as a fragment: a pairing no other public corpus offers.

## 4 Results

## 4.1 Text mass is extremely concentrated

Table 1 gives the page-count distribution in both units. Counting by page count alone regardless of orientation, documents over 50 pages are 5.00% of the corpus and 53.53% of its text: a document in that tail carries about eleven times an average document’s text. We use this threshold for the concentration claim here and in §4.2. The mirror image: the 46.2% of documents with three pages or fewer contribute 6.46% of the text. Over the 7,292,093 text-bearing documents, 3.02% hold half the tokens and 15.54% hold 80%; the Gini coefficient of tokens across documents is 0.807.

A corpus described by document count as “mostly flyers and forms” is, by text, a corpus of long documents. Both descriptions are arithmetically correct.

<table><tr><td></td><td>docs %</td><td>tokens %</td></tr><tr><td>Truncated by CC (1 MiB)</td><td>23.06</td><td>63.08</td></tr><tr><td>still truncated at 5 MiB</td><td>5.94</td><td>30.19</td></tr></table>

Table 2: Common Crawl’s truncation cap, priced in both units, over the 7,932,878 files with provenance records.

## 4.2 The concentration is not an artifact of one extractor

Since the skew could be a property of the extractor rather than of the corpus, we validate the distribution against page counts, which the corpus authors extracted with Poppler, an independent tool measuring an entirely different quantity. Page counts reproduce the same concentration: the Gini coefficient of pages across documents is 0.767 against 0.807 for tokens, 3.98% of documents hold half the pages against 3.02% for tokens, and documents over 50 pages account for 54.27% of all pages against 53.53% of all tokens, a difference of 0.74 percentage points. Two tools measuring two different quantities give the same answer: a shared extraction bias cannot produce this agreement.

## 4.3 The scholarly share more than doubles

Classifying the producer/creator strings into toolchain families, a T X toolchain accounts for 1.66% of documents but 4.05% of tokens (2.43×). PowerPoint runs the other way, 2.24% of documents against 0.82% of tokens (0.37×), and Microsoft Word is 24.95% of documents against 19.37% of tokens.

The conclusion that this corpus is not primarily scholarly can be inferred using both units, but its magnitude is off by a factor of 2.4 in the unit a language model consumes.

## 4.4 Truncation: 23% of documents, 63% of the text

Common Crawl truncated 1,829,061 of this corpus’s documents at the 1 MiB cap in force at the time, corresponding to 23.06% of them. Those documents hold 63.08% of the corpus’s text (Table 2). Truncated documents carry 2.74× the mean token mass; their median true size is 2.62 MB against 190 KB for the rest.

Applying the current 5 MiB cap to the corpus, 5.94% of documents and 30.19% of tokens would still be truncated. The March 2025 change therefore recovers roughly half of the exposure and leaves the other half in place. Our documentlevel counterfactual is consistent with Common Crawl’s own post-change measurement of 6.8% on 2025 crawls<sup>2</sup>, computed on a population five years younger; the token figure has no published counterpart.

In bytes, the 1 MiB cap kept 1.738 TiB of these files’ 9.449 TiB total, discarding 81.6%; storing them whole would have cost 7.712 TiB of additional archive. The 5 MiB cap lands between the two: it would keep 5.106 TiB (46.0% discarded), so the March 2025 change spends 3.368 TiB of that 7.712 TiB and leaves 4.344 TiB behind. At either cap, byte loss overstates text loss: 81.6% of bytes against 63.08% of tokens at 1 MiB, 46.0% against 30.19% at 5 MiB, because PDF bytes are largely images and embedded fonts. The byte figure prices the archive; the token figure prices the corpus.

## 4.5 Truncation destroys text rather than trimming it

To measure what is actually lost in truncation we reconstruct what Common Crawl held, cutting each of this corpus’s 1,829,061 truncated documents at 1 MiB, and extracting both versions. Whether a fragment is recoverable may be a property of the parser rather than of the file, so we run two independent engines, PyMuPDF (MuPDF) (Artifex Software, Inc., 2025) and PDFium (Chromium’s) (pypdfium2 team, 2025), and report on the 1,225,130 documents both tools successfully processed, and whose whole version yields text.

PyMuPDF recovers 11.4% of the tokens and 20.7% of the pages; PDFium recovers 1.4% and 1.9%, a gap of 8.4× on identical input. The two agree to within 1.7% on the intact versions of those same documents, so the divergence is specific to damaged input and is not a general difference in extraction quality. There is therefore no single “recovery rate” for a truncated PDF: the quantity is not defined until the extractor is named.

The shape of the failure differs as sharply as its size, and both shapes are invisible to a pipeline that counts successful parses. PyMuPDF opens 98.9% of truncated files, because MuPDF rebuilds a missing cross-reference table, but the page content streams lie beyond the cut, so the document opens, reports a page count, and returns no text: 72.4% yield nothing at all. PDFium attempts no such reconstruction and refuses the file outright, failing to open 96.7% of the same documents; its zero-yield rate of 96.8% is almost exactly its openfailure rate. One extractor logs these as successes with empty output, the other as parse errors.

The gap is not spread evenly across the corpus. A linearized PDF (Adobe “Fast Web View”) places its first page and a cross-reference table at the front of the file so that a partial copy is still renderable; 40.7% of the documents here are linearized. Under PyMuPDF those files recover 1.1% of their tokens against 18.5% for the non-linearized rest, while under PDFium it is 1.1% against 1.5%. Note that the gap persists when files of similar size are compared, so it is not a size effect. The difference between the two engines therefore lives in the non-linearized population, and the Fast Web View structure, designed to keep partial files usable, is the one from which least is recovered.

Applying the recovered fractions to the exposure in Table 2, roughly 55% to 62% of this corpus’s total text is destroyed by the 1 MiB cap, the range spanning the two extractors. That product multiplies an exposure counted in Tika tokens by a recovery counted in whitespace tokens. Since every truncated document has both counts, we re-weight each by its Tika count instead: the corpus figure moves by 0.42 points under PyMuPDF and by zero under PDFium. The two counts agree closely per document (median ratio 1.01–1.02) and diverge only for scripts without whitespace word separators: a property of the corpus, not of truncation. One caveat does remain: we count tokens rather than reading them, and a token count cannot distinguish text that a repair path recovered cleanly from reconstruction artefact, so “recovers more” must not be read as “recovers better”.

## 4.6 The 5 MiB cap recovers little of what it still exposes

Raising the cap to 5 MiB (Table 2) leaves 30.19% of tokens exposed; we tested whether they are recoverable, since every document truncated at 5 MiB is also truncated at 1 MiB and only the cut point moves between the runs.

On the 316,174 documents PyMuPDF processed at both caps, recovery rises from 3.3% to 13.2% of tokens, and 21,203 documents (6.7%) go from yielding no text at all to yielding some. PDFium moves from 0.1% to 2.4% on the same documents.

A five-fold larger prefix therefore multiplies recovered text roughly four-fold, but still leaves 77.6% of these documents (97.0% under PDFium) yielding no text. The change halves the exposure; what it leaves behind remains almost entirely unreadable.

The benefit is also concentrated immediately above the cap (Table 3, Appendix A): 74% of the rescued documents lie between 5 and 10 MiB, while above 25 MiB the larger cap is worth 2–3 percentage points. This is what a fixed prefix must do, 5 MiB is half of a 10 MiB file and a twentieth of a 100 MiB one, and it holds under both engines. Raising the cap further has sharply diminishing returns per additional TiB of archive: the documents that dominate the remaining token mass are those a larger fixed prefix helps least.

Both effects reflect the cut rather than the run: token counts from the whole documents agree exactly for all but 0.036% of paired documents under both engines (a shared, harness-imposed deadline). Recovery is not monotone, however: 800 documents (0.25%) yield text at 1 MiB and none at 5 MiB, so repair is sensitive to where a file stops, but the effect is 26× rarer than the reverse and does not disturb the aggregate.

## 5 Conclusion

Counted in documents and counted in text, the studied corpus CC-MAIN-2021-31-PDF-UNTRUNCATED is two different corpora: the units diverge by up to 11× per category, and Common Crawl’s cap cost 23% of documents but 55–62% of the text. Corpus statistics should therefore be reported in both units; every rate in this literature (coverage, filter drop, OCR routing) is per-document or unreported. Sizebased filters must be priced in text, because size is where the text is: PDFA’s 100 MB and 500 ms cuts, FinePDFs’ router, and the cap itself all select on it. For Common Crawl, §4.4–4.6 price the 1 MiB → 5 MiB change: it halves the exposure but recovers little of what it still truncates; the remaining exposure calls for a size-aware fetch policy for large PDFs, not a higher cap.

## Limitations

Every figure describing corpus composition and truncation exposure is computed over all 7.9M documents from the shipped metadata. The recovery figures cover the 1.2M truncated documents both extractors successfully processed, 67% of the 1.83M Common Crawl truncated. The shortfall is concentrated in ten of our sixty-four PyMuPDF shards, which produced nothing because of technical errors in the extractor (e.g. out of memory, hanging); since shards partition the corpus by archive index, the loss is arbitrary with respect to document content, and per-shard recovery rates vary by about a percentage point. It is nonetheless not a random sample, and we report no confidence intervals. The metadata is third-party and derived (Digital Corpora et al., 2021), and producer/creator strings are self-reported with 8.08% unclassified, so the T X share is a lower bound in both units. page\_size is recorded for the first page only, making the landscape category approximate. We study a single crawl snapshot, from before the policy change, and the only one for which re-fetched originals exist; the 5 MiB figure is therefore a counterfactual computed on 2021 file sizes rather than a measurement of a current crawl. Finally, the two extractors we compare are both text-layer parsers; an OCR pipeline would fail differently again, and we do not measure one.

## Acknowledgement

We thank DFKI (Deutsches Forschungszentrum für Künstliche Intelligenz GmbH) for supporting this work with computation and storage.

## Data and Code availability

The data was collected from the resource provided by Digital Corpora et al. (2021). The code is available at https://github.com/lfoppiano/cc-w acky-pdf.

## References

Artifex Software, Inc. 2025. PyMuPDF. Version 1.28.2; Python bindings for MuPDF.

Common Crawl Foundation. 2025. March 2025 crawl archive now available. https://commoncrawl.or g/blog/march-2025-crawl-archive-now-avail able. Truncation limit raised from 1 MiB to 5 MiB at CC-MAIN-2025-13.

Digital Corpora, NASA JPL, and DARPA SafeDocs. 2021. CC-MAIN-2021-31-PDF-UNTRUNCATED (SAFEDOCS). https://digitalcorpora.org/c orpora/file-corpora/cc-main-2021-31-pdf-u ntruncated/.

Ying-Hsiang Huang, Claire Gong, Shreya Shaji, Alison Yan, Leslie Harka, Albert Du, Anjali Gopal, Samuel J Klein, Shannon Zejiang Shen, Mark Phillips, Trevor Owens, Kyle Deeds, and Benjamin Charles Germain

Lee. 2026. Govscape: A public multimodal search system for 70 million pages of government pdfs. Preprint, arXiv:2511.11010.

Hynek Kydlícek, Guilherme Penedo, and Leandro Vonˇ Werra. 2025. Finepdfs: Liberating 3t of the finest tokens from pdfs.

Pablo Montalvo and Ross Wightman. 2024. PDFA: pdfa-eng-wds. https://huggingface.co/datas ets/pixparse/pdfa-eng-wds.

Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. 2025. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24838–24848. IEEE.

PDF Association. 2023. New large-scale pdf corpus now publicly available. https://pdfa.org/new -large-scale-pdf-corpus-now-publicly-ava ilable. Reporting WARC storage and truncation shares for CC-MAIN-2023-06.

Jake Poznanski, Aman Rangapur, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. 2025. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443.

pypdfium2 team. 2025. pypdfium2. Version 5.12.1; Python bindings for PDFium.

Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. 2025. Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining dataset. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 2459–2475.

Michał Turski, Tomasz Stanisławek, Karol Kaczmarek, Paweł Dyda, and Filip Gralinski. 2023. CCpdf:´ Building a high quality corpus for visually rich documents from web crawl data. In Document Analysis and Recognition – ICDAR 2023. ArXiv:2304.14953.

## A Recovery by document size

Table 3 breaks the paired recovery measurement of §4.6 down by document size.
<table><tr><td></td><td colspan="3">recovered</td><td></td></tr><tr><td>size</td><td>docs</td><td>1MiB</td><td>5MiB</td><td>gain</td></tr><tr><td>5-10 MiB</td><td>179,810</td><td>4.88%</td><td>22.44%</td><td>+17.6</td></tr><tr><td>10–25 MiB</td><td>99,831</td><td>2.62%</td><td>8.97%</td><td>+6.4</td></tr><tr><td>25-50 MiB</td><td>25,700</td><td>1.64%</td><td>4.47%</td><td>+2.8</td></tr><tr><td>50-100 MiB</td><td>8,683</td><td>1.36%</td><td>3.28%</td><td>+1.9</td></tr><tr><td>&gt;100MiB</td><td>2,150</td><td>1.01%</td><td>2.96%</td><td>+2.0</td></tr></table>

Table 3: What raising the cap buys, by document size (PyMuPDF, gain in percentage points). The same 316,174 documents are cut at both points; the benefit is concentrated immediately above the cap.