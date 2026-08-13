# Large Language Model-Driven Small-Capitalization Trading: Integrating Financial News Sentiment, Macroeconomic Indicators, and Technical Signals

Alireza Kargarzadeh<sup>1</sup>, Nariman Khaledian<sup>2</sup>, Navid Parvini<sup>3</sup>, Arman Khaledian<sup>3</sup>

<sup>1</sup>Tailstate Intelligence Ltd., <sup>2</sup>Independent Researcher, <sup>3</sup>Zanista AI Ltd. Correspondence: alireza.kargarzadeh@tailstate.ai

## Abstract

Large language models can extract richer sig nals from financial news than fixed sentiment lexicons, and recent work has explored feeding such signals into portfolio construction. We study an uncertainty-aware construction that feeds model-predicted risk – decomposed into aleatoric and epistemic components – directly into the covariance matrix of portfolio alloca tors, rather than treating portfolio risk as fixed or adjusting only expected returns. We evaluate the pipeline on Russell 2000 equities under three stock-selection regimes: a pure-alpha trigger that isolates abnormal stock moves not explained by macro indicators, a pure-beta trig ger that captures macro-indicator moves before the stock itself fires, and a beta trigger in which both channels agree. Across the full holding-period grid, the separated pure-alpha and pure-beta legs usually dominate the beta intersection on Sharpe and return. Two hori zons are especially informative. At one day, pure beta can work under low and moderate transaction costs because it captures immedi ate lead-lag spillovers from liquid macro and sector indicators into exposed small-cap stocks, but this advantage disappears at 100 bps when turnover and microstructure noise dominate. At 40 days, pure beta works for a different rea son: slower macro repricing overtakes the firmspecific pure-alpha channel. The strongest conservative row is pure beta with GPT-4o mini sentiment, a Student-t target, a 40-day hold ing period, and risk parity allocation, reaching Sharpe 2.33 at 100 bps. The results suggest that stock-selection regime and allocator choice matter at least as much as the sentiment model, and that separating firm-specific and macro exposure triggers is more informative than requiring both to fire simultaneously. Code is available at the following repository.

## 1 Introduction

Financial news moves markets (Tetlock, 2007), but fixed lexicon methods miss negation, context, and domain-specific financial language. Large language models (LLMs) address this by conditioning on the full news context, and recent evidence shows that LLM-derived financial sentiment can contain return-predictive information and outperform dictionary-based sentiment measures (Lopez-Lira and Tang, 2026; Chen et al., 2022; Kirtac and Germano, 2024).

Two gaps remain. First, most studies score isolated headlines rather than related news stories, even though repeated coverage can overweight a single event. Second, sentiment is usually treated as an expected-return input while portfolio risk remains historical and signal-independent (Colasanto et al., 2022; Lee et al., 2025; Chen, 2025; Taheripour et al., 2025). This is limiting because a predictive model can also provide information about uncertainty and cross-asset covariance.

We address both gaps with a pipeline that clusters related articles into representative newssummary sentiment features, combines them with macroeconomic and technical indicators, and predicts both expected returns and covariance for portfolio construction. To limit look-ahead bias, sentiment is scored with models whose knowledge cutoffs precede the 2025 test period. The main empirical result is that decomposing the macro-beta trigger matters: pure alpha and pure beta generally outperform the beta intersection, but for different reasons across horizons. Figure 1 previews this pattern.

Contributions. We build a news-summary LLM sentiment pipeline that clusters related coverage before scoring, using models whose knowledge cutoffs limit look-ahead bias. We combine this with an uncertainty-aware return-distribution model that decomposes predictive covariance into aleatoric and epistemic components and feeds it directly into standard portfolio allocators, rather than treating risk as a separate, historically-estimated quantity. We evaluate the resulting pipeline across sentiment scorer, target distribution, allocator, holding period, transaction cost, and stock-selection regime, to test how much each of these choices actually matters.

![](images/b311d94fc31bdd93e0ef9e3a593b9b8b553d45e2acd67e9e29fa12b4077cbfe9.jpg)  
Figure 1: Holding-period sweep at 100 bps. Pure alpha leads at 5, 10, and 60 days; pure beta leads at 20 and 40 days.

## 2 Related Work

Turning financial news into a usable sentiment signal predates LLMs by nearly two decades. Early work scored sentiment from fixed positive/negative lexicons, showing that media pessimism, negativeword frequency, and informal investor discourse predict returns, earnings, and price reaction speed at both the market and firm level (Das and Chen, 2007; Tetlock, 2007; Tetlock et al., 2008; Calomiris and Mamaysky, 2019; Glasserman et al., 2019). These methods are transparent, but their fixed vocabularies struggle with negation, context, and domain-specific phrasing. LLMs address this limitation by conditioning on richer text context: Lopez-Lira and Tang (2026) show that GPT-4 scores from post-cutoff headlines predict initial reactions and subsequent drift, especially for smaller stocks and negative news; Chen et al. (2022) extract contextual embeddings from financial news and find incremental information beyond technical predictors and simpler text representations; and Kirtac and Germano (2024) compare OPT, BERT, FinBERT, and the Loughran–McDonald dictionary, with the OPT long–short strategy achieving a Sharpe ratio of 3.05 at 10 bps transaction costs. Jadhav and Mirza (2025) survey the broader LLMequity literature, confirming that this has become a fast-growing research area. Across this literature, however, news is typically treated as a stream of independent observations: each headline or article is scored on its own, and repeated syndicated coverage of the same event can therefore give that event outsized weight simply because it appears many times. We instead first collapse related coverage into representative news-summary sentiment before scoring, so that a story’s influence reflects how much it matters rather than how often it was reported.

LLM trading studies also face a specific validation problem: a model may appear predictive because its training data overlaps the evaluation period. Glasserman and Lin (2024) distinguish lookahead bias from a distraction effect, where general company knowledge can distort sentiment judgments even without memorizing the subsequent return. Eliseev and Seleznev (2026) extend the warning to macroeconomic forecasting with “fake date” tests, finding that none of the modern LLMs they test passes the strictest look-ahead screen. Rizvani et al. (2026) add a complementary robustness concern by showing that LLM-driven trading pipelines can be vulnerable to adversarially manipulated headlines. These findings motivate our cutoff-aware sentiment scoring, strict trailing standardization, and representative-story construction.

A separate literature asks how a predictive signal should become portfolio weights. Equal weighting, risk parity (Maillard et al., 2010), and hierarchical risk parity (López de Prado, 2016) either ignore the expected-return signal or allocate primarily from covariance, while mean-variance optimization (Markowitz, 1952) treats predicted return and risk as fixed point estimates. Sentimentbased portfolio papers usually inject language information into expected returns or Black–Litterman views: Colasanto et al. (2022) use a BERT sentiment score to form Black–Litterman views, Lee et al. (2025) translate LLM forecasts and predictive uncertainty into views and confidence levels, and Chen (2025) and Taheripour et al. (2025) adjust MVO-style expected-return inputs directly. Predictive uncertainty has been used for trade sizing by Spears et al. (2020), but at the single-position level rather than as a cross-asset portfolio covariance.

Kargarzadeh (2024) develops a smallcapitalization trading framework that combines LLM-scored financial news with macroeconomic and technical signals, and motivates decomposing macro-beta events into pure-alpha, pure-beta, and beta-intersection legs, according to whether a stock’s own abnormal return, its exposure to a co-moving macroeconomic indicator’s abnormal move, or both, exceed a z-score threshold.

![](images/c52f841cc36ae50b6819c5f290efda9c91c383352f0008c9aa9e4c5117d73e84.jpg)  
Figure 2: End-to-end trading pipeline. The experiment begins with a Russell 2000 price and news panel, resolves a tradable cross-section using one of three macro-beta trigger legs, clusters related articles into news-summary sentiment features, and trains a return-distribution network that outputs both expected returns and predictive covariance. The same predicted moments are then passed to six allocation rules, allowing the benchmark to isolate the effects of the stock-selection regime, sentiment scorer, target distribution, and portfolio construction method.

The remaining gap is therefore not whether text can contain return information, but how to combine news-summary sentiment, macro/technical stock selection, predictive uncertainty, and allocation in one controlled benchmark. We test that full chain by feeding model-predicted covariance directly into the portfolio stage and evaluating whether performance depends on the sentiment backend, return target, allocator, holding period, transaction cost, and stock-selection regime.

## 3 Methodology

For assets $i \in \{ 1 , \ldots , N \}$ and trading days t, our target is the forward log-return vector $y _ { t } ^ { ( H ) } \in \mathbb { R } ^ { N }$ over horizon H. Rather than predicting return alone, we estimate the full conditional return distribution – its mean and covariance, $\hat { \mu } _ { t } = \mathbb { E } [ y _ { t } ^ { ( H ) } \mid$ F<sub>t</sub>] and $\hat { \Sigma } _ { t } = \mathrm { C o v } ( y _ { t } ^ { ( H ) } \mid \mathcal { F } _ { t } )$ – since it is the covariance estimate that ultimately distinguishes our portfolio construction from prior work. Figure 2 gives the full workflow. The remainder of this section follows the pipeline end to end: how the tradable universe is selected (Section 3.1), how news becomes a daily sentiment signal (Section 3.2), how that signal becomes a joint return-and-risk prediction (Section 3.3), and how that prediction becomes a portfolio (Section 3.4). Dataset and experimental protocol are described in Section 4.

## 3.1 Stock Selection

We compare three ways to define the traded crosssection, each decomposing a macro-beta trigger into a separate event set. All three start from the same eligible Russell 2000 pool and define a stockside event set $S _ { S }$ and an indicator-side event set $S _ { I }$ A stock’s own movement is flagged via its return z-score,

$$
Z = { \frac { R - \mu } { \sigma } }\tag{1}
$$

and its relationship to a given indicator is measured via beta,

$$
\beta = \frac { \mathrm { C o v } ( R _ { i } , R _ { i n d } ) } { \mathrm { V a r } ( R _ { i n d } ) }\tag{2}
$$

computed over a rolling window against a curated panel of macroeconomic releases, commodities, currencies, rates, volatility, world-index, and sector-ETF indicators (details in Appendix A). An indicator trigger fires when an indicator has an abnormal move and the stock is sufficiently exposed to that indicator; a stock trigger fires when the stock itself has an abnormal move. Following the smallcapitalization LLM-driven trading framework of Kargarzadeh (2024), the three selection legs are then pure alpha, S<sub>S</sub> \ $S _ { I }$ ; pure beta, $S _ { I } \setminus S _ { S } ;$ and beta, $S _ { I } \cap S _ { S }$ Pure alpha isolates firm-specific abnormal moves not contemporaneously explained by macro indicators. Pure beta is the anticipatory macro leg: an indicator has moved for a levered stock, but the stock itself has not yet registered its own abnormal move. Beta is the confirmatory leg, where stock and indicator triggers fire together and agree on direction. In all three regimes, candidates are ranked using only in-sample BUY events and capped at 50 names for model training and backtesting. At allocation time, the filter acts as a mask: the predictive model still emits full return and covariance forecasts, but the allocator may hold only names with an active BUY signal under the selected leg.

## 3.2 From News to Sentiment

Much of the LLM-sentiment trading literature scores individual headlines or articles as independent observations (Section 2). We instead treat sentiment as a property of a story, not an article: multiple outlets frequently cover the same event, and scoring each report separately lets syndicated coverage dominate a signal simply by being repeated, not by being informative. For each article–asset pair, the sentiment model returns a calibrated probability distribution over negative, neutral, and positive outcomes. These probabilities are transformed into a final normalized sentiment surprise through entityprior correction, strictly trailing rolling demeaning, and group-day cross-sectional standardization. The exact scoring and normalization formulas are reported in Appendix B.

We then collapse repeated coverage before it reaches the model. Within a trailing 30-day window, article summaries are embedded and clustered by single-linkage agglomerative clustering using cosine distance as the merge criterion, merging two articles into the same story once their similarity reaches 0.90. Each resulting cluster is treated as one underlying story. For every story cluster, only the article nearest to the cluster centroid is retained as the representative news item; all other articles in that cluster are treated as duplicate coverage. The clustering, centroid, and representative-selection formulas are given in Appendix B. The daily sentiment feature $\bar { s } _ { i , d }$ is then computed from these representative articles and paired with a log-scaled story count as a coverage-volume feature. Neither feature is forward-filled, so the signal never reflects information unavailable at the time it would have been used.

To score sentiment, we use GPT-4o mini, whose October 2023 knowledge cutoff precedes our evaluation period by construction, directly limiting lookahead bias (Glasserman and Lin, 2024) – though not the related distraction effect, since the model may still carry general knowledge of a company independent of the specific event being scored. To test whether our results are an artifact of this particular scorer, we repeat the full pipeline with three alternative scorers spanning domain-specific and general-purpose designs: FinBERT (Araci, 2019), Mistral 7B Instruct (Jiang et al., 2023), and Llama 2 13B Chat (Touvron et al., 2023).

## 3.3 Joint Return and Risk Prediction

The sentiment signal above feeds a multimodal network alongside daily price- and volume-derived technical features. The news branch uses the newssummary sentiment and coverage features summarized in Table 2, while the daily branch uses the OHLCV-derived indicators summarized in Table 3. The two branches are encoded separately, concatenated over the lookback window, and combined into a per-asset representation $h _ { i , t }$ . Rather than treating risk as a separate, historically-estimated quantity the way prior work does, we predict it from this same representation, alongside return, and we distinguish two sources of uncertainty rather than collapsing them into one number.

Dropout applied to $h _ { i , t }$ (rate 0.2) gives a stochastic representation $\tilde { h } _ { i , t , n } = \mathrm { D r o p o u t } ( h _ { i , t } ; \omega _ { n } )$ for the n-th of S forward passes; a mean head and covariance head operate on each draw, with the covariance head parameterized as rank-2 low-rank plus diagonal to guarantee a valid estimate by construction,

$$
\hat { \Sigma } _ { t , n } ^ { A } = F _ { t , n } F _ { t , n } ^ { \top } + \mathrm { d i a g } ( d _ { t , n } )\tag{3}
$$

This aleatoric term captures the market’s own conditional randomness, but says nothing about how much the model itself should be trusted on a given prediction. We capture that separately, as epistemic uncertainty, by keeping dropout active at inference and treating the spread across $S$ stochastic passes as our confidence in the forecast itself:

$$
\hat { \mu } _ { t } = \frac { 1 } { S } \sum _ { n = 1 } ^ { S } \hat { \mu } _ { t , n } , \qquad \hat { \Sigma } _ { t } ^ { A } = \frac { 1 } { S } \sum _ { n = 1 } ^ { S } \hat { \Sigma } _ { t , n } ^ { A }\tag{4}
$$

$$
\hat { \Sigma } _ { t } ^ { E } = \frac { 1 } { S - 1 } \sum _ { n = 1 } ^ { S } \left( \hat { \mu } _ { t , n } - \hat { \mu } _ { t } \right) \left( \hat { \mu } _ { t , n } - \hat { \mu } _ { t } \right) ^ { \top }\tag{5}
$$

Following the decomposition of Spears et al. (2020), originally used to size individual trades, we combine both into a single predictive covariance,

$$
\hat { \Sigma } _ { t } = \hat { \Sigma } _ { t } ^ { A } + \hat { \Sigma } _ { t } ^ { E }\tag{6}
$$

extending it from a single-signal quantity to a full $N \times N$ cross-asset covariance matrix. With $S = 1$ epistemic covariance is exactly zero, recovering a model with no explicit account of its own uncertainty.

The model is trained by maximizing the likelihood of realized returns under its own predicted aleatoric distribution, Gaussian or Student-t (full training objective in Appendix E), so risk is learned jointly with return rather than fit afterward from residuals. Before reaching the portfolio stage below, $\hat { \mu } _ { t }$ and $\hat { \Sigma } _ { t }$ are converted from log- to arithmetic-return moments, written $\mu _ { t }$ and $\Sigma _ { t }$ (full conversion in Appendix E).

## 3.4 Risk-Aware Portfolio Construction

This combined covariance is what we feed into portfolio construction. Where prior sentiment-driven work adjusts a portfolio’s expected-return input but leaves the covariance matrix at its historical value (Section 2), we feed $\Sigma _ { t }$ directly into a standard mean-variance optimizer (Markowitz, 1952) with transaction-cost regularization, full investment, and a per-asset position cap of 40%, at a risk-aversion setting of $\delta = 2 . 5$ . All reported filtered portfolios are long-only: the stock-selection signal is used only as a BUY eligibility mask, and constrained allocators impose non-negativity, full investment, and the same position cap. The optimizer is therefore penalized more heavily when the model itself is uncertain, not only when the asset has historically been volatile. The exact objective is reported in Appendix G (Equation 23).

We benchmark this against five standard alternatives: equal-weighted, risk parity (Maillard et al., 2010), hierarchical risk parity (López de Prado, 2016), and Black–Litterman and Bayesian Black– Litterman (Black and Litterman, 1992), the latter pair implemented following Lee et al. (2025)

in letting epistemic uncertainty widen the viewconfidence matrix alongside $\Sigma _ { t }$ . Standard Black– Litterman and its Bayesian variant differ only in the allocation step applied to an identical posterior: closed-form renormalized weights versus our own constrained optimizer (Equation 23) applied to the same posterior moments, isolating the effect of the allocation step from the effect of the posterior itself. Each baseline consumes the identical $\left( \mu _ { t } , \Sigma _ { t } \right)$ for a controlled comparison; full baseline formulas are given in Appendix G.

## 4 Experimental Setup

## 4.1 Data and Sample

We evaluate the pipeline on Russell 2000 equities with daily OHLCV data from October 2, 2023 through December 31, 2025 and scored news from October 1, 2023 through December 31, 2025. Macro-beta selection additionally uses a macroindicator panel spanning January 1, 2022 through January 1, 2026. All universe selection, standardization, model fitting, and early stopping use only the in-sample period ending December 31, 2024. Within that in-sample period, the first 80% of trading days are used for training and the remaining 20% for validation, with a holding-period embargo at split boundaries. The 2025 calendar year is held out for backtesting.

## 4.2 Benchmark Grid

For each selection regime, sentiment backend, target distribution, holding period, and transaction-cost setting, we train the predictive model and evaluate all six allocation methods from Section 3.4. The three selection regimes are pure alpha, pure beta, and beta. Holding periods are $H ~ \in ~ \{ 1 , 2 , 3 , 5 , 1 0 , 2 0 , 4 0 , 6 0 \}$ trading days. Transaction-cost scenarios are {0, 1, 2, 5, 10, 20, 50, 100} basis points, with a fixed 5 bps one-way slippage component in addition to the reported transaction-cost setting. The main results emphasize the conservative 100 bps setting across the full holding-period grid; Appendix D reports the corresponding 50 bps sweep, detailed best rows, sentiment-backend effects, and allocator-level diagnostics. A 30-day holding period was not part of the executed grid, so it is not reported. Performance is reported using cumulative net return, annualized return, annualized volatility, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, average turnover, total costs, hit rate, and exposure. Because the out-of-sample period is one calendar year, annualized metrics should be read as annualized summaries of a single test year rather than as estimates over many independent years.

## 5 Results and Discussion

## 5.1 Selection Regime and Holding Period

Figure 1 summarizes the conservative 100 bps benchmark. The main result is that the separated pure-alpha and pure-beta legs are more useful than the beta intersection. Pure alpha leads at 5, 10, and 60 trading days; pure beta leads at 20 and 40 days. The one-day result is cost-sensitive: pure beta is strongest at low costs, but the edge disappears under the 100 bps stress case (Appendix D).

The pattern has a direct financial interpretation. Pure alpha selects abnormal stock moves without a simultaneous macro trigger, a setting where small-cap information diffusion can be slow because coverage, liquidity, and arbitrage capacity are limited (Bernard and Thomas, 1989; Johnson and Schwartz, 2001; Shanthikumar, 2003). Pure beta instead selects stocks exposed to an abnormal macro or market-indicator move before the stock itself has moved abnormally. This makes it plausible both as a one-day lead–lag signal at low costs and as a slower 40-day macro-transmission signal, where investors gradually map rates, sector, commodity, volatility, and demand shocks into heterogeneous small-cap fundamentals (Andersen et al., 2007). Table 1 reports the best cost-sensitive configurations, and Figure 3 shows the 60-day cumulative path supporting the long-horizon pure-alpha interpretation.

This 60-day advantage does not depend on the specific 100 bps cost point emphasized elsewhere in this section. Figure 4 sweeps the best allocator within each selection regime across the full range of explicit transaction costs at the 60-day horizon, and pure alpha remains the strongest filtered regime from zero costs through 100 bps; if anything, its lead over pure beta and the beta intersection widens rather than narrows as costs rise. Because 60-day rebalancing already trades infrequently, the cost sweep moves absolute Sharpe levels only modestly. What it rules out is the possibility that pure alpha’s edge at this horizon is an accident of the particular cost assumption used in Table 1.

![](images/14ca73da45dfdba8d4630aa8bd1f130d98854c1a1cda98f3b229d8b7a6c613e7.jpg)  
Figure 3: Cumulative net-return paths for the filtered selection regimes at 100 bps transaction cost and a 60-day holding period, compared with Buy-and-Hold, an equalweight (EW) portfolio, and the Russell benchmark index. The longer-horizon results favor pure alpha, while pure beta remains positive and the beta-intersection strategy exhibits weaker performance.

## 5.2 Strategy Comparison Across Holding Horizons

The horizon comparison is not monotone because the two separated legs capture different adjustment speeds. At one day, pure beta has the clearer economic channel: the signal starts from a liquid macro, sector, rate, commodity, volatility, or index move that has already occurred, and then buys small-cap stocks with measured exposure whose own abnormal move has not yet appeared. This is a short lead–lag trade. It can work before slower constituents fully incorporate the factor shock, but it is fragile because one-day rebalancing pays the spread and opening-price noise repeatedly; under 100 bps costs the advantage disappears.

Pure alpha is more plausible at the horizons where it leads, especially 5, 10, and 60 trading days. These trades start from a stock-specific abnormal move without a simultaneous macro trigger, so the information is more likely to be firm-level: earnings revisions, financing news, contracts, governance changes, litigation, product updates, or other events not immediately summarized by broad indicators. Small-cap names have thinner analyst coverage and weaker arbitrage capacity, so this information can diffuse over several sessions rather than being fully priced by the next open (Bernard and Thomas, 1989; Johnson and Schwartz, 2001;

<table><tr><td>Cost</td><td>Selection</td><td>Sentiment</td><td>Target</td><td>H</td><td>Allocator</td><td>Net</td><td>Ann.</td><td>Sharpe</td><td>Max DD</td></tr><tr><td>20</td><td>Pure beta</td><td>GPT-4o mini</td><td>Student-t</td><td>40</td><td>RP</td><td>111.1%</td><td>102.5%</td><td>2.51</td><td>-18.3%</td></tr><tr><td>20</td><td>Pure alpha</td><td>FinBERT</td><td>Gaussian</td><td>60</td><td>HRP</td><td>47.1%</td><td>57.9%</td><td>2.10</td><td>-14.3%</td></tr><tr><td>20</td><td>Beta</td><td>FinBERT</td><td>Student-t</td><td>20</td><td>MVO</td><td>48.7%</td><td>56.2%</td><td>1.19</td><td>-21.7%</td></tr><tr><td>50</td><td>Pure beta</td><td>GPT-4o mini</td><td>Student-t</td><td>40</td><td>RP</td><td>106.9%</td><td>100.0%</td><td>2.44</td><td>-18.3%</td></tr><tr><td>50</td><td>Pure alpha</td><td>Mistral-7B</td><td>Gaussian</td><td>60</td><td>HRP</td><td>50.3%</td><td>61.6%</td><td>2.05</td><td>-15.5%</td></tr><tr><td>50</td><td>Beta</td><td>FinBERT</td><td>Student-t</td><td>20</td><td>MVO</td><td>44.7%</td><td>53.2%</td><td>1.12</td><td>-21.9%</td></tr><tr><td>100</td><td>Pure beta</td><td>GPT-4o mini</td><td>Student-t</td><td>40</td><td>RP</td><td>100.1%</td><td>95.9%</td><td>2.33</td><td>-18.3%</td></tr><tr><td>100</td><td>Pure alpha</td><td>Mistral-7B</td><td>Gaussian</td><td>60</td><td>HRP</td><td>47.8%</td><td>59.3%</td><td>1.96</td><td>-15.5%</td></tr><tr><td>100</td><td>Beta</td><td>FinBERT</td><td>Student-t</td><td>20</td><td>MVO</td><td>38.4%</td><td>48.2%</td><td>1.01</td><td>-22.4%</td></tr></table>

Table 1: Best 2025 net backtest for each filtered stock-selection regime at 20, 50, and 100 bps transaction costs, restricting the holding period to 20, 40, or 60 trading days. Each row is selected by Sharpe over sentiment backends, available return targets, and six allocation methods.

![](images/7ed225e9d5cb3b2f9b1c0f4561b8fa1ac261713178454516d85766e9a4dab8b7.jpg)  
Figure 4: Transaction-cost robustness for the filtered selection regimes at a 60-day holding period. Each line reports the best allocator within a selection regime at each explicit cost setting, averaged over sentiment backends and target distributions.

Shanthikumar, 2003). The 5- and 10-day results fit this delayed-incorporation channel. The 60-day result is consistent with a smaller set of persistent firm-specific events, where lower turnover allows fundamental repricing to dominate trading frictions. By contrast, 20 and 40 days favor pure beta, suggesting that macro repricing can remain important while investors translate aggregate shocks into heterogeneous small-cap cash-flow and discount-rate effects. The main conclusion is therefore not that one leg dominates all horizons, but that separating $S _ { S } \backslash S _ { I }$ from $S _ { I } \setminus S _ { S }$ reveals distinct firm-specific and macro-transmission mechanisms that the intersection $S _ { I } \cap S _ { S }$ obscures.

This interpretation also explains why the beta intersection is generally weaker. Requiring both the stock and indicator channels to fire at the same time is economically conservative, but it removes two useful cases: early macro spillovers before the stock has reacted, and idiosyncratic stock events without a broad factor shock. The separated regimes therefore act less like minor variants of one filter and more like different trading hypotheses. Pure beta is a lead–lag or macro-transmission strategy; pure alpha is a delayed firm-specific repricing strategy. The empirical benchmark supports this distinction because the preferred leg changes with holding period and transaction cost instead of showing a single universally dominant selection rule.

## 6 Conclusion

This paper studies small-capitalization trading with LLM-derived news sentiment, macroeconomic indicators, technical signals, and uncertainty-aware portfolio construction. The central result is that the way the tradable set is defined matters at least as much as the sentiment backend or allocator. Separating the macro-beta trigger into pure alpha and pure beta usually produces stronger Sharpe and return than requiring the stock-side and indicator-side triggers to fire together.

The horizon pattern is economically interpretable. Pure beta works best when the signal is a cross-asset transmission problem: at very short horizons, liquid macro or sector instruments can lead slower small-cap constituents; at 20–40 trading days, aggregate shocks can continue to be mapped into heterogeneous firm fundamentals. Pure alpha works best when the signal is firmspecific and slower to diffuse, especially at 5, 10, and 60 trading days. These results suggest that LLM news sentiment is most useful when embedded in a broader trading architecture that controls the opportunity set, distinguishes macro and idiosyncratic channels, and passes predicted risk into portfolio construction rather than using sentiment only as an expected-return overlay.

## Limitations

Several limitations qualify the results. First, GPT-4o mini’s October 2023 knowledge cutoff helps limit look-ahead bias, but it does not eliminate distraction from general pre-existing company knowledge. The same cutoff constraint also prevents use of newer models with later training cutoffs.

Second, long articles are summarized before scoring. This controls cost and context length, but any nuance lost during summarization cannot be recovered downstream.

Third, the evaluation is not a live deployment. It does not measure real-time retrieval, summarization, scoring latency, or timestamp precision within the trading day. Executing at the next available open may therefore be imperfect for less liquid stocks.

Finally, the horizon explanations are financial interpretations of a one-year out-of-sample benchmark, not causal identification of the underlying news events. Because the design compares many models, allocators, targets, costs, and holding periods, the best cells should be read as economically motivated evidence rather than as stable production parameters. A longer live sample, formal multiple-testing adjustment, and event-level attribution would be needed before treating any single horizon–selection pair as a persistent anomaly.

## References

Torben G. Andersen, Tim Bollerslev, Francis X. Diebold, and Clara Vega. 2007. Real-time price discovery in stock, bond and foreign exchange markets. Journal ofInternational Economics, 73(2):251–277.

Dogu Araci. 2019. Finbert: Financial sentiment analysis with pre-trained language models. arXiv preprint arXiv:1908.10063.

Victor L. Bernard and Jacob K. Thomas. 1989. Postearnings-announcement drift: Delayed price response or risk premium? Journal of Accounting Research, 27:1–36.

Fischer Black and Robert Litterman. 1992. Global portfolio optimization. Financial Analysts Journal, 48(5):28–43.

Charles W. Calomiris and Harry Mamaysky. 2019. How news and its context drive risk and returns around the world. Journal ofFinancial Economics, 133(2):299– 336.

Qizhao Chen. 2025. Sentiment-aware meanvariance portfolio optimization for cryptocurrencies. arXiv:2508.16378.

Yifei Chen, Bryan T. Kelly, and Dacheng Xiu. 2022. Expected returns and large language models. SSRN Electronic Journal. Available at SSRN: https: //ssrn.com/abstract=4416687; posted April 21, 2023; last revised February 24, 2026.

Francesco Colasanto, Luca Grilli, and Domenico Santoro. 2022. Bert’s sentiment score for portfolio optimization: A fine-tuned view in black and litterman model. Neural Computing and Applications, 34:17507–17521.

Sanjiv R. Das and Mike Y. Chen. 2007. Yahoo! for amazon: Sentiment extraction from small talk on the web. Management Science, 53(9):1375–1388.

Alexander Eliseev and Sergei Seleznev. 2026. Fake date tests: Can we trust in-sample accuracy of llms in macroeconomic forecasting? arXiv:2601.07992.

Paul Glasserman, Fulin Li, and Harry Mamaysky. 2019. Time variation in the news-returns relationship. SSRN Electronic Journal.

Paul Glasserman and Caden Lin. 2024. Assessing lookahead bias in stock return predictions generated by gpt sentiment analysis. Journal of Financial Data Science, 6(1):25–42.

Aakanksha Jadhav and Vishal Mirza. 2025. Large language models in equity markets: Applications, techniques, and insights. Frontiers in Artificial Intelligence, 8.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

W. Bruce Johnson and William C. Schwartz. 2001. Evidence that capital markets learn from academic research: Earnings surprises and the persistence of post-announcement drift. SSRN working paper.

Alireza Kargarzadeh. 2024. Developing and backtesting a trading strategy using large language models, macroeconomic and technical indicators. MSc thesis, Imperial College London.

Kemal Kirtac and Guido Germano. 2024. Sentiment trading with large language models. Finance Research Letters, 62:105227.

Youngbin Lee, Yejin Kim, Juhyeong Kim, Suin Kim, and Yongjae Lee. 2025. Llm-enhanced blacklitterman portfolio optimization. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM ’25), FinAI Workshop.

Marcos López de Prado. 2016. Building diversified portfolios that outperform out-of-sample. Journal of Portfolio Management, 42(4):59–69.

Alejandro Lopez-Lira and Yuehua Tang. 2026. Can chatgpt forecast stock price movements? return predictability and large language models. Journal of Financial Economics,forthcoming.

Sébastien Maillard, Thierry Roncalli, and Jérôme Teiletche. 2010. The properties of equally weighted risk contribution portfolios. The Journal ofPortfolio Management, 36(4):60–70.

Harry Markowitz. 1952. Portfolio selection. The Journal ofFinance, 7(1):77–91.

Advije Rizvani, Giovanni Apruzzese, and Pavel Laskov. 2026. Adversarial news and lost profits: Manipulating headlines in llm-driven algorithmic trading. arXiv:2601.13082.

Devin M. Shanthikumar. 2003. Small and large trades around earnings announcements: Does trading behavior explain post-earnings-announcement drift? SSRN working paper.

Trent Spears, Stefan Zohren, and Stephen Roberts. 2020. Investment sizing with deep learning prediction uncertainties for high-frequency eurodollar futures trading. arXiv:2007.15982.

Esmaeil Taheripour, Seyed Jafar Sadjadi, and Babak Amiri. 2025. A novel approach to portfolio construction: An application of finbert sentiment analysis and credibilistic cvar criterion. IEEE Access, 13:76775– 76795.

Paul C. Tetlock. 2007. Giving content to investor sentiment: The role of media in the stock market. The Journal ofFinance, 62(3):1139–1168.

Paul C. Tetlock, Maytal Saar-Tsechansky, and Sofus Macskassy. 2008. More than words: Quantifying language to measure firms’ fundamentals. The Journal ofFinance, 63(3):1437–1467.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

## Appendix Overview

A Stock-Selection Details. Universal liquidity selection and the pure-alpha, pure-beta, and beta trigger legs used in Section 3.1.

B Feature Definitions. The daily technical and news feature tables underlying the model in Section 3.3.

C Sentiment-Distribution Diagnostics. Distribution, class-mix, and time-series plots for the four sentiment scorers.

D Selection and Sharpe Diagnostics. Full holding-period, sentiment-backend, targetdistribution, and allocator diagnostics supporting Section 5.

E Training Objective and Return-Scale Conversion. Derivations of the Gaussian and Student-t likelihoods used to train the model in Section 3.3, along with the log-toarithmetic return conversion.

F Backtesting Procedure. Rebalancing, transaction-cost accounting, selection masking, and metric computation.

G Baseline Portfolio Formulations. Posterior and allocation formulas for the Black– Litterman, Bayesian Black–Litterman, and hierarchical risk parity baselines from Section 3.4.

## A Stock-Selection Details

Macro-beta indicator panel. The three filtered regimes use a fixed panel of 58 exogenous indicators: 50 Yahoo Finance market series and 8 FRED macroeconomic releases. The Yahoo portion covers 11 GICS sector ETFs, major world equity indices, VIX, metals, energy, agriculture, rates, crypto, and broad commodity exposure. The FRED portion covers GDP, inflation, unemployment, capacity utilization, consumer confidence, housing starts, building permits, and the federal funds rate. Daily market indicators are sourced from Yahoo Finance; lower-frequency macroeconomic releases are sourced from the public FRED CSV endpoint and forward-filled only after their returns and z-scores are computed.

Rolling windows and tail co-movement. Beta (Equation 2) is computed over a rolling 120-tradingday window for daily-frequency indicators and a 240-day window for FRED indicators; the stockside return z-score (Equation 1) uses the same 120-trading-day window, so that self-triggered and indicator-triggered abnormal moves are judged against a consistent look-back horizon. In addition to standard beta, we separately compute comovement conditional on the indicator being in the tails of its own return distribution – specifically, when its return z-score is above +1 or below −1 – to capture whether a stock’s sensitivity to an indicator changes under stressed conditions rather than only on average. The implementation requires at least 25% of the conditional regime window to be populated, avoiding a degenerate case in which the shorter price panel produces no conditional beta estimates.

Trigger sets. A stock-side trigger fires when the stock’s own return z-score satisfies $| Z _ { i } | \geq 2 ;$ the stock-side direction is the sign of $Z _ { i }$ . An indicator-side trigger fires when an indicator satisfies $| Z _ { i n d } | \ge 2$ , the stock’s absolute beta to that indicator exceeds 1, and the corresponding positivetail or negative-tail conditional beta also exceeds 1 in absolute value. If several indicators fire for the same stock-date pair, the indicator direction is accepted only when their signs are unanimous; mixed directions become HOLD. This produces two daily sets: $S _ { S }$ , the self-triggered stocks, and $S _ { I }$ , the indicator-triggered stocks.

Pure alpha, pure beta, and beta. The three filtered strategies are set operations over $S _ { S }$ and $S _ { I }$ Pure alpha is $S _ { S } \ \backslash \ S _ { I } .$ the stock moved abnormally on its own, with no contemporaneous macroindicator trigger explaining the event. Pure beta is $S _ { I } \ \backslash \ S _ { S } \colon$ a macro or market indicator moved and the stock is exposed to it, but the stock has not yet made its own abnormal move. This is the anticipatory leg. Beta is $S _ { I } \cap S _ { S } { : }$ both channels fired, and the position is allowed only if stock-side and indicator-side directions agree.

Ranking, cap, and live mask. A filtered name qualifies for the training universe only if it has at least one in-sample BUY under the selected leg. Qualified names are ranked by in-sample BUY frequency and tie-broken by mean signal strength. Pure alpha uses the absolute stock z-score as strength; pure beta and beta use the number of triggering indicators. The top 50 names are retained across all three regimes. During the 2025 backtest, the same filter creates the live allocation mask: when no BUY signal is active for a filtered regime on a rebalance date, the portfolio holds cash until the next rebalance rather than reverting to an unfiltered universe.

## B Feature Definitions

This appendix details the two feature groups that feed the multimodal network in Section 3.3: newsderived sentiment and daily price-based indicators.

News features. The scorer returns a calibrated probability over negative, neutral, and positive classes,

$$
\begin{array} { r } { p _ { \tau , i } = \big ( p _ { \tau , i } ^ { - } , p _ { \tau , i } ^ { 0 } , p _ { \tau , i } ^ { + } \big ) , ~ \sum _ { k } p _ { \tau , i } ^ { k } = 1 } \end{array}\tag{7}
$$

after clipping and renormalization if the model returns slightly off-sum probabilities. The raw directional score is

$$
s _ { \tau , i } ^ { r a w } = p _ { \tau , i } ^ { + } - p _ { \tau , i } ^ { - }\tag{8}
$$

The production sentiment variable is the normalized score $s _ { \tau , i } ^ { f i n a l }$ . First, an entity-prior correction subtracts the measured named-versus-masked bias $\delta _ { i }$ for ticker $i ,$ falling back to the group prior when the ticker prior is unavailable. Second, a strictly trailing winsorized location estimate removes recent ticker-level drift without using the current article. With $m _ { \tau , i }$ denoting the ticker-level trailing median, $m _ { \tau , g }$ the corresponding group-level trailing median, $n _ { \tau , i }$ the ticker history count, and $\kappa _ { \mathrm { s h r i n k } } = 2 0 . 0$ the shrinkage constant,

$$
\begin{array} { l } { { \tilde { m } _ { \tau , i } = w _ { \tau , i } m _ { \tau , i } + ( 1 - w _ { \tau , i } ) m _ { \tau , g } , } } \\ { { w _ { \tau , i } = \displaystyle \frac { n _ { \tau , i } } { n _ { \tau , i } + \kappa _ { \mathrm { s h r i n k } } } . } } \end{array}\tag{9}
$$

The pre-standardized surprise is therefore

$$
u _ { \tau , i } = s _ { \tau , i } ^ { r a w } - \delta _ { i } - \tilde { m } _ { \tau , i } .\tag{10}
$$

Finally, for the article’s trading day $d ( \tau )$ and group $g ( \tau )$ , the score is converted to a cross-sectional z-score,

$$
s _ { \tau , i } ^ { f i n a l } = \frac { u _ { \tau , i } - \mu _ { g ( \tau ) , d ( \tau ) } } { \operatorname* { m a x } \{ \sigma _ { g ( \tau ) , d ( \tau ) } , \sigma _ { \operatorname* { m i n } } \} } ,\tag{11}
$$

where $\mu _ { g , d }$ and $\sigma _ { g , d }$ are computed across rows in the same group and trading day, with floor $\sigma _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ . If the group-day cell contains fewer than three rows, the unstandardized surprise $u _ { \tau , i }$ is retained and the row is flagged.

Each article is mapped to the first trading day on or after publication, using a publication cutoff of 20:00 UTC: articles published at or after this time are treated as arriving on the next trading day, so that no article is attributed to a decision date that had already closed before the article appeared. For each asset-date pair, articles in the trailing $M = 3 0 \mathrm { \cdot }$ day window are embedded and grouped into story clusters by single-linkage agglomerative clustering with cosine-distance threshold $1 - \eta _ { ; }$

$$
\begin{array} { r l } & { d ( \tau , \tau ^ { \prime } ) = 1 - e _ { \tau , i } ^ { \top } e _ { \tau ^ { \prime } , i } , } \\ & { } \\ & { d ( \tau , \tau ^ { \prime } ) \leq 1 - \eta } \end{array}\tag{12}
$$

We set $\eta = 0 . 9 0$ , so two articles are merged into the same story only when their embedding cosine similarity is at least 0.90. For cluster $C _ { k }$ , the centroid is

$$
c _ { k } = \frac { \sum _ { \tau \in C _ { k } } e _ { \tau , i } } { \left\| \sum _ { \tau \in C _ { k } } e _ { \tau , i } \right\| _ { 2 } } ,\tag{13}
$$

and the representative article is the article nearest to that centroid,

$$
r _ { k } = \arg \operatorname* { m a x } _ { \tau \in C _ { k } } e _ { \tau , i } ^ { \top } c _ { k }\tag{14}
$$

Writing $K _ { i , d }$ for the resulting story count on day d, the daily sentiment feature uses only these representative news items:

$$
\bar { s } _ { i , d } = \frac { 1 } { K _ { i , d } } \sum _ { k = 1 } ^ { K _ { i , d } } { s _ { r _ { k } , i } ^ { f i n a l } }\tag{15}
$$

paired with $\log ( 1 + K _ { i , d } )$ as a coverage-volume feature. The resulting news feature group is listed in Table 2; both features are zero on days with no relevant coverage.

Daily features. The daily branch consumes a standard set of price- and volume-derived technical indicators computed from adjusted daily OHLCV data, listed in Table 3. These range from simple return measures (close-to-close, open-to-close, overnight) to volatility and momentum indicators (rolling volatility, RSI, ATR) that are common in technical trading strategies and provide the model with a market baseline against which the sentiment signal can add incremental information.

<table><tr><td>Feature</td><td>Definition or interpretation</td></tr><tr><td>mean_sent</td><td>Mean normalized sentiment of centroid-representative stories in the trailing 30-day window</td></tr><tr><td>log_n_articles</td><td> $\log ( 1 + K _ { i , d } )$  , the log count of unique representative stories in the same window</td></tr></table>

Table 2: News feature group.
<table><tr><td>Feature</td><td>Definition or interpretation</td></tr><tr><td>ret_close</td><td>Close-to-close log return</td></tr><tr><td>ret_oc</td><td>Open-to-close log return</td></tr><tr><td>ret_overnight</td><td>Previous-close-to-open log return</td></tr><tr><td>ret_adj</td><td>Adjusted-close log return</td></tr><tr><td>hl_range</td><td>Daily high-low range scaled by close</td></tr><tr><td>roll_vol</td><td>20-day rolling volatility of adjusted log returns</td></tr><tr><td>roll_mean_ret</td><td>20-day rolling mean adjusted log return</td></tr><tr><td>rel_volume</td><td>Volume divided by 20-day rolling mean volume</td></tr><tr><td>vol_change</td><td>Log-volume first difference</td></tr><tr><td>amihud</td><td>Same-day absolute adjusted log return divided by same-day dollar volume</td></tr><tr><td>mom_5, mom_20</td><td>Five-day and twenty-day adjusted-price momentum</td></tr><tr><td>rsi</td><td>14-day relative strength index scaled to [0, 1]</td></tr><tr><td>atr</td><td>14-day average true range scaled by close</td></tr><tr><td> $\mathrm { ~ z ~ } _ { - } \mathrm { r e t } , \mathrm { z } _ { - } \mathrm { v o l }$ </td><td>20-day rolling z-scores of return and log volume</td></tr><tr><td>ma_ratio</td><td>Close relative to 20-day rolling moving average</td></tr></table>

Table 3: Daily OHLCV feature group.

## C Sentiment-Distribution Diagnostics

The following figures summarize the scored news feed before it enters the portfolio model. All four backends score the same article–ticker universe where available; each distributional difference is therefore a property of the sentiment model rather than of the downstream allocator. Figure 5 reports score dispersion, Figure 6 reports class mix and confidence, and Figure 7 reports time-series calibration.

## D Selection and Sharpe Diagnostics

## D.1 Holding-Period Robustness

Figure 8 repeats the main holding-period sweep at 50 bps. The pattern is similar to Figure 1: the beta intersection is not the preferred filtered regime beyond the one-day cell, pure alpha dominates the intermediate firm-specific horizons, and pure beta dominates the 40-day macro-transmission cell. Table 4 gives the numeric 100 bps sweep behind the main-text comparison.

## D.2 Best Cost-Sensitive Configurations

Table 5 reports the pure-alpha leadership cells, while Table 6 separates the best medium- and longhorizon rows by target distribution. The broader

![](images/d1d2b6aee66d07c6f7fb99fc13623dc02a18da7856fa125c12290937d0ad4775.jpg)  
Figure 5: Distribution of net sentiment score, $p _ { + } - p _ { - } ,$ by backend. The histogram shows where each scorer concentrates its article-level mass, while the empirical CDF makes tail mass and median shifts easier to compare. FinBERT has the widest score dispersion, GPT-4o mini and Llama-2-13B skew positive, and Mistral-7B places more mass near neutral. These calibration differences explain why identical news coverage can induce different portfolio forecasts.

cost-sensitive best-configuration table is reported in the main text as Table 1.

## D.3 Cross-Horizon Strategy Diagnostics

Figure 9 shows allocator-level Sharpe dispersion at the lower transaction-cost setting, and Figure 10 summarizes the 60-day pure-alpha risk-return evidence; the corresponding cost-robustness sweep is reported in the main text as Figure 4. Figure 11 gives a representative short-horizon cumulative path, while the 60-day path is reported in the main text as Figure 3.

<table><tr><td rowspan="2">H</td><td colspan="3">Sharpe</td><td colspan="3">Annualized return</td></tr><tr><td>Pure alpha</td><td>Pure beta</td><td>Beta</td><td>Pure alpha</td><td>Pure beta</td><td>Beta</td></tr><tr><td>1</td><td>-1.28</td><td>-2.15</td><td>-0.84</td><td>-89.2%</td><td>-146.5%</td><td>-27.5%</td></tr><tr><td>2</td><td>-0.15</td><td>-2.44</td><td>-0.19</td><td>12.4%</td><td>-150.7%</td><td>-8.0%</td></tr><tr><td>3</td><td>-0.72</td><td>-1.73</td><td>-1.27</td><td>-33.8%</td><td>-102.0%</td><td>-52.5%</td></tr><tr><td>5</td><td>1.01</td><td>-0.79</td><td>-1.34</td><td>89.0%</td><td>-42.0%</td><td>-40.7%</td></tr><tr><td>10</td><td>1.11</td><td>-0.60</td><td>-0.83</td><td>49.6%</td><td>-22.2%</td><td>-22.6%</td></tr><tr><td>20</td><td>0.06</td><td>0.09</td><td>-0.15</td><td>7.7%</td><td>3.2%</td><td>0.2%</td></tr><tr><td>40</td><td>0.62</td><td>2.05</td><td>0.15</td><td>14.6%</td><td>74.7%</td><td>6.4%</td></tr><tr><td>60</td><td>1.87</td><td>1.48</td><td>0.69</td><td>57.0%</td><td>41.8%</td><td>23.9%</td></tr></table>

Table 4: Best-allocator filtered-regime sweep at 100 bps. For each holding period and selection regime, the reported value is the best allocation model after averaging that model over sentiment backends and target distributions. This is the numeric table behind the main-text interpretation.
<table><tr><td>H</td><td>Best allocator</td><td>Figure</td><td>Sharpe</td><td>Ann. return</td><td>Cum. net</td></tr><tr><td>5</td><td>Risk parity</td><td>Fig. 1</td><td>1.01</td><td>89.0%</td><td>70.0%</td></tr><tr><td>10</td><td>Hierarchical RP</td><td>Fig. 1</td><td>1.11</td><td>49.6%</td><td>43.8%</td></tr><tr><td>60</td><td>Equal weight</td><td>Fig. 1</td><td>1.87</td><td>57.0%</td><td>43.3%</td></tr></table>

Table 5: Pure-alpha leadership cells at 100 bps transaction cost.

![](images/bf0fb5d8fe586537e9d1f451ce4c85cb39b459db6a8c79a1b4743d8d14e4a9af.jpg)

![](images/c4c8a41ef3fea5774eac4813b4f1e63316a5eff4f0f38a92a4ba2c7d6447858c.jpg)  
Figure 6: Predicted class mix and confidence by sentiment backend. The left panel reports the share of article–ticker pairs assigned to positive, negative, or neutral by argmax probability; the right panel reports the CDF of the winning class probability. This distinguishes directional bias from conviction. Llama-2-13B is highly confident on many rows, while GPT-4o mini and Mistral-7B use lower-confidence assignments more often.

## D.4 Allocator Strategy Diagnostics

The allocator panels below are shown at each filtered leg’s most informative high-cost horizon rather than at a single common horizon. Figure 12 shows pure alpha at $H ~ = ~ 6 0$ , where it has its strongest 100 bps Sharpe in the holding sweep. Figure 13 shows pure beta at $H = 4 0$ , its strongest conservative configuration. Figure 14 shows the beta intersection at $H \ = \ 2 0 .$ , its best high-cost medium/long-horizon cell even though it remains weaker than the separated legs.

![](images/e0be370781cd62e103df02d6bcfccdbdff3802b42d71bb4cbc2db510ba192164.jpg)  
Figure 7: Monthly mean net sentiment by backend through the scored-news sample. Persistent level differences reveal standing model bias, while parallel monthto-month movement indicates that the backends react to the same news cycle but place it on different scales. This plot is useful for interpreting portfolio differences that arise from calibration rather than from distinct article coverage.

## D.5 Sentiment-Backend Diagnostics

The sentiment-backend panels repeat the same horizon choices by selection leg: Figure 15 for pure alpha, Figure 16 for pure beta, and Figure 17 for the beta intersection.

## E Training Objective and Return-Scale Conversion

## E.1 Training Procedure

Each training example is a synchronized crosssectional window. For decision day t, the model receives the previous $T = 3 0$ trading days of daily technical features and news features for all assets, and the supervised target is the forward H-day log return from t to t + H. Samples are assigned by decision date, and a sample is retained only if its entire forward target window stays inside the same split. Validation and test splits are additionally embargoed by at least H trading days, so no training horizon overlaps a later evaluation horizon.

<table><tr><td>Cost</td><td>Selection</td><td>Target</td><td>Sentiment</td><td>H</td><td>Allocator</td><td>Net</td><td>Ann.</td><td>Sharpe</td><td>Max DD</td></tr><tr><td>50</td><td>Beta</td><td>Gaussian</td><td>Mistral-7B</td><td>60</td><td>EW</td><td>15.6%</td><td>26.2%</td><td>0.76</td><td>-17.7%</td></tr><tr><td>50</td><td>Beta</td><td>Student-t</td><td>FinBERT</td><td>20</td><td>MVO</td><td>44.7%</td><td>53.2%</td><td>1.12</td><td>-21.9%</td></tr><tr><td>50</td><td>Pure alpha</td><td>Gaussian</td><td>Mistral-7B</td><td>60</td><td>HRP</td><td>50.3%</td><td>61.6%</td><td>2.05</td><td>-15.5%</td></tr><tr><td>50</td><td>Pure alpha</td><td>Student-t</td><td>Mistral-7B</td><td>60</td><td>EW</td><td>45.6%</td><td>56.9%</td><td>1.96</td><td>-15.2%</td></tr><tr><td>50</td><td>Pure beta</td><td>Gaussian</td><td>FinBERT</td><td>40</td><td>HRP</td><td>78.4%</td><td>79.8%</td><td>2.15</td><td>-18.5%</td></tr><tr><td>50</td><td>Pure beta</td><td>Student-t</td><td>GPT-4o mini</td><td>40</td><td>RP</td><td>106.9%</td><td>100.0%</td><td>2.44</td><td>-18.3%</td></tr></table>

Table 6: Best high-cost medium/long-horizon row within each filtered stock-selection and target-distribution pair. Rows are selected by Sharpe over sentiment backends, holding periods, transaction costs, and allocation methods.

![](images/eb76e8ead8275e5455e94417558deb036e50bb117abd684faefe19672a793c47.jpg)  
Figure 8: Holding-period sweep at 50 bps transaction cost. Each point selects the best allocator within a selection regime after averaging that allocator over sentiment backends and target distributions. The lower-cost sweep confirms that the separated pure-alpha and pure-beta legs, not the beta intersection, drive most of the useful filtered performance.

Feature standardization is fit on training lookback days only and then reused unchanged for validation and test. Warm-up NaNs from rolling indicators, undefined ratios, and no-news observations are filled only after standardization. Each branch encoder is a 1D convolutional network with 32 channels and kernel size 3, followed by a shared 64- dimensional hidden representation; dropout with rate 0.2 is applied throughout the network, including at the prediction head used for the stochastic passes described below. The network is trained with AdamW, batch size 32, learning rate 10<sup>−3</sup>, weight decay 10<sup>−5</sup>, gradient clipping at 1.0, and early stopping on validation negative log-likelihood with patience 8 over a maximum of 40 epochs. The covariance head uses a rank-2 low-rank-plusdiagonal parameterization, with a Cholesky jitter of 10<sup>−4</sup> added before decomposition for numerical stability. At inference, dropout remains active for 50 stochastic forward passes, producing the epistemic covariance term in Equation 5. All reported results use a fixed random seed of 7 for weight initialization and dropout sampling.

![](images/532dcb036e2402a6ce1057fb92fc636fa57aaa3c80142ef48a8996c65d98a707.jpg)  
Figure 9: Allocator-level Sharpe comparison for the filtered selection regimes at 50 bps transaction cost and a 40-day holding period. Points are means over sentiment backends and target distributions; horizontal bars show the min–max range across those configurations. Pure beta is the strongest filtered selection regime in this setting.

## E.2 Distributional Objective

Training likelihoods are fit only to the aleatoric distribution per draw (Section 3.3); MC-dropout is used only at inference. Under the Gaussian setting,

$$
y _ { t } ^ { ( H ) } \mid \mathcal { F } _ { t } , \omega \sim \mathcal { N } ( \hat { \mu } _ { t , n } , \hat { \Sigma } _ { t , n } ^ { A } ) ,
$$

with negative log-likelihood, for a batch of $B _ { s }$ samples and $e _ { b } = y _ { b } - \hat { \mu } _ { b }$

![](images/bb140da42bc08e95c1b27ec4facb6a24e976ef59a79daaf68179654a0b3d0984.jpg)  
Figure 10: Risk-return plane for the filtered regimes in the 100 bps, 60-day evaluation slice. Each point is a selection-regime and allocator pair averaged over sentiment backends and target distributions; color identifies the selection regime and marker shape identifies the allocator. The figure shows the long-horizon setting where pure alpha offers the strongest risk-adjusted profile among the filtered regimes.

$$
\begin{array} { r l r } {  { \mathcal { L } _ { G } = \frac { 1 } { B _ { s } } \sum _ { b = 1 } ^ { B _ { s } } \frac { 1 } { 2 } \Big [ e _ { b } ^ { \top } ( \hat { \Sigma } _ { b } ^ { A } ) ^ { - 1 } e _ { b } } } \\ & { } & { + \log | \hat { \Sigma } _ { b } ^ { A } | + N \log ( 2 \pi ) \Big ] . } \end{array}\tag{16}
$$

This is computed via the aleatoric Cholesky factor for numerical stability.

A Student-t alternative accommodates heavier tails,

$$
y _ { t } ^ { ( H ) } \mid \mathcal { F } _ { t } , \omega \sim t _ { \nu } ( \hat { \mu } _ { t , n } , S _ { t , n } ) ,
$$

where

$$
S _ { t , n } = \frac { \nu - 2 } { \nu } \hat { \Sigma } _ { t , n } ^ { A } .
$$

We fix the degrees of freedom at $\nu = 5$ across all Student-t configurations, chosen to give meaningfully heavier tails than the Gaussian case without pushing into a regime so fat-tailed that the covariance itself becomes poorly identified. The model is trained by minimizing the negative log-likelihood

![](images/d175e84f3e97433091c1dd6ac0f8334fceb0d86b7ac727a527564d8cee963072.jpg)  
Figure 11: Cumulative net-return paths for the filtered selection regimes at 100 bps transaction cost and a 10- day holding period, compared with Buy-and-Hold, an equal-weight (EW) portfolio, and the Russell benchmark index. The shorter-horizon results show pure alpha separating from pure beta and beta, consistent with firm-specific information diffusion over several trading days.

$$
\mathcal { L } _ { t } = - \frac { 1 } { B _ { s } } \sum _ { b = 1 } ^ { B _ { s } } \left[ \log \Gamma \left( \frac { \nu + N } { 2 } \right) - \log \Gamma \left( \frac { \nu } { 2 } \right) \right.\tag{17}
$$

$$
\begin{array} { l } { { \displaystyle - \frac { 1 } { 2 } \log | S _ { b } | - \frac { N } { 2 } \log ( \nu \pi ) } } \\ { { \displaystyle - \frac { \nu + N } { 2 } \log \left( 1 + \frac { q _ { b } } { \nu } \right) \biggr ] . } } \end{array}
$$

Here,

$$
\begin{array} { r } { q _ { b } = ( y _ { b } - \hat { \mu } _ { b } ) ^ { \top } S _ { b } ^ { - 1 } ( y _ { b } - \hat { \mu } _ { b } ) , } \end{array}\tag{18}
$$

is the squared Mahalanobis distance under the Student-t scale matrix.

The training objective is the selected distributional negative log-likelihood:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d i s t } } , } \end{array}\tag{19}
$$

where ${ \mathcal { L } } _ { \mathrm { d i s t } }$ is $\mathcal { L } _ { G }$ or $\mathcal { L } _ { t }$ , depending on the configured predictive distribution.

## E.3 Return-Scale Conversion

Training operates on log returns; portfolio construction requires arithmetic-return moments:

$$
\mu _ { i , t } = \exp \left( \hat { \mu } _ { i , t } + \frac { 1 } { 2 } \hat { \Sigma } _ { i i , t } \right) - 1 .\tag{20}
$$

![](images/057c2fecbe4ecc83e2b9b29666559db432af95d1d4f1d8dd26174363ac24daa8.jpg)  
Figure 12: Portfolio-strategy comparison for pure alpha at 100 bps and a 60-day holding period. This is the strongest pure-alpha horizon in the conservative sweep; equal weight is the best allocator after averaging over sentiment backends and target distributions.

![](images/962ef8c724196b94b46b2e3dde43b2618dc674e52547f21cd6bef5aa76455c20.jpg)  
Figure 13: Portfolio-strategy comparison for pure beta at 100 bps and a 40-day holding period. Risk parity is strongest in this cell, supporting the interpretation that the pure-beta leg acts as a risk-filtered macro-exposure strategy rather than only an expected-return signal.

$$
\begin{array} { r } { \Sigma _ { i j , t } = \exp \left( \hat { \mu } _ { i , t } + \hat { \mu } _ { j , t } + \frac { 1 } { 2 } \hat { \Sigma } _ { i i , t } + \frac { 1 } { 2 } \hat { \Sigma } _ { j j , t } \right) } \\ { \mathrm { ( } } \\ { \qquad \times \left( \exp ( \hat { \Sigma } _ { i j , t } ) - 1 \right) . } \end{array}\tag{21}
$$

After return-scale conversion, each covariance matrix is symmetrized and positive-definite regularized by diagonal loading: if its smallest eigenvalue is below $\epsilon = 1 0 ^ { - 4 } , ( \epsilon - \lambda _ { \operatorname* { m i n } } ) I$ is added so that the minimum eigenvalue is floored at ϵ.

## F Backtesting Procedure

For each holding period H, the trained model emits out-of-sample mean and covariance forecasts on the 2025 decision dates. These log-return moments are converted to arithmetic-return moments using the equations above, then regularized to ensure positive definiteness before allocation. Rebalance dates are every H trading days. A portfolio chosen at decision day t earns the realized daily returns from t + 1 through the next rebalance date, with weights allowed to drift between rebalances as asset prices move.

![](images/440d0d1e02d1b1ac390315d5614b8e5e372096c39f5f3a5e02ffba4c865dedbf.jpg)  
Figure 14: Portfolio-strategy comparison for the beta intersection at 100 bps and a 20-day holding period. This is the beta leg’s strongest high-cost medium/longhorizon cell, but the narrow intersection mask still leaves it behind the separated pure-alpha and pure-beta legs.

![](images/9f37be492d66666ada92717d33fe7a3957a6aee2305529a3e7769f197e9a6a8a.jpg)  
Figure 15: Sentiment-backend sensitivity for pure alpha at 100 bps and a 60-day holding period. The pure-alpha trigger isolates self-triggered abnormal stock moves with no simultaneous macro-indicator trigger, so differences across backends reflect how sentiment features interact with the strongest long-horizon firm-specific event set.

All six allocation methods consume the same forecast pair $\left( \mu _ { t } , \Sigma _ { t } \right)$ on a given rebalance date. In the pure-alpha, pure-beta, and beta regimes, forecasts are produced for the full selected crosssection, and the allocation is then masked to names with an active BUY signal under the selected trigger leg. If no name is selected on a rebalance date, the benchmark holds cash until the next rebalance rather than falling back to the unfiltered portfolio.

![](images/806e0a12293ba545927adddeb8e327994851def6dbed215aef267eade63a0e36.jpg)  
Figure 16: Sentiment-backend sensitivity for pure beta at 100 bps and a 40-day holding period. Pure beta is the anticipatory macro leg, $S _ { I } \setminus S _ { S } ;$ the plot shows whether language-derived news features add value once the eligible stocks are selected by macro-indicator exposure rather than by their own abnormal move.

![](images/71bf8ae47ca285e8cc3391646d64312690985c005c915a6c8ab13552fab96ef7.jpg)  
Figure 17: Sentiment-backend sensitivity for the beta intersection leg at 100 bps and a 20-day holding period. This is the beta leg’s strongest high-cost medium/longhorizon setting; requiring both stock-side and indicatorside triggers to agree still produces a narrower opportunity set than pure alpha or pure beta.

Transaction costs are applied at the end of each holding segment. The one-way cost rate is the explicit transaction-cost setting plus the fixed 5 bps slippage component, divided by 10,000. Let R be the gross holding-period return and let turnover be the L1 change in target weights at the rebalance. The effective one-way rate is $x = ( \mathrm { c o s t r a t e } ) \ \times$ turnover/2, and the round-trip net return is

$$
R ^ { \prime } = \frac { R ( 1 - x ) - 2 x } { 1 + x }\tag{22}
$$

The daily equity curve is reported net of these segment-level cost charges. Portfolio metrics are computed from the resulting daily net value path; cumulative gross return is retained separately by rerunning the same strategy with transaction costs and slippage set to zero.

## G Baseline Portfolio Formulations

Mean–variance optimization. The primary riskaware allocator solves a constrained mean–variance problem using the model’s predicted arithmeticreturn mean $\mu _ { t }$ and covariance $\Sigma _ { t }$ , with risk aversion $\delta = 2 . 5$ shared across all allocators that use it,

$$
\begin{array} { r } { w _ { t } ^ { M V O } = \arg \operatorname* { m a x } _ { w } \Big ( \mu _ { t } ^ { \top } w - \frac { \delta } { 2 } w ^ { \top } \Sigma _ { t } w \qquad ( } \\ { - \kappa _ { \mathrm { t u r n } } \| w - w _ { t ^ { - } } \| _ { 1 } \Big ) , } \end{array}\tag{23}
$$

subject to full investment, non-negativity, and a per-asset position cap of $w _ { i } \leq 0 . 4 0$ . The $\ell _ { 1 }$ term penalizes turnover from the previous tradable portfolio $w _ { t ^ { - } }$ . In the reported experiments $\kappa _ { \mathrm { t u r n } } = 0 . 0 $ so transaction costs are applied by the backtest cost model rather than by an optimizer-internal turnover penalty.

Black–Litterman. With $P \ = \ I .$ view vector $Q _ { t } = \mu _ { t }$ , equal-weight market portfolio $w ^ { e q }$ , and equilibrium prior $\pi _ { t } = \delta \Sigma _ { t } w ^ { e q }$ , the posterior mean and covariance follow the standard BL update,

$$
\begin{array} { r } { \mu _ { t } ^ { B L } = \left[ ( \tau \Sigma _ { t } ) ^ { - 1 } + P ^ { \top } \Omega _ { t } ^ { - 1 } P \right] ^ { - 1 } } \\ { \times \left[ ( \tau \Sigma _ { t } ) ^ { - 1 } \pi _ { t } + P ^ { \top } \Omega _ { t } ^ { - 1 } Q _ { t } \right] , } \end{array}\tag{24}
$$

$$
\Sigma _ { t } ^ { B L } = \Sigma _ { t } + \left[ ( \tau \Sigma _ { t } ) ^ { - 1 } + P ^ { \top } \Omega _ { t } ^ { - 1 } P \right] ^ { - 1 }\tag{25}
$$

with $\Omega _ { t } = \mathrm { d i a g } ( \Sigma _ { 1 1 , t } , \dots , \Sigma _ { N N , t } )$ and $\tau = 0 . 0 5$ controlling prior uncertainty throughout. The resulting portfolio is

$$
\tilde { w } _ { t } ^ { B L } = ( \delta \Sigma _ { t } ^ { B L } ) ^ { - 1 } \mu _ { t } ^ { B L } , \qquad w _ { t } ^ { B L } = \frac { \tilde { w } _ { t } ^ { B L } } { { \bf 1 } ^ { \top } \tilde { w } _ { t } ^ { B L } }\tag{26}
$$

falling back to equal weight if this normalization is unsafe.

Bayesian Black–Litterman. Treating the expected-return vector $\theta _ { t }$ as latent, with prior $\begin{array} { r l r } { \theta _ { t } } & { { } \sim } & { \mathcal { N } ( \pi _ { t } , \tau \Sigma _ { t } ) } \end{array}$ and view model $Q _ { t } \ \mathrm { ~  ~ { ~ \vert ~ } ~ } \theta _ { t } \ \sim \ \mathcal { N } ( P \theta _ { t } , \Omega _ { t } )$ , the posterior precision, covariance, and mean are

$$
\Lambda _ { t } ^ { p o s t } = ( \tau \Sigma _ { t } ) ^ { - 1 } + P ^ { \top } \Omega _ { t } ^ { - 1 } P ,\tag{27}
$$

$$
V _ { t } ^ { p o s t } = ( \Lambda _ { t } ^ { p o s t } ) ^ { - 1 } ,\tag{28}
$$

$$
m _ { t } ^ { p o s t } = V _ { t } ^ { p o s t } \bigl [ ( \tau \Sigma _ { t } ) ^ { - 1 } \pi _ { t } + P ^ { \top } \Omega _ { t } ^ { - 1 } Q _ { t } \bigr ]\tag{29}
$$

with posterior predictive covariance $\begin{array} { r l } { ~ } & { { } \Sigma _ { t } ^ { p r e d , p o s t } = } \end{array}$ $\Sigma _ { t } + V _ { t } ^ { p o s t }$ (numerically identical to $\Sigma _ { t } ^ { B L }$ above). These moments are then substituted for $\left( \mu _ { t } , \Sigma _ { t } \right)$ in the constrained optimization of Equation 23, subject to the same investment and position-cap constraints.

Hierarchical Risk Parity. Covariance is first converted to a correlation-based distance,

$$
\begin{array} { r } { \rho _ { i j , t } = \frac { \sum _ { i j , t } } { \sqrt { \sum _ { i i , t } \sum _ { j j , t } } } , \qquad d _ { i j , t } = \sqrt { \frac { 1 - \rho _ { i j , t } } { 2 } } } \end{array}\tag{30}
$$

and hierarchical clustering orders assets so that similar assets are adjacent. For a cluster $C$ , inversevariance weights and cluster variance are

$$
\begin{array} { r l } & { \omega _ { i } ^ { C } = \frac { 1 / \Sigma _ { i i , t } } { \sum _ { j \in C } { 1 / \Sigma _ { j j , t } } } , } \\ & { \sigma _ { C } ^ { 2 } = ( \omega ^ { C } ) ^ { \top } \Sigma _ { C , t } \omega ^ { C } } \end{array}\tag{31}
$$

For sibling clusters $L$ and $R ,$ capital assigned to the left cluster is

$$
\alpha _ { L } = \frac { \sigma _ { R } ^ { 2 } } { \sigma _ { L } ^ { 2 } + \sigma _ { R } ^ { 2 } } , \qquad \alpha _ { R } = 1 - \alpha _ { L }\tag{32}
$$

applied recursively down the cluster hierarchy.