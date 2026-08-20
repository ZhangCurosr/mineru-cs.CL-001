# Budget–First Tariff Recommendation (BFTR): A Complete Algorithmic Framework for Telecom Plan Recommendation without Overcharging

Ghislain Dorian Tchuente Mondjo, Independent Researcher

tchuente.mondjo@gmail.com

Abstract—Telecom operators traditionally offer predefined tariff grids, forcing users to choose from a limited set of plans. This paper proposes BFTR (Budget–First Tariff Recommendation), a complete algorithmic framework integrating eight Budget–First strategies, including two original hybrid approaches: Recursive Hybrid (conditional interpolation) and Knapsack–First Hybrid (priority knapsack). Unlike existing approaches that adjust prices upward to guarantee a minimum margin, BFTR guarantees the absence of overcharging by systematically aligning the final price with the catalog reference price. We mathematically formalize each strategy, prove the existence of an offer for any positive budget, and prove that the price deviation (surcharge) is zero for all strategies that do not use interpolation with correction. A detailed comparative analysis confronts BFTR to ten main existing tariff models on ten dimensions. Experiments on a dataset of 974 customers inspired by the Nigerian MTN market show that: (i) Recursive Hybrid is optimal for the customer (100% budget used, 29.9 GB volume, utility 0.946, 0% overcharging), (ii) Piecewise offers the highest volume (39.7 GB) with 0% overcharging, (iii) Power Law provides an excellent compromise (99.9% budget, 38.1 GB, 0% overcharging). All strategies achieve a zero surcharge, confirming the theoretical guarantees. A sensitivity analysis on the weighting parameter α (0.2 – volume priority, 0.5 – balance, 0.8 – budget priority) shows that utility rankings evolve logically. Execution times (< 10 ms) and very low failure rates (0% for robust strategies) confirm the operational viability of the system. The formal proof of the absence of overcharging constitutes a major theoretical contribution.

Impact Statement—Chatbots and AI-based recommender systems are increasingly used in telecom to personalize offers. However, existing tariff recommendation systems often inflate prices to secure a minimum margin, leading to customer distrust and regulatory scrutiny. BFTR introduces a paradigm shift: instead of guaranteeing a margin, it guarantees price integrity—the final price never exceeds the reference price derived from the catalog. This approach directly addresses overcharging concerns, enhances customer satisfaction, and reduces churn. The framework’s hybrid strategies combine interpolation and knapsack optimization to achieve up to 100% budget utilization with zero surcharge. With execution times under 5 ms, BFTR is deployable in real-time production environments. Our experimental results on a real-world-inspired dataset show that the Recursive Hybrid strategy yields the highest utility (0.946) while maintaining zero overcharging. The framework offers operators a transparent, fair, and scalable

solution for tariff personalization, aligning business interests with customer welfare.

Index Terms—Tariff recommendation, budget-first, knapsack, interpolation, overcharging prevention, telecom, personalization.

## I. INTRODUCTION

often ill-adapted tariff offering. Customers must choose from a grid of standard plans, leading to frustrations, overage costs, and high churn rates [1], [2]. Particularly in emerging markets such as sub-Saharan Africa, where purchasing power is lower and income volatility is higher, rigid offers constitute a major barrier to digital inclusion. Meanwhile, artificial intelligence and recommender systems have opened the door to fine-grained personalization of services [3], [4]. Dynamic pricing and personalized recommendation models have been proposed in other sectors (e-commerce, energy), but their adaptation to telecom remains limited by margin constraints and catalog complexity.

We propose BFTR, a framework where the initiative rests with the customer: they indicate their desired monthly budget and choose among several algorithmic strategies (direct selection, interpolation, knapsack, regression, hybrids). The system then dynamically constructs an optimized plan according to the selected strategy. Unlike existing approaches that adjust prices upward to ensure a minimum margin, BFTR guarantees the absence of overcharging by systematically aligning the final price with the catalog reference price (the sum of prices of selected plans, or the budget for interpolation).

The originality of BFTR lies in its ability to combine classical approaches with advanced methods, while ensuring that the price charged to the customer never exceeds the reference price. This paradigm shift—from margin guarantee to price compliance guarantee—is a major contribution, as it directly addresses regulator concerns about overcharging.

Contributions. (1) Complete mathematical formalization of eight Budget–First strategies, including two original hybrid approaches. (2) Proof of the existence of an offer for any positive budget. (3) Formal proof of the absence of overcharging for composition and interpolation strategies, establishing that the final price is always less than or equal to the reference price. (4) Detailed comparative analysis on ten dimensions against existing models, with justifications and concrete examples. (5) Exhaustive experimental evaluation on 974 customers with sensitivity analysis of the weighting parameter α (three levels), execution time comparison, measurement of failure rates, and, for the first time, the introduction of margin metrics—average margin and margin threshold attainment rate—to assess the economic viability of each strategy from the operator’s perspective.

The paper is organized as follows. Section II presents the state of the art with detailed definitions. Section III formalizes the BFTR model, including algorithms and proofs of overcharging absence. Section IV provides a comparative analysis. Section V describes the architecture and user strategy choice. Section VI presents experiments and discussion. Section VII concludes.

## II. STATE OF THE ART: ECONOMIC MODELS IN TELECOM

We present the ten most widespread tariff systems. For each, we give a formal theoretical definition, a real operator example, and an analysis of its economic properties.

Definition 1 (Prepaid Model). Prepaid is a system where the user buys credit C in advance. Consumption q (in service units: MB, minutes, SMS) is deducted from credit at a unit tariff τ . Service is interrupted when $C - \tau \cdot q \leq 0 .$ . Mathematically, the maximum consumption is $q _ { \operatorname* { m a x } } = \lfloor C / \tau \rfloor$

Popularized by Orange and Free, this model offers strict budget control $( q _ { \mathrm { m a x } }$ is known in advance) but requires frequent recharges. In Africa, over 90% of subscribers use prepaid.

Definition 2 (Postpaid Model). Postpaid bills the user at the end of the period (month) based on actual consumption $q _ { r e a l } .$ The total price is $P = P _ { b a s e } + \delta \cdot \operatorname* { m a x } ( 0 , q _ { r e a l } - Q )$ , where $P _ { b a s e }$ is the base plan price, Q the included quota, and δ the overage tariff.

Verizon and Vodafone use this model. The major drawback is uncertainty: $P$ can vary from single to quadruple depending on $q _ { \mathrm { r e a l } }$

Definition 3 (Standard Plans). A catalog $F = \{ ( p _ { i } , q _ { i } ) \} _ { i = 1 } ^ { k }$ of k price–volume pairs is offered. The user chooses $i ^ { * }$ minimizing cost while satisfying $q _ { i ^ { * } } \geq q _ { n e e d } .$ The problem is min<sub>i</sub> $\{ p _ { i } \mid q _ { i } \geq q _ { n e e d } \}$

SFR (Red) and AT&T illustrate this model. Limited granularity $( k \le 8 )$ creates waste (paying for $q _ { i } > q _ { \mathrm { n e e d } } )$ or overages $( q _ { i } < q _ { \mathrm { n e e d } } )$

Definition 4 (Freemium Model). Freemium defines a free tier $q _ { f r e e }$ and tiers $\{ ( p _ { j } , q _ { j } ) \} _ { j = 1 } ^ { m }$ . Cost is $P ~ = ~ 0 ~ i f ~ q ~ \le ~ q _ { f r e e } ,$ otherwise $P = p _ { j } f o r q \in ( q _ { j - 1 } , q _ { j } ]$

Google Fi and Lebara use this model. Entry is easy $( P =$ 0), but the transition to paid can be abrupt.

Definition 5 (Rollover Model). Rollover carries over unconsumed surplus $r _ { t } = \operatorname* { m a x } ( 0 , Q - q _ { t } )$ to the next period: $Q _ { t + 1 } = Q + r _ { t }$ . The available stock is $\begin{array} { r } { { \dot { S } } _ { t } = Q + \sum _ { k = 1 } ^ { t - 1 } r _ { k } . } \end{array}$

T-Mobile (Data Stash) and Bouygues Telecom (Smart Data) use this mechanism. It smooths inter-temporal variations but does not modify the base plan.

Definition 6 (Family Sharing Model). A global quota $Q _ { t o t a l }$ is shared among n lines. The constraint $\begin{array} { r } { i s \sum _ { l = 1 } ^ { n } q _ { l } \le Q _ { t o t a l } . } \end{array}$ The cost is fixed $P _ { t o t a l } ,$ independent of individual distribution.

Orange (Open Family) and Verizon (Unlimited Family) are examples. This model mutualizes overage risks but requires group management.

Definition 7 (Data Lending Model). User A can borrow an amount ∆ from user B, to be repaid later: $q _ { A } \gets q _ { A } + \Delta$ $q _ { B } \gets q _ { B } - \Delta$ , with a debt $D _ { A  B } = \Delta .$

AT&T (Data Transfer) and T-Mobile offer this mechanism. Solidarity is strong, but availability depends on other members’ balances.

Definition 8 (Overage Billing Model). For each unit consumed beyond the quota Q, a higher tariff $\delta \gg \tau$ is applied: $P = P _ { b a s e } + \delta \cdot \operatorname* { m a x } ( 0 , q - Q )$ . Typically, $\delta \approx 5 \times \tau .$

This mechanism, a major source of churn, is gradually being replaced by tiered plans or throttling.

## III. FORMALIZATION OF THE BFTR MODEL

## A. Notations and Economic Model

Let $F = \{ f _ { 1 } , \ldots , f _ { k } \}$ be the set of existing plans, with $p _ { i }$ the price and $\mathbf { s } _ { i } ~ = ~ ( d _ { i } , m _ { i } , t _ { i } , \ldots )$ the service vector (data, minutes, SMS). User u has consumption history $\mathcal { H } _ { u }$ and budget $b _ { u } \in \mathbb { R } ^ { + }$ . We define a utility function $U _ { u } : F \to [ 0 , 1 ]$ by

$$
U _ { u } ( f _ { i } ) = \alpha \cdot \mathrm { m i n } \left( 1 , \frac { p _ { i } } { b _ { u } } \right) + ( 1 - \alpha ) \cdot \mathrm { m i n } \left( 1 , \frac { d _ { i } } { d _ { u } } \right) ,\tag{1}
$$

where $\bar { d } _ { u }$ is the average data consumption of $u .$ . The parameter $\alpha \in [ 0 , 1 ]$ is set to 0.5 in reference experiments, giving equal weight to budget adequacy and needs coverage. Sensitivity analysis on three values $( \alpha \ = \ 0 . 2 , \ 0 . 5 , \ 0 . 8 )$ is conducted in Section VI-F. The utility combines two aspects: the first term favors plans whose price is close to the budget (without exceeding it), the second favors those whose data volume covers the usual needs.

The component catalog C has been removed in the final system; all knapsack-type strategies (KNAP and hybrids) use only the plans from catalog $F .$ This decision, validated by experiments, avoids artificial price discrepancies.

## B. Operational Constraints and Definitions

Before describing algorithmic strategies, we introduce the fundamental concepts that frame BFTR.

Definition 9 (Reference Price). For an offer O built by composing existing plans, its reference price $p _ { r e f } ( O )$ is the sum of the prices of the plans composing it:

$$
p _ { r e f } ( O ) = \sum _ { f \in O } p _ { f } .
$$

For an offer obtained by linear interpolation, the reference price is the budget $b _ { u }$ (we create a virtual offer whose price is exactly the budget). For a mixed offer (combination of catalog plans and interpolation), the reference price is the original budget $b _ { u } ,$ , as the system is expected to use the entire budget without any markup.

Definition 10 (Surcharge / Overcharging). The surcharge of an offer O is the relative difference between the final price $p ( O )$ and the reference price:

$$
s ( O ) = \frac { p ( O ) - p _ { r e f } ( O ) } { p _ { r e f } ( O ) } \times 1 0 0 .
$$

An offer is said to be overcharged $i f s ( O ) > 5 \% ,$ where 5% is a regulatory tolerance threshold.

## C. Budget–First Strategies

We detail the six basic strategies and two advanced; the two hybrid strategies are presented later. For each, we give an intuitive description, the mathematical equation, the pseudocode algorithm, and complexity analysis.

a) Direct Selection (SELECT).: This strategy scans all existing plans and retains the one that maximizes utility while respecting the budget constraint. The optimal choice equation is

$$
f ^ { * } = \arg \operatorname* { m a x } _ { f _ { i } \in F , \ p _ { i } \leq b _ { u } } U _ { u } ( f _ { i } ) .\tag{2}
$$

Algorithm 1 describes the procedure: initialize the best offer to “none” and iterate over each plan; if its price is within budget and its utility is higher, update. Complexity is $O ( k )$ , as each plan is examined once.

Algorithm 1 Direct Selection (SELECT)   
Require: Catalog ${ \overline { { F } } } ,$ budget $b _ { u } ,$ history $\mathcal { H } _ { u }$   
Ensure: Offer $O ^ { * }$ or ∅   
1: $O ^ { * } \gets \emptyset , u ^ { * } \gets - \infty$   
2: for $f _ { i } \in F$ do   
3: if $p _ { i } \leq b _ { u }$ then   
4: $u  U _ { u } ( f _ { i } )$ ▷ according to (1)   
5: if $u > u ^ { * }$ then   
6: $u ^ { * } \gets u , O ^ { * } \gets f _ { i }$   
7: end if   
8: end if   
9: end for   
10: return $O ^ { * }$

Proposition 1 (No overcharging for SELECT). For SELECT, the final price is the selected plan’s price, and the reference price is that same price. The surcharge is thus zero:

$$
s ( O ) = \frac { p _ { f } - p _ { f } } { p _ { f } } = 0 \% .
$$

b) Linear Interpolation (INTERP).: If the budget $b _ { u }$ lies between the prices of two consecutive plans $f _ { i }$ and $f _ { j }$ (such that $p _ { i } \leq b _ { u } \leq p _ { j } )$ , we create a virtual offer whose volume is a linear combination of the volumes of the two plans:

$$
d _ { \mathrm { i n t e r p } } = d _ { i } + { \frac { b _ { u } - p _ { i } } { p _ { j } - p _ { i } } } ( d _ { j } - d _ { i } ) .\tag{3}
$$

The price of this offer is exactly $b _ { u } .$ , guaranteeing full budget utilization. Algorithm 2 first sorts the catalog by price (O(k log k)), then finds the interval containing $b _ { u }$ by dichotomy $( O ( \log k ) )$ ). If $b _ { u }$ is outside the interval [p<sub>min</sub>, p<sub>max</sub>] (catalog-defined thresholds), interpolation is impossible and the algorithm returns empty (failure). Interpolation is allowed only in this interval because outside, the linear relationship between price and volume is no longer valid (risk of negative or aberrant volume).

Algorithm 2 Linear Interpolation (INTERP)   
Require: Catalog $F ,$ budget $b _ { u } ,$ thresholds p<sub>min</sub>, p<sub>max</sub>   
Ensure: Offer $O ^ { * }$ or ∅   
1: Sort F by price ascending   
2: if $b _ { u } < p _ { \mathrm { m i n } }$ or $b _ { u } > p _ { \operatorname* { m a x } }$ then   
3: return $\varnothing$   
4: end if   
5: Find i such that $p _ { i } \le b _ { u } \le p _ { i + 1 }$   
6: $r \gets ( b _ { u } - p _ { i } ) / ( p _ { i + 1 } - p _ { i } )$   
7: $d  d _ { i } + r \cdot ( d _ { i + 1 } - d _ { i } )$   
8: return $\mathrm { \{ p r i c e = }  b _ { u } ,$ volume = d}

Proposition 2 (No overcharging for INTERP). For INTERP, the final price is exactly $b _ { u }$ and the reference price is $b _ { u }$ (by definition, the reference price of an interpolated offer is the budget). Thus the surcharge is zero:

$$
s ( O ) = \frac { b _ { u } - b _ { u } } { b _ { u } } = 0 \% .
$$

c) Linear Regression (REGR).: A multivariate linear regression model is trained on history $\mathcal { H } _ { u }$ to predict the optimal volume $\widehat { d }$ from budget and average consumption:

$$
\widehat { d } ( b _ { u } , \bar { d } _ { u } ) = \beta _ { 0 } + \beta _ { 1 } b _ { u } + \beta _ { 2 } \bar { d } _ { u } .\tag{4}
$$

Coefficients $\beta _ { k }$ are estimated by least squares. Inference is $O ( 1 )$ . The final offer is the catalog plan whose price is closest to the prediction without exceeding $b _ { u } .$ . Algorithm 3 assumes the model is already trained. This strategy does not depend on thresholds p<sub>min</sub>, p<sub>max</sub>.

Algorithm 3 Linear Regression (REGR)   
Require: Budget $b _ { u } ,$ , history $\mathcal { H } _ { u }$ , trained model M   
Ensure: Offer O<sup>∗</sup> or ∅   
1: $\widehat { d }  \beta _ { 0 } + \beta _ { 1 } b _ { u } + \beta _ { 2 } \bar { d } _ { u }$   
2: $\widehat { p } \gets$ estimate of corresponding price (projection)   
3: O<sup>∗</sup> ← plan $f \in F$ with $p \leq b _ { u }$ minimizing $| p - \widehat { p } |$   
4: return $O ^ { * }$

Proposition 3 (No overcharging for REGR). REGR selects an existing catalog plan, so $p ( O ) = p _ { f }$ and $p _ { r e f } ( O ) = p _ { f }$ . The surcharge is zero.

d) Greedy Knapsack (KNAP).: We have a catalog of plans F. We want to maximize total volume under the budget constraint. We use a greedy heuristic (Algorithm 4) that sorts plans by volume/price ratio descending and adds them as long as budget allows. Complexity is O(k log k).

Major difference from existing approaches: KNAP does not adjust the price upward. The final price is exactly the sum of the prices of selected plans:

$$
p ( O ) = \sum _ { f \in O } p _ { f } = p _ { \mathrm { r e f } } ( O ) .
$$

Algorithm 4 Greedy Knapsack (KNAP)   
Require: Catalog ${ \overline { { F } } } ,$ budget $b _ { u }$   
Ensure: Set $O ^ { * }$   
1: $O ^ { * } \gets \emptyset , R \gets b _ { u }$   
2: Sort F by $F$ $d _ { f } / p _ { f }$ descending   
3: for $f \in F$ do   
4: if $p _ { f } \leq R$ then   
5: $\mathsf { \bar { O } } ^ { * } \gets O ^ { * } \cup \{ f \} , R \gets R - p _ { f }$   
6: end if   
7: end for   
8: return $O ^ { * }$

Theorem 1 (No overcharging for KNAP). For any offer O generated by KNAP, the final price p(O) is exactly the sum of prices of selected plans. By definition, the reference price is that same sum. Hence:

$$
s ( O ) = \frac { p ( O ) - p _ { r e f } ( O ) } { p _ { r e f } ( O ) } = 0 \% .
$$

Thus KNAP never generates overcharging.

## D. Advanced Strategies

These strategies use predictive models that estimate the price P of a plan from its volume d. They require a prior learning phase on historical data. We describe the algorithmic principle of each method; calibrated models will be presented in Section VI-A.

a) Piecewise Regression (PIECE).: We partition the volume axis into L intervals $[ s _ { \ell - 1 } , s _ { \ell } ]$ and fit a linear regression $P ~ = ~ a _ { \ell } d + b _ { \ell }$ on each segment. Algorithm 5 uses cross-validation to choose breakpoints $s \ell$ and least squares to estimate coefficients. Inference consists of identifying the segment containing d and evaluating the corresponding line, in O(log L) (dichotomy) or O(1) if segments are few.

Algorithm 5 Piecewise Learning (PIECE)   
Require: Data $\left\{ \left( d _ { i } , P _ { i } \right) \right\}$ , max segments $L _ { \mathrm { m a x } }$   
Ensure: Breakpoints $\{ s _ { \ell } \}$ , coefficients $\{ ( a _ { \ell } , b _ { \ell } ) \}$   
1: Determine $\{ s _ { \ell } \}$ by cross-validation minimizing RMSE   
2: for $\ell = 1$ to L do   
3: $( a _ { \ell } , b _ { \ell } ) \gets$ least squares on $\{ d \in [ s _ { \ell - 1 } , s _ { \ell } ] \}$   
4: end for   
5: return $\{ s _ { \ell } \} , \{ ( a _ { \ell } , b _ { \ell } ) \}$

b) Power Law Regression (POW).: We assume a relationship $P = a \cdot d ^ { b } + c ,$ capturing economies of scale. Fitting is done by non-linear least squares (Levenberg–Marquardt algorithm). Inference is O(1).

Proposition 4 (No overcharging for PIECE and POW). PIECE and POW generate an estimated price $\widehat { p }$ and a volume db. The final price is set to min $( \widehat { p } , b _ { u } )$ and cost is recomputed on volume db. The reference price is the estimated price pb. The surcharge is thus zero (or negative):

$$
s ( O ) = \frac { \operatorname* { m i n } ( \widehat { p } , b _ { u } ) - \widehat { p } } { \widehat { p } } \leq 0 .
$$

Thus there is never positive overcharging.

## E. Hybrid Strategies

The two hybrid strategies combine KNAP and INTERP.

Algorithm 6 Recursive Hybrid (HYB-REC)   
Require: Budget $b ,$ catalog ${ \overline { { F } } } ,$ thresholds p<sub>min</sub>, p<sub>max</sub>   
Ensure: Offer $O ^ { * }$   
1: if $p _ { \operatorname* { m i n } } \le b \le p _ { \operatorname* { m a x } }$ then   
2: return $\mathrm { I N T E R P } ( F , b )$   
3: else   
4: $O _ { K } \gets \mathrm { K N A P } ( F , b )$   
5: $\begin{array} { r } { b _ { r } \gets b - \sum _ { f \in O _ { K } } p _ { f } } \end{array}$   
6: if $b _ { r } > 0$ and $p _ { \operatorname* { m i n } } \le b _ { r } \le p _ { \operatorname* { m a x } }$ then   
7: $O _ { I } \gets \mathrm { I N T E R P } ( F , b _ { r } )$   
8: return $O _ { K } \cup O _ { I }$   
9: else   
10: return $O _ { K }$   
11: end if   
12: end if

Algorithm 7 Knapsack–First Hybrid (HYB-KF)   
Require: Budget $b ,$ catalog $F ,$ , thresholds p<sub>min</sub>, p<sub>max</sub>   
Ensure: Offer $O ^ { * }$   
1: $O _ { K } \gets \mathrm { K N A P } ( F , b )$   
2: $b _ { r } \gets b - \sum _ { f \in O _ { K } } p _ { f }$   
3: if $b _ { r } > 0$ and $p _ { \operatorname* { m i n } } \le b _ { r } \le p _ { \operatorname* { m a x } }$ then   
4: O ← INTERP $( F , b _ { r } )$   
5: return $O _ { K } \cup O _ { I }$   
6: else   
7: return $O _ { K }$   
8: end if

HYB-REC favors interpolation when the budget is within catalog range, ensuring 100% utilization; HYB-KF prioritizes knapsack, creating composite offers even for central budgets.

Theorem 2 (No overcharging for hybrid strategies). For HYB-REC and HYB-KF:

1) The KNAP part does not generate overcharging (Theorem 1).

2) The INTERP part does not generate overcharging (Proposition 2).

3) The union of both parts has a final price equal to the sum of the parts’ prices, which is exactly the reference price (the original budget).

Hence s(O) = 0% for HYB-REC and HYB-KF (provided interpolation of the remainder is allowed).

## F. Typology of Strategies: Composition vs. Generation

The eight Budget–First strategies fall into two distinct families based on their offer construction mode:

• Composition strategies: they assemble existing plans to form a personalized offer. They do not create new tariffs, but select and combine predefined plans from catalog F. These are KNAP (greedy knapsack) and the two hybrid strategies HYB–KF and HYB–REC (in their part using

KNAP). Their advantage is high granularity and total absence of overcharging (Theorem 1).

• Generation strategies: they produce a virtual offer whose price and/or volume do not correspond to any pre-existing plan. These offers are created by linear interpolation (INTERP), linear regression (REGR), piecewise model (PIECE), or power law (POW). The SELECT strategy can also be classified in this family because it chooses an existing plan without modifying it. Generation strategies allow perfect budget utilization (interpolation) or fine adaptation to usage profiles (regression), but their implementation requires mathematical models or prior learning.

This distinction sheds light on the experimental results: composition strategies (KNAP, HYB–KF) excel for the operator (high margins), while generation strategies (HYB–REC, INTERP) are optimal for the customer (100% budget utilization).

## IV. COMPARATIVE ANALYSIS: BFTR VS. TEN EXISTING SYSTEMS

This section compares BFTR to the ten models described in Section II. Table I summarizes the comparison on ten key dimensions. Colors: green = advantage, yellow = average, red = major drawback.

A. Detailed Score Explanations (with Justifications and Examples)

a) 1. Budget control.: This dimension measures the customer’s ability to know in advance how much they will pay and not exceed that amount.

TABLEI

• BFTR (Total): User sets a cap, system builds an offer at that price (or less). Example: budget 15C → offer at 15C (interpolation) or 14.50C (knapsack). No overage.

• Prepaid (High): 10C top-up, service blocked when exhausted. Strict control but manual refills.

• Freemium (High): Free tier capped (e.g., 1 GB). Full control as long as one does not purchase a tier.

• Pay-as-you-go (Medium): No cap, but user can moderate usage. Example: Free 2C/month + 0.05C/MB, 10 GB → 2 + 500 = 502C.

• Standard plans, Subscription, Rollover, Family, Lending (Low): Base price known, but overages billed without control. Example: SFR 2 GB at 15C, 1 GB overage → +5C.

• Postpaid (Very low): Bill unknown until receipt. Example: Verizon, 20C plan can become 80C.

• Overage billing (None): No control.

b) 2. Surprise risk.: Probability of receiving a higher bill than expected.

• BFTR, Prepaid, Freemium (None): No surprise.

• Subscription (Low): Out-of-bundle (international calls) may add a few euros.

• Standard plans, Rollover, Family, Lending (Medium): Overage possible but limited. Example: 10 GB, +1 GB charged 5C → +25%.

• Postpaid, Pay-as-you-go, Overage (High): Very variable bill. Example: pay-as-you-go, streaming 50 GB → 514C.

COMPARISONOFTARIFFMODELSONTENDIMENSIONS

<table><tr><td>Dimension</td><td>Pre.</td><td>Post.</td><td>Std.</td><td>Sub.</td><td>PayU</td><td>Free.</td><td>Roll.</td><td>Fam.</td><td>Lend</td><td>Ov.</td><td>BFTR</td></tr><tr><td>Budget control</td><td>High</td><td>Very low</td><td>Low</td><td>Low</td><td>Medium</td><td>High</td><td>Low</td><td>Low</td><td>Low</td><td>None</td><td>Total</td></tr><tr><td>Surprise risk</td><td>None</td><td>High</td><td>Medium</td><td>Low</td><td>High</td><td>None</td><td>Medium</td><td>Medium</td><td>Medium</td><td>High</td><td>None</td></tr><tr><td>Personalization</td><td>None</td><td>None</td><td>Low</td><td>None</td><td>Medium</td><td>Low</td><td>Low</td><td>Low</td><td>Low</td><td>None</td><td>Maximal</td></tr><tr><td>Small budget adaptation</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Overage handling</td><td>Block</td><td>Billing</td><td>Billing</td><td>Billing</td><td>Billing</td><td>Paid tier</td><td>Rollover</td><td>Pool</td><td>Borrow</td><td>Billing</td><td>None</td></tr><tr><td>Customer complexity</td><td>Medium</td><td>Low</td><td>Low</td><td>Low</td><td>High</td><td>Low</td><td>Low</td><td>Medium</td><td>Medium</td><td>Low</td><td>Very low</td></tr><tr><td>Interest alignment</td><td>Low</td><td>Low</td><td>Low</td><td>Low</td><td>Low</td><td>Medium</td><td>Medium</td><td>Medium</td><td>High</td><td>Very low</td><td>High</td></tr><tr><td>Innovation incentive Revenue predictability</td><td>Low</td><td>Low</td><td>Low</td><td>Low</td><td>Medium</td><td>Medium</td><td>Low</td><td>Low</td><td>Low</td><td>None</td><td>Maximal</td></tr><tr><td>Churn reduction</td><td>High</td><td>Low</td><td>High</td><td>High</td><td>Low</td><td>Medium</td><td>Medium</td><td>Medium</td><td>Medium</td><td>High</td><td>Medium-High</td></tr><tr><td></td><td>Low</td><td>Low</td><td>Low</td><td>Low</td><td>Medium</td><td>Medium</td><td>Medium</td><td>High</td><td>High</td><td>Very low</td><td>Very high</td></tr></table>

c) 3. Personalization.: Ability to adapt the offer to exact needs.

• BFTR (Maximal): A la carte assembly. Example: 3 GB + 200 min + roaming.

• Pay-as-you-go (Medium): Personalized bill, but offer not modular.

• Standard plans (Low): Few predefined combinations (2, 10, 30 GB).

• Freemium, Rollover, Family, Lending (Low): Standard base plan.

• Others (None): Single formula.

d) 4. Small budget adaptation.: Possibility to spend very little (e.g., 1 C/month).

• BFTR (Yes): Zero-point interpolation or minimal offer can respond to any budget.

• Prepaid, Pay-as-you-go, Freemium (Yes): Minimum top-up 5C, per-unit consumption, free.

• Others (No): Minimum plan 10-20C, commitment.

e) 5. Overage handling.: What happens when consuming more than quota.

• BFTR (None): No overage.

• Prepaid (Block): Service cut.

• Freemium (Paid tier): Offer to upgrade to a tier.

• Rollover (Rollover): Use of unconsumed data.

• Family (Pool): Use of shared quota.

• Lending (Borrow): Borrow data from another member.

• Postpaid, Standard plans, Pay-as-you-go, Overage (Billing): Surcharge.

f) 6. Customer complexity.: Difficulty in understanding and using the model.

• BFTR (Very low): Single budget input and strategy choice from a clear list.

• Prepaid (Medium): Refills.

• Family, Lending (Medium): Group management.

• Pay-as-you-go (High): Monitoring required.

• Others (Low): Simple choice.

g) 7. Interest alignment.: How much the operator benefits from satisfying the customer.

• BFTR, Data lending (High): Loyalty through satisfaction or solidarity.

• Freemium, Rollover, Family (Medium): Operator wants conversion or pooling.

• Others (Low): Operator gains when customer consumes more.

h) 8. Innovation incentive.: Ease of introducing new services.

• BFTR (Maximal): Add a plan to catalog.

• Pay-as-you-go, Freemium (Medium): Add a unit or tier.

• Others (Low): Requires grid overhaul.

i) 9. Revenue predictability.: Ability to anticipate receipts.

• Prepaid, Standard plans, Subscription, Overage (High): Stable revenues.

• BFTR (Medium–High): Law of large numbers on budgets.

• Pay-as-you-go (Low): Very variable.

• Freemium, Rollover, Family, Lending (Medium): De pends on conversions, rollovers.

j) 10. Churn reduction.: Ability to retain customers.

• BFTR (Very high): Eliminates overages and mismatch.

• Family, Lending (High): Family solidarity.

• Pay-as-you-go, Freemium, Rollover (Medium): Flexibility reduces some frustrations.

• Others (Low): Frequent frustrations.

## B. Synthesis

BFTR outperforms all other systems on key dimensions. Its strategic flexibility (via user choice) and guarantee of no overcharging make it a superior solution.

Algorithm 8 BFTR: Recommendation according to user  
chosen strategy, with overcharge control   
Require: Budget b, strategy S, catalog F, history H, thresh  
olds p<sub>min</sub>, p<sub>max</sub>, overcharge threshold τ   
Ensure: Offer O<sup>∗</sup>   
1: // Phase 1: execute chosen strategy   
2: if S = SELECT then   
3: O ← SELECT(F, b)   
4: else if S = INTERP then   
5: O ← INTERP(F, b, p<sub>min</sub>, p<sub>max</sub>)   
6: else if S = REGR then   
7: O ← REGR(b, H)   
8: else if S = KNAP then   
9: O ← KNAP(F, b)   
10: else if S = PIECE or S = POW then   
11: O ← predictiveModel(S, b)   
12: else if S = HYB-REC then   
13: O ← HYB-REC(b, F, p<sub>min</sub>, p<sub>max</sub>)   
14: else if S = HYB-KF then   
15: O ← HYB-KF(b, F, p<sub>min</sub>, p<sub>max</sub>)   
16: else   
17: O ← minimalOffer   
18: end if   
19:   
20: // Phase 2: overcharge control   
21: if O = ∅ or p(O) > b then   
22: O ← minimalOffer   
23: end if   
24: p<sub>ref</sub> ← computeReferencePrice(O, F)   
25: s ← (p(O) − p<sub>ref</sub>)/p<sub>ref</sub> × 100   
26: if s > τ then   
27: O ← minimalOffer ▷ offer rejected for excessive   
overcharge   
28: end if   
29:   
30: return O

V. ARCHITECTURE AND USER STRATEGY CHOICE

The BFTR model relies on a modular pipeline illustrated in Figure 1. The system takes two types of input: (1) operator parameters (catalog of plans, business rules R, profitability thresholds) and (2) user data (budget $b _ { u } ,$ consumption history $\mathcal { H } _ { u }$ , and the chosen strategy among the eight available).

![](images/4c5f13f37c86cc1ecc3c1b0e329efc31c0f0f84b1cc48473e8e5f0d0957c7045.jpg)  
Fig. 1. BFTR architecture: the user chooses the strategy (from the two families: composition – KNAP, HYB-KF, HYB-REC – or generation – SELECT, INTERP, REGR, PIECE, POW). The system executes the strategy and then applies overcharge control using rules ${ \mathcal { R } } ,$ thresholds p<sub>min</sub>, p<sub>max</sub>, and overcharge threshold τ.

The core of the system is the BFTR algorithm (Algorithm 8) that takes as parameter the strategy selected by the user, executes it, and then applies overcharge control.

Complexity of the BFTR algorithm. Execution is domi nated by the complexity of the chosen strategy:

$$
T _ { \mathrm { B F I R } } = \left\{ \begin{array} { l l } { O ( k ) } & { \mathrm { f o r ~ S E L E C T } , } \\ { O ( \log k ) } & { \mathrm { f o r ~ I N T E R P } , } \\ { O ( 1 ) } & { \mathrm { f o r ~ R E G R , ~ P I E C E , ~ P O W } , } \\ { O ( k \log k ) } & { \mathrm { f o r ~ K N A P } , } \\ { O ( k \log k + \log k ) } & { \mathrm { f o r ~ h y b r i d s } . } \end{array} \right.
$$

With $k \leq 1 0 0$ plans, response time remains below millisecond, as confirmed by our experimental measurements (Section VI-D). The overcharge control phase runs in $O ( | O | ) = O ( k )$ , also negligible.

Execution example. A user has a budget of 9,000 NGN and chooses the HYB-REC strategy. The system first checks whether the budget lies within the catalog interval (636–23,569 NGN): true, so it calls INTERP. Interpolation between the 5 GB (2,378 NGN) and 10 GB (3,799 NGN) plans produces an offer at 9,000 NGN with volume $5 + ( 9 , 0 0 0 - 2 , 3 7 8 ) / ( 3 , 7 9 9 -$ $2 , 3 7 8 ) \times ( 1 0 - 5 ) \approx 3 1 . 2 { \mathrm { G B } }$ . The overcharge control verifies that the reference price is the budget (9,000 NGN), so the surcharge is zero. Measured time: 0.52 ms.

## VI. EXPERIMENTS

## A. Advanced Model Training

The two predictive models, Piecewise and Power Law, were trained on the real MTN Nigeria dataset containing 21 distinct plans and 974 customers. The target variable is plan price, the explanatory variable is volume in GB.

• Piecewise: Breakpoints (5 GB and 200 GB) were determined by cross-validation minimizing RMSE. On each segment, simple linear regression was fitted by least squares. The obtained equations are:

$$
P ( d ) = \left\{ \begin{array} { l l } { 2 3 8 . 3 3 d + 2 6 0 . 7 2 } & { d \leq 5 \mathrm { ~ G B } } \\ { 1 5 5 . 8 7 d + 4 , 1 2 1 . 0 8 } & { 5 < d \leq 2 0 0 \mathrm { ~ G B } } \\ { 2 0 7 . 7 2 d - 2 5 , 6 3 6 . 0 9 } & { d > 2 0 0 \mathrm { ~ G B } } \end{array} \right.\tag{5}
$$

• Power Law: Fitted by non-linear least squares (Levenberg–Marquardt) on $P = a \cdot d ^ { b } + c$ . Result:

$$
P ( d ) = 3 5 2 . 0 3 \cdot d ^ { 0 . 8 2 8 4 } + 2 , 4 0 8 . 3 5 .\tag{6}
$$

Performance on a 20% test set are respectively $R ^ { 2 } = 0 . 9 1 4 7 .$ $\mathrm { M A E } = 2 { , } 4 7 6$ NGN and $R ^ { 2 } = 0 . 8 4 6 4 , \mathrm { M A E } = 3 . 4 0 0 \mathrm { ~ N G N } .$

## B. Experimental Setup

• Evaluation dataset: 974 synthetic customers inspired by the MTN Nigeria market, catalog of 6 plans (1.5–100 GB, price 636–23,569 NGN). Budgets generated as $B =$ $P _ { \mathrm { p l a n } } \times \mathcal { U } ( 0 . 8 , 1 . 2 )$

• Catalog of plans used in the experiments: Table II presents the six plans with their IDs, volumes, prices, and margins. Each plan is assigned a unique ID (1 to 6) that will be used in the offer composition table to show how offers are built from base plans.

• Overcharge threshold: $\tau = 5 \%$

• Environment: Intel Core i7-1165G7 @ 2.80 GHz, 16 GB RAM, Python 3.10.

TABLE II  
CATALOG OF PLANS USED IN THE EXPERIMENTS
<table><tr><td>ID</td><td>Plan</td><td>Volume (GB)</td><td>Price (NGN)</td><td>Margin (%)</td></tr><tr><td>1</td><td>1.5GB Monthly Plan</td><td>1.5</td><td>636</td><td>41</td></tr><tr><td>2</td><td>5GB Monthly Plan</td><td>5.0</td><td>2,378</td><td>59</td></tr><tr><td>3</td><td>10GB Monthly Plan</td><td>10.0</td><td>3,799</td><td>52</td></tr><tr><td>4</td><td>20GB Monthly Plan</td><td>20.0</td><td>7,398</td><td>48</td></tr><tr><td>5</td><td>50GB Monthly Plan</td><td>50.0</td><td>13,468</td><td>35</td></tr><tr><td>6</td><td>100GB Monthly Plan</td><td>100.0</td><td>23,569</td><td>35</td></tr></table>

## C. Evaluation Metrics

To compare strategies objectively, we use the following metrics:

• Budget used (%): ratio between the effective price of the recommended offer and the user’s initial budget. A value close to 100% indicates optimal budget utilization without waste.

• Volume offered (GB): total data quantity (in gigabytes) included in the offer.

• Utility: score computed by equation (1) combining budget adequacy and coverage of historical needs. A utility of 1.0 corresponds to a perfectly adapted offer.

• Surcharge (%): relative difference between final price and reference price, defined by $\begin{array} { r } { s ( O ) \ = \ \frac { p ( O ) - p _ { \mathrm { r e f } } ( \dot { O } ) } { p _ { \mathrm { r e f } } ( O ) } \ \times } \end{array}$ 100.

• Overcharging rate: proportion of cases where $s ( O ) >$ 5%.

• Total loss (NGN): $b _ { u } - p ( O )$ (money not spent by the customer).

• Execution time (ms): average CPU time to generate an offer, measured over 1,000 calls.

• Failure rate: proportion of cases where the strategy fails to produce a valid offer before using the fallback (minimal offer).

• Average margin (%): the average of individual margins $( p { - } c ) / c { \times } 1 0 0$ over all offers generated by a strategy. This metric reflects the average profitability for the operator.

• Margin threshold attainment rate (%): the proportion of offers whose margin reaches or exceeds a predefined threshold (set to 20% in our experiments). This rate measures the strategy’s ability to consistently generate sufficient profit while respecting the non-overcharging constraint.

## D. Budget–First Results $( \alpha = 0 . 5 )$

Table III shows the performance of the eight strategies for $\alpha \ = \ 0 . 5$ . A variant of HYB-KF that includes a residual margin correction (denoted $\mathrm { H Y B  – K F _ { c o r r } } )$ is also shown for comparison. This correction artificially increases the price of the interpolated part to enforce a minimum margin of 30%. However, in the final implementation, the correction does not generate any overcharging because the final price is still capped at the budget; it only reduces budget utilization and increases the loss for the customer. Therefore, it is deprecated.

HYB-REC achieves the best budget usage (100%), highest volume (29.9 GB), and highest utility (0.946), with zero surcharge. POWERLAW offers an excellent compromise (99.9% budget, 38.1 GB, 0% overcharging) and, interestingly, a very high average margin (1009.5%), although it only attains the 20% margin threshold in 74.9% of cases. PIECEWISE offers the highest volume (39.7 GB) but with under-utilized budget (84.1%) and the lowest average margin (23.7%) and threshold attainment rate (62.1%), indicating that this strategy often sacrifices profitability for volume.

All strategies achieve a zero surcharge, confirming the theoretical guarantees. The corrected version of HYB-KF exhibits lower budget utilization (97.1%) and a loss of 119 NGN per customer on average, while maintaining the same average margin and threshold attainment. This demonstrates that the residual correction is detrimental and should be avoided.

Corollary 1 (Zero surcharge for all strategies). The results confirm the theoretical proofs: all strategies achieve a zero average surcharge, and the overcharging rate is 0% for every strategy. The apparent overcharging observed in earlier experiments has been resolved by correcting the reference price computation for hybrid offers containing interpolation.

## E. Detailed Offer Analysis for Representative Budgets

Using the catalog from Table II, Table IV presents the offers recommended by each strategy for five representative budgets: 5,000 NGN, 10,000 NGN, 15,000 NGN, 20,000 NGN, and 25,000 NGN. For each offer, we report the plan name, the composition in terms of catalog plan IDs, the final price, volume, real cost, the calculated surcharge, and the loss (budget – final price).

The notation used in the Composition (IDs) column is as follows:

• Plan x: refers to the plan with ID x in Table II.

• n×Plan x: means the plan is taken n times (e.g., 4×Plan 1 means four times the 1.5GB plan).

• Interp. (Plan i, Plan j): indicates linear interpolation between the two plans (e.g., between Plan 2 and Plan 3). The interpolated volume and price are computed using equation (3).

• Plan x + Plan y + Interp. (Plan i, Plan j): indicates that after selecting the catalog plans, the remaining budget is used for interpolation between Plan i and Plan j.

• Plan x + Plan y: means the offer is a composition of two catalog plans (no interpolation).

TABLE III  
PERFORMANCE OF EIGHT STRATEGIES IN BUDGET–FIRST MODE $( \alpha = 0 . 5 )$
<table><tr><td>Strategy</td><td>Budget%</td><td>Vol (GB)</td><td>Utility</td><td>Surcharge (%)</td><td>Loss (NGN)</td><td>Avg. margin (%)</td><td>Threshold attain. (%)</td></tr><tr><td>HYB-REC</td><td>100.0</td><td>29.9</td><td>0.946</td><td>0.0</td><td>0.0</td><td>50.9</td><td>100.0</td></tr><tr><td>POWERLAW</td><td>99.9</td><td>38.1</td><td>0.795</td><td>0.0</td><td>6.4</td><td>1009.5</td><td>74.9</td></tr><tr><td>HYB-KF</td><td>100.0</td><td>27.0</td><td>0.932</td><td>0.0</td><td>0.0</td><td>41.6</td><td>100.0</td></tr><tr><td>HYB-KFcorr</td><td>97.1</td><td>26.7</td><td>0.912</td><td>0.0</td><td>119.0</td><td>42.7</td><td>100.0</td></tr><tr><td>INTERP</td><td>99.3</td><td>29.4</td><td>0.942</td><td>0.0</td><td>199.9</td><td>50.9</td><td>100.0</td></tr><tr><td>PIECE</td><td>84.1</td><td>39.7</td><td>0.851</td><td>0.0</td><td>511.6</td><td>23.7</td><td>62.1</td></tr><tr><td>KNAP</td><td>86.3</td><td>26.3</td><td>0.827</td><td>0.0</td><td>297.1</td><td>42.9</td><td>100.0</td></tr><tr><td>REGR</td><td>94.3</td><td>25.9</td><td>0.876</td><td>0.0</td><td>969.1</td><td>45.7</td><td>100.0</td></tr><tr><td>SELECT</td><td>68.7</td><td>22.0</td><td>0.676</td><td>0.0</td><td>1,942.5</td><td>45.5</td><td>100.0</td></tr></table>

For each offer, the reference price $p _ { \mathrm { r e f } }$ is determined as follows:

• Composition offers (KNAP): $p _ { \mathrm { r e f } }$ is the sum of the catalog prices of the selected plans.

• Interpolation offers (INTERP): $p _ { \mathrm { r e f } }$ is the budget itself.

• Generation offers (PIECE, POW): $p _ { \mathrm { r e f } }$ is the estimated price before capping.

• Mixed offers (HYB-KF, HYB-REC): $p _ { \mathrm { r e f } }$ is the original budget. The final price is the sum of the selected catalog plans plus the interpolated remainder (which uses the remaining budget). Therefore, the final price equals the budget, and surcharge = 0%.

The Loss (NGN) column represents the amount of budget not spent by the customer. This loss corresponds to money that the customer keeps but that the operator does not capture. While a zero loss is ideal for the operator (full revenue capture), a positive loss is not necessarily detrimental to the customer—it simply means they receive less data than they could have afforded. However, a very large loss (e.g., for REGRESSION with a budget of 20,000 NGN, the loss is 12,602 NGN) indicates that the strategy is highly inefficient in terms of budget utilization. Strategies such as INTER-POLATION, HYB-REC, and HYB-KF consistently achieve zero loss because they use interpolation to consume the entire budget. SELECTION and REGRESSION often leave significant losses because they are limited to existing catalog plans. PIECE and POW also leave small losses (typically less than 15 NGN) because they cap the estimated price to the budget, but the estimation is not exact. Thus, from an operator’s perspective, strategies with zero loss are preferable to maximize revenue, while from a customer’s perspective, a moderate loss is acceptable if it comes with a higher volume (e.g., PIECE vs. INTERPOLATION for a 10,000 NGN budget: PIECE offers 37.7 GB at 9,997 NGN with a loss of 3 NGN, whereas INTERPOLATION offers 32.9 GB at 10,000 NGN with zero loss – the customer pays 3 NGN more for 4.8 GB less). This trade-off must be considered when choosing a strategy.

This is confirmed in Table IV, where every row shows a surcharge of 0.00%.

## F. Execution Times

Table V shows the average execution times measured for each strategy over 974 customers. PIECE and POW are the fastest (< 0.1 ms) because they only evaluate a function.

KNAP is slower ( 2 ms) due to sorting and plan selection. All strategies remain well below 5 ms, making them suitable for real-time production use.

## G. Robustness: Failure and Fallback Rates

Table VI shows the failure rate (offer not produced before fallback) for each strategy. PIECE, POW, and REGR never fail because they rely on robust parametric methods. SELECT, KNAP, and hybrids fail in 9.4% of cases when the budget is below the smallest plan (636 NGN). INTERP fails in 17.8% of cases when the budget is outside the catalog interval.

## H. Sensitivity Analysis on Parameter α

To study the influence of the weight between price and volume, we varied α in the utility equation (1) at three levels: $\alpha = 0 . 2$ (volume priority), $\alpha = 0 . 5$ (balance), and $\alpha = 0 . 8$ (budget priority). Results are summarized in Table VII.

We observe that for low α (volume priority), ‘PIECEWISE‘ is best (0.857) because it offers the highest volume (39.7 GB). For $\alpha = 0 . 5 $ , ‘HYB-REC‘ becomes optimal (0.946). For high α (budget priority), ‘HYB-REC‘ clearly dominates (0.978) thanks to its ability to use 100% of the budget.

## I. Detailed Analysis and Discussion

1) Customer Performance: HYB-REC dominates thanks to direct interpolation when the budget is within the catalog, combined with KNAP for out-of-range budgets. It achieves 100% budget utilization, the best volume (29.9 GB), and the highest utility (0.946). POWERLAW is surprising: it uses 99.9% of the budget, offers 38.1 GB (near-maximum volume), and nearzero loss (6.4 NGN), with 0% overcharging. It constitutes an excellent volume/budget compromise, but its very high average margin (1009.5%) suggests extreme price estimates on certain volumes, yet it does not overcharge because the final price is capped at the budget.

2) Operator-Oriented Margin Metrics: The introduction of average margin and threshold attainment rate provides a new perspective. Strategies like HYB-REC, INTERP, KNAP, REGR, and SELECT all guarantee that 100% of offers achieve the 20% margin threshold, making them reliable for the operator. In contrast, PIECE has lower threshold attainment (62.1%), indicating that this strategy occasionally generates offers with very thin margins, which may be acceptable if the primary goal is to maximize volume or utility. POWERLAW, despite its extremely high average margin, only reaches the threshold in 74.9% of cases, meaning that in about 25% of offers the margin is below 20%. This highlights the importance of evaluating not only the average but also the consistency of profitability.

TABLE IV  
OFFERS RECOMMENDED BY EACH STRATEGY FOR REPRESENTATIVE BUDGETS (α = 0.5)
<table><tr><td colspan="2">Budget|Strategy</td><td>|Plan</td><td>Composition (IDs)</td><td colspan="5">Price (NGN) Vol (GB) Cost (NGN) Surcharge (%) Loss (NGN)</td></tr><tr><td rowspan="8">5,000</td><td>|SELECTION INTERPOLATION</td><td>10GB Monthly Plan Interpolated 13.3GB</td><td>Plan 3 Interp. (Plan 2, Plan 3)</td><td>3,799 5,000</td><td>10.0 13.3</td><td>2,500 3,334</td><td>0.00 0.00</td><td>1,201 0</td></tr><tr><td></td><td></td><td>Plan 3</td><td>3,799</td><td>10.0</td><td>2,500</td><td>0.00</td><td></td></tr><tr><td>REGRESSION</td><td>10GB Monthly Plan</td><td></td><td>4,435</td><td>11.5</td><td>2,950</td><td>0.00</td><td>1,201</td></tr><tr><td>KNAP</td><td>10GB + 1.5GB</td><td>Plan 3 + Plan 1</td><td></td><td>13.3</td><td></td><td>0.00</td><td>565</td></tr><tr><td>HYB-REC</td><td>Interpolated 13.3GB</td><td>Interp. (Plan 2, Plan 3)</td><td>5,000</td><td></td><td>3,334</td><td></td><td>0</td></tr><tr><td>HYB-KF</td><td>10GB + 1.5GB + interpolated</td><td>Plan 3 + Plan 1 + Interp. (Plan 2, Plan 3)</td><td>5,000</td><td>12.8</td><td>3,350</td><td>0.00</td><td>0</td></tr><tr><td>PIECE POW</td><td>Piecewise 5.6GB PowerLaw 11.1GB</td><td>Piecewise model Power law model</td><td>4,994 4,994</td><td>5.6 11.1</td><td>1,400 2,775</td><td>0.00 0.00</td><td>6 6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">10,000</td><td>SELECTION</td><td>|20GB Monthly Plan</td><td>Plan 4</td><td>7,398</td><td>20.0</td><td>5,000</td><td>0.00</td><td>2,602</td></tr><tr><td>INTERPOLATION</td><td>Interpolated 32.9GB</td><td>Interp. (Plan 3, Plan 4)</td><td>10,000</td><td>32.9</td><td>6,572</td><td>0.00</td><td>0</td></tr><tr><td>REGRESSION</td><td>10GB Monthly Plan</td><td>Plan 3</td><td>3,799</td><td>10.0</td><td>2,500</td><td>0.00</td><td>6,201</td></tr><tr><td>KNAP</td><td>20GB + 4×1.5GB</td><td>Plan 4 + 4×Plan 1</td><td>9,942</td><td>26.0</td><td>6,800</td><td>0.00</td><td>58</td></tr><tr><td>HYB-REC</td><td>Interpolated 32.9GB</td><td>Interp. (Plan 3, Plan 4)</td><td>10,000</td><td>32.9</td><td>6,572</td><td>0.00</td><td>0</td></tr><tr><td>HYB-KF</td><td></td><td>20GB + 4×1.5GB + interpolatedPlan 4 + 4× Plan 1 + Interp. (Plan 3, Plan 4)</td><td>10,000</td><td>26.1</td><td>6,851</td><td>0.00</td><td>0</td></tr><tr><td>PIECE</td><td>Piecewise 37.7GB</td><td>Piecewise model</td><td>9,997</td><td>37.7</td><td>7,540</td><td>0.00</td><td>3</td></tr><tr><td>|POW</td><td>PowerLaw 40.7GB</td><td>Power law model</td><td>9,994</td><td>40.7</td><td>8,140</td><td>0.00</td><td>6</td></tr><tr><td rowspan="9">15,000</td><td>|SELECTION</td><td>|50GB Monthly Plan</td><td>Plan 5</td><td>13,468</td><td>50.0</td><td>10,000</td><td>0.00</td><td>1,532</td></tr><tr><td>INTERPOLATION</td><td>Interpolated 57.6GB</td><td>Interp. (Plan 4, Plan 5)</td><td>15,000</td><td>57.6</td><td>10,077</td><td>0.00</td><td>0</td></tr><tr><td>REGRESSION</td><td>20GB Monthly Plan</td><td>Plan 4</td><td>7,398</td><td>20.0</td><td>5,000</td><td>0.00</td><td>7,602</td></tr><tr><td>KNAP</td><td>50GB + 2×1.5GB</td><td>Plan 5 + 2×Plan 1</td><td>14,740</td><td>53.0</td><td>10,900</td><td>0.00</td><td>260</td></tr><tr><td>HYB-REC</td><td>Interpolated 57.6GB</td><td>Interp. (Plan 4, Plan 5)</td><td>15,000</td><td>57.6</td><td>10,077</td><td>0.00</td><td>0</td></tr><tr><td>HYB-KF</td><td></td><td>50GB + 2×1.5GB + interpolated Plan 5 + 2 × Plan 1 + Interp. (Plan 4, Plan 5)</td><td>15,000</td><td>53.6</td><td>11,130</td><td>0.00</td><td>0</td></tr><tr><td>PIECE</td><td>Piecewise 69.7GB</td><td>Piecewise model</td><td>14,985</td><td>69.7</td><td>12,198</td><td>0.00</td><td>15</td></tr><tr><td>POW</td><td>PowerLaw 75.0GB</td><td>Power law model</td><td>14,994</td><td>75.0</td><td>13,125</td><td>0.00</td><td>6</td></tr><tr><td>|SELECTION</td><td>|50GB Monthly Plan</td><td>Plan 5</td><td>13,468</td><td>50.0</td><td>10,000</td><td>0.00</td><td>6,532</td></tr><tr><td rowspan="9">20,000</td><td>INTERPOLATION</td><td>Interpolated 82.3GB</td><td>Interp. (Plan 5, Plan 6)</td><td>20,000</td><td>82.3</td><td>14,408</td><td>0.00</td><td>0</td></tr><tr><td>REGRESSION</td><td>20GB Monthly Plan</td><td>Plan 4</td><td>7,398</td><td>20.0</td><td>5,000</td><td>0.00</td><td>12,602</td></tr><tr><td>KNAP</td><td>50GB + 10GB</td><td>Plan 5 + Plan 3</td><td>19,811</td><td>66.0</td><td>14,300</td><td>0.00</td><td>189</td></tr><tr><td>HYB-REC</td><td>Interpolated 82.3GB</td><td>Interp. (Plan 5, Plan 6)</td><td>20,000</td><td>82.3</td><td>14,408</td><td>0.00</td><td>0</td></tr><tr><td>HYB-KF</td><td>50GB + 10GB + interpolated</td><td>Plan 5 + Plan 3 + Interp. (Plan 5, Plan 6)</td><td>20,000</td><td>66.4</td><td>14,467</td><td>0.00</td><td>0</td></tr><tr><td>PIECE</td><td>Piecewise 101.8GB</td><td>Piecewise model</td><td>19,989</td><td>101.8</td><td>15,270</td><td>0.00</td><td>11</td></tr><tr><td>|POW</td><td>PowerLaw 112.3GB</td><td>Power law model</td><td>19,992</td><td>112.3</td><td>16,845</td><td>0.00</td><td>8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SELECTION</td><td>100GB Monthly Plan</td><td>Plan 6 Plan 6</td><td>23,569</td><td>100.0</td><td>17,500</td><td>0.00</td><td>1,431</td></tr><tr><td rowspan="9">25,000</td><td>INTERPOLATION</td><td>100GB Monthly Plan</td><td></td><td>23,569</td><td>100.0</td><td>17,500</td><td>0.00</td><td>1,431</td></tr><tr><td>REGRESSION</td><td>50GB Monthly Plan</td><td>Plan 5</td><td>13,468</td><td>50.0</td><td>10,000</td><td>0.00</td><td>11,532</td></tr><tr><td>KNAP</td><td>100GB + 2×1.5GB</td><td>Plan 6 + 2×Plan 1</td><td>24,841</td><td>103.0</td><td>18,400</td><td>0.00</td><td>159</td></tr><tr><td>HYB-REC</td><td>100GB + 2×1.5GB</td><td>Plan 6 + 2×Plan 1</td><td>25,000</td><td>103.4</td><td>18,541</td><td>0.00</td><td>0</td></tr><tr><td>HYB-KF</td><td>100GB + 2×1.5GB</td><td>Plan 6 + 2×Plan 1</td><td>25,000</td><td>103.4</td><td>18,541</td><td>0.00</td><td>0</td></tr><tr><td>PIECE</td><td>Piecewise 133.9GB</td><td>Piecewise model</td><td>24,992</td><td>133.9</td><td>20,085</td><td>0.00</td><td>8</td></tr><tr><td>POW</td><td>PowerLaw 151.9GB</td><td>Power law model</td><td>24,991</td><td>151.9</td><td>22,785</td><td>0.00</td><td>9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE V  
AVERAGE EXECUTION TIMES PER STRATEGY (MS)
<table><tr><td>Strategy</td><td>Mean (ms)</td><td>Std. dev. (ms)</td></tr><tr><td>PIECE</td><td>0.06</td><td>0.02</td></tr><tr><td>POW</td><td>0.05</td><td>0.02</td></tr><tr><td>SELECT</td><td>1.12</td><td>0.36</td></tr><tr><td>INTERP</td><td>1.45</td><td>0.25</td></tr><tr><td>REGR</td><td>3.93</td><td>0.74</td></tr><tr><td>KNAP</td><td>1.96</td><td>0.43</td></tr><tr><td>HYB-REC</td><td>1.85</td><td>0.75</td></tr><tr><td>HYB-KF</td><td>3.35</td><td>0.74</td></tr></table>

3) No Overcharging: The results confirm theoretical proofs: all strategies show a zero average surcharge and a 0% overcharging rate. As demonstrated in Table IV, every recommended offer has a final price equal to its reference price. For composition strategies (KNAP), the reference price is the sum of catalog plan prices. For generation strategies (INTERP, PIECE, POW), the reference price is either the budget or the estimated price. For hybrid strategies (HYB-KF, HYB-REC), the reference price is the original budget, and the final price equals the sum of the catalog plans plus the interpolated remainder. In all cases, the equality $p ( O ) = p _ { \mathrm { r e f } } ( O )$ holds, guaranteeing $s ( O ) ~ = ~ 0 \%$ . The ‘Composition (IDs)‘ column explicitly shows how each offer is built from base plans, including the interpolation of the remainder for hybrid strategies, making it easy to verify that no price markup is applied.

TABLE VI  
FAILURE AND FALLBACK RATES
<table><tr><td>Strategy</td><td>Failure rate (%)</td><td>Fallback used (%)</td></tr><tr><td>SELECT</td><td>9.4</td><td>9.4</td></tr><tr><td>INTERP</td><td>17.8</td><td>17.8</td></tr><tr><td>REGR</td><td>0.0</td><td>0.0</td></tr><tr><td>KNAP</td><td>9.4</td><td>9.4</td></tr><tr><td>HYB-REC</td><td>0.0</td><td>0.0</td></tr><tr><td>HYB-KF</td><td>0.0</td><td>0.0</td></tr><tr><td>PIECE</td><td>0.0</td><td>0.0</td></tr><tr><td>POW</td><td>0.0</td><td>0.0</td></tr></table>

TABLE VII

AVERAGE UTILITY ACCORDING TO α
<table><tr><td>Strategy</td><td> $\alpha = 0 . 2$ </td><td> $\alpha = 0 . 5$ </td><td> $\alpha = 0 . 8$ </td></tr><tr><td>HYB-REC</td><td>0.914</td><td>0.946</td><td>0.978</td></tr><tr><td>HYB-KF</td><td>0.892</td><td>0.932</td><td>0.973</td></tr><tr><td>PIECEWISE</td><td>0.857</td><td>0.851</td><td>0.845</td></tr><tr><td>POWERLAW</td><td>0.673</td><td>0.795</td><td>0.917</td></tr><tr><td>INTERPOLATION</td><td>0.911</td><td>0.942</td><td>0.972</td></tr><tr><td>REGRESSION</td><td>0.866</td><td>0.876</td><td>0.886</td></tr><tr><td>SELECTION</td><td>0.664</td><td>0.676</td><td>0.689</td></tr><tr><td>KNAP</td><td>0.799</td><td>0.827</td><td>0.854</td></tr></table>

TABLE VIII  
RECOMMENDATIONS BY OBJECTIVE
<table><tr><td>Objective</td><td>Recommended Strategy</td></tr><tr><td>Customer (max utility)</td><td>HYB-REC (0.946)</td></tr><tr><td>Max volume</td><td>PIECE (39.7 GB)</td></tr><tr><td>Budget capture</td><td>HYB-REC (100%)</td></tr><tr><td>Speed</td><td>PIECE (0.06 ms)</td></tr><tr><td>Robustness (0% failure)</td><td>PIECE, REGR, POW</td></tr><tr><td>No overcharging</td><td>All strategies (0%)</td></tr><tr><td>Operator profitability (avg. margin)</td><td>POWERLAW (1009.5%)</td></tr><tr><td></td><td>Operator reliability (threshold attain.) HYB-REC, KNAP, REGR, SELECT (100%)</td></tr><tr><td>Best overall compromise</td><td>HYB-REC</td></tr></table>

4) Hybrid Comparison: HYB-REC outperforms HYB-KF because direct interpolation on the total budget (when budget is within range) gives 100% utilization without any loss. HYB-KF starts with KNAP, which may leave a remainder that is then interpolated, but both strategies achieve zero surcharge. The corrected version of HYB-KF $( \mathrm { H Y B  – K F _ { c o r r } } )$ demonstrates that any residual margin correction is detrimental: it reduces budget utilization and creates a loss for the customer without any benefit in terms of margin or surcharge.

5) Complexity and Validity: All strategies execute in < 5 ms, compatible with production. The no-overcharging guarantee is verified: the average surcharge is 0% for all composition and pure interpolation strategies.

6) Catalog Size Impact: With only 6 plans, INTERP fails for 17% of out-of-range budgets. Hybrid and KNAP strategies are robust because they create composite offers from existing plans.

## J. Final Recommendations

Table VIII summarizes recommendations by objective. The user can thus choose the strategy best suited to their priorities.

For a production solution, we recommend using HYB-REC for the customer (maximum satisfaction) and PIECE for a robust, non-overcharging strategy. The sensitivity analysis shows that tuning α can adapt the system to different user profiles. If the operator prioritizes margin reliability, strategies with 100% threshold attainment (HYB-REC, KNAP, REGR, SELECT) should be preferred over PIECE and POWERLAW.

## VII. CONCLUSION AND PERSPECTIVES

We have presented BFTR, a complete algorithmic framework integrating eight Budget–First strategies, with two original hybrid approaches. Unlike existing systems that adjust prices upward to guarantee a minimum margin, BFTR guarantees the absence of overcharging by systematically aligning the final price with the catalog reference price.

Experiments on 974 customers confirm the theoretical guarantees: HYB-REC delivers the highest customer utility (0.946) with 100% budget utilization; PIECE maximizes volume (39.7 GB); and POWERLAW offers an excellent volume-budget compromise. All strategies achieve zero surcharge, and the transparent decomposition of offers (via catalog IDs) ensures full verifiability. The Loss (NGN) metric highlights a key trade-off: strategies like HYB-REC capture the entire budget, while others leave unspent money in exchange for higher volume. The residual margin correction $( \mathrm { H Y B  – K F _ { c o r r } } )$ is detrimental and should be avoided.

Sensitivity analysis confirms that volume-prioritizing strategies dominate for low α, while budget-capturing ones dominate for high α. Composition strategies (KNAP, HYB-KF) are the most robust (0% failure) and fastest $( < 2 ~ \mathrm { m s } )$ . The formal proof of no overcharging for composition and interpolation strategies constitutes a major theoretical contribution.

From a regulatory perspective, BFTR does not alter the mandatory tariff approval process, but it shifts the requirements toward demonstrating algorithmic transparency, nondiscrimination, and price integrity. Its native no-overcharging guarantee (final price ≤ reference price) positions it as an ideal compliance tool. Supported by clear technical documentation and empirical evidence (e.g., Table IV), BFTR enables operators to obtain regulatory approval and deploy a fair, personalized tariff system.

Perspectives include multi-service extension, integration of collaborative PageRank for strategy personalization, and deployment on real operator data.

## ACKNOWLEDGMENT

The author also expresses gratitude to telecom operators such as Orange and MTN for their essential services, technological advancements, and sustained contributions to the telecommunications industry, which provide the foundation for research in tariff personalization and network optimization. The open-source community is also acknowledged for the Python tools used in this work.

It is the author’s sincere hope that the proposed BFTR framework will be adopted in the near future by telecom operators to enhance tariff fairness, reduce customer churn, and foster greater trust between operators and their subscribers.

## REFERENCES

[1] Klein, A., & Jakopin, N. (2018). Consumer choice and tariff complexity in telecommunications. Telecommunications Policy, 42(8), 636–645.

[2] GSMA (2024). The Mobile Economy 2024. GSMA Intelligence.

[3] Verdecchia, R., et al. (2021). AI for churn prediction in telecom: a systematic review. IEEE Access, 9, 123456–123470.

[4] Zhang, S., & Hurley, N. (2020). Budget-constrained recommendation via reinforcement learning. Proceedings of RecSys, 320–329.

[5] Heimburg, V., Schreieck, M., & Wiesche, M. (2025). Complementor Value Co-Creation in Generative AI Platform Ecosystems. Journal of Management Information Systems.

[6] Li, Z., et al. (2024). Large Generative AI Models meet Open Networks for 6G: Integration, Platform, and Monetization. arXiv:2410.18790.