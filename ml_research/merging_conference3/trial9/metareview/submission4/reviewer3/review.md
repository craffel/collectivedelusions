# Peer Review

## Summary of the Paper
The submission addresses the fundamental challenge of serving Parameter-Efficient Fine-Tuning (PEFT) adapters (specifically, task-specific LoRA experts) on a sequential, heterogeneous, and unlabeled serving stream. 

In deep network architectures under online serving, stateless routing systems (e.g., SABLE) suffer from extreme layer-to-layer ensembling weight oscillations, termed **routing jitter**, which is induced by local representational noise and cascading non-linearities. This jitter blends incompatible expert parameters in successive layers, initiating a cascade of representational drift that degrades overall classification accuracy. 

To resolve this, the state-of-the-art stateful routing system ChemMerge models ensembling weights as chemical concentrations governed by biochemical kinetics, Arrhenius reaction rates, and continuous Ordinary Differential Equations (ODEs) integrated via numerical solvers. 

Through the lens of Occam's razor, this paper mathematically deconstructs ChemMerge's complex continuous physical metaphor. The authors prove that under explicit Euler discretization, the continuous rate equations are equivalent to a simple constant Exponential Moving Average (EMA). Stripping away the biochemical machinery, the authors propose **Momentum-Merge**, a training-free, single-parameter dynamic ensembling framework that stabilizes routing trajectories across network depth using a simple constant EMA update:
$$\alpha_k^{(l)} = (1 - \beta) w_k^{(l)} + \beta \alpha_k^{(l-1)}$$
where $\beta \in [0, 1]$ is the constant momentum coefficient and $w_k^{(l)}$ represents the raw similarity-routing weights.

Evaluating within the Analytical Coordinate Sandbox (ICS), the paper exposes a fundamental **Accuracy-Stability trade-off**: stateless calibrated routing (SABLE + Layer Centroids) achieves the highest joint accuracy (77.24%) but suffers from high routing jitter, whereas stateful smoothing acts as a low-pass filter that trades a minor fraction of accuracy to achieve high routing stability. Under this trade-off, basic Momentum-Merge achieves 74.85% joint accuracy and reduces routing jitter by 5.7$\times$ over tuned SABLE, while its advanced variant (incorporating layer-wise centroid anchoring and raw boundary initialization) reaches 74.98% joint accuracy and reduces routing jitter to a near-zero 0.000374 (a massive 195.7$\times$ reduction over tuned SABLE and a 41.1$\times$ reduction over tuned ChemMerge). 

---

## Overall Recommendation
**Recommendation:** **5: Accept**  
The submission is a mathematically rigorous, highly thorough, and outstandingly written paper. It applies Occam's razor to deep learning design, demonstrating that a classical, well-understood mathematical operator (the Exponential Moving Average) can completely deconstruct and outperform a highly convoluted, metaphor-driven state-of-the-art physical ODE model. The theoretical derivations are clean, and the empirical verification across 10 independent random seeds is exceptionally thorough. The paper's conceptual parsimony, combined with extensive stability sweeps, a detailed scaling trajectory for LLMs, and high-integrity disclosures of vulnerabilities (such as recurrence trapping), makes it a high-value contribution to the PEFT serving and Mixture of Experts communities.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Parsimony and Rigor (Originality & Soundness):** The paper applies a refreshing and highly rigorous approach to deep learning design. By deconstructing ChemMerge'scontinuous biochemical kinetics (Theorem 1), it exposes a highly strained physical metaphor as redundant complexity. Proposing a single-line constant EMA replacement (Momentum-Merge) is a major contribution toward parsimonious and interpretable deep learning architectures.
2. **Exhaustive Empirical Evaluation (Soundness):** Unlike typical papers, the authors evaluate all methods across 10 independent random seeds, provide standard deviations, and perform pairwise seed checks using paired $t$-tests to prove statistical significance ($p \approx 0.0061$ over ChemMerge and $p \approx 0.0212$ over SABLE).
3. **Comprehensive Sensitivity Sweeps (Soundness):** The appendices contain highly rigorous and high-signal sweeps over Softmax temperature ($\tau$), joint hyperparameter space ($\beta \times \tau$), depth-wise momentum schedules (V-shaped Momentum), task-asymmetric noise regimes, and expert pool scaling ($K=10$), mapping the full boundaries of stateful ensembling physics.
4. **Honest Disclosure of Core Phenomena (Presentation & Significance):** The paper clearly maps the **Accuracy-Stability trade-off** in dynamic serving pipelines and exposes a significant theoretical vulnerability of stateful recurrences—**Recurrence Trapping**—under scarce calibration data. 
5. **Practical, Actionable LLM Blueprint (Significance):** The mathematical scaling trajectory (Layer-wise Centroid Anchoring, Layer-wise Temperature Scaling, Depth-wise Momentum Modulation) and the standardized physical evaluation protocol (LLaMA-7B with GLUE/HumanEval/GSM8K) in Appendix B provide a complete blueprint for real-world deployment.

### Weaknesses
1. **Lack of Real-World LLM Evaluation (Significance):** While the synthetic Analytical Coordinate Sandbox (ICS) is highly controlled and perfectly suited for a clean comparative study, the paper is completely situated within this simulated environment. The ecological validity would be significantly stronger if the authors ran even a small-scale real-world experiment (e.g., LLaMA-7B with 2-3 LoRA adapters) to empirically confirm the sandbox findings.
2. **Artificial rate-matching constraint in Theorem 1's Proof (Soundness):** In Theorem 1's proof, the authors assume $\kappa = k_{\text{decay}}$ as a "physical constraint" to keep concentrations on the probability simplex. As shown in our mathematical check below, this rate-matching assumption is completely redundant, as step-wise normalization automatically guarantees the EMA form for *any* values of $\kappa$ and $k_{\text{decay}}$.

---

## Detailed Evaluation of Dimensions

### 1. Soundness
**Rating:** **Excellent**

The paper is technically very sound and demonstrates high scientific hygiene. The mathematical derivations are precise, and the experimental protocol is rigorous. 

#### Deeper Mathematical Analysis: Extending Theorem 1
In Theorem 1, the authors prove mathematical equivalence by assuming $\kappa \Delta t = k_{\text{decay}} \Delta t = \gamma$. They note that forcing $\kappa = k_{\text{decay}}$ is physically strained, as there is no thermodynamic reason why the rate of expert species creation must equal its rate of degradation. 

However, we prove that this rate-matching assumption is entirely unnecessary. Any probability-conserving system must perform a step-wise simplex projection (normalization) on the concentration vector $C_k^{(l)}$:
$$C_k^{(l)} = (\kappa \Delta t) w_k^{(l)} + (1 - k_{\text{decay}} \Delta t) \alpha_k^{(l-1)}$$
Summing over $k$ (where $\sum_k w_k^{(l)} = 1$ and $\sum_k \alpha_k^{(l-1)} = 1$):
$$\sum_k C_k^{(l)} = \kappa \Delta t + 1 - k_{\text{decay}} \Delta t = Z$$
Normalizing to obtain the ensembling weights $\alpha_k^{(l)} = C_k^{(l)} / Z$:
$$\alpha_k^{(l)} = \frac{\kappa \Delta t}{Z} w_k^{(l)} + \frac{1 - k_{\text{decay}} \Delta t}{Z} \alpha_k^{(l-1)}$$
Letting $\gamma = \frac{\kappa \Delta t}{Z}$. Then:
$$1 - \gamma = 1 - \frac{\kappa \Delta t}{\kappa \Delta t + 1 - k_{\text{decay}} \Delta t} = \frac{1 - k_{\text{decay}} \Delta t}{Z}$$
This yields:
$$\alpha_k^{(l)} = \gamma w_k^{(l)} + (1 - \gamma) \alpha_k^{(l-1)}$$
This is *exactly* equivalent to a constant-inertia EMA with momentum parameter $\beta = 1 - \gamma$, where $\gamma$ is a constant because $\kappa$, $k_{\text{decay}}$, and $\Delta t$ are all constants!

This is a profound theoretical result: **step-wise normalization automatically guarantees mathematical equivalence to a constant EMA for *any* values of $\kappa$ and $k_{\text{decay}}$, meaning the biochemical metaphor is even more redundant than the authors claim.** The rate-matching assumption is completely bypassed by the simplex constraint itself.

#### Theoretical Limitations: Recurrence Trapping
In Appendix G, the authors expose a critical theoretical vulnerability of stateful recurrences which they term **Recurrence Trapping**. 
Under small calibration subset sizes ($|\mathcal{C}_k| \le 16$), the pre-computed layer centroids are highly noisy, making the initial boundary weight $w_k^{(L_{\text{frozen}}+1)}$ highly inaccurate. Because the stateful model possesses temporal momentum memory, this initial boundary error propagates through network depth, trapping the ensembling coefficients in highly sub-optimal states throughout the forward pass and collapsing joint accuracy to **71.20%** when $|\mathcal{C}_k| = 8$ (a **4.80% absolute degradation** compared to the stateless SABLE + Layer Centroids which achieves **76.00%**).

This highlights a fundamental theoretical trade-off: **stateful smoothing introduces a structural vulnerability where the routing trajectory is highly sensitive to initialization errors (recurrence trapping) when representation anchors are noisy.** Stateless systems, evaluating layers independently, are self-correcting across depth. This is an excellent, highly honest contribution to our theoretical understanding of stateful merging.

---

### 2. Presentation
**Rating:** **Excellent**

The presentation is outstanding. 
- The paper is written with high mathematical clarity, employing precise terminology and explicit, well-structured equations.
- The narrative flow is highly logical, starting from the identification of routing jitter, deconstructing the stateful SOTA, introducing a minimalist framework, and systematically mapping the resulting dynamics and trade-offs.
- Figures 1 and 2 are exceptionally clean, high-signal, and directly reinforce the text.
- The tables (Tables 1-8) are beautifully organized and provide comprehensive statistics and parameters.

---

### 3. Significance
**Rating:** **Good**

The significance is good, with high conceptual value. 
- **Philosophical Value:** This paper acts as a vital sanity check against the growing trend of wrapping simple mathematical operators in convoluted, "pseudo-physical" metaphors. Proving that EMA outperforms complex continuous-time ODEs in model merging should encourage parsimony across other sub-areas of deep learning.
- **Practical Serving Utility:** For production PEFT serving, suppressing layer-to-layer ensembling oscillations is highly critical. Momentum-Merge provides a training-free, zero-overhead, highly stable ensembling framework that virtually eliminates routing oscillations, making it extremely attractive for low-latency serving pipelines.
- **Theoretical Extensions:** The insights regarding depth-wise momentum scheduling (V-shaped Momentum) and its dynamic, variance-based estimation could directly influence how routing choices are regularized across depth in large Mixture of Experts (MoE) architectures.

Its significance is slightly limited by the lack of physical LLM experiments (which are outlined as a blueprint in Appendix B, but not executed).

---

### 4. Originality
**Rating:** **Excellent**

The originality of the paper is excellent, of a highly refreshing, **conceptually reductive** nature. 
Instead of introducing a highly complex new method, the paper's novelty lies in its mathematical deconstruction and simplification of the state-of-the-art, demonstrating that a classical, well-understood baseline (EMA) is actually the superior choice. 

Furthermore, the introduction of **Raw Boundary Initialization** (Eq. 7) is a small but highly effective original contribution, eliminating transient startup jitter and reducing routing jitter by 70.1$\times$ by starting the recurrence in its stationary state. The depth-wise scheduling (V-shaped Momentum) and task-asymmetric noise analysis further enrich the paper's original contributions.

---

## Questions and Constructive Suggestions for the Authors

1. ** Broader Mathematical Derivation of Theorem 1:** Consider incorporating the step-wise normalization derivation outlined in our Soundness section. By showing that step-wise normalization guarantees the constant EMA form for *any* values of $\kappa$ and $k_{\text{decay}}$, you can completely remove the restrictive and physically strained rate-matching assumption ($\kappa = k_{\text{decay}}$), which significantly strengthens your theoretical deconstruction.
2. **Preliminary Real-World Verification:** Even a small-scale real-world experiment (e.g., deploying 2-3 LoRA adapters on a 7B parameter Transformer model and evaluating on 100 shuffled samples) would enormously enhance the ecological validity of the paper and ground the excellent scaling blueprint you provide in Appendix B.
3. **Investigation of Recurrence Trapping Mitigation:** Since Raw Boundary Initialization under noisy calibration data causes Recurrence Trapping, have you considered employing a hybrid approach? For example, starting with a higher-entropy uniform boundary or applying a dynamic, layer-wise noise-dependent momentum scale $\beta^{(l)}$ (lower momentum early in the network) to allow the recurrence to escape noisy initial states before applying heavy temporal smoothing.
