# Final Peer Review

## Summary of the Paper
The paper addresses the challenge of serving multiple task-specific Low-Rank Adaptation (LoRA) expert adapters on a highly heterogeneous, sample-by-sample online stream without task labels. To resolve the issue of **routing jitter**—high-frequency layer-to-layer oscillations in ensembling weights caused by representational noise in deep networks—previous state-of-the-art work (ChemMerge) introduced a continuous-time framework modeling ensembling weights as chemical concentrations governed by biochemical Ordinary Differential Equations (ODEs). 

Applying Occam's razor, the authors mathematically prove that under uniform activation energies and constant temperature, ChemMerge's rate equations simplify to a simple, discrete **constant Exponential Moving Average (EMA)** update on ensembling weights across depth. Stripping away the biochemical metaphor and ODE solvers, they propose **Momentum-Merge**, a training-free, single-parameter dynamic ensembling framework requiring only one line of code. They evaluate Momentum-Merge in the Analytical Coordinate Sandbox (ICS) against SABLE, ChemMerge, Uniform, and Oracle baselines. They map out a fundamental **Accuracy-Stability trade-off**: un-smoothed calibrated routing (SABLE + Layer Centroids) achieves the highest joint accuracy (**77.24%**) due to absolute local expert plasticity, while stateful temporal smoothing (Momentum-Merge Advanced) trades a minor fraction of classification accuracy to virtually eliminate routing jitter, dropping it by **76.2$\times$** to a near-zero **0.000374**.

---

## Strengths and Weaknesses

### 1. Soundness
**Rating: Excellent**
- **Strengths:** 
  - **Exemplary Experimental Hygiene:** The empirical evaluation is conducted across 10 independent random seeds. The authors provide seed-by-seed pairwise t-tests to confirm the statistical significance of their findings ($p < 0.01$ against SOTA ChemMerge, and $p < 0.05$ against SABLE), which is highly commendable.
  - **Intellectual Honesty in Boundary Analyses:** The Appendix contains highly insightful sweeps over the calibration subset size $|\mathcal{C}_k|$ and task-asymmetric noise regimes, mapping out the precise physical boundary conditions of the constant-inertia assumption of Momentum-Merge.
- **Weaknesses (Constructive Critiques):**
  - **Gap in Theorem 3.1 Proof Regarding Simplex Projection:** The proof of Theorem 3.1 asserts that for the discretized concentration vector $C^{(l)}$ to remain on the probability simplex, "conservation of mass" forces $\kappa \Delta t = k_{\text{decay}} \Delta t = \gamma$, which requires the rate constants to be identical: $\kappa = k_{\text{decay}}$. In physical chemistry, there is no thermodynamic or physical reason why the rate of expert species creation must equal its rate of degradation. Furthermore, as noted in Appendix A, ChemMerge utilizes a non-linear simplex projection mechanism. If an explicit projection step is used, the discretization step itself *does not* need to be naturally simplex-conserving! Therefore, the constraint $\kappa = k_{\text{decay}}$ is not a natural physical constraint, but an artificial mathematical restriction imposed by the authors to force an exact algebraic equivalence to EMA. Under a more general regime where $\kappa \neq k_{\text{decay}}$, the discretization yields $C_k^{(l)} = \gamma w_k^{(l)} + (1 - \delta) C_k^{(l-1)}$ followed by non-linear projection, which represents a more complex dynamical system that is not dual to constant EMA. The authors should clarify this distinction.
  - **Ecological Validity:** Evaluated purely on the synthetic coordinate sandbox. While Appendix B outlines a concrete scaling trajectory for physical Transformer deployment, actual empirical results on massive pre-trained Transformer architectures (e.g., LLaMA or Mistral) are not provided in the paper.

### 2. Presentation
**Rating: Excellent**
- **Strengths:** 
  - The paper is beautifully written, clear, and logically structured.
  - Figures 1 and 2 are professional and successfully convey the performance-jitter frontier and the momentum parameter sweeps.
  - The mathematical notation is highly consistent, precise, and rigorous.
- **Weaknesses:**
  - Highly significant physical findings—such as "Recurrence Trapping" under scarce calibration data and task-asymmetric noise boundary sweeps—are currently buried in Appendix D. Elevating these analyses to the main text would balance the paper's narrative.

### 3. Significance
**Rating: Good**
- **Strengths:** 
  - The paper delivers immense practical value for low-latency, resource-constrained edge serving by replacing an entire continuous ODE numerical integrator with a 1-line constant EMA update equation.
  - It carries significant philosophical value for the machine learning community, de-escalating the growing trend of wrapping simple mathematical operators in convoluted, pseudo-physical metaphors.
- **Weaknesses:**
  - The restriction to synthetic evaluations slightly limits its immediate real-world deployment significance, though the theoretical scaling trajectory in Appendix B provides a clear blueprint for practitioners.

### 4. Originality
**Rating: Excellent**
- **Strengths:** 
  - The conceptual deconstruction of continuous stateful routing into discrete-time EMA represents a highly original and refreshing contribution.
  - The characterization of "routing jitter" and the formal mapping of the "Accuracy-Stability trade-off" are novel and provide deep insights into the physics of deep representation flow.

---

## Overall Recommendation
**Rating: 5: Accept**
*Justification:* The paper is technically solid, theoretically grounded, and scientifically rigorous. Its deconstruction of ChemMerge's biochemical metaphors is a rare and highly original service to the community. While the lack of physical Transformer evaluations and a slight gap in Theorem 3.1's proof are minor weaknesses, the rigorous pairwise sweeps, statistical hygiene, and highly insightful boundary sweeps in the Appendix make this paper a strong contribution that others are highly likely to build on.

---

## Detailed Questions and Constructive Feedback for the Authors

### 1. Suggestion: Incorporating a Formal Mathematical Theory of Routing Jitter
While the paper treats "routing jitter" reduction as an empirical observation, it is possible to provide a rigorous, closed-form mathematical explanation of how the momentum parameter $\beta$ scales the jitter variance. We suggest the authors integrate the following noise-propagation derivation into Section 3:

Let the raw similarity-routing weight at layer $l$ be modeled as $w^{(l)} = w^{*(l)} + \epsilon^{(l)}$, where $w^{*(l)}$ is the clean, noise-free task routing vector and $\epsilon^{(l)}$ represents independent layer-wise noise with zero mean and isotropic variance $\sigma^2_w$. Under Momentum-Merge (constant EMA with parameter $\beta \in [0, 1]$):
$$\alpha^{(l)} = (1-\beta) w^{(l)} + \beta \alpha^{(l-1)}$$
Expanding this recurrence for large depth, the variance of the final ensembling weights $\alpha^{(l)}$ scales as:
$$\text{Var}(\alpha^{(l)}) \approx \frac{1-\beta}{1+\beta} \sigma^2_w$$
The Routing Jitter is defined as the expected squared difference between successive layers: $\mathbb{E}[(\alpha^{(l)} - \alpha^{(l-1)})^2]$. Substituting the EMA equation:
$$\alpha^{(l)} - \alpha^{(l-1)} = (1-\beta) (w^{(l)} - \alpha^{(l-1)})$$
Under the assumption of independent layer-wise noise, the covariance $\text{Cov}(w^{(l)}, \alpha^{(l-1)}) = 0$ because $\alpha^{(l-1)}$ only depends on past weights. Therefore, the variance of the step change at steady state is:
$$\text{Var}(\alpha^{(l)} - \alpha^{(l-1)}) = (1-\beta)^2 \text{Var}(w^{(l)}) + (1-\beta)^2 \text{Var}(\alpha^{(l-1)})$$
$$\text{Var}(\alpha^{(l)} - \alpha^{(l-1)}) \approx (1-\beta)^2 \sigma^2_w + (1-\beta)^2 \left(\frac{1-\beta}{1+\beta}\right) \sigma^2_w$$
$$\text{Var}(\alpha^{(l)} - \alpha^{(l-1)}) \approx \frac{2(1-\beta)^2}{1+\beta} \sigma^2_w$$

#### Theoretical Insights:
- This formula proves that the routing jitter is scaled by a factor of $f(\beta) = \frac{2(1-\beta)^2}{1+\beta}$ relative to the raw noise variance $\sigma^2_w$.
- For stateless routing ($\beta = 0$), $f(0) = 2$, meaning the jitter variance is $2\sigma^2_w$.
- For the empirical optimum ($\beta = 0.60$), $f(0.60) = \frac{2(0.40)^2}{1.60} = 0.20$. This represents an exact **10$\times$ reduction in jitter variance**, explaining the empirical trend observed in Table 3.
- Under high task-pool complexity ($K=10$), "distraction entropy" increases the raw noise variance $\sigma^2_w$. To keep the routing jitter stable and bounded, the network mathematically requires a larger optimal momentum coefficient $\beta = 0.80$ to apply a heavier damping scaling factor $f(0.80) = \frac{2(0.20)^2}{1.80} \approx 0.044$ (an over **45$\times$ reduction in jitter variance**), fully explaining the empirical shift observed in Table 5.
- Adding this formal derivation would elevate the paper's status from a purely empirical deconstruction to a highly complete theoretical contribution.

### 2. Elaborate on the "Recurrence Trapping" Vulnerability (Appendix D.3)
In Appendix D.3 (Table 4), the authors sweep $|\mathcal{C}_k|$ and expose a major vulnerability in Momentum-Merge (Advanced) under data scarcity ($|\mathcal{C}_k| \le 16$), which they term **Recurrence Trapping**:
- When calibration data is scarce ($|\mathcal{C}_k| = 8$), the computed layer centroids are noisy, making the initial boundary weight highly inaccurate. Because Momentum-Merge has stateful temporal memory, this initial boundary error propagates through network depth, trapping the ensembling coefficients in highly sub-optimal states throughout the forward pass and collapsing joint accuracy to **71.20%** (a 4.80% absolute degradation compared to stateless SABLE + LC, which achieves **76.00%**).
- **Theoretical Insight:** Stateless routing evaluates activations independently at each layer, making errors localized and allowing the network to recover in subsequent layers. Momentum-Merge (Advanced) initializes its stateful recurrence using Raw Boundary Initialization (Eq. 7). This couples the entire ensembling trajectory directly to the first layer's routing weight. Under noisy centroids, the initial weight acts as a biased prior, and the low-pass filter (high $\beta$) prevents the recurrence from adjusting quickly, "trapping" the ensembling weights.
- **Action:** We suggest moving this discussion to the main text, as it maps the exact physical boundary of the constant-inertia assumption and highlights the intellectual honesty of the work.

### 3. Clarification on the Equivalence Boundary of Theorem 3.1
The proof of Theorem 3.1 should explicitly clarify that the exact algebraic equivalence to EMA holds when ChemMerge operates in a simplified, projection-free parameter space where $\kappa = k_{\text{decay}}$. Under a more realistic regime where an explicit non-linear simplex projection mechanism is active, ChemMerge can operate with $\kappa \neq k_{\text{decay}}$, which introduces an additional degree of freedom not captured by the simple constant EMA. Please add a brief remark in Section 3.5.1 addressing this boundary.

### 4. Evaluation on Physical Transformer Architectures
While Appendix B provides a comprehensive scaling trajectory (Layer-wise Centroid Anchoring, Layer-wise Temperature Scaling, Depth-wise Momentum Modulation, and an experimental protocol), the main text remains synthetic. To achieve full ecological validity, we strongly encourage the authors to deploy Momentum-Merge on a physical Transformer architecture (e.g., LLaMA-7B or Mistral-7B) serving specialized LoRA task adapters, validating whether the near-zero jitter and robust accuracy trends translate directly to physical representational manifolds under inter-task overlap.
