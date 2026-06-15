# Peer Review for ICML 2026

## Paper Title
**Grassmannian Subspace Consensus Merging: A Spectral Filter for Multi-Task Parameter Alignment**

---

## 1. Summary of the Paper
This paper introduces **Grassmannian Subspace Consensus Merging (GSC-Merge)**, a mathematically rigorous and provably optimal framework for weight-space model merging. Model merging is a computationally efficient strategy to consolidate multiple specialized expert models into a single multi-task network. However, existing methods suffer from parameter interference and representation collapse, and rely heavily on discrete, coordinate-wise heuristics (such as sign voting or absolute magnitude thresholding) that discard structural correlations and lack theoretical guarantees.

To address this, GSC-Merge targets the major linear projection layers inside Transformer blocks (comprising over 95% of the block parameters) while keeping lightweight normalization, biases, and embedding layers task-specific. It constructs a joint multi-task update matrix representing the downstream adaptation trajectories of all experts, and performs Singular Value Decomposition (SVD) to project these updates onto a shared low-rank Grassmannian manifold $\mathbf{Gr}(r, d_{out})$. 

The paper's core contributions are:
1. **Mathematical Optimality:** Leveraging the Eckart-Young-Mirsky Theorem to prove that SVD projection yields the mathematically optimal low-rank reconstruction of multi-task updates under the Frobenius norm, minimizing representation drift.
2. **Resolution of the Overfitting-Optimizer Paradox:** Exposing and mathematically proving that unconstrained optimization of layer-wise merging coefficients on small validation sets (OFS-Tune) is prone to transductive overfitting to validation noise. Proving that the Grassmannian projection operator serves as an implicit spectral regularizer (acting as a non-strict contraction in both spectral and Frobenius norms) that bounds the optimization space and reduces split-sensitivity variance.
3. **Rigorous Empirical Evaluation:** Testing GSC-Merge using a Vision Transformer (ViT-Tiny) backbone across four highly conflicting datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) over 5 independent validation calibration splits. GSC-Merge consistently outperforms coordinate-wise baselines (Uniform, STA, TIES-Merging, Task Arithmetic) and achieves competitive accuracy to unconstrained tuning but with significantly reduced variance across splits.
4. **Pragmatic Scalability & Architectural Analysis:** Demonstrating SVD scalability via **Randomized SVD** (achieving a massive 23.56$\times$ speedup on LLaMA-7B sizes with $<2.5\%$ relative error), validating SVD projection directions (Left/Output-Space dramatically outperforming Input-Space and Bilateral projections), and analyzing performance under both task-conditional swapping and truly task-agnostic settings.

---

## 2. Strengths of the Paper
- **Rigorous Mathematical Foundation:** The paper provides a much-needed theoretical grounding to the largely heuristic field of model merging. It connects the Eckart-Young-Mirsky Theorem to representation drift and provides an elegant proof for the implicit regularizing effect (non-strict contraction) of Grassmannian projection.
- **Deep Geometric & Architectural Insight:** The empirical pilot study and mathematical justification for output-space (left-singular) SVD projection over input-space (right-singular) SVD projection (Appendix C) is exceptionally insightful, connecting matrix factorization properties directly to downstream coordinate alignment.
- **Addressing the Overfitting-Optimizer Paradox:** The identification and resolution of transductive overfitting in OFS-Tune is a highly realistic, original, and impactful contribution, offering a robust bias-variance trade-off that is thoroughly verified via a multi-seed split analysis.
- **Exemplary Experimental Rigor:** The authors go far beyond typical merging papers. They execute sweeps on all baselines (TA scaling factors, STA/TIES-Merging pruning thresholds) to prevent under-tuning bias, evaluate over 5 independent random splits to report variance, and perform an extensive ablation on a truly task-agnostic setting.
- **Excellent Transparency regarding Limitations:** The paper is highly honest and transparent about the remaining performance gap between GSC-Merge and the expert ceiling under highly conflicting task suites. To address this, the authors propose a promising hybrid routing framework (**GSC-Route**) in the appendix.
- **Excellent Computational Scalability:** The introduction of Randomized SVD and the corresponding benchmarks on LLaMA-7B sizes (Appendix A) successfully mitigate any potential CPU/time scalability concerns.

---

## 3. Weaknesses/Limitations
Given the exceptional quality of the paper, there are no critical flaws. However, there are minor limitations and areas that can be further clarified:
1. **The Inherent Performance Gap:** Although GSC-Merge outperforms coordinate-wise merging baselines and provides elegant spectral regularization, there remains a notable performance gap between the merged model ($42.13\%$) and the expert ceiling ($74.96\%$) in highly conflicting settings. Under a truly task-agnostic setting, this gap is even wider. While the authors discuss this in Section 4.3 and formulate the hybrid **GSC-Route** in Appendix D.2 to address it, the static merged model's performance on highly conflicting domains is still bounded.
2. **Applicability to Fully Non-Task-Conditional Settings:** While the ablation study in Section 4.4 shows that GSC-Merge is highly competitive when non-target parameters are fixed to pre-trained base values, the overall accuracy drop across all methods in this setting is severe. Resolving representation statistics and normalization boundaries in a purely task-agnostic manner remains a critical bottleneck for the entire model merging paradigm.
3. **Mathematical Looseness Regarding the Optimizer's Search Space:** In Section 3.6 (Proposition 3.2), the authors state that GSC-Merge "restricts the active parameter search space of the optimizer from $d_{out}$ dimensions to a low-dimensional $r$-dimensional subspace." This is mathematically loose. The learnable parameters being optimized are the merging coefficients $\alpha^{(l)} \in \mathbb{R}^K$. Therefore, the optimizer's parameter search space has dimension $K$ (the number of tasks, e.g., 4) in both unconstrained OFS-Tune and GSC-Merge. The true regularizing effect of GSC-Merge does not come from reducing the optimizer's parameter degrees of freedom, but rather from projecting the task vectors $V_k^{(l)}$ onto the low-rank subspace. This ensures that the blended task updates $\sum_k \alpha_k^{(l)} \tilde{V}_k^{(l)}$ are restricted to a $K$-dimensional subspace of the $r$-dimensional consensus space, filtering out the high-frequency/task-specific noise contained in the tail singular vectors of $\{V_k^{(l)}\}$. The authors should clarify this distinction to maintain absolute mathematical precision.
4. **The Mean Generalization Trade-off (The Bias-Variance Nuance):** The paper claims that GSC-Merge "resolves" the Overfitting-Optimizer Paradox by "substantially boosting out-of-distribution generalizability on unseen test data." However, a close look at the tables reveals that GSC-Merge does *not* actually achieve higher mean test accuracy than unconstrained OFS-Tune in either Table 1 or Table 2.
   - In Table 1 (Task-Conditional Swapping), unconstrained OFS-Tune achieves a joint mean accuracy of $44.08 \pm 4.31\%$, whereas the best GSC-Merge configuration ($\gamma=0.5$) achieves $43.88 \pm 4.07\%$, and GSC-Merge ($\gamma=0.3$) achieves $42.13 \pm 2.76\%$.
   - In Table 2 (Truly Task-Agnostic setting), unconstrained OFS-Tune achieves $20.86 \pm 4.81\%$, while GSC-Merge ($\gamma=0.5$) achieves $20.61 \pm 4.80\%$.
   While GSC-Merge successfully reduces the variance across calibration splits (e.g., standard deviation of $\pm 2.76\%$ at $\gamma=0.3$ compared to $\pm 4.31\%$ for unconstrained), it does so at the cost of slightly lower mean performance. This indicates a minor underfitting/representation bias from the low-rank projection. The claim that it "resolves" the Overfitting-Optimizer Paradox by boosting out-of-distribution generalizability is therefore slightly overstated or at least needs to be reframed as a classic bias-variance trade-off (trading a small drop in average accuracy for a major reduction in split-sensitivity variance).

---

## 4. Questions and Actionable Suggestions for the Authors
- **Adaptive Layer-Wise Ranks:** In the conclusion, you suggest an adaptive layer-wise spectral thresholding scheme based on local singular value decay. Since you have already extracted and plotted the singular decay spectra in Appendix B, did you observe whether certain layers (e.g., MLP contraction vs. attention projections) exhibit a much faster decay? Highlighting which layers can tolerate extremely compact ranks (e.g., $\gamma = 0.1$) vs. those that require more capacity (e.g., $\gamma = 0.5$) would be highly valuable for practitioners.
- **Gradient-Free Coefficient Search in Low Data Regimes:** Section 3.8 discusses CMA-ES and Bayesian Optimization as alternatives to Adam for coefficient search. Have you conducted any preliminary runs comparing gradient-free optimization to Adam? If derivative-free methods show further reduction in validation split sensitivity or can operate on fewer steps/samples, this would be an excellent practical tip to include.
- **Evaluating on More Modern/Larger Backbones:** Vision Transformer (ViT-Tiny) is a relatively small model. While you include Randomized SVD CPU benchmarks on a LLaMA-7B sized layer in the appendix to prove computational scalability, actually evaluating the end-to-end multi-task merging accuracy on larger models (such as ViT-Base, ViT-Large, or LLaMA models) would significantly strengthen the empirical claims of the paper and show whether the bias-variance trade-off behaves similarly in high-capacity regimes.
- **Formatting Minor:** In Table 1, the Uniform Merging and Task Arithmetic rows don't have rank entries, which is correct, but adding a "N/A" or "—" in the Rank column makes the table cleaner.

---

## 5. Overall Ratings and Recommendation

### Ratings
- **Soundness:** Excellent
- **Presentation:** Excellent
- **Significance:** Excellent
- **Originality:** Excellent

### Recommendation
**6: Strong Accept**
*Rationale:* This is a technically flawless, mathematically elegant, and highly rigorous paper that addresses a crucial bottleneck in multi-task model consolidation. By replacing heuristic coordinate-wise "hacks" with principled spectral projections onto the Grassmannian, the paper provides a solid theoretical and geometric foundation for model merging. The analysis of the Overfitting-Optimizer Paradox is highly original, and the empirical evaluations—backed by extensive appendices addressing scalability, projection directions, and future routing frameworks—are exceptionally thorough. This paper is highly ready for publication and has the potential to significantly influence future research in model merging, PEFT/LoRA adapter fusion, and parameter routing.
