# 5. Impact and Presentation Quality

## Major Strengths
1. **Exceptional Empirical Honesty and Transparency:**
   The authors deserve immense credit for their willingness to highlight a negative result—that no test-time adaptive configuration outperforms the zero-overhead static Uniform baseline of 30.41%. In machine learning literature, negative results are often buried or manipulated. This paper's empirical transparency is refreshing, highly credible, and provides a significant service to researchers and practitioners alike.
   
2. **Elegant Deconstruction of Zero-Order Dynamics:**
   The mathematical deconstruction of 1+1 ES's robustness (the "sluggishness hypothesis") is brilliant. Rather than accepting the surface-level conclusion that ES is a superior optimizer, they rigorously explain that its performance rise at higher dimensions is due to optimization failure (underfitting), which preserves the robust initialization. This is a high-signal, high-quality insight.

3. **Polished Structure and Presentation:**
   The paper is beautifully written, highly structured, and easy to follow. The transition from task-vector formulation to the nested hierarchy, optimizer dynamics, and regularizers is logical and seamless. The tables and formulas are formatted to standard LaTeX publication quality.

---

## Areas for Improvement

1. **Practical Generalizability and Scale:**
   To have a real-world impact, the authors must validate whether the "Generalization-Granularity Trade-off" holds on high-fidelity, fully converged experts and larger backbones (e.g., CLIP-Large, LLaMA-7B). In high-fidelity regimes, task vectors are less noisy, and representation structures are more stable. It is possible that fine-grained merging *can* outperform uniform baselines in these settings, making this evaluation crucial.

2. **Computational and Efficiency Evaluation:**
   Adding a table or discussion comparing the computational costs (number of forward/backward passes, latency, memory overhead, edge-deployment feasibility) of the different granularities and optimizers would significantly enhance the paper's value for practitioners.

3. **Hyperparameter Tuning Feasibility:**
   Provide a sensitivity analysis of the ESR and TV regularizers ($\beta, \gamma$) and discuss how practitioners can set these hyperparameters at test-time when no labeled validation set is available.

4. **Missing Direct Competitors:**
   Incorporate a direct empirical benchmark against RegCalMerge (Jin et al., 2026), which specifically addresses transductive overfitting in multi-task model merging.

---

## Overall Presentation Quality
**Excellent.** The paper represents a high-standard academic write-up. The narrative is cohesive, the arguments are well-reasoned, and the empirical analyses are exceptionally honest and diagnostic.

---

## Potential Impact and Significance
- **Aesthetic and Conceptual Impact:** High. The paper acts as a valuable "cautionary tale" for the model merging community, establishing a clear warning boundary against over-parameterizing test-time adaptation. It exposes prediction entropy minimization as a potentially hazardous surrogate loss on compact test-time streams.
- **Practical Utility Impact:** Low. Because the proposed methods (ESR and TV regularizers at Level 5) still fail to beat the zero-overhead, static Uniform baseline, a practitioner reading this paper will conclude that they should simply use the static baseline and avoid test-time adaptation entirely. However, this is still a highly significant practical takeaway, as it saves deployment engineers from wasting computational resources.
