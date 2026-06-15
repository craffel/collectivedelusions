# Peer Review of Conference Submission: Exclusive Parameter Merging

## Strengths and Weaknesses

### Soundness
*   **Strengths:**
    *   The paper is written in a highly structured, mathematical style. Equations are clearly stated and consistent.
    *   The mathematical interpretation of Soft-EPA in Equation 9 as a convex/linear combination of pure coordinate exclusivity and standard Task Arithmetic is an elegant way to conceptualize the parameter blending.
    *   The distinction between utilizing standardization exclusively as a decision-making filter for masking/routing and performing physical parameter updates in the original, unstandardized weight space is a sensible heuristic to avoid scale mismatch.
*   **Weaknesses:**
    *   **Absence of Formal Theoretical Grounding:** While the paper employs advanced mathematical terminology (e.g., "activation manifolds," "topology," "gradient physics"), it lacks any actual proofs, theorems, or formal mathematical guarantees. The core routing mechanism (Soft-EPA) is an empirical heuristic rather than a mathematically derived or bounded operator.
    *   **Loose Derivation of Dynamic Coherence Scheduling (DCS):** The authors justify the quadratic schedule $\gamma(p) = \gamma_0 + (1 - \gamma_0) \cdot p^2$ by claiming that coordinate-level collisions under independent random pruning masks scale proportionally to $(1-p)^2$. This derivation is theoretically flawed on several levels:
        1.  *Mask Correlation:* The assumption of independent, uniformly distributed pruning masks is highly questionable. Since all experts are fine-tuned from a shared pre-trained base model, their learned updates are highly correlated, not independent.
        2.  *Multi-Task Scale:* The quadratic collision probability of $(1-p)^2$ is only exact for $K=2$ tasks. For $K > 2$ tasks, the probability of at least two active coordinate collisions is more complex and does not scale quadratically.
        3.  *Heuristic Schedule:* No theoretical bounds or proofs are provided to show why $\gamma(p)$ scaling quadratically guarantees the prevention of representational collapse or capacity starvation.
    *   **Arbitrary Optimization Objective:** The validation metric $\mathcal{S}_{\text{val}} = \min_{k} \text{Acc}_k + 0.1 \cdot \text{Mean}(\text{Acc})$ used in TLC-Tune is a heuristic combination. The choice of the weight $0.1$ is highly arbitrary. There is no principled theoretical derivation for how this weight should be set to guarantee generalization or balanced multi-task learning.
    *   **Lack of Convergence and Generalization Guarantees for TLC-Tune:** The authors claim TLC-Tune is a highly stable and robust alternative to high-dimensional optimization because of its $K$-dimensional global scaling space. However, they provide no theoretical analysis of the convergence rate of the greedy zero-order (1+1)-ES search on non-convex, non-differentiable validation accuracy landscapes, nor any statistical generalization bounds (e.g., VC-dimension or Rademacher complexity) to prove why this 4-dimensional parameterization generalizes flawlessly from a tiny validation split of 128 samples per task to the test set.

### Presentation
*   **Strengths:**
    *   The manuscript is exceptionally well-structured, with a clear narrative flow.
    *   The visual conceptual overview (Figure 1) and the optimization study trajectory (Figure 2) are clean and informative.
    *   The mathematical notation is rigorous and clearly presented.
*   **Weaknesses:**
    *   **Unsupported Empirical Assertions:** In Section 4.3 (Sensitivity Analysis of Coherence Retention Factor $\gamma$), the authors write: *"To formalize this feature-routing coherence, we can analyze the activation manifolds using Centered Kernel Alignment (CKA) and t-SNE visualizations of layer-wise activations... CKA similarity between the merged model and the target experts decays exponentially with layer depth... recovering high CKA values and maintaining cohesive clustering of task classes in t-SNE projections..."* However, **there are no CKA plots, tables, or t-SNE visualizations anywhere in the paper or supplementary files.** Making detailed empirical conclusions about exponential CKA decay and clustering without actually providing the quantitative data or figures is a major scientific and presentation flaw.
    *   **Baseline Optimizer Mismatch:** The authors evaluate the continuous optimization baselines (AdaMerging and ZipMerge) by restricting their search to a non-differentiable validation score under a zero-order (1+1)-ES search. These baselines were natively designed to be optimized via first-order gradient descent on differentiable validation cross-entropy losses. Forcing a 56- or 70-dimensional non-convex search space to optimize via a greedy single-point random mutation search is mathematically expected to fail (under-converge), creating a biased baseline setup that does not represent their actual capabilities.

### Significance
*   **Strengths:**
    *   The paper tackles an important problem in weight-space model merging: resolving parameter conflicts without the heavy computational overhead of multi-stage test-time optimization.
    *   The detailed decoder-only LLM algorithm (Algorithm 1) provides useful guidance for practitioners trying to scale coordinate-level routing.
*   **Weaknesses:**
    *   **Extreme Performance Gaps:** Individual unmerged expert ceilings achieve accuracies of 91%–98% (with a Joint Mean Ceiling of 94.91%). The best-performing merged model (EPM TLC-Tune) achieves only **46.19%** joint mean accuracy under dense merging and **42.60%** under 50% sparsity. This represents an absolute performance collapse of **48% to 52%** compared to the expert ceilings. In a practical deployment, a model that operates at ~46% accuracy on simple classification tasks is barely functional, casting doubt on the practical significance of the proposed method under severe task conflicts.
    *   **Unrepresentative Backbone Scale:** The experiments are restricted to a compact ViT-Tiny backbone containing 5.7 million parameters. In these highly constrained regimes, weight-space properties are fundamentally different from modern billion-parameter LLMs/VLMs where model merging is primarily applied. The lack of scale generality limits the significance of the empirical findings.
    *   **The Minimax Optimization Zero-Sum Game:** In Table 1, untuned EPM ($\Lambda=\mathbf{1.0}$) achieves **68.89%** on CIFAR-10 and **59.41%** on SVHN. Applying TLC-Tune to optimize the worst-case floor elevates MNIST from 15.86% to 48.07% but **collapses CIFAR-10 from 68.89% to 36.98% (a massive 31.9% absolute drop)**. Sacrificing a highly complex visual classification task to slightly improve a trivial, grayscale digit classifier is a highly undesirable trade-off in practical deployment, demonstrating that TLC-Tune merely shuffles representation capacity in a zero-sum game rather than performing meaningful multi-task optimization.

### Originality
*   **Strengths:**
    *   The formulation of a soft coordinate-wise parameter routing protocol based on standardized absolute magnitude is a logical and interesting combination of coordinate-level masking, $z$-score standardization, and linear task arithmetic.
*   **Weaknesses:**
    *   **Incremental Novelty:** The individual components—task vector extraction, $z$-score standardization, magnitude-based thresholding, and (1+1) Evolution Strategy—are standard techniques from the literature. Mathematically, EPM's "cooperative soft exclusivity" (Equation 9) is simply an interpolation between hard exclusive coordinate-wise masking and standard Task Arithmetic. The originality lies in their empirical combination rather than any fundamental algorithmic or theoretical breakthrough.

---

## Technical Ratings

### Soundness: Fair
The mathematical style of the paper is clean, but the core methodologies (Soft-EPA, TLC-Tune, and DCS) are built entirely on empirical heuristics and lack formal proofs, theorems, or mathematical guarantees. The derivation of the Dynamic Coherence Scheduling (DCS) formula relies on a loose and highly questionable independent uniform mask assumption that is violated in practice. Furthermore, the validation size and optimization step sweeps do not provide rigorous generalization or convergence bounds.

### Presentation: Fair
The paper is well-written and structured. However, the authors make detailed empirical claims in Section 4.3 regarding Centered Kernel Alignment (CKA) decay and t-SNE clustering without actually presenting any CKA plots, t-SNE figures, or quantitative data. This is a severe presentation and scientific oversight. Additionally, evaluating gradient-based baselines under a zero-order optimizer mismatch represents a biased setup.

### Significance: Fair
The problem is highly relevant, but the absolute accuracies achieved (~46% average compared to ~95% expert ceilings) are too low for practical, production-level deployment. The minimax optimization (TLC-Tune) behaves as a zero-sum game, severely degrading the performance of complex visual tasks to boost trivial digit classifiers. The evaluation is limited to an unrepresentative 5.7M parameter ViT-Tiny model, lacking scale generality.

### Originality: Fair
The soft parameter routing operator is an interesting heuristic, but its mathematical formulation is an incremental interpolation between hard routing and standard Task Arithmetic. No new optimization theory or algorithmic paradigms are introduced.

---

## Overall Recommendation

**Score: 3: Weak Reject**

**Detailed Rationale:**
This submission presents an elegant and well-written exploration of coordinate-level routing in weight-space model merging. However, from a rigorous, theory-minded perspective, the paper falls short of the standards of a top-tier machine learning conference:

1.  **Lack of Theoretical Foundations:** The submission utilizes mathematical notation to present empirical heuristics but fails to provide any actual proofs, theorems, or formal guarantees. Concepts like "activation manifold alignment," "representational coherence," and "generalizability" are discussed descriptively but never formalized mathematically.
2.  **Unsupported Empirical Claims:** The detailed claims regarding CKA similarities and t-SNE activations in Section 4.3 are completely unsupported, as no CKA tables, t-SNE plots, or underlying data are presented in the manuscript.
3.  **Loose Derivations:** The derivation of the quadratic Dynamic Coherence Scheduling (DCS) schedule rests on a questionable independent random mask assumption that is violated in practice given shared model initializations.
4.  **Zero-Sum Minimax Trade-offs:** TLC-Tune's minimax optimization behaves as a zero-sum capacity shifter, collapsing CIFAR-10 performance by nearly 32% absolute accuracy to pull MNIST from 15.86% to 48.07%, which lacks practical utility.
5.  **Optimizer Mismatch:** The continuous optimization baselines (AdaMerging and ZipMerge) are evaluated under an artificial zero-order ES optimizer mismatch rather than their native first-order gradient descent pipelines, leading to a biased comparison.

To raise the paper to an Accept, the authors should:
- Provide formal mathematical proofs or Rademacher generalization bounds for TLC-Tune's low-dimensional parameter space.
- Mathematically model coordinate-level mask correlations to derive a principled coherence schedule.
- Include the actual CKA similarities and t-SNE plots mentioned in Section 4.3, or remove those unsupported claims entirely.
- Resolve the optimizer mismatch by evaluating AdaMerging and ZipMerge under their native, first-order gradient-based validation pipelines.
- Verify the scale generality of EPM empirically on modern Large Language Models (LLMs) with billions of parameters.
