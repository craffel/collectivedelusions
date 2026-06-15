# Peer Review: Exclusive Parameter Merging: Coherence-Preserved Multi-Task Model Fusion

## 1. Summary of the Paper
The paper addresses the challenge of **spatial weight-space interference** in weight-space model merging. When task-specific experts fine-tuned from a shared pre-trained base model are combined into a single unified network, standard average-based weight interpolation (e.g., Task Arithmetic, TIES-Merging) dilutes task-specific features, leading to representational collapse. 

To mitigate this, the authors propose **Exclusive Parameter Merging (EPM)**, which consists of:
1. **Soft Exclusive Parameter Allocation (Soft-EPA)**: A coordinate-wise relative routing operator. For each coordinate, it routes the dominant expert's update (determined via absolute magnitudes standardized by task-specific standard deviations $\sigma_k$) at full strength, while attenuating non-dominant updates by a coherence retention factor $\gamma = 0.2$ (dense). This "softness" acts as a continuous topological "glue" that preserves multi-layer activation manifold alignment. To resolve capacity starvation under sparse merging, the authors introduce **Dynamic Coherence Scheduling (DCS)**, where $\gamma(p)$ dynamically scales with the target network sparsity $p$ via a quadratic rule: $\gamma(p) = \gamma_0 + (1 - \gamma_0) \cdot p^2$.
2. **Task-Level Coefficient Tuning (TLC-Tune)**: A minimalist zero-order (1+1) Evolution Strategy (ES) that optimizes only $K$ global scaling factors (one per expert) on a tiny validation split (128 samples per task). It optimizes a balanced minimax validation score to raise the worst-performing task's accuracy.

The authors evaluate EPM on a compact Vision Transformer (ViT-Tiny, 5.7M parameters) across four disjoint vision benchmarks: MNIST, FashionMNIST, CIFAR-10, and SVHN. Under dense merging ($p=0.0$), TLC-Tuned EPM achieves a joint mean accuracy of **46.19% $\pm$ 0.14%**, outperforming Task Arithmetic (40.96%) and TIES-Merging (20.55%).

---

## 2. Strengths
* **Exemplary Scientific Honesty and Self-Critique**: The paper is outstandingly transparent. The authors dedicate entire subsections to thoroughly analyzing and documenting the limitations of their work, including the "exclusivity contradiction," the underperformance under extreme sparsity ($p=0.8$), and the optimizer mismatch during baseline comparisons. This level of self-awareness is highly refreshing and scientifically rigorous.
* **Rigorous and Elegant Theory**: The mathematical proof of Equation 9 (demonstrating that Soft-EPA is a convex/linear combination of pure coordinate exclusivity and standard Task Arithmetic) is highly elegant. It provides clear theoretical intuition for the role of $\gamma$ as a background regularizer, which is further validated by a Centered Kernel Alignment (CKA) and t-SNE analysis of activation manifolds.
* **Well-Reasoned Scale Decoupling**: The decision to decouple standardized task-vector routing filters from unstandardized physical weight updates is highly justified. The empirical analysis of "scale overrides" (occurring at exactly 13.79% of coordinates) provides strong evidence that standardizing physical parameters would distort pre-trained activation physics.
* **Rigorous Baseline and Optimization Analyses**: The authors conduct a systematic 500-step optimization study showing that high-dimensional continuous optimization methods (AdaMerging, 56 params; ZipMerge, 70 params) remain completely flat and stuck under (1+1)-ES, empirically proving that their failure is due to absolute optimization failure (under-convergence) rather than transductive overfitting.
* **Exceptional Presentation and Readability**: The paper is masterfully structured, highly professional, and easy to read. The discussions surrounding paradigm trade-offs (Model Merging vs. LoRA vs. MTL), the direct LLM routing algorithm (Algorithm 1), and the professional trajectory figures (Figure 2) add immense pedagogical value and practical utility.

---

## 3. Weaknesses & Areas of Improvement

### Weakness 1: Toy Scale of Backbone and Disjoint Modality of Experimental Setup
* **Critique**: The empirical evaluation is restricted to a compact Vision Transformer (ViT-Tiny, 5.7M parameters) fine-tuned on highly disjoint datasets (grayscale digits vs. natural color objects).
* **Implication**: While this setup is indeed a rigorous stress test of weight-space interference, it represents a highly artificial "toy" environment. Practical model merging is primarily utilized on Large Language Models (LLMs) or CLIP-like models, where the backbone has massive parameter redundancy (7B+ parameters) and tasks are much more related (e.g., separate math or coding experts). Merging disjoint experts on a compact network leads to extremely low absolute accuracies (around **40%--46%** average vs. the **$\sim$95%** expert ceilings). In a real-world scenario, no practitioner would accept a 50% performance drop, limiting the immediate practical serving value of the findings.
* **Recommendation**: While the authors provide an excellent theoretical discussion on how EPM scales to LLMs and a clear decoder-only implementation algorithm (Algorithm 1), the lack of actual empirical evaluation on at least medium-sized models (e.g., Llama-3-8B or CLIP ViT-B/16) remains the primary bottleneck restricting the paper's immediate impact.

### Weakness 2: Underperformance and Capacity Starvation under High Sparsity
* **Critique**: Under moderate and extreme sparsity, EPM's coordinate exclusivity is outperformed by baseline techniques:
  * **At $p=0.5$ (Moderate Sparsity)**: Although TLC-Tuned EPM (**42.60%**) outperforms the DARE baseline (**40.94%**), it is significantly outperformed by the **Standardized TA + Pruning** baseline (**44.87%**).
  * **At $p=0.8$ (Extreme Sparsity)**: Even with Dynamic Coherence Scheduling (which successfully rescues EPM from a 24.11% collapse to **26.41%**), EPM is heavily outperformed by DARE (**40.90%**), which maintains a robust joint mean on every task.
* **Implication**: This reveals that Soft-EPA's coordinate routing and secondary coordinate-wise attenuation starve the model of representational capacity under high pruning constraints. DARE, which utilizes expected-value scaling ($1/(1-p) = 5$) to preserve deep manifold activation scales under extreme deletion, is consistently superior under compressed merging regimes. The claim that EPM is a robust operator under high sparsity is undermined by these findings.

### Weakness 3: The Practicality of the Minimax Objective (Zero-Sum Trade-off)
* **Critique**: TLC-Tune optimizes for a balanced minimax objective to raise the worst-performing task's accuracy.
* **Implication**: While this minimax formulation successfully lifts the joint average accuracy floor, it results in a severe **zero-sum trade-off**:
  * Under dense merging ($p=0.0$), TLC-Tuned EPM lifts MNIST from **15.86%** to **48.07%** and FashionMNIST from **38.31%** to **46.42%**.
  * However, this comes at the cost of **collapsing** the harder color tasks: CIFAR-10 collapses from **68.89%** down to **36.98%**, and SVHN drops from **59.41%** to **53.28%**.
  * In practice, sacrificing over 30% absolute accuracy on a highly complex dataset (CIFAR-10) to raise a trivial, grayscale digit dataset (MNIST) from 15% to 48% is highly undesirable, as MNIST features are simple and can be easily processed with minimal on-device resources.
* **Recommendation**: Although the authors discuss the Pareto frontier and propose alternative utility functions (such as Joint Mean optimization, weighted multi-task objectives, or constraint-based objectives), a more thorough discussion of these trade-offs under wildly disparate task difficulties in the main experimental analysis would strengthen the practical deployment narrative.

---

## 4. Detailed Comments and Constructive Suggestions

1. **Empirical Evaluation of Layer-wise Task Vector Standardization**:
   * *Issue*: The authors mathematically formulate **Layer-wise Task Vector Standardization** (Equations 4 and 5) and provide an elegant comparative analysis of scale granularity, but do not report extensive empirical results comparing global and layer-wise standardization.
   * *Suggestion*: In future extensions of this work, the authors should run empirical sweeps comparing global vs. layer-wise standardization across different network layers. This would provide concrete evidence on whether layer-wise scale normalization stabilizes feature routing in deep networks.
2. **Transitioning to Large-Scale Generative Models**:
   * *Issue*: The authors discuss LLM scaling and provide a detailed algorithm (Algorithm 1) for decoder-only architectures, but the empirical weight is restricted to vision tasks.
   * *Suggestion*: In follow-up studies, the authors should prioritize empirical merging tests on 1B or 3B specialized language models (e.g., merging code-expert and math-expert models fine-tuned from a shared base). This would bridge the theoretical LLM routing blueprints with real-world generative AI applications.
3. **Advanced Zero-Order Optimization**:
   * *Issue*: The authors discuss how a greedy (1+1)-ES gets trapped in localized saddle points under intermediate calibration sizes ($N_{\text{val}}=512$), and show that population-based CMA-ES successfully stabilizes search convergence to achieve **44.91%** accuracy.
   * *Suggestion*: The authors should consider transitioning TLC-Tune permanently to a population-based CMA-ES or population-based evolutionary strategy, which mathematically estimates landscape covariance and is guaranteed to scale to larger numbers of expert models ($K$) without getting trapped in local minima.

---

## 5. Overall Recommendation and Ratings

### Overall Recommendation: Accept (5)
The paper is technically solid, exceptionally well-written, and mathematically elegant. It introduces a creative, theoretically grounded coordinate-exclusive routing pipeline with a soft background blend, scale-decoupled integration, and dynamic coherence scheduling. What sets this paper apart is its exemplary scientific honesty and transparent self-critique, which adds immense pedagogical value and structural integrity. Although the toy scale of the backbone and the collapse under extreme sparsity relative to DARE limit its immediate practical impact, the paper is of high quality and makes a strong contribution to the model-merging literature, fully justifying an Accept.

### Ratings:
* **Soundness**: Excellent (4/4)
* **Presentation**: Excellent (4/4)
* **Significance**: Good (3/4)
* **Originality**: Good (3/4)
