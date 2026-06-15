# Peer Review

## Summary of the Submission
This paper presents **Q-PolyMerge**, a parameter-efficient, quantization-aware test-time adaptation (TTA) framework for multi-task model merging on edge devices. Multi-task model merging consolidates task-specific experts sharing a base initialization directly in the parameter space. However, combining merging with post-training quantization (PTQ) to low-bit integers introduces representation alignment noise (Merge-then-Quantize) or breaks continuous linear mode connectivity (Quantize-then-Merge). 

To resolve this, previous test-time adaptation methods (e.g., AdaMerging) optimize layer-wise merging coefficients on small streams of unlabeled calibration images via entropy minimization, but they suffer from what the authors call the **Overfitting-Optimizer Paradox**: unconstrained high-dimensional search over layer-wise coefficients fits transductive noise, producing jagged, physically nonsensical trajectories.

Q-PolyMerge resolves this by restricting the search space of merging coefficients to a low-dimensional continuous polynomial subspace of normalized layer depth:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
where $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ represents the learnable parameters of degree $d$ (typically $d=2$). This formulation reduces the parameter search space by over 78% (e.g., from 56 to 12 parameters for a ViT-Tiny backbone). 

The paper outlines two optimization pathways:
1. **First-Order Optimization via Straight-Through Estimator (Adam STE):** Backpropagation through the non-differentiable rounding operator.
2. **Zero-Order Optimization via 1+1 Evolution Strategy (1+1 ES):** A derivative-free black-box search that bypasses activation caching entirely, reducing peak volatile memory (SRAM) footprint from **158.40 MB** to **4.05 MB** (a 97.5% reduction) under 4-bit per-channel PTQ.

The method is evaluated on a Vision Transformer (ViT-Tiny) across four benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) under 8-bit and 4-bit PTQ, demonstrating stabilized optimization trajectories and competitive accuracies.

---

## Strengths
1. **Compelling Systems-Level Motivation and Mathematical Footprint Derivation:** 
   The paper is highly meticulous in its systems-level justification. The detailed mathematical derivation of the 158.40 MB activation cache cached under PyTorch autograd for a ViT-Tiny backbone is extremely clear and highlights a genuine physical bottleneck for on-device backpropagation.
2. **Exceptional Parameter and Compute Efficiency:** 
   Projecting the layer-wise coefficients onto a low-degree polynomial subspace dramatically reduces the search dimension. Under the zero-order pathway, this compact space enables the 1+1 ES to converge in just 100 iterations (100 forward passes, 0 backward passes), making it 16.7% computationally cheaper than first-order gradient descent, which is traditionally a major latency bottleneck for black-box search.
3. **High-Quality Presentation:** 
   The paper is beautifully written, exceptionally well-structured, and easy to follow. The mathematical notation is consistent, the figures are clean, and the literature is thoroughly contextualized with concurrent quantization-aware merging works (such as TVQ, E-PMQ, and 1bit-Merging).

---

## Weaknesses
While the systems-level motivations and engineering parameterizations are highly commendable, the paper exhibits several critical weaknesses regarding **theoretical rigor, mathematical consistency, and experimental design**:

### 1. Lack of Formal Theoretical Grounding for "Low-Pass Filtering" Claims
The authors repeatedly state that constraining the merging coefficients to a continuous, low-degree polynomial subspace "acts as a mathematical low-pass filter, filtering out high-frequency optimization noise and preventing overfitting." 
* **The Theoretical Gap:** This is used as a qualitative metaphor rather than being theoretically established. A formal low-pass filter must be defined in the frequency domain (e.g., Fourier or spectral representation of the parameter updates across layer indexes). The paper lacks any spectral analysis of the parameter trajectories or gradient updates, and does not define what "high-frequency optimization noise" means in the context of layer-wise coefficient optimization.

### 2. Discontinuity-Continuity Mismatch in 4-Bit Zero-Order Search
The uniform post-training rounding operator $q(W)$ is a step function, which divides the test-time loss landscape $\mathcal{L}_{\text{TTA}}(\boldsymbol{\alpha})$ into flat plateaus separated by sharp, discontinuous step-cliffs.
* **The Mismatch:** The authors propose a *smooth, continuous* polynomial constraint. While this works well for first-order gradient descent (approximated via STE), under the zero-order 1+1 ES pathway, **Block-wise Constant (ES) scaling slightly outperforms Q-PolyMerge (ES) by 0.28%** (43.33% vs. 43.05%, Table 3).
* **Implication:** This empirical anomaly indicates that forcing a smooth polynomial trajectory on a discontinuous rounding landscape is mathematically mismatched for derivative-free search. The hard boundaries of block-wise constant parameterization provide localized step-perturbations that help random search escape flat plateaus, whereas the smooth polynomial constraint restricts mutations to a global continuous trajectory, smoothing out the localized adjustments needed to navigate fragmented rounding landscapes. This contradicts the authors' general claim of mathematical superiority across both optimization pathways.

### 3. Lack of Generalization Bounds for the "Overfitting-Optimizer Paradox"
The core thesis is that unconstrained layer-wise optimization overfits on tiny streams of 16 images, whereas the polynomial subspace generalizes well.
* **The Theoretical Gap:** While this is intuitive, there is no formal statistical learning theory analysis (e.g., Rademacher complexity, VC dimension, or generalization error bounds) to mathematically prove this generalization gap in the transductive setting. The paper bypasses formal learning bounds, relying instead on purely empirical observations.

### 4. Speculative Chebyshev Condition Number Analysis
In Appendix B.7, the authors analyze the 2-norm condition number of the monomial Vandermonde matrix $V_{L, d}$ across varying polynomial degrees $d$.
* **The Critique:** For the quadratic setting ($d=2$) utilized in all of their experiments, the condition number is $\approx 20$, which is highly well-conditioned. Thus, the ill-conditioning problem does not exist in their actual implementation. Furthermore, the discussion of Chebyshev polynomials preventing "Runge's phenomenon" is highly speculative. Runge's phenomenon is a classic property of polynomial interpolation of specific functions on equidistant grids. Model merging is a variational optimization problem (entropy minimization), not an interpolation of a fixed function. The authors provide no proof or evidence that Runge's phenomenon actually occurs for neural network layer coefficients, and they do not conduct any experiments with Chebyshev polynomials to show an empirical difference.

### 5. Weaknesses in the Experimental Setup
* **Toy-Scale Evaluation on Mismatched Datasets:** Evaluating a pre-trained Vision Transformer (ViT-Tiny, 5.7M parameters) on **MNIST and FashionMNIST** is highly artificial and representative of a toy-scale setup. MNIST consists of simple greyscale $28 \times 28$ images, which is not a representative workload for a pre-trained ViT.
* **Underfitted experts:** The individual experts are trained on only 512 samples per dataset, resulting in weak, underfitted baseline models (e.g., 8-bit experts get only 79.05% average accuracy). It is unclear if their proposed regularization benefits would hold for highly optimized, fully converged experts.
* **Performance Gap vs. Offline Optimization:** In both 8-bit and 4-bit regimes, standard **AdaMerging (Adam $\to$ Low-Bit) offline optimization strictly outperforms Q-PolyMerge**:
  - *8-Bit PTQ:* AdaMerging (Adam $\to$ 8-Bit) achieves **62.27%** vs. Q-PolyMerge (Adam) at **59.76%**.
  - *4-Bit PTQ:* AdaMerging (Adam $\to$ 4-Bit) achieves **50.20%** vs. Q-PolyMerge (Adam) at **48.87%**.
  This raises a major conceptual question: **Why adapt on-device at all if practitioners can simply perform the adaptation offline on a server (where they have unlimited SRAM) prior to deployment, achieving superior performance?** 
* **Statistical Rigor:** The experiments are conducted on only **3 random seeds**. Given the high standard deviations observed in some treatments (e.g., standard deviation of up to 21.12% in Table 1), 3 seeds are statistically insufficient to establish confidence.

---

## Rubric Ratings

### Soundness: Fair
The systems-level derivations of SRAM memory footprints and activation caches are sound and meticulous. However, the theoretical foundations of the paper are weak. The core claims (such as "low-pass filtering" and "Runge's phenomenon") are qualitative metaphors rather than proven mathematical properties. There is a fundamental mathematical mismatch between the smooth polynomial prior and the discontinuous rounding landscape under zero-order search, which is empirically reflected in the block-wise constant scaling outperforming Q-PolyMerge. Additionally, there are no convergence proofs for the STE under the Vandermonde constraint.

### Presentation: Excellent
The writing is exceptionally clear, precise, and well-structured. The narrative is easy to follow, and the mathematical formulations of the polynomial projection and post-training quantization are well-articulated. The appendices are exhaustive and cover a wide range of practical deployment concerns.

### Significance: Fair
While the practical SRAM and compute reductions are highly valuable for the edge-AI community, the theoretical significance is modest. The polynomial parameterization is a standard parameter-reduction heuristic rather than a novel theoretical paradigm. Because standard offline adaptation followed by post-hoc quantization strictly outperforms their direct on-device method, the practical necessity of on-device adaptation remains conceptually questionable.

### Originality: Good
The paper provides a creative combination of polynomial subspace constraints, test-time adaptation via entropy minimization, and quantization-aware model merging. While the individual components are standard (e.g., STE, 1+1 ES, Tent/AdaMerging), their synthesis into a unified, parameter-efficient framework for edge deployment is original and interesting.

---

## Overall Recommendation
**Rating & Score: 3: Weak Reject**

**Justification:**
This submission has clear engineering and systems-level merits, particularly in identifying the activation-cache memory bottleneck of first-order test-time adaptation and demonstrating how parameter-space projection can unlock SRAM-efficient zero-order search. However, from a theoretical perspective, the paper lacks mathematical rigor. The central "low-pass filter" claims are qualitative analogies, and there is a mathematical mismatch between the smooth polynomial constraint and the discontinuous rounding landscape under zero-order optimization (as shown by block-wise constant scaling outperforming the proposed method). Additionally, the evaluation is highly toy-scale (using a ViT-Tiny on MNIST and FashionMNIST with only 512 training samples), and standard offline optimization followed by post-hoc quantization strictly outperforms the proposed direct on-device adaptation. The paper requires a more rigorous theoretical treatment, convergence or generalization proofs, and validation on larger-scale, more representative models (such as CLIP or LLaMA) before it is ready for publication.

---

## Questions and Constructive Feedback for the Authors

1. **Theoretical Formalization of "Low-Pass Filtering":** 
   Can you provide a formal spectral analysis of the parameter trajectories or gradient updates? Specifically, can you mathematically define "high-frequency optimization noise" in the context of layer-wise coefficient search and formally prove how the polynomial projection filters this noise?
2. **Discontinuity Mismatch in Zero-Order Search:** 
   How do you reconcile the fact that **Block-wise Constant (ES) scaling outperforms Polynomial Continuous (ES) by 0.28%** in the 4-bit regime? Since the rounding landscape is discontinuous, is a smooth prior fundamentally disadvantaged for gradient-free search? Can you provide a theoretical characterization of this trade-off?
3. **Generalization Analysis:** 
   Can you provide a formal statistical learning theory analysis (e.g., generalization bounds or Rademacher complexity bounds) to mathematically substantiate the "Overfitting-Optimizer Paradox" and prove why the $(d+1) \times K$ polynomial subspace generalizes better than the $L \times K$ layer-wise space on tiny calibration streams?
4. **Convergence Guarantees for STE under Vandermonde Constraint:** 
   The STE introduces gradient bias because of mismatched forward and backward passes. When combined with a Vandermonde mapping and trajectory clamping, does this bias accumulate? Can you provide convergence guarantees or stability boundaries for Adam STE under this specific linear projection constraint?
5. **Practical Use Case of On-Device TTA:** 
   Since **offline optimization (AdaMerging Adam $\to$ Low-Bit) strictly outperforms Q-PolyMerge by +1.33% to +2.51% absolute accuracy**, why would a practitioner choose to perform expensive and lower-accuracy adaptation on the device instead of optimizing offline on a server prior to deployment? What is the specific, high-priority scenario where offline adaptation is impossible?
6. **Scalability and Deeper Architectures:** 
   While the appendix outlines elegant theoretical blueprints for Chebyshev orthogonal splines on CLIP and LLaMA, why are these not implemented and evaluated? Testing on at least one representative larger-scale model (e.g., CLIP-ViT-B/16 or a 1B/3B language model) is critical to prove that the polynomial constraint scales beyond toy-scale ViT-Tiny on MNIST.
7. **Physical Hardware Validation:** 
   You outline a detailed hardware-in-the-loop testbed in the appendix. Can you provide physical measurements (such as latency, actual SRAM utilization, and dynamic power consumption) on the STM32H7 or RISC-V GAP8 microcontrollers to validate your theoretical memory projections?
