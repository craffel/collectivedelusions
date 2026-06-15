# Peer Review of Conference Submission

## Summary of the Paper
This paper presents a rigorous post-mortem and limitation-mapping study of joint model merging and pruning on resource-constrained edge hardware. The authors investigate **ZipMerge**, a framework designed to co-optimize layer-wise merging coefficients and magnitude-pruning boundaries at test-time using an unsupervised minimum entropy objective on tiny calibration sets. Two optimization engines are evaluated: first-order Straight-Through Estimators (STE) and zero-order 1+1 Evolution Strategies (ES).

Rather than curating a narrative of triumph, the authors honestly expose severe empirical boundaries under extreme domain shift (MNIST, FashionMNIST, CIFAR-10, SVHN):
1. **Catastrophic Representational Collapse:** Every single joint merged configuration (including Uniform, AdaMerging, and ZipMerge) collapses to near random-guessing levels (~10%-14% accuracy) on a compact ViT-Tiny backbone.
2. **The Overfitting-Optimizer Paradox:** Unconstrained test-time adaptation on tiny calibration sets overfits transductively, successfully minimizing entropy while destroying generalizable features.
3. **Decoupled Baseline Outperformance:** A simple, unoptimized decoupled baseline, **Prune-then-Merge (P-then-M)**, consistently and significantly outperforms the complex test-time optimization loops because pre-merging pruning acts as a spatial regularizer that removes orthogonal parameter noise.
4. **Noisy Expert Noise Injection:** Poorly converged experts (such as SVHN evaluated at 19.59%) act as "poison pills" in weight space, injecting noise that collapses the entire merged system.

The paper translates these failures into actionable architectural guidelines, advocating for domain-aligned task selections, parameter-efficient adapters (PEFT), and explicit structural regularizations. It also validates a highly elegant, post-hoc **Orthogonal Procrustes SVD Alignment** step that rotates separately learned LoRA adapter weight spaces into a shared coordinate system before averaging, yielding a massive +16.45% absolute accuracy gain with completely negligible computational overhead.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Scientific Honesty and Rigor:** The paper is exceptionally refreshing in its refusal to spin a failed optimization method as a success. By thoroughly exposing and mapping the limitations of linear weight-space operations under extreme task conflict, it provides a highly valuable cautionary tale and reference point for both researchers and real-world practitioners.
2. **Deconstruction of Complexity in Favor of Simplicity:** The paper demonstrates that a simple, unoptimized decoupled baseline (**Prune-then-Merge**) consistently outperforms the complex, computationally expensive test-time adaptive co-optimization of coefficients and pruning masks. This is a powerful demonstration that over-engineering optimization loops is often unnecessary and structurally inferior to simpler, decoupled operations.
3. **Elegance of the Orthogonal Procrustes SVD Alignment:** The introduction and empirical validation of post-hoc SVD-based rotation for LoRA adapters is highly elegant. It directly addresses the mathematical root cause of linear merging errors (coordinate basis mismatch) post-hoc without requiring data or complex training. Achieving a massive +16.45% absolute improvement over unaligned LoRA merges with completely negligible CPU overhead (under a millisecond) is a masterclass in solving problems via simple, clean mathematical formulation rather than brute-force optimization.
4. **Exceptionally Comprehensive and Rigorous Evaluation:** The authors went to great lengths to isolate and address potential confounds:
   - Evaluated alternate backbones (ResNet-18) to prove findings generalize across CNNs.
   - Evaluated larger models (ViT-Base) to isolate model capacity constraints.
   - Tested alternative unsupervised regularizers (MMI, soft pseudo-labeling, Likelihood Ratio, CBC loss).
   - Profiling of structured block-pruning execution latency on an ARM mobile CPU, demonstrating a practical 1.89x speedup.
   - Generalizability to language models (GPT-2) and thorough systems-level scaling studies (VRAM and CPU sorting overhead).

### Weaknesses
1. **Framing and Title Mismatch:** The paper’s current title ("ZipMerge: Joint Model Merging and Pruning...") and abstract initially position ZipMerge as the primary framework. However, the core empirical finding of the paper is that ZipMerge is highly over-engineered, computationally expensive, and ineffective under task conflicts, being beaten by simpler baselines (P-then-M) and PEFT methods. The framing could be improved by shifting the title and central narrative to focus more heavily on the *limitation-mapping study* and the superior, simpler alternatives (P-then-M and Procrustes SVD Alignment).
2. **Complexity of Certain Proposed Regularizers:** While the authors rightly identify that unconstrained entropy TTA collapses, some of their proposed regularized objectives (such as the self-supervised Class-Balanced Contrastive loss) introduce even more complexity and optimization hyper-parameters. The paper should more strongly emphasize that simpler structural distance penalties (like Reg-ZipMerge) or PEFT manifold restrictions are the primary, most robust solutions to prevent overfitting.

---

## Soundness
**Rating:** Good
**Justification:** The empirical methodology is extremely solid, honest, and highly reproducible. The authors evaluate an extensive suite of baselines, backbones, and alternate regularizers, and their analysis of why the optimization fails (Overfitting-Optimizer Paradox, representation collapse) is highly sound and convincing. Although the proposed ZipMerge algorithm itself is over-engineered and fails to solve the representational collapse problem under extreme domain shift, the paper's scientific value lies in the correctness of its empirical findings and its rigorous mapping of these boundaries.

---

## Presentation
**Rating:** Excellent
**Justification:** The paper is exceptionally clear, well-structured, and easy to follow. The mathematical formulations of model merging, dynamic magnitude pruning, and binarized Straight-Through Estimators are precise. Algorithm 1 provides a highly useful systems-level trace of the adaptation loop. Figures and tables are clean and directly reinforce the core findings (such as the collapse curves in Figure 1).

---

## Significance
**Rating:** Good
**Justification:** This paper has high practical significance for systems engineers attempting on-device multi-task model composition. By establishing that linear weight-space merging and test-time co-optimization are fundamentally bounded by representation incompatibility rather than expert convergence quality, the paper steers the community away from complex, unconstrained on-the-fly optimization loops that overfit and collapse. Instead, it redirects attention toward simpler, structurally robust alternatives like pre-merging spatial regularizers, PEFT adapter restrictions, and lightweight post-hoc coordinate rotations.

---

## Originality
**Rating:** Good
**Justification:** While combining magnitude pruning, STEs, and entropy minimization is relatively standard, the paper’s originality lies in its rigorous mapping of failure modes of these methods. Furthermore, the formulation and empirical validation of post-hoc Orthogonal Procrustes SVD-based coordinate rotation represents a highly novel, lightweight, and elegant alignment mechanism that directly resolves coordinate mismatch in parameter space.

---

## Overall Recommendation
**Rating:** 5: Accept
**Justification:** This submission represents a rare, outstanding example of scientific honesty in machine learning literature. Instead of curating a narrative of triumph around an over-engineered method, the authors systematically deconstruct their own co-optimization framework (ZipMerge), revealing that a simpler decoupled baseline (Prune-then-Merge) is structurally superior. The paper provides a highly valuable cautionary lesson, complete empirical proof of the limits of linear weight-space merging under domain shift, and introduces an elegant post-hoc SVD rotation step that delivers massive gains with negligible overhead. This is a highly valuable, well-written contribution that aligns perfectly with the principles of simplicity, clarity, and mathematical elegance.
