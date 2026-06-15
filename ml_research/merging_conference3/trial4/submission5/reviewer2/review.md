# Peer Review Report

## Summary of the Submission
This submission focuses on **weight-space model merging**, proposing **Sparsity-Guided Task Arithmetic (SG-TA)** as a post-hoc, deterministic weight-space regularization framework to fuse multiple task-specific neural network experts. To mitigate the catastrophic representational collapse caused by parameter interference (unregularized additions), SG-TA applies magnitude-based binary masking to individual task-specific update vectors before merging. 

The authors evaluate two masking scopes: **Global Quantile (GQ) masking** (calculating a single magnitude threshold globally across the entire model) and **Layer-wise Quantile (LQ) masking** (enforcing uniform keep-ratios per layer). They also introduce **Task Vector Magnitude Normalization (TV-Norm)** to handle absolute magnitude imbalance across tasks, **Sigmoid-Gated Soft Masking (SG-TA-Soft)** to evaluate continuous sparsification, and **Non-Uniform Coordinate Search (CS)** as a linear-time alternative to exponential task-specific hyperparameter grid sweeps. Hyperparameters are calibrated using **Offline Few-Shot Validation Tuning (OFS-Tune)** on 10 samples per task. Experiments on a 4-dataset visual classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a Vision Transformer (ViT-Tiny) backbone show that SG-TA GQ achieves a joint mean accuracy of 61.40%, outperforming unregularized Task Arithmetic (46.32%), TIES-Merging (60.64%), and DARE-Merging (58.44%).

---

## Overall Recommendation
**Rating: 3: Weak Reject**

**Justification**: 
The submission has clear empirical merits, including an exceptionally thorough evaluation, rigorous 5-seed statistical runs, a comprehensive baseline comparison (including unpruned layer-wise scaling, Fisher-Weighted Averaging, and a trained Joint MTL upper bound), and outstanding scientific honesty regarding the absolute performance gap and statistical significance boundaries. 

However, from a scientific and conceptual standpoint, the paper is **predominantly heuristic and lacks any rigorous theoretical grounding, mathematical proofs, or formal guarantees**. While the empirical sweeps are comprehensive, the paper does not offer a deep theoretical understanding of *why* weight-space consolidation behaves the way it does under these operations. Crucial qualitative claims (such as the Orthogonal Noise Hypothesis, the surrogacy to diagonal Fisher Saliency, the convergence of Coordinate Search, and landscape smoothing in soft masking) are treated as intuitive assumptions rather than being mathematically proven or analyzed. For a conference paper, it is vital that proposed heuristics are supported by a solid mathematical framework that explains their underlying mechanics. Therefore, the theoretical weaknesses currently outweigh the empirical strengths, and a revision incorporating formal mathematical analysis is recommended.

---

## Strengths and Weaknesses

### Strengths
1. **Methodological Rigour and Statistical Validity**: The authors evaluate all merging variants and baselines across **5 random calibration seeds**, reporting means and standard deviations. This level of statistical completeness is exemplary.
2. **Comprehensive and Honest Baseline Comparison**: The authors compare against a wide range of baselines (naive/optimized TA, TIES, DARE, Decoupled Prune-then-Merge, L-Scale, and Fisher-Weighted Averaging). Crucially, they train and report a physical **Joint Multi-Task Learning (MTL)** baseline (95.55%) and individual **Dense Expert Ceilings** (95.91%), providing a highly transparent look at the absolute performance gap.
3. **High Scientific Honesty**: The authors explicitly point out that their method's superiority over TIES-Merging is not statistically significant due to overlapping standard deviations. They also dedicate a comprehensive section to discussing the absolute performance collapse (a 34.51% gap between the merged model and individual experts), avoiding over-claiming.
4. **Insightful Algorithmic Extensions**:
   - **Coordinate Search**: Rebalances representations in linear time $\mathcal{O}(T)$ by optimizing task-specific parameters sequentially.
   - **Validation Size Sweeps**: Successfully validates the sensitivity of pre-masking normalization, showing that a modest increase in the validation pool size to $N_{\text{val}}=20$ stabilizes calibration and cuts variance by 4x.
   - **Sigmoid-Gated Soft Masking**: Demonstrates that continuous soft-gating stabilizes the hyperparameter landscape, halving calibration variance across seeds.

### Weaknesses
1. **Lack of Theoretical Grounding and Formal Proofs**: The primary limitation is the lack of mathematical proofs or guarantees behind the proposed heuristics:
   - **The Orthogonal Noise Hypothesis**: The claim that low-magnitude updates represent uncorrelated background noise is backed only by empirical cosine similarities of trained task vectors. There is no formal probabilistic modeling of these updates (e.g., treating them as random high-dimensional vectors) to mathematically prove their orthogonality.
   - **Mathematical Surrogacy to Diagonal Fisher Saliency**: The authors argue qualitatively in Section 4.4 (item 7) that magnitude-based pruning acts as a zero-order surrogate to diagonal Fisher Saliency. This claim must be formalized with a mathematical proof showing that magnitude pruning bounds or approximates diagonal Fisher entries under specific loss landscape assumptions (e.g., quadratic approximation).
   - **No Optimization Theory for Coordinate Search**: Since coordinate descent is applied to a non-convex, non-smooth, and discontinuous validation accuracy objective, there are no convergence guarantees. The paper should provide a formal convergence analysis or bound the optimization sub-optimality.
   - **No Smoothness Analysis for Soft Masking**: The landscape stabilization of Sigmoid-Gated Soft Masking is discussed qualitatively, but lacks a mathematical derivation showing how the Lipshitz continuity or gradient variance changes with the temperature parameter $\beta$.
2. **Limited Architectural Scale and Over-Parameterization**: The experiments are confined to a compact Vision Transformer (\texttt{vit\_tiny\_patch16\_224}, 5.7M parameters) and simple low-resolution datasets (MNIST, CIFAR-10, SVHN). In theoretical deep learning, compact networks behave very differently from over-parameterized foundation models, which possess massive null spaces and low-rank parameter updates. It remains mathematically unverified whether the observed global quantile budget flexibility and magnitude-based regularizations generalize to large-scale over-parameterized architectures.

---

## Soundness
**Rating: Fair**

**Justification**:
While the experimental methodology and statistical runs are highly sound and reproducible, the paper is fundamentally weak in its **theoretical soundness**. The central claims (that magnitude-based binary masking acts as an optimal spatial regularizer, that low-magnitude updates are orthogonal noise, and that Coordinate Search is a stable optimization protocol) are supported purely by empirical trends rather than formal theoretical analysis. There is no mathematical verification of why absolute parameter magnitude is the correct metric for identifying non-essential updates, leaving the framework as a collection of well-tuned heuristics.

---

## Presentation
**Rating: Excellent**

**Justification**:
The submission is exceptionally well-written, logically organized, and easy to follow. The mathematical notation is clean and clearly defines all proposed variants (equations 1-13). The tables and figures are well-structured, presenting statistical means, standard deviations, and calibration costs clearly. The authors are highly transparent about prior work, explicitly discussing the mathematical equivalence of SG-TA (LQ) and Decoupled Prune-then-Merge under optimal calibration.

---

## Significance
**Rating: Fair**

**Justification**:
The paper addresses a highly important problem—weight-space model consolidation—and offers highly practical, actionable insights (such as the vital role of global budget flexibility and the scalability of Coordinate Search for calibration). However, the significance is limited by the lack of theoretical proofs and the restriction to a small ViT-Tiny sandbox. Without a mathematical framework explaining the underlying mechanics, it is difficult for other researchers to build upon these heuristics theoretically, restricting the overall significance of the contribution.

---

## Originality
**Rating: Fair**

**Justification**:
The algorithmic novelty of SG-TA is incremental:
- Magnitude-based pruning prior to merging is a well-known technique (used as the first step in TIES-Merging). Removing subsequent steps (sign election) represents a simplification rather than a conceptually novel contribution.
- SG-TA (LQ) is mathematically equivalent to the existing Decoupled Prune-then-Merge baseline.
- Sigmoid gating, vector L1 scaling, and coordinate descent are standard textbook mathematical operations applied to this specific domain.
The primary originality lies in the systematic empirical investigation of global budget allocation (GQ) and task-specific coordinate optimization, which represents a valuable empirical contribution but a minor algorithmic delta.

---

## Questions and Constructive Suggestions for the Authors
1. **Formalize the Orthogonal Noise Hypothesis**: Can you model the task vector updates mathematically as random vectors in high-dimensional space under specific distribution assumptions to derive formal probabilistic bounds on their cosine similarities as a function of the keep-ratio $k$?
2. **Prove the Diagonal Fisher Surrogacy**: Can you provide a mathematical proof showing that magnitude-based pruning bounds or approximates the diagonal entries of the Fisher Information Matrix (or the Hessian of the loss landscape) under a quadratic Taylor approximation?
3. **Provide Convergence Guarantees for Coordinate Search**: Given that Coordinate Search is applied to a non-smooth validation accuracy objective, can you provide a formal proof of convergence or bound the optimization error relative to the joint global optimum?
4. **Analyze Landscape Smoothness Mathematically**: Can you mathematically derive how the smoothness (e.g., Lipschitz constant of the gradient) of the validation objective changes under Sigmoid-Gated Soft Masking (SG-TA-Soft) as a function of the temperature parameter $\beta$?
5. **Demonstrate Scalability**: Although ViT-Tiny serves as an excellent sandbox, the behavior of compact models differs theoretically from over-parameterized ones. Can you run a small-scale pilot experiment on a larger model (e.g., CLIP-ViT-B or a 1B LLM) to verify if the global budget flexibility (GQ outperforming LQ) and magnitude regularization trends mathematically scale with over-parameterization?
