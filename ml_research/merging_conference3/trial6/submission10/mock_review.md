# Peer Review Report

**Paper Title:** Task-Correlation Prior Regularization (TCPR): Guiding Calibration via Representation Cosine Similarity  
**Review Date:** June 14, 2026  

---

## 1. Summary of the Paper
This paper addresses the problem of low-data calibration for dynamic routing heads in multi-task model merging. Linearly interpolating specialized expert parameters (fine-tuned from a common pretrained base model) enables parameter-efficient multi-task inference. Dynamic model merging uses a lightweight routing head to compute sample-specific or batch-specific merging coefficients at test time. However, optimizing these heads under severe data constraints (e.g., 16 calibration samples per task) frequently leads to catastrophic overfitting or representational collapse, particularly on high-conflict, heterogeneous datasets.

To address these limitations, the authors propose:
1. **Bounded Sigmoidal Router (BSigmoid-Router):** Replaces the standard Softmax activation with independent, decoupled sigmoidal pathways to eliminate the competitive, zero-sum bottleneck of Softmax routing.
2. **Task-Correlation Prior Regularization (TCPR):** Introduces a pre-computed task-relationship similarity matrix $S \in \mathbb{R}^{K \times K}$ as a prior. The prior is implemented in two variants: **TCPR-Param** (parameter-space cosine similarity of task vectors across all layers) and **TCPR-Rep** (representation-space cosine similarity of base model intermediate activations). To prevent collinear collapse and scale mismatches, the authors center the off-diagonal elements of the similarity matrix and normalize the routing signatures to unit spheres before computing their pairwise dot products.

The authors evaluate their method on a heterogeneous four-task image classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a Vision Transformer backbone, claiming state-of-the-art results that outperform standard L2-regularized baselines and state-of-the-art wave-interference methods like QWS-Merge.

---

## 2. Overall Recommendation
*   **Recommendation:** **3: Weak Reject**
*   **Soundness Rating:** **Fair**
*   **Presentation Rating:** **Excellent**
*   **Significance Rating:** **Fair**
*   **Originality Rating:** **Good**

**Justification:**  
The authors have made highly commendable efforts to refine this manuscript. In particular, they successfully addressed several critical presentation and methodology issues from previous rounds of reviews:
1.  **Resolved Scientific Misrepresentation:** The paper now correctly states in Section 4.1 that the evaluation was performed on 250 test samples per task, aligning honestly with the codebase.
2.  **Restored Professional Tone:** All self-indulgent faction references (such as capitalized/bolded "Empiricists") have been removed, resulting in a neutral, objective, and scholarly academic tone.
3.  **Mathematical Improvements:** The introduction of off-diagonal centering and unit sphere signature normalization are elegant additions that resolve previous concerns about collinear-collapse and weight explosion.

However, a close inspection of the empirical results, mathematical scaling, and training protocol reveals **three remaining critical flaws** that currently prevent recommendation for publication. Most notably, the proposed regularizer has absolutely zero active effect at its claimed "optimal" hyperparameter (behaving identically to the unregularized baseline), the second variant (TCPR-Rep) actively degrades performance, and the underlying expert models are severely under-optimized. Addressing these concerns is necessary to establish the scientific and practical validity of the proposed framework.

---

## 3. Major Weaknesses and Critical Flaws (3 Main Flaws)

### Flaw 1: The "Dead Regularizer" and Lack of Empirical Improvement (The "Zero-Delta" SOTA)
The central claim of the paper is that the proposed TCPR regularizer significantly stabilizes and enhances joint multi-task performance during low-data calibration. However, a direct inspection of the reported results in Table 1 reveals that **the proposed regularizer achieves absolutely zero performance improvement over the simple unregularized baseline**:
*   **BSigmoid-Router (Unreg) Joint Mean:** **25.40%** (MNIST 34.80%, Fashion 26.00%, CIFAR-10 30.00%, SVHN 10.80%)
*   **TCPR-Param (Ours) ($\beta = 10^{-4}$) Joint Mean:** **25.40%** (MNIST 34.80%, Fashion 26.00%, CIFAR-10 30.00%, SVHN 10.80%)

Converting these percentages into the exact number of correct predictions across the evaluated 1000 total test samples (250 samples $\times$ 4 tasks):
*   **BSigmoid-Router (Unreg):** $254$ out of $1000$ correct.
*   **TCPR-Param (Ours) ($\beta = 10^{-4}$):** $254$ out of $1000$ correct.

The predictions are **exactly identical down to the last decimal place** because the regularizer has zero active effect during training. This "zero-delta" behavior is caused by a severe scaling mismatch:
1.  **Microscopic Priors:** Because the parameter-space similarities are centered, their off-diagonals are extremely small ($S^{\text{centered}}_{i, j} \approx \pm 0.01$).
2.  **Bounded Cosine terms:** Projecting the routing signatures to unit spheres bounds the pairwise dot products $\cos(\mathbf{w}_i, \mathbf{w}_j)$ in $[-1, 1]$.
3.  **The Scale Mismatch:** The total unscaled prior loss $\sum_{i \ne j} S^{\text{centered}}_{i, j} \cos(\mathbf{w}_i, \mathbf{w}_j)$ is on the order of $12 \times 0.01 \times 1 \approx 0.12$. When scaled by $\beta = 10^{-4}$, the prior loss term in the total objective is on the order of $1.2 \times 10^{-5}$. Since the cross-entropy loss $\mathcal{L}_{\text{CE}}$ is around $2.3$, the regularizer is **five orders of magnitude smaller than the cross-entropy loss**. Its gradients are completely drowned out, and the optimizer behaves identically to the unregularized router.

Thus, the reported "SOTA" performance is entirely an artifact of the sigmoidal router architecture rather than the proposed regularization.

### Flaw 2: Extreme Under-optimization of Specialist Experts
The "specialist expert" models that serve as the foundation of the model merging benchmark are poorly trained and far from true convergence:
*   **The Numbers:** The MNIST expert achieves only **73.20% accuracy** (where standard MNIST convolutional/ViT models easily exceed $99\%$), and the SVHN expert achieves a dismal **23.20% accuracy** (barely above the $10\%$ random-guess baseline for a 10-class dataset).
*   **The Protocol:** Each expert was trained on a tiny subset of 1000 images per task for only 2 epochs.
*   **Why this is critical:** Model merging is mathematically designed and practically used to fuse fully specialized, converged expert models. Merging models that are barely trained and poorly converged makes any findings about "bridging the gap to specialist experts" and "preventing representational collapse" highly unconvincing. Under-optimized experts introduce massive parameter noise, rendering the resulting merging and routing analysis of limited relevance to actual deep learning practices.

### Flaw 3: Performance Degradation of the Representation-Space Prior (TCPR-Rep)
While the parameter-space variant achieves identical performance to the baseline, the representation-space variant (**TCPR-Rep**) actually **degrades** performance compared to the unregularized sigmoidal router baseline:
*   **BSigmoid-Router (Unreg) Joint Mean:** **25.40%** (MNIST 34.80%, Fashion 26.00%, CIFAR-10 30.00%, SVHN 10.80%)
*   **TCPR-Rep (Ours) ($\beta = 10^{-4}$) Joint Mean:** **21.70%** (MNIST 21.60%, Fashion 20.00%, CIFAR-10 34.00%, SVHN 11.20%)

This represents a substantial **-3.70% absolute joint accuracy drop** compared to doing nothing (BSigmoid-Router Unreg), and is even worse than standard isotropic L2 weight decay (`BSigmoid-Router (Reg)`, **24.00%**).
This degradation directly contradicts the paper's central claim that incorporating task-relatedness priors provides a "robust, scale-invariant pathway" for dynamic model merging.

---

## 4. Strengths of the Paper
*   **Highly Relevant Problem:** Low-data calibration of dynamic model merging routing heads is an active, important bottleneck in parameter-efficient multi-task deployment.
*   **De-mystifying Metaphors:** The goal of deconstructing complex physics-inspired dynamic routing methods (such as QWS-Merge) and showing that simple sigmoidal routing can achieve comparable or better performance is a highly refreshing and commendable stance.
*   **Excellent Responsiveness:** The authors have shown exceptional diligence in revising previous structural, stylistic, and presentation concerns, establishing high scientific transparency.
*   **Pristine Structure and Visuals:** The paper is exceptionally well-written, easy to follow, and features clean, professional visualizations (such as the sweep trajectory in Figure 1).

---

## 5. Detailed Evaluation on Criteria

### Soundness
*   **Rating: Fair**
*   **Justification:** Mathematically, the formulations for centered similarity and signature normalization are correct and address previous collinear-collapse issues. However, methodologically, evaluating on severely under-trained experts (e.g., SVHN expert at 23.20%) and claiming SOTA improvement when the regularizer is mathematically inactive at the optimal hyperparameter limits the scientific soundness.

### Presentation
*   **Rating: Excellent**
*   **Justification:** The writing is exceptionally clear, fluent, and well-structured. The transition from static merging to dynamic routing and then to regularization is natural and easy to follow. The removal of capitalized faction names ("Empiricists") and correction of the test split size (250) successfully restored scholarly credibility.

### Significance
*   **Rating: Fair**
*   **Justification:** The potential significance of the paper is constrained. Because the proposed method does not offer any actual performance benefit over the basic unregularized sigmoidal router, and the absolute accuracies are extremely poor (25.40% joint mean) due to under-trained experts, the practical utility of the resulting merged models is virtually zero for real-world deployments.

### Originality
*   **Rating: Good**
*   **Justification:** Introducing pre-computed task-relationship similarity priors to guide dynamic routing calibration is a reasonable and relatively original concept. The addition of off-diagonal centering and sphere projection are creative combinations of existing techniques.

---

## 6. Questions and Constructive Suggestions for the Authors

1.  **Scale the Regularizer Strength appropriately:** To make the regularizer mathematically active without weight explosion, you must scale $\beta$ proportionally to match the magnitude of the cross-entropy loss (e.g., sweeping $\beta \in [10^{-1}, 10^1]$ or scaling the prior term by the reciprocal of the similarity matrix norm). Showing a positive performance delta over the unregularized sigmoidal router is crucial to justify the proposed regularizer.
2.  **Train Specialist Experts to Convergence:** Training experts on only 1000 images for 2 epochs does not achieve convergence. SVHN and MNIST specialist models should achieve over 90% and 99% accuracy respectively. Please train your specialist experts properly to ensure a realistic and challenging starting point for your model merging benchmark.
3.  **Investigate the Failure of TCPR-Rep:** Since TCPR-Rep actively degrades performance (dropping to 21.70% joint accuracy), please analyze why representation-space similarity acts as a destructive prior. It is possible that intermediate activation similarities do not align with optimal weight-space updates, causing conflict during calibration.
4.  **Report Multi-seed Statistical Variance:** Because low-data calibration has high variance, please run your calibration and sweeps across at least 5 independent random splits/seeds and report the mean and standard deviations (e.g., $25.40\% \pm 0.45\%$). This is essential to confirm whether any minor differences are statistically significant.
