# Peer Review of Conference Submission

**Paper Title:** The Layer-Averaging Collapse Paradox: Exposing the Limits of Dimensionality in Layer-Wise Dynamic Model Merging  
**Overall Recommendation:** **2: Reject**  
**Soundness:** **Poor**  
**Presentation:** **Good**  
**Significance:** **Poor**  
**Originality:** **Fair**  

---

## 1. Summary of the Submission
The submission investigates the spatial granularity and dimensionality of routing trajectories in dynamic, layer-wise model merging in weight space. Specifically, it positions itself as an empirical deconstruction of a "Layer-Averaging Collapse" (rank-1 collapse) theorem, which claims that layer-wise dynamic routing coefficients inevitably collapse to a single global dimension, rendering layer-wise routers redundant. To audit this claim, the paper introduces:
1. A **Bounded Sigmoid (BSigmoid) Router** that utilizes independent, element-wise sigmoids followed by a post-gating sum-to-1 normalization.
2. An **SVD Collinearity Audit** that computes the **Collinearity Ratio** ($\rho_{collinear} = \sigma_1 / \sum \sigma_i$) of the learned Batch-Averaged Layer-wise Coefficient Matrix $A \in \mathbb{R}^{L \times K}$ to measure its dimensionality.
3. An empirical evaluation on Split-MNIST subsets using a 12-layer MLP (DeepMLP-12) and a 4-layer CNN (TinyCNN-4) under various task-conflict suites.
4. A conceptual discussion of the **Batch-Averaged Multi-Task Inference Paradox**, highlighting systems-level and logical boundaries of dynamic weight blending.

---

## 2. Strengths and Weaknesses

### Major Strengths
1. **Outstanding Intellectual Honesty:** The authors deserve high praise for Section 3.5, where they explicitly articulate and analyze the *Batch-Averaged Multi-Task Inference Paradox*. Identifying that dynamic model merging under batch-averaging is either logically redundant (for homogeneous batches) or functionally degraded to a static compromise (for mixed batches) shows exceptional scientific rigor and self-reflection.
2. **Detailed Systems-Level Analysis:** In Appendix 3, the authors present an insightful systems-level audit of memory-bandwidth and latency bottlenecks for large-scale architectures (such as 7B LLMs on H100 GPUs). This mathematically explains why full-parameter on-the-fly weight blending is a memory-bound bottleneck and provides a strong justification for restricting dynamic merging to parameter-efficient modules (like LoRA).
3. **Rigorous Diagnostic and Ablation Studies:** The appendices include robust checks, such as testing sensitivity to projection dimensions and random projection seeds, proving that their SVD Collinearity Ratio is a statistically stable diagnostic tool.

### Major Weaknesses

#### A. Mathematical Flaw in "Gradient Gating Decoupling" Claim
In Section 4.4, the authors claim that their Bounded Sigmoid (BSigmoid) router avoids the competitive gradient clashing of standard Softmax because it "decouples" the gradient paths during calibration. This claim is mathematically incorrect. While the *pre-normalized* sigmoidal gate $\tilde{\alpha}_i$ has a local derivative, the actual routing coefficients $\lambda_i$ are normalized via a post-gating division (Equation 10):
$$\lambda_{l, k}(x) = \frac{\tilde{\alpha}_{l, k}(x)}{\sum_{j=1}^K \tilde{\alpha}_{l, j}(x) + \epsilon}$$
By the quotient rule, the derivative of any routing coefficient $\lambda_k$ with respect to *any* input logit $\alpha_i$ is heavily coupled. Specifically, for $i \neq k$:
$$\frac{\partial \lambda_k}{\partial \alpha_i} = -\sigma'(\alpha_i) \frac{\tilde{\alpha}_k}{\left(\sum_j \tilde{\alpha}_j\right)^2}$$
This forces the overall gradient of the loss $\mathcal{L}$ with respect to logit $\alpha_i$:
$$\frac{\partial \mathcal{L}}{\partial \alpha_i} = \sum_k \frac{\partial \mathcal{L}}{\partial \lambda_k} \frac{\partial \lambda_k}{\partial \alpha_i}$$
to be mathematically coupled across all task dimensions $k$, exactly like standard Softmax. The normalization re-introduces the competitive zero-sum constraint at both the forward and backward gradient levels. The assertion that BSigmoid "decouples gradient paths" is a mathematical error.

#### B. Severe Performance Collapse and Lack of Practical Utility
The absolute classification accuracies of the merged models are extremely poor, rendering the proposed system functionally unusable:
- **DeepMLP-12 Cross-Domain:** The Layer-wise Router achieves only **$16.15\% \pm 5.60\%$** accuracy on an 8-class classification task. Since the random guessing threshold is **$12.5\%$**, the model is essentially broken and completely unusable. The authors admit that "full-parameter linear interpolation of deep, fully connected layers... is fundamentally a failed paradigm." If it is a failed paradigm, evaluating it here without solving it severely limits the significance of the paper.
- **TinyCNN-4 Cross-Domain:** The proposed dynamic Layer-wise Router (scoring **$52.52\% \pm 5.95\%$**) is consistently **outperformed** by the simple, static baseline **OFS-Tune** (scoring **$53.40\% \pm 7.16\%$**) on the primary data split (128 samples per task). Why should the community adopt a complex, high-capacity dynamic router that is more computationally expensive yet performs *worse* than a simple static global baseline?
- **Massive Oracle Gap:** There is an unaddressed gap of **nearly 46%** on CNN and **over 82%** on MLP compared to the Oracle ceiling (which simply routes inputs directly to the specialized unmerged experts).

#### C. Overstated Interpretation of the SVD Collinearity Ratio
The authors define the Collinearity Ratio as $\rho_{collinear} = \sigma_1 / \sum_{i=1}^{\min(L, K)} \sigma_i$.
- For $K=2$ tasks, the absolute minimum possible ratio is **0.5** (when $\sigma_1 = \sigma_2$).
- The authors report ratios of $0.64$ to $0.74$ for $K=2$. This is extremely close to $1.0$ (perfect collinearity) and represents a heavily dominated single-dimensional trajectory. For instance, a ratio of $0.65$ means that the first singular value is nearly double the magnitude of the second.
- Even for $K=4$ (where the minimum is $0.25$), the reported ratios are $0.50$ and $0.57$. The first singular value still accounts for more than half of the total energy.
Thus, the routing coefficients are still highly collinear. The authors' claim that they have "completely deconstructed" the rank-1 collapse is an overstatement of their empirical data.

#### D. Weak Evaluation and Lack of Code
- **Toy Dataset Sandboxes:** The entire physical evaluation is conducted on Split-MNIST. This is an extremely simple benchmark that does not reflect modern representational challenges. The transition to "natural images" is relegated to a brief, unverified discussion in Section 4.4 and a small ViT simulation in the appendix where only collinearity (and not classification accuracy) is evaluated.
- **Fragility of Hyperparameters:** The few-shot calibration loop (128 samples per task) is extremely fragile. Increasing the projection dimension $d$ beyond 8 or making the projection matrix learnable causes immediate generalization collapse (e.g., dropping to $26.12\%$ on TinyCNN-4).
- **Inadequate Reproducibility:** No open-source code, repository link, or reproduction script is provided.

---

## 3. Rating Justification

- **Soundness (Poor):** The core mathematical argument for gradient decoupling in the BSigmoid router is incorrect. The SVD collinearity ratios are mathematically misinterpreted to overstate the refutation of rank-1 collapse. The calibration setup is extremely fragile, and single-task experts are trained on non-standard, tiny subsets (512 samples/task).
- **Presentation (Good):** The paper is clearly structured and written in a mathematically sophisticated tone. However, the framing is overly reactive and adversarial against a single anonymous preprint.
- **Significance (Poor):** Because the proposed dynamic router is consistently outperformed by a simple static baseline on CNNs, performs at the level of random guessing on MLPs, and is plagued by a fatal unresolved Batch-Averaged Inference Paradox, the practical utility of this work is virtually non-existent.
- **Originality (Fair):** The proposed router and diagnostic tools are simple combinations of standard, pre-existing ML techniques (SVD, random projections, sigmoid gating, few-shot Adam calibration).

---

## 4. Questions and Suggestions for the Authors

1. **Address the Mathematical Gating Coupling:** In Equation 10, how does the post-gating sum-to-1 division not re-introduce the exact same competitive zero-sum constraint as Softmax? Please provide a correct mathematical derivation for the gradients of $\lambda_k$ with respect to logit $\alpha_i$, and explain how this represents "gradient path decoupling" when the denominator couples all logits in backpropagation.
2. **Explain the Superiority of Static OFS-Tune:** Since OFS-Tune consistently outperforms your Layer-wise Router across all three task-conflict suites in TinyCNN-4, what is the practical justification for using a more complex, high-capacity dynamic router?
3. **Scale Beyond MNIST:** Can you provide classification accuracies for your ViT-B/16 simulation on CIFAR-10 + SVHN? If weight-space merging on deep MLPs is a "failed paradigm" due to coordinate misalignment, how does ViT-B/16 avoid this, and what are its actual multi-task accuracies?
4. **Solve the Batch-Averaged Inference Paradox:** Since your paper identifies this paradox as a fundamental barrier to on-the-fly dynamic merging, why does the paper not attempt to implement and physically evaluate one of your proposed solutions (such as LoRA-level dynamic routing) instead of presenting a failed grayscale MNIST pipeline?
5. **Open Source Code:** To guarantee scientific reproducibility, please provide a public, anonymous GitHub link with the complete training, calibration, and diagnostic codebase.
