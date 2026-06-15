# Peer Review

## 1. Summary of the Paper
This paper proposes **Grassmannian Subspace Consensus Merging (GSC-Merge)**, a weight-space model merging framework designed to consolidate multiple fine-tuned expert models into a single multi-task network without full retraining. GSC-Merge targets the major linear projection layers of Transformer blocks (representing over 95% of block parameters) while keeping lightweight normalization, biases, and embedding parameters task-specific (swapped task-conditionally at inference time).

Specifically, the method horizontally concatenates the task vectors across $K$ expert models to construct a joint multi-task update matrix $\mathbf{M}^{(l)}$ at each layer. It then performs Singular Value Decomposition (SVD) on this matrix to extract the top $r$ left-singular vectors, constructing an orthogonal projection matrix $P^{(l)}$ onto a low-rank Grassmannian manifold. The task updates are projected onto this subspace to filter out high-frequency parameter-space noise. Finally, the layer-wise blending coefficients $\alpha$ are optimized using Offline Few-Shot Validation Tuning (OFS-Tune) on a tiny validation calibration set. 

The paper provides a theoretical guarantee of representation drift using the Eckart-Young-Mirsky Theorem and evaluates the approach on a ViT-Tiny backbone across four conflicting image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## 2. Strengths and Weaknesses

### Strengths:
1. **Mathematical Rigor:** The paper provides an elegant, continuous, and differentiable spectral consensus mechanism grounded in the geometry of the Grassmannian manifold. The application of the Eckart-Young-Mirsky Theorem to prove optimal low-rank reconstruction of joint task updates under the Frobenius norm is mathematically sound and well-derived.
2. **High-Quality Presentation:** The writing style is polished, logical, and structured. The exposition of SVD, projection matrices, and few-shot optimization is extremely clear and easy to follow.
3. **Fair and Comprehensive Baseline Comparison:** The paper compares GSC-Merge against 5 established baselines (Uniform Merging, Task Arithmetic, Sparse Task Arithmetic, TIES-Merging, and unconstrained OFS-Tune). It rigorously sweeps baseline hyperparameters (such as magnitude thresholds for STA and TIES) over independent random validation splits to avoid under-tuning bias.
4. **Insightful Ablation Study:** The inclusion of a truly task-agnostic setting as an ablation study is highly transparent, demonstrating how weight merging behaves when domain-specific layer normalization statistics and biases are kept at pre-trained base values rather than being swapped task-conditionally.

### Weaknesses:
1. **Unsupported Empirical Claims Regarding Variance Reduction:** The central empirical claim of the paper is that GSC-Merge acts as a robust spectral regularizer that "dramatically reduces split-sensitivity variance across validation splits" compared to unconstrained OFS-Tune. However, a close examination of the reported standard deviations (SD) in Table 1 and Table 2 reveals that this claim is not supported by the data:
   - For **SVHN** under Task-Conditional Swapping: Unconstrained tuning exhibits an SD of $\pm 19.70\%$, while GSC-Merge exhibits SDs of $\pm 19.60\%$ ($\gamma=0.5$) and $\pm 19.09\%$ ($\gamma=0.3$). These variances are statistically indistinguishable.
   - For **MNIST** under Task-Conditional Swapping: Unconstrained tuning has an SD of $\pm 12.54\%$. GSC-Merge with $\gamma=0.5$ actually exhibits a *higher* SD of $\pm 12.89\%$. While GSC-Merge with $\gamma=0.3$ reduces the SD to $\pm 8.92\%$, it does so at a severe cost to accuracy, which drops from $54.37\%$ to $48.70\%$.
   - For **CIFAR-10** under Task-Conditional Swapping: GSC-Merge with $\gamma=0.5$ has an SD of $\pm 8.93\%$, which is higher than the unconstrained baseline's $\pm 8.87\%$.
   - Under **Truly Task-Agnostic Settings (Table 2)**: The unconstrained tuning baseline joint mean is $20.86 \pm 4.81\%$, while GSC-Merge ($\gamma=0.3$) is $19.08 \pm 4.85\%$, showing lower mean accuracy and *higher* variance.
2. **Lack of Performance Superiority Over Unconstrained Tuning:** GSC-Merge does not outperform unconstrained OFS-Tune in terms of absolute average accuracy in either setting (43.88% vs 44.08% for task-conditional, and 20.61% vs 20.86% for task-agnostic). If the "Overfitting-Optimizer Paradox" were causing catastrophic validation overfitting, the regularized GSC-Merge model should statistically outperform unconstrained tuning on the unseen test set. Instead, unconstrained tuning consistently achieves the highest average test accuracy. This undermines the paper's primary optimization narrative.
3. **Unproven Scalability claims:** SVD of the joint multi-task update matrix is computationally expensive, scaling as $\mathcal{O}(d_{out}^2 \cdot K \cdot d_{in})$ per layer. While the authors discuss randomized SVD and block-wise decompositions as potential mitigations, they do not provide any empirical execution timings or experiments on larger models (e.g., LLMs or even a ViT-Base backbone) to substantiate these claims.
4. **Data-Text Inconsistency / Typo:** In Section 4.4 ("Addressing the Remaining Performance Gap"), the authors state: *"Under the truly task-agnostic setting, this gap becomes even wider (17.19% vs 74.96%)."* However, looking at Table 2, none of the GSC-Merge joint mean performances correspond to 17.19% (they are 16.77% for $\gamma=0.1$, 18.91% for $\gamma=0.2$, 19.08% for $\gamma=0.3$, and 20.61% for $\gamma=0.5$). This inconsistency suggests an error in the text.
5. **Practical Limitations in Extreme Settings:** All merged models perform very poorly on this conflicting task suite compared to the expert performance ceiling (74.96%). In the truly task-agnostic setting, getting ~19-20% accuracy on a 4-task suite is barely above random guessing. This makes the real-world utility of merging models across highly disparate domains questionable, a point that should be discussed more critically.

---

## 3. Detailed Dimension Ratings

### Soundness: Fair
**Justification:** While the mathematical derivations and proofs (such as the Eckart-Young-Mirsky Theorem application) are sound and correct, the core empirical claims of the paper are unsupported. Specifically, the reported standard deviations demonstrate that GSC-Merge does not reduce split-sensitivity variance compared to unconstrained tuning. Furthermore, GSC-Merge fails to outperform the unconstrained baseline on the test set, casting doubt on the practical significance of the "Overfitting-Optimizer Paradox" and the proposed regularizer.

### Presentation: Excellent
**Justification:** The paper is beautifully written, clearly structured, and easy to follow. The mathematical notation is clean and well-defined. The related work section is thorough and contextualizes the proposed framework perfectly. The tables and captions are presented clearly.

### Significance: Fair
**Justification:** The practical significance of the proposed method is limited. GSC-Merge is a low-rank extension of OFS-Tune that does not outperform the unconstrained baseline on the test set. Additionally, the massive performance gap relative to the expert ceiling on disparate tasks (getting only 20% on task-agnostic evaluation) suggests that the current formulation has limited real-world utility. The scalability of the SVD computation also remains unproven on high-capacity models.

### Originality: Good
**Justification:** Grounding model merging in spectral theory and Grassmannian projection is an original, elegant idea that moves parameter fusion away from coordinate-wise heuristics (like TIES and STA) towards structural outputactivation alignment.

---

## 4. Overall Recommendation
**Rating: 3: Weak Reject**

**Justification:** 
GSC-Merge represents a highly elegant mathematical framework that connects weight-space merging with the geometry of the Grassmannian manifold. However, from an empirical perspective, the paper falls short of supporting its central claims. The proposed regularizer does not lead to a statistically significant reduction in split-sensitivity variance, nor does it yield superior test-set performance compared to the unconstrained OFS-Tune baseline. 

Furthermore, there are minor typos (e.g., the 17.19% vs. Table 2 discrepancy), and the scalability of SVD is left unverified. For this paper to be suitable for publication, the authors must revise their empirical claims to align with their reported data, critically discuss why the unconstrained optimizer consistently achieves higher test accuracy, and demonstrate empirical scalability on larger network architectures.

---

## 5. Constructive Questions and Feedback for the Authors

1. **Variance Reduction Claims:** In the discussion, you claim GSC-Merge "dramatically reduces split-sensitivity variance across validation splits." However, in Table 1, the standard deviation for SVHN is nearly identical across unconstrained tuning and GSC-Merge ($\pm 19.70\%$ vs. $\pm 19.60\%$ for $\gamma=0.5$). For MNIST, the standard deviation of GSC-Merge ($\gamma=0.5$) is actually *higher* than unconstrained tuning ($\pm 12.89\%$ vs. $\pm 12.54\%$). Under the task-agnostic setting in Table 2, GSC-Merge ($\gamma=0.3$) also exhibits higher variance ($\pm 4.85\%$ vs. $\pm 4.81\%$). Could you please clarify this discrepancy and revise the strong claims of variance reduction to accurately reflect the reported standard deviations?
2. **The Overfitting-Optimizer Paradox:** Since unconstrained OFS-Tune consistently outperforms GSC-Merge on the unseen test set, what is the practical advantage of restricting the search space to the low-rank Grassmannian manifold? If validation overfitting were a catastrophic issue, we would expect the unconstrained baseline to generalize poorly to the test set, which is not what the empirical results indicate.
3. **Data Consistency:** Section 4.4 refers to a task-agnostic performance of 17.19% for GSC-Merge. However, Table 2 reports accuracies of 19.08% for $\gamma=0.3$ and 20.61% for $\gamma=0.5$. Could you please double-check the values in the text and the tables to ensure consistency?
4. **SVD Scalability:** Have you measured the actual Wall-clock execution times of computing the SVD per layer? For a ViT-Tiny model, SVD is fast, but for high-capacity LLMs (e.g., LLaMA-7B where $d_{out}$ and $d_{in}$ are 4096), a full SVD on concatenated task vectors would be extremely slow. Empirical timings or experiments utilizing randomized SVD on larger models would greatly strengthen your scalability arguments in Section 3.7.
