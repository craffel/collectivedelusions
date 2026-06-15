# Peer Review: Sparse Task Arithmetic (STA)

## Overall Recommendation
- **Score:** 4: Weak Accept
- **Soundness:** Good
- **Presentation:** Excellent
- **Significance:** Good
- **Originality:** Good

---

## 1. Summary of the Submission
The submission, titled **"Sparse Task Arithmetic (STA): Deconstructing the Redundancy of Sign-Resolution in Model Merging"**, presents a deconstructive critique of training-free sparse model merging techniques, specifically targeting **TIES-Merging** and **DARE**. The authors challenge the prevailing research trend of introducing complex, multi-stage heuristics (such as coordinate-wise sign voting, sign consensus, and stochastic drop-and-rescale) to resolve parameter interference in weight space.

Guided by Occam's razor, the authors propose a minimalist baseline called **Sparse Task Arithmetic (STA)**. STA consists of two basic steps:
1. **Layer-wise Magnitude Pruning:** Pruning each task vector to retain only the top-$s$\% largest absolute updates in each layer.
2. **Linear Addition:** Directly summing the sparse updates and adding them to the pre-trained base model, without any sign voting, sign consensus, or disjoint merging.

To address the **under-scaling confounder** (where zeroing out updates at low survival densities reduces update energy and degrades performance), the authors propose two scale-preserving variants:
- **Rescaled STA (R-STA):** Surviving updates are divided by the survival density $s/100$, scale-preserving them exactly.
- **Tuned STA:** The scaling coefficient $\lambda$ is dynamically adjusted (e.g., to $\lambda = 0.8$ at $s = 20\%$) to match the optimal weight space energy level.

Following a previous review cycle that highlighted a severe hyperparameter tuning confounder, the authors have revised their paper to include a **fully symmetric hyperparameter sweep** across all methods. The results are reported at each method's individual optimal scaling coefficient $\lambda^*$:
- **Tuned Task Arithmetic (s=100%):** Peaks at $\lambda^* = 0.5$ with **88.64%** average accuracy.
- **Tuned DARE (s=20%):** Peaks at $\lambda^* = 0.4$ with **88.95%** average accuracy.
- **Tuned TIES-Merging (s=20%):** Peaks at $\lambda^* = 0.5$ with **90.16%** average accuracy.
- **Tuned STA (Ours, s=20%):** Peaks at $\lambda^* = 0.8$ with **90.53%** average accuracy.

The paper evaluates these variants on a 4-task vision-transformer multi-task suite (MNIST, FashionMNIST, CIFAR-10, SVHN) using a pre-trained ViT-B-32 backbone.

---

## 2. Major Strengths
1. **Scientific Rigor and Fairness (Symmetric Tuning):** The authors are highly commended for implementing a fully symmetric hyperparameter sweep over the scaling coefficient $\lambda \in [0.1, 1.0]$ with a step size of $0.1$ across ALL baseline methods. Evaluating all methods at their peak tuned configurations is the gold standard of scientific fairness. It directly resolves the previous confounder and establishes a rock-solid empirical comparison.
2. **Exemplary Writing and Structure:** The paper is exceptionally well-written, clear, and logically organized. The narrative framing around Occam's razor is compelling, and the deconstructive tone is intellectually engaging.
3. **Deep Scholarly Insights (Tail-Bias Analysis):** The authors have added a deep, mathematically intuitive analysis of the "tail-bias" of magnitude-based pruning vs. the "uniform variance" of stochastic dropout (Section 4.3). This provides an elegant, scholarly explanation for why Rescaled STA fails at low densities while DARE succeeds. It represents a highly valuable conceptual contribution to our understanding of sparsified weight spaces.
4. **Valuable Demystification of Sign Consensus:** By demonstrating that a much simpler baseline (Tuned STA) performs comparably to TIES-Merging once update scale is balanced, the paper provides a crucial course correction for the model merging community, demonstrating that complex sign-consensus and majority-voting steps are largely redundant.

---

## 3. Remaining Limitations and Weaknesses

While the revised paper has successfully resolved the most critical technical flaws from the previous cycle, several minor and moderate limitations remain. 

### Limitation 1: Outdated and Restricted "Toy" Benchmark Suite
The evaluation remains restricted to a 4-task vision-transformer multi-task suite: MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Grayscale / Toy Tasks:** MNIST and FashionMNIST are low-resolution (28x28) grayscale datasets that are extremely easy to classify and have been considered "solved" or "toy" benchmarks for over a decade. They are not representative of modern, challenging vision tasks.
- **Omission of Challenging Tasks:** Standard sparse model merging papers (including TIES-Merging and DARE) typically evaluate on a larger 8-task suite that includes much more challenging vision domains such as **Stanford Cars**, **DTD** (Textures), **EuroSAT**, **GTSRB**, **RESISC45**, and **SUN397**. 
- **High Risk of Overlap Overestimation:** On SVHN (which has the largest domain shift of the 4 evaluated datasets), standard un-rescaled STA (68.70%) is beaten by TIES-Merging (73.97%) by **5.27%** and DARE (78.71%) by **10.01%** at equivalent sparsity. This indicates that task interference is highly active. It is highly likely that on harder datasets (Cars, DTD, SUN397) where task interference is much more severe, STA's performance would drop catastrophically without sign consensus. The omission of these datasets is a significant limitation of the empirical validation.

### Limitation 2: Marginal Performance Gains and Statistical Significance
In Table 1, Tuned STA (90.53% average accuracy) is reported as outperforming Tuned TIES-Merging (90.16% average accuracy).
- The margin of improvement is only **0.37% absolute**.
- Evaluated on a 16-batch validation split containing 2,048 samples per dataset, a difference of 0.37% corresponds to only **7 samples** out of 2,048. 
- For a pre-trained Vision Transformer, a difference of 7 samples is well within the margin of statistical error and random variance.
- Therefore, the claim that Tuned STA "substantially outperforms" or "outperforms" TIES-Merging is overstated. Scientifically, the results indicate that Tuned STA performs *comparably* to Tuned TIES-Merging, which still supports the minimalist hypothesis but refutes any claim of superior performance.

### Limitation 3: Technical Terminology Misnomer ("Isotropic Pruning")
Throughout the paper, the authors refer to their pruning process as **"isotropic layer-wise magnitude pruning."**
- In multidimensional space, **isotropic** means having physical properties that are identical in all directions (i.e., invariant under rotation).
- Magnitude-based pruning is coordinate-dependent and highly **anisotropic** because it selectively retains parameter updates with the largest absolute values. 
- Indeed, in Section 4.3, the authors explicitly explain that magnitude pruning suffers from **"tail-bias"** because it selectively retains the extreme tails of the update distribution, which distorts the parameter variance. 
- Calling a process "isotropic" when it is highly anisotropic and selectively distorts variance is a technical contradiction. The terminology should be corrected to "layer-wise magnitude pruning" or "layer-wise uniform magnitude pruning."

### Limitation 4: Omission of the Stronger DARE-TIES Baseline
The paper's DARE baseline is implemented as delta-dropout and rescaling followed by direct linear addition (which is DARE-TA).
- In the original DARE paper (Yu et al., 2024), DARE's delta-dropout is combined with TIES-Merging (DARE-TIES) to achieve peak state-of-the-art performance. DARE-TIES incorporates TIES-style sign consensus and disjoint merging.
- By evaluating only DARE-TA (without sign consensus), the authors omit the stronger version of the DARE baseline, which represents an incomplete comparison against the state-of-the-art.

---

## 4. Minor Suggestions and Questions

1. **Terminology Clarification:** Please replace the term "isotropic magnitude pruning" with "layer-wise magnitude pruning" or "layer-wise uniform magnitude pruning" throughout the paper to ensure mathematical precision.
2. **Inference Footprint:** Since the final merged model $\theta_{\text{merged}}$ is a dense model, does having sparse task vectors prior to merging provide any inference-time computational or memory benefit? If not, please clarify the practical motivation of sparsity in this context.
3. **Collision Rate under Task Similarity:** The collision probability analysis in Section 3.2 assumes independent pruning masks. However, fine-tuned models on a shared base often update similar parameter regions when adapted to similar domains. While the authors show that the overlap in their specific 4-task suite is 3.1%–4.3% (matching the 4.0% independence bound), this is a consequence of their chosen toy datasets spanning highly diverse and unrelated domains (digits, apparel, general objects). For tasks within the same domain (e.g., multiple text classification tasks fine-tuned on an LLM), the mask overlap is expected to be much higher, leading to severe parameter collisions that make sign voting essential. Can the authors evaluate and report overlap rates for more closely related tasks?

---

## 5. Actionable and Constructive Feedback for Revisions

To further elevate this work for its final camera-ready version, the authors should address the following points:
1. **Correct Technical Terminology:** Replace "isotropic magnitude pruning" with a technically precise term such as "layer-wise magnitude pruning."
2. **Report Full-Set Evaluation:** Replace the 2,048-sample validation subset evaluations with evaluations on the full validation/test sets to ensure statistical significance.
3. **Include DARE-TIES Baseline:** Run and report results for DARE-TIES (which combines delta-dropout with sign consensus) to ensure a complete baseline comparison.
4. **Discuss Task Similarity and Overlap:** Add a brief discussion in Section 3.2 addressing the limitations of the mask independence assumption under high task similarity.
