# Peer Review

## Summary of the Paper
This paper proposes **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a post-hoc weight sparsification and merging framework designed to compress task-specific expert models for resource-constrained edge and IoT devices. The authors introduce two deterministic magnitude-based pruning schemes: global **Uniform Pruning (NP-BTVP-U)** and layer-wise **Adaptive Saliency-Based Pruning (NP-BTVP-S)**. To counteract update norm shrinkage when setting a large portion of parameters to zero, they incorporate a reciprocal scale factor ($1/p$ or $1/p_l$) as a "norm-preserving rescaling" heuristic. 

Through an empirical sweep using a pre-trained CLIP ViT-B/32 backbone across four vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), the authors show that:
1. When paired with rescaling, both standard AdamW and flatness-aware SAM experts demonstrate an extraordinary and nearly identical level of resilience to heavy sparsification (e.g., maintaining ~90.3% classification accuracy under 90% sparsity, extremely close to the dense 91.0% unpruned baseline).
2. Training-stage loss landscape flatness (via SAM) does not provide an additional coordinate-aligned pruning buffer under well-converged regimes compared to standard AdamW.
3. Layer-wise budget allocation in NP-BTVP-S is subject to a "Saliency Double-Bind" (scale instability under global scaling vs. local noise amplification under layer-wise scaling), making the simpler Uniform Pruning (NP-BTVP-U) the optimal and most stable choice.
4. Combining NP-BTVP-U ($p=0.10$) with post-hoc INT8 quantization yields an overall 40x parameter compression footprint (reducing expert size to 5.74 MB) with a negligible accuracy drop of only 0.12%.

---

## Strengths
1. **Engaging Narrative and Practical Directives:** The paper is exceptionally well-written, clearly motivated, and highly structured. The focus on real-world edge deployment footprints and the inclusion of concrete CSR/COO storage equations and on-disk memory savings make it highly valuable for practitioners.
2. **Analytical Derivations:** The authors provide rigorous analytical derivations in the appendix of the expected $L_1$ norm ratio and expected $L_2$ reconstruction error under standard Laplace and Gaussian weight-residual distributions. This elevates the work above a purely empirical heuristic paper.
3. **Rigorous Statistical Verification:** Evaluated over three independent random seeds with standard deviations reported. The use of a two-tailed paired $t$-test to show that Saliency and Uniform pruning are statistically indistinguishable ($p$-values of 0.96 and 0.68) provides strong, pragmatically grounded design guidance (supporting the choice of the simpler Uniform method).
4. **Quantization Synergy:** The empirical demonstration of a joint sparsification-quantization pipeline (90% pruning + INT8 quantization) achieving a 40x footprint reduction with only a 0.12% drop in accuracy is a powerful engineering result.

---

## Weaknesses

### 1. Mathematical Equivalence of "Norm-Preserving Rescaling" to Merging scale Reparameterization
The core theoretical claim that "norm-preserving rescaling" is a crucial new post-hoc weight sparsification operator is undermined by its mathematical equivalence to scaling up the merging coefficient $\lambda_k$. 

In Uniform Pruning (NP-BTVP-U), the rescaled sparse task vector is defined as:
$$\tilde{\tau}_k^{(p)} = \tau_k \odot M_k^{(p)} \times \frac{1}{p}$$
where $M_k^{(p)} \in \{0, 1\}^d$ is the magnitude mask. The final merged model is reconstructed as:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \lambda_k \tilde{\tau}_k^{(p)}$$
By substituting the definition of $\tilde{\tau}_k^{(p)}$, we get:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \lambda_k \left( \tau_k \odot M_k^{(p)} \times \frac{1}{p} \right) = \theta_{\text{base}} + \sum_{k=1}^K \left( \frac{\lambda_k}{p} \right) \left( \tau_k \odot M_k^{(p)} \right)$$
Let $\bar{\tau}_k^{(p)} = \tau_k \odot M_k^{(p)}$ be the standard, unrescaled pruned task vector. Let $\bar{\lambda}_k = \frac{\lambda_k}{p}$ be a reparameterized merging coefficient. The merged model becomes:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \bar{\lambda}_k \bar{\tau}_k^{(p)}$$
This proof shows that **the proposed norm-preserving rescaling is mathematically identical to performing unrescaled magnitude-based pruning and scaling up the merging coefficient by $1/p$**. 

During merging, the coefficients $\lambda_k$ are always swept and optimized post-hoc. The authors state that they sweep $\lambda \in [0.1, 1.0]$. Under $p=0.10$, the optimal unrescaled coefficient should be $\bar{\lambda}_k \approx 10 \lambda_k \approx 3.0$. However, because the authors capped their hyperparameter search space for the unrescaled baseline at $1.0$, the baseline was artificially prevented from reaching its optimal scale. The reported "performance collapse" (80.45% accuracy) of the unrescaled baseline is therefore **an artifact of a restricted hyperparameter search space rather than a physical or mathematical limitation of unrescaled pruning itself**. If the baseline sweep for $\bar{\lambda}_k$ had been extended to $[0.1, 10.0]$, the unrescaled baseline would have achieved the exact same performance as the rescaled method. This severely weakens the theoretical novelty of the proposed rescaling operator.

### 2. Conceptual Contradiction in the "Norm-Preserving" Naming
The framework is named "Norm-Preserved" Budgeted Task-Vector Pruning. However, as the authors formally derive in Appendix Sections 1.1 and 1.2, the $1/p$ scale factor does **not** preserve the $L_1$ norm. Under Laplace and Gaussian distributions with $p=0.10$, the expected $L_1$ norm of the rescaled vector is **boosted by 3.30x and 2.58x**, respectively. 
Calling the framework "Norm-Preserved" when it mathematically multiplies the norm by ~3x is a conceptual misnomer. It should be more accurately framed as "Signal-Strength Boosting" or "Over-scaled Pruning".

### 3. Hand-Waved Explanation of the Reconstruction Error Paradox
In Appendix Section 1.3, the authors derive the expected $L_2$ reconstruction error and note that for $p = 0.10$, the quadratic multiplier is $(\frac{1}{p} - 1)^2 = 81$, which causes a massive $L_2$ distortion in parameter space. They state: *"This elegant trade-off explains the minor performance gap (0.60%) between the rescaled 10% sparse model and the uncompressed dense model..."*
Theoretically, this is hand-waved. A massive $L_2$ error would normally imply that the weights are far from their optimal values, which should *increase* the performance gap. 
A rigorous explanation must account for why such a massive parameter-space distortion results in only a 0.60% drop. Formally, this is because the final merged parameter difference $\Delta_{\text{MTL}}$ is scaled down by the small merging coefficients squared:
$$\mathbb{E}[\|\Delta_{\text{MTL}}\|_2^2] = \sum_{k=1}^K \lambda_k^2 \mathbb{E}[\|\tilde{\tau}_k^{(p)} - \tau_k\|_2^2]$$
Since $\lambda_k \approx 0.3 \implies \lambda_k^2 \approx 0.09$, the massive individual reconstruction error is scaled down by a factor of ~11. Additionally, because the pre-trained base model is highly overparameterized, the loss landscape has massive flat directions (null spaces of the Hessian), which project away the remaining parameter-space distortion.

### 4. Experimental Toy Scope and Overparameterization Regimes
The empirical evaluation is conducted on low-scale toy vision datasets (MNIST, Fashion, CIFAR-10, SVHN) in an extremely restricted low-data regime (**1024 samples**) for only **5 epochs** with a small learning rate ($10^{-5}$). 
Fine-tuning a 28.7 million parameter encoder under these settings ensures that the model operates in a highly overparameterized, interpolating regime. The resulting task vectors $\tau_k$ are extremely close to zero. Consequently, pruning 90% of them introduces an infinitesimal absolute perturbation in weight space, which naturally explains the "extraordinary resilience to heavy sparsification". On larger datasets or standard full fine-tuning regimes (where task vectors undergo significant drift and have larger norms), magnitude-based pruning would likely cause much more severe degradation.

---

## Detailed Ratings

### Soundness: Fair
The analytical derivations in the appendix are mathematically correct and elegant. However, the core experimental baseline comparison is fundamentally flawed due to a restricted hyperparameter search space that artificially handicaps the unrescaled baseline. Furthermore, the explanation of why a massive $L_2$ reconstruction error results in a minor performance gap is theoretically hand-waved.

### Presentation: Excellent
The paper is exceptionally well-written, structured, and easy to read. The equations are clean, the figures are professional, and the tables are complete and clear.

### Significance: Fair
The practical edge-deployment storage recipes (INT8 quantization + 10% sparse expert) and the analysis of the Saliency Double-Bind are useful for practitioners. However, because the core "Norm-Preserving Rescaling" is mathematically equivalent to scaling the merging coefficient $\lambda_k$, the theoretical and scientific impact of the proposed mechanism on the model merging community is limited.

### Originality: Fair
The framework combines standard magnitude-based pruning with a scaling factor ($1/p$) that is mathematically equivalent to a reparameterization of the merging coefficient $\lambda_k$. The layer-wise saliency scheme (NP-BTVP-S) is shown to be inferior/equivalent to the uniform baseline, and the findings regarding SAM and AdamW are easily explained by standard overparameterization and Hessian eigenvalue properties. Thus, the overall novelty is incremental.

---

## Overall Recommendation

**Rating:** 3: Weak Reject

**Justification:** 
The paper has clear merits in its writing quality, detailed practical edge-deployment analyses, and mathematically sound appendix derivations. However, the core mechanism ("Norm-Preserving Rescaling") is mathematically equivalent to simply scaling the merging coefficient $\lambda_k$, making the key baseline comparison in the paper's ablation study unfair and misleading. Furthermore, the evaluation is restricted to a highly specific, low-data, overparameterized regime on toy datasets, making the claimed sparsification resilience a likely artifact of the training regime. For these reasons, the paper requires significant revisions (specifically, expanding the baseline's sweep range to make the comparison fair, correcting the "Norm-Preserving" misnomer, and validating on larger datasets/LLMs) before it can be accepted.

---

## Constructive Suggestions and Questions for the Authors
1. **Explain the Merging Coefficient Equivalence:** How do you justify the claim that "norm-preserving rescaling" is a distinct, necessary post-hoc operator when it is mathematically identical to performing unrescaled pruning and setting $\bar{\lambda}_k = \lambda_k / p$?
2. **Expand the Baseline Sweep Range:** To prove that norm-preserving rescaling is a fundamental physical necessity, please run an experiment where you evaluate the unrescaled magnitude-pruning baseline with an expanded merging coefficient sweep range, e.g., $\bar{\lambda}_k \in [0.1, 10.0]$ for $p = 0.10$. Does it not achieve the exact same performance as your rescaled method?
3. **Correct the Naming Misnomer:** Consider renaming the framework to "Signal-Boosted Budgeted Task-Vector Pruning" or similar, since your appendix derivations formally prove that the $1/p$ factor does not preserve the $L_1$ norm but rather boosts it by 2.58x to 3.30x.
4. **Validate on Standard/Large-Scale Benchmarks:** To prove that your sparsification resilience is not an artifact of the low-data (1024 samples) interpolating regime, please provide empirical results on standard full-dataset fine-tuning (e.g., full CIFAR-100 or ImageNet-1K) or modern LLM merging benchmarks.
5. **Rigorous $L_2$ Reconstruction Analysis:** Please replace the hand-waved explanation of the "elegant trade-off" regarding $L_2$ reconstruction error with a rigorous mathematical sensitivity analysis of the final merged parameter space and Hessian null space, as outlined in the review.
