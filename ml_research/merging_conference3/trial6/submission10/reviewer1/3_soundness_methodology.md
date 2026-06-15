# 3. Soundness and Methodology

## Clarity of Description
The proposed methodology is clearly written and structured. The mathematical formulations for the BSigmoid-Router (Eq 8-10) and TCPR (Eq 11-17) are explicit and easy to follow. The appendix provides sufficient detail regarding hyperparameters, architecture ($\mathtt{vit\_tiny\_patch16\_224}$), and the optimization setup.

## Appropriateness of Methods & Potential Technical Flaws

1. **The Core Self-Contradiction Flaw:**
   The paper suffers from a catastrophic disconnect between its theoretical pitch and its empirical results.
   - **The Claim (Abstract & Intro):** The authors pitch TCPR as a "simple yet highly effective approach," claiming that "TCPR consistently prevents high-conflict task collapse and bridges the performance gap to specialist experts" and "provides a robust, scale-invariant pathway."
   - **The Reality (Section 4.4 & 5):** The authors' own results show that TCPR **does not work**. It fails to improve performance over the unregularized BSigmoid-Router baseline at any scale. At small scales ($\beta = 10^{-6}$), it is "mathematically dead" and has no effect. At larger scales ($\beta \ge 1.0$), it severely degrades performance due to representation interference.
   - **Conclusion:** It is highly inappropriate to write a paper that claims to propose and validate a "highly effective" method (TCPR) when the empirical results demonstrate that the method is a complete failure and that unregularized sigmoidal routing is actually the optimal approach.

2. **Under-Trained and Sub-Optimal Experts:**
   - The task-specific experts are severely under-trained, achieving only **73.20% on MNIST** (where standard classifiers exceed 99.5%) and **23.20% on SVHN** (barely above random guessing for a 10-class dataset).
   - The authors attempt to justify this by claiming it "simulates realistic edge-AI scenarios where practitioners cannot afford full-scale converged pre-training." This justification is highly skeptical. Fine-tuning a ViT-Tiny on MNIST to >95% accuracy takes less than a few minutes on a standard GPU. 
   - Because the experts are sub-optimal and filled with parameter noise, the dynamic routing head is essentially learning to route between noisy representations. This makes the entire setup highly unrealistic. Model merging is designed to combine the strengths of high-performing models, not to patch together poorly trained ones.

3. **Arbitrary Centering of the Similarity Matrix:**
   - In Equation 13, the similarity matrix $S_{i, j}$ is centered by subtracting the off-diagonal mean $\mu_{\text{off}}$. This is mathematically arbitrary and lacks theoretical grounding.
   - Centering forces some task relationships to be negative (conflicting) and others to be positive, purely relative to the average similarity of the chosen task set. If a set contains only highly similar tasks, centering will falsely categorize some of them as conflicting. If the set contains only highly dissimilar tasks, centering will falsely categorize some as highly similar. This dependency on the specific task mixture makes the regularization highly unstable and unprincipled.

4. **Flawed Signatures Cosine Similarity Assumption:**
   - Equation 16 regularizes the cosine similarity of the routing signatures (the projection weights $\mathbf{w}_i, \mathbf{w}_j \in \mathbb{R}^D$).
   - The raw logit for task $i$ is $o(x)_i = \mathbf{w}_i^T z(x) + b_i$. Regularizing $\cos(\mathbf{w}_i, \mathbf{w}_j)$ assumes that the intermediate representations $z(x)$ are uniformly distributed on a sphere.
   - In practice, intermediate activations $z(x)$ lie in a highly structured, narrow cone or subspace. Aligning or orthogonalizing the weight vectors $\mathbf{w}_i$ in isolation—without considering the covariance structure of the representation space $z(x)$—is a major theoretical oversight and likely explains why the regularization degrades performance when active.

## Reproducibility
- While the paper provides hyperparameter tables, architectures, and states that they used strict seed control ($\mathtt{seed=42}$), the lack of statistical validation is a major flaw.
- The results are reported for a **single seed** ($\mathtt{seed=42}$) on a tiny calibration split of 64 images. Because the dataset is so small, the optimization is highly sensitive to the specific samples chosen and the initialization.
- The authors do not provide means, standard deviations, or confidence intervals across multiple random seeds or different calibration splits. A difference of 0.3% (between 25.50% and 25.20%) could easily be within the margin of noise. Without statistical significance tests, any claims about the regularizer being "dead" vs. slightly worse are speculative.
