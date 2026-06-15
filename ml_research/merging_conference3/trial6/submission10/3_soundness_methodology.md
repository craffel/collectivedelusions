# 3_soundness_methodology.md: Soundness and Methodology Critique of the Revised Paper

## 1. Description Clarity
The mathematical notations and formulas are clearly laid out in Section 3. The introduction of off-diagonal centering and signature normalization represents a positive response to previous reviews. However, the theoretical explanation of why the proposed TCPR regularizer stabilizes optimization is highly misleading because it completely ignores that the regularizer has **zero active effect** at the optimal hyperparameter.

## 2. Technical and Methodological Flaws

We have identified two critical methodological flaws that severely compromise the paper's soundness, accuracy, and scientific relevance:

### A. Extreme Under-optimization of Specialist Experts
The "specialist expert" models that serve as the foundation of the model merging framework are poorly trained and far from true convergence:
- **The Numbers:** The MNIST expert achieves only **73.20% accuracy** (where standard MNIST models easily exceed $99.5\%$ with a simple CNN or ViT), and the SVHN expert achieves a dismal **23.20% accuracy** (barely above the $10\%$ random-guess baseline for a 10-class dataset).
- **The Protocol:** Each expert was trained on a tiny subset of 1000 images per task for only 2 epochs. 
- **The Impact:** Merging models that are barely trained and poorly converged makes any findings about "bridging the gap to specialist experts" and "preventing representational collapse" completely unconvincing. Real-world model merging is performed on highly specialized, fully converged models. Under-optimized experts introduce massive parameter noise, rendering the resulting merging and routing analysis of limited practical relevance to deep learning practices.

### B. Mathematical Ineffectiveness of the Regularizer at the "Best" Hyperparameter
The paper claims that $\beta = 10^{-4}$ is the optimal regularization strength for both TCPR-Param and TCPR-Rep. However, a scaling analysis of the loss function shows that **the regularizer has virtually zero influence on optimization at this scale**:
1. **Prior Loss Magnitude:** The pairwise cosine similarities $\cos(\mathbf{w}_i, \mathbf{w}_j)$ are bounded in $[-1, 1]$. Since $S^{\text{centered}}_{i, j}$ values are extremely tiny (on the order of $0.01$), the total unscaled prior loss $\sum_{i \ne j} S^{\text{centered}}_{i, j} \cos(\mathbf{w}_i, \mathbf{w}_j)$ is on the order of $12 \times 0.01 \times 1 \approx 0.12$.
2. **Scaled Prior Loss:** When scaled by $\beta = 10^{-4}$, the regularization term is of order $1.2 \times 10^{-5}$.
3. **Cross-Entropy Loss:** The cross-entropy loss $\mathcal{L}_{\text{CE}}$ is of order $2.3$ (since the model is performing close to random guess initially).
4. **The Mismatch:** Because the cross-entropy loss ($2.3$) is **five orders of magnitude larger** than the regularizer term ($1.2 \times 10^{-5}$), the regularizer gradients are completely dwarfed during optimization. The optimizer behaves identically to the unregularized router, which explains why the optimized alphas and test accuracies are nearly identical.

## 3. Reproducibility and Statistical Rigor
While the code executes cleanly and quickly, reproducibility is heavily compromised because the reported improvements are completely non-existent under proper scaling, and are highly sensitive to the specific subset of 250 test samples. No seed variance or multi-run standard deviations are reported, which violates standard empirical research guidelines.
