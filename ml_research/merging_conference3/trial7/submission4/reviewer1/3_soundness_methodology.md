# Soundness and Methodology Evaluation - 3_soundness_methodology.md

## Description Clarity
The methodology is exceptionally well-written, with clean mathematical formulations and step-by-step progressions. The paper presents a highly clear narrative and formalizes its steps (SVD centroid extraction, Gram matrix computation, Löwdin orthogonalization, absolute projection, temperature Softmax gating) with rigorous notation. The additions of Top-$k$ Sparse Gating, self-calibrated temperature scheduling, and covariance whitening are clearly integrated.

---

## Appropriateness of Methods
- **SVD Centroid Extraction:** SVD is highly appropriate for extracting task centroids compared to naive averaging because it captures the principal axis of prototype variance, avoiding the sum-to-zero cancellation of symmetrically distributed class weights.
- **Löwdin Orthogonalization:** Löwdin's method is mathematically elegant for symmetric orthogonalization because it is order-invariant (unlike Gram-Schmidt) and minimizes the least-squares distance to original centroids.
- **Absolute Projection:** Taking the absolute value of the projection $u_{k, b} = |\bar{v}_k \cdot \tilde{z}_b|$ is a practical heuristic to handle bidirectional class prototype distributions, ensuring both halves of the class distribution project to positive gating scores.

---

## Potential Technical Flaws & Methodological Concerns

### 1. Degenerate Singular Values under Perfectly Symmetric/Orthogonal Prototypes
The authors assume that the top right-singular vector $V_{k, 1}$ is a unique and stable direction representing the task. However, in perfectly symmetric or orthogonal classification spaces (e.g., where class prototypes are mutually orthogonal and have equal norm, representing a standard maximum-margin configuration), the Gram matrix of the prototypes $W_k W_k^T$ has degenerate (identical) eigenvalues. 
When the top eigenvalues are degenerate or extremely close, the corresponding top singular vector is **not unique**; any linear combination in the eigenspace is a valid top singular vector. This makes the SVD centroid highly sensitive to minor numerical perturbations or floating-point differences. This mathematical edge case is unaddressed, and could lead to extreme centroid instability across different architectures, seeds, or floating-point precisions.

### 2. High Susceptibility of Absolute Projection to Out-of-Distribution (OOD) Noise
By utilizing the absolute value of the projection $u_{k, b} = |\bar{v}_k \cdot \tilde{z}_b|$, the gating coordinate loses directional specificity. A feature vector $z_b$ that is completely unrelated to the target task but has high magnitude along the centroid direction (even in the negative semantic direction) will be strongly routed to that task's expert. This bidirectionality increases the susceptibility of the router to false positives and OOD representation noise compared to a directional or bounded similarity metric.

### 3. Hyperparameter Sensitivity of Temperature $\tau$ and Standard Deviation Gating
The gating mechanism is highly sensitive to the temperature $\tau$. While the authors propose a self-calibrated temperature $\tau_b = \gamma \cdot \text{std}_k(u_{k, b})$, this introduces another hyperparameter $\gamma$. Under heavy task overlap, a small $\gamma$ behaves like hard gating (causing a hard gating penalty), whereas a large $\gamma$ softens routing to uniform merging. The claim that this is "completely self-calibrated" is slightly overstated as the selection of $\gamma$ still requires manual tuning depending on the active overlap and noise scale of the registry.

---

## Reproducibility Analysis
- **Availability of Code:** Crucially, there is **no implementation source code** (Python, PyTorch, etc.) included in the submission files. The authors evaluate their methods on a 10-seed simulation sandbox and a ResNet-18 feature manifold, but the code to reproduce these environments, the baselines (such as QWS-Merge and L3-Softmax), or the results is completely missing.
- **Mathematical Reproducibility:** The mathematical details provided are sufficient for a competent researcher to reimplement the formulas. However, reproducing the exact empirical results (e.g., the 74.46% ceiling accuracy, the noise scales, and the asymmetric sandbox layout) is impossible without the exact data generation scripts and experimental setups. This is a significant concern for empirical validation.
