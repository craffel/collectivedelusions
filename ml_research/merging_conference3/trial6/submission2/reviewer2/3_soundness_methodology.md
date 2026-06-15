# 3. Soundness & Methodology Evaluation

## Clarity of Description
The description of the methodology is mathematically rigorous and highly structured. The paper outlines the parameter blending formulation, the low-dimensional state projection, the Rademacher complexity proof, and the formulation of the CFR regularizer in detail. The narrative flow is cohesive, and the notation is consistent.

---

## Technical Soundness and Methodological Flaws
Despite the impressive mathematical formalism, several critical flaws, hidden assumptions, and logical inconsistencies undermine the soundness of the proposed methodology:

### 1. The Triviality of "Absolute Resilience" (Dynamic-to-Static Collapse)
The central claim of the paper is that R2D-Merge solves the problem of "heterogeneity collapse" in dynamic model merging. However, a close inspection of the authors' own analysis reveals a profound logical flaw:
* Under the recommended regularization strength ($\lambda_{\text{wd}} = 10^{-2}$), the CFR penalty is so dominant that the optimized router weights $w_{l, k}$ are crushed to near-zero (weight-to-bias ratio $\mathcal{M}_{\text{drift}} \approx 0.012$).
* When $w_{l, k} \approx 0$, the routing function (Equation 3) simplifies to:
  $$\alpha_{l, k}(x_i) \approx b_{l, k}$$
* This means the merging coefficients are practically independent of the input sample $x_i$.
* Consequently, the model is no longer dynamic; it has collapsed into a completely static, layer-wise merger.
* The "absolute resilience (0.00% accuracy drop)" under heterogeneous batch-averaged streams is a trivial, mathematically obvious consequence of this static collapse. Since the merging coefficients do not depend on the input sample, averaging them across the batch ($\bar{\alpha} = \frac{1}{B} \sum_{b=1}^B \alpha_b$) yields exactly the same static coefficients.
* The paper essentially proposes a complex, dynamic routing network, only to regularize it so heavily that it becomes a static model. The fact that the proposed method matches the **Static Layer-Wise Optimized** baseline (where $w_{l,k} = 0$ by design) exactly in performance (65.62%) and collapse impact (0.00%) proves that the dynamic routing mechanism is entirely redundant in the robust regime.

### 2. Lack of Normalization in $C_{l, k}$ (Scale Imbalance Flaw)
The task-specific empirical covariance matrix is formulated as:
$$C_{l, k} = \frac{1}{N} \sum_{i=1}^N \|z_i^{(l)} V_k^{(l)}\|_2^2 \cdot \psi(x_i) \psi(x_i)^T$$
* The scaling term is the squared $L_2$ norm of the product of the activation $z_i^{(l)}$ (a 192-dimensional vector) and the task vector $V_k^{(l)}$ (which represents weight deviations of an entire Transformer block).
* This norm can be extremely large, and its magnitude varies dramatically across different layers $l$ and task experts $k$ depending on the scale of fine-tuned weights and activations.
* However, there is **no normalization** of this scaling term. 
* Consequently, layers or experts with larger task vectors or activation scales will exhibit disproportionately huge eigenvalues in $C_{l, k}$.
* This scale imbalance causes the joint CFR penalty to act as an overwhelming constraint on specific layers, crushing their weights to zero while leaving others under-regularized. A theoretically sound approach would normalize the activation-weight product by the number of layer parameters or the average activation scale to ensure a balanced, scale-invariant regularization across the network.

### 3. The Representational De-coupling Assumption (Remark 3.2)
To prove Theorem 3.1, the authors assume that intermediate activations $z_i^{(l)}$ are independent of the upstream routing parameters. While they attempt to justify this with the "Representational De-coupling Approximation," this assumption is highly questionable in deep, multi-layer architectures:
* In a deep network, any change in the routing weights at layer $l-1$ alters the representations at layer $l$. Errors and perturbations propagate exponentially with depth.
* The authors measure the relative activation drift empirically and find it to be small (e.g., $0.12\%$ at Block 11). However, as noted above, this drift is only small because their CFR penalty has completely crushed the routing weights to zero. If the weights are not allowed to change, the representations obviously will not drift. 
* This creates a highly circular justification: the theoretical approximation is only valid because the regularization prevents any dynamic routing, which in turn makes the dynamic routing mechanism itself redundant.

---

## Reproducibility Assessment
The methodology is mostly reproducible due to the extensive details provided in the text. However, several critical details are omitted:
1. **PCA Extraction Details:** The paper states that globally pooled representations of Block 0 are projected using a frozen PCA matrix. It is unclear if the PCA was fitted on the training split of the datasets or purely on the $N=64$ calibration samples. Fitting PCA on only 64 samples in a 192-dimensional space is highly susceptible to high sampling variance.
2. **Diagonal Loading ($\gamma$):** Section 3.4 introduces diagonal loading ($\tilde{C}_{l, k} = C_{l, k} + \gamma I$), but the experiments do not specify the value of $\gamma$ used in the main runs (e.g., in Table 4.1). Was $\gamma$ set to 0, or was some default shrinkage applied?
3. **Numerical Stability Constant ($\epsilon$):** The value of $\epsilon$ in the unit-sphere normalization (Equation 6) is not specified.
4. **Task Vectors and Scale:** The absolute scales of $V_k^{(l)}$ are not discussed, which makes replicating the exact magnitude of $C_{l, k}$ difficult without the exact checkpoint weights.
