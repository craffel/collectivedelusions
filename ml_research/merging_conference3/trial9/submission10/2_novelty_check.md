# 2_novelty_check.md - Novelty and Literature Positioning Check

## Conceptual and Algorithmic Novelty
The core novelty of the submission lies in the integration of Dirichlet distribution modeling with PAC-Bayesian generalization theory to regularize test-time dynamic model ensembling on the probability simplex $\Delta^{K-1}$. 

1. **Simplex-Constrained Routing Uncertainty:** Previous regularized routers, notably **PAC-ZCA (2026)**, modeled ensembling temperatures in unconstrained spaces using Gaussian priors and posteriors. Dirichlet-PAC represents a significant conceptual shift by treating the ensembling coefficients themselves as random vectors on the simplex, constrained naturally by a Dirichlet posterior distribution.
2. **Analytical Dirichlet Complexity Penalty:** While the Kullback-Leibler (KL) divergence between Dirichlet distributions is a known mathematical result, its application as a self-stabilizing prediction-space complexity penalty in a PAC-Bayesian bound to regulate deep neural network ensembling is a novel and elegant design. The information barrier property of the Dirichlet KL divergence (preventing log-temperature collapse/explosion under finite precision bounds) is a highly clever and practical mechanism.
3. **Subspace Energy Projection (SEP):** Utilizing Singular Value Decomposition (SVD) on early-layer activations to perform unsupervised coordinate extraction is a solid engineering design. While SVD/PCA is classical, combining it with energy normalization to map the coordinates to the Dirichlet concentration parameters provides a mathematically scale-invariant and basis-independent routing driver.
4. **Unsupervised PEM-Div:** Extending the framework to a fully unsupervised setting via Prediction Entropy Minimization (PEM) is a highly valuable contribution, making the system viable for real-world privacy-preserving or label-free online streaming applications.

## Distinctions from Prior Work
The paper positions itself clearly and explicitly against its immediate predecessors:
- **Weight-Space Merging (TIES-Merging, DARE, Task Arithmetic):** Dirichlet-PAC operates in activation-space, preventing representation collapse and task interference when experts are highly heterogeneous.
- **SABLE (2025):** SABLE relies on static, hand-tuned temperatures, which cannot adapt to target stream characteristics or noisy calibration splits. SABLE also relies on centroid-based cosine similarities (which require supervised task centroids), whereas SEP is unsupervised.
- **PAC-ZCA (2026):** PAC-ZCA uses unconstrained Gaussian modeling over log-temperatures. Dirichlet-PAC addresses the "log-temperature explosion" and entropy collapse of PAC-ZCA by imposing simplex constraints directly via the Dirichlet distribution.

## Critique on Novelty
* **Textbook Mathematics:** The steps to derive the Dirichlet-to-Dirichlet KL divergence (Equation 12/Appendix A) are standard textbook statistics. The paper should make it clear that the closed-form KL expression is not a new mathematical discovery but rather a novel application within PAC-Bayes for deep model ensembling.
* **SVD as Feature Extraction:** The use of SVD for subspace identification is a widely used method. However, proving its basis-independence and scale-invariance in Proposition 3.1 is an excellent theoretical addition that strengthens the novelty of its specific application to deep neural networks.
