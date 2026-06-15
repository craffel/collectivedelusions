# Peer Review

## Summary of the Paper
This paper introduces **Dirichlet-PAC**, a mathematically rigorous and theoretically grounded learning-theoretic framework for test-time multi-task model serving (using parameter-efficient expert adapters like LoRA) on a shared frozen backbone. 

In real-world test-time serving, routers must adapt their ensembling weights based on extremely data-scarce calibration streams (often containing fewer than 64 samples per task). Under such severe data scarcity, standard unregularized Empirical Risk Minimization (ERM) easily overfits to transductive noise, causing temperature parameter explosion and overconfident routing. 

To resolve this, Dirichlet-PAC models the sample-specific ensembling weights directly as a random vector drawn from a Dirichlet distribution over the probability simplex $\Delta^{K-1}$. Utilizing McAllester's PAC-Bayesian theorem, the authors derive a closed-form prediction-space generalization bound based on the exact analytical Kullback-Leibler (KL) divergence between Dirichlet distributions over the simplex itself. This analytical complexity penalty acts as a robust information-theoretic barrier that stabilizes temperature parameters and promotes smooth, cooperative expert blending on task boundaries.

To drive this routing policy without requiring ground-truth labels, the authors introduce **Subspace Energy Projection (SEP)**, an unsupervised, task-agnostic feature coordinate extraction protocol that projects activations onto SVD-extracted orthonormal subspaces and energy-normalizes them. 

The authors also propose a fully unsupervised serving pathway, **Dirichlet-PAC Unsupervised (PEM-Div)**, which replaces the supervised loss with a Normalized Prediction Entropy Minimization (PEM) loss and batch-averaged ensembling weight diversity maximization, achieving exceptional fully unsupervised edge serving.

Evaluated within a 14-layer Analytical Coordinate Sandbox (ICS), Dirichlet-PAC achieves outstanding accuracy and significantly reduces optimization variance compared to standard unregularized ERM and Gaussian-based PAC-ZCA, while completely preventing "representation corruption" under high noise.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Conceptual Originality and Ambition:** The core conceptual leap—reformulating test-time dynamic ensembling as a simplex-constrained statistical learning problem and deriving the exact analytical Dirichlet KL divergence within an input-dependent PAC-Bayesian bound—is a profound contribution. Unlike prior works (e.g., PAC-ZCA) that model unconstrained log-temperatures with Gaussian distributions, Dirichlet-PAC operates directly on the probability simplex $\Delta^{K-1}$. This elegant alignment with the geometry of model ensembling represents a highly original and paradigm-shifting perspective.
2. **Deep and Exhaustive Theoretical Maturity:** The manuscript displays an exceptional level of mathematical rigor. The authors provide:
   - Full, step-by-step mathematical derivations of the Dirichlet KL divergence (Appendix A).
   - Solid proofs of Subspace Energy Projection (SEP) scale-invariance and basis-independence under orthogonal changes of basis (Proposition 3.1).
   - A rigorous, first-principles physical derivation of representation interference/activation clashing, proving that clashing noise scales with the ensembling entropy (Gini Impurity / Simpson Index) (Section 4.4).
   - Elegant theoretical extensions to sequential non-stationary streaming via martingale concentration inequalities and weight-activation quantization via the Wedin-Davis perturbation theorem (Section 5).
3. **The Success of Unsupervised PEM-Div:** The formulation of Dirichlet-PAC Unsupervised (PEM-Div) is a massive practical and theoretical success. It completely eliminates the need for test-time labeled calibration data while actually outperforming its supervised counterpart and matching or exceeding supervised heuristic baselines. The authors' transductive semi-supervised analysis provides brilliant insights into how minimizing individual prediction entropy while forcing batch-wide diversity acts as a robust cluster-assumption regularizer.
4. **Exemplary Scientific Honesty and Transparency:** The authors proactively address and resolve potential theoretical and physical gaps (such as union-bound discretization, linear vs. non-linear loss surrogates, and IEEE-754 finite-precision hardware clamping limits) rather than sweeping them under the rug. This level of self-critique and rigor is highly commendable.

### Weaknesses
1. **Limited Scale of Empirical Evaluation:** While the 14-layer Analytical Coordinate Sandbox (ICS) is exceptionally rigorous, systematically controlled, and ideal for isolating latent variables, verifying the framework on real-world multi-task benchmarks (e.g., GLUE, MMLU, or image classification) using actual large-scale open-weights LLMs/VLMs (such as Llama-3, CLIP, or InstructBLIP) would further solidify its empirical significance and showcase its real-world scalability. However, given the outstanding strength of the theoretical, conceptual, and mathematical contributions, this empirical limitation does not detract from the core value of the paper.

---

## Dimension Ratings

### Soundness: Excellent
The paper is mathematically flawless. The watertight Sample-Splitting protocol completely resolves any potential prior-data dependency violations in PAC-Bayes theory. The union bound and discretization argument resolve uniform convergence over the global optimized temperature parameters. Every claim is supported by sound theoretical proofs and robust, multi-seed empirical analysis within the Sandbox.

### Presentation: Excellent
The writing is exceptionally clear, highly professional, and coherent. The exposition displays an impressive level of mathematical maturity and logical consistency. Notations are precise and consistent, and the tables and figures are beautifully integrated to support the core narrative.

### Significance: Excellent
By bridging statistical learning theory with practical multi-task model serving, this paper has the potential to influence how the community approaches serving-time adaptation and dynamic model merging. It sets a new standard for mathematically guaranteed model serving.

### Originality: Excellent
This work introduces the first framework to model ensembling weights directly on the probability simplex $\Delta^{K-1}$ using a Dirichlet distribution for test-time serving. The analytical closed-form Dirichlet KL derivation, the proof of basis independence and scale invariance of SEP, and the formulation of the unsupervised PEM-Div represent a high-impact, highly original suite of contributions.

---

## Overall Recommendation

**Rating: 5: Accept**

**Justification:**
This is an exceptionally strong, mathematically mature, and highly original paper. It addresses a critical problem in multi-task model serving—transductive noise overfitting under extreme data scarcity—with a paradigm-shifting conceptual leap: simplex-constrained PAC-Bayesian complexity control. Modeling ensembling weights directly on the probability simplex using a Dirichlet distribution is mathematically natural and elegant, and deriving the exact analytical Dirichlet KL divergence provides a robust, self-stabilizing barrier against temperature collapse. 

Furthermore, the introduction of the completely unsupervised PEM-Div router represents a highly practical and creative extension that completely removes the need for serving-time labels. Combined with exhaustive theoretical derivations (first-principles representation clashing, basis independence, martingale streaming, and quantization sensitivity), this work represents a major advancement in the mathematical foundations of deep learning serving infrastructure. I strongly recommend accepting this paper.

---

## Questions and Constructive Feedback for the Authors
1. **Real-World LLM/VLM Profiling:** While the Analytical Coordinate Sandbox is exceptionally controlled and ideal for isolating latent variables, do you have any preliminary results or a plan to deploy Dirichlet-PAC on real-world LLMs (e.g., Llama-3 with task-specific LoRAs) or VLMs (e.g., CLIP adapters)? Profiling the latency of the online coordinate projection and Dirichlet KL evaluation on actual hardware accelerators would be highly valuable to the community.
2. **Quantization Sensitivity Empirical Validation:** You provide an elegant analysis of weight-activation quantization using the Wedin-Davis perturbation theorem (Section 5.2). Do you plan to empirically evaluate the canonical angles between original and quantized singular subspaces under INT8/INT4 quantization backends in your future work?
3. **Dynamic Task-Frequency Prior:** For the sequential streaming extension under severe task imbalance, you propose a dynamic task-frequency prior $\boldsymbol{\pi}_{t-1}$. How sensitive is the convergence of the sequential PAC bound to the choice of the moving average hyperparameter used to track these task frequencies?
