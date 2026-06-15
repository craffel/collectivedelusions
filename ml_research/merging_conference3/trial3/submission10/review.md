# Official Peer Review (Mock Review)

## 1. Summary of the Submission
The submission introduces **ChebyMerge** (Stable and Optimal Continuous Subspace Model Merging), a continuous parameterization framework designed to resolve the limitations of dynamic layer-wise merging under unsupervised test-time adaptation (TTA). Dynamic model-merging methods (such as AdaMerging) optimize blending coefficients ($\lambda_{k,l}$) on-the-fly using self-supervised objectives (such as Shannon entropy minimization) on unlabeled local test batches. 

The authors identify and address two critical challenges:
1. **The Overfitting-Optimizer Paradox:** Unconstrained optimizers possessing high degrees of freedom ($K \times L$ free variables) overfit to the transductive local batch sampling noise rather than aligning to the network's underlying semantic layer sensitivity profile, leading to representation collapse.
2. **Exponential Ill-Conditioning of Power Bases:** Prior continuous subspace methods (such as PolyMerge) represent layer-wise coefficients using standard monomials ($1, \bar{l}, \bar{l}^2, \dots$), yielding a Vandermonde-type design matrix whose Gram matrix condition number scales exponentially as $\mathcal{O}(4^d)$ (exceeding $10,400$ for a cubic parameterization). This extreme ill-conditioning creates highly anisotropic, "stiff" loss landscapes that cause vanishing/exploding gradients and impede convergence.

To resolve these, ChebyMerge projects the spatial coefficient space onto a low-dimensional continuous subspace spanned by **Chebyshev polynomials of the first kind**. The framework ensures near-perfect numerical conditioning, minimax-optimal uniform approximation, and an implicit boundary sensitivity matching that concentrates representational resolution at the highly sensitive early and late layers. To separate numerical stability from parameter regularization, the paper further introduces **Controllable Spectral Decay (CSD)**, which explicitly decays the learning rates of higher-order Chebyshev coefficients based on their frequency order.

The method is evaluated using a physically-grounded coupled non-convex optimization simulator that models the non-uniform layer sensitivities, inter-layer covariance couplings (via a non-diagonal covariance matrix $\boldsymbol{\Sigma}$), and multi-scale transductive noise of deep vision-language networks.

---

## 2. Strengths
- **Rigorous Mathematical Foundations:** The theoretical analysis is highly elegant. The proof of the exponential condition number scaling of standard monomials (Theorem 3.1) by establishing the continuous limit as the celebrated **Hilbert matrix** is mathematically beautiful and rigorous. It provides a solid theoretical explanation for the optimization stiffness observed in prior works.
- **Novel Conceptual Frameworks:** The exposure and framing of the **Conditioning-Generalization Paradox** is an outstanding, highly original conceptual contribution. Revealing that PolyMerge's extreme ill-conditioning acts as an accidental, implicit spectral damping regularizer is an incredibly deep insight. 
- **The Principle of Controllable Regularization:** The proposed **Controllable Spectral Decay (CSD)** is a highly principled solution. It decouples numerical conditioning (keeping the optimization landscape isotropic and well-conditioned) from parameter regularization (damping high-frequency terms), which represents a mathematically superior and far more controllable design pattern than relying on accidental matrix singularity.
- **Deep Optimization Diagnostics:** The empirical validation is exceptionally detailed. Rather than simply presenting final accuracy scores, the authors report design matrix condition numbers across different depths ($L=12$ and $L=32$), optimize under both SGD and Adam, analyze convergence rates, and visualize optimization trajectories and coefficient profiles, providing high-signal insights into the landscape's geometry.
- **Excellent Clarity and Presentation:** The paper is beautifully written, scholarly, and logically organized. The terminology is precise, and the formulas are rigorous. The appendix provides comprehensive implementation workflows, distributed scalability strategies, and thorough hyperparameter listings.

---

## 3. Weaknesses (Including Critical Flaws)

Despite its mathematical elegance, the submission exhibits one critical empirical flaw and two notable theoretical gaps:

### **Critical Flaw 1: Exclusive Reliance on Synthetic Simulation**
The most critical weakness of the paper is that **the entire empirical evaluation is conducted on a synthetic, simulated optimization landscape rather than on physical, pre-trained deep neural networks.**

While the simulator is highly sophisticated—incorporating physically grounded layer-wise sensitivities, non-diagonal covariance matrices to model inter-layer couplings, highly non-convex Rastrigin-type surfaces, and multi-scale transductive noise—it remains a simplified proxy. 
In the deep learning model-merging literature (e.g., Task Arithmetic, AdaMerging, PolyMerge), standard practice is to evaluate on physical models:
1. Loading pre-trained foundation models (such as Vision Transformers like CLIP ViT-B/32 or LLMs like LLaMA-3-8B).
2. Physically performing on-the-fly test-time optimization of merging coefficients using real, unlabeled streaming datasets (e.g., MNIST, CIFAR-10, SVHN, or NLP benchmarks).
3. Evaluating and reporting actual classification accuracies or generation perplexities.

Without evaluating ChebyMerge on physical model checkpoints, it is impossible to guarantee that the proposed mathematical benefits translate directly to real deep learning landscapes. Real-world gradient noise, out-of-distribution shifts, and functional layer interactions are highly complex and may exhibit behaviors not fully captured by the Rastrigin-type simulation. This exclusive reliance on synthetic experiments significantly limits the paper's immediate practical impact.

### **Theoretical Gap 1: Unproven Diagonal Dominance in Theorem 3.2**
The proof of Theorem 3.2 asserts that because Chebyshev polynomials $T_j(x)$ oscillate orthogonally, the off-diagonal elements of the Gram matrix on a uniform grid are "exceptionally small" and the Gram matrix $\mathbf{C}^T \mathbf{C}$ behaves as a "diagonally dominant" system. 

However, the authors provide no mathematical proof of this diagonal dominance condition ($\sum_{j \neq i} |G_{i,j}| < |G_{i,i}|$). On a uniform discrete grid, discrete orthogonality is lost, and diagonal dominance depends heavily on the specific layer count $L$ and degree $d$. For arbitrary $L$ and $d$, the Gram matrix is not guaranteed to be strictly diagonally dominant. The proof should either contain formal bounds on the off-diagonal elements or soften this claim to "approximately diagonal," since diagonal dominance is not formally proven.

### **Theoretical Gap 2: Frequency Distortion and Warping in "DCT Isomorphism" Claim**
The paper claims a spectral interpretation where ChebyMerge behaves as a high-quality discrete low-pass filter isomorphic to the Discrete Cosine Transform (DCT) of Type I. 

However, because ChebyMerge is evaluated on a uniform grid rather than Chebyshev-Gauss-Lobatto (CGL) nodes, the basis functions are $T_j(x_l) = \cos(j \arccos(x_l))$ where $x_l$ is linear in $l$. This is fundamentally different from the DCT-I basis functions $\cos(j \pi l / (L-1))$. The non-linear mapping $\arccos(x_l)$ warps the frequencies significantly. For example, for $j=1$, the Chebyshev basis is linear, while the DCT-I basis is a cosine wave. Consequently, the uniform grid Chebyshev projection does not correspond to a simple frequency truncation in the traditional Fourier sense. The spectral interpretation should be updated to account for this frequency warping.

### **Minor Theoretical Gap 3: Theoretical Contradiction in the Conditioning-Generalization Paradox under Adam**
The "Conditioning-Generalization Paradox" argues that PolyMerge's extreme ill-conditioning acts as an "implicit spectral damping" filter that prevents the optimizer from updating higher-degree spectral coefficients, acting as a regularizer. However, the experiments utilize the **Adam** optimizer. Adam scales the learning rates of individual parameters by the inverse of their root-mean-square gradients. This coordinate-wise rescaling is designed to neutralize ill-conditioning, meaning Adam will scale up updates along the "stiff" directions of the ill-conditioned monomial basis. If Adam neutralizes this damping, the implicit regularization argument should not hold, and PolyMerge should overfit. The authors must reconcile why PolyMerge still exhibits noise filtering and generalizes well when optimized with Adam.

---

## 4. Questions and Suggestions for the Authors
1. **Physical Validation:** Could you provide results for merging physical CLIP ViT-B/32 experts on real classification datasets (e.g., MNIST, FashionMNIST, CIFAR-10, SVHN) to verify if the simulated generalization improvements hold in practice? This would elevate the paper's significance enormously.
2. **Adam vs. Spectral Damping:** Under adaptive optimizers like Adam, coordinate-wise learning rate scaling divides by the gradient magnitude, which should theoretically neutralize the "implicit spectral damping" of monomial bases. Could the authors explain why PolyMerge still generalizes well under Adam? Is it solely due to the low-dimensional subspace restriction rather than the ill-conditioning acting as a regularizer under Adam?
3. **SGD vs. Adam Performance Discrepancy:** In Table 4 of the appendix, the final TTA loss achieved by ChebyMerge ($d=3$) under SGD with momentum at $\eta=10^{-2}$ is $3.8422 \pm 0.4753$, which is significantly lower than the final loss achieved under Adam in Table 3 ($3.9466 \pm 0.5205$). This suggests that SGD with momentum is not only viable but actually superior to Adam when paired with the well-conditioned ChebyMerge. Why did the authors default to Adam in their main experiments if SGD yields a better optimization minimum? This is a very interesting result that deserves discussion.
4. **DCT Isomorphism Frequency Warping:** Can the authors provide a formal spectral analysis of what frequencies are actually filtered by the Chebyshev projection on a uniform grid? The connection to DCT-I is mathematically precise only on CGL nodes, and on a uniform grid, the mapping is not a simple frequency truncation in the standard Fourier sense.

---

## 5. Overall Recommendation

**Rating:** 3: Weak Reject

**Justification:** 
The paper is an exceptional piece of scholarly writing with mathematically beautiful proofs (aside from the diagonal dominance assertion), deep optimization diagnostics, and brilliant conceptual insights (specifically the Conditioning-Generalization Paradox and CSD). However, evaluating a deep learning model-merging technique **entirely on a synthetic simulator without any real-world foundation model validation** falls short of the empirical standard required for top-tier machine learning conferences (such as ICML, NeurIPS, or ICLR). 

If the authors can integrate physical experiments on actual pre-trained Vision Transformers or Large Language Models using real datasets to verify their findings, this paper would easily be a Strong Accept. As it stands, the paper requires these real-world evaluations before it can be published in a machine learning venue.
