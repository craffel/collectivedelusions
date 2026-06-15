# Intermediate Evaluation 4: Experimental Evaluation Check

## 1. Experimental Setup and Datasets
The authors evaluate EHPB and baselines in a **Controlled Representation Sandbox** environment:
* **Backbone:** A pre-trained Vision Transformer ($\mathtt{vit\_tiny\_patch16\_224}$) with $L=14$ layer groups and feature dimension $D=192$.
* **Datasets:** MNIST, FashionMNIST, CIFAR-10, and SVHN.
* **Overlapping Task Configuration:** Rather than mapping tasks into disjoint orthogonal subspaces, the authors establish a dense, overlapping coordinate space configuration across all four tasks. This simulates realistic coordinate conflicts and parameter interference.

## 2. Baselines Evaluated
The paper evaluates a highly comprehensive set of baselines:
1. **Expert Ceiling:** Individual expert models trained on their respective tasks.
2. **Uniform Merging:** A static average of expert weights (representing standard post-hoc model merging).
3. **Linear Router (Global):** Global linear routing of inputs.
4. **vmap-Linear-Router:** A direct, sample-by-sample dynamic additive ensembling baseline implemented using $\mathtt{torch.vmap}$ to avoid a "strawman" comparison.
5. **QWS-Merge:** A SOTA wave-inspired quantum phase-interference model ensembling method.
6. **L3-Routers:** Low-dimensional layer-wise routers (L3-Linear, L3-Softmax, L3-Tanh, both regularized and unregularized) proposed in prior work.

## 3. Analysis of Experimental "Confounders" and Intellectual Honesty
A major strength of the experimental section is the authors' exceptional scientific integrity in identifying and explaining potential confounders in their own setup:

1. **The SVHN Floor Effect Confounder:** The expert ceiling for SVHN is set to an intentionally low 16.8% (only 6.8% above random guessing) by applying a high simulation noise scale ($\sigma = 0.90$). The authors candidly point out that this low-SNR setup introduces a **floor effect**, compressing all ensembling models into a narrow band (9.6% to 14.4% accuracy). They note that this floor effect inadvertently masks the true absolute severity of EHPB's performance collapse compared to cleaner tasks (e.g., CIFAR-10 drops from 81.6% to 12.0%, whereas SVHN drops from 16.8% to 9.6%).
2. **The Synthetic Sandbox Limitation:** The simulated expert weights $V_k$ are generated using independent Gaussian parameters ($\mathcal{N}(0, I_d)$). The authors acknowledge that in real-world fine-tuning, specialized expert weights are highly correlated and reside on low-dimensional manifolds. Generating them as independent Gaussian vectors represents the absolute worst-case scenario. Thus, the toy sandbox acts as a **pessimistic stress-test lower bound**, and real-world weight superposition is likely to experience significantly lower reconstruction noise.

## 4. Whether the Results Support the Claims
The empirical results provide robust, direct support for all the paper's primary claims:

* **Immunity to Heterogeneity Collapse (Claim 3):** The deployment audit ($B=256$) confirms that under mixed-task batching, standard dynamic routers degrade due to batch-averaged routing coefficients, while EHPB and $\mathtt{vmap}$-Linear-Router remain completely immune (Delta = 0.0%) because unbinding is sample-specific.
* **Hadamard Scale-Invariance (Claim 4):** A systematic logarithmic dimension sweep ($D \in [64, 2048]$) confirms that EHPB's relative activation-space weight reconstruction error remains invariant across scales, hovering between 170% and 179%. This validates the **Coordinate Isolation Confounder** theory.
* **Circular Convolution Roadmap (Claim 4):** A low-dimensional proof-of-concept simulation shows that while continuous coordinate-wise reconstruction error is flat at 173%, the cosine similarity of the correct template remains flat at 50% while incorrect template similarity decays as $O(1/\sqrt{D})$, verifying that circular convolution recovers the classic VSA noise decay.
* **Stabilizing Weight Superposition via Residual-EHPB:** Storing the top 5% of critical coordinates uncompressed boosts EHPB's Joint Mean from 28.4% to 33.7%, and MNIST from 61.2% to 75.2%. Furthermore, the **Structured Row-wise Residual-EHPB** simulation shows that row-wise masking incurs only a tiny relative error penalty (+7.77% absolute increase) while bypassing sparse coordinate index lookups, enabling hardware-friendly dense GEMMs.
* **Continuous Cleanup Networks (CCN):** Denoising is highly effective, reducing activation MSE by 8.1$\times$ at Layer 3 (from 0.000859 to 0.000106). This rescues MNIST accuracy from 61.2% to 81.2% (+20.0% absolute improvement). The projection distortion on complex tasks is successfully mitigated by transitioning to a non-linear bottleneck MLP (+0.3% Joint Mean improvement and higher accuracies on FashionMNIST/SVHN).
* **Activation-Space Projection Layers (ASPL):** Orthonormal projections degrade performance below 17.1% Joint Mean, confirming that unsupervised linear projection is limited because signal and noise subspaces are not orthogonal.
* **ReLU Post-Hoc Bias Correction:** Learnable scale/shift correction achieves a **31.4% reduction in final layer representation propagation error** (MSE dropping from 0.3835 to 0.2630) and increases final cosine similarity to 0.9492, confirming the theoretical models in Appendix B.
