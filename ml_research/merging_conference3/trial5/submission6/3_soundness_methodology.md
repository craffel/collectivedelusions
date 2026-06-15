# 3. Soundness and Methodology Check

## 3.1. Theoretical and Mathematical Soundness
The mathematical formulation of SLD-Merge is highly sound, rigorous, and logically consistent:
- **Offline SVD Task-Vector Factorization (Eq 1 to 4):** Factorizing the dense parameter shift vector $V_k^{(l)}$ using Singular Value Decomposition (SVD) and dividing the singular values equally as $\sqrt{\Sigma_k}$ onto both low-rank matrices $B_k^{(l)}$ and $A_k^{(l)}$ is a mathematically sound and highly stable method. It ensures that the norm of the adapter matrices remains balanced, preventing exploding or vanishing activations during the forward pass.
- **Bounded Cosine-Similarity Router (Eq 5 to 6):** Spatial average pooling of token representations to get $z(x)_b$ is a standard, robust practice in Vision Transformers. Computing cosine-similarity instead of an unconstrained linear projection acts as a bounded, spherical projection, restricting routing scores to $[-1, 1]$. This suppresses high-frequency activation noise and acts as a strong regularizer that prevents representation collapse under domain shifts.
- **Top-1 Gating and Forward Pass (Eq 7 to 9):** Applying hard Top-1 expert selection (represented by a sparse one-hot coefficient $\alpha$) completely isolates the selected expert adapter, bypassing task interference.
- **Vectorized Parallel Formulation (Eq 10):** The vectorized batch equation $Y = X W_{base}^{(l)} + \sum_{k=1}^K \alpha_k \odot ( (X A_k^{(l)}) B_k^{(l)} )$ is mathematically equivalent to processing each sample in complete isolation, which guarantees complete batch-independence and zero cross-sample leakage.
- **Activation-Space Mean Initialization (Eq 11):** Setting the routing basis vectors $\Phi_k^{(l)}$ to the empirical mean activations of task $k$ over a small unlabeled calibration split is mathematically clean and places the routing basis at the representative feature centroid of each task's activation space.
- **Autonomous Head Selection (Eq 12 to 13):** Dynamic selection of classification heads based on the layer-averaged routing score across the specialized late layers $\{9, 10, 11\}$ is mathematically rigorous and ensures stateless, oracle-free deployment.

## 3.2. Code Implementation Soundness
The code in `run_experiments.py` is exceptionally well-structured and faithfully implements the mathematical framework:
- **Model Surgery:** Correctly replaces standard PyTorch linear layers in blocks 9, 10, and 11 of a pre-trained `vit_tiny_patch16_224` backbone with custom `MergedLinear` modules.
- **Vectorized Dynamic Forward Pass:** Intercepts activations and routes them dynamically through the parallel low-rank SVD adapters.
- **Differentiable Straight-Through Estimator (STE):** Resolves a critical autograd bug in the optimized router baseline. Because hard Top-1 gating (`argmax` and `one_hot`) is non-differentiable, the code implements a mathematically rigorous STE:
  `coefficients = hard_coefficients + (soft_coefficients - soft_coefficients.detach())`
  And it scales the parallel low-rank adapter outputs by the exact routing coefficients at the active expert index:
  `delta_W_out = delta_W_out * coeff_scale`
  This allows gradients to flow back through the active low-rank paths to the routing basis parameters `self.sld_basis` during gradient descent, validating both the zero-shot warm-start and the backpropagation optimization path.
- **Oracle-Free Autonomous Evaluation:** The code implements and evaluates the autonomous classification head selection rule (`use_autonomous_head=True`), resolving previous oracle leakage concerns.

## 3.3. Methodological Limitations & Defense Strategies
The authors proactively identify, discuss, and address potential methodological limitations in both the main text and expanded appendices:
1. **Extreme Subsampling of Datasets:** The training/evaluation subsets are limited to 256 training, 128 calibration, and 256 test samples per dataset, resulting in low standalone expert ceilings (such as SVHN with 29.30%). The authors defend this as modeling a highly challenging "low-shot streaming stress-test" representing edge deployment and test-time cold-starts. Additionally, they provide a mathematical analysis in Appendix F showing that with fully converged experts, SVD truncation is expected to be *even more effective* since specialized representations saturate and stabilize, concentrating singular values in the top dimensions and further reducing reconstruction error.
2. **Hand-Coded Task Specialization (Blocks 9--11):** Restricted to 3 layers of a 12-block network. The authors address full-network merging in Appendix F, showing that applying SVD dynamic merging to all 12 blocks scales parameter overhead linearly but still achieves a massive **91.1%** task-specific parameter savings over duplicating the full 12-block network, while freezing early layers is a strategic choice to preserve consistent routing.
3. **Domain Classification Task Selection:** MNIST, FashionMNIST, CIFAR-10, and SVHN are highly distinct domains, making domain classification easy (93.26% accuracy). To scale to fine-grained or overlapping domains, the authors propose three concrete strategies in Appendix F: Hierarchical Routing, Task-Vector Clustering, and Shared Basis Projection.
