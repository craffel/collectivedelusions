# Idea Proposal: The "No-Data" Strawman: Demystifying Test-Time Adaptation vs. Offline Few-Shot Validation Tuning

## 1. Persona Alignment
This proposal is directly aligned with the core philosophy of **The Methodologist**. The current literature on test-time model merging (e.g., AdaMerging, RegCalMerge, PolyMerge, Q-Merge) claims state-of-the-art multi-task performance by running online test-time adaptation (TTA) with unsupervised objectives (like entropy minimization) on unlabeled test streams. However, as a skeptic of flashy but fragile SOTA claims, we identify two severe methodological flaws in this paradigm:
1. **The "No-Data" Strawman:** These papers compare their highly complex, online backpropagation-dependent adaptation against a completely unoptimized uniform baseline, creating a false dichotomy of "either zero-shot uniform merging or online TTA." In real-world deployment, practitioners almost always have access to a tiny labeled validation set (e.g., 5 to 50 samples per task).
2. **Hidden Assumptions of Stream Stability:** Online TTA relies on the assumption of a stable, balanced, i.e. i.i.d. unlabeled stream. We hypothesize that under realistic deployment scenarios (e.g., label shift, highly imbalanced or bursty task streams, and batch sizes of 1), online entropy minimization experiences severe representation collapse and transductive overfitting, destroying the model's multi-task capabilities.

To expose these issues, we propose **Offline Few-Shot Validation Tuning (OFS-Tune)**. By optimizing a highly constrained, low-dimensional merging search space (task-wise global scalars or low-degree polynomials) offline on a tiny validation set, we completely eliminate the need for test-time adaptation. This baseline requires zero test-time compute (no backpropagation or forward-pass optimization on streams), is completely immune to label/task shift, and we hypothesize it matches or outperforms SOTA TTA across standard benchmarks.

---

## 2. Core Techniques
We introduce and evaluate the following core techniques and search space configurations:
- **Global Task-wise Coefficients (GT-Merge):** Instead of optimizing layer-wise coefficients (which are highly prone to transductive overfitting and optimization noise), we constrain the search space to a single scalar merging weight $\alpha_k \in \mathbb{R}$ per task $k$, constant across all layers. The search dimensionality is exactly $K$ (number of tasks).
- **Polynomial Coefficient Profiles (Poly-Val-Merge):** Following the parameterization of PolyMerge, we model the coefficient of task $k$ at layer $l$ as a low-degree polynomial of normalized depth: $\alpha_k(l) = \sum_{j=0}^d c_{kj} (\frac{l}{L})^j$. For a degree $d$ polynomial, this requires $(d+1)$ parameters per task.
- **Offline Black-Box Optimization:** We perform optimization of the search parameters offline using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) on a tiny labeled validation set $D_{val}$ containing $M$ samples per task (where $M \in \{5, 10, 20, 50\}$).
- **Robustness Stress-Testing Protocol:** We systematically evaluate all methods under three adversarial stream conditions:
  1. *Extreme Label Shift:* The test stream has severe class imbalance (e.g., Dirichlet distribution $\text{Dir}(\alpha)$ where $\alpha=0.1$).
  2. *Bursty Task Streams (Temporal Shift):* Test samples arrive grouped by task rather than shuffled, violating the i.i.d. assumption.
  3. *Ultra-small Batch Sizes:* Streams arrive with a batch size of 1 or 2, where online batch normalization and entropy statistics fail.

---

## 3. Mathematical Formulation
Let $W_{base}$ be the pre-trained base model weights, and $W_k$ be the fine-tuned expert weights for task $k \in \{1, ..., K\}$. The task vector for task $k$ is:
$$V_k = W_k - W_{base}$$

The merged model weights $W_{merged}^{(l)}$ at layer $l \in \{1, ..., L\}$ are defined as:
$$W_{merged}^{(l)} = W_{base}^{(l)} + \sum_{k=1}^K \alpha_k(l) V_k^{(l)}$$

We propose two search parameterizations $\theta$:
1. **GT-Merge:** $\alpha_k(l) = \alpha_k$ for all $l \in \{1, ..., L\}$. The parameters are $\theta = \{\alpha_k\}_{k=1}^K \in \mathbb{R}^K$.
2. **Poly-Val-Merge:** $\alpha_k(l) = \sum_{j=0}^d c_{kj} \left(\frac{l}{L}\right)^j$. The parameters are $\theta = \{c_{kj}\}_{k=1, j=0}^{K, d} \in \mathbb{R}^{K \times (d+1)}$.

Let $f(x; W_{merged}(\theta))$ be the prediction function of the merged model. Given a tiny labeled validation set $D_{val} = \bigcup_{k=1}^K D_{val}^k$ where each $D_{val}^k$ has $M$ labeled examples, our objective is to find the parameters $\theta^*$ that minimize the cross-entropy loss over $D_{val}$:
$$\theta^* = \arg\min_{\theta} \mathcal{L}_{val}(\theta) = \frac{1}{K} \sum_{k=1}^K \frac{1}{|D_{val}^k|} \sum_{(x, y) \in D_{val}^k} \mathcal{L}_{CE}\left(f\left(x; W_{merged}(\theta)\right), y\right)$$

Because the validation loss landscape is highly non-convex due to the deep representations of $f$, but the parameter dimension of $\theta$ is extremely low (e.g., $K=4$ for GT-Merge, or $K(d+1)=12$ for Poly-Val-Merge with $d=2$), we optimize $\mathcal{L}_{val}(\theta)$ using **CMA-ES**.

---

## 4. Architecture Specifications
- **Backbone Models:** Vision Transformer (ViT-Tiny and ViT-B/32), as well as ResNet-18, fine-tuned on individual tasks from a shared pre-trained base.
- **Tasks ($K=4$):** Image classification on four visual domains: MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Input Representations:** Images $x$ resized and normalized to $224 \times 224 \times 3$ for ViT models, or $32 \times 32 \times 3$ for ResNet-18.
- **Task Classifiers:** Linear classification heads $H_k$ for each task $k$ appended to the backbone. During evaluation, the model routes the input to the correct task head or performs a joint multi-task prediction.

---

## 5. Baselines
We evaluate our proposed offline-tuned baselines against a comprehensive suite of competitive baselines:
1. **Uniform Task Arithmetic (Uniform TA):** Naive weight merging with uniform coefficients ($\alpha_k = 1/K$ or $\alpha_k = 0.3$).
2. **AdaMerging (TTA-Adam):** SOTA online test-time adaptation optimizing layer-wise coefficients $\alpha_{kl}$ via entropy minimization on the test stream using Adam.
3. **PolyMerge (TTA-Adam / TTA-ES):** SOTA online TTA parameterizing coefficients as polynomials and optimizing them via entropy minimization.
4. **RegCalMerge (TTA-Adam):** SOTA online TTA with Class-Capacity Normalization and Elastic Spatial Regularization.
5. **Offline Oracle (Full-Data Search):** Upper bound representing the best possible coefficients found by searching over the entire, labeled training/test sets.

---

## 6. Step-by-Step Interaction
The lifecycle of our proposed **OFS-Tune** is divided into two phases:

### Phase A: Offline Few-Shot Validation Tuning (Pre-Deployment)
1. **Validation Sampling:** From each task $k \in \{1, ..., K\}$, sample $M$ labeled examples (e.g., $M=10$) to form the few-shot validation set $D_{val}^k$.
2. **Optimizer Initialization:** Initialize the CMA-ES search with mean $\mu$ set to uniform weights (e.g., $0.3$) and step size $\sigma = 0.1$.
3. **Black-Box Search Loop (for $T$ generations):**
   - Sample a population of candidate parameter vectors $\{\theta_p\}_{p=1}^P$ from the current Gaussian distribution.
   - For each candidate $\theta_p$:
     - Construct the merged model weights $W_{merged}(\theta_p)$.
     - Perform a forward pass on the validation set $D_{val}$ to compute the multi-task cross-entropy loss $\mathcal{L}_{val}(\theta_p)$.
   - Update the CMA-ES mean and covariance matrix based on the evaluated losses.
4. **Finalization:** Output the optimal parameter vector $\theta^*$. Construct the static merged model $W_{merged}(\theta^*)$.

### Phase B: Deployment and Inference (Online)
1. **Zero-Compute Deployment:** Deploy the static, merged model $W_{merged}(\theta^*)$ directly to the target environment.
2. **Robust Inference:** 
   - A stream of unlabeled test data $x_t$ arrives.
   - The model directly computes the prediction $y_t = f(x_t; W_{merged}(\theta^*))$ in a single forward pass.
   - **Zero backpropagation steps, zero auxiliary forward passes, and zero coefficient updates are performed at test time.**
   - The system is completely unaffected by whether the stream is i.i.d., bursty, small-batch, or label-shifted, guaranteeing predictable and robust latency and classification performance.
