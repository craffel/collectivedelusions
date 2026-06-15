# Intermediate Evaluation 3: Soundness and Methodology

## Clarity of the Description
The paper's overall description is clear and logical. The proposed algorithm, **Sparse Task Arithmetic (STA)**, is simple and explicitly formulated in Section 3.1. The complete PyTorch implementation provided in Appendix A is self-contained and easy to follow. The writing style is professional, and the transition from introducing the problem of model merging to criticizing existing complex heuristics is well-structured.

---

## Technical and Mathematical Soundness: Detailed Critique
As a reviewer focused on rigorous theoretical grounding and mathematical correctness, I have identified several critical flaws, hand-waving assumptions, and theoretical gaps in the methodology.

### 1. The Expected Energy Equation Flaw (Section 3.1)
The authors attempt to justify the under-scaling confounder by modeling the expected squared $L_2$ norm of the sparse task vector. They write:
$$\mathbb{E}[\|v^{\text{sparse}}_{k, l}\|_2^2] \approx \frac{s}{100} \mathbb{E}[\|v_{k, l}\|_2^2]$$

**This equation is mathematically false for magnitude-based pruning.**

*   **The Error:** This linear scaling relation is only exact for *random* pruning (such as in DARE), where each coordinate is kept with a uniform probability $s/100$ independent of its magnitude.
*   **The Mathematical Reality of Magnitude-based Pruning:** In magnitude-based pruning, the binary mask $M_{k,l}$ is deterministic and depends directly on the absolute value of $v_{k,l}$. By keeping only the top-$s$\% largest coordinates by absolute magnitude, we are selectively preserving the extreme tails of the parameter update distribution. Because these are the *largest* elements, they contribute disproportionately to the total $L_2$ norm (or energy).
*   **Analytical Verification:**
    Let the elements of the task vector $v$ be distributed as a Gaussian variable, $v_i \sim \mathcal{N}(0, \sigma^2)$, which is a standard assumption in weight analysis.
    If we keep the top $s = 20\%$ largest elements, the pruning threshold $\tau$ corresponds to the $0.8$-quantile of the folded normal distribution, giving $\tau \approx 1.28 \sigma$.
    The expected squared value of the pruned elements is given by:
    $$\mathbb{E}[v_i^2 \mathbb{I}(|v_i| \ge \tau)] = 2 \int_{\tau}^{\infty} x^2 \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}} dx$$
    Using integration by parts, this evaluates to:
    $$\sigma^2 \left( 2 \Phi(-\tau/\sigma) + \frac{2\tau}{\sqrt{2\pi}\sigma} e^{-\frac{\tau^2}{2\sigma^2}} \right)$$
    Plugging in $\tau \approx 1.28 \sigma$ and $\Phi(-1.28) \approx 0.1$, we get:
    $$\mathbb{E}[v_i^2 \mathbb{I}(|v_i| \ge \tau)] \approx \sigma^2 \left( 2(0.1) + \frac{2(1.28)}{\sqrt{2\pi}} e^{-0.82} \right) \approx \sigma^2 \left( 0.2 + 0.45 \right) = 0.65\sigma^2$$
    Thus, for a standard normal distribution, keeping $s=20\%$ of the weights retains **$65\%$** of the total energy, not $20\%$!
    For heavy-tailed distributions (like Laplace, Student's $t$, or Cauchy) which are more representative of deep learning updates, the top 20% of weights can retain **$80\%$ to $90\%$** of the total energy.
*   **Impact on the Paper's Claims:** 
    This mathematical error invalidates the theoretical derivation of the Rescaled STA (R-STA) scaling factor. The authors scale R-STA by multiplying by $100/s$ (which is $5.0$ at $s=20\%$). Since the sparse vector already retains $65\%+$ of the original energy, multiplying by $5.0$ (energy scaling of $25.0$) results in massive over-scaling and "parameter explosion," which the authors observe in Section 4.3 but incorrectly attribute to a "fundamental variance-distortion phenomenon of magnitude pruning." In reality, the failure of R-STA at low densities is a direct consequence of this incorrect scaling equation.

---

### 2. Unrealistic Independence Assumption in Collision Probability (Section 3.2.1)
The authors assume that the pruning masks $M_a$ and $M_b$ for different tasks are completely independent, asserting that the probability of coordinate collision is $(s/100)^2$.

*   **Critique:** This assumption is highly unrealistic. Different task experts are fine-tuned from the *same* pre-trained base model $\theta_0$. It is well-established that:
    1. Fine-tuning updates are highly concentrated in specific layers (such as late layers, output projections, or key/query/value self-attention projections) rather than being uniformly distributed across the model.
    2. Certain sensitive parameters or generalizable features are updated across almost all fine-tuning tasks.
*   **The Mask Overlap Verification:** 
    The authors claim that the empirical overlap rate of $3.1\%$ to $4.3\%$ at $s=20\%$ matches the $4\%$ independence bound. However, they report this as a global average. 
    If different tasks have disjoint classification heads, their overlap is $0\%$, which artificially pulls the global average overlap down. 
    To be theoretically sound, the authors must report the mask overlap layer-by-layer—specifically for shared backbone layers (e.g., attention blocks and MLP blocks)—where parameter collision actually occurs and can cause degradation.

---

### 3. Vague and Metaphorical "Noise-Filtering" Model (Section 3.2.3)
The authors attempt to ground magnitude-based pruning as a symmetric noise filter using the decomposition:
$$v_k = v_k^{\text{salient}} + \epsilon_k$$

This model is presented in a highly hand-wavy and metaphoric manner, lacking formal mathematical definitions and proofs:
1. **Definition of Noise ($\epsilon_k$):** What is the formal mathematical definition of "high-frequency parameter noise" in a discrete, non-spatial parameter space? "High-frequency" is a spectral signal processing term. Its application here is purely metaphorical.
2. **Separation Guarantee:** Why is the noise strictly low-magnitude, while the salient features are high-magnitude? The authors provide no theoretical proof or citation to support the assertion that SGD noise is concentrated in low-magnitude coordinates.
3. **Weight-Space Drift:** The authors claim that $\sum_k \lambda_k \epsilon_k$ "creates a significant drift in weight-space direction" and pushes parameters "off the low-loss manifold," but they provide no mathematical model of the loss landscape (e.g., Hessian analysis or local curvature) to prove why this drift occurs or how magnitude pruning prevents it.

---

### 4. Problematic Assertion of "Self-Resolving" Sign Conflicts (Section 3.2.2)
The authors claim that when opposite sign updates cancel out ($[v_a]_j > 0$ and $[v_b]_j < 0$), "this cancellation is mathematically sound: it indicates that the tasks require conflicting adjustments to the shared representation, and the network should maintain a neutral state."

*   **Critique:** From a functional perspective, this is a major conceptual flaw. If task A requires a parameter to be positive ($+1.0$) to activate a feature, and task B requires it to be negative ($-1.0$) to repress a feature, setting the parameter to $0.0$ satisfies *neither* task. Both tasks A and B will experience performance degradation at this coordinate.
*   Calling this "mathematically sound" is a cop-out that ignores the functional reality of destructive interference. The only reason STA succeeds is because such collisions are empirically rare, not because cancellation is "functionally sound."

---

## Reproducibility
The reproducibility of the empirical results is **high**:
1. The authors use a standard ViT-B-32 backbone and standard datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) that are widely available.
2. The PyTorch code in Appendix A provides the exact implementation details, including the use of PyTorch's `torch.quantile` to construct the layer-wise binary masks.
3. The hyperparameters (scaling coefficient $\lambda \in [0.1, 1.0]$, survival density $s \in \{5\%, 10\%, 20\%, 50\%\}$) are fully documented.
4. However, the authors use a subset of $2{,}048$ validation samples per dataset to report accuracy. They do not report standard deviations or run multiple seeds. For a rigorous scientific review, the authors should report error bars to confirm that the marginal difference between Tuned STA (90.53%) and Tuned TIES-Merging (90.16%) is statistically robust.
