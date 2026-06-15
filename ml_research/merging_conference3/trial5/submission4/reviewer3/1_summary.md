# 1. Summary of the Paper

## Main Topic
The paper presents a critical methodological and empirical deconstruction of **Quantum Wavefunction Superposition Merging (QWS-Merge)**, a state-of-the-art dynamic model merging protocol. The authors adopt the principle of Occam's razor to examine whether the highly complex quantum metaphor (modeling parameters as quantum eigenstates and utilizing wave phase-interference equations) is scientifically necessary, or if its reported superior performance is an artifact of under-tuned/unregularized classical baselines. 

## Proposed Approach & Methodology
To systematically isolate confounding variables in dynamic weight routing, the authors introduce the **Bounded Classical Router (BC-Router)** framework. They design three specific classical router variants to serve as controlled baselines:
1. **Bounded Linear Router (BL-Router):** Restricts individual task routing coefficients to a ceiling of $\lambda_{\max} = 0.3$, aiming to control for the "Over-Scaling Confounder" (where unconstrained Softmax routing outputs near $1.0$, pushing merged weights outside the local basin of convergence).
2. **Global Router with Layer-wise Scaling (GLS-Router):** Uses a shared global routing head combined with trainable layer-specific task scaling amplitudes. This controls for the "Layer-wise Specialization Confounder" (where QWS-Merge's layer-specific parameters give it a massive capacity advantage over a simple global linear router).
3. **Bounded Sigmoidal Router (BSigmoid-Router):** Replaces the Softmax activation function with independent, uncoupled Sigmoids ($\alpha_{k} = \lambda_{\max} \times \text{Sigmoid}(o_k)$). This is designed to eliminate the "Softmax Zero-Sum Competitive Bottleneck" where tasks compete for a shared routing budget, often causing the optimizer to sacrifice high-conflict tasks (like SVHN) to minimize joint loss.

The authors build a converged evaluation pipeline using a compact Vision Transformer (`vit_tiny_patch16_224`) backbone. They fine-tune specialized experts to high convergence across four high-conflict vision datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. Trainable models are calibrated on a tiny balanced subset of 64 samples (16 per task).

## Key Findings & Claimed Contributions (with Evidence)
1. **Paradigm Distinction (TTA vs. Offline Calibration):** 
   - *Claim:* Comparing online Test-Time Adaptation (AdaMerging) to offline dynamic routing (QWS-Merge) purely on accuracy conflates two different operational paradigms. 
   - *Evidence:* AdaMerging achieves higher joint accuracy ($89.30\%$), but requires on-the-fly backpropagation/gradient descent during inference, creating massive latency ($495.0$ ms per batch at $B=16$). Offline-calibrated routers (QWS-Merge, BC-Router) are calibrated once on a tiny dataset and run as a pure, lightweight forward pass with zero test-time active optimization or latency overhead ($18.5$ ms per batch).
2. **Deconstructing Classical Failures via L2 Regularization:** 
   - *Claim:* The catastrophic collapse on SVHN of the classical Linear Router baseline ($74.00 \pm 16.14\%$) is an artifact of unregularized overfitting on the tiny calibration set.
   - *Evidence:* Applying standard L2 regularization (weight decay $\gamma = 1 \times 10^{-4}$) to the classical Linear Router completely resolves the SVHN collapse, boosting SVHN accuracy to **$91.73 \pm 3.71\%$** (outperforming QWS-Merge by **$+12.00\%$**) and improving the joint homogeneous accuracy to $82.80 \pm 4.85\%$. 
3. **Over-Scaling vs. Overfitting (Deconstruction of BL-Router):**
   - *Claim:* The failure of the bounded Softmax formulation (unregularized BL-Router SVHN accuracy of $31.73 \pm 16.03\%$) is not caused by the scale ceiling itself, but is an optimization failure combined with a structural under-scaling design flaw of Softmax bounding (which caps the global sum of coefficients at $0.3$, leading to a meager $0.075$ scale per task under uniform uncertainty).
   - *Evidence:* Adding L2 regularization rescues the BL-Router SVHN accuracy to $43.20 \pm 8.02\%$. Furthermore, the unconstrained regularized Linear Router achieves $91.73\%$ on SVHN, proving that global static scale ceiling constraints are redundant and counter-productive once proper regularization is applied.
4. **Resolving the Zero-Sum Bottleneck via Independent Sigmoids:**
   - *Claim:* Replacing Softmax with independent Sigmoids in the BSigmoid-Router completely eliminates the zero-sum competitive bottleneck during mixed-batch calibration.
   - *Evidence:* The BSigmoid-Router achieves a highly stable joint homogeneous accuracy of **$83.73 \pm 1.93\%$** and heterogeneous stream accuracy of **$83.96 \pm 2.27\%$** ($B=1$), matching or outperforming QWS-Merge ($83.97 \pm 0.53\%$ and $83.29 \pm 0.36\%$, respectively).
5. **The Value of QWS-Merge as a Structural Regularizer:**
   - *Claim:* While classical regularized routers can achieve higher peak performance, they exhibit severe task-imbalance and sensitivity to seed variation (e.g., GLS-Router produces a standard deviation of $24.30\%$ on SVHN). QWS-Merge's wave projection equations serve as a stable, structural regularizer that constrains the optimization search space, preventing task-sacrificing behavior and guaranteeing a balanced, Pareto-optimal multi-task profile across seeds (joint standard deviation of only **$0.53\%$**).
   - *Evidence:* GLS-Router and unregularized Linear Routers overfit or exhibit extreme instability under low-data budgets, but can be stabilized by scaling the calibration dataset size (as shown in the 128 and 256 sample ablation study). However, wave projection equations remain a remarkably sample-efficient structural constraint in extremely low-data regimes.
