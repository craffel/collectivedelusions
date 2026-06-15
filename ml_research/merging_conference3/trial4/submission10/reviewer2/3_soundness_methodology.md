# 3. Soundness and Methodology Evaluation

## Clarity of the Technical Description
The mathematical formulation of QWS-Merge (Equations 1-9) is presented clearly and structured logically. However, several critical details are omitted or require further clarification for a complete understanding and reproducibility:

1. **Random Projection Matrix $P$**:
   - The paper states that $P \in \mathbb{R}^{D \times d}$ is a "frozen random projection matrix."
   - *Missing Details*: How is $P$ initialized (e.g., standard normal, uniform, orthogonal)? Since it is frozen, does the specific random seed used to initialize $P$ significantly impact the downstream performance of the router? A sensitivity analysis on the initialization of $P$ is missing.
2. **Task-Specific Classification Heads**:
   - In a multi-task setting with four distinct visual domains (MNIST, FashionMNIST, CIFAR-10, SVHN), the classification heads must be task-specific due to different label spaces or semantics.
   - *Missing Details*: How are these heads handled at inference time? Is there an oracle routing mechanism that selects the appropriate head based on the ground-truth task label? If so, this must be explicitly stated. If the head selection is done based on the dynamic router's output, how does it handle mixed-task batches?
3. **Calibration Optimization Details**:
   - The authors mention that QWS-Merge is optimized using standard Adam on a tiny validation set of 64 samples for 100 steps.
   - *Missing Details*: What learning rate, weight decay, or beta parameters were used for the Adam optimizer during this calibration phase?

---

## Appropriateness of the Methodology
From an **Empiricist** perspective, there are several methodological decisions that are highly questionable:

1. **The Purpose of Dynamic Routing vs. Homogeneous Evaluation**:
   - In Table 1, the authors evaluate the model on "homogeneous test streams" (where a batch contains samples from only a single task) and show that QWS-Merge outperforms static methods.
   - *Methodological Critique*: In a homogeneous test stream, the task identity of the batch is implicitly known (e.g., we know the entire batch is SVHN). If the task is known, we do not need a dynamic router; we can simply route the entire batch directly to the specialized SVHN expert and achieve the expert ceiling ($34.50\%$). 
   - The only scenario where dynamic merging is actually valuable is when the input stream contains a **mix of tasks (heterogeneous stream)** and we do not have a prior task label.
2. **Heterogeneous Performance and "Heterogeneity Collapse"**:
   - Table 2 evaluates the models on a heterogeneous (mixed) test stream across batch sizes $B \in \{1, 16, 256\}$.
   - At $B=1$ (single-sample online inference), QWS-Merge achieves $54.90\%$ accuracy, which is **worse than static AdaMerging ($57.20\%$) and OFS-Tune ($55.60\%$)**.
   - At $B=16$, QWS-Merge drops to $48.80\%$, and at $B=256$, it drops to $48.70\%$ (both performing below uniform merging at $49.20\%$).
   - *Methodological Critique*: This is a severe practical flaw. In the only setting where dynamic routing is practically useful (mixed-task streams where task identity is unknown), **QWS-Merge actually performs worse than simple static merging methods** (such as AdaMerging) and collapses back to the performance of uniform averaging. The authors honestly document this "heterogeneity collapse," but the methodology fails to provide a solution that is practically superior to existing static merging methods in realistic deployment scenarios.

---

## Potential Technical Flaws and Weaknesses

1. **Suboptimal Expert Performance (Especially on SVHN)**:
   - The individual expert ceiling on SVHN is reported as **$34.50\%$**. 
   - *Technical Critique*: For a 10-class dataset of street view house numbers, $34.50\%$ is extremely low. Standard Vision Transformers (even tiny ones) typically achieve over $90\%$ accuracy on SVHN. A performance of $34.50\%$ suggests that the SVHN expert is highly under-trained, has converged to a poor local minimum, or was trained with highly suboptimal hyperparameters. If the expert model itself is fundamentally flawed, any merging results involving it (including QWS-Merge's $31.60\%$ and the Linear Router's $15.30\%$) are highly suspect and may not represent the true behavior of merged convergent networks.
2. **The "Strawman" Linear Router Baseline**:
   - The classical Linear Router is presented as a baseline that collapses to $15.30\%$ on SVHN. The authors claim this proves the necessity of QWS-Merge's "wave-like subspace regularization."
   - *Technical Critique*: The Linear Router contains 772 parameters and is optimized on a tiny set of 64 samples for 100 steps. Without any regularization, a linear projection is guaranteed to overfit catastrophically in this data-scarce regime. Did the authors attempt to apply standard, classical regularization techniques (e.g., L2 regularization/weight decay, dropout, or spectral normalization) to the Linear Router? If a regularized Linear Router can achieve similar or better performance than QWS-Merge on SVHN, then the entire "quantum wave-like subspace regularization" hypothesis is weakened, as classical regularization would suffice.
3. **Violation of the I.I.D. Assumption**:
   - The wavefunction collapse step (averaging sample-level coefficients across the batch dimension) introduces a **batch dependency** during inference. 
   - *Technical Critique*: This means that the model's prediction for a specific image depends on the other images present in the same batch. This directly violates the Independent and Identically Distributed (I.I.D.) assumption of statistical machine learning. In real-world applications, this makes the system highly unpredictable and difficult to verify, as an identical input could produce different classifications depending on its batch context.

---

## Reproducibility
The reproducibility of the work is **fair**. While the paper describes the core mathematical steps and lists the number of trainable parameters, it lacks key hyperparameter values (e.g., Adam learning rate for calibration, random projection initialization scheme, head selection mechanism) and does not provide public source code or a repository link. To meet the standard of reproducibility, the authors must release their code and provide a complete hyperparameter table.
