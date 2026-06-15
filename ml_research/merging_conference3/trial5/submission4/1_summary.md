# Paper Summary: Demystifying Dynamic Model Merging via Bounded Classical Routing (BC-Router)

## 1. Context and Motivation
Parameter-space model merging is an emerging paradigm in deep learning that fuses multiple specialized models (experts) fine-tuned from a shared pre-trained initialization into a single unified network. This approach avoids the high computational costs and data requirements of joint multi-task training. However, standard static merging approaches (like Task Arithmetic, TIES-Merging, and DARE) suffer from representation conflicts: they compress task-specific parameters into a single, input-agnostic weight state, which inevitably degrades performance on high-conflict domains.

To address this, recent work has shifted toward **dynamic model merging**, where parameter-space merging coefficients are routed dynamically on a per-input or per-batch basis. Within this space, **Quantum Wavefunction Superposition Merging (QWS-Merge)** has emerged as a prominent representative. QWS-Merge models specialized experts as quantum eigenstates in a parameter Hilbert space and uses wave phase-interference equations to project representations onto a unit sphere to compute routing weights. The authors of QWS-Merge reported that classical Linear Router baselines catastrophically collapse on high-conflict tasks such as SVHN, claiming that their quantum wavefunction superposition formulation is necessary to avoid representation collapse.

This paper adopts a critical, methodological perspective and applies **Occam's razor** to these claims. The authors argue that complex, over-engineered mathematical metaphors are often over-engineered solutions to simple baseline tuning issues, such as unregularized routing heads overfitting to tiny calibration sets, rather than representing an inherent limitation of classical linear projections.

---

## 2. Proposed Method: Bounded Classical Router (BC-Router)
To systematically isolate, control, and analyze the true drivers of dynamic model-merging performance, the paper proposes the **Bounded Classical Router (BC-Router)** framework. BC-Router introduces three training-free, parameter-efficient classical variants optimized on a tiny, 64-sample offline calibration set:
1.  **Bounded Linear Router (BL-Router):** Restricts task-vector coefficients to a maximum scale of $\lambda_{max} = 0.3$ to isolate and control the **Over-Scaling Confounder**.
2.  **Global Router with Layer-wise Scaling (GLS-Router):** Combines a shared global linear routing head with trainable, layer-specific task-scaling amplitudes ($R_k^{(l)}$ initialized to 0.3) to control the **Layer-wise Specialization Confounder**.
3.  **Bounded Sigmoidal Router (BSigmoid-Router):** Replaces standard Softmax routing with independent, Softmax-free Sigmoid activations to eliminate the **Zero-Sum Competitive Bottleneck** during mixed-batch multi-task calibration.

The paper also investigates the role of **L2 regularization** (weight decay $\gamma = 1\times 10^{-4}$) during calibration to prevent routing head overfitting in extremely low-data calibration regimes.

---

## 3. Key Findings & Empirical Discoveries
Evaluating these methods on a compact Vision Transformer backbone (`vit_tiny_patch16_224`) fine-tuned to true convergence across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) yields several highly significant empirical discoveries:
*   **Deconstructing Classical Failures via Proper Regularization:** The classical Linear Router's reported SVHN collapse ($74.00 \pm 16.14\%$) is entirely a low-data overfitting artifact. Applying standard L2 regularization during calibration completely resolves this collapse, boosting SVHN accuracy to **$91.73 \pm 3.71\%$** (outperforming QWS-Merge by **$+12.00\%$**), proving that previous failures were purely a baseline tuning artifact.
*   **Redundancy of Scale-Ceiling Constraints:** With proper regularization, the unbounded **Linear Router (Reg)** significantly outperforms the scale-capped **BL-Router (Reg)** on SVHN ($91.73\%$ vs. $43.20\%$), indicating that explicit scale-bounding constraints are counter-productive and redundant once overfitting is controlled.
*   **QWS-Merge as a Structural Regularizer:** The unregularized layer-wise **GLS-Router** exhibits severe overfitting, collapsing on FashionMNIST ($64.80 \pm 3.53\%$) and showing extreme sensitivity to calibration seeds (SVHN standard deviation of $24.30\%$). This reveals a profound scientific insight: QWS-Merge's complex wave projection equations serve as a highly robust **structural regularizer** that constrains the optimization search space, preventing task-sacrificing behavior under tight calibration budgets.
*   **Resolving the Zero-Sum Bottleneck:** Standard Softmax routing forces tasks to compete for a shared routing budget, leading the optimizer to sacrifice hard, high-conflict domains like SVHN during mixed-batch calibration. The proposed Softmax-free **BSigmoid-Router** completely resolves this, achieving a highly stable **$83.73 \pm 1.93\%$** joint homogeneous and **$83.96 \pm 2.27\%$** heterogeneous stream accuracy ($B=1$, outperforming QWS-Merge's $83.29 \pm 0.36\%$).
*   **The Batch-Averaging Bottleneck:** Under temporal stream noise, sample-level batch averaging collapses dynamic routing at larger batch sizes ($B=256$), causing all dynamic methods to converge to static Uniform Merge performance.
