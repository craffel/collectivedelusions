# 3. Soundness & Methodology

## Clarity of the Description
The description of the mathematical and systems-level frameworks is exceptionally clear and structured:
* **Mathematical Formulation**: The paper provides a clear and rigorous formulation of parameter-space merging, the feature projection matrix, and layer-wise classical routing projections (using Softmax vs. QWS wave activations).
* **Clear Conceptual Explanations**: Concepts like **Vectorization Collapse**, **Heterogeneity Collapse**, and the **Batch-Average Smoothing Confounder** are defined intuitively and supported with precise mathematical notation.
* **Systems Analysis**: The systems-level complexity and hardware-level memory/latency constraints (Appendix A) are explained with outstanding technical depth, transitioning smoothly from theoretical constraints to physical latency profiles.

## Appropriateness of Methods
* **Controlled Sandbox**: Evaluating on a 192-dimensional synthetic **Analytical Coordinate Sandbox** is highly appropriate. It allows the authors to systematically isolate and analyze core routing mechanics under customizable parameter-space overlap without confounding visual pre-training variables or high-dimensional representation noise.
* **Prior-Driven Classical Routing**: Utilizing a **Zero-Initialized Softmax routing layer** with standard $L_2$ weight decay is an incredibly elegant and appropriate choice. It acts as a maximum-entropy uniform prior, naturally restricting parameters to stay close to the uniform compromise under extreme data scarcity, which perfectly resolves overfitting and Vectorization Collapse.
* **Normalized Random Projections**: Projecting latent features onto a low-dimensional unit sphere is statistically sound, ensuring scale invariance and reducing overfitting risk on tiny calibration splits ($|D_{\text{cal}}| = 64$).
* **Real-World Validation**: Re-evaluating the findings by merging actual convolutional visual experts on MNIST and FashionMNIST images bridges the gap between the synthetic sandbox and real deep neural networks.

## Technical Flaws & Mitigations
The paper has no major technical flaws; the authors have anticipated and thoroughly addressed potential limitations:
1. **Redundancy of $\mathcal{L}_{VR}$**: The authors openly admit and empirically prove that explicit Task-Variance Regularization ($\mathcal{L}_{VR}$) is empirically redundant once the zero-initialized Softmax prior is established. Rather than hiding this, they present it as a key diagnostic finding of their study, emphasizing that simple architectural priors are the true drivers of stability.
2. **Layer-Averaging Simplification**: The authors note that the sandbox averages layer-wise routing weights, bypassing sequential multi-layer dynamics (routing jitter) present in deeper models. To address this, they theoretically propose a **Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$)**, conduct a sensitivity sweep over its coefficient, and empirically validate that it reduces sequential layer-to-layer routing weight jitter by over $57.5\%$ without degrading multi-task classification accuracy.
3. **Systems Bottlenecks**: The authors address the severe memory/latency overheads of full-parameter dynamic assembly by proposing and empirically evaluating **Low-Rank Parameter Assembly (Dynamic LoRA)**. They show that Dynamic LoRA ($r=10$) captures 99% of parameter variance with zero accuracy loss, while matching the latency of static Uniform Merging ($1.01\times$ slowdown) at batch size $512$ where full-parameter assembly slows down by $110.06\times$.

## Reproducibility
The methodology is **highly reproducible**:
* **Explicit Hyperparameters**: All training details, including calibration split size ($64$ samples), number of epochs ($100$), optimizer (Adam), learning rate ($10^{-3}$), weight decay ($\lambda_{wd} \in [10^{-4}, 10^{-1}]$), and variance penalty weight ($\lambda_{var} \in [0.0, 10.0]$) are clearly documented.
* **Statistical Rigor**: The authors execute all experiments across 10 independent random seeds (seeds 42 to 51) to ensure findings are statistically significant and not cherry-picked.
* **Real-World Roadmap**: In Appendix C, the authors outline a concrete, high-integrity experimental protocol to replicate their findings using pre-trained CLIP ViT-B/16 checkpoints on MNIST, FashionMNIST, CIFAR-10, and SVHN, providing a complete roadmap for real-world validation.
