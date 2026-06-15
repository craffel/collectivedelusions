# Evaluation Step 3: Soundness & Methodology

## 1. Description Clarity & Quality of Formulation
The mathematical formulation and description of the methods are **exceptionally clear, precise, and well-structured**:
- **Problem Formulation:** The definition of task vectors ($V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$) and the integration with the Vision Transformer (`vit_tiny_patch16_224`) backbone are mathematically rigorous.
- **BC-Router Framework:** Each variant (BL-Router, GLS-Router, and BSigmoid-Router) is introduced with clean, self-contained equations that explicitly map sample-level representations to dynamic, batch-collapsed merging coefficients.
- **Deconstruction of Design Flaws:** The paper stands out for its high-fidelity mathematical deconstruction. For example, it exposes the subtle structural under-scaling design flaw of Softmax-based bounding (where $\sum_{k=1}^K \alpha_{k} = 0.3$), contrasting it with Uniform Merge (which has an independent scale of $0.3$ per task, totaling $1.2$). This level of detail makes the mathematical arguments highly convincing.

## 2. Appropriateness of Methods
- **Backbone & Expert Convergences:** The use of a capacity-constrained Vision Transformer backbone is a highly appropriate choice. Small backbones exaggerate representation conflicts and weight interference, making them the ultimate "stress test" for dynamic weight routing. Crucially, the authors resolve previous literature flaws by training specialized task-specific experts to true convergence (e.g., SVHN at $96.80\%$), creating a high-fidelity baseline.
- **Few-Shot Calibration Protocol:** Optimizing the 772-parameter routing heads on a tiny 64-sample dataset (16 per task) for 100 Adam steps is standard, realistic, and highly practical. It reflects real-world constraints where large training sets or high-compute resources may not be available for post-merging tuning.
- **Homogeneous vs. Heterogeneous Stream Evaluation:** Evaluating under both static, independent test sets (homogeneous) and shuffled, interleaved temporal streams across various batch sizes ($B \in \{1, 16, 256\}$) is highly appropriate and reflects realistic deployment environments.

## 3. Potential Technical Flaws & Limitations
While the paper is methodologically exceptionally strong, several limitations and areas for improvement exist from a practical standpoint:
- **Scope & Scaling Restrictions:** The empirical verification is entirely conducted on a compact backbone (`vit_tiny_patch16_224` with 5.7M parameters) and four standard vision datasets. While highly controlled, a practitioner would be concerned about scalability: Do these insights (e.g., L2 regularization on routing heads, sigmoidal routing advantages) scale to larger vision backbones (ViT-Base, ViT-Large, Swin), multimodal models (CLIP, LLaVA), or billion-parameter Large Language Models (LLMs) with $K \ge 10$ tasks?
- **Simplified AdaMerging Stream Evaluation:** The authors evaluate AdaMerging statically on the heterogeneous stream using its offline-calibrated joint mean accuracy. While operationally justified to avoid expensive real-time gradient descent during benchmarking, this bypasses the active online optimization loop. In real deployments, online Test-Time Adaptation (TTA) suffers from temporal shift noise, label imbalance, and potential parameter drift, meaning the reported constant performance represents an optimistic upper bound.
- **Unregularized Layer-wise Scaling Amplitudes:** In GLS-Router, standard L2 regularization (weight decay) was applied to the routing weights $W_{route}$ but *not* to the 56 layer-wise scaling parameters $R_k^{(l)}$ itself. This led to severe overfitting and a collapse on FashionMNIST ($64.80\%$). While the authors correctly identify this optimization gap, a more complete analysis would have included a regularized variant where $R_k^{(l)}$ is also penalized or constrained (e.g., using a smaller learning rate or a local weight decay), which could have rescued the GLS-Router baseline.

## 4. Reproducibility
The reproducibility of the paper is **exemplary and exceeds standard conference bars**:
- The authors provide their code, converged expert checkpoints, and complete evaluation scripts at a public repository (`https://github.com/anonymous-researcher/bc-router`).
- The explicit optimization parameters (100 steps of Adam, learning rate of $1 \times 10^{-2}$, weight decay of $1 \times 10^{-4}$ for Reg variants) and calibration set configurations are completely documented.
- Mean and standard deviation are reported across 3 random calibration-sampling seeds, validating statistical robustness.
