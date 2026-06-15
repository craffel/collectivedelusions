# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The mathematical and structural descriptions in the paper are exceptionally clear, logical, and easy to follow. 
- **Section 3 (Methodology)** clearly details the backbone setup (compact Vision Transformer with 14 layer groups, $D=192$, and spatial averaging of patch tokens to extract a representation $z(x)_b \in \mathbb{R}^D$).
- It defines the unregularized Linear Router baseline with clear equations (projecting representation to logits, mapped via Softmax to batch-averaged merging coefficients $\bar{\alpha}_k$).
- The math of the proposed variants (**BL-Router**, **GLS-Router**, **BSigmoid-Router**) is cleanly stated.
- Crucially, the authors provide a rigorous mathematical deconstruction of the **structural under-scaling flaw of Softmax bounding**, which is very insightful and clearly explains the hidden mathematical bottleneck of previous bounded classical formulations.

## Appropriateness of Methods
The scientific method of isolating and controlling individual confounding variables is highly appropriate and rigorous. 
- **Variable Isolation:** Instead of merely proposing another complex method, the authors systematically design ablation models to control for over-scaling (BL-Router), layer-wise specialization capacity (GLS-Router), and competitive Softmax (BSigmoid-Router).
- **CONVERGED Experts:** A major strength is ensuring that the expert checkpoints are trained to *true convergence* (e.g., $100\%$ MNIST, $92.8\%$ FashionMNIST, $96.4\%$ CIFAR-10, $96.8\%$ SVHN) before performing merging. This resolves previous experimental flaws where under-tuned or unconverged experts distorted merging behaviors.
- **Evaluation Protocols:** Testing under both Homogeneous (multi-task independent test sets) and Heterogeneous (fully shuffled interleaved streaming data across batch sizes $B \in \{1, 16, 256\}$) scenarios is highly appropriate. This captures both static multi-task capacity and dynamic adaptation under stream noise.
- **Ablations and Sensitivity Analyses:** The paper goes far beyond standard empirical work by including:
  1. A sensitivity analysis of the L2 regularization strength ($\gamma \in [0, 10^{-5}, 10^{-4}, 10^{-3}]$).
  2. A data-scaling ablation study (increasing calibration set size from 64 to 128 and 256 samples).
  3. Physical latency benchmarks comparing feed-forward routing to test-time adaptation (AdaMerging).

## Potential Technical Flaws, Limitations, and Nuances
1. **Scope of the Empirical Sandbox:** The experimental setup is restricted to a compact Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) on four image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). While this is mathematically ideal for isolating and deconstructing weight routing mechanisms, it is a relatively small scale. Whether these insights (L2 regularization, Softmax-free Sigmoid routing) generalize to massive LLMs (e.g., LLaMA-1B/3B) or larger vision models remains an open question. The authors honestly acknowledge this limitation in their conclusion.
2. **The Generalist-Specialist Paradox / Underperformance vs. Static Averaging:** The authors make a highly transparent and honest admission: *none* of the trainable dynamic classical routing variants outperform a simple, non-trainable **Uniform Merge (Task Arithmetic)** in overall homogeneous joint mean accuracy ($85.10\%$). The authors explain that weight merging is a zero-sum game of parameter capacity: dynamically steering weights toward one task pulls them away from others, degrading overall average performance. This represents a fundamental conceptual limitation of the entire sub-field of dynamic weight-space routing (including QWS-Merge). While this is a scientific limitation of the field, the authors' transparency in highlighting this "practical utility paradox" is a major strength.
3. **AdaMerging Stream Evaluation Simplification:** On heterogeneous streams, AdaMerging (TTA) is modeled statically using its offline-calibrated joint mean accuracy to avoid prohibitive real-time gradient optimization latency. The authors openly admit that this is an optimistic upper bound that fails to capture real-world online temporal dynamics (parameter drift, label shift, gradient noise on the stream). This is a reasonable and practical simplification, but represents a minor limitation of the stream evaluation.
4. **GLS-Router Overfitting detail:** The unregularized GLS-Router collapses on FashionMNIST ($64.80 \pm 3.53\%$) because its 56 layer-wise scaling parameters $R_k^{(l)}$ are unregularized during calibration on the 64-sample set. The authors identify this optimization gap and recommend applying regularization directly to layer-wise amplitudes in future designs, which is a constructive methodological refinement.

## Reproducibility
The reproducibility of the submission is **excellent**:
- Detailed hyperparameter configurations are provided for both the expert fine-tuning (AdamW, 15 epochs, dual learning rate schedule) and the calibration optimization (Adam, 100 steps, LR $1\times 10^{-2}$, weight decay $\gamma = 1 \times 10^{-4}$).
- The authors provide a public GitHub URL in a footnote containing code, expert checkpoints, and evaluation scripts, ensuring complete reproducibility of the reported figures and tables.
