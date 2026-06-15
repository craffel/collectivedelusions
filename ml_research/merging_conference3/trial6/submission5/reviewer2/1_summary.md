# Comprehensive Summary

## 1. Main Topic and Scope
This paper investigates the vulnerabilities of **test-time dynamic model merging** (such as L3 routing) under low-data calibration splits (64 samples). The core research focuses on how dynamic routers, which adapt model merging coefficients per layer on-the-fly during inference, behave under different deployment streams (homogeneous vs. heterogeneous) and batch sizes (large batch sizes vs. single-sample vectorized streaming, $B=1$).

## 2. Technical Approach
The authors construct a controlled, 192-dimensional **Analytical Coordinate Sandbox** simulating a 14-layer backbone processing streams from four downstream visual expert domains of varying noise levels. They evaluate several baseline and state-of-the-art dynamic model merging systems (such as QWS-Merge, unregularized L3-Linear, and L3-Softmax) and compare them to static Uniform Merging.

To address the observed vulnerabilities of these methods, the authors introduce the **Prior-Driven Classical Routing Framework** (specifically **Zero-Initialized Softmax Routing** with $L_2$ weight decay). They also formulate:
- **Task-Variance Regularization ($\mathcal{L}_{VR}$)**: A group-level variance-limiting penalty designed to suppress intra-task routing variance and prevent "heterogeneity collapse".
- **Sequential Smoothness Regularization ($\mathcal{L}_{\text{smooth}}$)**: An auxiliary sequential penalty that suppresses high-frequency layer-to-layer routing weight fluctuations.
- **Sample-Specific Vectorized Assembly**: Running true batch-independent, sample-specific parameter assembly during training and deployment (e.g. using `torch.vmap` or highly optimized `einsum` operations) to bypass the batch-average compromise bottleneck.

## 3. Key Findings
- **The Batch-Average Smoothing Confounder**: Standard large-batch evaluation ($B=256$) on heterogeneous streams masks severe overfitting of dynamic routers. Batch-averaging the predicted coefficients acts as an implicit smoothing operator that hides wild coefficient variations.
- **Vectorization Collapse**: When evaluated on true batch-independent, single-sample streams ($B=1$), the batch-average smoothing mask is removed. Standard unregularized dynamic routers (like random-initialized L3-Softmax) catastrophically drop in performance (e.g., L3-Softmax drops to $41.09\% \pm 3.73\%$, nearly $17\%$ below the static Uniform Merging baseline of $58.00\%$).
- **Prior-Driven Classical Routing Stability**: Zero-initialized Softmax routing with $L_2$ weight decay completely resolves Vectorization Collapse, maintaining a flatline joint accuracy of $59.16\% \pm 1.17\%$ across all batch sizes ($B=1$ to $B=512$).
- **The Dynamic Routing Paradox**: To achieve stability on data-scarce splits, the router must be regularized so heavily that its learned coefficients barely deviate from their initial uniform prior (achieving a Mean Absolute Deviation of only $0.0236$ from the $0.25$ uniform baseline). This heavy constraint leaves the router with marginal functional flexibility, yielding a tiny $+1.16\%$ joint accuracy improvement over naive, training-free Uniform Merging (59.16% vs. 58.00%).
- **Empirical Redundancy of $\mathcal{L}_{VR}$**: An extensive baseline audit and ablation study show that standard L3-Softmax with zero-initialization and weight decay performs identically to VR-Router, proving that simple architectural priors (zero-initialization) carry all the regularizing weight, making explicit loss penalties empirically redundant.

## 4. Explicitly Claimed Contributions (with Evidence)
- **Identification of Batch-Average Confounder & Vectorization Collapse**: Grounded in Table 1 and Table 3, where unregularized L3-Softmax collapses to $41.09\%$ at $B=1$ but achieves $59.35\%$ at $B=256$.
- **Demonstration of Quantum-Inspired Vulnerabilities**: Shows that QWS-Merge (cos-based phase-interferometry) suffers from rugged landscapes under severe data constraints, dropping to $56.19\%$ under $B=1$ (Table 1).
- **Prior-Driven Classical Routing Framework**: Establishes that zero-initialization and weight decay are sufficient to resolve collapse, achieving stable flatline performance of $\approx 59.16\%$ (Table 1, Table 3).
- **Formulation of the Dynamic Routing Paradox**: Quantified by computing the Mean Absolute Deviation (MAD) of learned weights under well-regularized routing, revealing it is only $0.0236$ (or $2.36\%$) from uniform.
- **Exhaustive Empirical Audit**: Proves the claims through a statistical significance sweep (10 seeds), regularization sensitivity sweeps ($\lambda_{var} \in [0.0, 10.0]$), inference stream heterogeneity stress tests ($B \in \{1, 8, 32, 128, 512\}$), loss ablations (Table 4), and real-world CNN expert merging validation (Table 7).
