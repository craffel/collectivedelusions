# Experimental Rigor & Results Check: BC-Router

## 1. Experimental Assessment
**Rating: Excellent**

The experimental design and empirical rigor in this paper are of exceptionally high quality. The authors evaluate their framework on fully converged specialized task experts, benchmark under both homogeneous and heterogeneous stream protocols, perform comprehensive scaling and sensitivity analyses, and honestly address the fundamental trade-offs and physical limitations of weight-space dynamic routing.

### A. Thorough and Fair Evaluation Pipelines
The empirical results are highly reliable and robust:
*   **Converged Experts:** The authors fine-tune their ViT backbone to high convergence, establishing strong upper bounds (92.8% to 100.0% accuracy), preventing under-tuned experts from confounding the merging analysis.
*   **Multi-Seed Evaluation:** All homogeneous and heterogeneous results are reported as means and standard deviations over 3 random calibration-sampling seeds, providing clear visibility into optimization stability.
*   **Heterogeneous Stream Simulation:** Evaluating adaptation under temporal noise across batch sizes $B \in \{1, 16, 256\}$ is a highly realistic, rigorous setup.

### B. The Generalist-Specialist Paradox & Operational Trade-offs
A major strength of the paper's experimental discussion is Section 4.4, where the authors address the "practical utility paradox" of dynamic model merging. They honestly analyze why simple static **Uniform Merge (Task Arithmetic)** achieves a superior overall homogeneous joint mean accuracy ($85.10\%$) compared to trainable dynamic routers ($82.80\%$ for Linear Router (Reg) and $83.63\%$ for BSigmoid-Router (Reg)). 

Instead of overselling their method, they explain the physical limitation: weight-space merging is a zero-sum game of parameter capacity. Merging cannot create new capacity; it can only reallocate existing parameters on-the-fly. Consequently, steering weights toward a hard task (like SVHN, where Linear Router (Reg) achieves $91.73\%$ vs. Uniform Merge's $77.60\%$) inevitably degrades accuracy on the other tasks. They define three critical scenarios where dynamic routing is practically useful:
1.  **Peak Domain Specialization:** Forcing the model to operate at its specialized peak accuracy on a difficult domain when static compromise is unacceptable.
2.  **Inference-Time Domain Steering:** Adapting on-the-fly when processing local streams of a single domain for prolonged periods.
3.  **Dynamic Safety & Capability Masking:** Real-time disabling or filtering of specific capabilities without requiring re-merging and redeployment.

### C. Comprehensive Scaling & Sensitivity Analyses
The paper includes highly informative auxiliary analyses that strengthen its claims:
*   **Calibration Set Size Ablation:** Scaling the calibration set from 64 to 256 samples demonstrates that larger budgets act as data-driven stabilizers for unregularized classical routers, reducing GLS-Router's SVHN standard deviation from 24.30% to 4.60%.
*   **Regularization Strength Sensitivity:** Sweeping weight decay $\gamma \in [0, 10^{-5}, 10^{-4}, 10^{-3}]$ clearly maps the transition from unregularized overfitting to flat, static uniform blending.
*   **Inference Latency Profiling:** Demonstrates that BSigmoid-Router runs as a pure forward pass taking only 18.5ms per batch (over 25x faster than AdaMerging's 495.0ms) and identifies PyTorch tensor copying as the main physical bottleneck.

---

## 2. Experimental Strengths and Weaknesses
*   **Strengths:**
    *   Highly converged task-specific expert baselines.
    *   Dual homogeneous and heterogeneous stream evaluations across batch sizes.
    *   Superb academic honesty regarding the performance-balance trade-offs and physical capacity limits.
    *   Excellent inclusion of calibration budget scaling curves, regularization sensitivity, and latency profiling.
*   **Weaknesses:**
    *   The stream evaluation protocols model AdaMerging statically on the stream to bypass its prohibitive active optimization latency, which is an optimistic upper-bound simplification that fails to capture real-world online gradient noise and parameter drift.
