# 4. Experimental Evaluation and Baselines Check

## Empirical Strengths of the Setup

The experimental validation is highly structured, transparent, and rigorous within its scope:
1. **Multi-Baseline Comparison:** The paper compares GraviMerge against a comprehensive suite of baselines including:
   - **Uniform Merging:** Static baseline.
   - **SPS-ZCA:** Early shared-layer single-pass routing.
   - **SABLE:** State-of-the-art stateless cosine-similarity routing.
   - **EMA Smoothing (DSP):** Standard first-order signal filter.
   - **WMomentum:** Weight-space second-order momentum filter.
   - **ChemMerge (SOTA Kinetics):** First-order non-equilibrium chemical reaction kinetics.
   - **Kalman Filter (DSP):** Mathematically optimal first-order state-space tracking filter.
2. **Comprehensive Evaluation Metrics:** The evaluation measures both **Joint Serving Accuracy** (performance) and **Layer-to-Layer Routing Weight Jitter** (using Mean Absolute Deviation, MAD, of ensembling coefficients across sequential depth) to thoroughly assess the accuracy-stability Pareto frontier.
3. **Multi-Stream Workload Evaluation:** The models are tested across three edge-serving scenarios to verify numerical consistency:
   - Simulated Homogeneous Batching.
   - Simulated Heterogeneous Batching ($B=256$).
   - Simulated Real-Time serving ($B=1$, vectorized single-stream serving typical of resource-constrained devices).
4. **Hardware Latency Benchmarks:** In Section 4.5, the authors conduct a sequential wall-clock latency evaluation on LLM-scale dimensions ($D = 4096$, equivalent to Llama-3-8B) for $K \in \{4, 8, 16, 32, 64\}$. They demonstrate that GraviMerge’s routing equations scale sub-linearly and add under 4 ms of CPU execution overhead across 12 layers, verifying negligible computational overhead in real-world servers.
5. **Robustness & Ablation Sweeps:** 
   - **OOD Resilience:** Evaluates Sentinel Attractor Dynamics (SAD), showing it reduces ensembling weight variance to $0.0578$ under OOD inputs, locking the probe at the geometric barycenter.
   - **Adaptive Viscous Drag:** Shows dynamic drag modulation successfully optimizes the accuracy-stability trade-off.
   - **Noise Robustness:** Evaluates performance under Gaussian representational noise, verifying that GraviMerge naturally dampens high-frequency noise.

---

## Key Limitations and Empirical Gaps

Despite the exceptional thoroughness of the evaluation, there is a **major structural limitation** regarding the dataset and setting:

1. **Toy-Scale Simulation (Projected Digit Manifolds):**
   - The primary RDS benchmark is built from scikit-learn's real **handwritten digits dataset** (MNIST-like 8x8 grayscale images, containing only 1,797 samples total).
   - To simulate a "deep representation space," the 64-dimensional features are projected using a random orthogonal matrix to $D=192$ dimensions.
   - The "14-layer backbone" is not a real deep network (like a pretrained LLaMA or ResNet), but a simulated propagation model where intermediate representations are linearly updated based on blended centroids.
2. **Lack of Downstream NLP/Vision Task Evaluation:**
   - There are no experiments on standard Large Language Model (LLM) benchmarks (such as MMLU, GSM8k, or GLUE) or large-scale computer vision benchmarks (such as ImageNet-split or VTAB) where LoRA expert ensembling is actually used in production.
   - Although the appendix includes a 12-layer deep Transformer scale verification ($D=768$, $K=4$ tasks) demonstrating complete immunity to representation collapse and $1.06 \times 10^6\times$ jitter reduction, this verification still operates on a projected task simulator rather than a real text generation or image classification pipeline.
3. **Absence of Real GPU Serving and Execution Profiling:**
   - The authors report sequential execution wall-clock latency on a CPU core showing sub-4ms execution. However, physical edge serving on modern GPUs is highly bottlenecked by memory bandwidth, sequential kernel launch overhead, and thread synchronization. The paper would be significantly stronger with physical GPU serving benchmarks and an evaluation of a fused CUDA/Triton kernel.
4. **Hyperparameter Tuning Complexity:**
   - The system relies on multiple interdependent hyperparameters ($G$, $\epsilon$, $\gamma_{\text{drag}}$, $\tau_{\text{grav}}$, $\eta_{\text{feedback}}$, etc.). High-dimensional spaces ($D=4096$) present unique geometric properties (e.g., near-orthogonality of random vectors), and while the authors propose excellent auto-tuning extensions in the appendix—specifically **Adaptive Gravitational Scheduling (AGS)** and **Adaptive Viscous Drag Scheduling**—these are only evaluated on simulated geometries rather than realistic high-dimensional LLM manifolds.

---

## Conclusion on Empirical Soundness
The paper's experiments are exceptionally complete, transparent, and reproducible *within the bounds of a simulated coordinate sandbox*. However, the absence of evaluation on a real, large-scale deep learning model (e.g., Llama-3-8B or Mistral-7B) performing downstream NLP generation tasks on real GPUs is a significant limitation that prevents immediate empirical generalizability. The paper is positioned as a foundational geometric and mathematical validation rather than a downstream systems deployment paper.
