# 4. Experimental Evaluation Check

This section provides a thorough check and audit of the experimental setup, baselines, and empirical results.

## Audit of Experimental Results

### 1. Baseline Selection
The authors compare against a comprehensive and competitive set of baselines:
- **Expert Oracle**: Upper ceiling.
- **Uniform Merging**: Static baseline.
- **SABLE & ChemMerge**: High-overhead non-parametric ensembling methods.
- **Shared Router**: Parameter sharing across depth.
- **L2-Fixed Router**: Standard weight regularization with fixed temperature.
- **Linear Router**: Unregularized parametric baseline.

This represents an exceptionally rigorous baseline comparison for this problem domain.

### 2. Active Gating Evaluation Metrics
- Standard evaluation in sandboxes suffers from "routing accuracy illusions" where static Uniform Merging can achieve near-oracle routing accuracy in orthogonal spaces.
- To resolve this, the authors introduce **Direct Gating Accuracy** and **Gating Cross-Entropy**, which measure the actual layer-wise gating decisions. This is an excellent, high-integrity methodological improvement that ensures the evaluation is fair and leak-free.

### 3. Key Findings & Trade-offs
- **CR-Router vs. L2-Fixed:** In Experiment 3 (Real-World Vision Embedding), CR-Router significantly outperforms L2-Fixed by **+6.37%** in classification accuracy (53.70% vs. 47.33%). This demonstrates that joint spectral-temperature regularization is highly superior to simpler fixed-temperature parameter regularizers in realistic non-orthogonal manifolds.
- **The Gap with Non-Parametric Baselines (SABLE / ChemMerge):** SABLE achieves 70.60% classification accuracy in Experiment 3, whereas CR-Router achieves 53.70% (and 62.45% with Adaptive Test-Time Temperature Annealing). This represents a performance gap of **~8.15%** absolute. The authors candidly attribute this gap to "expert dilution" and the fact that CR-Router is constrained to be a strict contraction, whereas non-parametric methods are unconstrained. 
- **Validation of Online Heuristics:** The empirical sweeps in Table 6 beautifully validate the three proposed label-free online tuning heuristics, proving that the optimal contraction regime corresponds to a minimized depth-variance shelf and a stable entropy valley.

### 4. Serving Efficiency Benchmarks (Table 9)
- Unlike many papers that only theoretically argue about serving efficiency, this paper includes a concrete **forward-pass latency and throughput profiling benchmark** (Table 9) executed on a batch size of $B=400$ over 100 warm-started iterations.
- The results elegantly support the authors' claims: CR-Router reduces latency to **25.34 ms** (a **33.7% latency reduction** compared to SABLE's 38.23 ms) and increases throughput to **15,785.1 samples/s** (a **1.51x speedup** compared to SABLE's 10,464.1 samples/s). This provides highly convincing empirical validation of the practical advantages of parametric learned routers in production serving.

## Experimental Weaknesses & Suggestions for Improvement

1. **In-depth Hyperparameter Sensitivity across Seeds:**
   - Table 4 (regularization sensitivity analysis), Table 6 (label-free heuristics), and Table 7 (fine-grained sweep over lambda) report results on **Seed 42 only**. While the main tables (Tables 2, 3, and 5) and the test-time temperature annealing table (Table 8) report results averaged across 10 random seeds with standard deviations, these sensitivity sweeps should also be reported with mean and standard deviation across seeds to demonstrate that these behaviors are robust and not seed-dependent.

2. **Scaling to GPU Accelerators and Larger Batches:**
   - The serving efficiency benchmark (Table 9) was conducted on a standard CPU machine on a batch size of $B = 400$. To better represent realistic high-throughput enterprise model serving, the authors should evaluate these latency and throughput numbers on modern GPU accelerators (e.g., NVIDIA A100 or H100) and scale the batch size up to larger scales (e.g., $B = 1024, 2048$).

3. **Scaling to LLM Benchmarks:**
   - The paper's experiments are limited to the Analytical Coordinate Sandbox (synthetic) and PCA-projected ResNet18 embeddings of MNIST, Fashion-MNIST, KMNIST, and USPS.
   - To make a high-impact contribution, the authors must validate CR-Router on **large-scale language model servings** (e.g., LLaMA-7B or Mixtral backbones with routed LoRA expert adapters on multi-task LLM benchmarks such as GLUE, SuperGLUE, or MMLU). Vision embeddings are a good proxy, but modern deep model merging is heavily focused on NLP and LLMs.
