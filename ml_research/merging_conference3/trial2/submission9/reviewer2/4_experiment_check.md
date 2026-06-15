# Experimental Evaluation and Baseline Integrity: "Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging"

## 1. Evaluation of the Experimental Setup and Baselines
The authors evaluate BPAM on an 8-task image classification benchmark using CLIP ViT-B/32. While the benchmark dataset selection is standard, the experimental design, baseline comparisons, and the authors' interpretation of their results suffer from severe flaws:

### A. Marginal, Non-Significant Gains Over Static Baselines
Under frozen heads (Part A), BPAM-Static (69.21% average accuracy) achieves a measly **+0.11%** absolute improvement over standard, training-free **Task Arithmetic** (69.10%). 
*   **The Cost-Benefit Paradox:** To achieve this +0.11% gain, BPAM requires **200 epochs of test-time optimization**, taking **14.2 minutes** of GPU calibration runtime. Task Arithmetic requires **0.0 minutes** of runtime and zero optimization. Spending 14.2 minutes of high-end GPU compute to achieve a 0.11% improvement is highly impractical and demonstrates that their proposed adaptive weight-space optimization is virtually useless.

### B. Severe Underperformance Compared to Static TIES-Merging
Under both frozen and active head settings, the proposed adaptive method is **strictly dominated by static, training-free alternatives**:
*   **Frozen Heads:** BPAM-Static (69.21%) underperforms compared to static, zero-compute **TIES-Merging** (72.90%) by a substantial **-3.69%** absolute.
*   **Active Heads:** BPAM-Full (75.22%) underperforms compared to **TIES-Merging + Head Tuning** (78.50%) by a massive **-3.28%** absolute.
*   **The Practical Implication:** Simply performing downstream decision-boundary head tuning on top of a strong, conflict-resolved static weight model (TIES-Merging) is strictly superior to running joint adaptive weight-and-head optimization under low-parameter constraints. This completely undermines the utility of the proposed BPAM method. Why would any practitioner adopt a complex test-time optimization pipeline that performs significantly worse than a static, zero-compute merging baseline combined with basic linear head tuning?

### C. Gaping Performance Deficit Against High-Capacity Adaptive Methods
Under frozen heads, SOTA high-capacity adaptive merging methods like **SyMerge** and **FoldMerge** achieve **83.56%** average accuracy. This represents a massive **+14.35%** absolute improvement over BPAM-Static (69.21%). 
Similarly, **AdaMerging** (which utilizes layer-wise parameters) achieves **83.17%** (+13.96% higher).

While the authors frame BPAM as a "boundary probe" to map limits, this massive 14% gap confirms that extremely low-parameter regimes (like 8 scalars) are fundamentally incapable of performing genuine weight-space alignment. The proposed method is essentially a non-competitive baseline that simply documents that global task scalars are too mathematically constrained to work.

### D. The Latency and Memory Overhead Illusion
The authors argue that BPAM's 14.2-minute calibration time is "highly efficient" compared to FoldMerge (45.0 mins) and AdaMerging (25.4 mins), and that once calibrated, downstream inference has zero overhead. 

However, they gloss over a massive practical bottleneck: **The Expert Teacher Memory Footprint.**
To compute the KL-divergence objective during test-time adaptation, the pipeline must feed each batch of target images through **all $K$ expert teacher networks** simultaneously to obtain prediction distributions. For our $K=8$ expert benchmark, this requires hosting and running **9 parallel foundation models in GPU memory** (8 teachers + 1 merged model) during the 14.2-minute calibration phase. 
In real-world settings where model merging is used precisely to *reduce* hosting costs and fit models onto resource-constrained hardware, requiring a 9x memory footprint for adaptation is highly self-defeating and severely restricts the practicality of this approach.

### E. Complete Lack of Statistical Significance and Variance Metrics
Test-time adaptation is notoriously sensitive to batch ordering, local data distribution shifts, and small calibration set sizes. Despite this inherent volatility, the authors present all experimental results in Tables 1, 2, 4, and 5 as single-run deterministic values. 
They provide **no error bars, no standard deviations, and no statistical significance tests (such as p-values)** over multiple random seeds or batch shuffles. Reporting marginal improvements (like +0.11% or +0.42%) as meaningful scientific findings without verifying their statistical significance is a major empirical weakness.
