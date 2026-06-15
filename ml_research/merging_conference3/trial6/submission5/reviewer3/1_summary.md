# 1. Summary of the Paper

## Main Topic
The paper investigates **test-time dynamic model merging**, a paradigm that combines specialized expert neural networks into a single multi-task model at inference time using an input-dependent routing network. Specifically, the work focuses on identifying and resolving critical deployment vulnerabilities and deconstructing the fundamental statistical and systems-level limitations of dynamic routing under severe data scarcity (e.g., a 64-sample calibration split).

## Proposed Approach
The authors adopt a rigorously empirical and statistical methodology to evaluate dynamic routing mechanisms. Their approach consists of:
1. **Identifying Vulnerabilities**: Exposing **Vectorization Collapse** (severe performance drop under sample-wise vectorized deployment, $B=1$) and **Heterogeneity Collapse** (performance dilution under mixed-task batch streaming due to batch-averaged coefficients).
2. **Exposing Confounders**: Detailing the **Batch-Average Smoothing Confounder**, where standard large-batch evaluations average predicted coefficients and mask severe router overfitting on low-data splits.
3. **Prior-Driven Classical Routing**: Proposing a simple, mathematically elegant framework consisting of a **Zero-Initialized Softmax prior** combined with standard $L_2$ weight decay, projected onto a unit sphere via normalized random projections.
4. **Task-Variance Regularization ($\mathcal{L}_{VR}$)**: Formulating an optional group-level variance penalty to suppress intra-task routing variance, serving as a mathematically precise limit target.
5. **Experimental Evaluation**: Creating a controlled 192-dimensional synthetic **Analytical Coordinate Sandbox** alongside a high-fidelity calibrated simulator to evaluate methods across 10 independent random seeds. 
6. **Real-World Validation**: Fine-tuning convolutional visual experts on MNIST and FashionMNIST using a shared backbone to validate their findings in a realistic scenario.
7. **Systems-Level Mitigation**: Proposing **Low-Rank Parameter Assembly (Dynamic LoRA)** to bypass high latency and memory overheads ($O(B \cdot M)$) of dynamic full-parameter assembly.

## Key Findings
* **Catastrophic Collapse**: Unregularized global and layer-wise routers (such as L3-Softmax and standard linear routers), as well as complex quantum-inspired architectures (QWS-Merge), suffer severe performance degradation (e.g., unregularized L3-Softmax drops to 41.09% accuracy at $B=1$, nearly 17% below static Uniform Merging) under sample-wise vectorized streaming.
* **Power of Simple Priors**: Proper zero-initialization of Softmax routing layers combined with standard weight decay acts as a highly robust implicit regularizer. Both the well-regularized standard Softmax baseline (`L3_Softmax_WellReg`) and the variance-regularized VR-Router completely resolve Vectorization Collapse, maintaining flatline joint accuracies of **59.16%** and **59.14%** respectively across all batch sizes ($B=1$ to $B=512$).
* **Redundancy of Complex Objectives**: Explicit Task-Variance Regularization ($\mathcal{L}_{VR}$) is empirically redundant once the zero-initialized Softmax architectural prior is established, indicating that the baseline with zero-initialization naturally satisfies the group-level variance limit.
* **The Dynamic Routing Paradox**: To prevent Vectorization Collapse under data scarcity, the router must be regularized so heavily that its learned coefficients barely deviate from their initial uniform prior (Mean Absolute Deviation of only $0.0236$ or $2.36\%$). This heavy restriction limits the router to a tiny $+1.16\%$ joint accuracy improvement over naive, training-free **Uniform Merging** (59.16% vs. 58.00%).
* **Practical Recommendation**: Since dynamic full-parameter assembly introduces a massive memory footprint expansion and a $110.06\times$ latency slowdown for large batches, naive static Uniform Merging is a highly practical, zero-cost, and superior default in data-scarce settings.

## Explicitly Claimed Contributions (with Evidence)
1. **Discovery of Vectorization Collapse and Batch-Average Smoothing**: Supported by empirical batch-size sweeps ($B \in \{1, 8, 32, 128, 512\}$) where unregularized models collapse under $B=1$ but appear successful under $B=256$.
2. **Prior-Driven Classical Routing Framework**: Proven via `L3_Softmax_WellReg` and `VR_Router`, which completely eliminate collapse and outperform all unregularized dynamic baselines.
3. **Rigorous Statistical Evaluation**: Demonstrated by conducting sweeps across 10 independent random seeds and mapping the regularization sensitivity frontier ($\lambda_{var} \in [0, 10]$).
4. **Deconstruction of the Dynamic Routing Paradox**: Supported by quantitative weight analyses demonstrating a low MAD of $2.36\%$ from the uniform $0.25$ baseline.
5. **Dynamic LoRA and Physical Latency Benchmarks**: Supported by empirical CPU latencies showing Dynamic LoRA ($r=8$) resolves the full-parameter assembly bottleneck, scaling seamlessly to $B=512$ with only a $1.01\times$ slowdown.
6. **Real-World Demonstration**: Demonstrated using actual visual experts on MNIST/FashionMNIST, confirming that dynamic parameter merging works and exhibits the batch-averaging confounder on real-world image datasets.
