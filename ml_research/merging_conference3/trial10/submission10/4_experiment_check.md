# 4. Experimental Evaluation Check

This section evaluates the empirical design, results, and potential limitations of the paper's experimental validation.

### 1. Strengths of the Experimental Evaluation
*   **Comprehensive Baselines:** The authors compare 2D-STEM against a wide variety of representative baselines spanning stateless (SABLE), spatial-only (Momentum-Merge), temporal-only (PAC-Kinetics), and spatio-temporal (ChemMerge) ensembling, testing both constant-inertia and fully adaptive (Dynamic ODE) variations of ChemMerge.
*   **Insightful Analysis:** The analysis of the ChemMerge (Dynamic ODE) failure mode on homogeneous streams is highly insightful. 2D-STEM's decoupled architecture (measuring stream similarity at frozen layers) successfully solves the dynamic-ODE trade-off.
*   **Statistical Rigor & Ablations:** Paired t-tests across 5 independent seeds (Table 4) and extensive sweeps over ATG-PL exponent $\gamma$ (Table 6) and spatial/temporal momentum coefficients (Table 7) isolate individual physical contributions and demonstrate robust Pareto-frontier configurations.
*   **Robustness to Calibration Scarcity:** The ablation of calibration set sizes (Section 4.10) shows that 2D-STEM is highly robust, retaining near-optimal performance with as few as 5 samples per task.
*   **Generalizability on Pre-Trained ViT:** The evaluation in Table 3 utilizing a physical pre-trained ViT backbone (`vit_tiny`) with CLS-token activations demonstrates that 2D-STEM reduces absolute routing jitter by over $5.2\times$ compared to stateless SABLE while maintaining stable ensembling trajectories.

### 2. Minor Empirical Limitations

#### A. Performance of PAC-Kinetics and ChemMerge Proxy in Table 3
*   Under the pre-trained Vision Transformer environment (Table 3), PAC-Kinetics achieves the highest alignment accuracy ($70.57\%$ homogeneous, $67.08\%$ heterogeneous) and the lowest routing jitter ($0.0063$ homogeneous, $0.0369$ heterogeneous).
*   **Honest Discussion:** The authors are highly honest and scientifically rigorous in discussing this behavior in Section 4.4.1. They explain that PAC-Kinetics is a temporal-only tracker operating at a static depth. Because it uses constant ensembling coefficients across all blocks within a single sample's forward pass, it is completely immune to the depth-wise layer propagation noise that corrupts deeper layers.
*   **Trade-off:** However, the authors also correctly contextualize this: in more challenging representation environments with severe manifold overlaps (Section 4.3), this complete lack of depth-wise adaptation causes PAC-Kinetics' ensembling accuracy to collapse to $42.02\%$. This highlights that local spatio-temporal coupling (as performed by 2D-STEM) is mandatory for generalizable ensembling.

#### B. Poor Separability and High Overlaps in Table 3
*   Under the ViT setting (Table 3), Uniform Merging achieves high alignment accuracy ($65.06\%$).
*   **Scientific Explanation:** The authors explain that the pre-trained ViT CLS token representations of the 4 visual tasks exhibit a high baseline cosine similarity ($\approx 0.95$) at early routing layers. Because the representations are already highly correlated, any linear mixture (even task-agnostic uniform blending) retains high baseline cosine similarity to the target representation.
*   **Utility:** The authors use this to argue that the primary benefit of stateful ensembling in real-world models is not necessarily finding a sparse task coordinate, but stabilizing trajectories (e.g., reducing jitter by over $5.2\times$) to prevent expensive cache thrashing and DRAM expert transfers. This is a very convincing physical and system-level argument.

#### C. Lack of Hardware serving Benchmarks
*   Since the paper emphasizes the low computational overhead and edge-serving latency advantages of 2D-STEM, reporting actual serving metrics (latency in milliseconds, memory footprint in MB, peak throughput, or power draw) on actual physical edge hardware (such as an NVIDIA Jetson or Raspberry Pi) comparing 2D-STEM, SABLE, and ChemMerge would further strengthen the hardware-efficiency claims.

### Overall Experimental Rating: Excellent
The experimental validation is outstanding. By adding statistical significance tests, extensive ablations, and a highly honest discussion of the pre-trained ViT results (Table 3), the authors have resolved all previous empirical gaps. The results are fully trustworthy and scientifically rigorous.
