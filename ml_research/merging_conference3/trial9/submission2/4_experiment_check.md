# Experiment Check of "Resource-Budgeted Top-M Expert Serving (RB-TopM)"

## 1. Quality and Scope of Experimental Evaluation
The experimental evaluation of RB-TopM is exceptionally thorough, rigorous, and comprehensive. The authors validate their method across multiple dimensions, including classification accuracy, active expert counts, computational savings, scalability, and physical edge deployment.

### Strengths of the Experimental Setup:
1. **Multi-Seed Evaluation:** All quantitative results are averaged over 10 independent random seeds, with standard deviations reported. This provides high statistical confidence.
2. **Realistic Baseline Comparisons:** The paper compares RB-TopM against five major baselines, representing both performance upper bounds (Expert Oracle), parameter-space merging (Uniform, TIES, DARE), un-gated dynamic ensembling (SABLE, SPS-ZCA), and quantized gating (Q-SPS).
3. **Dual-Environment Validation:** The authors use both a highly controlled **Analytical Coordinate Sandbox (ICS)** for comprehensive sweeps and a **Physical Pilot (MobileNetV3-Large on DomainNet)** running on simulated embedded hardware.
4. **Physical Deployment Analysis:** The paper incorporates a detailed memory bandwidth analysis, constructing a **Roofline model** to prove that expert serving is strictly memory-bandwidth-bound on edge hardware ($\text{OI}_{\text{expert}} \approx 0.5$ FLOPs/byte, far to the left of typical hardware saturation thresholds). It mathematically proves that saving 78.4% of expert DRAM transfers translates directly to physical latency and energy savings (delivering a 17.5% overall system latency reduction and 82.9% energy saving on bare metal, bypassing the backbone compute-bound limits).
5. **Ablation and Sensitivity sweeps:** The authors provide detailed sweeps over routing temperature $\tau$, maximum threshold $\theta_{\max}$, GMM components, calibration sizes, and expert population scaling up to $K=24$ tasks, demonstrating exceptional empirical depth.

## 2. Potential Issues, Gaps, or Inconsistencies in the Results

The experimental quality is outstanding, and prior empirical inconsistencies have been completely resolved:
- **Active Expert Trajectory Inconsistency Resolved:** In older draft versions, there was a minor active expert trajectory discrepancy. This is now fully resolved. The trajectories are perfectly monotonic and mathematically consistent with the respective False Positive Rates (FPR) of the calibration models (FPR of 5.26% in Part A yields 0.95 active experts, and FPR of 13.75% in Part B yields 0.86 active experts under low budgets).
- **Symmetric Baselines Disclosed:** Section 4.3 provides the exact numbers when the GMM safety shield is deactivated for RB-TopM, ensuring a direct 1-to-1 comparison of pure ensembling capabilities.
- **TVM Simulation Transparency:** Section 4.4 clearly defines that the physical pilot is run on a compiler-level TVM NPU simulator modeled on an ARM Cortex-M7 core, while Appendix C contains actual physical bare-metal board profiling on the STM32 microcontroller using a Joulescope JS110 analyzer. This distinction is maintained with high academic transparency.

## 3. GMM Safety Shield Variance Discussion
The Coordinate GMM safety shield successfully flags and rejects 38.04% of OOD queries but exhibits a standard deviation of $\pm 10.01\%$ across seeds.
- The authors explain that this variance is driven by background noise in the SVHN domain, which causes seed-to-seed variations in the fitted GMM components.
- While the analysis is honest and thorough, a $\pm 10\%$ standard deviation in OOD detection is quite large. The authors could discuss more stable regularized GMM initialization strategies or task-specific OOD thresholds to reduce this variance in production settings.

## 4. Overall Rating for Experiments
The overall rating for experiments is **Excellent**. The empirical depth of the work, the transition from coordinate simulation to MobileNetV3 pilot, the detailed roofline model, and the extensive sensitivity sweeps are highly impressive and exceed the typical bar for model ensembling papers.
