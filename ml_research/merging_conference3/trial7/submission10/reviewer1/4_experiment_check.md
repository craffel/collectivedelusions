# Experimental Check and Evaluation

## Critical Evaluation of the Experimental Setup
The experimental evaluation is highly thorough, leveraging both high-fidelity analytical simulations and physical PyTorch profiling across two distinct modalities:
- **Vision Modality:** Pre-trained `vit_tiny_patch16_224` backbone evaluated on MNIST, Fashion-MNIST, CIFAR-10, and SVHN (K=4 tasks).
- **Text Modality:** Pre-trained decoder-only `gpt2` evaluated on Legal, Medical, and Code domains (K=3 tasks).

The authors also include extensive edge-hardware execution profiling on a **Raspberry Pi 4 (ARM Cortex-A72 CPU)**.

## Datasets and Baselines
- **Datasets:** The chosen datasets are standard and representative of typical multi-task suites in literature.
- **Baselines:** The baselines are highly appropriate and comprehensive, covering:
  1. *Expert Ceiling:* Upper bound representing isolated experts.
  2. *Uniform Merging:* Static weight-space merging baseline.
  3. *Linear Router (Reg):* Parametric linear routing baseline.
  4. *QWS-Merge SOTA:* Recent advanced quantum-inspired merging baseline.
  5. *PFSR + MBH SOTA:* Prior non-parametric state-of-the-art that uses classification heads and micro-batch partitioning.
  
The physical PyTorch evaluation also compares four distinct implementations of their proposed method (SPS-FP, SPS-SG, SPS-VSG, and SPS-Compiled) to isolate the impact of framework overhead.

## Analysis of Claims vs. Experimental Evidence
1. **Claim: SPS-ZCA recovers 100% of the Expert Ceiling.**
   - *Evidence:* Table 1 (Simulation) and Table 3 (Physical PyTorch) show that SPS-ZCA achieves 79.80% (simulation) and 76.14% (physical) Joint Mean accuracies, which match the Expert Ceiling exactly.
   - *Check:* In the physical PyTorch world, ZCA achieves 100.0% routing accuracy over the test suite, meaning there is zero routing error. Thus, the downstream classification accuracy naturally matches the isolated expert ceiling. This is a very strong and verified result.
2. **Claim: SPS-ZCA overcomes the latency bottleneck of MBH.**
   - *Evidence:* Table 2 (Simulation costs), Section 4.7 (PyTorch latencies), and Appendix B.4 (Raspberry Pi 4 benchmarks) provide detailed timings.
   - *Check:* The authors honestly characterize the "serving gap". Under standard PyTorch with large batch sizes ($B=256$), framework overheads cause SPS to run slightly slower than MBH. However, they show that:
     - At low batch scales ($B=16$), SPS-VSG is physically 1.17$\times$ faster than MBH out of the box (16.63 ms vs. 19.42 ms).
     - Under a compiled C++ custom operator (ONNX Fused), they achieve a verified physical **3.91$\times$ speedup** at $B=1$ and **3.61$\times$ speedup** at $B=256$ on a Raspberry Pi 4 CPU.
     - This physical evidence fully supports their claims and resolves any potential skepticism regarding the "reality gap" of analytical FLOP speedups.
3. **Claim: UNC, IDC, and GMM Shield provide robustness.**
   - *Evidence:*
     - UNC: Figure 3 (Ablation C) shows UNC restores Joint Mean from 79.22% to 79.80% under 5$\times$ scale imbalance.
     - IDC: Section 4.6.4 (Ablation D) shows IDC stabilizes routing under asymmetric task manifold dispersions (restoring balanced routing from 95.40% misrouting down to 47.00% random-chance baseline).
     - GMM Shield: Figure 4 (ROC Curve) and Table 2 (Ablation E) show GMM OOD rejection achieves 95.2% TPR at 4.3% FPR.
   - *Check:* The experimental evidence is extensive and highly convincing. The authors specifically ensure that the 1000 validation samples used to analyze GMM sensitivity are completely distinct and disjoint from the 64-sample calibration split, ruling out coordinate leakage.
4. **Claim: Generalizability to Text Modalities and KV Cache Sharing.**
   - *Evidence:* Section 4.8.2 profiles GPT-2 and shows a Joint Mean accuracy of 91.83% (matching Expert Ceiling and beating Uniform's 52.40%) and near-perfect perplexity preservation (12.18 vs. 12.15).
   - *Check:* This is a robust extension that demonstrates the framework's broad applicability.

## Methodological Strengths in the Experiments
- **Ablation completeness:** Sweeps are provided for almost every hyperparameter ($\tau$, $\eta$, $|\mathcal{C}_k|$, $K$, $B$, scale drift, dispersion scale).
- **Physical Hardware Verification:** Benchmarking on an actual Raspberry Pi 4 CPU with compiled C++ CustomOps bridges the theory-practice divide beautifully.
- **Empirical Capacity Verification:** Section 4.6.7 verifies that freezing the first 3 layers causes a negligible (-0.02%) drop in capacity, validating the feasibility of early-layer routing.
