# Experimental Evaluation Critique: HyperMerge

## Evaluation Setup
The authors evaluate HyperMerge inside an "Analytical Coordinate Sandbox" (14 layers, 192 dimensions). This is a completely synthetic, artificial coordinate space designed to simulate representation flows, rather than a physical evaluation on real-world pre-trained foundation models.

## Baselines
The baselines used are appropriate and comprehensive, including:
- Expert Ceiling (upper bound)
- Uniform Merging (static baseline)
- PFSR (with and without MBH)
- SABLE (Early Routing and Late Adaptation)
- SPS-ZCA (SOTA Euclidean baseline)

## Do the Results Support the Claims?
No, the empirical results directly contradict the core thesis of the paper.

1. **Failure to Outperform Euclidean Baselines in Clean Regimes:**
   - In Table 1 (Standard Sandbox), the simple Euclidean baseline **SABLE (Early Routing)** achieves an accuracy of **84.03% ± 5.15%**, which is superior to HyperMerge's **83.40% ± 5.15%**.
   - SABLE achieves this without any of the elaborate non-linear hyperbolic machinery, exponential/logarithmic mappings, or Klein-space coordinate transformations.

2. **Failure to Outperform Euclidean Baselines in Crowded/Overlapping Regimes:**
   - The authors introduce the "Overlapping Subspace Sandbox Regime" specifically to prove that HyperMerge's hyperbolic space resolves representation crowding.
   - However, even under this heavily crowded substrate (Table 3), **SABLE (Early Routing)** still achieves a joint mean accuracy of **77.98% ± 2.12%**, outperforming HyperMerge ($c=0.1$) at **76.62% ± 3.96%** and HyperMerge (Tuned) at **76.50% ± 3.36%**.
   - **SPS-ZCA** (the other Euclidean baseline) also outperforms HyperMerge under this regime (**77.32% ± 1.98%** vs **76.62% ± 3.96%**).
   - This is a highly damaging result: in the very scenario designed to showcase the advantage of negative curvature in resolving representation crowding, the flat Euclidean methods perform significantly better.

3. **The "Tuned" Parameter Degradation:**
   - In Table 3, the "Tuned" version of HyperMerge (at $c=0.2, \tau=0.08$) actually performs *worse* than the default version ($c=0.1$) (**76.50%** vs **76.62%**). This raises doubts about the stability of the hyperparameters and whether the hyperbolic curvature actually helps at all.

4. **Artificial Nature of the Evaluation:**
   - The entire evaluation is based on synthetic 192-dimensional vector partitions. There is no validation on physical datasets (e.g., GLUE, MMLU, ImageNet) using physical pre-trained models (e.g., LLaMA, RoBERTa, CLIP), despite the authors providing a "real-world blueprint" in Appendix B. An appendix blueprint cannot substitute for empirical evidence.
