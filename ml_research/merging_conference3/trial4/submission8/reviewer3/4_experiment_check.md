# Experimental Evaluation

## Experimental Setup and Dataset Choices
The experimental design is highly rigorous and appropriate:
- **Backbone and Experts:** The choice of a lightweight Vision Transformer (`vit_tiny_patch16_224`) with 14 layer-wise modules provides a clear structural testbed. The expert fine-tuning setup is solid, achieving high-quality task ceilings on MNIST (96.30%), FashionMNIST (86.90%), CIFAR-10 (90.20%), and SVHN (81.30%).
- **Multi-Domain Composition:** Merging experts across MNIST, FashionMNIST, CIFAR-10, and SVHN represents a highly challenging, disjoint multi-domain composition problem. This is a robust test for representation interference.
- **PTQ Target Schemas:** The authors evaluate 6 distinct deployment schemas, covering both tensor-wise and channel-wise, symmetric and asymmetric, and INT8/INT4 precision levels. This is an exhaustive and highly practical evaluation of hardware robustness.

---

## Baseline Coverage
The baseline coverage is exemplary:
- It includes a static, zero-compute baseline (**Uniform Task Arithmetic**).
- It covers unregularized test-time adaptation (**AdaMerging**).
- It includes spatial smoothness regularization (**RegCalMerge**), which uses a Total Variation penalty.
- It covers quantization-aware optimization (**Q-Merge**) with Straight-Through Estimators.
- It includes the subspace-constrained adaptive baseline (**PolyMerge**).
- It compares against unconstrained coefficient-space sharpness minimization (**HessMerge**).
This diverse selection ensures a fair and comprehensive comparison.

---

## Support for Core Claims
The empirical results provide direct, statistically significant support for all core claims:

1. **Synergy of Subspaces and Sharpness (CR-PolySACM's Superiority):**
   Under the aggressive **INT4 Symmetric per-Channel** target, CR-PolySACM achieves **19.07%** joint mean accuracy, outperforming the state-of-the-art PolyMerge baseline (18.10%) by $+0.97\%$ (statistically significant, $p < 0.01$). This directly supports the claim that local flatness optimization within a stable, global polynomial subspace provides crucial robustness under severe discretization noise.

2. **The Task-Vector Norm Scale Pathology (Ablation of $\gamma$):**
   The ablation of the unconstrained regularizer strength $\gamma$ in Table 5 shows that as $\gamma$ increases, both FP32 and INT8 performance monotonically degrade. This supports the claim that unconstrained sharpness optimization is blind to sensitive, low-norm layers (such as the final layer norm) and instead overfits to less sensitive, high-norm layers, pushing the model into sub-optimal regions.

3. **Validation of CR-SACM (Ablation of $\beta$):**
   The non-monotonic performance trend across the clipping threshold $\beta$ (Table 6) perfectly validates the dual failure modes predicted by the theory:
   - Extremely small $\beta \le 0.01$ triggers division-by-zero/gradient explosion (11.20% accuracy).
   - Extremely large $\beta \ge 0.25$ triggers scale-blindness, recovering standard PolyMerge performance (18.15% accuracy).
   - $\beta = 0.10$ provides the mathematically optimal scale balance (19.07% accuracy).

4. **Correcting HessMerge (HessMerge's Breakthrough):**
   By incorporating CR-SACM to resolve scale-blindness, HessMerge (Ours) consistently and significantly outperforms unregularized AdaMerging across all schemas (e.g., $50.48\%$ vs $49.12\%$ in FP32, and $50.05\%$ vs $48.80\%$ in INT8 Asym Channel), proving that sharpness-aware adaptation is highly effective once scale-blindness is resolved.

5. **Theoretical Noise Decomposition (Table 3):**
   The empirical measurements of $\|J_{\mathbf{p}}\boldsymbol{\epsilon}\|_2$ and $\|\delta_{\perp}\|_2$ show that under INT4, out-of-subspace noise $\delta_{\perp}$ is $7.71\times$ larger than controllable in-subspace noise, explaining why unconstrained TTA collapses and highlighting the critical necessity of global structural constraints (PolyMerge).

---

## Critical Appraisal of Limitations
The authors are commendably honest and transparent about two key limitations:
- **Absolute INT4 Performance:** While CR-PolySACM achieves a clear and statistically significant relative improvement (+0.97% over PolyMerge), the absolute joint accuracy of 19.07% remains extremely low and practically unusable for production systems. The authors correctly state that the value of these results is primarily scientific (proving the theoretical flatness-quantization synergy) rather than practical.
- **Expert-to-Merge Gap:** There is a substantial performance drop from individual experts (average 88.67%) to the continuous merged model (57.40% in FP32). This $-31.27\%$ gap is an inherent challenge in weight-space merging when tasks are fine-tuned on highly disparate domains, where orthogonal task-vector interference is severe.
This transparency is highly refreshing and increases the scientific credibility of the work.
