# Experimental Evaluation

## Experimental Setup
The authors evaluate stateful routers in two environments:
1. **Analytical Coordinate Sandbox (ACS):** A 14-layer simulated representation space with 4 tasks. It simulates block-stable (Homogeneous, block length of 50) and rapid-switching (Heterogeneous, block length of 1) streams under Orthogonal and Overlapping manifold layouts.
2. **Pre-Trained Vision Transformer CLS-Token Simulation:** A 12-layer ViT-Tiny model. It simulates serving over 200 steps on 4 synthetic visual domains.

## Baselines
The paper compares 2D-STEM against an extensive list of relevant baselines:
- **Expert Oracle:** The hypothetical ceiling (100% accuracy, 0.0060/1.5095 jitter).
- **Uniform Merging:** Merges experts with a static weight of $1/K = 0.25$.
- **SABLE (Stateless):** Nearest-centroid routing at each layer and sample.
- **Momentum-Merge (Spatial-only):** Depth-wise EMA, resetting per sample.
- **ChemMerge (Constant-Inertia Proxy):** Fixed 2D EMA with $\beta_{\text{depth}} = 0.60, \beta_{\text{temp}} = 0.30$.
- **ChemMerge (Dynamic ODE):** Adaptive biochemical reaction kinetics with Arrhenius scaling and Euler ODE integration.
- **PAC-Kinetics (Temporal-only):** Test-time state-space recurrence with static weights across depth.

The baseline selection is highly thorough and fair.

## Do the Results Support the Claims?
- **Claim 1: 2D-STEM filters out representation noise and reduces jitter on homogeneous streams.**
  - *Supported:* Yes, in both environments. In ACS, 2D-STEM reduces routing jitter to 0.0070 (Orthogonal) and 0.0068 (Overlapping), which is extremely close to the Oracle's 0.0060 and represents a $2.75\times$ reduction compared to SABLE (0.0187). On the pre-trained ViT, 2D-STEM reduces jitter by $5.23\times$ compared to SABLE (0.0675 vs. 0.3530).
- **Claim 2: ATG-PL suppresses transition lag and outperforms constant-inertia baselines.**
  - *Supported in ACS:* Yes, on heterogeneous streams in ACS, 2D-STEM achieves $94.66\%$ (Orthogonal) and $92.82\%$ (Overlapping) accuracy, outperforming the constant-inertia ChemMerge Proxy ($42.78\%$ and $45.76\%$) by up to $51.88\%$ absolute accuracy.
  - *Not Supported on Physical ViT:* No, in the physical ViT experiment, the constant-inertia ChemMerge Proxy actually *outperforms* 2D-STEM in alignment accuracy ($65.04\%$ vs. $64.61\%$) and achieves lower routing jitter ($0.0428$ vs. $0.0679$).
- **Claim 3: 2D-STEM surpasses highly parameterized frameworks (PAC-Kinetics, ChemMerge).**
  - *Partially Supported:* In ACS, 2D-STEM outperforms PAC-Kinetics and ChemMerge Dynamic on homogeneous and heterogeneous streams under overlapping manifolds.
  - *Contradicted on Physical ViT:* In the pre-trained ViT CLS-token trajectory simulation, **PAC-Kinetics** outperfroms 2D-STEM by a massive margin: achieving **$70.57\%$** alignment accuracy (vs. 2D-STEM's $63.70\%$) and **$0.0063$** routing jitter (vs. 2D-STEM's $0.0675$) under homogeneous streams. This undermines the claim that 2D-STEM's spatio-temporal filter is universally superior. In a realistic model representation space, PAC-Kinetics' depth-wise isolation is highly beneficial, a property that 2D-STEM's localized recurrence fails to match.

## Missing Experiments and Gaps
The most significant experimental gap is the **complete absence of a real-world downstream classification task using fine-tuned physical LoRA experts**.
- A practitioner looking to deploy this method wants to see actual classification accuracy on standard vision datasets (e.g., CIFAR-100, SVHN, DomainNet) or NLP benchmarks, running on a physical backbone (e.g., ViT-Base or LLaMA-7B) with merged LoRA adapters.
- Instead, the authors only evaluate on a highly simulated coordinate sandbox (ACS) and a CLS-token activation trajectory simulation on synthetic visual noise patterns. Without real downstream accuracy and hardware latency measurements on physical edge devices (e.g., NVIDIA Jetson or Raspberry Pi), the practical utility and claimed speedups of 2D-STEM remain unproven.
- Furthermore, the ablation of the MLP coordinate-prior mapper (Section 4.5 and Appendix B) lacks details on training sensitivity, overfitting risk, and its actual impact on the physical ViT results.
