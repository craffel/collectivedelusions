# Peer Review

## Summary of the Paper
This paper addresses the problem of deploying deep neural networks on resource-constrained edge devices for concurrent multi-task execution. To avoid the high on-device DRAM/SRAM storage costs of separate experts, the paper builds on Test-Time Adaptation (TTA) for model merging (e.g., AdaMerging), which dynamically optimizes layer-wise blending coefficients on-device via unsupervised Shannon entropy minimization. 

The authors identify a critical vulnerability: under physical test-time input corruptions (e.g., sensor noise,weather artifacts), standard TTA suffers from "Noise-Entropy Collapse" because the optimizer overfits the high-frequency transductive noise, leading to highly jagged blending profiles that degrade out-of-distribution (OOD) generalization. 

To resolve this, the paper proposes **FlatMerge**, a dual-regularization TTA model merging framework:
1. **Subspace-Constrained Blending (PolyMerge):** Constrains blending coefficients to a quadratic polynomial of normalized layer depth ($d=2$), mathematically filtering out high-frequency spatial variation across layers.
2. **Zeroth-Order Flatness-Aware Randomized Smoothing (ZO-FlatMerge):** Optimizes the compact polynomial parameters using a gradient-free, zeroth-order randomized smoothing gradient estimator. This actively guides the adaptation toward flat entropy valleys to stabilize against low-frequency transductive drift.

Crucially, because ZO-FlatMerge is gradient-free and operates in a highly compact coefficient space, it completely bypasses weight-space backpropagation and requires **exactly zero activation memory caching** (0 MB peak adaptation memory), making it highly compatible with edge-device SRAM limitations.

The method is evaluated inside a continuous, multi-seed numerical simulation of a 12-layer Vision Transformer (ViT-B/32) weight-merging landscape and is validated physically on real MLP and CNN architectures fine-tuned on MNIST, FashionMNIST, and KMNIST.

---

## Strengths and Weaknesses

### Strengths
1. **Realistic and Critical Problem Focus:** The paper identifies and addresses a highly practical deployment issue—namely, the impact of physical test-time sensor and channel noise on adaptive model-merging algorithms.
2. **Hardware-Centric Optimization Design:** Bypassing backpropagation and activation caching via a zeroth-order randomized smoothing formulation over a compact 12-parameter polynomial subspace is a highly creative, hardware-friendly design. It aligns perfectly with the strict memory and computing limits of edge accelerators.
3. **Rigorous and Transparent Hardware Profiling:** Section 3.5 provides an exceptional, highly transparent analysis of hardware trade-offs (static memory overhead vs. dynamic activation caching and step latency). The discussion of the DRAM weight-reconstruction bottleneck and the proposed asynchronous adaptation mitigation are highly valuable for practical engineering.
4. **Strong Statistical Setup for Simulations:** The simulation-based main results are repeated across 15 independent random seeds with detailed standard deviations, ensuring high statistical reliability in those environments.
5. **Excellent Presentation Quality:** The paper is exceptionally well-written, with highly structured narratives, clear and correct mathematical equations, and informative, beautiful figures.

### Weaknesses
1. **The "Task Arithmetic" Paradox on Physical Weights:** In the physical MLP and CNN experiments (Tables 3 & 4), the proposed FlatMerge method is consistently and significantly outperformed by the simple, static, zero-overhead baseline of **Task Arithmetic** under clean, moderate, and heavy noise. For example, under moderate noise ($\gamma = 1.0$), Task Arithmetic beats ZO-FlatMerge by **$14.63\%$ absolute** on MLP and by **$11.47\%$ absolute** on CNN. Since Task Arithmetic requires zero dynamic updates, zero activation memory, and zero weight-reconstruction overhead, its superior performance raises severe questions about the actual practical utility of FlatMerge (and TTA model merging in general) on real physical weights.
2. **Reliance on Simulated Main Results and Gaps:** The primary results of the paper (Tables 1 & 2, Figures 2, 3, 4, 5, 7) are evaluated on a continuous numerical simulation of a Vision Transformer rather than actual Vision Transformer weights. Furthermore, in Table 2, the Task Arithmetic baseline's accuracy remains perfectly constant ($84.44\% \pm 0.00$) as noise increases from $\gamma = 1.0$ to $3.0$. This indicates that the simulation only corrupts the unlabeled adaptation batch during TTA but evaluates on clean test data. This is an artificial setup, as real edge systems encounter persistent noise during both adaptation and inference.
3. **Discrepancies Between Simulated and Real Behavior:** In Table 1, PolyMerge ($d=2$) is highly robust and outperforms Task Arithmetic. However, in Table 4, PolyMerge ($d=2$) catastrophically collapses on real CNN weights, dropping to a near-random **$14.27\%$** under clean data. This massive discrepancy suggests that the simulated loss landscapes do not capture real deep CNN behaviors, undermining the reliability of the paper's simulation-based conclusions.
4. **Lack of Statistical Rigor in Physical Validation:** Unlike the simulated results, the physical MLP and CNN results (Tables 3 & 4) appear to be single-run, deterministic results with no standard deviations, confidence intervals, or multi-seed trials. Reporting these single runs makes it impossible to determine if the observed physical behaviors are statistically significant.
5. **Missing Baselines in Physical Validation:** Prominent baselines like RegCalMerge are missing from the physical validations. Additionally, PolyMerge is missing from the MLP validations.
6. **Unevaluated Theoretical Formulations:** The authors introduce a mathematically elegant "Adaptive Perturbation Radius" formulation in Eq. 8, but explicitly state that they "leave the empirical exploration of this adaptive formulation to future work," leaving a core component of their conceptual robustness framework completely unevaluated.

---

## Evaluation Criteria Ratings

### Soundness
* **Rating: Fair**
* **Justification:** While the mathematical formulations of the polynomial subspace projection and zeroth-order randomized smoothing are sound, the central claims regarding the practical utility of FlatMerge are not supported by the physical experiments on actual neural networks. In almost all realistic physical settings (clean, moderate, and heavy noise), ZO-FlatMerge is significantly outperformed by the simple static Task Arithmetic baseline. Furthermore, the main results rely on an artificial simulation environment where test-time noise is only transductive and is not applied to the final test-set evaluation.

### Presentation
* **Rating: Excellent**
* **Justification:** The paper is exceptionally clear, logical, and beautifully structured. The mathematical equations are precise and the figures are highly professional, detailed, and informative. The authors are highly transparent about their limitations and hardware profiling benchmarks, which is highly commendable.

### Significance
* **Rating: Fair**
* **Justification:** The significance of the paper is limited by the "Task Arithmetic" Paradox. Since a zero-overhead, zero-compute static baseline outperforms the proposed method by up to $15\%$ in realistic clean and moderate noise regimes, the practical incentive to deploy FlatMerge on actual edge accelerators is currently weak. However, the concept of applying Zeroth-Order SAM inside compact hyperparameter spaces to bypass backpropagation could be significant if it is shown to yield benefits on larger, pre-trained architectures.

### Originality
* **Rating: Good**
* **Justification:** The combination of a spatial polynomial filter (PolyMerge) and zeroth-order randomized smoothing (Sharpness-Aware Minimization) to find flat entropy valleys in a compact hyperparameter space is highly creative and well-motivated. The authors clearly distinguish their method from prior works like AdaMerging, RegCalMerge, PolyMerge, and standard weight-space SAM.

---

## Overall Recommendation

* **Recommendation: 3: Weak reject**
* **Justification:** 
  This paper has clear merits: a highly practical problem focus, a very creative and resource-minded gradient-free mathematical formulation, exceptional presentation and writing, and highly transparent hardware profiling analysis. 

  However, the paper suffers from critical empirical weaknesses that outweigh these merits in its current form:
  1. **The Performance Paradox:** On actual, physical MLP and CNN weights, ZO-FlatMerge is consistently and significantly outperformed by the zero-overhead, zero-compute Task Arithmetic baseline under clean, moderate, and heavy noise.
  2. **Simulation-to-Real Gap:** The primary results rely heavily on simulated loss landscapes of a Vision Transformer, which do not reflect physical deep learning behaviors (as seen by the catastrophic collapse of PolyMerge on real CNNs compared to its high performance in simulation).
  3. **Artificial Noise Setup in Simulation:** In the simulation robustness sweep (Table 2), the final evaluation is performed on clean data rather than noisy data, which is an artificial setup that does not reflect real-world test-time corruptions.
  4. **Lack of Statistical Rigor on Real Models:** The physical validations do not report standard deviations, confidence intervals, or multi-seed trials, unlike the simulated experiments.

  To be suitable for publication, the authors must address these empirical gaps. Specifically, they need to:
  - Demonstrate that ZO-FlatMerge can actually outperform Task Arithmetic on high-capacity models (e.g., physical CLIP Vision Transformers fine-tuned on downstream tasks) under realistic clean and moderate-noise settings.
  - Run the physical validation experiments across multiple random seeds and report standard deviations and confidence intervals.
  - Include missing baselines (e.g., RegCalMerge) in the physical evaluations.
  - Evaluate the adapted models on noisy test sets rather than clean test sets in the simulation to align with realistic physical deployments.
