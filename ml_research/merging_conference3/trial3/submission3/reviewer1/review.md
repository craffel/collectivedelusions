# Peer Review

## Summary of the Paper
This paper addresses the critical challenge of deploying deep neural networks on resource-constrained edge devices in realistic, noisy physical environments. It focuses on the paradigm of unsupervised **test-time model merging (TTA)**, where fine-tuned task-specific expert weights are combined, and their layer-wise blending coefficients are dynamically optimized on-device using prediction entropy minimization on unlabeled local target-task input streams. 

The authors identify and analyze a devastating failure mode they term **Noise-Entropy Collapse** (a manifestation of the **Overfitting-Optimizer Paradox**): standard first-order optimizers easily minimize entropy on noisy test batches by overfitting high-frequency transductive noise, causing catastrophic out-of-distribution performance degradation and, on physical weights, inducing a "constant-prediction collapse."

To resolve this, the authors propose **FlatMerge**, a robust, backpropagation-free dual-regularization framework that:
1. **Subspace-Constrained Blending (PolyMerge):** Projects and restricts the blending coefficients to a smooth low-degree polynomial of normalized depth, forcing depth-wise smoothness and reducing parameter dimensionality by over 90%.
2. **Zeroth-Order Flatness-Aware Randomized Smoothing (ZO-FlatMerge):** Optimizes a smoothed entropy objective over randomized perturbations, guiding the adaptation toward flat entropy valleys that are robust to noise, using only forward passes.

Through extensive continuous simulated environments and physical validations on actual MLP and 5-layer CNN weights fine-tuned on MNIST, FashionMNIST, and KMNIST, the authors show that FlatMerge successfully prevents transductive collapse, achieving state-of-the-art robust accuracies in simulation and completely avoiding the representation collapse that standard first-order TTA suffers from on physical deep learning architectures.

---

## Strengths and Weaknesses

### Strengths
* **Highly Realistic Edge Motivation:** The paper addresses a major, frequently overlooked vulnerability of adaptive model merging at deployment—specifically, how real-world environmental sensor noise, defocus blur, weather artifacts, and compression distortions corrupt test-time adaptation.
* **Pragmatic, Backpropagation-Free Architecture:** FlatMerge completely eliminates backpropagation and intermediate activation memory caching during adaptation. Peak SRAM adaptation memory is exactly $0.00\text{ MB}$ (identical to standard forward inference), which resolves a critical hardware bottleneck for edge accelerators.
* **Elegant Theoretical Synergy:** Combining low-degree polynomial depth parameterization (spatial filtering of high-frequency layer-wise variations) with zeroth-order flatness-aware randomized smoothing (temporal stabilization of learned trajectories) is mathematically elegant and highly appropriate.
* **High Statistical Rigor in Simulations:** Evaluating the continuous simulation environments across **15 independent random seeds** and reporting thorough standard deviations demonstrates a highly disciplined and commendable experimental methodology.
* **Physical Validation on Real Weights:** Anchoring the simulated findings on live MLP and CNN models fine-tuned on real datasets successfully identifies the Overfitting-Optimizer Paradox and constant-prediction collapse on physical weights, validating the core theoretical hypotheses on actual neural network manifolds.

### Weaknesses
* **Critical Bibliographic Omissions (Broken Citations):** The paper suffers from severe bibliographic errors where foundational works cited in the text are completely omitted from the bibliography (`references.bib`). Specifically, **PolyMerge** (`\cite{polymerge}`) and **SAM** (`\cite{sam}`) are cited extensively but are completely missing from the `.bib` file, causing severe undefined citation warnings. For a scholarly paper, this is a critical oversight.
* **The Optimization Utility Paradox on Physical Models:** In physical validations (MLP and CNN), FlatMerge consistently performs **significantly worse than the simple static, uniform Task Arithmetic baseline** across almost all clean and moderate noise conditions (e.g., on the CNN, Task Arithmetic achieves $58.20\%$ clean and $40.67\%$ moderate noise, whereas FlatMerge achieves $48.57\%$ clean and $29.20\%$ moderate noise—a degradation of over $9.6\%$ and $11.4\%$ absolute). This raises serious questions about the practical utility of deploying a latency-heavy, complex optimization loop on edge devices if a simple static merge is 10% more accurate.
* **DRAM Bandwidth Bottleneck:** Evaluating the smoothed objective requires loading the base weights and task vectors from DRAM and reconstructing the merged weights $2 B_{\text{zo}} = 20$ times per adaptation step. For an 85M parameter ViT model (FP32), this translates to **$40.8\text{ GB}$ of DRAM transactions per step**, which is extremely energy- and latency-heavy on edge hardware, as evidenced by the reported $3.73\times$ latency penalty.
* **Static Weight Memory Storage Inflation:** While saving activation memory, FlatMerge increases **static weight memory by $1.5\times$** (from $1360.28\text{ MB}$ to $2040.42\text{ MB}$) to store base weights, task vectors, and active merged weights in memory simultaneously. This trade-off is not sufficiently highlighted or discussed in the context of extreme memory-constrained edge accelerators (such as microcontrollers).

---

## Soundness
**Rating: Good**

**Justification:** The core mathematical formulation of FlatMerge is highly sound and technically correct. Restricting layer coefficients to a polynomial subspace is an effective way to filter spatial noise, and zeroth-order randomized smoothing is a mathematically established tool to seek flat minima without backpropagation. The authors are honest about empirical limitations and conduct a rigorous hardware-profiling benchmark on standard processors to characterize latency and static memory overheads. 

However, the soundness rating is capped at "Good" due to the significant trade-offs (heavy DRAM bandwidth overhead, increased static memory) and the fact that the actual optimization objective (unsupervised prediction entropy minimization) underperforms static merging on physical models.

---

## Presentation
**Rating: Fair**

**Justification:** The overall writing style of the paper is Excellent—it is extremely clear, professional, well-structured, and easy to follow. The figures (Figures 1, 2, and 6) are exceptionally high quality and highly illustrative, particularly Figure 6's layer-wise blending trajectories.

However, the presentation rating is downgraded to "Fair" due to the **severe bibliographic omissions in `references.bib`**:
1. **PolyMerge (`\cite{polymerge}`)** is cited in Section 1, Section 2.3, Section 3.1, Section 3.3, and the experiment sections, but is completely missing from the `.bib` file. This is particularly critical because FlatMerge adopts its exact mathematical formulation (Equation 3) as its first core regularization pillar.
2. **SAM (`\cite{sam}`)** is cited in Section 2.4 and Section 3.4 as the foundation of FlatMerge's flatness-aware optimization, but is completely missing from the `.bib` file.
Compiling the manuscript in any standard LaTeX environment yields critical undefined citation warnings (producing double question marks `??` in the text), which severely compromises its scholarly and presentation quality.

---

## Significance
**Rating: Good**

**Justification:** The significance of this paper is high for large-scale, pre-trained architectures (like Vision Transformers or Large Language Models) where test-time coefficient optimization is known to significantly outperform Task Arithmetic, and where backpropagation is computationally prohibitive. FlatMerge provides an elegant, memory-efficient way to adapt these massive models without activation caching or backpropagation.

However, its significance is lower for small-scale physical models, where the unsupervised test-time optimization fails to match simple static merging (Task Arithmetic).

---

## Originality
**Rating: Good**

**Justification:** The individual components of the paper—namely polynomial depth constraints (PolyMerge) and flatness-aware optimization via randomized smoothing (SAM)—are adapted from prior work. However, the specific combination of these two regularizers in a highly compressed coefficient space to perform backpropagation-free, zeroth-order test-time model merging is highly creative, original, and pragmatic. It represents a very clever and practical application of flatness theory to edge robustness.

---

## Overall Recommendation
**Rating: 4 (Weak accept)**

**Justification:** This is a technically solid, highly pragmatic paper that addresses an important and realistic deployment challenge: the vulnerability of test-time model merging to environmental input noise. The mathematical formulations are clear and sound, and the dual-regularization framework (projecting coefficients to a polynomial subspace and optimizing for flatness via zeroth-order perturbations) is highly elegant and effective. The authors ground their claims with a thorough hardware profiling benchmark and validations on real physical MLP and CNN weights.

However, the paper has critical weaknesses that prevent an "Accept" rating at this stage:
1. **Critical bibliographic omissions:** The foundational citations `polymerge` and `sam` are completely missing from the `references.bib` file, causing severe LaTeX compilation warnings. This must be corrected.
2. **The physical performance gap:** The authors must explicitly discuss and contextualize why their proposed optimization-based TTA (and TTA in general) consistently and significantly underperforms the static uniform Task Arithmetic baseline on physical models under clean, moderate, and heavy noise.
3. **Explicit hardware tradeoffs:** The paper should highlight the static weight memory storage trade-off (storing $K+2$ copies of model weights, causing a 1.5$\times$ memory inflation) as openly as it highlights the activation memory savings.

If the authors correct these bibliographic errors and address these empirical/hardware caveats in their revision, this paper will represent a very strong and valuable contribution to the model merging and edge intelligence communities.
