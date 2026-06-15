# Mock Review of "Root-Mean-Square Scaling: Unifying Model Merging via Minimalist Scale Calibration"

## Overall Recommendation
* **Score:** 5: Accept
* **Soundness:** Excellent (4/4)
* **Presentation:** Excellent (4/4)
* **Significance:** Good / Excellent (3.5/4)
* **Originality:** Good / Excellent (3.5/4)

---

## 1. Summary of the Paper
This paper addresses **task interference** and **representation scale mismatches** in training-free parameter-space model merging. Standard linear parameter averaging (such as Task Arithmetic) suffers from severe task dominance because different experts adapt at vastly different scales (e.g., due to heterogeneous training schedules, learning rates, or data distributions), allowing the task with the largest parameter update to dominate the merged model. Recent methods have turned to increasingly complex, high-overhead approaches, such as SVD-based isotropic singular value balancing (e.g., SAIM) or active test-time optimization via gradient descent (e.g., AdaMerging, SyMerge).

In response, this paper champions a minimalist philosophy (under the principle of Occam's Razor) and introduces two training-free, layer-wise scaling techniques:
1. **Standard-Deviation Scaling (SD-Scale):** Normalizes task vectors layer-wise to unit standard deviation to establish balanced directional contributions, and rescales the average normalized update by the average original standard deviation.
2. **Root-Mean-Square Scaling (RMS-Scale):** A mathematically stable, non-translation-invariant alternative to SD-Scale that avoids standard deviation's translation-invariance vulnerability on small, low-variance tensors (such as bias parameters). It normalizes task vectors layer-wise to unit RMS and rescales them by the average original RMS.

Furthermore, the authors introduce **Parameter-Free RMS-Scale (PF-RMS)**, which analytically counteracts the natural shrinkage of merged updates in high dimensions by inverting the layer-wise alignment ratio $\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$, removing any hyperparameter tuning or disjoint validation requirements.

The paper evaluates the proposed methods on a 3-seed multi-task image classification benchmark (MNIST, FashionMNIST, KMNIST) with uncoordinated fine-tuning schedules, demonstrating that tuned RMS-Scale (73.22%) and SD-Scale (73.23%) match or exceed SVD Isotropic Merging (73.13%) and outclass AdaMerging (62.79%). To bridge the evaluation scale gap, the authors also perform a physical model-merging evaluation on **36 high-dimensional projection weight layers of the official OpenAI CLIP ViT-B/32 visual encoder**, proving that RMS-Scale achieves the exact same optimal activation cosine alignment (57.74%) and isotropic balance (0.15% std) as SVD Isotropic, but delivers a spectacular **100x wall-clock speedup** (5.67ms vs. 571.92ms per layer).

---

## 2. Strengths

* **Conceptual Elegance and Occam's Razor:** The paper makes a brilliant, refreshing argument against the "complexity escalation" and "heuristic bloat" in modern model-merging literature. It demonstrates that highly complex, multi-stage, or SVD-heavy pipelines can be matched or exceeded by simple, two-line, element-wise scaling.
* **Outstanding Computational Efficiency:** By operating strictly in linear time $O(K \cdot N)$ where $K$ is the number of tasks and $N$ is the parameter count, RMS-Scale bypasses the cubic $O(d^3)$ bottleneck of SVD. The physical wall-clock comparison on actual CLIP ViT-B/32 layers (averaging 5.67ms vs. 571.92ms for SVD Isotropic) provides compelling, concrete proof of this advantage.
* **Rigorous Mathematical Grounding:** The paper includes a beautiful proof showing that layer-wise RMS normalization on matrices is mathematically equivalent to parameter-count-scaled Frobenius-norm normalization ($\hat{W}_k^l = \sqrt{N^l} \cdot W_k^l / \|W_k^l\|_F$). This links the element-wise heuristic directly to classical Riemannian manifold alignments.
* **Excellent Scientific Rigor and Transparency:** The empirical evaluation on the SimpleCNN benchmark is exceptionally rigorous, utilizing 3 independent seeds on completely separate, disjoint validation and test splits to prevent any target leakage. All baselines are carefully validation-tuned, and the authors are commendably honest about the multi-task trade-off on the dominant task (FashionMNIST).
* **The Breakthrough of Parameter-Free Calibration (PF-RMS):** Deriving the analytical shrinkage correction factor $\lambda^l = 1 / \alpha^l$ where $\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$ is a major conceptual contribution. It elegantly explains why task vectors shrink when averaged (due to high-dimensional parameter conflicts and partial orthogonality) and solves it dynamically layer-by-layer, outperforming default un-tuned baselines.
* **Deep Ablations and Visualizations:** The paper includes outstanding secondary investigations, such as:
  1. *Channel-wise structural partitioning (CW-RMS)* mapping to Transformer attention-head splits.
  2. *Alternative scale estimators* (Geometric and Harmonic means), showing that the Harmonic mean is highly effective at damping extreme outlier updates.
  3. *Sensitivity analyses* of the stability constant $\epsilon$ and the clipping safeguard threshold $\gamma \in [1.5, 3.0]$.
  4. *Visualizations of layer-wise alignment ratios (Figure 2)*, showing theoretical convergence to the high-dimensional orthogonal limit ($1/\sqrt{K}$).

---

## 3. Areas for Improvement (Minor Suggestions)

The paper has successfully addressed the major weaknesses identified in prior drafts (such as test-set leakage, limited evaluation scale, and bias translation-variance overemphasis). However, a few minor areas of improvement remain to polish the manuscript for final publication:

1. **Bridge the Evaluation Scale Gap for Classification Accuracy:**
   While the activation-alignment analysis is conducted on real-world CLIP ViT-B/32 weights, the end-to-end multi-task classification accuracy evaluation remains restricted to the SimpleCNN benchmark on grayscale MNIST datasets. Although the authors have added a detailed Limitations section discussing this, the paper would be even stronger if they could eventually report the final end-to-end multi-task accuracy of a merged CLIP model on standard downstream vision benchmarks (e.g., Stanford Cars, DTD, EuroSAT, SUN397, etc.).
2. **Elaborate on the Low-Memory LoRA Implementation:**
   In Section 3.7, the authors discuss the memory overhead of reconstructing large projection matrices for merging and recommend a sequential, layer-by-layer merging workflow. It would be helpful to explicitly mention if there are existing software packages or scripts that can be used to easily implement this sequential weight generator loop (e.g., using Hugging Face's PEFT or SafeTensors libraries), which would increase the practical utility for deep learning practitioners.
3. **Slight Absolute Gains on SimpleCNN:**
   On the SimpleCNN benchmark, the absolute performance improvements of tuned RMS-Scale over tuned Task Arithmetic (73.22% vs. 72.50%) are relatively modest. The authors should make sure to emphasize that the primary strength of RMS-Scale is not just the tuned performance, but its *parameter-free* variant (PF-RMS) which delivers highly competitive results (72.23%) out-of-the-box, completely bypassing the need for disjoint validation sets or expensive post-hoc hyperparameter searches.

---

## 4. Questions for the Authors

1. **Geometric Consistency on Diverse Tensor Dimensions:**
   Your Frobenius Equivalence proof in Section 3.6 holds elegantly for any weight matrix shape. Could you discuss how this equivalence manifests geometrically for extremely non-square matrices (such as attention query/key projections in Transformers) versus 1D bias vectors?
2. **Active Optimization Initialization:**
   In Section 4.3, you suggest that PF-RMS could serve as a highly stable, mathematically grounded initialization for parameterized merging methods (like AdaMerging). Have you experimented with or do you plan to evaluate this hybrid approach to see if it prevents optimization collapses?
3. **LoRA Factorized Scaling:**
   In Section 3.7, you mention that scaling the low-rank updates can be achieved by applying the layer-wise correction factor $1/\alpha^l$ to one of the factors (e.g., scaling $B_k^l \leftarrow \lambda^l B_k^l$). Is this factorized scaling mathematically identical to full weight reconstruction scaling under homogeneous LoRA ranks?
