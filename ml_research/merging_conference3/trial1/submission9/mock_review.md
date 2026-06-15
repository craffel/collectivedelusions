# Mock Review of "Root-Mean-Square Scaling: Unifying Model Merging via Minimalist Scale Calibration"

## Overall Recommendation
* **Score:** 5: Accept (Technically solid paper, with high impact on at least one sub-area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.)
* **Soundness:** Excellent (4/4)
* **Presentation:** Excellent (4/4)
* **Significance:** Excellent (4/4)
* **Originality:** Excellent (4/4)

---

## 1. Summary of the Paper
This paper addresses **task interference** and **representation scale mismatches** in training-free parameter-space model merging. When independent expert models are fine-tuned under uncoordinated downstream schedules (with varying learning rates, epoch counts, or datasets), standard linear parameter averaging (such as Task Arithmetic) often fails. This is because tasks with larger parameter updates or layers with disproportionate magnitudes dominate the merged model's representations, causing catastrophic performance degradation on weaker tasks. Recent works have turned to increasingly complex, high-overhead approaches, such as SVD-based isotropic singular value balancing (e.g., SAIM, OrthoMerge) or active test-time optimization via gradient descent (e.g., AdaMerging, SyMerge).

In response, this paper champions a **minimalist philosophy** (under the principle of Occam's Razor) and introduces two training-free, layer-wise scaling techniques:
1. **Standard-Deviation Scaling (SD-Scale):** Normalizes task vectors layer-wise to unit standard deviation to establish balanced directional contributions, and rescales the average normalized update by the average original standard deviation.
2. **Root-Mean-Square Scaling (RMS-Scale):** A mathematically stable, non-translation-invariant alternative to SD-Scale that avoids standard deviation's translation-invariance vulnerability on small, low-variance tensors (such as bias parameters) where subtracting the mean update can lead to division by zero. It normalizes task vectors layer-wise to unit RMS and rescales them by the average original RMS.

Furthermore, the authors introduce **Parameter-Free RMS-Scale (PF-RMS)**, which analytically counteracts the natural shrinkage of merged updates in high dimensions by inverting the layer-wise alignment ratio:
$$\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$$
where $\alpha^l$ represents the layer-wise alignment ratio. This completely removes any post-hoc grid search or disjoint validation data requirements, providing robust, out-of-the-box model merging.

The paper evaluates the proposed methods on a 3-seed multi-task image classification benchmark (MNIST, FashionMNIST, KMNIST) with uncoordinated fine-tuning schedules, demonstrating that tuned RMS-Scale (73.22%) and SD-Scale (73.23%) match or exceed SVD Isotropic Merging (73.13%) and outclass AdaMerging (62.79%). To bridge the evaluation scale gap, the authors also perform a physical model-merging evaluation on **36 high-dimensional projection weight layers of the official OpenAI CLIP ViT-B/32 visual encoder**, proving that RMS-Scale achieves the exact same optimal activation cosine alignment (57.74%) and isotropic balance (0.15% std) as SVD Isotropic, but delivers a spectacular **100x wall-clock speedup** (5.67ms vs. 571.92ms per layer).

---

## 2. Strengths

* **Conceptual Elegance and Occam's Razor:** The paper makes a brilliant, refreshing argument against the recent trend of "complexity escalation" and "heuristic bloat" in modern model-merging literature. It demonstrates that highly complex, multi-stage, or SVD-heavy pipelines can be matched or exceeded by simple, two-line, element-wise scaling.
* **Outstanding Computational Efficiency:** By operating strictly in linear time $O(K \cdot N)$ where $K$ is the number of tasks and $N$ is the parameter count, RMS-Scale completely bypasses the cubic $O(d^3)$ bottleneck of SVD. The physical wall-clock comparison on actual CLIP ViT-B/32 layers (averaging 5.67ms vs. 571.92ms for SVD Isotropic) provides compelling, concrete proof of this advantage, unlocking isotropic scale balancing for multi-billion parameter foundation models.
* **Rigorous Mathematical Grounding:** The paper includes a beautiful proof showing that layer-wise RMS normalization on matrices is mathematically equivalent to parameter-count-scaled Frobenius-norm normalization:
$$\hat{W}_k^l = \sqrt{N^l} \cdot \frac{W_k^l}{\|W_k^l\|_F}$$
This elegantly links the element-wise heuristic of RMS-Scale directly to classical Riemannian manifold alignments.
* **Excellent Scientific Rigor and Transparency:** The empirical evaluation on the SimpleCNN benchmark is exceptionally rigorous, utilizing 3 independent seeds on completely separate, disjoint validation and test splits to prevent any target leakage. All baselines are carefully validation-tuned, and the authors are commendably honest about the multi-task learning trade-offs on the dominant task (FashionMNIST).
* **The Breakthrough of Parameter-Free Calibration (PF-RMS):** Deriving the analytical shrinkage correction factor $\lambda^l = 1 / \alpha^l$ where $\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$ is a major conceptual contribution. It elegantly explains why task vectors shrink when averaged (due to high-dimensional parameter conflicts and partial orthogonality) and solves it dynamically layer-by-layer, outperforming default un-tuned baselines.
* **Exceptional Engineering Completeness (LoRA and Bounded Memory):** 
  - To handle Low-Rank Adapters (LoRA) without memory explosion on foundation models, the authors propose **sequential layer-by-layer Safetensors streaming**, which bounds the peak memory footprint to <150MB.
  - To retain the modular, swap-at-runtime, parameter-efficient serving benefits of LoRA, the authors derive a **post-merging SVD re-factorization** step ($B_{\text{merged}}^l = U_r \Sigma_r$, $A_{\text{merged}}^l = V_r^T$).
  - For factorized scaling, the authors prove that independent scaling of factors is mathematically unsound and show how to scale one factor ($B_k^l \leftarrow \lambda^l B_k^l$) to preserve the exact product's RMS calibration.
* **Generalization Safeguards:** The authors formalize the task-scaling relationship of the safeguard clipping threshold dynamically as $\gamma(K) = C \cdot \sqrt{K}$, preventing premature clipping for larger task pools ($K \ge 4$) while securing the method against adversarial division-by-zero or noise-amplification scenarios.
* **Hybrid Merging (PF-Ties-RMS):** The authors formulate a hybrid variant that resolves coordinate-wise sign conflicts first (via Ties-Merging's pruning and sign-election) before applying layer-wise scale calibration. This ensures that only coherent, sign-aligned signals are amplified rather than conflicting noise.
* **Deep Ablations and Visualizations:** The paper includes outstanding secondary investigations, such as:
  1. *Channel-wise structural partitioning (CW-RMS)* mapping to Transformer attention-head splits.
  2. *Alternative scale estimators* (Harmonic, Geometric, and Maximum means), showing that the Harmonic mean is highly effective at damping extreme outlier updates.
  3. *Sensitivity analyses* of the stability constant $\epsilon$ and the clipping safeguard threshold $\gamma \in [1.5, 3.0]$.
  4. *Visualizations of layer-wise alignment ratios (Figure 2)*, showing theoretical convergence to the high-dimensional orthogonal limit ($1/\sqrt{K}$).

---

## 3. Areas for Improvement (Minor Suggestions & Gaps)

While the paper is highly complete, scientifically rigorous, and ready for publication, we identify a few minor conceptual areas that the authors should address to further polish the manuscript:

### 1. The Evaluation Scale Gap for Classification Accuracy
While the activation-space alignment analysis is conducted on real-world CLIP ViT-B/32 weights, the end-to-end multi-task classification accuracy evaluation remains restricted to the SimpleCNN benchmark on grayscale MNIST datasets. Grayscale MNIST-like datasets are extremely low-resolution (28x28) and are considered toy tasks in modern deep learning (2026). Report of the final end-to-end multi-task classification accuracy of a merged CLIP model on standard downstream vision benchmarks (e.g., Stanford Cars, DTD, EuroSAT, SUN397, etc.) would make the paper's empirical claims much more powerful. Although the authors have discussed this in the Limitations section, explicitly acknowledging this evaluation scale gap as a target for future library releases is critical.

### 2. Practical Irrelevance of the Bias Translation-Invariance Argument
The paper devotes significant theoretical and mathematical space to the "translation-invariance vulnerability" of standard deviation on bias parameters, which serves as the primary justification for introducing RMS-Scale. In practice, however, bias parameters account for $<0.03\%$ of modern networks and are frequently omitted entirely in modern Transformer architectures (such as LLaMA). A standard stability constant $\epsilon = 10^{-8}$ completely prevents division-by-zero, and the authors' own "bias-free scaling" ablation shows that leaving biases unnormalized achieves identical performance (73.22%). De-emphasizing this minor theoretical risk would streamline the presentation and keep the focus on the high-dimensional weight matrices.

---

## 4. Questions for the Authors

1. **Alternative Averages under Explicit Sign-Conflict Resolution:**
   How does PF-RMS interact with explicit sign-conflict resolution techniques like Ties-Merging in practice? Have you evaluated a hybrid *Ties-RMS-Scale* where sign conflicts are resolved first before applying RMS-Scale? If so, does it show superior performance over standard Ties-Merging?
2. **Active Optimization Initialization:**
   In Section 4.3, you suggest that PF-RMS could serve as a highly stable, mathematically grounded initialization for parameterized merging methods (like AdaMerging). Have you experimented with or do you plan to evaluate this hybrid approach to see if it prevents optimization collapses?
3. **Harmonic Mean as Default:**
   Given that the Harmonic Mean PF-RMS outperforms the Arithmetic Mean by +0.40% on average and naturally dampens extreme outlier updates, do you plan to make it the default scale estimator in future releases of your merging framework?
