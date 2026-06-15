# Evaluation Component 1: Paper Summary

## 1. Main Topic and Scope
The submission addresses the problem of **representation scale mismatches** in training-free parameter-space **model merging** (e.g., combining multiple task-specific expert neural networks fine-tuned from a shared pretrained base model into a single multi-task model). The paper focuses on resolving "task dominance"—where tasks with larger parameter updates or layers with disproportionate magnitudes overshadow others during simple linear averaging (like Task Arithmetic). It operates within the paradigm of lightweight, training-free, and data-free model merging, proposing two main element-wise scaling frameworks to establish balanced, isotropic update directions at the layer level.

## 2. Core Technical Approach
The paper introduces two layer-wise scaling techniques:
1. **Standard-Deviation Scaling (SD-Scale):** Normalizes each task vector layer-wise to unit standard deviation (which is translation-invariant but susceptible to instability on small, low-variance tensors like biases) and then scales by the average of the original standard deviations.
2. **Root-Mean-Square Scaling (RMS-Scale):** Normalizes each task vector layer-wise to unit root-mean-square (RMS), establishing a balanced, isotropic update direction. It is mathematically stable and non-translation-invariant. The averaged normalized update is then rescaled by the average of the original task-wise RMS scales.
   - **Primary Practical Recommendation:** *Weight-Only Scaling*, which applies normalization and calibration exclusively to high-dimensional weights while using standard linear averaging for low-dimensional biases.
3. **Parameter-Free RMS-Scale (PF-RMS):** Addresses "shrinkage" caused by parameter conflicts and partial task orthogonality in high-dimensional spaces. PF-RMS dynamically calibrates the scale at each layer by inverting the layer-wise alignment ratio $\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$, thus requiring no validation tuning.
   - It incorporates a **Clipping Safeguard** $\gamma(K) = C \cdot \sqrt{K}$ (typically $C = 1.2$ or $1.5$) to prevent over-amplification in extreme conflict scenarios ($\alpha^l \to 0$).

The authors also suggest:
- **Channel-wise RMS-Scale (CW-RMS):** Performing partition-level scaling at the head or channel granularity.
- **Ties-RMS-Scale / PF-Ties-RMS:** Combining coordinate-wise sign-conflict resolution (from Ties-Merging) with layer-wise scale calibration.

## 3. Key Findings
- **Rigorous Statistical Evaluation (CNN):** On a multi-task image classification benchmark (MNIST, FashionMNIST, KMNIST) fine-tuned with uncoordinated downstream schedules (different epochs and learning rates) across 3 seeds:
  - Tuned SD-Scale and RMS-Scale achieve **73.23%** and **73.22%** average accuracy, respectively, slightly exceeding standard Task Arithmetic (72.50%) and Ties-Merging (71.77%), while matching/exceeding SVD Isotropic Merging (73.13%).
  - Out-of-the-box un-tuned PF-RMS achieves **72.23%**, outperforming un-tuned Task Arithmetic (71.68%) and un-tuned Ties-Merging (71.81%) without requiring any validation dataset.
  - SVD Isotropic is computationally expensive, and AdaMerging (active test-time entropy minimization) is unstable under heterogeneous schedules (achieving only 62.79%).
- **High-Dimensional Verification (CLIP ViT-B/32):** Real-weight merging experiments on 36 projection layers from OpenAI's CLIP visual encoder verify that RMS-Scale achieves identical activation cosine alignment to SVD Isotropic Merging (57.74%) but runs **over 100x faster** (5.67ms vs. 571.92ms per layer).
- **Ablation Validity:** Combining normalization and scale calibration is shown to be essential. Omitting either component results in severe performance degradation (RMS-Norm gets 19.23%; RMS-Calib gets 53.20%).

## 4. Explicitly Claimed Contributions
1. **Identification of Scale Mismatch:** Highlights the representation scale mismatch in task vector merging under uncoordinated fine-tuning schedules.
2. **Proposed Methods (SD-Scale, RMS-Scale, PF-RMS):** Introduces training-free and parameter-free scale calibration methods that operate in linear time $O(K \cdot N)$.
3. **Mathematical Proof of Frobenius Parity:** Formally demonstrates that layer-wise RMS normalization is mathematically equivalent to parameter-count-scaled Frobenius-norm normalization on matrix layers.
4. **Clipped Inversion Safeguard:** Derives a dynamic, task-pool-dependent clipping threshold $\gamma(K)$ to safeguard against division-by-zero or noise-amplification in extreme task-conflict scenarios.
5. **High-Dimensional Real-Weight Evaluation:** Validates the linear complexity and SVD alignment parity of the proposed methods on actual OpenAI CLIP ViT-B/32 weight matrices.
6. **Granularity & Hybrid Extensions:** Introduces channel-wise scaling (CW-RMS) and a hybrid pipeline combining parameter sign-conflict resolution with scale calibration (Ties-RMS-Scale).
