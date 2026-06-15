# 3. Soundness and Methodology Check

This section provides a rigorous critique of the technical soundness and methodology presented in the paper, highlighting both its areas of exceptional rigor and its key logical or practical limitations.

### 1. Areas of High Rigor and Methodological Strengths

- **Chronological Data Leakage Prevention:** The authors identify and rigorously resolve a subtle but critical chronological data leakage bottleneck in online centroid updates. By strictly enforcing that the routing and activation-blending of sample $x_b$ use only the centroids from the previous step ($\mu_{k, b-1}$), they ensure an honest, leakage-free evaluation of EPL-OCA (Eq. 6). This is a level of methodological honesty rarely seen in streaming papers.
- **Normalized Shannon Entropy (Eq. 4):** The mathematical formulation of Normalized Shannon Entropy is clean and highly rigorous. It addresses the inherent bias of Shannon entropy toward smaller vocabularies under heterogeneous registries in a mathematically sound, scale-invariant manner.
- **Activation Divergence and Systems FLOP Complexity:** The systems-level FLOP complexity calculation is exceptionally detailed and grounded in physical architecture. The authors identify exactly why activations diverge (at Layer 4 out of 12) and provide a mathematically rigorous FLOP complexity of $0.25 + 0.75K$ passes (Eq. 12).
- **Physical Wall-Clock Benchmarking:** The authors do not stop at theoretical complexity; they profile execution speed on physical hardware (single core AMD EPYC CPU) and analyze energy efficiency under SRAM/DRAM transfer costs on edge processors, providing an exceptionally solid engineering foundation for "The Pragmatist" view.

### 2. Critical Flaws, Contradictions, and Weaknesses

#### A. The "Calibration-Free" Contradiction in CG-EER
To resolve the "Entropy Calibration Discrepancy" and out-of-distribution (OOD) overconfidence on real ResNet-18 embeddings, the authors introduce **Centroid-Gated Entropy Routing (CG-EER)**. 
However, they define the gating mechanism as follows (Section 4.10):
> *"We apply an unsupervised threshold ($\delta \ge 0.7$) on the representation-space cosine similarity to each task's pre-computed centroid (obtained exactly like SPS-ZCA)."*

This introduces a severe logical contradiction:
1. The paper’s entire premise and core objective is **Zero-Shot Calibration-Free Model Merging**, which eliminates the dependency on offline, labeled calibration splits ($|\mathcal{C}_k|=64$) that bottlenecked frameworks like SPS-ZCA.
2. SPS-ZCA calculates its task centroids using **offline labeled calibration splits**.
3. If CG-EER depends on pre-computed centroids "obtained exactly like SPS-ZCA," then **CG-EER is NOT calibration-free or zero-shot**. It re-introduces the exact offline calibration data dependency that the paper claims to eliminate.

To maintain methodological integrity, the authors must address this:
- If CG-EER is to be truly calibration-free, the gating must utilize **unsupervised, on-the-fly centroids** (such as those discovered via EPL-OCA or online K-Means) rather than supervised pre-computed centroids.
- If pre-computed centroids are indeed required, CG-EER should be classified as a **semi-supervised or calibration-dependent hybrid method**, and its performance under truly calibration-free unsupervised centroids must be evaluated and reported. (The authors do re-classify CG-EER as hybrid in Section 4.10, but the overall presentation in the abstract and intro still tends to lump it under the "calibration-free" umbrella).

#### B. Total Collapse of the "Efficiency-First" Paradigm (EPL-OCA) on Real Features
The authors propose EPL-OCA as a major online ensembling paradigm designed to achieve $1.3\times$ amortized serving complexity. While it performs reasonably well in the highly orthogonal synthetic sandbox (61.62% for EPL-OCA Soft), it **completely collapses on real ResNet-18 embeddings**:
- **EPL-OCA Hard:** $27.45 \pm 1.34\%$ Joint Mean accuracy.
- **EPL-OCA Soft:** $31.52 \pm 1.37\%$ Joint Mean accuracy.
Given that static **Uniform Weight Merging** achieves **$31.66 \pm 0.91\%$** on these same features, EPL-OCA Soft is actually *worse* than or statistically equivalent to a simple uniform average of the expert weights, while EPL-OCA Hard is significantly worse.
- **Why this happens:** EPL-OCA relies entirely on the EER pseudo-labeler to direct online centroid updates. However, on real features, the pseudo-labeler is heavily corrupted by the Entropy Calibration Discrepancy (e.g., routing 75.2% of all samples to the overconfident MNIST expert). This corrupts the MNIST running centroid with SVHN and CIFAR-10 features, while other expert centroids are never updated, resulting in a total collapse of the centroid space.
- **The Consequence:** This is a major methodological limitation of EPL-OCA. It proves that the "Efficiency-First" online centroid adaptation paradigm is **completely non-functional on real features** under standard serving conditions without offline calibration data or pre-computed spatial anchors. The paper should state this limitation more directly.

#### C. Overlapping Class Label Namespace and Evaluation Bias
In both the synthetic sandbox and the real-world ResNet-18 experiments, all $K=4$ tasks have exactly $C=10$ classes, and their labels are represented by overlapping integers $\{0, \dots, 9\}$. 
- **The Issue:** Since the ensembling model's prediction is evaluated via a simple argmax match `(pred == y).item()`, if a sample from CIFAR-10 with class index 3 ("cat") is routed incorrectly to the MNIST expert, and the MNIST expert happens to predict class index 3 ("3"), the evaluation script will count this as a **correct** prediction.
- **The Bias:** While orthogonal representations in the synthetic sandbox and well-separated ResNet-18 features limit the probability of such cross-task prediction matches, they are still theoretically possible (occurring with a background probability of $\approx 10\%$). This introduces a slight optimistic bias in absolute accuracy values across all evaluated ensembling models, and represents an artifact of overlapping label spaces that should be explicitly discussed.

#### D. Artificial Simplification of the Synthetic Sandbox
The 192-dimensional multi-task representation sandbox relies on several highly simplified geometric and statistical assumptions:
1. **Subspace Orthogonality:** The 192-dimensional representation space is divided into $K=4$ orthogonal 48-dimensional task subspaces via QR decomposition.
2. **Class Orthogonality:** Within each task's 48-dimensional subspace, all 10 class prototypes are generated as mutually orthogonal unit vectors.
3. **Isotropic Gaussian Noise:** Representations are modeled using Gaussian distributions with calibrated noise scales.

While this allows precise mathematical control and isolates the **Representational Sparsity Paradox**, real-world deep neural network activation spaces do not exhibit perfect subspace or class-wise orthogonality. In real vision or language backbones, class manifolds are highly non-linear, non-isotropic, and have substantial topological overlap. The extreme orthogonality in the sandbox artificially amplifies the Representational Sparsity Paradox (and the resulting centroid jitter in EPL-OCA), making the synthetic baseline collapse look much worse than what might occur under smoother, correlated real-world representations.
