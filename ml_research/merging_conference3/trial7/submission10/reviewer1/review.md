# Peer Review

## Summary of the Paper
The paper introduces **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment), a training-free and computationally efficient dynamic model-merging framework designed for edge CPUs. It aims to solve the latency bottleneck of prior dynamic routing systems (such as Micro-Batch Homogenization) and the "routing paradox" where routers rely on late-stage features.

To accomplish this, the authors propose:
1. **Single-Pass Activation-Space Dynamic Blending (SPS):** A method to execute the shared base model and its expert adapters in a single parallel forward pass, blending activations sample-wise on-the-fly and keeping execution latency constant $O(1)$.
2. **Zero-Shot Centroid Alignment (ZCA) Routing:** Operating in the shared early-stage representation space (specifically Layer 3 CLS tokens), inputs are projected against pre-computed task centroids to compute dynamic routing coefficients, resolving the temporal dependency.
3. **Calibration & Robustness Modules:** To neutralize scale drift and asymmetric task manifold variance, the paper proposes Unit-Norm Calibration (UNC), Intra-Task Dispersion Calibration (IDC), and a low-dimensional diagonal Gaussian Mixture Model (GMM) Coordinate Density Estimator to reject out-of-distribution (OOD) queries prior to blending.

The method is evaluated via high-fidelity simulation and physical PyTorch profiling (on both Vision Transformer and GPT-2 models), as well as physical edge-hardware benchmarks on a Raspberry Pi 4 CPU.

---

## Strengths and Weaknesses

### Soundness
* **Rating: Fair**
* **Strengths:** The paper exhibits exemplary empirical soundness. The authors conduct detailed scaling sweeps and hyperparameter sensitivity audits ($\tau$, $\eta$, batch sizes, scale imbalances). They thoroughly validate their claims regarding capacity preservation when omitting adapters from early layers, and the physical Raspberry Pi 4 hardware profiling using custom ONNX Runtime CustomOps beautifully bridges the "reality gap" of analytical FLOP speedups.
* **Weaknesses (Grave Theoretical Underpinnings Gap):**
  While the empirical execution is commendable, the paper falls short of the rigorous mathematical standards expected of a top-tier machine learning conference. It relies heavily on empirical "discoveries" and heuristic calibrations rather than starting from first principles with a solid theoretical framework or proving mathematical guarantees.
  1. **Lack of Formal Proofs / Guarantees:** There is no formal mathematical guarantee proving that sample-wise activation-space blending (SPS) preserves task pathways without cross-expert activation interference or "activation bleeding" under realistic representation bounds. Under what geometric conditions on the representation manifolds does this training-free projection guarantee near-optimal recovery of the Expert Ceiling?
  2. **Heuristic Calibration Formulations:** The proposed calibration modules are mathematically descriptive but heuristically formulated. For example, dividing the routing similarity coordinates by the expected in-distribution similarity scale ($u'_{k,b} = u_{k,b}/s_k$) in IDC is an intuitive, hand-crafted scaling. There is no rigorous theoretical justification showing that this specific scaling factor optimizes decision boundaries or minimizes classification/routing error bounds under multi-task manifold assumptions.
  3. **Stated PAC Bound without Derivation:** In Section 4.8, the authors state a standard Probably Approximately Correct (PAC) sample complexity bound: $N = \mathcal{O}\left( \frac{K + \log(1/\delta)}{\epsilon^2} \right)$ to justify the sample efficiency of Supervised Head Fine-Tuning (SHFT). However, they merely copy this standard formula from general learning theory literature. They do not define the precise hypothesis class, loss function, or data-generation process associated with their specific low-dimensional coordinate routing space, nor do they provide a formal derivation or context-specific proof demonstrating that this bound holds in their setting.
  4. **Simplistic Diagonal Covariance GMM Assumption:** The GMM Shield assumes a diagonal covariance structure. While computationally efficient, there is no mathematical analysis analyzing how much statistical information is lost by ignoring off-diagonal covariance terms, especially when expert task coordinates are correlated.
  5. **No Theoretical Bounds on Activation Bleeding:** The authors qualitatively describe "activation bleeding" under overlapping task domains as a boundary condition. However, they do not formalize this relationship mathematically. A rigorous paper should analytically prove how performance degrades as a function of task overlap (e.g., using mutual information, Wasserstein distance, or Fisher Separability Criterion).

### Presentation
* **Rating: Excellent**
* **Strengths:** The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is consistent, and the figures are high-quality. The authors do an excellent job of describing their hardware-aware cost model, outlining their physical PyTorch variants, and providing compiled pseudo-code in Appendix A.

### Significance
* **Rating: Excellent**
* **Strengths:** The practical significance of this work is outstanding. Efficiently serving multiple task-specific adapters under strict hardware constraints is a crucial bottleneck for TinyML and on-device assistants. The framework is training-free, requires zero trainable parameters, and is directly deployable on sequential edge CPUs. The custom ONNX Runtime C++ CustomOp benchmarks on Raspberry Pi 4 CPU proving a physical 3.91$\times$ wall-clock speedup make this an actionable and highly valuable contribution to practitioners.

### Originality
* **Rating: Good**
* **Strengths:** The paper offers a creative, systems-ML co-designed combination of existing concepts. The idea of routing in early-layer representation spaces to resolve the temporal circular dependency of dynamic adapters is clever and elegant. Combining this with activation blending, scale-invariance (UNC), dispersion calibration (IDC), and low-dimensional GMM density estimation represents a comprehensive, cohesive system.
* **Weaknesses:** From a mathematical or algorithmic standpoint, the individual components are relatively standard. Cosine similarity, Softmax, Gaussian Mixture Models, and vector normalization are well-established. The originality lies in their combination and systems-level application rather than the introduction of fundamentally new learning theory or mathematical paradigms.

---

## Overall Recommendation
* **Rating: 4 (Weak Accept)**
* **Justification:** SPS-ZCA is an incredibly polished and empirically complete Systems-ML contribution with outstanding practical significance and exceptional presentation. The authors have gone to great lengths to profile physical framework overheads and benchmark their compiled CustomOps on physical Raspberry Pi hardware. However, the paper is fundamentally empirical and heuristic. It lacks the mathematical rigor, formal proofs, and theoretical guarantees that are essential to fully explain why the proposed geometric operations work and how they scale under overlapping manifolds. I recommend a Weak Accept, but strongly encourage the authors to elevate the theoretical rigor of the manuscript by addressing the mathematical gaps outlined below.

---

## Detailed Comments & Questions for the Authors

1. **Analytical Performance Bounds:** Can you provide a formal proof or analytical bound demonstrating under what geometric conditions (e.g., representation manifold angles or sub-Gaussian distributions) the activation-space blending in Equation 4 is guaranteed to match the isolated expert ceiling?
2. **Derivation of the PAC Bound:** In Section 4.8, you quote the standard PAC bound $N = \mathcal{O}\left( \frac{K + \log(1/\delta)}{\epsilon^2} \right)$. Please provide a formal, context-specific derivation of this bound for your low-dimensional coordinate routing space, defining the precise hypothesis class and loss function being analyzed.
3. **IDC Scaling Justification:** What is the theoretical justification for the specific scaling factor $u'_{k,b} = u_{k,b}/s_k$ used in Intra-Task Dispersion Calibration? Can you mathematically prove that dividing by the expected similarity scale optimizes decision boundaries or minimizes routing error under asymmetric task variances?
4. **Activation Bleeding Formalization:** Please provide a formal mathematical analysis/bound of the relationship between task manifold overlap (e.g., measured via mutual information, Wasserstein distance, or Fisher Separability Criterion) and the expected downstream classification or perplexity degradation due to "activation bleeding."
5. **GMM Covariance Analysis:** Have you mathematically analyzed the information loss or decision boundary distortion caused by assuming a diagonal covariance structure in the coordinate GMM rather than a full covariance matrix, especially as $K$ scales and coordinates become correlated?
