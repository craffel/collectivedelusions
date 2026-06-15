# Evaluation Task 5: Impact and Presentation Quality

## 1. Major Strengths
* **Highly Realistic Edge Motivation:** The paper focuses on a critical, frequently ignored vulnerability of test-time model merging: the impact of real-world environmental sensor noise, weather artifacts, and distortions on unsupervised test-time adaptation.
* **Pragmatic, Backpropagation-Free Architecture:** FlatMerge completely eliminates backpropagation and intermediate activation memory caching during adaptation. Peak SRAM overhead is identical to standard forward inference, satisfying a major constraint of edge accelerators.
* **Elegant Theoretical Synergy:** Combining low-degree polynomial depth parameterization (spatial filtering) with zeroth-order randomized smoothing (flatness-aware optimization) is mathematically elegant and conceptually sound.
* **High Statistical Rigor:** Simulating over 15 independent random seeds and presenting thorough standard deviations represents a high level of experimental discipline.
* **Physical Validation on Live Weights:** Anchoring simulated findings on actual MLP and 5-layer CNN models fine-tuned on real datasets (MNIST, FashionMNIST, KMNIST) provides vital empirical proof of the Overfitting-Optimizer Paradox and constant-prediction collapse on physical deep learning architectures.

---

## 2. Key Areas for Improvement

### A. Critical Bibliographic Omissions (Broken Citations)
* **Action:** The authors must fix the severe bibliographic errors in `references.bib` where foundational papers cited in the text (namely **PolyMerge** `\cite{polymerge}` and **SAM** `\cite{sam}`) are completely missing. Compiling the paper in LaTeX produces severe undefined citation warnings.

### B. Address the Practical Utility Gap (Task Arithmetic Dominance)
* **Action:** The authors must explain why a practitioner should deploy a complex, latency-heavy zeroth-order optimization loop like FlatMerge on edge devices when the simple, training-free static uniform merge (**Task Arithmetic**) consistently outperforms FlatMerge by **5% to 16% absolute accuracy** on physical MLP and CNN models under clean and moderate noise scales.

### C. Explicit Hardware Trade-offs (Static Weight Memory Storage)
* **Action:** The authors should openly discuss and highlight the static weight memory storage overhead. FlatMerge requires a **1.5$\times$ increase in static weight memory** ($2040.42$ MB vs $1360.28$ MB in simulation) to keep base weights, task vectors, and active merged weights simultaneously in memory. This is a critical hardware trade-off for memory-constrained MCUs and should be reflected in the conclusion/limitations section.

### D. Simulation-to-Real Gap
* **Action:** The authors should acknowledge that their continuous simulation environments (Model I and Model II) are highly stylized, decoupled from actual weight manifolds, and significantly over-estimate the benefits of unsupervised test-time optimization compared to physical networks.

---

## 3. Overall Presentation Quality
* **Writing Style (Excellent):** The paper is extremely clear, professional, and well-structured. The narrative flows logically from the introduction of the Noise-Entropy Collapse to the mathematical formulation of FlatMerge and the subsequent experiments.
* **Figures and Illustrations (Excellent):** The figures are high-quality, professional, and exceptionally informative. Figure 1 and Figure 2 clearly illustrate the robustness curves, and Figure 6 provides an intuitive qualitative visualization of the learned blending profiles.
* **Mathematical Precision (Excellent):** The mathematical descriptions of the polynomial subspace, Shannon entropy loss, and zeroth-order smoothing gradient estimator are mathematically precise, clear, and complete.

---

## 4. Potential Impact and Significance
* **Significance for High-Capacity Pre-trained Models (High):** On large-scale models (like pre-trained CLIP ViTs or LLMs) where test-time coefficient tuning is known to yield significant improvements over Task Arithmetic, FlatMerge's backpropagation-free flatness-aware adaptation is highly significant. It provides an elegant, memory-efficient way to adapt large models without activation caching or backpropagation.
* **Significance for Small-Scale Networks (Low-to-Medium):** For small-scale, resource-constrained networks (such as MLPs and CNNs built from scratch), the utility of FlatMerge is limited, as static merging (Task Arithmetic) remains the dominant paradigm.
* **Open-Source Impact:** If the authors release their calibrated continuous simulation sandbox code and hardware profiling benchmarks, it will provide a highly valuable low-compute prototyping library for the broader model merging community.
