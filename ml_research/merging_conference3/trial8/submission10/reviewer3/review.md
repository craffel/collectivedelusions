# Peer Review Report

## 1. Summary of the Paper
This paper addresses the systems-level deployment bottleneck of serving multiple specialized Parameter-Efficient Fine-Tuning (PEFT) adapters (such as Low-Rank Adaptation, or LoRA) concurrently to a heterogeneous stream of noisy requests on edge-compute hardware.

To resolve this, the authors propose a bio-inspired paradigm shift: treating the ensemble of specialized experts as a dynamic, self-organizing symbiotic ecosystem in activation space. The proposed framework, **Evolutionary Symbiotic Merging via Lotka-Volterra Cooperation (ESM-LVC)**, implements classical Lotka-Volterra competition-cooperation differential equations directly into the activation space of a transformer during inference. The framework consists of three training-free and parameter-free components:
1. **Lotka-Volterra Activation Dynamics (LVAD):** A non-linear dynamical system that models ensembling coefficients (expert populations) evolving over a virtual local timescale govern by Lotka-Volterra equations.
2. **Symbiotic Interaction Tensor (SIT):** A pre-computed semantic interaction matrix derived from intermediate centroid similarity alignments that governs task mutualism (cooperative co-activation of similar/complementary tasks) and competitive exclusion (mutual suppression of conflicting/dissimilar tasks) using an automatic off-diagonal threshold heuristic or a Localized Pairwise Threshold Heuristic.
3. **Discrete Euler Symbiosis Solver (DESS):** An ultra-lightweight Projected Euler method solver to integrate the differential equations on-the-fly with sub-millisecond latency. It features an Adaptive Step-Size Heuristic to guarantee trajectory stability and Decoupled Activation-Inference Sharpening (DAIS) / Exponential Information-Theoretic Adaptive Sharpening (E-ITAS) / Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC) to separate continuous routing dynamics from final, sharp dilution-free classification decisions.

The entire setup is deployed in a **Paradox-Free Execution Layout** consisting of shared feature extraction layers, an intermediate routing layer, and a specialized expert region where adapter activations are dynamically blended sample-wise using a **Dynamic Scale Alignment (DSA)** operator to maintain norm stability.

### Key Quantitative Findings:
* **ICS Sandbox Simulation:** In a 192-dimensional synthetic vector space emulating a 14-layer Vision Transformer serving 4 simulated vision tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN):
  - ESM-LVC achieves **75.12% Joint Mean accuracy** under standard settings, outperforming recent training-free methods like SABLE (74.13%) and SPS-ZCA (74.31%).
  - ESM-LVC significantly outperforms the trained **Linear Router (Act)** (64.03%) by **+11.09% absolute** without requiring any parameter updates or backpropagation.
  - Under extreme domain noise (Scale 2.5), ESM-LVC maintains **65.37% accuracy**, outperforming SPS-ZCA by **+2.63% absolute** due to the dynamic, self-regulating noise-filtering properties of Lotka-Volterra competitive exclusion.
  - ESM-LVC exhibits absolute immunity to serving stream heterogeneity and batch size variations (maintaining flatline **75.12%** performance from batch sizes $B=1$ to $B=512$), whereas weight-space ensembling suffer from severe **batch heterogeneity collapse** (decaying to 44.18% at $B=512$).
  - In similarity sweeps ($\rho = 0.40$), ESM-LVC's mutualistic co-activation achieves **75.80% accuracy** (outperforming SPS-ZCA by **+0.58% absolute**), demonstrating the benefits of organic task cooperation over winner-take-all routing.
  - Under a Destructive Interference Penalty ($iw = 0.3$), ESM-LVC experiences a tiny degradation of only -0.32% (maintaining **74.80%** accuracy), outperforming SPS-ZCA by **+0.50% absolute**, verifying its self-sharpening competitive dynamics.
  - Serve-time routing solver latency benchmark on a single CPU core is strictly below **0.6 milliseconds** across all scales (up to $K=16$ experts and batch size $B=256$).
* **Physical Model Verification:** In offline evaluations using CLS activations extracted from Layer 12 of a pre-trained physical ViT-Tiny model across four real-world datasets:
  - **GMC-BSC** (Gaussian Mixture Centroids) successfully breaks the single-centroid attractor bottleneck, boosting routing accuracy to **93.50%** (clean) and **89.75%** (severe noise scale $\sigma=2.0$), outperforming single-centroid zero-shot baselines by **+3.25%** and the Fully-Optimized Linear Router by **+4.75%** absolute under noise.
  - **DM-BSC** (Dirichlet-Multinomial Bayesian Self-Calibration) achieves **28.25%** downstream classification accuracy, outperforming SABLE ($27.25\%$) and matching the Fully-Optimized Linear Router within $0.50\%$ absolute without requiring any trainable parameters, while maintaining exceptionally low routing entropy ($0.0343 \to 0.0748$).
  - The Projected Euler DESS solver is empirically confirmed to have a **0.00% fallback rate** across all real-world test batches, validating its absolute numerical stability under realistic, non-orthogonal manifolds.

---

## 2. Strengths and Weaknesses

### Strengths:
* **S1. Outstanding Conceptual Originality:** The paper introduces a highly creative and mathematically elegant bio-inspired paradigm shift. Treating multi-expert model ensembling as a self-organizing symbiotic ecosystem in activation space governed by Lotka-Volterra dynamics is a highly innovative contribution.
* **S2. Rigorous Theoretical Foundation:** Theorem 1 provides mathematical proofs for the trajectory boundedness and stability of the DESS Projected Euler solver, establishing a solid theoretical foundation and giving practitioners confidence in the algorithm's reliability.
* **S3. Excellent Writing & Narrative Structure:** The narrative is cohesive, engaging, and exceptionally clear. It links classical connectionist literature (lateral inhibition, Self-Organizing Maps, Hopfield Networks, Adaptive Resonance Theory) with modern PEFT serving workloads.
* **S4. Thorough Empirical Validation:** The experiments thoroughly evaluate performance, extreme noise resilience, batch heterogeneity sweeps, task mutualism, and destructive interference, accompanied by CPU serve-time latency benchmarks.
* **S5. Highly Practical Extensions:** The paper proposes excellent extensions like **E-ITAS** (Exponential Information-Theoretic Adaptive Sharpening), **DM-BSC** (Dirichlet-Multinomial Bayesian Self-Calibration), and **GMC-BSC** (Gaussian Mixture Centroids) which successfully break the single-centroid prototype bottleneck and address core peer-reviewer concerns.
* **S6. High Transparency & Honest Disclosures:** The authors are highly transparent and honest about their assumptions, the limitations of the synthetic sandbox, and the offline nature of their physical verification, outlining a concrete systems-level and physical adapter validation roadmap in Section 5.1.

### Weaknesses:
* **W1. Critical Bibliography and Citation Omissions:**
  This is the most glaring deficiency of the manuscript. The authors compare their method against two crucial dynamic ensembling baselines, **SABLE** and **SPS-ZCA**, in almost every single table and figure (Tables 1, 2, 4, 5, 6, 7, 8). SABLE and SPS-ZCA are referred to as standard/predecessor baselines and evaluated extensively. However, they fail to provide a single bibliographic entry or citation for either baseline in the text or the `references.bib` file! SABLE and SPS-ZCA are simply discussed in the text without any references (e.g., `\cite{...}`). This is a major violation of scholarly integrity and makes it impossible to verify the baseline definitions or trace their true origins.
* **W2. Uncited References in the Bibliography:**
  The `references.bib` file contains several high-impact entries that are completely uncited in the LaTeX sections of the paper:
  - `chatterjee2024robustness` ("On the Robustness of Dynamic Model Merging on the Edge" by Chatterjee and Vance, 2026), which is highly relevant to dynamic model merging robustness on edge devices.
  - `pfsr2025` ("Parameter-Free Subspace Routing for Dynamic Adapter Merging", CVPR 2025), which is directly related to parameter-free routing.
  - `mbh2025` ("Micro-Batch Homogenization for Heterogeneous On-Device Inference", NeurIPS 2025), which is relevant to serving stream heterogeneity.
  - `mehta2021mobilevit`, `lane2015deep`, `warden2019tinyml`, `zhou2019edge` which are foundational for on-device and edge deep learning.
  
  Citing papers in the bibliography without referencing them in the text indicates a lack of scholarly rigor and rushed editing.
* **W3. Stylized Destructive Interference Penalty:**
  The destructive interference penalty in the simulation (Equation 3) is a stylized theoretical surrogate ($P_{k, b} = \sum_{j \neq k} (1 - \rho_{k, j}) \alpha_k \alpha_j$). While its bilinear form is physically motivated by first-order perturbative interactions, the authors did not collect physical empirical data to prove that actual multi-adapter ensembling accuracy drops according to this exact mathematical product. In actual deep networks, interference is highly non-linear, layer-dependent, and subject to multi-expert crosstalk.
* **W4. No Active End-to-End Physical Adapter Serving:**
  While CLS token activations were extracted and classified offline to verify the routing solver, no physical PEFT/LoRA adapters were actually trained or served end-to-end to measure physical multi-task accuracy under active activation blending. Thus, the downstream impact of Dynamic Scale Alignment (DSA) and actual physical representation-space interference remains untested.

---

## 3. Soundness

**Rating: Good**

**Justification:** The paper is highly solid mathematically and empirically. The formulation of LVAD, SIT, DESS, and DSA is sound, and Theorem 1 provides rigorous stability proofs. The simulation sandbox (ICS) is well-designed to isolate the mathematical properties of the solver under varying similarities, noise scales, and batch sizes. The physical verification on ViT CLS activations is highly informative. However, the rating is capped at "Good" because (1) the physical verification remains offline and lacks end-to-end multi-adapter forward execution, and (2) the destructive interference penalty model remains a stylized bilinear surrogate without empirical physical validation.

---

## 4. Presentation

**Rating: Fair**

**Justification:** The writing style is polished, professional, and dense with high-quality equations, rigorous proofs, and clear explanations. The overall narrative is exceptionally easy to follow. However, the presentation falls short due to severe deficiencies in the bibliography compiling and citation completeness:
1. **Critical Omission of Citations for Core Baselines:** The lack of any bibliography entries or citations for SABLE and SPS-ZCA. This is a severe deficiency in scholarly reporting.
2. **Uncited Bibliography Entries:** Sloppy compilation of the `.bib` file where multiple relevant works are listed but never cited in the text.

---

## 5. Significance

**Rating: Good**

**Justification:** The paper addresses an important and timely edge-compute serving problem. If integrated with specialized multi-adapter serving frameworks like Punica or S-LoRA using weighted Triton/CUDA blending kernels, ESM-LVC can enable highly robust, noise-resilient, and training-free on-device serving. However, the downstream classification probes yield very low absolute accuracies (20.75% to 28.75%). While theoretically explained by data-starved calibration and frozen pre-trained representations, the low absolute numbers limit the immediate practical utility of the current proof-of-concept.

---

## 6. Originality

**Rating: Excellent**

**Justification:** The conceptual bridge between non-linear mathematical ecology (Lotka-Volterra equations) and activation-space model ensembling is highly original. The Continuous-to-Discrete solver (DESS), Projected Euler clipping, Adaptive Step-Size Heuristic, Information-Theoretic Adaptive Sharpening (E-ITAS), Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC), and multi-modal Gaussian Mixture Centroids (GMC) are highly creative and novel contributions to the PEFT-serving literature.

---

## 7. Overall Recommendation

**Rating: 3: Weak Reject**

**Justification:** This paper has remarkable merits, including exceptional conceptual originality, rigorous theoretical proofs (Theorem 1), a highly polished writing style, and thorough evaluations across simulation and physical representation manifolds.
However, as a matter of scholarly integrity and reporting completeness, the paper cannot be accepted in its current form. The authors compare their method extensively against two core baselines, SABLE and SPS-ZCA, in almost every table and figure, yet **completely omit any bibliographic entry or citation** for these baselines. This major citation omission makes it impossible for peers to verify the baseline definitions or trace their true origins. Additionally, the bibliography contains several relevant, high-impact entries that are completely uncited in the text.
Therefore, a **Weak Reject** is recommended. The paper requires revision to:
1. Provide proper citations and bibliography entries for SABLE and SPS-ZCA (or explicitly clarify if they are baselines introduced in one of the uncited works, e.g., `pfsr2025` or `mbh2025`).
2. Clean up the bibliography by either removing or citing the completely unused references in `references.bib` (`chatterjee2024robustness`, `pfsr2025`, `mbh2025`, `mehta2021mobilevit`, `lane2015deep`, `warden2019tinyml`, `zhou2019edge`).

Once these bibliography and citation completeness issues are resolved, this paper would easily deserve a strong Accept.

---

## 8. Constructive Questions & Actionable Feedback for Authors

1. **Address Baseline Citations (Critical):** Please provide the correct citations and bibliography entries for SABLE and SPS-ZCA. If SABLE and SPS-ZCA are techniques introduced in other works listed in your bibliography (such as `pfsr2025` or `mbh2025`), make this connection explicit in both the Related Work and Experiments sections.
2. **Bibliography Cleanup (Critical):** Clean up the bibliography by either citing or removing the uncited entries in `references.bib`. For example, `chatterjee2024robustness` (On the Robustness of Dynamic Model Merging on the Edge) is highly relevant to your edge-serving motivation and can be easily cited in Section 1 or Section 2.
3. **Clarify Parameter Cancellation:** Under standard configurations, you set $\beta_k = 1.0$ and have $\Gamma_{k, k} \approx 0.9999$, leading to $\Gamma_{k, k} \alpha_k - \beta_k \alpha_k \approx 0$. Please add a discussion in Section 3.1 explaining the physical implications of this cancellation (i.e., that diagonal self-growth is effectively suppressed, and the system's non-linear dynamics are driven entirely off-diagonal).
4. **Discuss Destructive Interference Model Limitations:** In Section 5.1, please explicitly discuss that your Destructive Interference Penalty model (Equation 3) is a stylized theoretical surrogate, and that actual physical multi-adapter interference is highly non-linear, layer-dependent, and subject to multi-expert crosstalk.
5. **Physical Adapter Blending Roadmap:** To elevate the paper's significance, could you discuss the feasibility of training actual specialized LoRA adapters on the physical ViT backbone and serving them end-to-end, rather than performing offline classification on CLS activations? Providing a preliminary result of end-to-end serve-time accuracy under active blending would significantly boost the paper's impact.
