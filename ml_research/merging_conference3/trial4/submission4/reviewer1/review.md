# Peer Review: SpectralMerge: Rethinking Model Merging in the Frequency Domain

---

## 1. Summary of the Submission
The paper addresses post-hoc model merging, specifically parameterized layer-wise model merging. It challenges the standard "spatial coordinate paradigm" (where layer-wise coefficients are treated as independent physical variables across depth) and argues that this parameterization is ill-conditioned, highly collinear, and prone to overfitting validation noise (the "Overfitting-Optimizer Paradox"). 

To resolve these issues, the paper proposes **SpectralMerge**, which re-parameterizes layer-wise task-combining coefficients in the frequency domain using the **Discrete Cosine Transform (DCT-II)**. It introduces two regularization variants: 
1. **SpectralMerge-LP (Low-Pass Hard Cutoff):** Restricting trainable parameters to the first $F$ low-frequency spectral components and setting higher frequencies to zero.
2. **SpectralMerge-Reg (Soft Spectral Regularization):** Optimizing the entire spectrum under a quadratic Spectral Decay Penalty.

The authors evaluate SpectralMerge on a Vision Transformer (ViT-B/32) multi-task simulation landscape (Model II), a physical 12-layer MLP model merging pipeline, and a pre-trained ResNet-18 model on CIFAR-10 tasks.

---

## 2. Strengths and Weaknesses

### Strengths
* **Elegant Mathematical Formulation:** Re-parameterizing the layer-wise merging trajectory in the frequency domain via DCT-II is a highly intuitive and mathematically elegant concept. The authors provide a compelling theoretical argument showing that the orthonormal spectral basis achieves perfect numerical conditioning ($\kappa \approx 1.0$) at all scales, bypassing the severe ill-conditioning of polynomial representations (PolyMerge).
* **Deep Boundary and Orthonormality Analysis:** The formal analysis showing that the DCT-II's even-symmetric boundary extension guarantees a flat spatial derivative (zero slope) at the virtual network boundaries is thorough and well-explained, especially in contrast to the Discrete Sine Transform (DST).
* **Comprehensive Simulation Stress-Testing:** The simulation benchmark (Model II) is exhaustively evaluated over 30 independent seeds, covering sequential clean streams, sample complexity sweeps, validation selection bias sweeps, and three severe adversarial stream conditions (extreme label shift, bursty streams, small batch noise).
* **Thoughtful Algorithmic Extensions:** The paper introduces valuable extensions such as **Block-wise Spectral Merging** to handle functional layer type heterogeneity and **Adaptive Bandwidth SpectralMerge (LP-Adaptive)** to dynamically adjust representation capacity. Both are empirically validated on physical networks.

### Weaknesses
* **The Inactive Layer Optimization Paradox (A Methodological Flaw):** In the physical ResNet-18 experiments (Section 4.6), the experts are fine-tuned by updating only the final classification head and the `layer4` convolutional block (layers 13 to 17), while layers 0 to 12 are kept completely frozen. Consequently, the task vectors for layers 0 to 12 are exactly zero ($V_k^{(l)} = 0$), and any merging coefficients for these layers have **absolutely zero physical effect** on the model weights. Despite this, the authors attempt to optimize merging coefficients across *all 18 physical layers*. This couples the inactive and active layers through the global spectral transform, creating a sharp step-function discontinuity in parameter sensitivity across depth. The authors frame this "PEFT-Induced Step-Function Discontinuity" as a fundamental challenge and use it to explain why the hard-cutoff SpectralMerge-LP ($F=3$) collapses to a random guessing baseline of 29.00%. 
  However, this discontinuity is entirely self-inflicted by a poorly designed optimization setup. If the authors had restricted the optimization to the **5 active layers** that actually changed, the search space would have been tiny, there would be no step-function discontinuity, and standard unconstrained spatial search or simple low-pass smoothing would have functioned perfectly. Introducing an artificial optimization bottleneck by optimizing inactive layers, and then claiming a soft spectral penalty is a major breakthrough for resolving it, is a highly suspect and circular methodology.
* **Toy-like Physical Evaluation Regimes:** While the paper includes physical neural network experiments to bridge the simulation gap, they are conducted in extremely weak and artificial regimes. 
  - The MLP experiment uses a simple 12-layer model on synthetic multi-task classification.
  - The ResNet-18 experiment uses a binary split of CIFAR-10 tasks (Task 0: Vehicles vs Task 1: Animals) fine-tuned on only 120 samples per task. The expert models themselves are incredibly weak (Expert 1 gets only 65.00% accuracy on its binary task).
  - The absolute accuracies of the merged models are very low (e.g., SpectralMerge-Reg gets only 54.00% multi-task accuracy). A standard, high-performance, large-scale model-merging evaluation (e.g., merging fully trained ResNet-50s or Vision Transformers on full image classification datasets, or LLMs on GLUE) is entirely missing.
* **Lack of Statistical Significance for Physical Experiments:** The paper reports standard deviations over 30 seeds for the simulation landscape, but **fails to report any standard deviations, error bars, or multiple runs for the physical MLP and ResNet-18 experiments.** Since offline validation tuning is performed on extremely tiny datasets ($M=10$ or $M=15$), the results are highly susceptible to sampling variance and selection bias. Reporting single numbers (e.g., 54.00% for SpectralMerge-Reg vs 29.00% for spatial/polynomial) without error bars or multiple runs raises concerns about the statistical stability and potential cherry-picking of these results.
* **DC-Only Baseline Superiority under Test-Time Adaptation:** Under sequential clean streams in Table 1, the extremely simple **Online Global Task-Wise (DC-Only)** baseline (which optimizes a single scalar per task and shares it globally across all layers) achieves **85.91%** accuracy, which actually **outperforms** both Online SpectralMerge-LP (85.32%) and Online SpectralMerge-Reg (85.17%). This suggests that under online TTA, allowing layer-wise variations is unnecessary and harmful. The authors do not adequately justify why a practitioner should accept the added mathematical and algorithmic complexity of SpectralMerge when a simple 1D global scalar baseline is more effective in this regime.

---

## 3. Soundness
* **Rating:** **Fair**
* **Justification:** While the mathematical derivations of the DCT-II and its boundary properties are correct and mathematically sound, the empirical methodology is undermined by several major flaws. Most notably, the "PEFT-Induced Step-Function Discontinuity" that causes SpectralMerge-LP to collapse to 29.00% on ResNet-18 is a self-inflicted artifact of optimizing merging coefficients on 13 completely frozen layers where task vectors are zero. Furthermore, the physical experiments are toy-like, utilize extremely weak expert models, and completely lack error bars or statistical significance analysis despite utilizing tiny, high-variance validation sets ($M \in [10, 15]$).

---

## 4. Presentation
* **Rating:** **Good**
* **Justification:** The paper is exceptionally well-structured, clear, and easy to read. The equations are clean, and the signal-processing analogies are integrated with high semantic clarity. However, the presentation is occasionally misleading. For example, the authors frame the offline "OFS-Tune" SpectralMerge's immunity to adversarial test-time streams as a major robustness breakthrough. In reality, any offline-tuned frozen model (including the simple Global Task-Wise baseline) is naturally immune to test-time stream non-stationarity because it does not adapt online. This should be framed more transparently.

---

## 5. Significance
* **Rating:** **Fair**
* **Justification:** The core idea of frequency-domain regularization of layer trajectories has potential, but its practical significance is severely restricted by the toy-like empirical evaluations. Since the simple DC-only baseline outperforms SpectralMerge under Online TTA, and because the physical evaluations are restricted to binary CIFAR splits with weak experts and no error bars, practitioners will find it difficult to justify adopting a mathematically complex DCT-based optimization framework over simple, active-layer-only spatial tuning or a global scalar baseline.

---

## 6. Originality
* **Rating:** **Fair**
* **Justification:** Conceptually, the paper is very close to PolyMerge, as both attempt to regularize layer-wise merging coefficients by constraining them to follow a smooth continuous trajectory across depth to prevent overfitting. Replacing PolyMerge's polynomial representation with a trigonometric cosine basis is a relatively incremental modification. However, the mathematical analysis regarding numerical conditioning and even-symmetry boundary derivatives is highly original and adds useful, rigorous insights to the model-merging literature.

---

## 7. Overall Recommendation
* **Recommendation:** **3: Weak reject**
* **Justification:** The submission presents a mathematically elegant framework (SpectralMerge) with solid theoretical properties regarding numerical conditioning and boundary derivative flattening. However, the paper's weaknesses currently outweigh its merits. The empirical evaluation relies heavily on a simplified simulation landscape and extremely toy physical experiments with weak experts. The optimization design suffers from a fundamental logical flaw (optimizing coefficients for frozen layers), which artificially creates a "PEFT-induced discontinuity" that causes their hard-cutoff variant to fail. Finally, the physical experiments lack the necessary statistical averages and error bars to guarantee stability.

To be suitable for acceptance, the authors should perform the following essential revisions:
1. **Address the Inactive Layer Flaw:** Re-run the ResNet-18 experiments by optimizing the merging coefficients **only** on the active layers that were actually fine-tuned. Show how unconstrained, polynomial, and spectral parameterizations compare when this logical error is resolved.
2. **Standard, Scale-Appropriate Benchmarks:** Replace the toy binary CIFAR-10 experiments with a standard, realistic model-merging evaluation (e.g., merging fully fine-tuned models on GLUE or full image-classification datasets).
3. **Statistical Significance:** Report averages and standard deviations over multiple random validation splits and seeds for all physical neural network experiments.
4. **Discuss the DC-Only Superiority:** Provide a transparent discussion explaining why the simple global task-wise scalar baseline outperforms SpectralMerge under test-time adaptation, and clarify the practical trade-offs.
