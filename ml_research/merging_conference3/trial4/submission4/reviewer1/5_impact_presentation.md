# 5. Presentation, Strengths, Weaknesses, and Impact Evaluation

## Overall Presentation Quality
The presentation quality of the submission is **good to excellent**. 
* **Writing and Formatting:** The paper is exceptionally well-written, clear, and logically organized. The mathematical terminology from digital signal processing (such as DC/AC components, even-symmetry, orthonormal bases, and energy compaction) is applied accurately and integrated seamlessly into the deep learning model-merging context.
* **Mathematical Rigor:** The mathematical derivations are complete and easy to follow. For instance, the authors mathematically prove that the DCT-II's even-symmetry boundary extension guarantees a flat derivative (zero slope) at the virtual network boundaries, contrasting this rigorously with the Discrete Sine Transform (DST).
* **Visualizations:** The figures and tables are highly informative and clearly captioned, providing concrete visual and quantitative evidence of the authors' claims (e.g., the 2D contour plots of the surrogate loss landscapes and the matrix condition number comparisons).

---

## Major Strengths

1. **Strong Mathematical and Theoretical Motivation:**
   Using the Discrete Cosine Transform (DCT-II) to re-parameterize the merging trajectory is a highly elegant mathematical idea. The authors provide a compelling theoretical argument showing that the orthonormal spectral basis achieves perfect numerical conditioning ($\kappa \approx 1.0$) at any scale, bypassing the severe ill-conditioning that plagues polynomial trajectory smoothing (PolyMerge).
2. **Deep Boundary Condition Analysis:**
   The formal proof and empirical comparison of DCT-II's even-symmetry boundary extension versus the DST's odd-symmetry boundary extension is very thorough. It provides a solid theoretical and empirical justification for their specific transform choice.
3. **Thorough Simulation Stress-Testing:**
   The simulation evaluation (Model II) is highly comprehensive, covering standard clean streams, sample complexity sweeps, validation selection bias sweeps, and multiple adversarial conditions (extreme label shift, bursty streams, small batch noise) averaged over 30 independent seeds.
4. **Practical Extensions for Heterogeneity and Dynamics:**
   The paper proposes highly practical extensions to the core framework, such as **Block-wise Spectral Merging** (to handle architectural layer type heterogeneity) and **Adaptive Bandwidth SpectralMerge (LP-Adaptive)** (to dynamically expand representation capacity during training). Both are validated empirically.

---

## Major Areas for Improvement (Weaknesses)

1. **The Inactive Layer Optimization Paradox:**
   The authors optimize merging coefficients across all 18 layers of ResNet-18, despite layers 0 to 12 being completely frozen. This couples active and inactive layers through the global spectral transform, creating an artificial step-function discontinuity. If the optimization were properly restricted to only the 5 active layers, the search space would be tiny, there would be no step-function discontinuity, and the hard-cutoff SpectralMerge-LP would likely not collapse. The authors must address this fundamental logical flaw.
2. **Toy Physical Evaluation Regimes:**
   The physical experiments are conducted in extremely simplified, weak setups (a 12-layer MLP on synthetic classification, and a ResNet-18 fine-tuned on only 120 samples per task for binary CIFAR-10 tasks). The absolute accuracies of the expert models and the merged model (54.00%) are extremely low and fail to match the standard gold-standard benchmarks used in model merging literature (such as merging full ViTs on full image datasets or merging LLMs on GLUE).
3. **Lack of Statistical Significance in Physical Experiments:**
   While the simulation results report standard deviations over 30 seeds, the physical MLP and ResNet-18 results do not include any standard deviations or error bars. Given that validation tuning is performed on tiny datasets ($M=10$ or $M=15$), the results are highly sensitive to sampling variance. The authors must report statistical averages over multiple random seeds/splits for the physical models to confirm their stability.
4. **Simplistic DC-Only Baseline Outperforms SpectralMerge under Online TTA:**
   The Online Global Task-Wise (DC-Only) baseline achieves 85.91% accuracy under sequential clean streams (Table 1), which is better than Online SpectralMerge-LP (85.32%) and Online SpectralMerge-Reg (85.17%). Under online test-time adaptation, the simplest baseline wins, suggesting that the complexity of the frequency transform is unnecessary in this regime.

---

## Potential Impact and Significance
The paper has the potential for **moderate impact**.
* **If the empirical gap is addressed:** If the authors can validate SpectralMerge on modern, large-scale deep learning models (such as merging LLMs like LLaMA/Mistral or large Vision-Language models), the frequency-domain regularization of layer-wise combining coefficients could become a standard, highly robust technique for parameterized model merging under data-scarce tuning conditions.
* **In its current state:** The impact is restricted by the toy-like physical evaluations and the reliance on a simplified simulation landscape. Practitioners will be hesitant to adopt a mathematically complex, transform-based coefficient parameterization when a simple global scalar (DC-only baseline) or simple active-layer-only spatial tuning might perform comparably well without the added algorithmic overhead.
