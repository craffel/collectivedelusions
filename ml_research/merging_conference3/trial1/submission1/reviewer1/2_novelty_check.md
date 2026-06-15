# 2. Novelty Check

This section evaluates the key novel aspects of QP-Merge, its technical delta from existing research, and characterizes its novelty from a deployment-oriented practitioner's perspective.

---

## Key Novel Aspects
1. **Co-Design of Merging and Compression:**
   While academic model merging (e.g., Task Arithmetic, Ties-Merging, DARE) and post-training quantization (PTQ) have been heavily researched in isolation, QP-Merge is one of the first frameworks to recognize and address their joint bottlenecks—namely, that task vector additions compound heavy-tailed weight outliers and that distinct tasks exhibit incompatible activation scales.
2. **Task-Vector Centric Outlier Isolation (ORD):**
   Existing hybrid dense-sparse quantization methods (e.g., SqueezeLLM, SpQR) isolate outliers directly from static weights or based on activations. QP-Merge applies outlier separation to **task vector updates** ($\Delta W_t = W_t - W_{\text{base}}$). This is a highly logical choice, as task vector subtraction naturally highlights sparse, localized specialized updates that stretch symmetric quantization grids.
3. **Non-Equivalent Scale Calibration (QE-Calib):**
   Unlike traditional weight-activation scaling methods (e.g., SmoothQuant, AWQ) which must apply an inverse scaling factor to activations to preserve mathematical equivalence, QP-Merge permanently applies column-wise diagonal scaling matrices $D_l$ directly to the weight updates. This is a highly pragmatic design choice: in multi-task merging, there is no single activation scaling factor that can satisfy all merged tasks simultaneously. Abandoning mathematical equivalence in favor of an end-to-end representation alignment loss (unsupervised MSE) is a highly practical workaround.

---

## Technical Delta from Prior Work

### Post-Training Quantization (PTQ) Baselines
*   **SmoothQuant / AWQ:** SmoothQuant migrates quantization difficulty from activations to weights, maintaining equivalence via $Y = (X D^{-1})(DW)$. AWQ scales weights based on activation magnitudes. Both require runtime mathematical equivalence and do not handle multi-task conflicts. QP-Merge does not use runtime activation scaling, avoiding inference-time complexity and supporting platforms that cannot run dynamic scaling.
*   **SqueezeLLM / SpQR:** These methods use sensitive outlier separation on static pre-trained models. QP-Merge extends this to task vector updates in merged models, preventing the combined task vectors from expanding the quantization grid.

### Parameter-Space Model Merging
*   **Task Arithmetic, Ties-Merging, DARE, SyMerge:** These methods assume high-precision execution (FP16/FP32). They are blind to quantization. QP-Merge acts as a quantization-aware layer on top of model merging, allowing these base merging methods to be deployed in low-bit formats.

---

## Characterization of Novelty
The novelty of QP-Merge is characterized as **Incremental but Highly Pragmatic (Moderate-to-High Significance for Edge Engineers)**. 

- **Conceptually:** The individual building blocks (outlier separation with SpMM runtimes, scale searching via MSE minimization) are established techniques in LLM quantization. Thus, the work does not introduce a fundamentally new mathematical paradigm.
- **Practically:** The true originality lies in the **creative combination and adaptation** of these concepts to solve a pressing, real-world deployment challenge in multi-task model merging. The introduction of non-equivalent weight scaling (QE-Calib) specifically to resolve conflicting multi-task activation scales is a notable engineering insight that breaks from traditional PTQ constraints to deliver a highly functional, deployable edge solution.
