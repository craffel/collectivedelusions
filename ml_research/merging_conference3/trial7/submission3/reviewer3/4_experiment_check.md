# 4. Experimental Setup and Baselines Evaluation

## Critical Evaluation of the Experimental Setup and Datasets

* **The Isolating Coordinate Sandbox:**
  The primary evaluation is conducted on a synthetic, self-contained $14$-layer, $192$-dimensional "Isolating Coordinate Sandbox." While this sandbox is mathematically clean and provides a controlled environment, it represents an idealized, block-coordinate orthogonal setting. In real-world merged models, representations are rarely block-orthogonal; feature channels share significant overlap, correlation, and anisotropic distribution.

* **Mitigation via Real-World and Coupled Sweeps:**
  To the authors' credit, they actively mitigate this synthetic sandbox limitation by:
  1. Evaluating varying levels of **representation coupling $\gamma$** (introducing feature leaks up to $\gamma = 0.75$).
  2. Performing a **real-world multi-task model merging evaluation on GLUE text classification datasets** (SST-2, CoLA, MRPC) using a pre-trained BERT-Tiny backbone.
  3. Conducting a **generative pilot study using a pre-trained GPT-2 backbone** on sentiment analysis, translation, and math prompts.
  These additional evaluations are highly commendable, bridging the gap between synthetic abstractions and actual deep learning features.

## Baseline Comparisons
The baseline selection is comprehensive and highly thorough, comparing GP-DR against:
* **Static baselines:** Static Uniform Merging.
* **Parametric routers:** Global Linear (unregularized and regularized), L3-Linear, L3-Softmax.
* **State-of-the-art dynamic merging systems:** Quantum Wave Superposition Merging (QWS-Merge) and Parameter-Free Subspace Routing (PFSR).
This wide range of baselines ensures that the proposed method is evaluated against both historical and contemporary model-merging routers.

## Do the Results Support the Claims?

### Claim 1: Bypassing the Overfitting-Optimizer Paradox
* **Verdict: Supported.** 
* *Evidence:* The unregularized and regularized Global Linear Routers achieve near-perfect calibration split scores but collapse catastrophically to $\sim 30.00\%$ Joint Mean accuracy on test data. By contrast, GP-DR achieves **$72.40\%$** Joint Mean accuracy under standard batching with zero training loops. This empirically verifies that non-parametric Bayesian routing successfully avoids over-parameterized optimization loops and overfitting under low-data constraints.

### Claim 2: Bypassing Stream-Level Collapse via MBH
* **Verdict: Supported.**
* *Evidence:* In heterogeneous streaming environments, standard parallel forwarding collapses all dynamic routers to uniform merging performance ($\sim 25\% - 31\%$). Integrating Micro-Batch Homogenization (MBH) results in a dramatic recovery across all normalized models: GP-DR recovers to **$70.20\%$** accuracy ($+42.80\%$ recovery margin) and L3-Softmax recovers to **$72.00\%$** ($+47.10\%$ recovery margin). This demonstrates that MBH is highly effective at preventing representation-averaging collapse in streams.

### Claim 3: Accurate and Robust OOD Detection via GPR Posterior Variance
* **Verdict: NOT Supported.**
* *Evidence:* While the paper claims a "$100.00\%$ OOD Rejection Rate" with $0.00\%$ False Rejection Rate (FRR) for orthogonal OOD inputs, Section 4.5 and Table 8 reveal a severe empirical collapse under realistic settings:
  1. Under pure unit-sphere OOD noise ($\beta = 0.00$), GPR posterior variance exhibits severe "variance collapse" (RBF variance of $0.0024$, S.D. $0.0058$), which is statistically indistinguishable from in-distribution samples (mean $0.0014$, S.D. $0.0041$). This leads to a massive False Rejection Rate of **$80.80\%$** (RBF) and **$79.20\%$** (Cosine).
  2. Under representational coupling ($\gamma = 0.50$) and overlap ($\beta = 0.75$), the AUROC of GP posterior variance drops to **$82.10\%$** (RBF) and **$67.12\%$** (Cosine).
  3. Crucially, **simple coordinate-space distance-based heuristics (particularly 5-Nearest Neighbor distance) substantially outperform GP-DR's posterior variance by a massive margin.** At $\beta = 0.75$, 5-NN Euclidean distance achieves **$99.77\%$ AUROC** and a vastly lower FRR of **$30.40\%$**.
  This demonstrates that the claim of GPR providing a robust, superior uncertainty metric for OOD detection is fundamentally false. Simpler geometric distance metrics are far more robust, simpler to implement, and do not suffer from local variance collapse.

### Claim 4: Performance Superiority or Competitiveness
* **Verdict: Partially Supported.**
* *Evidence:* On standard orthogonal coordinates, GP-DR's raw accuracy (**$72.40\%$**) is significantly lower than PFSR SOTA (**$77.60\%$**). The authors explain that GPR's continuous approximation causes prior-mean shrinkage, which allows irrelevant task heads to compete and drag down accuracy. While reducing $\sigma_n^2$ or sharpening the posterior mean recovers performance to $76.10\%$, it degrades the Lipschitz smoothness and uncalibrates the OOD variance. Under severe representational coupling ($\gamma = 0.75$), GP-DR achieves a stable Joint Mean accuracy of **$77.90\%$**, matching PFSR SOTA (**$77.80\%$**). Thus, while GP-DR is highly competitive under blurred representational manifolds, PFSR remains simpler and more accurate in standard settings.

### Claim 5: Soft Blending vs. Hard Model Selection
* **Verdict: Supported.**
* *Evidence:* GP-DR's soft parameter blending achieves **$72.40\%$** Joint Mean accuracy compared to **$71.50\%$** for the Hard Model Selection baseline. This supports the claim that continuous weight-space blending preserves cooperative parameter structures and slightly outperforms discrete model selection.
