# 4. Experimental Evaluation Check

## Experimental Setup and Dataset
The primary evaluation of GraviMerge is conducted on the **Projected Digit Representation Space (RDS) Proxy** benchmark:
* **Dataset:** Uses scikit-learn's handwritten digits dataset (1,797 samples of 8$\times$8 grayscale images across 10 classes).
* **Task Formulation:** A $K = 4$ task stream is constructed using four distinct digit domains ('0', '1', '2', and '3').
* **Manifold Simulation:** To mimic deep representation manifolds, the 64-dimensional features are projected to $D = 192$ (and $D = 4096$ in the Appendix) using a random orthogonal projection matrix unique to each seed. Centroids are extracted from 64 calibration samples per task, and the remaining samples are used for evaluation.
* **Workloads:** Bounded Homogeneous, Heterogeneous ($B=256$), and Real-Time ($B=1$) edge-serving configurations are evaluated.

**Critique on Dataset Choice:** While the RDS proxy is highly effective for isolating and mathematically validating the differential routing equations with geometric precision, it is a **coordinate simulation** rather than a live pretrained neural network. Validating GraviMerge on actual text/vision representations extracted from a live transformer (e.g., Llama-3 or ViT) on standard downstream datasets (such as MMLU or GLUE) would provide a stronger real-world validation. The authors openly acknowledge this limitation in their discussion.

---

## Evaluation of Baselines
The paper includes a highly complete and robust set of baseline methods:
1. **Uniform Merging:** Simple static baseline (0 jitter, low accuracy).
2. **SPS-ZCA:** Single-pass centroid alignment (0 layer jitter, but cannot track layer-specific representations).
3. **SABLE:** Stateless, layer-wise cosine similarity routing (high accuracy, high layer jitter).
4. **EMA Smoothing:** Standard signal-processing first-order smoother (fails to stabilize, introduces phase lag).
5. **WMomentum:** Second-order weight-space momentum (degrades accuracy, explodes jitter due to simplex clamping discontinuities).
6. **ChemMerge:** State-dependent first-order chemical ODE kinetics (fails to stabilize, suffers from extreme phase lag and accuracy penalties).
7. **Kalman Filter:** Mathematically optimal first-order state-space estimator (fails to stabilize ensembling jitter due to lack of second-order states).

**Scholarly Enhancement Suggestion:** While the compared baselines are excellent, the literature review and discussion could be further enriched by discussing how GraviMerge positions itself relative to other emerging real-world test-time LoRA ensembling/merging frameworks such as **LoRA on the Go (LoGo)** (ACL 2026), **DA-MergeLoRA**, and **TTMM (Test-Time Model Merging)**.

---

## Supporting Evidence for Claims

The empirical results provide exceptionally strong and comprehensive support for the paper's central claims:
* **Highest Joint Serving Accuracy:** GraviMerge achieves **88.69%** serving accuracy, outperforming stateless SABLE ($87.65\%$) and SPS-ZCA ($88.51\%$), proving that second-order smoothing preserves and enhances representational alignment.
* **Unprecedented Jitter Reduction:** Slashes routing jitter to **0.00190 MAD**, achieving a **6.01$\times$** reduction compared to SOTA ChemMerge and **2.40$\times$** compared to SABLE.
* **Control-Theoretic and Mathematical Soundness:**
  * **Table 2 (Feedback Ablation):** Shows Coupled GraviMerge is highly robust and maintains low jitter even under tight feedback coupling ($\eta_{\text{feedback}} = 1.0$), demonstrating stable closed-loop tracking.
  * **Table 3 (Temporal Persistent Stream):** Under sequential task blocks, carrying over the velocity state vector achieves optimal accuracy ($89.30\%$) and the lowest layer-wise routing jitter ($0.00181$ MAD), resolving the statelessness and boundary-switching challenges.
  * **Table 4 (Deep Transformer Scale):** Under layer-specific representational drift and massive unnormalized scale variation ($D=768$), GraviMerge achieves a spectacular **$1.06 \times 10^6\times$ jitter reduction** compared to SABLE, validating geometric scalability.
  * **Table 5 (Noise Robustness):** Demonstrates that GraviMerge behaves as an active physical low-pass filter, maintaining superior accuracy and decreasing routing jitter under additive Gaussian noise.
  * **Table 6 (OOD Gating):** Incorporating Sentinel Attractor Dynamics (SAD) successfully handles out-of-distribution streams, reducing ensembling weight standard deviation to **$0.0578$** (converging to a uniform fallback mixture at the sphere's barycenter).
  * **Table 7 (High-dimensional auto-tuning):** Empirically verifies AGS, Adaptive Drag, and SAD on $D=4096$, showing significant improvements.
  * **Table 8 (Wall-Clock Latency):** Shows that physical execution of the multi-layer routing on Llama-3 scale dimensions ($D=4096$) is extremely fast (under 4 ms even for $K=64$ experts), confirming negligible real-world overhead.
* **Workload Resilience:** Maintains flat accuracy profiles across Homogeneous, Heterogeneous ($B=256$), and Real-Time ($B=1$) configurations, indicating high theoretical resilience to both "Heterogeneity Collapse" and "Vectorization Collapse."
