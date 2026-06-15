# Experimental Evaluation Analysis: GraviMerge

## 1. Evaluation of the Experimental Setup and Datasets
The primary evaluation of GraviMerge relies on the **Projected Digit Representation Space (RDS) Proxy benchmark**:
* **The Construction:** The benchmark utilizes scikit-learn's toy handwritten digits dataset (1,797 samples of 8x8 grayscale images) and projects the 64-dimensional pixel features to $D = 192$ using a random orthogonal projection matrix. A 14-layer network is mathematically simulated, modeling multi-expert blending as coordinate operations on these projected vectors.
* **The Practitioner's Critique (Major Weakness):** For a method targeting multi-task edge serving of deep foundation models, evaluating *strictly* on a simulated coordinate sandbox using 8x8 projected handwritten digit pixels is a **critical limitation**. Real-world latent spaces of deep LLMs or Vision Transformers are highly structured, non-linear, and undergo complex semantic shifts across depth. A random orthogonal projection of 8x8 pixel values cannot capture the true manifold structure or noise distributions of a real pretrained transformer. This is a classic **contrived toy problem** that fails to demonstrate real-world utility or deployment readiness.

---

## 2. Evaluation of Baselines
The paper includes a robust selection of baselines in Table 1:
* **Stateless Dynamic Merging:** SABLE (Sample-wise Activation Blending).
* **Static Model Merging / Routing:** SPS-ZCA (Single-Pass Zero-Shot Centroid Alignment) and Uniform Merging.
* **Stateful Dynamic Merging:** ChemMerge (SOTA first-order chemical kinetics).
* **Classic Signal Processing:** EMA Smoothing (first-order low-pass), WMomentum (weight-space second-order momentum), and a Kalman Filter baseline (state-space tracking).

### Analysis of Baselines:
* This represents a comprehensive and fair selection of baselines. The inclusion of the Kalman Filter as a rigorous state-space tracking baseline is particularly strong, as it provides a mathematically optimal linear alternative.
* **The Unaddressed Simplification:** However, the paper fails to emphasize the extreme competitiveness of **SPS-ZCA**. SPS-ZCA achieves an accuracy of **$88.51\% \pm 1.68\%$** with a layer-to-layer weight jitter of exactly **$0.00000$** (since it aligns centroids once at Layer 3 and uses static weights thereafter). In contrast, GraviMerge achieves **$88.69\% \pm 1.68\%$** with a jitter of **$0.00190$**. This means the highly complex, mathematically intensive, and computationally heavy GraviMerge yields a mere **$0.18\%$ absolute accuracy gain** over a simple, zero-overhead static method. For any practitioner, a $0.18\%$ improvement is completely negligible and does not justify the massive engineering and runtime complexity of implementing geodesic integration and parallel transport.

---

## 3. Evaluation of Results and Supporting Evidence
* **Accuracy-Stability Pareto Frontier:** The results in Figure 2b show that GraviMerge occupies the optimal top-left corner of the Pareto frontier. Standard SABLE has high accuracy but volatile jitter. ChemMerge and EMA smoothing fail, incurring severe accuracy penalties (~78-79%) due to phase lag-induced overshoots. This supports the theoretical claim that second-order dynamics successfully filter high-frequency noise without lagging.
* **Robustness to Dynamic Workloads:** The authors show that accuracy remains constant across Homogeneous, Heterogeneous ($B = 256$), and Real-Time ($B = 1$) serving setups. This is expected because each sample is processed independently in their execution model, but it is good to confirm that batching does not introduce numerical divergence.
* **Theoretical/Simulated Scaling (Appendix):**
  * *Appendix 7.4 (Deep Transformer Verification):* Tests GPT-2 dimensions ($D = 768$, 12 layers) under simulated layer-specific centroid drift. This is a strong theoretical check showing that GraviMerge handles representational drift better than SABLE.
  * *Appendix 7.6 (High-Dimensional Auto-Tuning):* Validates AGS and SAD at $D = 4096$, showing that auto-tuning schedules can stabilize velocity spikes.
* **The Key Empirical Gap (No Downstream Task Evaluation):**
  * Despite the scaling validations in the appendix, **there is zero downstream task evaluation on real models** (e.g., text generation benchmarks like MMLU, GSM8k, or GLUE on models like Llama or Mistral).
  * In Section 4.2 ("Empirical Scope, Limitations..."), the authors openly acknowledge this gap: *"we acknowledge that it is a projected simulation and does not evaluate downstream language task generation on standard LLM benchmarks... we frame full downstream NLP evaluation on pretrained LLMs as a critical immediate future research milestone."*
  * For an applied ML conference, this is a major empirical weakness. Without validating the method on actual downstream generation tasks using real pretrained LoRA weights, it is impossible to know if the stable routing weights generated by GraviMerge translate to coherent generation outputs or if the L2-normalization approximations in Decoupled Mode introduce representational distortions.

---

## 4. Serving Profiles and Latency Benchmarks
* **The CPU Benchmarks (Table 5):** The authors report CPU execution latencies for $D = 4096$ across 12 adapted layers. For $K = 4$ experts, GraviMerge takes **$1.29 \text{ ms}$**; for $K = 64$ experts, it takes **$3.98 \text{ ms}$**.
* **Practitioner's System Critique:**
  * While the authors claim this is "negligible" compared to LLM forward propagation, **$1.29 \text{ to } 3.98 \text{ ms}$ of sequential CPU routing overhead per sample is actually quite significant** for latency-sensitive, high-throughput edge serving.
  * In auto-regressive generation, where tokens are generated sequentially (one-by-one), adding several milliseconds of sequential routing calculations at every token step can severely degrade the overall Token Generation Latency (TGL).
  * The benchmarks are conducted on a single CPU core. No physical GPU-based profiling (e.g., measuring Triton/CUDA execution or kernel launch latency) is provided, leaving a notable gap in practical systems validation.
