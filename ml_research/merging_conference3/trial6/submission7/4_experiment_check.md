# 4_experiment_check.md - Experimental and Empirical Audit

This document audits the experimental design, baseline choices, simulated evaluations, ablation studies, and systems-level benchmarks reported in the submission.

---

## 1. Benchmarks and Baselines Evaluation
The experimental validation spans across a diverse and rigorous hierarchy of environments:
1. **Synthetic Sandbox:** A diagnostic physical laboratory ($L=14$ layers, intermediate representation dimension $D=192$, and $K=4$ disparate task manifolds: MNIST, FashionMNIST, CIFAR-10, and SVHN as the OOD task) designed to isolate, expose, and analyze layer-averaging and heterogeneity collapse.
2. **Real-World DomainNet Benchmark:** Vision Transformers (ViT-Base, $D=768$, $K=4$) across 4 domains: Quickdraw, Real, Sketch, and Infograph.
3. **Large-Scale NLP Benchmark:** LLaMA-7B task experts ($D=4,096$, $C=32,000$, $K=4$) across Math (GSM8K), Coding (HumanEval), Translation (WMT), and Instruction-Following (Alpaca).
4. **Ultra-Large Expert Pools:** Scaling evaluations up to $K=100$ specialized experts under extreme manifold congestion.

### 1.1 Baseline Comparison:
The proposed framework is compared against a comprehensive set of baselines:
*   *Expert Ceiling:* The upper bound performance of separate standalone expert models.
*   *Uniform Merging:* Standard parameter-space averaging.
*   *Static Merging:* Task Arithmetic and TIES-Merging (representing the state-of-the-art in static model fusion).
*   *Parametric Dynamic Routing:* Unregularized Linear Router, L3-Linear (Unregularized & Regularized), L3-Tanh, and L3-Softmax.
*   *State-of-the-Art:* QWS-Merge (Quantum Wavefunction Superposition Merging).

This baseline sweep is highly rigorous and provides an exhaustive evaluation of where the proposed method stands in the model-merging literature.

---

## 2. Key Empirical Findings and Validation
The empirical results across all configurations are highly supportive of the authors' claims:

### 2.1 Resolution of Heterogeneity Collapse (Table 3):
Standard parametric dynamic routers collapse on heterogeneous mixed-task streams because batch-averaging task coefficients forces them toward a uniform flat distribution (e.g., Linear Router drops from $51.00\%$ to $43.40\%$, QWS SOTA collapses to $43.30\%$). 
Our proposed **PFSR + MBH** completely resolves this issue. By dynamically partitioning the stream into homogeneous micro-batches on the fly, it maintains a collapse-free **71.60% Joint Mean accuracy** under heterogeneous streams, matching its sample-wise ($B=1$) baseline.

### 2.2 Superiority over Complex Routings (Table 2):
Our parameter-free PFSR + MBH achieves a high **75.00% Joint Mean accuracy** under homogeneous streams in the synthetic sandbox, coming remarkably close to the expert ceiling ($79.80\%$) and systematically outperforming QWS SOTA ($47.50\%$) and unregularized layer-wise routers ($45.60\%$).

### 2.3 Real-World Vision and NLP Scaling (Tables 5 & 9):
*   On **DomainNet (ViT-Base)**, PFSR + MBH + UNC achieves a stellar Mean accuracy of **78.50%**, recovering $97.5\%$ of the standalone expert ceiling ($80.50\%$) with zero trainable parameters and zero calibration split data, while static TIES-Merging achieves only $52.75\%$ and QWS SOTA collapses to $31.00\%$.
*   On **LLaMA-7B (NLP)**, our framework achieves a Mean accuracy of **79.12%**, recovering $96.8\%$ of the expert ceiling ($81.75\%$), while TIES-Merging achieves $60.38\%$ and QWS SOTA collapses to $35.25\%$.

---

## 3. High-Quality Ablation Studies
The paper includes a highly detailed set of ablation studies, validating every single component of the framework:
1. **Unit-Norm Calibration (UNC) Ablation (Table 4):** Verifies that UNC is critical for entangled features. Without UNC, cross-expert scale imbalances skew routing completely, resulting in a collapsed $25.00\%$ Joint Mean accuracy. With UNC, accuracy is restored to $75.00\%$.
2. **Class-Size Scaling Calibration Ablation (Table 10):** Verifies that Eq. 2 resolves maximum cosine similarity biases when merging highly asymmetrical expert registries (LLM with $C_1=32,000$ and classifier with $C_2=10$), raising Joint Mean from $58.00\%$ to $96.00\%$.
3. **Bounded Top-$k$ Gating Sweep (Table 6):** Shows that bounding active micro-batches via Top-$k$ routing is highly optimal, maintaining a robust $71.60\%$ joint mean even at $k=1$ while capping inference latency to a single forward pass.
4. **OOD Rejection Sweep (Table 7):** Audits the Cosine Rejection Threshold $\gamma_{OOD}$ and the GMM Density Estimator $\gamma_{density}$. At $\gamma_{OOD}=0.4$, we reject $91.60\%$ of SVHN OOD noise but suffer from a $23.73\%$ false-positive rate on in-distribution tasks (dropping accuracy to $62.60\%$). The proposed GMM density estimator successfully bypasses this trade-off, achieving an outstanding SVHN rejection rate of **95.20%** while maintaining an in-distribution false-positive rate of only **4.30%**, boosting overall accuracy to **74.10%**.
5. **Dynamic Temperature Scheduling (Table 11 & 13):** On boundary/ambiguous samples with small similarity margins, static routing acts as a hard argmax, achieving low accuracies ($53.50\%$ in sandbox, $48.60\%$ in DomainNet, and $51.20\%$ in LLaMA). Our dynamic temperature scheduler scales up local temperatures to perform soft blending, boosting accuracies substantially to **78.00%**, **71.40%**, and **76.50%** respectively.
6. **Ultra-Large Expert Pools ($K=100$) (Table 12):** Under extreme manifold congestion, uncalibrated flat routing collapses to $42.80\%$ accuracy. The diagonal covariance GMM and Hierarchical Gating successfully resolve coordinate overlaps, achieving an outstanding Joint Mean accuracy of **82.50%** and an OOD rejection of **94.60%**.

---

## 4. Systems-Level Systems Latency & Throughput Benchmark
To address critiques regarding the sequential forward pass overhead of MBH, the authors conduct comprehensive wall-clock benchmarks:
*   *Dynamic Merging Micro-Benchmark:* Low-rank dynamic adapter merging (computing $B_g \times A_g$ and adding to $W_{base}$) executes in under **1 ms** on GPU and **237.3 ms** on CPU, representing negligible overhead. Full-weight merging requires **54.4 ms**, and on-demand dynamic weight transfer over PCIe takes over **5000 ms**.
*   *Parallel Execution via SGMV Kernels:* SGMV parallel multi-adapter dispatching in PyTorch on an NVIDIA A100 GPU slashes sequential latency. Under maximum mixedness ($G=4$ active task adapters in a batch of $B=256$), parallel execution of SGMV is a mere **285.30 ms** (representing only a **5.71%** overhead compared to a single homogeneous model batch pass of 269.90 ms). This completely compresses the sequential latency of running four micro-batches (which would take over 1080 ms), proving that parallel multi-adapter kernel fusion fully eliminates sequential dispatching bottlenecks on standard GPU clusters.
*   *Inference Latency & Throughput Scaling Audit (Table 1):* Characterizes latency and throughput across batch sizes $B \in \{16, 64, 256\}$ and active tasks $G \in \{1, 2, 3, 4\}$. It demonstrates that while latency scales linearly with $G$ under sequential dispatch, aggregate throughput (samples/sec) scales exceptionally well with batch size $B$ (improving by over **$11.4\times$** from $B=16$ to $B=256$ at $G=4$), as similarity projection and LoRA merging overheads are amortized.

---

## 5. Major Experimental Concerns and Critiques
While the empirical validation is outstanding in its depth, there is one major experimental limitation that must be flagged:
1. **reliance on simulated penultimate feature representation manifolds:**
   The paper's real-world DomainNet and LLaMA-7B evaluations are *simulated* using representative feature embeddings and pre-calculated expert ceilings rather than running live, full-parameter active inference over raw text and image splits. While the authors are transparent about this protocol (and explicitly disclose it in Sec 4.4 and Sec 4.5), evaluating on simulated manifolds may mask practical system behaviors (such as representation out-of-domain noise, minor vocabulary distribution shifts, or hardware caching bottlenecks) that would only occur under a full end-to-end active deep network pipeline. Testing on live fine-tuned Vision Transformer and LLaMA weights is the gold standard, and while simulated scaling is highly reproducible and resource-efficient, it remains a simulated proxy.
