# Experimental Setup and Results Evaluation

## 1. Experimental Setup and Sandbox Calibration
The primary evaluation of the paper is conducted on the synthetic **Isolating Coordinate Sandbox**. While synthetic evaluations can sometimes limit real-world generalizability, the authors are exceptionally transparent about its purpose. They use the sandbox as a controlled physical laboratory to study representation-space coordinate dynamics, feature scale drifts, and systems latency trade-offs without the confounding noise and training stochasticity of organic pre-training.
Importantly, the sandbox is **deliberately calibrated** to match the organic accuracy ceilings of real-world counterparts (e.g., MNIST and F-MNIST at 100%, CIFAR-10 at 96%, SVHN at 30%). This represents a highly rigorous complexity calibration, ensuring that mathematical evaluations occur across realistic domain-complexity boundaries.

## 2. Real-World Organic Pilot Validations
To bridge the simulated bounds with real-world features, the authors include two highly rigorous real-world pilots:
1. **DomainNet (ViT-B/16):** Evaluates a larger library of $K=4$ diverse visual domains (Real, Sketch, Painting, Clipart) and $C=20$ classes per domain. Under an interleaved heterogeneous batch ($B=64$), Uniform Merging and Linear Routers suffer from severe representation collapse (9.35% Joint Mean accuracy). In sharp contrast, **PFAB-BOP matches the Expert Ceiling perfectly (78.80% Joint Mean accuracy)** and delivers a substantial wall-clock speedup over MBH (**19.80 ms vs 25.84 ms, a 1.31$\times$ speedup**).
2. **LLaMA-3-8B Text Generation:** Validates token-level routing (TSVHA) and the Dynamic Gate Reset (DGR) safeguard on pre-trained LLaMA-3-8B. Using three specialized LoRA experts (GSM8K, Alpaca, WikiText), TSVHA with Unit-Norm Calibration achieves a stellar **94.50% Gating Synchrony**. Furthermore, by implementing EMA entropy smoothing ($\beta=0.8$), the system filters out local syntactic stop-word noise, reducing the DGR false-alarm rate to a negligible **1.20%**.

## 3. Comprehensiveness of Baselines
The authors evaluate a comprehensive and highly robust selection of baselines:
- **Expert Ceiling:** Solves each task using its own dedicated model (theoretical upper bound).
- **Static Uniform Merging:** Averages the specialized expert adapter weights.
- **Learnable Parametric Routers (Linear Router, L3-Linear, QWS-Merge SOTA):** Evaluates standard parametric gating networks.
- **SOTA Systems-ML (PFSR + MBH):** Micro-Batch Homogenization with Parameter-Free Subspace Routing (prior state-of-the-art sequential dispatching baseline).
- **Jointly Trained Multi-Task Adapter:** A single low-rank adapter fine-tuned on the joint union of all tasks. This represents an exceptionally robust systems baseline. It processes heterogeneous streams in a single parallel pass with flat, constant latency (matching PFAB-ELC). However, due to gradient conflicts and capacity bottlenecking, its peak performance remains substantially below PFAB-BOP (**64.10% vs 81.50% Joint Mean accuracy** on the Sandbox), highlighting the benefits of PFAB's non-parametric routing over joint training.

## 4. Thoroughness of Ablations and Stress Tests
The paper features an exceptionally thorough set of ablation studies and stress tests that systematically evaluate every physical component of the design:
- **Ablation of UNC:** Shows that without Unit-Norm Calibration, representation scale drift causes certain experts to dominate routing, degrading accuracy to 53.40% on Sandbox. UNC restores it to 81.50%.
- **Subspace Entanglement Stress Test:** Sweeps cross-task representation leakage ($\epsilon$) from 0.0 to 0.5. Under extreme leakage ($\epsilon=0.5$), standard activation blending drops to 51.30%. Applying **SVD-based parameter-space orthogonalization** prior to serving successfully restores accuracy to **80.50%** (virtually matching the expert ceiling) while preserving **99.87%** of the experts' original specialized capabilities, with a completely negligible Joint Mean accuracy drop of only -0.10% in isolation.
- **High-K Scaling Sweep:** Sweeps installed task library size $K$ from 4 up to 64. Demonstrates that while dense activation blending latency scales linearly, applying **Sparse Top-2 Gating** slashes execution latency under $K=64$ from 24.84 ms down to just **11.22 ms** (a 54.8% compute saving) while preserving outstanding routing accuracy (79.10% accuracy, within 0.60% of the dense ceiling).
- **Unsupervised Streaming ELC (Streaming ELC):** Demonstrates that Layer 0 activations naturally cluster in distinct directions. Running online PyTorch-native K-means on Layer 0 activations achieves a highly competitive **58.20% Joint Mean accuracy** with zero offline labels or pre-computation, and stabilizes within just 50 to 100 unlabeled streaming samples.
- **FP8/INT8 Mixed Precision and Quantization Stability:** Simulates severe quantization noise by adding uniform Gaussian noise ($\sigma=0.05$) to intermediate blending coefficients and representations. PFAB-BOP preserves a robust **45.90% Joint Mean accuracy**, verifying that the Log-Sum-Exp mathematical stabilization trick successfully prevents numerical instability in low-precision regimes.

## 5. Support for Claims
The quantitative results **strongly and consistently support all of the paper's core claims**:
- PFAB-BOP matches prior SOTA (PFSR+MBH) and the expert ceiling perfectly (81.50% Sandbox, 78.80% DomainNet accuracy) under heterogeneous streams, successfully avoiding "heterogeneity collapse" while completely pruning the database-level systems scheduling layer.
- Both PFAB pathways exhibit completely flat and constant wall-clock execution latency profiles as task diversity ($G$) increases (Figure 2), whereas MBH latency scales linearly. Under $G=4$ active tasks, PFAB-BOP delivers a **2.52$\times$ latency speedup** and PFAB-ELC delivers a **3.26$\times$ speedup** over MBH.
- The proposed systems optimization techniques (Sparse Top-$p$ filtering and Chunked Layer-Wise Execution) successfully bound the intermediate activation memory footprint and parallel adapter computation, preventing Out-Of-Memory (OOM) failures under generative LLM workloads while preserving mathematically identical execution outputs.
- Early-layer centroids (PFAB-ELC) are highly sample-efficient (within 0.50% of peak accuracy with $|S_k|=5$ samples), though the authors honestly warn of their fragility under severe organic covariate shifts (DomainNet), showing excellent academic rigor.
