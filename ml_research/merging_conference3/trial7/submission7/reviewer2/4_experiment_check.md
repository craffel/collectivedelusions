# 4. Experiment Check

## Evaluation of Experimental Setup
The experimental setup spans multiple random seeds (10 seeds) and compares ELATI against a comprehensive suite of static (Uniform, DARE, TIES) and dynamic (Linear Routers, PFSR) ensembling baselines. However, a critical inspection reveals several key weaknesses and inconsistencies.

## Critical Inconsistencies and Weaknesses

### 1. Suspiciously Weak Static Model Merging Baselines
In Table 1, the authors report that:
- **DARE-Merging** achieves only **32.56% ± 2.66%** Joint Mean accuracy.
- **TIES-Merging** achieves only **37.39% ± 3.03%** Joint Mean accuracy.
- **Uniform Merging** achieves **48.27% ± 2.23%** Joint Mean accuracy.
- **Critique:** In the established model-merging literature, DARE and TIES-Merging consistently and significantly outperform standard Uniform averaging by resolving sign interference and pruning redundant updates. The fact that both advanced methods perform drastically worse than standard Uniform Merging suggests a flawed, unoptimized, or highly pathological implementation in the authors' sandbox, which undermines the integrity of these comparisons.

### 2. Complete Absence of Autoregressive LLM Experiments
The paper repeatedly highlights the applicability of ELATI to **Large Language Models (such as LLaMA-7B or 70B)** and discusses sequence pooling operators ($\Psi_{\text{final}}$, causal mask constraints, attention sinks) at length.
- **The Gap:** Despite this emphasis, the authors **do not run a single experiment on a real text dataset or an autoregressive LLM**. The sequence pooling simulations are synthetic (using 3D noise tensors with a sequence length of 16), and the only physical model evaluated is a tiny Vision Transformer (ViT-Tiny). 
- **Systems Scaling Extrapolation:** The LLaMA-7B "scaling micro-benchmarks" (Figure 8) are purely a CPU-bound mathematical extrapolation based on single-matrix additions, not physical serving evaluation. For a paper claiming to solve the serving bottleneck of massive multi-tenant LLM deployments, the complete lack of physical text-generation or causal decoder benchmarks is a major experimental deficit.

### 3. CPU Wall-Clock Latencies lack Industrial Relevance
- Symmetrical physical end-to-end forward propagation is benchmarked on a multi-core CPU. Under this setup, PFSR+MBH takes **36.90 ms** and ELATI takes **26.43 ms** (a ~10 ms absolute saving) on a batch of 1,000 samples.
- **Critique:** CPU wall-clock latencies are highly unrepresentative of industrial serving environments. In high-performance clouds, models are executed on massive parallel accelerators where kernel launch scheduling and memory bus latency are completely different from CPU execution threads. Timings on a CPU provide no guarantees of actual systems speedups in production-grade vLLM or S-LoRA servers.

### 4. Overlooked Hyperparameter Sensitivity
ELATI's performance and stability are highly sensitive to multiple hyperparameters:
- Gating temperature ($\tau$)
- Active expert pruning threshold ($\epsilon_{\text{prune}}$)
- Calibration split size ($|X_{\text{cal}}^{(k)}|$)
- Routing depth ($l_{\text{route}}$)
While the authors present several isolated sweeps in the "Deep-Dive" section, there is no joint optimization analysis. The interaction between routing depth ($l_{\text{route}}$) and representation entanglement ($\eta$) remains uncharacterized, meaning the optimal configuration might be highly fragile across different network architectures and stream noise levels.
