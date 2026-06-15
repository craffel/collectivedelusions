# Empirical and Experimental Evaluation Audit

This document provides a critical assessment of the experimental setup, datasets, baselines, and empirical results in "Parameter-Free Activation Blending (PFAB)". We examine whether the empirical evidence truly supports the paper's broad claims.

## 1. Heavy Reliance on the Synthetic "Isolating Coordinate Sandbox"
The primary empirical backbone of the paper—including the main performance sweeps (Tables 1, 2), the latency profiles (Table 4, Figure 2), the subspace entanglement tests (Table 3), and the ablation studies—is executed within a synthetic, simulated environment: the **Isolating Coordinate Sandbox**.

### A. The Reality Gap of Simulated Coordinate-Space Dynamics
In the Sandbox, representations are modeled as low-dimensional vector spaces, and task-specific noise scales are manually calibrated to mimic realistic dataset performance limits. While this environment is highly useful for isolating mathematical variables (such as representation scale drifts, feature leakage, and execution latency under controlled conditions), it is **not** a substitute for real-world pre-trained networks.
Real-world pre-trained representations (from LLMs or ViTs) live on complex, high-dimensional, stochastic, and highly intertwined manifolds. The coordinate-space scrambling and simple additive Gaussian noise of the sandbox are highly idealized. Consequently, the pristine empirical results (such as perfect expert-ceiling matching of 81.50% Joint Mean accuracy) are a artifact of the sandbox's controlled environment, and do not guarantee equivalent performance on organic, complex representations.

## 2. Incompleteness and Fragility of the Organic DomainNet Pilot
To address the limitations of the sandbox, the authors present a real-world validation of PFAB on the DomainNet dataset using a pre-trained Vision Transformer (ViT-B/16). However, a critical audit of these results reveals severe limitations:

### B. Severe Collapse of the Single-Pass Pathway (PFAB-ELC)
The DomainNet pilot results (Table 6) expose the extreme fragility of the single-pass pathway (PFAB-ELC) in organic settings:
- On the synthetic sandbox, PFAB-ELC achieves a respectable $66.50\%$ Joint Mean accuracy.
- On the organic DomainNet corpus, PFAB-ELC's accuracy collapses to **42.50%** (a massive **36.30% absolute gap** below the expert ceiling of $78.80\%$).

This collapse occurs because early layers extract low-level, style-dependent features (edges, color, textures) which undergo severe covariate shifts across Real, Sketch, Painting, and Clipart. The pre-computed offline task centroids $\boldsymbol{\mu}_k^{(early)}$ become highly overlapped and uncalibrated, resulting in severe routing mis-classifications. This empirical collapse confirms that **the single-pass pathway is highly fragile and virtually unusable** in realistic organic multi-domain serving environments, contradicting any claims of high systems scalability without accuracy compromises.

### C. Small-Scale Validation on ViT
The DomainNet pilot is restricted to a small ViT-B/16 backbone and only $K=4$ domains with $C=20$ classes. Modern multi-adapter serving and merging research is primarily targeted at generative Large Language Models (e.g., LLaMA, Mistral) with dozens or hundreds of task experts. The lack of an organic, large-scale multi-task LLM validation is a major gap in the empirical evaluation.

## 3. Lack of Downstream Organic LLM Metrics
In Section 4.5 and Table 5, the authors evaluate their generative LLM proposals (TSVHA and the DGR safeguard) using a token-by-token sequence generation simulation across $T=50$ tokens.

### D. Artificiality of the LLM Simulation
This simulation is conducted using synthetic PyTorch-native tensor sequences with simulated transition boundaries, rather than on a real pre-trained autoregressive LLM (e.g., LLaMA-3-8B) running standard text-generation tasks (such as GSM8K, Alpaca-eval, or translation).
- Reporting "100.00% Gating Synchrony" and "78.00% compute savings" on a toy 50-token simulation fails to validate the practical viability of TSVHA and the DGR safeguard.
- In real language modeling, vocabularies heavily overlap, and local prediction entropy fluctuates naturally due to syntax, which would introduce massive routing noise and trigger high false-alarm rates in the DGR monitor.
- The paper does not report any real downstream language generation metrics (e.g., perplexity, ROUGE scores, or GSM8K accuracy). Without these downstream metrics, there is no empirical proof that the one-token physical routing lag and periodic vocabulary projections do not degrade generated text quality.

## 4. Latency and Compute Crossover Trade-offs
The authors claim that PFAB-BOP delivers constant systems-serving latency ($O(1)$ backbone passes) and a major speedup over MBH. However, we analyze the compute and throughput scaling profiles:

### E. The 2$\times$ FLOPs and Saturated Throughput Penalty
PFAB-BOP requires executing two complete passes of the backbone model per batch (a base-only prototyping pass followed by an activated execution pass). Under peak-throughput serving conditions where the GPU is fully saturated:
- Doubling the backbone passes cuts the serving throughput in half: $QPS_{PFAB-BOP} \propto B / 2\mathcal{C}_{backbone}$.
- Under low task diversity ($G \le 2$), MBH requires only 1 or 2 sequential passes, making MBH more FLOP-efficient and providing superior saturated serving throughput.
- PFAB-BOP is only FLOP-efficient when task diversity is high ($G \ge 3$), which represents a specialized, high-diversity regime.

### F. Hidden Memory Bottleneck in Vectorized Evaluation
To avoid PyTorch kernel launch bottlenecks at large batch sizes, the authors design a vectorized parallel adapter execution layer (`torch.bmm`) which expands activations.
As the authors' own memory scaling analysis in Section 3.6 notes, vectorized parallel evaluation of $K$ experts on an LLM batch of size $B=256$, sequence length $S=2048$, and dimension $D=4096$ requires expanding activation tensors to over **68.7 GB of GPU VRAM per layer**, causing instant Out-Of-Memory (OOM) failures on standard hardware.
While the authors propose Sparse Gating and Chunked Execution to bound memory, **these memory optimizations are only evaluated on the synthetic sandbox** (Tables 1, 2) where sequence length is $S=1$ and intermediate dimension is $D=192$. The paper does not profile the memory usage or latency profiles of these optimizations on real-world, large-scale networks, leaving their practical safety unverified.
