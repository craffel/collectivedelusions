# Intermediate Evaluation 4: Experimental Evaluation and Claims Verification

## Soundness and Quality of the Experimental Setup
The experimental evaluation is highly systematic, consisting of two main testbeds:
1. **Analytical Coordinate Sandbox:** A 14-layer sequential feedforward simulation ($D=192$, $d=8$, $K=4$ tasks) designed to isolate coordinate propagation and study eigenvalue decay. It evaluates models under orthogonal and overlapping manifold structures, across homogeneous and heterogeneous deployment streams.
2. **Simulated GLUE LoRA Benchmark:** A high-fidelity sequential simulation scaled to RoBERTa-Large dimensions ($D=1024$, $r=8$) over four tasks (SST-2, MRPC, CoLA, RTE), which measures representation propagation and downstream classification.

The baseline suite is exceptionally comprehensive, including **11 distinct methods** representing the state-of-the-art and classic approaches in flat ensembling (SABLE), temperature-optimized ERM, and PAC-Bayesian ensembling (PAC-ZCA) across multiple projection extraction modalities (Block, PCA, and UN-PCA).

---

## Critical Evaluation of Empirical Results and Claims

### 1. Verification of Projected Coordinate Collapse
The empirical results in Table 1 strongly support the authors' primary claim. Under Overlapping Manifolds (overlap=12) with a uniform routing configuration, **Uniform Merging collapses to exactly $25.00\% \pm 0.00\%$ accuracy** (the random baseline for 4 classes). The average deviation from projection idempotency is high ($\Delta_{\text{idem}} \approx 0.187$), proving that flat linear ensembling acts as a lossy filter that shrinks eigenvalues and destroys out-of-intersection representation norms.

### 2. Effectiveness of C-Lie-MM
The proposed C-Lie-MM successfully avoids coordinate collapse, maintaining $\Delta_{\text{idem}} \approx 1.24 \times 10^{-7}$ (perfect numerical idempotency in float32 precision).
- Under overlapping manifolds, C-Lie-MM achieves **$70.30\% \pm 4.01\%$ accuracy**, outperforming Uniform Merging by $+45.30\%$ absolute and SABLE (SEP-UN-PCA) by $+13.70\%$ absolute.
- Under both Homogeneous and Heterogeneous streaming workloads, C-Lie-MM maintains identical performance. This strongly supports the claim of **complete immunity to heterogeneity collapse**, which is a major system-level advantage over static weight-merging methods.

### 3. Transparent and Rigorous Ablations
The paper stands out for its high level of scientific transparency, actively addressing two potential critiques:
- **The Residual/LayerNorm "Strawman" Critique:** In actual networks, residual connections and LayerNorm can act as buffers against representation decay. The authors address this by conducting a dedicated sandbox ablation with residuals and normalization. Uniform Merging's performance rises from 25.0% to $51.90\% \pm 5.22\%$, validating the critique. Crucially, however, C-Lie-MM still significantly outperforms Uniform Merging by **$+20.40\%$ absolute** ($72.30\% \pm 4.71\%$), confirming that even with structural buffers, preserving manifold geometry remains vital to prevent feature distortion.
- **The Tuned Flat Baselines Critique:** Table 1 shows that flat baselines with optimized parameters (like Temp-Only ERM Block and PAC-ZCA Block) achieve $\sim 70\%$ accuracy, seemingly avoiding collapse. The authors analyze this and show that these baselines survive *only* by collapsing their routing temperatures ($\tau \to 0.010$) and routing entropy ($H/H_{\max} < 10^{-4}$), effectively becoming hard gating/selection operators and abandoning soft ensembling. To prove this, they conduct a frozen-temperature ablation ($\tau = 1.0$), where flat baselines collapse catastrophically ($38.40\%$ and $25.00\%$), while C-Lie-MM remains highly robust ($68.50\% \pm 4.21\%$). This is a brilliant and highly convincing experimental defense.

---

## Crucial Scholarly Distinctions and Limitations

### 1. Simulated vs. Physical Evaluation on GLUE
Section 4.4 presents the "Simulated GLUE LoRA Benchmark" at RoBERTa-Large scale, showing C-Lie-MM outperforming flat parameter-merging baselines (Task Arithmetic, TIES-Merging, SABLE) by $+42.0\%$.
*   **Scholarly Criticism:** While the simulation is mathematically high-fidelity (it genuinely propagates features through $L=8$ sequential projection layers using weights matched to RoBERTa's scale), it is still a **feature-propagation simulation** rather than an empirical evaluation on actual, physical fine-tuned weights of a RoBERTa-Large model on GLUE datasets. 
*   **Implication:** The authors claim that "curved geodesic-based subspace ensembling is vital for maintaining representation quality... in modern deep network architectures." However, they must clearly distinguish between this high-fidelity simulation and standard, physical downstream fine-tuning. The evaluation is simulated, and the claims of "real-world NLP performance" should be slightly tempered to reflect this.

### 2. Absence of Direct serving through End-to-End LLM Pipelines
The authors present PyTorch-based GPU forward pass benchmarks (SVD-based C-Lie-MM takes $0.51$ ms and Chebyshev polynomial C-Lie-MM takes $0.11$ ms on an NVIDIA A100 for batch size 256).
*   **Scholarly Criticism:** While these isolated GPU latency micro-benchmarks are highly informative and confirm the efficiency of the SVD-free polynomial formulation, they are not integrated into an actual end-to-end LLM serving engine (like vLLM or Hugging Face PEFT) to evaluate sequence-to-sequence token generation throughput (tokens/sec). The authors should acknowledge that actual production serving throughput is subject to additional pipeline overheads, although their prompt-level frozen routing policy (which caches the merged basis alongside the KV cache) provides a strong theoretical mitigation.
