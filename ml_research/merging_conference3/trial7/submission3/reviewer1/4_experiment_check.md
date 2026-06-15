# Evaluation Step 4: Experimental Evaluation and Claims Verification

## Experimental Setup and Datasets
The empirical evaluation is highly thorough, structured, and validated across three distinct regimes:
1. **Synthetic Coordinate Sandbox**: A 14-layer, 192-dimensional sandbox simulating a Vision Transformer (ViT-Tiny) backbone over 4 distinct image tasks (MNIST, F-MNIST, CIFAR-10, SVHN). Expert mean ceiling is $83.00\%$. Extremely low-data constraints are enforced ($N=64$ calibration samples).
2. **Real-World Text Classification (GLUE Benchmark)**: SST-2, CoLA, and MRPC tasks merged using a pre-trained `BERT-Tiny` backbone (4.4M parameters) with $N=48$ calibration samples. It evaluates the framework on a noisy, coupled real-world pre-trained representation space.
3. **Generative LLM Blueprint Pilot**: A focused evaluation on a pre-trained `gpt2` model (124M parameters) merging Sentiment and French Translation experts with out-of-distribution math prompts, addressing representation anisotropy (narrow cone effect) with centered and clamped cosine similarity.

## Baselines
The paper includes an exceptionally comprehensive suite of baselines:
- **Static Merging**: Static Uniform Merging.
- **Parametric Gating / Routing**: Global Linear (Unregularized / Regularized), L3-Linear (Unregularized / Regularized), L3-Softmax (Unregularized / Regularized), and SOTA QWS-Merge.
- **Non-Parametric Routing**: SOTA PFSR (Parameter-Free Subspace Routing).
- **Out-of-Distribution Rejection**: Min Euclidean, 5-NN Euclidean, Min Cosine, Raw Mahalanobis, and Raw Energy-Based OOD.
- **Alternative Routing Schemes**: Hard Model Selection (discrete routing).

## Verification of Claims
- **Overfitting-Optimizer Paradox**: Confirmed. Parametric global linear routers achieve perfect training scores but collapse to $30.00\% - 30.90\%$ test Joint Mean, whereas GP-DR achieves a stable **$72.40\%$** without any optimization loop.
- **Stream-Level Collapse and MBH Recovery**: Confirmed. Without MBH, all dynamic routers collapse to uniform levels ($24.90\% - 31.20\%$) under heterogeneous streams ($B=256$). Pairing GP-DR with MBH recovers performance to **$70.20\%$** (a $+42.80\%$ recovery margin).
- **Real-World Viability**: Confirmed. On BERT-Tiny GLUE tasks, Static Uniform collapses to $16.22\%$, and the parametric softmax router overfits at $34.22\%$. GP-DR achieves a competitive **$45.78\%$** Joint Mean accuracy with zero training, and MBH recovers it to **$45.76\%$** under streaming ($+31.70\%$ recovery margin).
- **Generative Blueprint**: Confirmed. Centered and clamped cosine similarity resolves the narrow-cone representation anisotropy, allowing GP-DR to achieve **$90.00\%$ routing accuracy** and **$66.00\%$ OOD rejection AUROC** on GPT-2.

## Critical Insights and Rigor
1. **The Sandbox Joint Evaluation Artifact**: The authors transparently analyze that Static Uniform Merging's poor score ($25.50\%$) is an artifact of unconditioned evaluation across all 40 classes. Under a task-conditioned evaluation, ALL models (including Uniform Merging) recover their stand-alone expert ceilings ($83.00\%$). This reveals that the sandbox unconditioned evaluation is actually a stress-test of the router's ability to drive irrelevant task coefficients to zero to mute competing heads.
2. **Dynamic Merging vs. Hard Model Selection**: The authors justify parameter-space merging over hard model selection, demonstrating that GP-DR's soft blending achieves $72.40\%$ accuracy compared to $71.50\%$ for Hard Selection. This is supported by mathematical proofs of global smoothness (Lipschitz continuity).
3. **Logit Scaling & Calibration**: The authors discuss key engineering practices (temperature scaling, LayerNorm, and vectorized softmax) to handle dominant expert classification heads under unconditioned ensembling, offering a vital bridge for practitioners.
4. **MBH Hardware Latency/Throughput Trade-Offs**: The authors provide comprehensive CPU and A100 GPU benchmarks for MBH. They show that while CPU throughput drops by $44\%$, GPU throughput drops by $55\% - 68\%$ due to small, variable workloads underutilizing Tensor Cores ($B_m < 32$). The authors' PyTorch concurrent CUDA streams implementation is empirically validated to recover $30\% - 45\%$ of throughput loss, proving its high-efficiency systems-level viability.
5. **GPR Posterior Variance Collapse vs. Distance Heuristics**: The authors' OOD mixture sweep (Table 5) reveals a crucial finding: **simpler distance-based heuristics (particularly 5-NN Euclidean distance) substantially outperform GPR posterior variance by a massive margin under representational coupling and overlap.** 
   - Under representational overlap ($\beta = 0.75$), 5-NN Euclidean distance achieves $99.77\%$ AUROC and a $30.40\%$ FRR, whereas GP-DR's RBF posterior variance drops to $82.10\%$ AUROC and $90.40\%$ FRR (and Cosine variance drops to $67.12\%$ AUROC, $98.40\%$ FRR).
   - Under pure unit-sphere OOD noise ($\beta = 0.00$), GPR posterior variance collapses locally, yielding a high FRR ($80.80\%$ for RBF and $79.20\%$ for Cosine), while distance-based heuristics maintain an FRR of just $4.40\%$. The authors provide a clear mathematical explanation of this performance gap (GPR's extreme spatial sensitivity to proximity of even a single landmark).
   
The empirical evaluation is exceptionally thorough, realistic, and honest, providing all the necessary data and transparent insights required to evaluate the practical and theoretical limits of the proposed framework.
