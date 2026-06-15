# Evaluation Step 2: Novelty Check

## Assessment of Key Novel Aspects
The paper introduces several highly novel concepts, particularly at the intersection of model composition and edge-device compression:
1. **Post-Merge Joint Weight Pruning and Coefficient Tuning (ZipMerge):** Co-optimizing continuous merging coefficients $\Lambda$ and binary pruning boundaries $M$ at test-time using an unsupervised minimum entropy objective is a highly fresh formulation. Traditional approaches either perform merging and pruning sequentially (which leads to suboptimality) or optimize them in isolation. 
2. **First- vs. Zero-Order Test-Time Co-Optimization:** Using Straight-Through Estimators (STE) or 1+1 Evolution Strategy (ES) to bypass non-differentiable threshold-based magnitude pruning operators directly during test-time calibration is novel. The exploration of their respective geometric trajectories (e.g., STE being superior at 80% sparsity due to variance reduction, and ES being superior at 50% due to global non-gradient evaluations) provides deep insights.
3. **Orthogonal Procrustes Alignment for PEFT Merging:** The introduction of an analytical, SVD-based post-hoc rotation to align coordinate spaces of separately trained LoRA adapters prior to linear merging is highly novel. It directly targets coordinate basis misalignment—the fundamental mathematical driver of averaging errors in PEFT space—without requiring access to training data or incurring substantial runtime overhead.
4. **Co-Design of Joint Quantization-Pruning (PTQ) under STE:** The theoretical formulation integrating Uniform Quantization and dynamic pruning directly into the Identity-pass STE represents an elegant multi-compression co-design.

## "The Delta" from Prior Work
The paper positions itself very clearly relative to three main bodies of literature:
* **Model Merging (e.g., Task Arithmetic, TIES-Merging, AdaMerging):** Prior works assume a fully dense merged model, which is a severe hurdle for deployment on resource-constrained hardware. ZipMerge introduces sparsity constraints and co-optimizes the mask with coefficients.
* **Network Pruning (e.g., Magnitude Pruning, Wanda, SparseGPT):** Pruning typically focuses on single-model setups. ZipMerge addresses the spatial and representational conflicts that arise uniquely during multi-task expert merging.
* **Test-Time Adaptation (e.g., Tent, MEMO, AdaMerging TTA):** Standard TTA adjusts parameters (like BatchNorm or merging weights) without pruning. ZipMerge introduces a dynamic pruning boundary inside the TTA loop.
* **Representational Post-Mortem (e.g., ESR/RegCalMerge):** Rather than presenting a curated narrative of triumph, the paper conducts a rigorous post-mortem to map boundary failures. It provides highly pragmatic analyses (such as showing the P-then-M baseline outperformance and the Overfitting-Optimizer Paradox) that are rarely published in mainstream papers but are incredibly valuable for engineers.

## Characterization of Novelty
The novelty is **significant and highly pragmatic**. 
While some individual components (magnitude pruning, STE, 1+1 ES, LoRA, and Orthogonal Procrustes alignment) are known in their respective domains, their synthesis into a cohesive co-optimization and alignment framework for edge deployment is highly original. 

For real-world practitioners, the paper is exceptionally refreshing. It avoids "toy" assumptions (such as assuming infinite on-device storage or perfect domain alignment) and instead maps the exact physical system realities. The addition of the SVD-based Procrustes alignment is a particularly powerful, data-free, negligible-overhead mathematical solution that bridges the massive performance gap in adapter merging. The paper's honest reporting of boundary conditions and its extensive system-level studies (profiling CPU latency, VRAM footprint, and sorting approximations) represent a high-value contribution to the field of applied machine learning.
