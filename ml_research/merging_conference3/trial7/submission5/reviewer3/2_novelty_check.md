# Novelty Check and Delta Analysis

## 1. Key Novel Aspects
The paper introduces several highly elegant and simple conceptual and mathematical innovations:
- **Shift from Parameter-Space to Activation-Space Blending:** Standard dynamic model merging compiles unified weight states on-the-fly, which is batch-bound. The primary novelty of PFAB is executing sample-wise blending directly in feature space within each layer. While multi-adapter or LoRA-MoE serving layers (e.g., S-LoRA, Punica) exist and evaluate adapters in parallel, they rely on complex learnable parametric routers. PFAB's novelty is establishing a **non-parametric, calibration-free gating alternative** that achieves sample-level routing with zero trainable parameters by projecting features directly onto pre-trained, frozen classification heads.
- **Unit-Norm Calibration (UNC):** Independently trained experts suffer from representation scale imbalances. UNC is a training-free spatial normalization technique that projects representations and classification weights onto the unit hypersphere, equalizing feature scales.
- **Class-Size Scaling Calibration:** Addresses the statistical bias of maximum cosine similarity towards experts with larger output vocabularies ($C_k$) by scaling similarities using a theoretical extreme-value divisor $\sqrt{2\log C'_k / D}$.
- **Decentralized Subspace Complement Projection (DSCP) & SVD Orthogonalization:** Introduces training-free weight-space preprocessing methods (centralized joint SVD or decentralized DSCP) to orthogonalize expert adapters, ensuring absolute representation-space insulation under high cross-task leakage without sequential batch partitioning.
- **Transition-Aware Sequence Routing (TSVHA & DGR):** Extends non-parametric routing to generative autoregressive LLMs by mapping high-cardinality vocabularies to low-cardinality task spaces and using a prediction entropy change EMA monitor ($\Delta H$) to trigger out-of-period gate resets instantly.

## 2. The 'Delta' from Prior Work
The paper positions its main delta against two main branches of literature:
1. **Dynamic Parameter Merging (e.g., AdaMerging, QWS-Merge):** These methods suffer from "heterogeneity collapse" when deployed on mixed-task streams because they force a single batch-wide weight compromise. The delta is that PFAB performs sample-wise activation-space blending, retaining the exact, fine-grained routing coordinates for each sample and completely avoiding collapse.
2. **Prior SOTA Systems-ML (PFSR + MBH):** MBH avoids collapse by dynamically partitioning heterogeneous streams into homogeneous micro-batches and dispatching them sequentially. The delta of PFAB is a complete architectural pruning of this database-level systems bloat. By shifting the blending to activation space, PFAB processes heterogeneous batches in a single parallel pass of standard PyTorch with constant $O(1)$ latency, delivering a massive 2.52$\times$ wall-clock speedup.

## 3. Characterization of Novelty
The novelty of this paper is **significant and conceptually profound**, serving as a prime example of applying Occam's razor to systems-ML co-design. 
Rather than proposing increasingly complex, parameterized networks or adding heavy database scheduling and scheduling dependencies to patch neural network limitations, this paper achieves superior serving performance through standard, hardware-agnostic tensor operations (`torch.bmm`, `torch.einsum`).

It demonstrates that a simple, elegant mathematical shift in representation space can render complex engineering infrastructures redundant. It bridges the gap between systems efficiency (single forward pass) and model capacity (retaining specialized isolated experts) without introducing any learnable parameters, presenting a clean and refreshing direction for multi-task expert serving.
