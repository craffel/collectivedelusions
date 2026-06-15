# Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is exceptionally well-written, logically structured, and mathematically rigorous. The descriptions of both the mathematical formulation and the systems-level implementation are clean and accessible:
* **The Mathematical Formulation (Section 3):** The transition from raw cosine similarity (Eq. 1) to statistical class-size calibration (Eq. 2), temperature Softmax gating (Eq. 3), and micro-batch partitioning (Eq. 5-7) is seamless and highly intuitive. 
* **The Systems-Level Co-design (Section 4 & 5.4):** The paper details the exact hardware footprint, memory scaling, and execution strategies (activation-space vs. parameter-space merging), making it highly clear how the theoretical model maps to actual hardware.
* **Algorithm 1:** Formulates the entire joint PFSR + MBH + UNC framework in a highly precise, step-by-step pseudocode, ensuring that the entire pipeline can be easily understood and implemented.

## Appropriateness of Methods
The proposed methods are highly appropriate and address the targeted failure modes directly:
* **PFSR for OOD Overfitting:** Removes trainable routing layers completely, eliminating the possibility of optimization overfitting and transductive OOD collapse.
* **MBH for Heterogeneity Collapse:** Solves batch-level mixed-task interference at the data-stream level. This is a highly appropriate co-design choice because the "averaging" of routing coefficients across a heterogeneous batch is a hardware/inference-engine constraint, not a fundamental model constraint. Shifting the solution to the data-orchestration layer is both elegant and effective.
* **Unit-Norm Calibration (UNC):** Prior to similarity projection, normalizing both features and classifier weights is mathematically essential to ensure scale-invariance across independently trained models.
* **Class-Size Scaling Calibration ($O(\sqrt{\log C_k / d})$):** A mathematically rigorous correction for asymmetrical label spaces. It correctly models the asymptotic behavior of the maximum of random Gaussian similarities in high dimensions.
* **Sub-Vocabulary Prototype Selection:** Pruning LLM vocabularies based on classification weight variance across experts is highly appropriate as a data-free, parameter-centric optimization. It successfully slashes gating latency by over $130\times$ without compromising task specificity.

## Potential Technical Flaws & Boundaries
The paper is remarkably honest and thorough in discussing its own boundary conditions and potential limitations, which is a testament to its strong scientific integrity:
1. **Representational Drift under Full Fine-Tuning:** PFSR assumes a shared, aligned penultimate feature space. If experts undergo full fine-tuning with large learning rates, their representation manifolds drift apart, making their classification heads incompatible. The authors thoroughly analyze this boundary in Section 5.5, measuring representation distance across learning rates, and propose concrete mitigations (e.g., representation alignment objectives $\mathcal{L}_{align}$).
2. **Generative & Regression Tasks (No Classification Heads):** For domains without class prototypes (e.g., diffusion models or continuous regression), the raw formulation of PFSR is undefined. The authors explicitly acknowledge this boundary and propose an unsupervised $K$-means clustering alternative ($M_{k,p}$) on a tiny calibration split, validating it quantitatively on the sandbox (Table 5.8).
3. **Sequential Dispatch Latency on Edge CPUs:** Sequential execution of $G \le K$ micro-batches under highly mixed streams can create a latency bottleneck on low-power CPUs. The authors address this by providing a comprehensive deployment matrix (Table 5.7) and recommending a clear hierarchy of edge-friendly mitigations (e.g., hard Top-1 routing fallback, which guarantees a single forward pass).
4. **Systems-Level Complexity Shifting:** PFSR+MBH successfully eliminates model parameter bloat but shifts the engineering burden to the underlying data-serving infrastructure (dynamic partitioning, indexing, re-sorting, parallel SGMV kernels). The authors are highly transparent about this trade-off (Section 5.4, Table 5.7).

## Reproducibility
The reproducibility of the paper is **excellent**:
* All mathematical equations are written out in full, explicit detail.
* Pseudocode is provided for the entire pipeline in Algorithm 1.
* All hyperparameters (e.g., temperature $\tau=0.001$, sub-vocabulary size $C_{sub}=256$, batch sizes, layer counts, dimensions) are explicitly disclosed.
* The variance-based token selection heuristic (Eq. 4) is completely deterministic and data-free, meaning any researcher can replicate the LLM token selection in milliseconds using only the model weights.
* A qualitative audit of the selected LLM tokens is provided (Table 5.1), allowing direct validation of the heuristic's behavior.
* Clear hardware benchmarks and specifications (A100 GPU, Xeon CPU) are reported, ensuring the systems-level throughput and latency speedups are verifiable.
