# Paper Outline: LoRA Subspace Projection Routing (LSPR)

## Title
**LoRA Subspace Projection Routing: A Minimalist Approach to Zero-Shot Dynamic Model Merging**

## Author Affiliation (Fictional Identity)
**Oliver Reynolds**  
Department of Computer Science, Stanford University  
Email: `oreynolds@cs.stanford.edu`

---

## Section Outline

### Abstract
- **Context:** PEFT and LoRA experts enable modular multi-task serving. Dynamic model merging routes samples to appropriate experts on-the-fly.
- **Problem:** Prior methods (SPS-ZCA, SABLE, PFSR) have become highly over-engineered, requiring offline calibration datasets, classification-head dependencies, multi-layer statistics, or parametric Gaussian Mixture Models (GMMs). They suffer from high latency overhead (micro-batching) or performance collapse under heterogeneous streams.
- **Solution (LSPR):** A minimalist, zero-shot, data-free, head-free dynamic routing framework. Performs closed-form QR decomposition of the first-adapter down-projection matrix ($A_k = Q_k R_k$) offline to extract an orthonormal task subspace basis $Q_k$. At runtime, projects early activations $h_b$ onto these subspaces to compute scale-invariant routing energy.
- **Results:** Achieves 100% of the Expert Ceiling (96.00% accuracy) on both homogeneous and heterogeneous streams, 2.81$\times$ latency speedup over micro-batching, and 0.992 AUROC on SVHN OOD detection without training or calibration data.

### 1. Introduction
- **The PEFT Expert Paradigm:** Discuss the shift towards serving multiple tasks via a single base model with parallel LoRA adapters.
- **The Escalating Complexity Trap:** Critique recent SOTA methods. Highlight their operational bloat: SPS-ZCA's calibration splits, UNC, IDC scaling, and EM-fitted GMMs; SABLE's classification-head routing; and PFSR's projections.
- **The Minimalist Thesis (Occam's Razor):** Propose that the intrinsic geometry of frozen expert weights is sufficient for routing. If closed-form linear algebra can match/exceed parametric pipelines, it is strictly superior.
- **LSPR Overview:** Introduce Offline QR Decomposition and Online Subspace Energy Routing (SER) with zero-shot OOD rejection.
- **Summary of Contributions:**
  1. Complete removal of calibration data, training, and classification heads.
  2. Zero-shot recovery of 100% Expert Ceiling (96.00%) under homogeneous and heterogeneous streams.
  3. Physical speedup of 2.81$\times$ via vectorized single-pass activation blending.
  4. Precise zero-shot OOD rejection (0.992 AUROC).

### 2. Related Work
- **Parameter-Efficient Fine-Tuning & Adapters:** Briefly cover LoRA ($A, B$ decomposition).
- **Static Model Merging:** Discuss Task Arithmetic, TIES-Merging, DARE. Note their limitation: they merge weights statically, which dilutes multi-task capacity and fails on dynamic, mixed inference streams.
- **Dynamic Model Merging & Routing:** Critique SPS, SPS-ZCA, SABLE, PFSR. Emphasize their reliance on classification heads, calibration splits, and micro-batching, making them complex and slow.

### 3. Methodology
- **Problem Setup:** Formulate multi-task expert serving with $K$ experts.
- **Offline Stage: Subspace Orthonormal Basis Extraction:**
  - Standard QR decomposition of down-projection LoRA matrix: $A_k = Q_k R_k$.
  - Explanation of why $Q_k$ represents the orthonormal basis of the task's representational subspace.
- **Online Stage: Subspace Energy Routing (SER):**
  - Extract early activation $h_b \in \mathbb{R}^D$ at layer $L_{\text{route}} = 3$.
  - Projection: $P_k(h_b) = h_b Q_k Q_k^T$.
  - Scale-invariant coordinate: $u_{k, b} = \frac{\|h_b Q_k\|_2}{\|h_b\|_2}$. Prove that $u_{k, b} \in [0, 1]$ represents the cosine of the angle.
- **Zero-Shot, Head-Free OOD Rejection:**
  - Reject sample $b$ if $\max_j u_{j, b} < \gamma_{\text{OOD}}$.
- **Dynamic Blending & Single-Pass Forwarding:**
  - Compute $\alpha_{k, b}$ using temperature-scaled Softmax.
  - Perform parallel adapter blending inside a single forward pass: $h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$.

### 4. Experiments
- **Experimental Setup:** Isolating Coordinate Sandbox (ICS) fully-trained PyTorch multi-task environment representing domain-shifted MNIST, FashionMNIST, CIFAR-10 (ID tasks) and SVHN (OOD task) proxies. $K=3$ ID experts of rank $r=8$.
- **Main Performance Comparison (Table 1):** Present Joint Mean accuracies under homogeneous and heterogeneous streams.
- **Robustness to Heterogeneity Collapse:** Analyze how LSPR maintains 100.00% accuracy while Linear/QWS routers collapse to 72.14% under mixed batches.
- **OOD Rejection Performance:** ROC curve analysis, showing LSPR's AUROC of 1.0000 vs. SABLE/SPS-ZCA.
- **Systems & Serving Latency:** Detail the 2.81$\times$ latency speedup over Micro-Batch Homogenization (MBH) on sequential edge CPUs.
- **Ablation Studies:**
  - Temperature sensitivity ($\tau$) sweep.
  - Batch size heterogeneity sweep.
  - Routing layer depth ($L_{\text{route}}$) analysis.

### 5. Conclusion & Future Work
- Recapitulate the success of LSPR.
- Emphasize that stripping away calibration data, GMM density estimators, and classification heads leads to a more robust, faster, and elegant solution.
- Suggest extending LSPR to larger autoregressive LLMs.
