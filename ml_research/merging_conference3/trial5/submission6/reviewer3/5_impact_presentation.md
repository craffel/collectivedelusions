# 5. Impact and Presentation

## Major Strengths

### 1. Thorough and Comprehensive Ablation Studies
The paper includes an exceptionally detailed set of ablation studies that explore almost every major dimension of the proposed framework. The authors systematically evaluate:
- Sensitivity to SVD truncation rank ($r \in \{4, 8, 16\}$).
- Zero-shot activation-mean initialization versus optimized routing using a Straight-Through Estimator (STE).
- Isolating SVD truncation error from routing error using a "Full-Rank + Top-1 Gating" baseline.
- Autonomous versus oracle classification head selection.
- Statistical robustness and sequence-ordering variance (using 5 random seeds).
- Quantitative routing jitter and layer-wise agreement analysis.
These ablations provide a clear view of the system's operational boundaries and behavior under different configurations.

### 2. Physical Edge Hardware Profiling
Unlike many model-merging papers that rely solely on theoretical FLOP calculations, the authors conduct physical execution profiling on a **Raspberry Pi 4** edge computer. They measure actual wall-clock latency (showing an $85.2\%$ latency reduction compared to weight-reconstruction baselines) and peak RAM utilization. This practical validation strongly supports their claims regarding edge-deployment feasibility.

### 3. Clear and Structured Writing
The paper is exceptionally well-written, polished, and structured. The mathematical formulations of the offline SVD, bounded cosine-similarity router, and parallel forward pass are presented clearly. The figures (Overview, Heterogeneity Collapse) and tables are professional and easy to interpret.

---

## Major Areas for Improvement

### 1. Resolve the Conceptual Misalignment
The authors must address the fundamental mischaracterization of their framework. **SLD-Merge is not a model merging method**; it is an activation-space **multi-LoRA Mixture-of-Experts (MoE)** framework. Because it keeps $K$ separate low-rank adapter pathways in memory and routes samples to them, it bypasses the core challenge of weight-space model merging (fusing parameters into a single set). The paper should be reframed to clearly position itself within the multi-adapter and MoE literature, and the claims regarding "resolving weight-space merging bottlenecks" must be toned down.

### 2. Evaluate on Standard-Scale, Converged Experts
Evaluating solely on tiny, 256-sample datasets with under-trained experts (such as the $29.3\%$ accurate SVHN model) is highly non-standard and acts as a major confounding variable. 
- The authors must evaluate SLD-Merge on standard, fully converged experts trained on full-scale datasets.
- This is critical to determine whether their highly suspicious claim that SVD truncation *outperforms* full-rank expert models (attributed to "implicit regularization") is a real scientific phenomenon or merely an artifact of severe overfitting in their artificial, low-data setup.

### 3. Address and Profile the Linear Complexity Scaling with $K$
The parallel PyTorch forward pass formulation:
$$Y = X W_{base}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
executes all $K$ low-rank adapters in parallel, scaling compute linearly with $K$.
- The authors must implement a truly sparse, conditionally executed forward pass (e.g., using scatter/gather) to ensure that compute is independent of $K$.
- They must provide a scalability profiling sweep showing FLOPs and latency as $K$ scales from $4$ to $50$ or $100$ tasks. This is necessary to validate their broad claims about the framework being "computationally lean" and suitable for scaling in production.

### 4. Compare with Multi-Adapter Routing and MoE Baselines
To demonstrate actual novelty, the paper must include comparisons to standard multi-LoRA routing and Mixture-of-Experts baselines, such as **LoRA-Hub, LoRA-MoE, or LLaVA-MoE**, rather than only comparing to weight-space model merging methods that are bound by completely different architectural constraints.

---

## Overall Presentation Quality
**Excellent.** The quality of the presentation, prose, mathematical notation, and visualizations is highly professional. The paper reads like a polished conference submission.

---

## Potential Impact and Significance
**Moderate to Low.**
- While the engineering execution and hardware profiling are impressive, the core conceptual contribution is highly incremental when viewed through the lens of existing multi-adapter and MoE literature.
- The severe linear scaling bottleneck with $K$ in their current implementation severely limits the framework's practical utility for large-scale multi-task suites.
- Unless the authors resolve the scalability bottleneck and validate their findings on standard-scale, fully converged experts, the paper's impact on the broader machine learning community will remain limited.
