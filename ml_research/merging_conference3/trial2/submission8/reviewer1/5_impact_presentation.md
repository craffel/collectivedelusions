# 5. Impact and Presentation

## Major Strengths of the Paper
1. **Outstanding Pragmatic Utility:** 
   The proposed framework is completely training-free, operates post-hoc, and runs in milliseconds due to $O(d \log d)$ sorting-based magnitude pruning. It enables storing task vectors in highly compressed formats like CSR or COO, reducing storage footprints by 10$\times$ to 20$\times$. This provides a massive, immediate solution to the storage, memory, and bandwidth bottlenecks in Edge AI and IoT deployments.
2. **Deep Rigor in Mathematical Derivations:** 
   Unlike many empirical model-merging papers, this work includes a thorough, formal mathematical analysis in **Appendix A**. It derives expected $L_1$ norm ratios under Laplace and Gaussian update distributions and analyzes expected $L_2$ reconstruction error, laying solid theoretical foundations for the "signal-strength boost" heuristic.
3. **Rigorous and Unbiased Evaluation:** 
   The authors ensure high statistical rigor (evaluating across 3 random seeds with reported standard deviations) and absolute evaluation fairness. By individually sweeping and optimizing the merging coefficient $\lambda$ for both **TIES-Merging** and **DARE-Merging** baselines, they prevent baseline-handicapping and establish highly trustworthy results.
4. **Honest and Counter-Intuitive Scientific Insights:** 
   The authors deserve high praise for reporting and analyzing a negative/counter-intuitive result: training-stage loss landscape flatness (SAM) does *not* provide an additional coordinate-aligned pruning buffer compared to standard AdamW under well-converged regimes. This helps separate isotropic loss landscape geometry from unstructured coordinate-wise sparsification.
5. **Practical Synergy with Quantization:** 
   By demonstrating that global Uniform Pruning (NP-BTVP-U) works seamlessly with INT8 quantization (achieving a **40$\times$ storage compression** with only **0.12%** accuracy degradation under SAM) while layer-wise Saliency Pruning collapses catastrophically due to noise amplification, the paper provides a complete, deployment-ready blueprint for Edge AI practitioners.

## Areas for Improvement
1. **Scale of Empirical Evaluation:** 
   The empirical validation is focused on a pre-trained CLIP ViT-B/32 backbone fine-tuned on 28.7 million parameters across small disjoint datasets (1024 samples). While the statistical and mathematical rigor is excellent, testing the framework on larger architectures (e.g., LLaMA-7B or ViT-L) on standard benchmarks would strongly reinforce the scalability of their conclusions.
2. **Conceptual Refinement of the Name "Norm-Preserved":** 
   Since the $1/p$ scaling factor actually boosts the expected $L_1$ norm by $2.58\times$ to $3.30\times$ (due to scaling the largest sorted parameters), the name "Norm-Preserved" is technically a misnomer. While the authors openly discuss this in Section 3.3 and Appendix A, referring to it as **"Norm-Scaled"** or **"Signal-Boosted" Budgeted Task-Vector Pruning** would reflect the mathematical reality more accurately.
3. **Open-Source Code Release:** 
   The paper does not mention a public repository link for their codebase. For a paper emphasizing practical utility and edge deployment, providing a ready-to-use open-source library (or integrating with popular tools like Hugging Face or `mergekit`) would maximize its adoption and real-world impact.

## Overall Presentation Quality
The presentation quality is **Excellent**:
* The paper is extremely well-written, clearly structured, and easy to follow.
* The transition from task-vector extraction to SAM optimization, pruning formulations, and empirical sweeps is logical and seamless.
* The terminology is precise, and the mathematical notation is standard and consistent.
* The figures and tables are clean, informative, and perfectly labeled, summarizing high-dimensional sweeps clearly (e.g., the pruning resilience curves and baselines comparison).

## Potential Impact and Significance
The potential impact of this work is **Very High**:
* **For Edge AI and IoT Practitioners:** This paper provides an incredibly simple, robust, and highly effective pipeline to compress specialized multi-task expert models by 10--40$\times$ with almost zero loss in accuracy. This enables on-device expert switching and over-the-air expert updates on severely bandwidth-constrained systems, which is of massive industrial interest.
* **For the Model Merging Community:** The "Saliency Double-Bind" analysis and the discovery of the geometric separation between loss landscape flatness (SAM) and coordinate sparsification resilience are highly significant. These insights will steer future research away from overly complex layer-wise coefficient search and Hessian calculations, guiding the community toward simpler, scale-harmonic uniform scaling solutions.
