# 2. Novelty and Originality Check

## Assessment of Originality
This paper is an exceptional **methodological deconstruction and scientific audit** of an existing state-of-the-art (SOTA) framework (SAIM). While it does not introduce an entirely new machine learning model or optimization algorithm from scratch, its originality and value to the machine learning community are outstanding and fully aligned with the definition of originality in the reviewing criteria:
> "Originality may arise from creative combinations of existing ideas... or removing restrictive assumptions... or providing novel insights by evaluating existing methods."

The specific novel contributions of this paper include:
1. **Multi-Axial Deconstruction Grid**: The systematic $5 \times 3$ crossing of optimizers and merging strategies is a highly original and rigorous way to decouple the causal drivers of performance. It avoids the common pitfall of evaluating multi-component frameworks as a single monolithic package.
2. **Boundary-Condition Analysis ($\lambda = 0.0$ vs. $\lambda = 0.2$)**: The paper identifies and formalizes how the performance of SVD-based Isotropic Merging is highly sensitive to the parameter-mixing boundary conditions. It reveals that SVD is a distortive operator on un-mixed parameters ($\lambda=0$), but a helpful spectrum regularizer under active mixing ($\lambda=0.2$).
3. **Exposing the Coordinate Selection Bottleneck**: The paper provides a novel empirical and practical analysis showing that coordinate-restricted sharpness optimization (SA-BCD) is not only mathematically suboptimal but also computationally inefficient on modern GPUs (slowing training by 18.5%) due to indexing and masking operations that disrupt GPU thread-coalescing.
4. **Integration of Scale-Calibrated and Modern Weight-Consolidation Baselines**: Rather than comparing SVD only to collapsing baselines, the authors introduce a Scale-Calibrated baseline to isolate the singular-value flattening mechanism from norm preservation, and cross SAM with TIES-Merging and DARE to demonstrate a massive, novel "structural synergy" between training-stage flatness and post-hoc sparsification.
5. **Practical PEFT Generalization (LoRA-SAM)**: The theoretical and empirical extension of loss landscape flatness to parameter-efficient fine-tuning (LoRA-SAM) is a highly original, forward-looking part of the paper (Section 5). The authors show that LoRA-SAM is exceptionally computationally feasible (less than 2.5% wall-clock overhead and $<1.5\%$ VRAM overhead compared to standard LoRA) and enables flawless post-hoc merging via naive Task Arithmetic ($74.12\%$ ACC), offering a practical and scalable SVD-free alternative for large-scale models.
6. **Empirical SVD Scaling Analysis**: Providing concrete SVD execution times (ms) on CPU and NVIDIA H100 GPUs across dimensions from $192$ to $4096$ is highly original. It translates theoretical complexity ($O(d^3)$) into actionable system-level insights that guide real-world deployment decisions.
7. **Empirical Scale Validation on ViT-Base (86M parameters)**: Testing the core flatness claims on a larger ViT-Base backbone and presenting it in Table 3 successfully confirms that the performance boost of training-stage flatness holds at a larger scale ($+3.89\%$ accuracy improvement), verifying the structural robustness of the findings.

## Positioning relative to Prior Work
The paper is excellently positioned in the context of:
- **Model Merging & Weight Averaging** (e.g., Model Soups, Task Arithmetic, TIES-Merging, DARE, OrthoMerge).
- **Sharpness-Aware Minimization & Loss Flatness** (e.g., SAM, SWA, Linear Mode Connectivity).
- **Continual Learning via Sequential Merging** (e.g., SyMerge, SAIM).

The authors clearly distinguish their work from prior literature. Instead of proposing yet another complex merging technique, they show that standard, established optimization methods (like SAM) can achieve equal or superior results when paired with simple merging techniques, thereby simplifying the model-merging pipeline.

## Suggestions to Enhance Originality and Value
While the paper represents an outstanding and highly complete piece of research, the authors could further enhance its originality and scientific impact with the following suggestions:

1. **Incorporate SVD Latency Benchmarks for LoRA matrices**:
   - In Section 5, the authors explain that SVD isotropic merging is mathematically redundant on low-rank adapters because they are already low-rank (typically $r \le 16$).
   - To empirically back up this claim, the authors could include a minor discussion or a quick benchmark of SVD on low-rank matrices (e.g., of dimensions $r \times r = 8 \times 8$ or $d \times r = 4096 \times 8$), demonstrating that SVD is virtually instantaneous at this scale (fraction of a millisecond) compared to the $O(d^3)$ complexity of full-parameter matrices ($4096 \times 4096$), further justifying why LoRA-SAM is highly scalable.
2. **Discuss Parallel Multi-Task Merging Applicability**:
   - The paper focuses on sequential merging for continual learning. However, many model merging frameworks are used for parallel, multi-task merging (e.g., merging fine-tuned experts from different tasks into a single multitask model).
   - The authors can discuss or briefly analyze whether their findings (flatness as a foundational driver, SVD redundancy under certain regimes) apply to parallel multi-task merging, broadening the paper's applicability and further enhancing its value to the wider model-merging community.
3. **Analyze the Sensitivity to Task Order in Sequential Merging**:
   - Since the task stream consists of 5 sequential tasks on Split CIFAR-100, the final performance could depend on the specific sequence of tasks.
   - The authors could briefly discuss whether training-stage flatness (SAM) makes the sequential merging process less sensitive to task ordering compared to AdamW, adding a valuable new dimension to their deconstruction.
