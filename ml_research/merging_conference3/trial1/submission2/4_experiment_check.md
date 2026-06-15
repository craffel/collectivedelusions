# 4. Experimental Check

## 1. Assessment of Empirical Quality
The empirical section of the paper is **outstandingly comprehensive, well-structured, and methodologically sound**. The authors have executed a highly systematic, multi-axial crossing of 5 optimizers and 3 merging strategies, split into two distinct weight-mixing regimes ($\lambda=0.0$ and $\lambda=0.2$).

### Major Empirical Strengths:
- **Systematic Multi-Axial Evaluation**: Crossing every optimizer with every merging strategy successfully decouples the effects of training-stage optimization (finding flat basins) from post-hoc weight consolidation.
- **Robust Empirical Scale Validation**: The addition of the **ViT-Base** (86M parameters) validation in Table 3 successfully confirms that the core conclusions hold at scale. Transitioning to SAM + Task Arithmetic boosts accuracy from **86.18%** to **90.07%** ($+3.89\%$), demonstrating that the benefits of optimizer-driven flatness are highly generalizable and robust to model capacity scaling.
- **Rigorous Integration of Weight Consolidation Baselines**: Crossing primary optimization regimes with established post-hoc weight-consolidation methods (**TIES-Merging** and **DARE**) in Table 2 reveals a powerful structural synergy: training experts to reside in flat, wide basins (via SAM) makes their parameters structurally resilient to post-hoc pruning and sign-consensus, leading to massive accuracy boosts (+16.89% for SAM + DARE over AdamW + DARE).
- **Scale-Calibrated Baseline**: Table 2 includes a Scale-Calibrated baseline that eliminates compounding scale shrinkage. Showing that it still underperforms SVD Isotropic Merging successfully isolates the precise, selective singular-spectrum flattening mechanism of SVD.
- **Consistent Error Bars**: Standard deviations are reported over 3 random seeds for every single configuration in Table 1 and Table 2.
- **Real-World GPU Wall-Clock Benchmark**: Table 1 reveals that coordinate-restricted sharpness optimization (SA-BCD) is 18.5% *slower* than globally perturbed SAM, and Table 5 benchmarks SVD execution times (ms) on CPU and NVIDIA H100 GPUs up to $4096 \times 4096$ dimensions. These benchmarks convert theoretical complexity claims into concrete, system-level warnings about post-hoc SVD latencies.
- **Rigorous PEFT (LoRA-SAM) Validation**: Section 5 reports LoRA-SAM (rank $r=8$) on ViT-Tiny in Table 4. Crucially, the authors include the standard LoRA-AdamW baselines under both Task Arithmetic and Isotropic Merging, which provides a pristine experimental control. It shows that low-rank flatness optimization yields $74.12\%$ average accuracy under naive Task Arithmetic, with negligible ($<2.5\%$) training overhead, demonstrating outstanding PEFT merging feasibility.
- **Visual Sensitivity Plots**: The inclusion of Figure 2 (`fig:sensitivity`) in the Appendix provides concrete, visual sensitivity plots of the perturbation radius $\rho$ and coordinate selection ratio $p$, which makes the findings highly intuitive.

---

## 2. Experimental Gaps and Areas of Improvement
Despite these impressive empirical strengths, there are minor experimental suggestions that, if addressed, would make the paper's claims completely indisputable:

### Suggestion 1: Discuss Cross-Domain NLP Generalization
- **The Issue**: The paper's empirical results are focused on computer vision (Split CIFAR-100 with ViT). 
- **The Suggestion**: To expand the generalizability of their deconstruction, the authors could discuss the feasibility and experimental design of verifying these findings in the Natural Language Processing (NLP) domain. For example, sequentially fine-tuning a BERT-Base model on GLUE tasks (such as SST-2, QQP, and MNLI) using BERT-SAM, and subsequently merging their weights under active parameter conflict ($\lambda > 0$). This would provide excellent cross-domain generalizability and guide NLP practitioners.

### Suggestion 2: SVD Benchmarks under low-rank adapters
- **The Issue**: In Section 5, the authors explain that SVD isotropic merging is redundant on low-rank adapters because they are already low-rank.
- **The Suggestion**: To empirically reinforce this, the authors could include a minor discussion or quick benchmark of SVD execution times on small low-rank matrices (e.g., $8 \times 8$ or $4096 \times 8$). This would show that SVD computation at this scale is virtually instantaneous (fraction of a millisecond) compared to full-parameter matrices ($4096 \times 4096$), further justifying LoRA-SAM's scalability.

### Suggestion 3: Discuss Generalization to Parallel Multi-Task Merging
- **The Issue**: The paper focuses on sequential merging for continual learning. However, many model merging frameworks are used for parallel, multi-task merging (e.g., merging fine-tuned experts from different tasks into a single multitask model in a parallel fashion).
- **The Suggestion**: The authors should discuss or briefly analyze whether their findings (flatness as a foundational driver, SVD redundancy under certain regimes) carry over to parallel multi-task merging where no chronological ordering or $t$ factor exists, broadening the paper's applicability.

### Suggestion 4: Report Statistical Variance (Multiple Seeds) for Scale Validation (Table 3)
- **The Issue**: Unlike Table 1, Table 2, and Table 4, which report standard deviations over 3 random seeds, the scale validation on ViT-Base (Table 3) only reports single-seed results.
- **The Suggestion**: While running ViT-Base (86M parameters) is computationally heavy, reporting statistical variance or at least acknowledging the single-seed nature of Table 3 as a limitation would improve the empirical completeness of the paper. If compute is available, running 3 seeds for Table 3 would make the scaling claims completely indisputable.
