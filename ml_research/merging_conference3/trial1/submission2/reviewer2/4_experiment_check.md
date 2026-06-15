# Experimental Evaluation Critique

## Evaluation of Experimental Setup
- **Controlled Environment:** The choice of task-incremental continual learning with oracle task-specific classification heads is highly appropriate. It successfully isolates the evaluation to focus strictly on backbone weight merging without being confounded by head interference or classification drift.
- **Ablation Axes:** The crossed multi-axial grid (5 optimizers $\times$ 3 merging strategies) is exceptionally rigorous, ensuring that no single component's effect is conflated with another.
- **Statistical Rigor:** Crucial primary configurations are averaged over 3 random seeds with reported standard deviations. This satisfies high empirical standards for checking variance and ensuring findings are statistically sound.

## Critique of Datasets and Baselines
- **Datasets:** Split CIFAR-100 is a standard, clean benchmark for classification. While small in capacity, it provides a highly controlled, fast-turnaround environment to perform massive grid searches that would be computationally prohibitive on massive datasets. The inclusion of a larger-scale vision transformer (ViT-Base, 86M parameters) validates that these findings scale effectively.
- **Exhaustive Baselines:** The baseline selection is incredibly strong and comprehensive:
  - It includes standard *AdamW* and standard *SAM*.
  - It includes two corrected variants of the coordinate-restricted optimizer (*SA-BCD Std Adam* and *SA-BCD Adam GT*).
  - It includes a simple scalar *Update Decay* baseline to isolate overall update scale shrinkage.
  - It introduces a *Norm-Matching* baseline and a *Scale-Calibrated* baseline to isolate SVD's spectral effects from simple global weight magnitude preservation.
  - It crosses optimizers with modern post-hoc weight consolidation baselines like *TIES-Merging* and *DARE* to contextualize the findings.

## Do the Results Support the Claims?
Yes, the results are highly convincing and directly support every central claim made in the paper:
1. **Claim: Flatness is the foundational driver of merging success.**
   - *Support:* Changing the optimizer from AdamW to standard SAM under Task Arithmetic provides a massive +9.87% ACC boost under sequential parity ($\lambda = 0.0$) and a +12.30% ACC boost under active mixing ($\lambda = 0.2$).
2. **Claim: SVD isotropic merging is boundary-condition-sensitive.**
   - *Support:* SVD consistently degrades performance under $\lambda = 0.0$ (e.g., from 68.31% to 61.33% for SAM), proving it is redundant/distortive without active mixing. Under active mixing ($\lambda = 0.2$), SVD isotropic merging successfully acts as a regularizer, boosting SAM from 73.83% to 76.42% and AdamW from 61.53% to 68.98%.
3. **Claim: SA-BCD is suboptimal and computationally inefficient.**
   - *Support:* SA-BCD (Std Adam) under Task Arithmetic ($\lambda = 0.0$) yields 62.94% ACC, falling 5.37% short of standard SAM (68.31%), yet requires 279.9s of training time compared to SAM's 236.1s (an 18.5% increase). This confirms that coordinate-restricted perturbation is less effective and computationally slower due to sorting and indexing bottlenecks.
4. **Claim: LoRA-SAM is a highly scalable, SVD-free alternative for PEFT.**
   - *Support:* LoRA-SAM + Task Arithmetic achieves 74.12% ACC (a +14.78% improvement over LoRA-AdamW). The addition of post-hoc SVD isotropic merging only improves this to 74.85% (a negligible +0.73% gain), proving SVD is redundant once optimizer flatness is established on a low-rank manifold. The profiled wall-clock (<2.5%) and VRAM (<1.5%) overhead is extremely low.
5. **Claim: Pre-merging flatness synergizes with and enables post-hoc pruning/consensus (DARE/TIES).**
   - *Support:* SAM + DARE yields 57.70% (+16.89% over AdamW + DARE's 40.81%) and shows remarkable robustness across a wide range of pruning rates $p_{\text{drop}} \in [0.1, 0.9]$. This directly supports Proposition 3.1, demonstrating that flat minima are structurally robust to pruning.
