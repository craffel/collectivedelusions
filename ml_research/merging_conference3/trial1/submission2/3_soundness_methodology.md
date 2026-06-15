# 3. Soundness and Methodology Check

## 1. Assessment of Technical Soundness
The methodology, mathematical derivations, and experimental design presented in this paper are **highly rigorous, logically consistent, and technically sound**. The authors have done an exceptional job of deconstructing the mathematical components of the SAIM framework and validating their critiques.

### Outstanding Strengths in Soundness:
- **Flawless Deconstruction of SA-BCD's Algebraic Bug**: The mathematical identification of the error in SA-BCD's literal formula is robust. Multiplying the Adam-style step (which already scales the step size and provides direction) by the raw perturbed gradient $g'_{t,i}$ again forces the update direction to be proportional to the square of the gradient ($[g'_{t, i}]^2$). This is a solid mathematical proof of why the "literal" optimizer causes complete divergence, which is empirically confirmed by a random-chance accuracy of ~4.5%.
- **Elegant Analysis of Norm-Matching's Compounding Shrinkage**: The high-dimensional geometric proof of why the Norm-Matching baseline collapses under sequential merging (Appendix A.3) is brilliant. Assuming near-orthogonality of high-dimensional sequential updates ($\langle \Delta_{\text{cum}}, \Delta_{T_t} \rangle \approx 0$), the combined update's norm is mathematically $\sim 1.442N$, while the target norm is $N$, leading to a per-step shrinkage factor of $\sim 0.693$. Over 5 tasks, this compounds to a cumulative update shrinkage of $\sim 0.231$ (under 24% of proper magnitude). This mathematically explains the performance collapse and separates SVD's actual singular-value flattening mechanism from simple norm-preservation.
- **Systematic, Decoupled Evaluation Grid**: The multi-axial $5 \times 3$ evaluation grid crossing 5 optimizers and 3 merging strategies is methodologically pristine. It successfully isolates whether empirical gains are driven by finding flatter minima (during training) or post-hoc weight alignment (during merging).
- **Thorough Resolution of Previous Methodological Criticisms**:
  - **Scale-Calibrated Baseline**: The authors designed and evaluated a Scale-Calibrated baseline that rescales the combined update to match the Frobenius norm of the current task expert $\|\Delta_{T_t}\|_F$ at each step. This directly eliminates the compounding scale shrinkage. Showing that this baseline still underperforms (55.13% vs. SVD's 76.42% under SAM) provides mathematically solid proof that SVD Isotropic Merging's success is due to selective *singular value spectrum flattening* rather than simple global scale adjustments.
  - **Modern Baselines (TIES/DARE)**: The authors integrated established, modern weight consolidation techniques into their grid, proving SVD's capabilities against strong, relevant baselines.
  - **Complete Statistical Variance**: Every single row in Table 1 (the 15-configuration scoreboard) and Table 2 (active weight-mixing results) includes standard deviations over 3 random seeds, meeting the highest engineering standards for empirical reporting.
  - **Robustness under Scale**: Conducting the key baseline experiments on **ViT-Base** (86M parameters) in Table 3 successfully confirms that the core conclusions remain sound as parameter capacity scales by over $17\times$, resolving any scaling concerns.
  - **Complete PEFT Baselines (Section 5)**: In Section 5.2 and Table 4, the authors include the standard LoRA-AdamW baselines and a dedicated table for PEFT model merging. This completely isolates the causal benefits of flatness (LoRA-SAM) on low-rank manifolds, addressing previous baseline concerns.

---

## 2. Methodology Flaws and Potential Areas of Improvement
While the paper is exceptionally strong, there are a few minor areas of reasoning and methodology that could be further refined:

### Critique 1: Computational Overhead of Full-Parameter SAM
- Standard full-parameter SAM requires a double-backward pass per iteration to compute sharpness-aware perturbations. This effectively doubles both the training wall-clock time and the compute costs during the fine-tuning phase of each task-expert.
- While the authors highlight that coordinate selection (SA-BCD) is 18.5% slower than globally-perturbed SAM due to GPU serialization bottlenecks, standard SAM itself still represents a $2\times$ training cost overhead over standard AdamW. The authors should discuss the practical cost-benefit trade-off of full-parameter SAM more extensively, or suggest methods to reduce its compute overhead (e.g., executing sharpness updates only every $k$ steps or restricting SAM updates to a subset of layers).

### Critique 2: Generalization of SVD Decay Schedule ($1/\sqrt{t}$)
- In Appendix A.4, the authors evaluate linear decay ($1/t$) and constant scale ($\gamma_t = 0.5$) schedules, concluding that $1/\sqrt{t}$ acts as an elegant middle ground that matches the standard Brownian scaling of random updates.
- While this is an insightful observation, the analysis is purely empirical on a single model (ViT-Tiny). It remains unclear whether the $1/\sqrt{t}$ schedule holds as a universal default across other architectures (e.g., LLMs) or if the decay schedule is heavily dependent on task-stream length $t$.
- **Actionable Suggestion**: The authors should add a brief sentence in the Appendix suggesting how the decay schedule might be scaled for extremely long streams (e.g., introducing a hyperparameter exponent $\beta$ as in $1/t^\beta$) to give practitioners guidance on handling longer task trajectories.

### Critique 3: Exploration of low-rank adapter choices in PEFT
- In Section 5, the authors target query, key, and value projection layers in all self-attention blocks for LoRA.
- Since other layers (such as feed-forward MLP blocks) also play a key role in representation learning, it remains unclear whether expanding LoRA-SAM to MLP blocks affects linear mode connectivity or structural flatness.
- **Actionable Suggestion**: The authors could add a brief comment or recommendation on whether practitioners should restrict LoRA-SAM to self-attention blocks or apply it to MLP blocks as well, further improving the practical value of Section 5.

### Critique 4: Sensitivity to Task Ordering in Continual Merging
- In sequential merging for continual learning, task order is a major confounding factor that can significantly affect performance and forgetting due to catastrophic interference. The authors evaluate their framework on a single task order.
- It remains unclear whether optimizer-driven flatness (SAM) reduces the model's sensitivity to the task fine-tuning order during sequential weight averaging compared to standard AdamW.
- **Actionable Suggestion**: The authors should briefly discuss this task-order sensitivity and hypothesize whether wide, flat basins make consolidated models more robust to task-order variations.
