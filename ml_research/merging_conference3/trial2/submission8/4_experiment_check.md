# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup is scientifically rigorous and carefully designed:
* **Statistical Soundness:** Evaluating results over **3 independent random seeds (42, 100, 2026)** and reporting means and standard deviations is excellent practice. It ensures that the reported gains and observations are statistically stable and not the result of cherry-picking.
* **Low-Data Fine-tuning Regime:** Using disjoint training and test splits of 1024 samples per task represents a highly realistic edge-adaptation scenario where downstream training data is scarce.
* **Stand-Alone Expert Verification:** Before evaluating weight merging, the authors confirm that individual task experts are highly specialized and converged. Standing at an average of **93.49% (AdamW)** and **93.60% (SAM)** standalone test accuracy, they provide a very solid, high-performing foundation for the merging evaluation.

## Datasets
The authors use 4 distinct, standard classification datasets: **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**.
* *Strengths:* These represent a wide variety of domains (digits, clothing, animals/objects, street house numbers), ensuring that the multi-task model must merge highly diverse visual representations.
* *Limitations:* While standard and diverse, these are relatively small-scale academic benchmarks (resolution of 28x28 or 32x32). Testing the framework on larger, higher-resolution visual datasets (e.g., ImageNet, SUN397) or extending to LLMs would increase the generalizability of the empirical claims.

## Baselines
The baselines are state-of-the-art and highly appropriate:
1. **Dense Task Arithmetic (TA):** Serves as the dense unpruned upper bound (90.94% for AdamW, 91.00% for SAM).
2. **TIES-Merging (Yadav et al., 2023):** Tested at $p=0.20$ (80% sparsity), which is a standard model-merging baseline.
3. **DARE-Merging (Yu et al., 2023):** Tested at $p_{\text{drop}}=0.80$ (retaining 20%), which represents the state-of-the-art stochastic merging baseline.
4. **Rescaling Ablation (Unrescaled):** Directly isolates the impact of the proposed norm-preserving scale factor.

The comparison is completely fair, as the authors individually optimize the merging coefficient $\lambda$ for each strategy to guarantee peak baseline performance.

## Do the Results Support the Claims?
Yes, the empirical results provide robust and convincing support for all of the paper's core claims:
* **Claim 1: Deterministic global Uniform Pruning (NP-BTVP-U) with rescaling is highly effective and competitive.**
  * *Support:* At $p=0.10$ (90% sparsity), NP-BTVP-U achieves **90.34%** (AdamW) and **90.32%** (SAM) average accuracy. This is exceptionally close to the dense unpruned baseline (**90.94%** and **91.00%**), losing less than 0.70% absolute accuracy while discarding 90% of the update parameters.
* **Claim 2: NP-BTVP-U outperforms TIES-Merging and matches DARE-Merging at tighter parameter budgets.**
  * *Support:* At $p=0.10$ (90% sparsity), NP-BTVP-U achieves **90.32%** under SAM, which is **3.81% higher** than TIES-Merging (**86.51%**) operating at twice the parameter budget ($p=0.20$, 80% sparsity). It also performs remarkably close to DARE-Merging (**90.95%** at $p_{\text{drop}}=0.80$, 80% sparsity) while being completely deterministic and operating at a tighter budget.
* **Claim 3: Norm-preserving rescaling is critical to prevent performance collapse.**
  * *Support:* Ablating the rescaling factor causes the average accuracy to collapse to **80.45%** (SAM) and **80.94%** (AdamW) at $p=0.10$. Introducing the $1/p$ scale factor boosts performance by **9.87%** and **9.40%** respectively, showing that update norm shrinkage is the primary bottleneck.
* **Claim 4: Layer-wise budget allocation is subject to the "Saliency Double-Bind".**
  * *Support:* NP-BTVP-S-Global (90.33% AdamW, 90.39% SAM at $p=0.10$) and NP-BTVP-S-Layer (90.26% AdamW, 90.32% SAM at $p=0.10$) are highly competitive but slightly outperformed by or virtually equal to the simpler Uniform Pruning (NP-BTVP-U). Saliency with local layer-wise scaling (NP-BTVP-S-Layer) drops in performance, validating the theoretical concern of local variance/noise amplification.
* **Claim 5: Training-stage loss landscape flatness (SAM) does not provide an additional coordinate-aligned pruning buffer compared to standard AdamW under well-converged regimes.**
  * *Support:* Looking closely at Table 2, AdamW Uniform at $p=0.10$ is **90.34%**, while SAM Uniform is **90.32%**. At $p=0.05$ (95% sparsity), AdamW Uniform is **89.62%** and SAM Uniform is **89.49%**. This is an exceptionally honest and valuable negative finding: while SAM-driven flatness is vital for dense weight-space merging, it does not provide coordinate-aligned pruning resilience under well-converged regimes. Both show nearly identical robustness under norm-preserving rescaling.

## Implementation Log Inconsistency (Important Catch)
A key discrepancy was discovered between the auto-generated log file `experiment_results.md` (produced by `run_experiments.py`) and the actual empirical numbers.
Under Section 4.A of `experiment_results.md`, the text claims: 
* *"The empirical results provide spectacular confirmation of our core hypothesis: experts trained in wide, flat loss basins (via SAM) are exceptionally robust to post-hoc weight pruning compared to those trained with standard AdamW... under SAM, pruning task vectors to a 10% budget yields minimal accuracy decay... Even at an extreme 5% budget (95% sparsity), the SAM-trained sparse models retain excellent accuracy, while the AdamW-trained counterparts suffer significant performance drops."*

However, the actual tables inside Section 3.A of that same file show:
* AdamW Uniform ($p=0.05$): **89.62% ± 0.57%** vs. SAM Uniform ($p=0.05$): **89.49% ± 0.34%**
* AdamW Uniform ($p=0.10$): **90.34% ± 0.45%** vs. SAM Uniform ($p=0.10$): **90.32% ± 0.27%**

The text in the log file's discussion directly contradicts the empirical data, repeating an outdated or anticipated hypothesis. Fortunately, **the authors' final LaTeX paper draft successfully resolved this contradiction.** The main text of the submission (Introduction and Section 4.3) adopts an objective, scientifically honest stance, correctly highlighting this as a "negative finding" and separating loss-landscape isotropic flatness from coordinate-aligned sparsification. This transition from a biased hypothesis to objective science shows excellent research integrity, though the old text remains in the code's log outputs and should be cleaned up.

