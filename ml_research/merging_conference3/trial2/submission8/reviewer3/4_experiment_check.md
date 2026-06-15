# Experimental Evaluation Check

A critical evaluation of the experimental setup, datasets, baselines, and whether the results support the claims made in the paper.

## Experimental Setup and Design
The experimental setup is characterized by high technical rigor but is limited in scale and task complexity:
* **The Model Backbone:** CLIP ViT-B/32 is a standard and appropriate backbone for evaluating vision-based task arithmetic.
* **Finetuning Parameters:** The authors target 28.7 million active parameters (including attention weights across all 12 blocks), which is a substantial portion of the visual encoder.
* **Datasets:** The evaluation is conducted on four standard, relatively simple classification datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN.
* **Low-Data Regime:** Each dataset is subsampled to exactly 1024 training and test samples. While this mirrors resource-constrained edge fine-tuning, these classification tasks are highly predictable and easily solved by a pre-trained CLIP backbone. The task vectors extracted in this low-data regime are likely to represent relatively small parameter shifts, which are naturally more resilient to pruning than larger shifts from extensive fine-tuning.
* **Rigour:** The inclusion of 3 independent random seeds with reported mean and standard deviation, and the use of individually optimized merging coefficients ($\lambda$) for all baselines, represents excellent scientific practice that ensures complete evaluation fairness.

---

## Evaluation of Claims vs. Evidence
The empirical results strongly support the specific claims made by the authors:
1. **Rescaling Necessity:** The claim that update norm shrinkage is the primary obstacle in task-vector pruning is convincingly supported by the ablation study in Section 4.3. Without rescaling, accuracy collapses to ~80% at $p=0.10$, whereas rescaling restores it to over 90%, closing the gap to the dense baseline.
2. **SAM vs. AdamW Pruning Resilience:** The counter-intuitive claim that loss landscape flatness (via SAM) does not inherently buffer task vectors against coordinate-wise pruning under well-converged regimes is backed by the results in Table 2. At $p=0.10$, the average accuracy of AdamW Uniform (90.34%) is virtually identical to SAM Uniform (90.32%). This is a key finding that challenges the common geometric assumption in model merging.
3. **The Saliency Double-Bind:** The claim that layer-wise budget allocation (NP-BTVP-S) suffers from scale instability and fails to outperform global Uniform pruning (NP-BTVP-U) is supported by the statistical indistinguishability of their performance ($p$-values of 0.96 under AdamW and 0.68 under SAM) and the performance drops observed under layer-wise scaling.

---

## Key Experimental Weaknesses and Gaps
While the experiments are technically sound, there are several notable gaps and limitations that affect the significance and generalizability of the findings:

1. **Lack of Head-to-Head Baseline Comparisons at Identical Sparsity Levels:**
   - In Table 3, the baselines are evaluated at different retention rates: TIES-Merging is evaluated at $p=0.20$ (80% sparsity) and DARE-Merging is evaluated at $p_{\text{drop}}=0.80$ (80% sparsity), while the proposed Uniform and Saliency methods are evaluated at $p=0.10$ (90% sparsity).
   - While the authors use this to show that their method at 90% sparsity outperforms TIES at 80% sparsity (which is impressive), they do not report TIES and DARE performance at $p=0.10$ or $p=0.05$. To establish a rigorous scientific comparison, all methods should have been evaluated across the entire sweep of parameter retention budgets ($p \in \{0.05, 0.10, 0.20\}$).

2. **Toy-Like Dataset Complexity:**
   - Evaluating exclusively on MNIST, FashionMNIST, CIFAR-10, and SVHN represents a very low bar. These datasets are often considered "toy" benchmarks in modern machine learning.
   - It is unclear if the high resilience of task-vector pruning (maintaining 90.34% accuracy at 90% sparsity compared to 90.94% dense) would hold on more challenging, high-dimensional vision tasks (e.g., ImageNet classification, object detection) or in the natural language processing domain (e.g., instruction following, reasoning), where task vectors are highly complex.

3. **Absence of Large Language Model (LLM) Evaluations:**
   - Model merging and task arithmetic are most critical and widely applied in the context of LLMs (such as LLaMA, Mistral, or OPT) due to the immense storage costs of these models.
   - While the authors dedicate Appendix C to discussing the theoretical extension of their findings to LLMs, they do not provide any actual experimental validation in this domain. Given that TIES-Merging and DARE-Merging were heavily validated on LLMs, the lack of LLM experiments in this work represents a major gap that limits the work's overall impact and generalizability.
