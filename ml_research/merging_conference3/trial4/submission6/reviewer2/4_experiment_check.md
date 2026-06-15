# Experimental Evaluation Check

## 1. Experimental Setup and Choice of Datasets
- **Backbone Model:** The authors evaluate using **ViT-B-32** (approx. 86M parameters). While this is a standard model in academia, it is relatively small compared to modern deep learning models (e.g., LLMs with billions of parameters, or CLIP-ViT-L/14 with 300M+ parameters).
- **Datasets:** The 4-task suite consists of **MNIST, FashionMNIST, CIFAR-10, and SVHN**. These are highly standard, historic, and simple image classification datasets. 
- **Validation Split:** The evaluation uses a standardized subset of $2{,}048$ samples per dataset.

### Practitioner's Critique of the Setup:
1. **Toy/Small-Scale Benchmark Bias:** Modern weight-space model merging is predominantly applied to **Large Language Models (LLMs)** (such as LLaMA-2/3, Mistral, Gemma, or T5) and large-scale **Vision-Language Models** (such as CLIP). Evaluating only on simple classification datasets like MNIST/FashionMNIST on a small ViT-B-32 backbone does not reflect the scale, representation complexity, or structural dynamics of real-world deployments. 
2. **Artificial Diversity:** The selected datasets span very different domains (grayscale digits, clothing, natural images, street numbers). Because the domains are highly disjoint, the models update completely different parameter regions during fine-tuning. This artificially lowers the mask overlap rate (to $3\%-4\%$) and makes parameter collisions negligible. In practical applications where homologous models (e.g., multiple instruction-tuned LLaMA-3 models) are merged, task similarity is high, leading to a much higher collision rate. The paper's experimental setup is designed in a way that minimizes the very problem that sign consensus was designed to solve.

## 2. Baseline Comparisons
The paper compares STA against:
- **Task Arithmetic (TA)** (full density, both un-tuned and tuned)
- **Tuned DARE** (stochastic delta-dropout at $s=20\%$)
- **Tuned TIES-Merging** (with thresholding at $s=20\%$)

### Practitioner's Critique of the Baselines:
- **Omission of DARE-TIES:** The authors evaluate **DARE-TA** (direct addition) but omit the stronger hybrid baseline **DARE-TIES** (which combines stochastic dropout with TIES's sign consensus and disjoint merge), arguing that they want to isolate the effects of sparsification. However, DARE-TIES is the most widely used and strongest variant of DARE in real-world libraries like `mergekit`. Excluding the strongest SOTA hybrid variant makes the baseline comparison less complete and weakens the claim that sign consensus is redundant.

## 3. Do the Results Actually Support the Claims?

### A. The General Redundancy Claim is Overgeneralized
- **The Claim:** "...the sign-resolution heuristics of methods like TIES-Merging are entirely redundant."
- **The Support:** Table 1 shows that Tuned STA ($s=20\%$, $\lambda=0.8$) achieves $90.53\%$ average accuracy, matching Tuned TIES-Merging ($90.16\%$) and outperforming Tuned DARE ($88.95\%$) and Tuned TA ($88.64\%$). Conceptually, yes, in this specific benchmark, the results support the claim.
- **The Counterargument:** Because this benchmark uses a small-scale model on highly disjoint, simple datasets with artificially low mask overlap, the results **cannot** be generalized to claim that sign-consensus heuristics are *entirely* redundant across all deep learning. On highly similar tasks (e.g., merging multiple language adapters or instruction-tuned LLMs), coordinate-wise collisions and sign conflicts could be much more frequent and severe. Without LLM or high-similarity experiments, this claim remains unproven for the settings where model merging is most widely used.

### B. The Performance Gains Rely Heavily on Tuning Overhead
- **The Claim:** STA is an "extremely elegant alternative" that "matches the performance of over-engineered alternatives."
- **The Support:** Tuned STA ($90.53\%$) matches TIES-Merging ($90.16\%$).
- **The Counterargument:** This matching performance relies completely on finding the optimal scaling factor $\lambda^* = 0.8$. If we look at standard, un-tuned STA at $s=20\%$ and $\lambda=0.3$, the average accuracy is **$82.91\%$**—which lags behind Tuned TIES-Merging ($90.16\%$) by **$-7.25\%$ absolute**, and even lags behind full-density Task Arithmetic ($87.45\%$) by **$-4.54\%$ absolute**. This means that without validation sweeps to tune $\lambda$, STA is substantially inferior to existing techniques. In contrast, TIES-Merging and DARE are much more robust across different configurations and do not exhibit such a severe drop when evaluated with standard scaling, making them much more practical for out-of-the-box deployments.
