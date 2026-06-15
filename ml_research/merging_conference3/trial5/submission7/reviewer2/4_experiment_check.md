# Experimental Setup and Results Evaluation

## Critical Evaluation of the Experimental Setup

### 1. Scale of the Evaluation
- **Model Backbone:** The authors employ `vit_tiny_patch16_224`, which contains only $14$ layer groups and $5.7$M parameters. While this is a reasonable, resource-efficient model for a proof-of-concept, it is highly compact compared to the massive foundation models (e.g., ViT-Huge, CLIP, RoBERTa, or 7B+ LLMs) where parameter-space model merging is most actively used and critical. 
- **Task Domains:** The experts are trained on standard, toy-scale datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN, utilizing only $1,024$ training images per task and evaluating on subsets of size $512$. 
- **Implication:** The extremely small scale of both the model and the datasets limits the generalizability of the findings. Optimization dynamics, gradient noise, and parameter-space interference behave differently in large-scale regimes. While acceptable as a baseline study, evaluating PG-Merge on at least one larger backbone (such as ViT-B or a RoBERTa-base model) would have greatly strengthened the empirical significance of the paper.

### 2. Missing Standard Baselines
The paper's related work section (Section 2.1) discusses several static model merging techniques, including **TIES-merging** (Yadav et al., NeurIPS 2023) and **Fisher Weighted Averaging** (Matena & Raffel, NeurIPS 2022). 
- **The Gap:** TIES-merging is widely recognized as the de facto standard baseline for static model merging because it explicitly prunes redundant parameters and resolves sign conflicts to combat parameter interference. 
- **Critique:** Despite its prominence and being cited in the text, **TIES-merging is completely missing from the quantitative evaluation in Table 1.** The authors only compare against Uniform Merging (Task Arithmetic) as a static baseline. Including TIES-merging is essential to understand whether PG-Merge's adaptive test-time updates actually outperform state-of-the-art static conflict-resolution techniques.

---

## Rigorous Evaluation of Empirical Results

A deep, scholarly examination of the quantitative results in Table 1 and Table 2 reveals several nuances that challenge the authors' sweeping claims of PG-Merge's dominance:

### 1. Nuanced Individual Task Performance
The authors highlight that PG-Merge ($p=0.05$) achieves the highest **Joint Mean Accuracy** ($62.70\%$). However, a task-by-task breakdown reveals a different story:
- **MNIST:** Standard static **Uniform Merging achieves $65.04\%$**, which is **$1.76\%$ higher** than PG-Merge ($p=0.05$) at $63.28\%$. Here, test-time adaptation actually *hurts* performance relative to a simple average.
- **SVHN:** Standard **PolyMerge achieves $40.43\%$**, which is **$8.40\%$ higher** than PG-Merge ($p=0.05$) at $32.03\%$, and **$5.66\%$ higher** than PG-Merge ($p=0.15$) at $34.77\%$. 
- **Implication:** PG-Merge is not uniformly superior. On SVHN, restricting the optimization trajectory to a quadratic curve (PolyMerge) is dramatically better than PG-Merge's dynamic coordinate selection. On MNIST, doing no optimization at all (Uniform) is superior. The authors' claim that PG-Merge "dramatically outperforms unconstrained AdaMerging and matches or exceeds the generalizability of SOTA regularizers" is true only on average (Joint Mean), but hides substantial underperformance on specific tasks. The paper would be more academically honest if it discussed these trade-offs rather than presenting PG-Merge as a universal panacea.

### 2. Marginal Gains over Simple Baselines
While PG-Merge ($p=0.05$) technically outperforms all active baselines in Joint Mean Accuracy, the absolute margin of improvement over the simplest static baseline (Uniform Merging) is extremely narrow:
- PG-Merge ($p=0.05$): **$62.70\%$**
- Uniform Merging (Static): **$62.16\%$**
- **Absolute Gain:** **$+0.54\%$**
Given that PG-Merge requires backpropagation and multiple optimization steps (100 steps) on test-time streams, an average improvement of just $0.54\%$ over a zero-compute static average raises questions about its practical utility. The paper should discuss whether this tiny performance gain justifies the additional computational cost of test-time backpropagation.

### 3. Incomplete Boundary Exploration in Ablation Study
The ablation study in Table 2 does an excellent job of showing that higher sparsity (smaller $p$) yields better performance, with a peak at $p=0.05$. 
- **The Gap:** The authors stop their evaluation at $p=0.05$. To fully characterize the "sparsity sweet spot," they should have explored even higher sparsity levels, such as $p = 0.02$ or $p = 0.01$ (which correspond to updating only 1 coefficient per step). 
- Knowing the point at which the model's capacity becomes *too* restricted (causing performance to collapse back toward the uniform initialization) is crucial for a complete scientific understanding of the optimization landscape.
