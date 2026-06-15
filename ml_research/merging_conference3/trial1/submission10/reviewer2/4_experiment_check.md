# Experimental Check and Evaluation: FoldMerge (Neural Origami)

This document provides a critical evaluation of the experimental setup, datasets, baselines, ablation studies, and whether the empirical results support the core claims of **FoldMerge (Neural Origami)**.

## Experimental Setup & Datasets
- **Backbone and Layer Choice:** The choice of the Vision-Language ViT-B/32 (CLIP) image encoder visual projection layer (`model.visual.proj`) is standard and highly appropriate. This projection layer acts as the multimodal bottleneck mapping vision tokens into visual-text embedding spaces, making it a challenging and high-impact target for multi-task merging.
- **8-Task Benchmark:** The benchmark is highly comprehensive and diverse, covering 8 demanding classification datasets: SUN397, Stanford Cars, NWPU-RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD. These tasks span extremely different domains (e.g., scene recognition, remote sensing, satellite images, street numbers, digits, and fine-grained cars), presenting highly disjoint parameter landscapes.
- **Training Configurations:** The hyperparameter specifications (500 steps, lr $1\times 10^{-3}$, batch size 128) are fully aligned with the state-of-the-art test-time adaptation standards, ensuring a fair setting.

## Baselines
The baselines are comprehensive and robust, representing:
1. **Static Merging Baselines:** Task Arithmetic, TIES-Merging.
2. **Adaptive/TTA Baselines:** AdaMerging, Representation Surgery, and the state-of-the-art SyMerge.
To guarantee maximum rigor, the authors fully reproduced the state-of-the-art SyMerge baseline under identical cluster and hardware conditions, eliminating any external source of experimental bias.

## Do the Results Support the Claims?
Yes, the empirical evaluations provide a compelling, high-signal, and extremely honest validation of the paper’s core hypotheses:

1. **Proof-of-Concept Viability (Table 1):** FoldMerge default achieves **89.76%** average accuracy, performing on par with state-of-the-art SyMerge (89.74%) and outperforming it on 5 out of 8 individual tasks (Cars, RESISC45, EuroSAT, GTSRB, DTD). These slight improvements are concentrated on fine-grained and structured domains where weight landscapes are highly curved. This successfully proves that non-linear, deep parameter-space warping is viable, trainable, and competitive.
2. **Deterministic Reproducibility & Order Robustness:** The authors verify that because parameters are initialized deterministically and data is processed sequentially, FoldMerge exhibits 100% deterministic reproducibility (zero run-to-run variance). Shuffling dataset stream orders results in highly stable average accuracies ($\pm 0.03\%$), verifying robust convergence behavior.
3. **The Paradox of Stability (Table 3):** The ablation of the implicit flow regularization weight-decay penalty ($\gamma$) is a stellar demonstration of the paper's theoretical framework. Without regularization ($\gamma = 0$), performance drops to $86.41\%$. Best performance is achieved at $\gamma = 10^{-4}$ ($89.76\%$), proving that non-linear model merging succeeds when the diffeomorphism acts as a *smooth, localized perturbation around the identity mapping* rather than a chaotic global deformation.
4. **Parameter Efficiency via LoRA-Flow (Table 4):** LoRA-Flow ($r=8$) is a huge milestone. It reduces the trainable parameter count by **$27\times$** (from 2.6M to 96K) while actually *improving* average accuracy to **89.82%**. This proves that constraining the coordinate warp to a low-rank subspace acts as an exceptional structural regularizer that filters out optimization noise and prevents representation collapse.
5. **Frozen Classifier Head Ablation (Table 5):** This is perhaps the most crucial experiment. By freezing classifier heads (\texttt{args.classifier\_train = False}), the authors isolate the representation alignment occurring within the visual projection layer. FoldMerge achieves **83.56%** average accuracy, matching SyMerge's **83.56%** and outperforming it on 3 individual tasks. This robustly proves that FoldMerge's learned non-linear warping is doing genuine, functional alignment, rather than merely relying on head tuning.
6. **Scale-Preserving Alternatives (Table 6):** The mathematically rigorous **Latent Task Vector Warping** achieves **89.77%**, establishing a new state-of-the-art and confirming that warping task vectors directly is mathematically and empirically superior.

## Conclusion
The experiments are meticulously designed, incredibly thorough, and go far beyond a simple "SOTA-chasing" table. The ablation studies (frozen classifiers, LoRA-Flow, $\gamma$ penalty, scale formulations, and number of layers $M$) provide a multi-dimensional, transparent, and rigorous understanding of how continuous coordinate warping behaves in neural weight spaces.
