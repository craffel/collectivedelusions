# Critical Evaluation of the Experimental Setup and Results

## Evaluation of Experimental Setup and Datasets
The experimental evaluation is divided into a synthetic diagnostic simulation (modeling a 12-layer Vision Transformer on MNIST, FashionMNIST, CIFAR-10, and SVHN) and actual physical weight-merging of a pre-trained CLIP ViT-B/32 backbone across 8 real-world datasets (SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD). 
The dataset choices are standard and diverse, representing various classification tasks (textures, satellite images, street signs, digit characters, natural objects, and fine-grained cars).

## Critical Analysis of Baselines and Evaluation Rigor

### 1. Complete Omission of Key Baselines in Physical Weight Merging
The most critical empirical weakness is the **selective omission of the strongest baseline methods** in the physical weight experiments. 
In Table 1 (simulated results), the proposed method is compared against **RegCalMerge (ESR)** and **PolyMerge (Subspace)**, which are recent and highly relevant spatial-regularization baselines. However, in Table 2 (actual physical weights), these two baselines are completely absent. 
Table 2 only compares GP-BayesMerge against standard Task-Wise and Layer-Wise AdaMerging variants. Without reporting RegCalMerge and PolyMerge on the physical ViT-B/32 experiments, it is impossible to verify whether the proposed GP-prior regularization actually outperforms prior spatial-regularization or subspace-constraint heuristics on real weight-merging deployments. This selective reporting is a major gap.

### 2. Limited Statistical Significance and Overlapping Confidence Intervals
The authors evaluate all physical experiments across 3 independent random seeds, which is a step in the right direction. However:
- A sample size of $N=3$ seeds is relatively small for establishing strong statistical confidence.
- The performance improvements of GP-BayesMerge and MT-GP-BayesMerge over Layer-Wise AdaMerging++ in Table 2 are modest (e.g., an average accuracy of $82.35\%$ vs $81.15\%$, a delta of just $1.2\%$).
- Several task-specific performance intervals overlap significantly. For instance, on the SVHN dataset:
  - Layer-Wise AdaMerging++: $89.62 \pm 0.98\%$
  - GP-BayesMerge: $90.15 \pm 0.35\%$
  The intervals $[88.64\%, 90.60\%]$ and $[89.80\%, 90.50\%]$ overlap heavily. The paper does not conduct any statistical significance tests (such as paired t-tests) to confirm that the reported improvements are statistically sound rather than random fluctuations.

### 3. Mischaracterization of "Out-of-Distribution (OOD)" Generalization
The authors repeatedly claim that GP-BayesMerge "fully preserves performance on challenging out-of-distribution domains" (e.g., in the abstract and intro). However, the evaluation protocol in Section 4.7 simply performs multi-task merging and evaluates on the standard in-distribution validation/test sets of each task. 
While test-time adaptation is performed on unlabeled calibration batches, the evaluation itself is not on out-of-distribution covariate shifts (such as evaluating on corrupted images like ImageNet-C, or domain generalization benchmarks like PACS). This mischaracterizes the standard multi-task testing as OOD evaluation.

## Do the Results Support the Claims?
- **Exposing the Overfitting-Optimizer Paradox**: Supported *only* in the synthetic simulation, where Standard AdaMerging SVHN performance collapses to $46.64\%$. On actual physical weights, however, Layer-Wise AdaMerging achieves $87.02\%$, outperforming Task Arithmetic's $82.05\%$. The claim of a "catastrophic generalization collapse" in real-world settings is not supported by the physical weight data.
- **Superiority of GP-BayesMerge**: Partially supported. GP-BayesMerge and MT-GP-BayesMerge do achieve the highest average classification accuracies on physical weights ($82.35\%$ and $82.68\%$), and they significantly reduce the standard deviation across seeds (e.g., SVHN variance drops from $\pm 1.84\%$ to $\pm 0.35\%$). This strongly supports the claim that the spatial GP prior stabilizes test-time optimization.
