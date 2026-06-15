# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
1. **Toy and Low-Resolution Datasets:**
   - The experiments are conducted on standard toy datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. While these datasets are diverse, they are extremely low-resolution and simple compared to the settings where model merging is practically relevant today (e.g., instruction tuning in Large Language Models, cross-domain adaptation in Vision-Language Models like CLIP, or large ViT / ResNet scales on high-resolution medical or satellite imagery).
   - Evaluating on a sub-10M parameter model ($\mathtt{vit\_tiny\_patch16\_224}$ with 5.7M parameters) limits the generalizability of the findings to modern, large-scale deep learning models.

2. **Severe Specialist Under-Training:**
   - The task experts are extremely weak: the SVHN specialist gets 23.20% accuracy, MNIST gets 73.20%, and CIFAR-10 gets 77.60%.
   - Fine-tuning each model for only 2 epochs on a tiny class-balanced subset of 1000 images means the "specialist experts" are highly sub-optimal.
   - While the authors argue that this simulates a "highly challenging, computationally-constrained and sub-optimal regime," evaluating model merging on noisy, poorly-trained models makes the results highly suspect. Model merging is typically designed to aggregate specialized *knowledge* from strong models. If the base experts themselves have barely learned the tasks, the dynamic router is merely blending parameter noise.

3. **Small Calibration Set and Seed Sensitivity:**
   - The calibration phase uses only 16 images per task (64 total).
   - Under such an extreme low-data regime, optimization is highly sensitive to the specific 16 samples selected and the random initialization seed.
   - The authors only report results for a single seed ($\mathtt{seed=42}$). They do not provide average accuracy, variance, or statistical significance across multiple random seeds or multiple calibration folds.

## Evaluation of Baselines
- The authors include a reasonable set of baselines, including Uniform Merge (Task Arithmetic), Softmax-based dynamic routers (BL-Router, with and without L2 regularization), independent sigmoidal routers (BSigmoid-Router, with and without L2 regularization), and a state-of-the-art wave-interference method (QWS-Merge).
- **Missing Baselines:** The authors do not compare their prior-regularization method to simpler, more standard regularizations on the routing head, such as:
  - **L1 Regularization:** To encourage sparsity in routing projection weights.
  - **Entropy Regularization:** To control the confidence/entropy of the routing coefficients.
  - **Dropout:** Applied to the router's projection layers to prevent overfitting on the 64-sample calibration set.

## Whether the Results Actually Support the Claims

**The results directly contradict the central claims of the paper.**

1. **The Abstract and Introduction Claim:** "TCPR consistently prevents high-conflict task collapse and bridges the performance gap to specialist experts... TCPR consistently outperforms standard L2-regularized baselines and SOTA wave-interference methods like QWS-Merge."
2. **The Actual Results (Table 1):**
   - The unregularized `BSigmoid-Router` baseline gets **25.50%** joint mean accuracy.
   - The proposed `TCPR-Param` gets **25.20%** ($\beta = 10^{-6}$).
   - The proposed `TCPR-Rep` gets **25.20%** ($\beta = 10^{-6}$).
   - Any active regularization scale ($\beta \ge 1.0$) leads to a severe performance collapse, dropping joint accuracy to 21.00% or worse.
3. **The Empirical Failure:**
   - The proposed method (TCPR) **never** outperforms the unregularized `BSigmoid-Router` baseline. It either slightly degrades performance (at extremely small $\beta = 10^{-6}$ where it is "mathematically dead" anyway) or completely collapses performance (when active at $\beta \ge 1.0$).
   - The only reason TCPR appears to outperform QWS-Merge (21.80%) or BL-Router (19.10%) is because it is backed by the unregularized `BSigmoid-Router` architecture. The "proposed regularizer" itself is completely useless and actively harmful.
   - Therefore, the empirical results **disprove** the value of TCPR. The paper is essentially pitching a regularization method that has been empirically demonstrated to be a failure.
