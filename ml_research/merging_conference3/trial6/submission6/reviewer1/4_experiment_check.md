# Peer Review Analysis - Part 4: Experimental Evaluation

## Critical Evaluation of the Experimental Setup
While the mathematical formulation of the paper is rigorous, the experimental setup exhibits severe **practical limitations and artificial constraints** that limit its real-world relevance:

1. **Highly Synthetic "Sandbox" Environment (Toy Setup):**
   - The authors evaluate their method on a **14-layer deep MLP with a width of only 64**. 
   - Instead of processing raw images, they flatten MNIST, FashionMNIST, CIFAR-10, and SVHN images and project them via **Johnson-Lindenstrauss (JL) random projections** to 192 dimensions.
   - For a practitioner in 2026, a width-64 MLP with JL-projected inputs is an extremely contrived, "toy" setup that does not reflect real-world deep learning. Real-world model merging is applied to large pre-trained architectures like **Vision Transformers (ViTs), ResNets, or Large Language Models (LLMs)** operating on raw high-dimensional data. This toy setup raises serious doubts about whether the findings generalize to physical vision backbones or LLMs.

2. **Extremely Low and Impractical Absolute Accuracies:**
   - The reported classification accuracies are exceptionally poor. The joint mean accuracy for PAC-Bayes Merge is only **35.37%**.
   - Looking at the individual tasks in Table 1:
     - **CIFAR-10 accuracy is 12.89%** (barely above the 10% random guessing baseline).
     - **SVHN accuracy is 15.71%** (random guessing is 10%).
     - **MNIST is 59.72%** and **FashionMNIST is 53.17%** (both are extremely low for these simple datasets, where standard networks easily achieve >95%).
   - These low accuracies are a direct result of the highly constrained architecture (width-64 MLP) and the JL projection. No practitioner would deploy a merged model that performs at 13% on CIFAR-10 or 15% on SVHN. Evaluating model merging on a setup where the underlying expert models themselves have extremely poor performance (e.g., CIFAR-10 expert ceiling is only 25.07%, SVHN is 17.57%) limits the practical utility of the results.

3. **Inappropriate Baseline Comparison for Ties-Merge and DARE-Merge:**
   - The authors compare their method against **Ties-Merge** and **DARE-Merge**, showing that both underperform the Static Uniform baseline.
   - This occurs because Ties-Merge and DARE-Merge are explicitly designed for large pre-trained models (like LLMs or large ViTs) where the weights reside in highly aligned, shared pre-trained spaces.
   - In the authors' setup, they apply **different seed-specific random projection matrices** to the inputs of each task! This means that each task expert is trained on entirely different input feature spaces. As a result, the input layer of each expert model must map completely different manifold structures. 
   - Merging these models in weight-space is structurally mismatched from the start. Ties-Merge and DARE-Merge are bound to fail here because the weight spaces are not aligned. Using this severe representation mismatch to claim that Ties-Merge and DARE-Merge "underperform" is misleading and represents a contrived comparison.

## Verification of Claims
- **Claim: $L_2$ regularizer outperforms $L_1$ regularizer (RBPM):**
  Table 1 shows that **PAC-Bayes (Deterministic Compiled)** achieves **35.37 $\pm$ 2.81%** Joint Mean accuracy, while **RBPM ($L_1$)** achieves **35.27 $\pm$ 2.72%**. 
  - The absolute performance margin is a microscopic **0.10%**.
  - Given the standard deviations of **2.81%** and **2.72%** across the 15 random seeds, a 0.10% difference is **statistically and practically insignificant**. The claim that the smooth $L_2$ penalty is superior because it "preserves continuous representative capacity" is theoretically appealing but practically unsupported by the empirical data.

- **Claim: PAC-Bayes Merge suppresses transductive overfitting under extreme scarcity:**
  Let's examine the calibration scarcity sweep in `scarcity_results.json`:
  - Under extreme scarcity (**$M = 2$ samples per task**, i.e., 8 total samples to optimize 12 trajectory parameters):
    - **Offline Unconstrained (unregularized)** achieves a Joint Mean accuracy of **34.16 $\pm$ 3.13%**.
    - **PAC-Bayes (isotropic)** achieves **33.86 $\pm$ 3.36%**.
    - **PAC-Bayes-FIM** achieves **33.43 $\pm$ 3.40%**.
  - Under the most severe scarcity regime ($M = 2$), the **unregularized offline unconstrained optimizer actually outperforms the proposed regularized models** by **0.30%** (over isotropic) and **0.73%** (over FIM).
  - This contradicts the core premise of the paper that unregularized optimization collapses under extreme scarcity and requires a PAC-Bayesian penalty. The authors explain this as implicit regularization from AdamW early stopping, but it raises questions about the absolute necessity of the complex mathematical framework in extreme few-shot regimes.
  - Furthermore, under $M = 2$, the FIM-weighted variant performs the worst (**33.43%**), illustrating that estimating Fisher Information on extremely small samples introduces massive estimation noise that degrades optimization.
