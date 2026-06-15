# Critical Evaluation of the Experimental Setup and Results

## 1. Experimental Setup and Dataset Limitations
The experimental design is well-structured but contains several scope limitations that warrant skepticism:
- **Toy/Low-Scale Datasets:** The evaluation is conducted on four highly standard and relatively simple vision datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. These datasets are far from modern, real-world deployment challenges (e.g., ImageNet, MS-COCO, or complex domain-shift benchmarks).
- **Extremely Low-Data Regime:** The training split for each task is restricted to exactly **1024 samples**. 
- **Highly Overparameterized Regime:** Fine-tuning a 28.7 million parameter vision encoder (CLIP ViT-B/32) on only 1024 samples creates a highly overparameterized, interpolating regime. Because the dataset is tiny and fine-tuning is limited to **5 epochs** with a tiny learning rate ($10^{-5}$), the resulting task vectors $\tau_k = \theta_k - \theta_{\text{base}}$ are extremely small in absolute magnitude.
- This tiny magnitude of task vectors explains why the models exhibit "extraordinary resilience to heavy sparsification". When the updates $\tau_k$ are already extremely close to zero, pruning 90% of them introduces an infinitesimal absolute perturbation in the weight space. Therefore, the reported resilience is likely a **direct artifact of the low-data, short-epoch fine-tuning regime**, rather than a general property of the NP-BTVP framework. On larger datasets or with longer training (where task vectors undergo significant drift and have larger norms), magnitude-based pruning would likely cause much more severe degradation.

## 2. Unfair Comparison and Methodological Flaw in the Ablation Study
The most critical methodological flaw in the paper lies in the **Ablation Study on Norm-Preserving Rescaling** (Section 4.3):
- The authors show that without rescaling, Uniform Pruning at $p = 0.10$ collapses to **80.45%** (SAM) and **80.94%** (AdamW), whereas introducing rescaling ($1/p$) restores performance to **90.32%** and **90.34%**.
- However, as mathematically proven in `2_novelty_check.md`, applying a global rescaling factor of $1/p = 10$ to the task vector is **mathematically identical** to multiplying the merging coefficient $\lambda_k$ by 10 (i.e., setting $\bar{\lambda}_k = \lambda_k / p$).
- In their experiments, the authors restricted the merging coefficient search space for all methods to $\lambda \in [0.1, 1.0]$. 
- This restriction is fair for the rescaled method (which uses $\lambda \approx 0.3$), but it **critically handicaps the unrescaled baseline**. To achieve the same optimal update signal strength, the unrescaled baseline required $\bar{\lambda} \approx 3.0$. Because the authors capped the baseline's search space at $1.0$, the unrescaled baseline was forced to operate at a severely sub-optimal scale factor.
- Thus, the reported "performance collapse" of the unrescaled baseline is an artificial result of the restricted hyperparameter search space, not a physical or mathematical limitation of unrescaled pruning itself. If the authors had swept $\lambda$ up to $10.0$ for the unrescaled baseline, it would have achieved **exactly the same performance** (90.34%) as the rescaled method. This undermines the core claim that "norm-preserving rescaling" is a necessary new operator for weight sparsification.

## 3. Analysis of Baselines
The authors compare NP-BTVP against TIES-Merging and DARE-Merging:
- **TIES-Merging:** Operates at $p=0.20$ (80% sparsity) and achieves 86.51% (SAM). NP-BTVP-U ($p=0.10$, 90% sparsity) outperforms it at 90.32%. This is a strong and valid comparison, showing that simple magnitude pruning with proper scale tuning is superior to the sign-consensus voting of TIES-Merging on these datasets.
- **DARE-Merging:** Operates at $p_{\text{drop}}=0.80$ (80% sparsity) and achieves 90.95% (SAM), while NP-BTVP-U ($p=0.10$, 90% sparsity) achieves 90.32%. While DARE-Merging has a slight edge (+0.63%), it operates at 80% sparsity compared to NP-BTVP's 90% sparsity. This makes NP-BTVP highly competitive, although the authors should have compared both at the exact same sparsity level (e.g., both at 90% or both at 80%) to make a scientifically rigorous claim of equivalence.

## 4. Summary of Experimental Check
- **Scale of Evaluation:** The evaluation is restricted to small-scale toy datasets (MNIST, Fashion, CIFAR-10, SVHN) and a tiny low-data regime (1024 samples), which artificially buffers both models against pruning.
- **Methodological Flaw:** The ablation study's comparison is fundamentally flawed due to a restricted hyperparameter search space for the unrescaled baseline, which mathematically hides the equivalence of the rescaling factor to a standard merging coefficient scaling.
- **Baseline Rigor:** Comparisons with TIES are strong, but the comparison with DARE is mismatched in sparsity levels (80% vs 90%).
