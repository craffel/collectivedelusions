# 4. Experimental Evaluation Check

## Evaluation of the Experimental Setup

A critical analysis of the experimental framework reveals severe limitations, weak baselines, and results that directly contradict the paper's main selling points:

### 1. Toy Scale & Lack of Domain Realism
* The entire empirical evaluation is restricted to a tiny Vision Transformer (`vit_tiny`, 5.7M parameters) and four small, outdated image classification benchmarks: MNIST, FashionMNIST, CIFAR-10, and SVHN.
* Merging task experts across these completely disjoint and artificial domains is highly unrealistic. In practice, model merging is utilized to combine LLMs or large multi-modal models on related text or visual domains (e.g., merging specialized models for code, mathematics, or translation). 
* The authors provide zero evidence that EPM scales or generalizes to modern foundation models or larger architectures. Their theoretical claim that "massive overparameterization provides a highly redundant high-dimensional weight space where conflicting updates can lie in orthogonal subspaces" is pure speculation unsupported by any empirical data.

### 2. Absurdly Low Absolute Accuracies (The Performance-Utility Gap)
* The individual task experts achieve high test accuracies (91.31% to 98.74%, with a joint ceiling of 94.91%). 
* However, after merging with EPM (TLC-Tune), the joint mean accuracy peaks at **46.19%** (dense) and drops to **42.60%** (sparse 50%) and **26.41%** (sparse 80%).
* A model with 46% accuracy on standard digit and clothing classification is completely useless in any real-world vision pipeline. If a practitioner needs to serve these four tasks, a simple 4-expert ensemble or a lightweight dynamic routing framework (like task-specific LoRA adapters or MoE) would preserve the expert-level accuracies ($>$90%) with negligible latency or parameter overhead. 
* The paper spends extensive text on "The Paradigm Trade-offs" defending this gap, but the absolute performance drop is too severe to make model merging a viable deployment option in this setting.

### 3. The "Exclusivity Contradiction" under Sparsity
* The paper's core thesis is that "coordinate exclusivity" is the driving mechanism for resolving weight conflicts. 
* However, the sensitivity analysis of the coherence factor $\gamma$ (Table 4) and the discussion on the "Exclusivity Contradiction" directly contradict this. Under 50% sparsity ($p=0.5$), as the coherence factor $\gamma$ increases from $0.0$ (pure coordinate exclusivity) to $1.0$ (standard Task Arithmetic with pruning, i.e., zero exclusivity), the joint average accuracy rises monotonically from **41.79%** to **46.14%**! 
* This demonstrates that standard average weight blending (where there is NO coordinate exclusivity) actually performs *better* than coordinate exclusivity. 
* To hide this, the authors introduce a "Standardized TA + Pruning" baseline in Table 2 ($p=0.5$), which achieves **44.87%** joint average accuracy, outperforming their own TLC-Tuned EPM (**42.60%**).
* The authors defend EPM by claiming it "raises the worst-case performance floor" (protecting MNIST from collapsing to 15.77%). However, raising a trivial grayscale digit dataset (MNIST) from 15% to 59% while dragging down CIFAR-10 (from 61.67% to 37.86%) and SVHN (from 67.14% to 41.99%) is a highly questionable trade-off. Sacrificing complex, high-value vision models to boost a toy digit classifier is an unacceptable compromise.

### 4. Poor Performance under Extreme Sparsity ($p=0.8$)
* In Table 3, under 80% sparsity, TLC-Tuned EPM with DCS achieves a joint mean accuracy of **26.41%**. 
* Under the exact same conditions, the standard **DARE** baseline achieves a joint mean of **40.90%**, completely dominating EPM by **14.49% absolute accuracy**!
* Even with the proposed Dynamic Coherence Scheduling (DCS), EPM remains highly uncompetitive under high sparsity, demonstrating that coordinate exclusivity is a fragile mechanism that suffers from capacity starvation when parameters are heavily pruned.

### 5. Overstated Claims of Optimization Robustness
* The authors claim TLC-Tune is immune to transductive noise due to its low dimensionality ($K=4$). 
* However, they utilize a tiny validation split of only 128 samples per task (512 total). Direct black-box search on such a small validation pool is highly susceptible to overfitting to the specific validation split's data distribution. 
* The localized performance dip of EPM (TLC $p=0.5$) at $N_{\text{val}}=512$ where the accuracy drops to 35.36% further proves that their zero-order (1+1)-ES is fragile and can easily get snared in suboptimal local minima.
