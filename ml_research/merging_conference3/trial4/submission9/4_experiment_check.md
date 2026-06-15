# Experiment Check

## 1. Strengths of the Experimental Evaluation

### Exceptionally Comprehensive and Rigorous Evaluation Suite
The paper's experimental section is remarkably thorough, extensive, and designed with high scientific rigor:
* **Strong and Representative Baselines**: It compares EPM against 8 representative baselines from across the model-merging literature, covering classical methods (Task Arithmetic, Prune-then-Merge), sign-consistent methods (TIES-Merging), random-dropout methods (DARE), continuous test-time adaptation methods (AdaMerging, ZipMerge), a critical control (Random Tensor Routing), and a newly introduced **Standardized TA + Pruning** control.
* **Multiple Sparsity Levels**: Results are reported across dense ($p=0.0$), moderate sparsity ($p=0.5$), and extreme sparsity ($p=0.8$) regimes, which is critical for evaluating parameter fusion under compression constraints.
* **Thorough Sensitivity Sweeps**: The paper includes three systematic sensitivity studies:
  * Coherence factor $\gamma$ sweep (from 0.0 to 1.0 in Table 4).
  * Optimization budget step count sweep (from 40 to 500 steps in Table 5) with professional convergence trajectory plots (Figure 2).
  * Calibration split size $N_{\text{val}}$ sweep (from 128 to 1024 samples per task in Table 6/Table 5).
* **Statistical Rigor**: Randomized optimization methods are run across 5 different calibration seeds, reporting clear standard deviations to demonstrate optimization stability (e.g., EPM TLC-Tune dense achieves **46.19% $\pm$ 0.14%**).
* **Empirical Scale Override Statistics**: The authors analyze all 5.52 million parameters of their ViT backbone to show that a scale override (where standardized routing elects a task, but physically another task's update is larger) occurs at exactly **13.67%** under Untuned EPM, and **13.79%** under TLC-Tuned EPM. This provides powerful empirical evidence of the need for scale-decoupled routing.

---

## 2. Weaknesses of the Experimental Evaluation

### Weakness 1: Toy Scale of Backbone and Disjoint Domains
* **Critique**: The empirical evaluation is restricted to a compact Vision Transformer (ViT-Tiny, 5.7M parameters) fine-tuned on highly disjoint datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
* **Implication**: While this setup is indeed a rigorous test of weight-space interference, it represents a highly artificial "toy" environment:
  * Grayscale digits (MNIST) and natural color objects (CIFAR-10) have completely non-overlapping feature spaces.
  * Merging these disjoint experts on a compact network leads to extremely low absolute accuracies (around **40%--46%** average vs. the **$\sim$95%** ceiling of individual experts). In a real-world scenario, no practitioner would deploy a merged model with a 50% performance drop.
  * Actual model merging is primarily utilized on Large Language Models (LLMs) or CLIP models, where the backbone is highly overparameterized (7B+ parameters) and tasks are much more related (e.g., merging different code or reasoning experts).
  * **Acknowledge authors' response**: The authors do discuss how overparameterization in billion-parameter LLMs/VLMs is expected to buffer conflicts, and provide a detailed autoregressive decoder-only LLM implementation pseudocode (Algorithm 1) to guide future large-scale empirical tests. Nonetheless, the immediate practical impact is bottlenecked by the toy-scale empirical backbone.

### Weakness 2: Underperformance and Capacity Starvation under High Sparsity
* **Critique**: Under moderate and extreme sparsity, the proposed Soft-EPA coordinate routing is highly fragile when parameters are pruned, and is outperformed by baseline techniques:
  * **At $p=0.5$ (Moderate Sparsity)**: Although TLC-Tuned EPM (**42.60%**) outperforms the DARE baseline (**40.94%**), it is significantly outperformed by the **Standardized TA + Pruning** baseline (**44.87%**).
  * **At $p=0.8$ (Extreme Sparsity)**: Even with Dynamic Coherence Scheduling (which successfully rescues EPM from a 24.11% collapse to **26.41%**), EPM is heavily outperformed by DARE (**40.90%**), which maintains a robust joint mean on every single task.
* **Implication**: This reveals that Soft-EPA's coordinate routing and secondary coordinate-wise attenuation starve the model of representational capacity under high pruning constraints. DARE, which utilizes expected-value scaling ($1/(1-p) = 5$) to preserve deep manifold activation scales under extreme deletion, is consistently superior under compressed merging regimes. The claim that EPM is a robust operator under high sparsity is undermined by these findings.

---

## 3. Experimental Rating: Good to Excellent
The experimental section is rated "Good to Excellent" because it is exceptionally thorough, multi-seed, includes comprehensive sensitivity sweeps, reports detailed scale override statistics, and is outstandingly self-honest and academically transparent about its limitations and the superiority of baselines (like DARE under extreme sparsity and Standardized TA + Pruning under moderate sparsity). The toy-scale backbone and disjoint datasets prevent it from being a flawless "Excellent".
