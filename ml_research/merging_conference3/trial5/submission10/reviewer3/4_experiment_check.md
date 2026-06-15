# 4. Experimental Evaluation Check

## Experimental Setup & Datasets
The authors design a highly controlled, multi-task visual classification benchmark to evaluate model merging methods:
- **Datasets:** MNIST, FashionMNIST, CIFAR-10, and SVHN are classic and standard in multi-task merging literature.
- **Backbone:** The pre-trained Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters) is a modern, block-based architecture.
- **Low-Resource Regime:** The setup employs 2,000 fine-tuning samples, $B=64$ calibration samples, and 500 evaluation samples per task. This low-resource setup is an excellent and rigorous stress test. It exposes whether routing modules are prone to overfitting on small validation splits (transductive overfitting) versus generalizes well to unseen test data.

## Quality and Comprehensiveness of Baselines
The paper includes a remarkably comprehensive and competitive suite of baselines:
1. **Static Merging Baselines:** Task Arithmetic (Uniform) and AdaMerging (Unsupervised TTA).
2. **Supervised Static Baselines:** OFS-Tune (Supervised Static) and OFS-Tune Task-Specific (Supervised Task-Conditional).
3. **Dynamic Routing Baselines:** Linear Router (Classical) and QWS-Merge (Quantum Wavefunction Superposition).
4. **Internal Reference:** Individual Experts (Ceiling).

This selection represents the state-of-the-art in both static and dynamic model merging. The inclusion of the Task-Specific OFS-Tune (Task-Conditional) baseline is particularly commendable, as it represents a highly competitive upper-bound for static task-level adaptation.

## Do the Results Support the Claims?
Yes, the experimental results directly support the paper's primary claims with strong empirical evidence:

- **Claim 1: G-CML tames gradient explosion and stabilizes optimization.**
  - *Evidence:* The original, ungated chaotic baseline collapses to **55.20%** average accuracy, whereas G-CML achieves **73.80%** average accuracy under task-specific routing (Table 1). This is an absolute improvement of **+18.60%**.
  - *Support:* Figure 2 provides further quantitative support by showing that G-CML's learned gating factor drives Lyapunov exponents into the negative regime, stabilizing the chaotic recurrence.
- **Claim 2: ChaosMerge achieves highly competitive results with a tiny parameter footprint.**
  - *Evidence:* G-CML achieves **73.80%** average accuracy using exactly **384 parameters**. This significantly outperforms static Uniform Merging (**54.75%**) and AdaMerging (**70.85%**), and is competitive with the unconstrained Linear Router (**77.10%**) and QWS-Merge (**77.05%**), which require $30\times$ more parameters (10,808 parameters).
- **Claim 3: Annealed Chaos-to-Order Merging achieves outstanding performance.**
  - *Evidence:* The Annealed framework achieves an outstanding **78.12%** average accuracy (Table 2), outperforming pure G-CML (**72.90%**), pure Tanh Gated (**75.45%**), and both over-parameterized dynamic routers: Linear Router (**77.10%**) and QWS-Merge (**77.05%**).
- **Claim 4: Task-Specific dynamic routing is critical.**
  - *Evidence:* Comparing Task-Averaged vs. Task-Specific routing in Table 1 shows consistent improvements across all dynamic models (e.g., G-CML improves from **71.20%** to **73.80%**; Linear Router from **73.50%** to **77.10%**). This validates that batch-level averaging washes out sensitive dynamic trajectories.

## Critical Gaps and Empirical Limitations
Despite the strong alignment between claims and results, three key empirical limitations must be highlighted:
1. **Restricted Empirical Scale (Toy-Scale Datasets):** The evaluations are confined to relatively simple, small-scale image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While appropriate for a controlled proof-of-concept, these datasets do not represent the complexity of modern large-scale vision tasks (e.g., ImageNet) or massive language benchmarks (e.g., GLUE, MMLU).
2. **Small Backbone Size:** The use of `vit_tiny` (5.7M parameters) limits the generalizability of the findings to modern massive models. In very large models, parameter-space interference and routing dynamics can differ significantly.
3. **Underperformance against Static Task-Conditional Baseline:** While G-CML's parameter efficiency is outstanding, its pure formulation (**73.80%**) and even its annealed formulation (**78.12%**) are still heavily outperformed by the *OFS-Tune Task-Specific (Supervised Task-Conditional)* baseline, which achieves **82.90%** average accuracy ($+4.78\%$ to $+9.1\%$ higher).
   - *Note:* The authors do a commendable job of contextualizing this gap, explaining that the task-conditional static baseline requires explicit, hard-switched Task IDs at test-time (which is impractical under domain shifts or mixed batches) and scales poorly with $K$, whereas G-CML represents a unified, task-agnostic, and continuous parameter steering prior.
