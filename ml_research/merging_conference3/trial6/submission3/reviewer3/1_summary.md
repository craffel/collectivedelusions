# Paper Summary

## Main Topic and Approach
The paper addresses the challenge of **dynamic model merging**, where multiple task-specific expert network weights are dynamically blended at runtime using input-dependent routing coefficients $\alpha_k(x)$ on a per-sample or per-batch basis. The authors focus on the capacity-generalization trade-off in dynamic merging. 
They identify two severe empirical flaws in existing unshared, layer-wise routing networks (such as the L3-Router), which learn independent routing parameters for each of the $L$ layers:
1. **Layer-to-Layer Coefficient Ruggedness and Representation Drift:** Learning independent routing weights across sequential layers increases optimization degrees of freedom linearly with $L$. On small calibration splits, this can cause routing coefficients to diverge abruptly layer-by-layer, causing sequential representation drift as features propagate.
2. **Parameter Scaling Excess:** Unshared routers require a high number of parameters, making them overfit to low-noise features and collapse catastrophically when calibration data is scarce (e.g., on out-of-distribution SVHN data with only 64 samples).

To resolve these issues, the paper introduces the **Block-wise Weight-Sharing Router (BWS-Router)**. This architecture:
- Groups the $L$ layers of the model into $G = L / M$ uniform block groups and shares the routing weights ($W_{group}^{(g)}, B_{group}^{(g)}$) within each block group of size $M$.
- Restricts the optimization search space and mathematically mitigates layer-to-layer coefficient ruggedness.
- Integrates an unsupervised PCA pre-projector and bounded independent sigmoidal gating ($\lambda_{max} = 0.3$) for parameter efficiency and stability.
- Extends the evaluation to a **physical sequential weight-space model-merging** framework on multi-layer MLP experts, where representations are physically propagated through blended layers at runtime without any virtual-layer ensembling averaging.

---

## Key Findings and Claims
1. **Static Uniform Collapse:** Under severe weight-space semantic conflicts (permuted label mappings across 4 tasks), static uniform merging collapses to **23.56 $\pm$ 2.91%** Joint Mean accuracy in the virtual sandbox, and **17.88 $\pm$ 3.78%** in the physical framework.
2. **BWS-Router Performance and Efficiency:** In the virtual sandbox, BWS-Router ($M=3$, 80 parameters) achieves **79.57 $\pm$ 1.14%** Joint Mean accuracy (climbing to **79.63 $\pm$ 1.18%** under optimal learning rates). In the physical sequential framework, BWS-Router ($M=3$, 12 parameters) achieves **45.26 $\pm$ 10.11%** Joint Mean accuracy.
3. **91.7% Parameter Footprint Reduction:** Sharing routing weights globally ($M=12$) reduces the trainable routing parameter footprint from 240 down to only 20 parameters in the sandbox, with zero loss in dynamic routing accuracy (retaining **79.60 $\pm$ 1.15%**).
4. **Task-Heterogeneous Batch Robustness:** BWS-Router maintains stable sample-wise gating performance under task-heterogeneous batch shifts (**79.30 $\pm$ 1.88%** accuracy) without suffering from "heterogeneity collapse".
5. **Mitigation of Representation Drift in Physical Setup:** Under a task-heterogeneous mixed-batch stream, block-shared physical BWS-Router ($M=3$) outperforms the unshared baseline ($M=1$) by **+10.93%** absolute accuracy (**43.20 $\pm$ 22.49%** vs. **32.27 $\pm$ 21.28%**), empirically proving that block weight sharing acts as a powerful structural regularizer.
6. **Vulnerability of Wave-Based Routing:** The non-linear wave-superposition routing in QWS-Merge exhibits optimization ruggedness and sensitivity across random seeds.
7. **Softmax vs. Sigmoid Gating:** While Softmax gating excels in closed-world, mutually exclusive classification sandbox settings due to implicit sum-to-one regularization, independent Sigmoidal routing is preferred for open-world, decoupled environments where OOD inputs require expert deactivation and non-exclusive multi-task feature mixing is necessary.

---

## Explicitly Claimed Contributions and Supporting Evidence
The authors explicitly claim the following contributions:
- **Mathematical and Empirical Deconstruction of Coefficient Ruggedness:** They formalize Expected Ruggedness incorporating depth-dependent variance scales and adjacent layer correlations. They show that block-wise sharing acts as a direct structural constraint that stabilizes adjacent weight blending.
- **Introduction of the BWS-Router Architecture:** Supported by a large-scale grid search of over 1,280 experiment runs across 5 independent seeds. Under optimal learning rates, BWS-Router achieves outstanding Joint Mean accuracy using only a fraction of the parameters of unshared routers.
- **Validation under Physical Sequential Model Merging:** They construct a physical sequential weight-space model-merging framework on PyTorch 3-layer MLP experts, demonstrating that BWS-Router outperforms static uniform merging and dramatically boosts heterogeneous mixed-batch accuracy over the unshared baseline by **+10.93%** absolute.
- **Comprehensive Ablations and Analyses:** The paper includes sweeps over:
  - Block size sensitivity ($M \in \{1, 2, 3, 4, 6, 12\}$)
  - Gating activation functions (Linear, Tanh, Softmax, Sigmoid)
  - Task scaling ceiling ($\lambda_{max} \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ and learnable $\lambda_{max}$)
  - Calibration sample complexity (scaling from 16 to 1024 samples)
  - Gating bias initialization ($B_{group} \in \{-2.0, -1.0, 0.0, 1.0, 2.0\}$)
  - PCA subspace dimension ($d \in \{2, 3, 4, 6, 8, 12, 16\}$)
  - Non-linear unsupervised projector kernels (Linear, RBF, Cosine, Polynomial)
  - Variance stabilization strategies (residual routing links vs. sequential smoothing regularization)
  - GPU-level scaling recipe and physical Vision Transformer demonstration
