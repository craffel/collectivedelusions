# 1. Summary of the Paper

## Main Topic and Approach
The paper addresses the challenge of **dynamic model merging** in parameter space for multi-task model fusion. Standard static fusions (e.g., Task Arithmetic, Ties-Merging) compute a single set of merging coefficients that remain static for all inference samples, which often causes catastrophic interference when merging disparate models. Dynamic merging methods attempt to predict sample-specific routing weights on-the-fly using lightweight neural routers. 

However, existing dynamic routers collapse spatial representations via global average pooling before routing. This leads to:
1. **Vulnerability to spatial occlusion/corruptions** due to loss of localized features.
2. **Task heterogeneity collapse** when processing mixed-task batches (pooling across different tasks averages out task-specific signatures, collapsing predictions to uniform compromises).
3. **Softmax bottleneck constraints** that impose artificial zero-sum competition among experts.

To address these, the paper proposes **Cross-Attention Multi-Expert Routing (CAM-Router)**:
* **Spatial Sequence Preservation:** It extracts and retains the un-pooled token sequence $H_0 \in \mathbb{R}^{B \times N \times D}$ from the first block of a pre-trained, static base model.
* **Multi-Head Cross-Attention (MHCA):** It introduces $K$ trainable task-expert query embeddings $Q \in \mathbb{R}^{K \times D}$ that attend to the spatial patch tokens via cross-attention, allowing queries to capture localized features.
* **Independent Bounded Gating:** It uses independent Bounded Sigmoid activations with a maximum scaling coefficient ($\lambda_{max} = 0.3$) instead of Softmax to eliminate zero-sum bottlenecks.
* **Decoupled Historical Gating (DHG):** For batched inference, it computes per-sample coefficients and keeps an exponential moving average (EMA) of these coefficients across a sliding window to decouple samples and prevent heterogeneity collapse.

---

## Key Findings and Claims
1. **Baseline Comparison:** CAM-Router achieves a Joint Mean Accuracy of **53.07%** (MNIST: 65.47%, FashionMNIST: 58.67%, CIFAR-10: 31.07%, SVHN: 57.07%) on a simulated 14-layer compact ViT-Tiny sandbox. This is a $+11.10\%$ absolute improvement over Static Uniform (41.97%) and significantly outperforms pooling-based routers (e.g., BSigmoid-Router at 28.70%, QWS-Merge SOTA at 24.90%). Note: The abstract inconsistently claims a Joint Mean Accuracy of **57.07%** and $+15.10\%$ improvement.
2. **Attention Heads Sweep:** Performance remains relatively stable when varying the number of attention heads $h \in \{1, 2, 4, 8\}$, peaking slightly at $h=1$ with 56.73%.
3. **Robustness to Spatial Occlusion:** Under systematic patch masking (up to 80%), CAM-Router maintains a stable accuracy of 50.57% (the abstract claims 53.63%), while pooling-based routers collapse.
4. **Batch Size/Heterogeneity Resilience:** With Decoupled Historical Gating (DHG), CAM-Router remains robust under mixed-task batches, achieving 54.30% accuracy at $B=256$, while standard pooling-based methods collapse.
5. **Ablations on Queries & Regularization:** Prototypic Task-Average initialization is the most effective strategy (53.07% vs. Random Gaussian at 47.03% and Orthogonal at 44.87%). Unregularized calibration training ($\lambda_{wd} = 0.0$) performs best.

---

## Explicitly Claimed Contributions (with Evidence)
* **CAM-Router Framework:** A spatial cross-attention-based dynamic model merging framework adding only $0.15\text{M}$ parameters ($2.61\%$ overhead on a ViT-Tiny backbone). *Evidence: Section 3.1, 3.2, and 3.5 parameter calculations.*
* **State-of-the-Art Performance:** Claims breakthrough Joint Mean Accuracy (+11.10% over static uniform, outperforming other dynamic routers). *Evidence: Section 4.2 Table 1.*
* **Optimization Landscape Mapping:** Systematic sweeps over structural configurations (attention heads, query initialization, and $L_2$ regularization). *Evidence: Section 4.3 Tables 2, 5, 6.*
* **Stress Test Robustness:** Extraordinary robustness under spatial occlusions (up to 80%) and heterogeneous large batch sizes ($B=256$). *Evidence: Section 4.3 Tables 3 and 4.*
