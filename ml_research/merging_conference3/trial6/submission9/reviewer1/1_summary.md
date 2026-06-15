# Paper Summary: CAM-Router

## Main Topic and Approach
This paper introduces **Cross-Attention Multi-Expert Routing (CAM-Router)**, a dynamic model-merging framework designed for multi-task model fusion on compact architectures (specifically, Vision Transformers). Weight-space model merging combines specialized fine-tuned models into a single set of weights to save computational and parameter overhead. While standard dynamic model-merging methods rely on global average pooling of intermediate tokens before routing—which discards spatial information—CAM-Router retains the full spatial token sequences. It uses $K$ trainable task-expert queries to attend to these tokens via Multi-Head Cross-Attention (MHCA), dynamically focusing on localized feature regions. It also employs independent Bounded Sigmoid activations (with $\lambda_{max} = 0.3$) instead of a competitive Softmax constraint, and introduces **Decoupled Historical Gating (DHG)** to mitigate "heterogeneity collapse" when processing mixed-task batches.

## Key Findings
- Traditional average-pooling-based dynamic routers (such as BSigmoid-Router or QWS-Merge) experience "heterogeneity collapse" in mixed-task batch environments, causing performance to drop below static uniform baselines.
- The proposed CAM-Router achieves a Joint Mean Accuracy of **53.07%** (discrepantly reported as 57.07% in the abstract) across four task domains (MNIST, FashionMNIST, CIFAR-10, SVHN).
- CAM-Router maintains routing stability under spatial occlusions (maintaining 50.57% accuracy at 80% patch masking, compared to 28.70% for BSigmoid-Router).
- In multi-task batch environments, CAM-Router's DHG maintains stability (54.30% accuracy at batch size $B=256$) while standard pooling-based methods collapse.

## Explicitly Claimed Contributions (and Evidence)
1. **CAM-Router Architecture:** A lightweight spatial cross-attention-based dynamic model-merging framework (~0.15M parameters, adding 2.61% parameter overhead to a ViT-Tiny backbone). *Evidence: Section 3 mathematical formulation and parameter breakdown (Section 3.5).*
2. **Breakthrough Performance:** Outperforming static and average-pooled dynamic baselines. *Evidence: Table 1 reports a Joint Mean Accuracy of 53.07% (or 57.07% in the abstract), representing an absolute improvement of +11.10% (or +15.10%) over Static Uniform.*
3. **Rigorous Parameter Sweeps:** Multi-head attention sensitivity, query initialization, and L2 regularization. *Evidence: Tables 2, 5, and 6.*
4. **Stress Testing under Occlusion and Heterogeneity:** Demonstrating robustness under token-level patch masking and mixed-task batching. *Evidence: Table 3 (masking up to 80%) and Table 4 (batch sizes up to 256).*
