# 1. Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of **dynamic model merging** for multi-task model fusion, where specialized expert parameters are combined on-the-fly depending on the input context. The objective is to design a dynamic routing method that is robust to spatial occlusions and "heterogeneity collapse" when processing mixed-task batches during inference.

## Proposed Approach: CAM-Router
To resolve the limitations of existing dynamic routing methods (which perform global average pooling of token features and use Softmax-based constraints), the paper introduces **Cross-Attention Multi-Expert Routing (CAM-Router)**. The core components of this framework are:
1. **Preservation of Spatial Tokens:** Instead of global pooling, CAM-Router retains the full spatial token sequence from the early layers (output of the first transformer block) of a Vision Transformer (ViT) backbone.
2. **Multi-Head Cross-Attention (MHCA) Gating:** Trains a set of task-expert queries ($Q$) to attend to the un-pooled token representations via cross-attention, allowing localized and domain-specific feature extraction.
3. **Independent Bounded Sigmoidal Gating:** Replaces the zero-sum competitive Softmax activation with independent, bounded sigmoidal gating (with $\lambda_{max} = 0.3$) to allow concurrent multi-expert activation.
4. **Decoupled Historical Gating (DHG):** For batched inference, DHG computes per-sample coefficients and smooths them across a sliding window via an Exponential Moving Average (EMA), rather than average pooling active concurrent batch elements.

## Key Findings and Claims
1. **Performance Lead:** The authors claim CAM-Router achieves a Joint Mean Accuracy of **53.07%** (though the Abstract claims **57.07%**) across four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) in a 14-layer ViT-Tiny coordinate-space sandbox, outperforming the Static Uniform baseline (41.97%) and average-pooling dynamic baselines (such as BSigmoid-Router at 28.70% and QWS-Merge at 24.90%).
2. **Occlusion Robustness:** CAM-Router is claimed to maintain stable performance under spatial occlusions, keeping **50.57%** accuracy (Abstract claims **53.63%**) under 80% patch masking, where global pooling-based methods collapse.
3. **Heterogeneity Resilience:** Under mixed-task batching up to $B=256$, CAM-Router with DHG maintains an accuracy of **54.30%** (Abstract claims **55.47%**), preventing the "heterogeneity collapse" observed in standard average-pooling methods.

## Explicitly Claimed Contributions (with Evidence)
- **Introduction of CAM-Router:** A lightweight ($0.15\text{M}$ parameter, 2.61% overhead) spatial cross-attention-based dynamic routing framework.
- **Improved Accuracy over Baselines:** Evidence provided in Table 4 (Main Baseline Comparison), showing significant gains on all four task domains.
- **Structural Optimization Insights:** Multi-dimensional sweeps over the number of attention heads (Table 5), query initialization strategies (Table 8), and $L_2$ regularization penalties (Table 9).
- **Robustness in Stress Tests:** Validation via systematic spatial occlusion masking (Table 6) and task heterogeneity sweeps (Table 7) verifying stability.
