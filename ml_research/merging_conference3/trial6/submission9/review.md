# Peer Review: Cross-Attention Multi-Expert Routing for Dynamic Model Merging

## Overall Recommendation and Summary Ratings

*   **Overall Recommendation:** **2: Reject**
*   **Soundness:** **Poor** (Significant technical flaws, logical contradictions, paper-to-code mismatches, and complete lack of genuine empirical validation)
*   **Presentation:** **Good** (Clear writing, well-structured sections, and professional-looking figures, though highly deceptive in its framing)
*   **Significance:** **Poor** (No valid real-world significance due to non-viable batch pooling, stateful temporal dependencies, and lack of actual physical evaluation)
*   **Originality:** **Fair** (Incremental combination of standard cross-attention queries with an existing classical sigmoid-gated routing head)

---

## 1. Summary of the Paper
This paper introduces **Cross-Attention Multi-Expert Routing (CAM-Router)**, a dynamic weight-merging framework designed to fuse specialized expert models in parameter space on-the-fly. The authors identify three key limitations in current global average pooling (GAP) based dynamic routers: (1) vulnerability to spatial occlusion due to GAP collapsing spatial coordinates, (2) "task heterogeneity collapse" where multi-task batches average out sample-specific representation features, and (3) zero-sum competitive bottlenecks arising from standard Softmax normalization.

To resolve these, CAM-Router:
1.  Retains the full, un-pooled sequence of spatial tokens from early layers of a Vision Transformer (ViT) backbone.
2.  Introduces a set of trainable task-expert query embeddings that dynamically attend to localized patch regions via Multi-Head Cross-Attention (MHCA).
3.  Replaces Softmax with independent, bounded sigmoidal gating to avoid zero-sum competition.
4.  Introduces **Decoupled Historical Gating (DHG)** for batched inference to smooth routing coefficients over a sliding historical window using an exponential moving average (EMA) to prevent heterogeneity collapse.

The paper claims that on a 14-layer compact Vision Transformer (`vit_tiny_patch16_224`) evaluated across MNIST, FashionMNIST, CIFAR-10, and SVHN, CAM-Router achieves a breakthrough Joint Mean Accuracy of **49.75%** (+31.00% over static fusions), is immune to spatial occlusions up to 80% patch masking, and is highly resilient to batch task heterogeneity up to batch size $B=256$.

---

## 2. Strengths
*   **Excellent Surface Presentation:** The paper is exceptionally well-written, with a clear narrative, logical flow, and sophisticated academic terminology. The equations are clean, and the LaTeX compiles flawlessly.
*   **Aesthetically Pleasing Figures:** The figures (representing the sweeps and stress tests) are visually sharp, clean, and highly professional, suggesting high-quality empirical work at a surface level.
*   **Well-Constructed Related Work:** The paper does a good job of positioning weight-space model merging within the context of Mixture-of-Experts (MoE) and parameter-efficient multi-task adaptation.

---

## 3. Major and Critical Weaknesses (The Fatal Flaws)

Despite its polished presentation, this submission contains severe scientific, mathematical, and ethical flaws that warrant an immediate and unambiguous **Reject**.

### Flaw 1: Complete Absence of Real Empirical Evaluation (Fabricated and Manipulated Results)
The most critical and disqualifying flaw is that **no actual model training, fine-tuning, or weight merging was ever performed on real datasets**. The entire empirical evaluation section is based on a fabricated Python simulation (`run_experiments.py`) that uses synthetic 192-dimensional noise vectors to simulate ViT token features.
*   **The Mock Environment:** The "accuracy" metrics are calculated not by running a Vision Transformer, but by mapping the predicted coefficients $\alpha_k$ directly to a classification probability via a hand-coded algebraic formula:
    $$\text{net\_routing} = \alpha_{task} - 0.8 \sum_{k \neq task} \alpha_k$$
    $$\text{prob\_correct} = 0.1317 + (\text{ceiling} - 0.1317) \cdot (\text{norm\_score})^2$$
    This formula is a completely artificial, non-physical proxy.
*   **Deliberate Manipulation of Sweep Data:** The sweep data is manually post-processed and artificially inflated with arbitrary offsets and clamps before being written to `experiment_results.md` and plotted in the figures:
    *   **Main Results (Table 1):** The CAM-Router accuracy is manually inflated by adding an arbitrary 12.00% to individual task accuracies and 10.00% to the joint mean:
        `cam_accs_final = [a + 0.12 for a in cam_accs]`
    *   **Attention Heads Sweep (Table 2 / Fig 1):** The joint mean accuracy is scaled using a logarithmic function of the head count $h$:
        `mean_acc_adj = mean_acc + (0.02 * np.log2(h))`
    *   **Spatial Occlusion Sweep (Table 3 / Fig 2):** For CAM-Router, accuracy is clamped to a minimum of 40.00% (`max(0.40, ...)`), resulting in a suspicious, perfectly flat line of exactly **40.00%** across all mask ratios ($0.0, 0.2, 0.4, 0.6, 0.8$).
    *   **Batch Size Resilience Sweep (Table 4 / Fig 3):** CAM-Router's accuracy is clamped to a minimum of 55.00% (`max(0.55, ...)`), while the baseline BSigmoid-Router is clamped to a minimum of 13.17% (`max(0.1317, ...)`). This results in a completely artificial flat line of exactly **55.00%** for CAM-Router and **13.17%** for the baseline across all large batch sizes.
    *   **Ablation Sweeps (Table 5 & 6):** Heuristic offsets of `0.10` and `0.05` are manually added to specific categories to force a desired hierarchy.

Representing a hand-tuned synthetic simulation as real-world Vision Transformer experiments on MNIST, FashionMNIST, CIFAR-10, and SVHN is a profound violation of scientific integrity. It completely invalidates every single empirical claim made in the paper.

### Flaw 2: The Batch-Pooling & Decoupled Historical Gating (DHG) Paradox (Theoretical Inconsistencies & Code Discrepancies)
The paper heavily criticizes standard average-pooling routers for "Task Heterogeneity Collapse" in mixed-task batches. To resolve this, the authors introduce **Decoupled Historical Gating (DHG)** in Section 3.3, where batch-level routing coefficients are smoothed using an exponential moving average (EMA):
$$\bar{\alpha}_k^{(t)} = \beta \bar{\alpha}_k^{(t-1)} + (1 - \beta) \left( \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(t)} \right)$$
This formulation contains severe mathematical contradictions and is completely contradicted by the actual codebase:
1.  **Temporal Statefulness and Order Dependency:** Maintaining an EMA of coefficients over a historical sliding window makes the model's weight state stateful during inference. This means that the model weights used to process a batch of images at step $t$ depend on the historical sequence of completely unrelated images processed at steps $t-1, t-2, \dots$. This violates standard stateless independent and identically distributed (i.i.d.) inference assumptions, as the model's prediction on the exact same sample will change depending on which unrelated queries happened to be executed in the past and in what order. This is a highly negative and dangerous property for production deployment.
2.  **Active Batch Dependency:** Even with DHG, the active batch average $\frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(t)}$ is folded into the EMA. Thus, predictions still suffer from batch-mate co-dependency.
3.  **Complete Paper-to-Code Discrepancy:** A close inspection of the codebase (`run_experiments.py`) reveals that **DHG is completely unimplemented in the actual router**. In `CAMRouter.forward()`, the router simply performs standard batch-average pooling:
    ```python
    logits_route = torch.cat(logits_route, dim=-1)  # [B, K]
    alphas_sample = 0.3 * torch.sigmoid(logits_route)  # [B, K]
    return alphas_sample.mean(dim=0)
    ```
    There is no historical tracking, no EMA state, and no $\beta$ parameter anywhere in the script! The "immunity to collapse" shown in Table 4 and Fig 3 was fabricated purely by hardcoding a `max(0.55, ...)` clamp in the evaluation loop, not through the implementation or testing of DHG.

### Flaw 3: The First-Block Inference Paradox and Weight-Merging Latency
In Section 3.1, the sequence of tokens $H_0$ is extracted from "the output of the first self-attention block" of the backbone. This introduces two serious architectural flaws:
1.  **The Extraction Paradox:** To run the first self-attention block of the backbone to extract $H_0$, the model must already possess a set of weights. What weights are used for the patch embedding and the first self-attention block? Are they the base model weights, or are they merged? If they are merged, how were the coefficients predicted before $H_0$ was extracted? If they are static base weights, then the first layer of the model is never dynamically merged, which is not discussed.
2.  **Weight-Merging Latency Overhead:** If $H_0$ is extracted during the forward pass, then the weights for layers 2 to 14 must be dynamically summed and overwritten *in the middle of the forward pass*:
    $$W_{merged}^{(l)} = W_{base}^{(l)} + \sum_k \bar{\alpha}_k V_k^{(l)}$$
    Summing large weight matrices for 13 layers of a ViT model on-the-fly during inference is an extremely memory-bandwidth-heavy operation. On modern GPUs, memory bandwidth is the primary bottleneck. Performing this weight summation on-the-fly inside the PyTorch forward pass introduces significant computational latency, completely invalidating the authors' claim of "zero additional memory or computational latency overhead."

---

## 4. Questions and Feedback for the Authors
1.  **On Empirical Rigor:** Can the authors explain why the results presented in the paper match the hardcoded outputs and artificial offsets (such as `+0.12`, `max(0.40, ...)`, and `max(0.55, ...)`) from `run_experiments.py` rather than actual physical measurements of a trained Vision Transformer?
2.  **On Batch and Historical Dependency:** How do the authors propose to address the stateful nature of Decoupled Historical Gating (DHG), where the classification of an individual image changes depending on which unrelated images happened to be in previous batches?
3.  **On Paper-to-Code Mismatch:** Why does the implementation of `CAMRouter` in `run_experiments.py` perform simple active-batch pooling and completely omit the Decoupled Historical Gating (DHG) mechanism described in Section 3.3?
4.  **On the Forward-Pass Paradox:** What weights are used to execute the patch embedding and the first transformer block before $H_0$ is extracted and before the routing coefficients are computed? If they are base weights, does this mean the first layer of the model does not benefit from expert-specific parameters?
5.  **On Weight-Merging Latency:** Have the authors measured the actual latency overhead (in milliseconds) of copying and summing the weight tensors of 13 ViT layers on-the-fly during a PyTorch forward pass, especially when running on a GPU?

---

## 5. Final Verdict
This paper is a classic example of an attractive surface-level presentation masking severe structural contradictions, fundamental logical paradoxes, and a complete lack of genuine scientific substance. The core methodology contains fundamental logical paradoxes, and the empirical section is a completely fabricated simulation with manually adjusted results. Furthermore, the claimed solution to mixed-task batching—Decoupled Historical Gating—is both theoretically stateful and completely unimplemented in the actual source code. For these reasons, the paper cannot be accepted and must be rejected.
