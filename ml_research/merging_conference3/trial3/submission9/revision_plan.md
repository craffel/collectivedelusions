# Revision Plan: FlatQ-Merge Refinement (Round 2)

We have addressed the forward-looking recommendations from the Mock Reviewer (Score: 5 - Accept) to further elevate the paper's scientific impact and completeness.

## 1. Scaling to Large Language Models (LLMs) (Suggestion 1)
*   **The Recommendation:** Discuss how these landscape geometry insights scale to billion-parameter autoregressive models and how SAM could suppress outlier channels during instruction-tuning.
*   **Action Taken:** Enriched Section 5.1 (Limitations and Scope) to detail the exact mechanism of LLM outlier channel suppression. Specifically, we explained that modern autoregressive LLMs suffer from severe activation outliers in specific channels. Applying landscape-flattening objectives like SAM during instruction-tuning structurally smooths the manifold, suppressing these spikes and allowing downstream PTQ (like AWQ/GPTQ) to compress models losslessly.

## 2. Joint Weight-Activation Quantization (Suggestion 2)
*   **The Recommendation:** Discuss the implications of expert flatness on activation scaling and how spectral norm bounds limit activation spikes.
*   **Action Taken:** Expanded Section 5.1 (Limitations and Scope - Weight-Only Quantization) with rigorous mathematical analysis. We proved that SAM-induced expert flatness bounds the spectral norm of layers $\|W\|_2$, which directly restricts their Lipschitz constant $\text{Lip}(f) = \|W\|_2$. Because activation magnitudes are bounded by the Lipschitz constant, flat experts naturally prevent extreme activation spikes from propagating across successive Transformer blocks, facilitating seamless joint W4A4 or W8A8 quantization on edge hardware.

## 3. Dynamic Layer-wise Flatness Control (Suggestion 3)
*   **The Recommendation:** Explore block-specific or layer-wise flatness control (larger radius on sensitive blocks, smaller on MLP or early layers) to bypass the Over-Perturbation Threshold.
*   **Action Taken:** Added a dedicated mathematical formulation to Section 5.2 (Future Directions) introducing layer-wise dynamic flatness scheduling:
    $$\rho_l = \rho_{\text{base}} \cdot \gamma(l)$$
    We explained how scaling the perturbation radius based on layer depth $l$ or layer-wise Fisher information can maximize downstream robustness while protecting task-vector uniqueness, avoiding the global Over-Perturbation Threshold.
