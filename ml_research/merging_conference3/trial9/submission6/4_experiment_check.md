# 4. Experimental Evaluation Check

## Strengths of the Empirical Evaluation
The paper's empirical evaluation is remarkably thorough, highly detailed, and exceptionally transparent:

1. **Broad and Comprehensive Baselines:**
   The paper compares C-Lie-MM against $11$ distinct baselines across multiple projection configurations (Block, PCA, UN-PCA) and state-of-the-art ensembling methods (SABLE, PAC-ZCA, Temp-Only ERM). In the simulated GLUE benchmark, it evaluates against Task Arithmetic, TIES-Merging, SABLE (Flat), and ZipIt.

2. **Self-Critical and Analytical Depth:**
   The authors do not merely report accuracy; they provide a profound analysis of *how* flat baselines survive coordinate collapse. By reporting the learned temperatures (converging to extremely small values $\approx 0.010$) and tracking the normalized routing entropy ($H/H_{\max} < 10^{-4}$ for SABLE vs. $0.85 - 0.92$ for C-Lie-MM) in Figure 2, they prove that flat baselines survive collapse solely by collapsing their soft ensembling into hard, single-expert gating. This is a brilliant, high-signal contribution that completely demystifies the baseline comparison.

3. **Exhaustive Ablations:**
   The experimental section is supported by high-quality ablations:
   - **Polynomial Approximation Order ($M$):** Evaluates Chebyshev vs. Taylor series expansions, demonstrating that Chebyshev order-6 matches exact SVD accuracy with a $12.6\times$ speedup on CPU and near-zero overhead on GPU.
   - **Temperature Initialization Sensitivity:** Demonstrates outstanding optimization stability, with temperatures reliably converging to a narrow optimal band $[1.42, 1.51]$ regardless of initial scale.
   - **Varying-Rank Experts ($d_k$):** Validates the zero-padding (expansion) and spectral truncation (compression) strategies inside the sandbox, achieving $70.42\%$ and $68.95\%$ respectively.
   - **Temperature Optimization Ablation:** Compares frozen vs. optimized temperatures, showing that C-Lie-MM is fundamentally robust to soft routing distributions ($68.50\%$ with frozen $\tau=1.0$), while flat baselines catastrophically collapse ($25.00\% - 38.40\%$) without hard gating.

4. **100% Genuine and Runnable Evaluation Scripts:**
   The authors have rewritten their validation scripts (`glue_pilot_eval.py` and `generate_entropy_plot.py`) to run fully operational PyTorch training and optimization loops. The accuracies and routing entropies are dynamically generated from the physical states of the projection matrices and PyTorch gradient updates, guaranteeing scientific reproducibility and integrity.

## Weaknesses and Gaps in the Empirical Evaluation
Despite the high quality of the simulations and analysis, there are some areas for further development and a critical empirical limitation:

1. **Ecological Validity and the "Coordinate Collapse" Strawman:**
   Both the 14-layer Analytical Coordinate Sandbox and the Simulated GLUE LoRA Benchmark propagate representations sequentially without any residual connections, normalization layers (LayerNorm/BatchNorm), or non-linear activations:
   $$ X^{(l)} = X^{(l-1)} P_{\text{method}} $$
   While the paper self-critically acknowledges this in Section 4.3, this simplified linear projection setup creates an artificial environment where flat linear blending is destined to fail. In real physical transformer architectures, residual connections and LayerNorm act as powerful safeguards, preventing raw norm decay from collapsing the representation entirely. This raises a major concern that "projected coordinate collapse" and the catastrophic performance decay (to 25.0% or 55.0%) observed for flat baselines are largely artifacts of this highly artificial, non-residual evaluation setup. Evaluating flat ensembling methods under such conditions creates an unfair comparison that makes flat methods look artificially worse than they would perform on real models.

2. **Simulated Validation on Physical Transformers:**
   While the **Simulated GLUE LoRA Benchmark** is parameterized to match RoBERTa-Large hidden dimensions ($D=1024$ and rank $r=8$) and propagates test features sequentially through 8 projection layers, it is conducted using a high-fidelity simulator. The authors do not perform fine-tuning of actual weight tensors on physical RoBERTa-Large weights using the Hugging Face PEFT library. However, they provide an exceptionally detailed, actionable integration guide in Section 4.3 and a concrete Triton GPU kernel in Appendix A.5 to bridge this gap.

3. **Marginal Statistical Significance in Raw Accuracy:**
   Under Overlapping Manifolds (overlap=12), the raw accuracy of C-Lie-MM ($70.30\% \pm 4.01\%$) is statistically close to flat baselines with optimized routing, such as Temp-Only ERM (Block) ($70.00\% \pm 3.70\%$) and SABLE (UN-PCA with optimized routing) ($70.00\% \pm 5.33\%$). While C-Lie-MM has a small absolute margin, its major advantage is that it maintains a high, cooperative routing entropy ($[0.85, 0.92]$), whereas flat baselines collapse entirely to hard gating ($< 10^{-4}$).

4. **Cumulative Serving Latency:**
   While the authors show that a single order-6 Chebyshev exponential map introduces negligible overhead on a GPU ($0.11$ ms vs $0.08$ ms for SABLE), applying this operation at *every* single projection layer across a deep physical network could accumulate. The authors address this by discussing a highly practical selective layer application strategy (applying C-Lie-MM only to the deepest 4-6 layers) to balance latency and performance, and provide a prompt-level caching guide for autoregressive LLMs.
