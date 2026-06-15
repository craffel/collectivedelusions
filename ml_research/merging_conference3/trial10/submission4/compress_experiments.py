import sys

# Read 04_experiments.tex
with open('submission/sections/04_experiments.tex', 'r') as f:
    content = f.read()

# 1. Shorten "Empirical Analysis of the Quantization Collapse"
old_p1 = r"""\textbf{Empirical Analysis of the Quantization Collapse.} In both Table 1 and Table 2, we observe a severe collapse in the accuracy of standard ensembling baselines under standard naive quantization. For example, at large-sample calibration (Table~\ref{tab:large_sample}), Momentum-Merge (Float32) achieves an accuracy of \textbf{78.00\%} at $\rho=0.0$, outperforming Uniform Merging (Float32) at \textbf{65.60\%} by over 12\% absolute. However, under naive quantization, SABLE (Quantized-Naive) and ChemMerge (Quantized-Naive) collapse completely, falling directly back to the level of static Uniform Merging (e.g., SABLE drops to \textbf{65.80\%} and ChemMerge to \textbf{65.80\%}).

This complete failure is caused by the rounding operator ($\lfloor \cdot \rceil$). In low-precision integer spaces, rounding acts as an aggressive step filter that collapses the multi-dimensional representational coordinates. Centroids computed in continuous spaces overlap on the discrete grid, erasing distance differences. The routing coefficients collapse, forcing the ensembling layers to act as uniform blenders and completely neutralizing the benefits of continuous adaptation."""

new_p1 = r"""\textbf{Empirical Analysis of the Quantization Collapse.} In Tables 1 and 2, standard ensembling baselines collapse under naive quantization. SABLE (Quantized-Naive) and ChemMerge (Quantized-Naive) drop directly to static Uniform Merging levels (e.g., SABLE collapses to \textbf{65.80\%}). Rounding ($\lfloor \cdot \rceil$) acts as an aggressive step filter that projects coordinates onto a coarse grid. Task centroids shift and overlap, erasing directional and distance differences. This causes routing coefficients to freeze, neutralizing any continuous adaptation benefits."""

if old_p1 in content:
    content = content.replace(old_p1, new_p1)
    print("Shortened p1")
else:
    print("Could not find p1")

# 2. Shorten "Mitigation and Complete Recovery via QA-Merge"
old_p2 = r"""\textbf{Mitigation and Complete Recovery via QA-Merge.} Our proposed QA-Merge successfully stabilizes representation trajectories, achieving a virtually 100\% recovery of continuous ensembling gains across all baselines, in both the small-sample and large-sample calibration regimes. For SABLE (QA-Merge), ChemMerge (QA-Merge), and Parametric Router (QA-Merge) under large-sample calibration (Table 2), accuracy tracks extremely closely to their Float32 counterparts across all entanglement levels, closing the gap to the unquantized ceiling to within 0.1--0.3\% absolute.

Most notably, for \textbf{Momentum-Merge (QA-Merge)} under the large-sample regime at $\rho=0.5$ (Table 2), our method achieves an accuracy of \textbf{90.50\%}, completely matching the unquantized Momentum-Merge performance of \textbf{90.50\%} and outperforming static Uniform Merging (Quantized) at \textbf{85.20\%} by \textbf{5.30\%} absolute, as well as Momentum-Merge (Quantized-Naive) at \textbf{87.60\%} by \textbf{2.90\%} absolute. These results demonstrate that dynamic ensembling is not only possible but highly practical under extreme low-precision constraints."""

new_p2 = r"""\textbf{Mitigation and Complete Recovery via QA-Merge.} QA-Merge successfully stabilizes trajectories, achieving $\approx 100\%$ recovery of ensembling gains. SABLE (QA-Merge), ChemMerge (QA-Merge), and Parametric Router (QA-Merge) track full-precision ceilings within 0.1--0.3\% absolute. Most notably, for \textbf{Momentum-Merge (QA-Merge)} at $\rho=0.5$ (Table 2), QA-Merge achieves \textbf{90.50\%}, matching the unquantized ceiling and outperforming static Uniform Merging by \textbf{5.30\%} and naive quantized ensembling by \textbf{2.90\%} absolute."""

if old_p2 in content:
    content = content.replace(old_p2, new_p2)
    print("Shortened p2")
else:
    print("Could not find p2")

# 3. Shorten "Overcoming the Small-Step Quantization Bottleneck"
old_p3 = r"""\textbf{Overcoming the Small-Step Quantization Bottleneck.} Standard low-precision dynamic ensembling models typically suffer from the \emph{Small-Step Quantization Bottleneck}, where subtle dynamic activations are smaller than the quantization grid spacing and are thus rounded away to zero, causing routing trajectories to freeze. QA-Merge completely overcomes this bottleneck by combining per-sample dynamic quantization (which prevents the quantization scale from blowing up due to noisy outlier tasks) with a layer-wise Activation Error Feedback (AEF) mechanism. AEF keeps a local accumulator of rounding errors and diffuses them back into the next layer's dynamic pull update:
\begin{equation}
    h^{(l)} = \text{quantize}(h^{(l-1)} + \text{pull} + e_{\text{act}})
\end{equation}
This ensures that even extremely small, sample-specific adjustments accumulate and cross the quantization rounding boundary rather than being lost. Our empirical results confirm that with these components, QA-Merge achieves near-zero downstream accuracy loss, establishing a new state of the art for low-precision dynamic ensembling."""

new_p3 = r"""\textbf{Overcoming the Small-Step Quantization Bottleneck.} Naive quantized routers suffer from the \emph{Small-Step Quantization Bottleneck}, where subtle activation changes (the pull vector) are smaller than the quantization step size and round to zero, freezing trajectories. QA-Merge combines per-sample dynamic activation scaling with layer-wise Activation Error Feedback (AEF). AEF residually accumulates representation rounding errors and adds them back to the next layer's update:
\begin{equation}
    h^{(l)} = \text{quantize}(h^{(l-1)} + \text{pull} + e_{\text{act}})
\end{equation}
This ensures that sub-grid adjustments accumulate and cross quantization thresholds, establishing near-zero loss Servings."""

if old_p3 in content:
    content = content.replace(old_p3, new_p3)
    print("Shortened p3")
else:
    print("Could not find p3")

# 4. Shorten "Bypassing and Isolating the Weak SVHN Expert"
old_p4 = r"""\textbf{Bypassing and Isolating the Weak SVHN Expert.} As described in Section 4.1, the SVHN expert adapter is deliberately calibrated to a highly underperforming accuracy of \textbf{22.80\%} (acting as a noisy representational distractor). To understand how QA-Merge handles such extreme expert imbalances, we analyze the routing coefficients ($\tilde{\boldsymbol{\alpha}}$) across the network. We observe that for non-SVHN input queries (MNIST, Fashion-MNIST, CIFAR-10), the routing weight allocated to the weak SVHN expert is consistently negligible (typically $\le 0.02$). This proves that our proposed scale-invariant cosine similarity gating (Eq. 4) and STE-gated routing (Eq. 5) successfully learn to bypass and isolate noisy representational pathways under extreme quantization. As a result, the classification performance of high-performing tasks is fully preserved at their Float32 limits (e.g., 100.0\% for MNIST and 92.40\% for CIFAR-10) with zero cross-task interference, while SVHN queries are safely routed to the SVHN expert without corrupting other representation paths."""

new_p4 = r"""\textbf{Bypassing and Isolating the Weak SVHN Expert.} The weak SVHN expert adapter (calibrated at \textbf{22.80\%}) acts as a representational distractor. For non-SVHN queries (MNIST, CIFAR-10), the routing weight allocated to this expert remains negligible ($\le 0.02$). This confirms that scale-invariant cosine similarity gating (Eq. 4) and STE-gated routing (Eq. 5) successfully isolate noisy representational pathways. Classification performance forMNIST (100.0\%) and CIFAR-10 (92.40\%) is preserved with zero cross-task interference, while SVHN queries are safely routed without representation corruption."""

if old_p4 in content:
    content = content.replace(old_p4, new_p4)
    print("Shortened p4")
else:
    print("Could not find p4")

# 5. Shrink Figure 2 size slightly
old_fig = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{results/fig2.png}"""

new_fig = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{results/fig2.png}"""

if old_fig in content:
    content = content.replace(old_fig, new_fig)
    print("Shrunk Figure 2")
else:
    print("Could not find Figure 2 code block")

with open('submission/sections/04_experiments.tex', 'w') as f:
    f.write(content)
print("Finished compress_experiments.py")
