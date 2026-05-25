text_to_append = """

# Phase 3: Paper Writing

## Synthesis and Drafting
- **Structure**: We created a complete, publication-quality 8-page paper based on the ICML 2026 LaTeX template.
- **Key Sections Written**:
  1. **Abstract**: Summarized our method (Hyper-TTMM), key quantitative results (48.34% accuracy), and massive efficiency advantages (48x faster than Oracle, 8.5x faster than Gradient-based TTA).
  2. **Introduction**: Outlined the test-time model merging (TTMM) setting, highlighted the latency bottleneck of iterative optimization on streaming batches, and introduced our amortized hypernetwork-based solution.
  3. **Related Work**: Positioned our method relative to model merging, test-time adaptation (TENT), and TTMM baselines (CP-AM, FDF-DPA, and BK-CoMerge).
  4. **Proposed Method (Hyper-TTMM)**: Described the mathematical formulation of linear weight merging, batch normalization statistics fusion, the 278-dimensional batch descriptor s (incorporating features and expert prediction distributions), and hypernetwork MLP training on offline label-aware oracle targets.
  5. **Experimental Setup**: Specified the expert training details on MNIST and FashionMNIST training splits, the custom custom differentiable forward pass (manual_batch_norm2d), and the 5-phase test stream.
  6. **Results and Discussion**: Presented a detailed table of overall and phase-specific accuracies and latency metrics, explaining the massive speedups and accuracy improvements of Hyper-TTMM (+10.18% absolute gain over TTA).
  7. **Conclusion & Future Work**: Summarized the contributions and mapped future paths (scaling, LLMs, unsupervised meta-training).

## Compilation and Submission
- **LaTeX Engine**: Compiled successfully using tectonic to download all style packages and build a clean PDF.
- **Submission**: Copied the output PDF to submission.pdf in the root directory.

- **Status of Phase 3**: **COMPLETE**
"""

with open("progress.md", "a", encoding="utf-8") as f:
    f.write(text_to_append)

print("Successfully appended Phase 3 details to progress.md")
