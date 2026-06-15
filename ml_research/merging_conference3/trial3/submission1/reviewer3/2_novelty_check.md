# 2. Novelty Check

## Originality of Ideas and Conceptual Leaps
This paper offers a highly original, paradigm-challenging perspective on quantization-aware model merging. Rather than following the incremental "state-of-the-art chasing" trend of inventing another weight fusion heuristic, the authors take a step back and perform a rigorous, critical **methodological deconstruction and independent audit**. This represents a significant conceptual leap. The authors identify and dismantle three foundational, unstudied assumptions that have dominated the literature:
1. **Quantization-Operator Monomorphism:** The false assumption that coefficients optimized under simulated "fake" quantization in PyTorch will generalize perfectly to heterogeneous physical accelerators (e.g., edge TPUs, Qualcomm Hexagon DSPs, Apple Neural Engine) with varying scale/offset arithmetic.
2. **Calibration Stream Purity:** The unrealistic assumption of pristine, class-balanced, and static test-time calibration streams.
3. **STE Gradient Path Fidelity:** The belief that straight-through gradient approximations provide a high-fidelity guide for weight-space search in highly non-convex, discontinuous quantized landscapes.

By exposing these blind spots, the paper establishes the **Cross-Schema Generalization Gap** as a crucial, previously undocumented phenomenon: coefficients learned on a source schema overfit intensely to simulated rounding thresholds, leading to catastrophic representation collapse (often sub-10% random guess) when deployed on mismatched physical hardware.

## Delta from Prior Work
Prior works (such as Q-Merge, PolyMerge, and RegCalMerge) focus exclusively on improving "matched" accuracy—evaluating the merged model on the exact same simulated operator used during optimization, and using clean, balanced datasets. 

This paper's delta from prior literature is substantial and multi-dimensional:
* **The "Quantized AdaMerging" Paradigm Shift:** The authors demonstrate that unquantized search in FP16 followed by post-hoc quantization (Quantized AdaMerging) consistently outperforms direct low-bit optimization via STE ($30.00\%$ vs $26.25\%$). This is a highly surprising finding that challenges the core thesis of direct quantization-aware merging, showing that direct optimization is not only unnecessary but actively detrimental due to STE-induced gradient noise.
* **The "Low-Capacity Generalization Illusion" (PEFT/Subspace):** The paper provides a highly novel critique of PEFT/low-rank ensembling space. While restricting merging to a low-rank SVD subspace closes the generalization gap ($+0.50\%$), the authors expose this as an illusion: the global SVD projection destroys vital representation directions and flattens activation ranges, making the model naturally insensitive to quantization parameters because its predictions are already highly degraded and approach noise.
* **Expected Gradient randomized smoothing (Axis 4):** Rather than just reporting that input noise helps, the authors mathematically formalize how Gaussian input-space perturbations act as randomized smoothing, transforming fragile, discrete rounding thresholds into smooth, expectation-based gradients that stabilize search trajectories.
* **Hybrid Optimization Pipeline (Algorithm 1):** The authors do not merely criticize STE; they propose a highly novel **Hybrid Optimization Pipeline** combining first-order coarse search (STE) with zero-order fine-grained search (1+1 ES) under Total Variation spatial regularization, offering a concrete algorithmic solution to navigate discontinuous landscapes.

## Characterization of Novelty
The novelty of this work is **significant and paradigm-shifting**. It moves the model merging field away from narrow, over-optimistic evaluation setups toward a realistic, hardware-aware, and deployment-centric standard. The conceptualization of quantization-operator overfitting, the exposing of the "Low-Capacity Generalization Illusion," and the formalization of the hybrid optimizer represent substantial theoretical and empirical advancements. This paper changes how the community must think about test-time weight-space fusion under compression constraints.
