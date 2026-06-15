# 2. Novelty Check & Delta Analysis

## Overview of Novelty Claims
The paper claims to introduce a "novel wave-theoretic paradigm" for model merging, shifting the optimization from static, real-space averaging to continuous, complex-valued phase rotation in the Fourier frequency domain. 

## The "Delta" from Prior Work
The closest prior work is **FREE-Merging** (Zheng & Wang, 2024 / arXiv:2411.16815), which also applies 2D Discrete Fourier Transforms to neural network weights to resolve task interference. 
* **FREE-Merging Approach:** Applies static, non-adaptive Fourier high-pass and low-pass filters to task vectors to prune harmful frequency bands.
* **PhaseMerge "Delta":** Instead of static, hard-coded frequency boundaries, PhaseMerge introduces *learnable, differentiable phase-rotation parameters* ($\tilde{\phi}_k^l$) optimized end-to-end via prediction entropy. This is a technical step forward, turning frequency-domain filtering into an adaptive, optimization-based process.

## Characterization of Novelty: Highly Incremental & Theoretically Flawed
While presented with highly evocative, wave-theoretic terminology ("constructive/destructive wave interference," "complex wave superposition," "directional Hilbert transform"), the core novelty of applying learnable phase rotation is **highly incremental** and suffers from **fundamental theoretical inconsistencies**:

1. **Permutation Invariance vs. Coordinate-Dependent Fourier Basis:**
   Neural network weights in dense layers (e.g., self-attention projection and MLP blocks) are permutation-invariant. Swapping the rows and columns (with corresponding adjustments in adjacent layers) does not alter the mathematical function of the network. However, the Discrete Fourier Transform is highly coordinate-dependent, as its basis functions are complex exponentials evaluated over grid coordinates. 
   * Treating the rows and columns of a reshaped dense weight matrix as "spatial grid coordinates" is mathematically arbitrary. Adjacent elements in a dense matrix do not possess topological or physical spatial correlation.
   * Consequently, the concept of a 2D spatial frequency in a dense weight matrix is an artifact of the arbitrary layout of the matrix. If one permutes the neurons prior to the FFT, the resulting Fourier representation, amplitude, phase, and directional Hilbert transform will change completely.
   * This means the entire wave-theoretic framing ("constructive and destructive wave interference") lacks physical or functional justification for dense layers. It is not a physically grounded wave alignment; rather, it behaves as an arbitrary, coordinate-dependent structured random projection that acts as a low-dimensional regularizer.

2. **The Failure of the Spatial Frequency Smoothing Grid ($r=2$):**
   This theoretical mismatch is empirically confirmed by the paper's own ablation study (Section 4.4). The upsampled $2 \times 2$ grid parameterization ($r=2$, PhaseMerge) is designed to enforce spatial-frequency smoothing. Yet, it consistently and significantly *underperforms* the uniform scalar phase shift ($r=1$, U-PhaseMerge) by $2.08\%$ in FP32 and $1.50\%$ in 8-bit PTQ. 
   Because adjacent elements in a dense matrix are physically unordered, assuming spatial-frequency continuity across these dimensions is a false inductive bias. This further demonstrates that the wave-theoretic mechanics described in the paper break down when applied to dense layers.

3. **Hilbert Transform Duality is Coordinate-Dependent:**
   The paper provides an elegant proof (Theorem 3.1) showing that a uniform phase shift in the half-complex Fourier domain is equivalent in the spatial domain to:
   $$\tau_{\text{merged}} = \cos\phi \cdot \tau + \sin\phi \cdot \mathcal{H}(\tau)$$
   where $\mathcal{H}(\tau)$ is the directional Hilbert transform. While mathematically sound, the Hilbert transform of a dense weight matrix is entirely dependent on the arbitrary initial layout of the weights. Mixing a task vector with its coordinate-dependent Hilbert transform has no clear semantic or functional relationship to task-specific feature alignment.

4. **Worse than Existing Real-Space Baselines:**
   A critical metric for evaluating the significance of any methodological novelty is whether it improves upon the state of the art. The paper's own experiments show that **PolyMerge** (a real-space depth-wise polynomial scaling baseline) consistently and substantially outperforms PhaseMerge and U-PhaseMerge by **5.17% to 6.00% absolute accuracy** across all settings (FP32, 8-bit PTQ, 4-bit PTQ). The proposed method fails to establish empirical superiority over much simpler, existing real-space alternatives, limiting its practical significance.
