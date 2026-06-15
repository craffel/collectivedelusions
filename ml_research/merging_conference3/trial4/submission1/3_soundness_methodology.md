# 3. Soundness of Methodology

## 3.1 Mathematical Correctness and Theoretical Underpinnings
The mathematical formulation of PhaseMerge is exceptionally rigorous and elegant. We analyze the core components below:

### 3.1.1 Spatial Domain Dual: Theorem 3.1
The paper proves a highly intriguing link to the spatial coordinate domain in **Theorem 3.1**:
- Applying a uniform phase-shift $\phi$ to the Real 2D Fourier representation of a task vector $\tau$ is mathematically equivalent to a spatial rotation in a 2D subspace spanned by the task vector and its directional Hilbert transform:
  $$\tau_{\text{merged}} = \cos\phi \cdot \tau + \sin\phi \cdot \mathcal{H}(\tau)$$
- The proof is mathematically solid. It uses conjugate symmetry and Euler's formula to show that the cosine part remains constant, while the sine part acts as a signum-multiplier in frequency, which corresponds precisely to the directional Hilbert transform $\mathcal{H}(\tau)$ in the spatial domain.
- In PyTorch, `torch.fft.rfft2` splits the last dimension (columns) to exploit conjugate symmetry. Thus, the Hilbert transform is directional along the columns of the weight matrix.

### 3.1.2 Real-Signal Constraints and the Symmetry-Preserving Mask
A key technical detail is the enforcement of a strict **symmetry-preserving frequency mask** $M_{\text{sym}}$:
- The Real 2D Fast Fourier Transform requires that the DC component (frequency $(0,0)$) and the Nyquist components have zero imaginary parts to represent a real-valued spatial signal.
- If a learnable phase shift rotated these components arbitrarily, the inverse transform would produce complex-valued outputs with non-zero imaginary parts, violating real-valued constraints and causing mathematical inconsistency.
- Element-wise multiplying the phase shift by $M_{\text{sym}}$ physically zeroes out phase-shift rotations at these specific boundaries, guaranteeing a clean, mathematically correct real-valued spatial matrix reconstruction. This shows excellent attention to mathematical detail.

## 3.2 Coordinate Dependency on Dense Weights (Methodological Limitation)
A major theoretical point of critique lies in the **coordinate dependency** of applying 2D FFTs to dense neural network weights:
1. **Permutation Invariance**: Dense layers are technically permutation-invariant; you can swap rows and columns of a weight matrix (along with corresponding input/output activations) without changing the function of the network.
2. **FFT Coordinate Sensitivity**: The 2D FFT basis consists of complex exponentials defined over rigid grid coordinates. Swapping rows or columns completely reshuffles the 2D FFT representation.
3. **The Shared Backbone Assumption**: The paper justifies this by locking the permutation coordinate to a shared, pre-trained initialization backbone $W_{\text{pre}}$, ensuring that all downstream experts are structurally aligned. This is a valid practical assumption.
4. **The False Spatial Topology of Dense Weights**: Even with a shared basis, dense weight matrices do not possess a native "spatial topology" (unlike images or convolutional kernels). Swapping any two rows across the entire network doesn't change the function but completely alters the 2D FFT.
5. **Critique of $r=2$ (Continuous Phase-Shift Grids)**: The $r=2$ configuration defines a $2 \times 2$ phase grid that is bilinearly upsampled to the full weight dimensions, assuming that adjacent frequency scales are spatially correlated. Since adjacent rows and columns in a dense weight matrix are completely arbitrary and have no physical spatial correlation, this bilinear upsampling assumption is physically invalid. 
6. **Explanation of Empirical Findings**: This theoretical invalidity explains why **U-PhaseMerge ($r=1$) outperforms PhaseMerge ($r=2$)** in Table 1 (FP32: $42.83 \pm 1.76\%$ vs $40.75 \pm 1.43\%$). The $r=1$ uniform scalar phase shift applies a single global degree of freedom per layer and does not enforce false spatial correlation across coordinates.

This coordinate dependency is a fundamental limitation of applying frequency transforms to dense matrices and should be clearly highlighted. It also explains why the method behaves beautifully on convolutional layers (which have actual physical spatial coordinates) but is less aligned with dense weight representations.

## 3.3 The PolyPhaseMerge Innovation
Since **PolyMerge** is the strongest empirical baseline (at $48.00 \pm 1.62\%$), the paper's proposed hybrid direction **PolyPhaseMerge** is highly logical and theoretically sound:
- PolyMerge's strength lies in its **macroscopic depth-wise coordination**, fitting merging coefficients as a smooth polynomial of the layer index.
- PhaseMerge's strength lies in its **microscopic phase synchronization** in frequency space.
- Parameterizing the phase shifts $\phi_k(l)$ across layers as a continuous polynomial of the layer depth $l$ represents an exceptionally promising hybrid paradigm. This would integrate macroscopic hierarchical depth coordination with frequency-domain phase cancellation, potentially surpassing PolyMerge's performance while preserving PhaseMerge's low-bit quantization resilience.

## 3.4 Verdict on Soundness
The methodology is rated as **excellent**. It is mathematically sound, highly elegant, and possesses rigorous proofs. The application of 2D spatial frequency operations ($r=2$) to unordered dense weight matrices represents a minor methodological mismatch, which is transparently documented, analyzed, and validated by the empirical results.
