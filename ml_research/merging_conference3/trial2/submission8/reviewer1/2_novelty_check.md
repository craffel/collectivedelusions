# 2. Novelty Check

## Key Novel Aspects and "Delta" from Prior Work
The core delta of **NP-BTVP** relative to the existing model merging literature (specifically **TIES-Merging** and **DARE-Merging**) lies in several distinct conceptual and empirical areas:

1. **Deterministic Reciprocal Rescaling ($1/p$) on Sorted Magnitude-Based Masks:** 
   Prior work like **DARE** employs random Bernoulli dropout and scales the remaining updates by $1/(1-p_{\text{drop}})$ to maintain mathematical expectation ($\mathbb{E}[\tilde{\tau}] = \tau$). In contrast, **NP-BTVP** applies reciprocal scaling ($1/p$) to a *deterministic, magnitude-sorted* mask. 
   * **The Delta:** Because the mask is coordinate-aligned and deterministic rather than stochastic, this scaling cannot be justified probabilistically. The authors show that instead of preserving the exact $L_1$ norm, this scale factor mathematically *amplifies* the $L_1$ norm (e.g., $2.58\times$ to $3.30\times$ at 90% sparsity). They frame this as a **"Signal-Strength Boost"** that prevents highly sparse task-specific updates from being drowned out by the dense base pre-trained model weights. This is a novel and important conceptual reframing.

2. **The "Saliency Double-Bind" in Layer-Wise Budget Allocation:**
   While layer-wise pruning budget allocation is a standard concept in network compression, the paper introduces a novel analysis of how rescaling interacts with layer-wise budget variations.
   * **The Delta:** The authors identify a fundamental scale-preservation trade-off:
     * **Global Rescaling:** Under-scales low-saliency layers (essentially silencing them) and over-scales high-saliency layers (making them dominate and drown out others).
     * **Layer-wise Rescaling:** Amplifies random coordinate noise and outliers into massive updates in highly sparse layers (where $p_l$ is very small).
   * This double-bind elegantly justifies why simple global Uniform Pruning (NP-BTVP-U) is practically and empirically superior, saving practitioners from unnecessarily complex layer-wise optimization.

3. **Curvature Flatness vs. Coordinate Sparsification Resilience:**
   It has been widely assumed in the model merging literature that loss-landscape flatness (e.g., via SAM) improves robustness to all post-hoc perturbations.
   * **The Delta:** The authors systematically deconstruct this assumption, showing that under well-converged regimes, SAM experts do *not* exhibit higher resilience to coordinate-aligned unstructured magnitude pruning compared to standard AdamW experts. This reveals a valuable geometric insight: isotropic loss landscape flatness (Hessian eigenvalues) does not directly translate to unstructured coordinate-wise sparsification buffer.

4. **Joint Sparsification-Quantization Interaction:**
   The paper investigates how post-hoc pruning and 8-bit quantization (INT8) interact in a model-merging context. 
   * **The Delta:** They show that NP-BTVP-U works beautifully with INT8 (losing only 0.12% accuracy), whereas layer-wise Saliency Pruning (NP-BTVP-S) suffers catastrophic collapse because its high local scale factors amplify quantization rounding noise.

## Characterization of Novelty
We characterize the novelty of this work as **Significant and Exceptionally Practical**:

* **Academic Novelty (Moderate-to-High):** The individual tools—magnitude pruning, binary search, and reciprocal scaling—are classical compression techniques. However, their combination in a post-hoc, training-free model merging framework, coupled with the rigorous mathematical derivations of $L_1$-amplification under Laplace/Gaussian distributions, represents a highly solid conceptual advance.
* **Pragmatic Novelty (High):** From a practitioner's perspective, this work is highly novel and valuable. Instead of proposing complex, test-time optimizations that require calibration data and risk transductive overfitting (such as AdaMerging), or heavy second-order Hessian calculations (like Fisher/optimal-brain-surgeon methods), the authors show that a simple, $O(d \log d)$ sorting-based uniform pruning with a $1/p$ scale factor delivers outstanding multi-task performance. This provides immediate, out-of-the-box utility for real-world deployments.
