# Review of "PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging"

## Summary of the Submission
This paper introduces **PhaseMerge**, a frequency-domain model-merging framework designed to blend multiple task-specific expert neural networks without joint multi-task retraining. To address task conflicts and post-training quantization (PTQ) noise, PhaseMerge projects real-valued spatial task vectors (weight updates) into complex-valued Fourier space using the 2D Real Fast Fourier Transform (RFFT2D), decomposing the updates into amplitude and phase. 

The framework optimizes a differentiable phase-rotation mechanism to align complementary features or cancel conflicting updates, exploring two parameterizations: a uniform scalar phase shift per layer (U-PhaseMerge, $r=1$) and a continuous bilinear phase grid upsampled to the frequency tensor dimensions (PhaseMerge, $r=2$). Both configurations are constrained by a symmetry-preserving frequency mask to ensure real-valued spatial reconstruction via the Inverse RFFT2D (IRFFT2D). The reconstructed updates are combined using learnable task-wise scaling coefficients ($s_k$). 

For edge deployment, the merged weights are projected through a PTQ operator (e.g., 8-bit or 4-bit), and the phase parameters are optimized by minimizing unsupervised prediction entropy on a small calibration stream ($D_{\text{cal}}$) using Straight-Through Estimation (STE) for backpropagation.

---

## Overall Assessment
The paper presents a highly creative and interdisciplinary concept, modeling parameter merging as wave superposition in the complex Fourier frequency domain. The mathematical formulation is thorough, and the paper is exceptionally well-written, with high-quality figures and an extensive appendix. 

However, the submission suffers from several fundamental conceptual and technical flaws that outweigh its merits:
1. **Conceptual Incoherence:** Applying Fourier transforms to permutation-invariant dense weights lacks physical justification, as "frequencies" are implementation-dependent artifacts.
2. **Severe Empirical Deficit:** The proposed method is heavily outperformed by PolyMerge, a simple 12-parameter real-space baseline, by $5\%$ to $7\%$ absolute accuracy.
3. **Toy-Scale Evaluation:** The experiments are restricted to a tiny 5M-parameter ViT and severely subsampled datasets with only 100 test samples.
4. **Optimization Pathology:** The method degrades in performance and spikes in variance when provided with larger calibration streams ($M=32$), contradicting the central claim of resolving the Overfitting-Optimizer Paradox.

Therefore, while the idea is academically interesting, it is not ready for publication in its current form. I recommend a **Weak Reject**, and outline the specific areas that require revision below.

---

## Rating Recommendations

- **Soundness:** **Fair**  
  *Justification:* While the mathematical derivations are detailed, the application of 2D FFT to permutation-invariant dense matrices is conceptually flawed. Furthermore, the optimization exhibits severe pathologies (performance degradation and variance spikes as calibration data increases), and a key claim regarding the symmetry-preserving mask is entirely un-ablated.
- **Presentation:** **Excellent**  
  *Justification:* The paper is beautifully written, highly structured, and easy to follow. The notation is clean, the figures are professional, and the appendix is exceptionally thorough, proactively addressing scalability, CNN extensions, and future roadmaps.
- **Significance:** **Poor**  
  *Justification:* Since the proposed framework is substantially outperformed by a simple, easy-to-implement, 12-parameter real-space polynomial baseline (PolyMerge), there is no practical incentive for researchers or practitioners to adopt this highly complex frequency-domain parameterization.
- **Originality:** **Good**  
  *Justification:* The shift from standard real-space weight interpolation to continuous, differentiable phase-rotation in the complex frequency domain represents a highly creative and refreshing interdisciplinary concept.
- **Overall Recommendation:** **3: Weak Reject**  
  *Justification:* The paper has clear conceptual and mathematical merits, but its core methodology is theoretically questionable for dense layers, its empirical performance is significantly behind a simple baseline, its evaluation is limited to toy-scale datasets, and the optimization shows instability under standard scaling. Revisions—such as evaluating on convolutional layers, implementing the proposed PolyPhaseMerge hybrid, and scaling to realistic datasets—are required before this work can be meaningfully built upon by the community.

---

## Detailed Strengths and Weaknesses

### Strengths
1. **Innovative Interdisciplinary Formulation:** Modeling model-merging as complex-valued wave superposition and phase rotation is a highly refreshing conceptual leap that departs from standard real-space coordinate interpolations.
2. **Theoretical Depth:** The proof of Theorem 1 establishes a formal dual connection between frequency-domain phase-rotations and spatial-domain rotations in a 2D subspace spanned by the task vector and its directional Hilbert transform.
3. **Zero Inference Latency:** Since the optimized frequency-space updates are reconstructed back into the spatial domain and added directly to the pre-trained weights prior to deployment, PhaseMerge introduces exactly zero computational overhead or library dependency at inference/deployment time.
4. **Structured Macro-Micro Decoupling:** Decoupling global task-wide scaling coefficients ($s_k$) from layer-wise phase-shift parameters is an excellent design choice that prevents the optimizer from navigating hundreds of unregularized amplitudes, preserving optimization stability.
5. **Outstanding Presentation Quality:** The manuscript is exceptionally polished, with clear figures, well-structured sections, and detailed appendices that proactively detail CNN topologies, generative foundation scaling, and formal formulations of future directions.

### Weaknesses

#### 1. Conceptual Incoherence: Fourier Transforms on Permutation-Invariant Weights
The fundamental mathematical justification for the 2D Discrete Fourier Transform relies on a **spatial or grid topology**, where adjacent elements share continuous coordinate relationships (such as adjacent pixels in an image). Dense layers in neural networks, however, are fundamentally **permutation-invariant**: the rows and columns can be arbitrarily swapped (provided corresponding input/output indices are aligned) without changing the network's function. 
Swapping neuron indices completely alters the spatial layout of the weight matrix, resulting in a completely different 2D FFT representation. Therefore, concepts such as "frequencies," "wave coherence," and "phase alignment" when applied to dense layers are completely arbitrary artifacts of the implementation's memory layout rather than physical properties of the parameters. While the authors lock the coordinate indices to a shared base model, this is a superficial fix that does not resolve the underlying mathematical mismatch of applying continuous Fourier transforms to non-spatial parameters.

#### 2. Severe Empirical Deficit Relative to a Simple Real-Space Baseline
The primary objective of introducing a new model parameterization is to unlock superior performance or efficiency. However, the proposed PhaseMerge framework is consistently and heavily dominated by **PolyMerge**—a simple, real-space baseline that constrains layer-wise merging coefficients using a low-degree polynomial:
- **FP32:** PolyMerge achieves **$48.00 \pm 1.62\%$** average accuracy, whereas U-PhaseMerge achieves $42.83 \pm 1.76\%$ (a drop of $5.17\%$) and PhaseMerge ($r=2$) achieves $40.75 \pm 1.43\%$ (a drop of $7.25\%$).
- **8-bit PTQ:** PolyMerge gets **$48.00 \pm 1.47\%$**, beating U-PhaseMerge ($42.33\%$) by $+5.67\%$.
- **4-bit PTQ:** PolyMerge gets **$43.42 \pm 1.30\%$**, beating U-PhaseMerge ($37.42\%$) by $+6.00\%$.

PhaseMerge introduces substantial mathematical and computational complexity (RFFT2D, IRFFT2D, complex numbers, Straight-Through Estimators, and symmetry-preserving masks) only to perform significantly worse than a straightforward 12-parameter real-space polynomial baseline. This severely limits the practical significance of the work.

#### 3. Core Proposed Method Underperforms Its Simplest Variant
In Table 4 (Ablation Study), the primary proposed method, **PhaseMerge ($r=2$)** which uses the $2\times 2$ bilinear upsampling grid, is consistently **outperformed by the simpler U-PhaseMerge ($r=1$)** which uses a single scalar phase shift per layer:
- U-PhaseMerge is $+2.08\%$ better in FP32 ($42.83\%$ vs $40.75\%$) and $+1.50\%$ better in 8-bit ($42.33\%$ vs $40.83\%$).

This result demonstrates that the spatially-continuous frequency-smoothing grid ($r=2$), which is a major part of the paper's novelty and complexity, is actually **detrimental** to performance on dense layers. The authors acknowledge that this is because dense layers lack spatial correlation. This empirical finding directly validates the theoretical critique that 2D frequency representations are fundamentally mismatched for dense weight matrices.

#### 4. Toy-Scale and Statistically Weak Evaluation
The experimental evaluation is restricted to an extremely small, obsolete setup:
- The backbone model is `vit_tiny_patch16_224` which has only ~5 million parameters.
- The downstream tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) are toy datasets.
- Fine-tuning is limited to only 500 training samples per task, and most critically, **evaluation test sets are subsampled to only 100 samples per task**.
Evaluating on 100 test samples introduces immense statistical noise, where a single correct or incorrect prediction shifts the accuracy by $1\%$. This explain why the reported standard deviations are so high (up to $4.03\%$). This toy-scale setup is insufficient to prove the modern applicability or statistical validity of the method.

#### 5. Catastrophic Performance Collapse After Merging
The absolute multi-task average accuracies reported are extremely low (collapsing from a $78.17\%$ average individual expert performance to $40.75\%$ for PhaseMerge and $48.00\%$ for PolyMerge). This catastrophic loss of over $30-37\%$ absolute performance indicates that the merged models are practically non-functional, barely outperforming random guessing. In a realistic model-merging scenario, merged models must maintain a significant portion of the expert capabilities. This artificial setup raises serious doubts about the practical validity of the experimental results.

#### 6. Optimization Pathology and Contradictory Scaling Behavior
The central motivation of PhaseMerge is to avoid the "Overfitting-Optimizer Paradox" under extreme data scarcity. However, in Table 2, as the calibration stream size $M$ increases from $4$ to $32$, **the performance of U-PhaseMerge ($r=1$) actually degrades and becomes highly unstable**:
- $M=4$: $42.42 \pm 1.64\%$
- $M=16$: $42.33 \pm 1.76\%$
- $M=32$: $40.67 \pm 3.65\%$ (accuracy drops and standard deviation spikes to $3.65\%$).
In contrast, the unconstrained baseline **AdaMerging improves steadily** as more data is provided ($40.75\% \to 41.67\% \to 42.50\%$). 
A robust optimization framework should exhibit better performance and lower variance when provided with more calibration data. The fact that PhaseMerge degrades and becomes highly unstable as $M$ increases indicates severe optimization pathology, completely undermining the claim that PhaseMerge successfully resolves the Overfitting-Optimizer Paradox.

#### 7. Fragile Calibration Budget
To prevent "class collapse" during unsupervised prediction entropy minimization, the authors restrict the test-time calibration budget to exactly 5 optimization steps. If an optimization framework is so fragile that running it for more than 5 steps triggers catastrophic parameter collapse, the underlying optimization space and objective are highly unstable. Relying on an extremely short, arbitrary stopping criterion is an ad-hoc band-aid rather than a robust, principled solution.

#### 8. Overstated and Un-Ablated Mask Claim
In Section 3.4, the authors claim that the symmetry-preserving frequency mask $M_{\text{sym}}$ "dramatically stabilizes optimization in complex frequency-space and prevents performance degradation," referencing "our experimental sweeps." However, there is no ablation study or quantitative comparison showing results without this mask anywhere in the paper. Proclaiming a component is essential without providing any empirical evidence of its omission is a major scientific flaw.

---

## Questions and Clarifications for the Authors

1. **Weight Reshaping/Padding:** How are the dense weight matrices of varying, non-square dimensions (e.g., $192\times 576$ or $192\times 768$) reshaped or padded to apply the 2D Real Fourier Transform? Since the DFT is coordinate-dependent, how does the choice of reshaping or transposition impact the results?
2. **Symmetry Mask Ablation:** Can you provide a quantitative ablation study showing the optimization stability and multi-task average accuracy of PhaseMerge with and without the symmetry-preserving frequency mask $M_{\text{sym}}$?
3. **M=32 Performance Degradation:** Why does U-PhaseMerge's performance degrade and its variance spike as the calibration stream size increases from $M=4$ to $M=32$, while the unconstrained baseline (AdaMerging) improves? How does this behavior align with the claim of resolving the Overfitting-Optimizer Paradox?
4. **PolyPhaseMerge Evaluation:** Since the proposed PolyPhaseMerge hybrid in Appendix A.3 is theoretically designed to integrate depth-wise coordinate constraint (remedying the lack of depth coordination in PhaseMerge) and parameter-efficiency, why was it not implemented and evaluated? Enforcing a quadratic polynomial constraint on phase angles across depth would likely close the empirical gap to PolyMerge, making it a critical addition.
5. **Convolutional Layer Evaluation:** In Appendix A.4, you discuss the physical topological alignment of convolutional layers, where $r=2$ continuous frequency-space phase rotations are expected to significantly outperform $r=1$ uniform phase shifts. Since dense layers are a mismatch for 2D FFTs, why did the authors choose to evaluate on a Vision Transformer (dense layers) rather than a CNN (convolutional layers) where the wave-mechanics framing is physically and mathematically coherent?
