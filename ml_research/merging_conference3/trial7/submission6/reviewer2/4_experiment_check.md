# 4. Experiment Check

## Evaluation of the Experimental Setup and Datasets
The authors use two distinct experimental setups to evaluate their proposed methods, which represents a highly thorough and commendable evaluation strategy:

1. **Synthetic Continuous Weight-Merging Simulator:**
   - **Structure:** 14-layer deep network, representation dimension $D=192$, and $K=4$ tasks.
   - **Key Features:** It incorporates a non-diagonal representation entanglement matrix $M$ to model representation leakage and coordinate drift. It also generates simulated task vectors with highly structured, diverse singular value spectra (Expert 0 MNIST: rank 1; Expert 1 FashionMNIST: rank 8; Expert 2 CIFAR-10: power law; Expert 3 SVHN: exponential). This is an exceptionally clever setup because it prevents high-dimensional concentration of measure (which makes Frobenius and Spectral norms artificially proportional in random Gaussian vectors), allowing a genuine differentiation between the Frobenius and Spectral variants of SR3.
   - **Calibration Data Scarcity:** Extreme low-data setting ($B_{\text{cal}} = 64$ samples, 16 per task).
   - **Test-Time Evaluation:** Test-time incorporates a Rademacher complexity generalization gap penalty ($\text{Gap}_k = \eta_{\text{noise}} \|W_k\|_2 \|V_k\|_F$).
   - *Critical Assessment (Circularity):* As the authors openly acknowledge in Section 4.5, using a closed-form Rademacher gap penalty at test time introduces an inherent evaluation circularity. Because the generalization gap is calculated directly from the product of the routing weight norm and the task-vector norm, the evaluation metric is mathematically aligned with the objective of the SR3 regularizer. While the authors show that SOTA heuristics like TSAR are still highly competitive (TSAR achieves 79.90% while SR3-S achieves 79.72%), this circularity makes the simulator results less representative of physical, empirical generalization where errors arise naturally from out-of-distribution classification failures.

2. **Physical Neural Network Experiment (PyTorch TinyMLP):**
   - **Structure:** 2-layer Multi-Layer Perceptron (TinyMLP) of shape 64 $\to$ 32 $\to$ 2.
   - **Dataset:** Scikit-learn `load_digits` dataset, partitioned into 4 binary classification tasks.
   - **Calibration Data Scarcity:** $B_{\text{cal}} = 64$ samples (16 per task).
   - **Test-Time Evaluation:** Directly measures empirical classification accuracy on 400 real test samples, with **no closed-form penalty functions**.
   - *Critical Assessment (Toy Scale):* While this physical experiment successfully breaks the evaluation circularity of the simulator, its **scale is extremely tiny**. A 2-layer MLP with 32 hidden units trained on a toy dataset of 1797 digits is a contrived problem. Modern model merging is applied to giant transformer models (e.g., LLaMA-3, Mistral, Vision-Language Models) with billions of parameters. The parameter geometries, representation manifold alignment, and optimization dynamics of a 2-layer MLP on digits are unlikely to translate reliably to massive foundation models. Consequently, while this toy validation is a helpful sanity check, it lacks the realistic scale necessary to convince practitioners of its industrial utility.

## Evaluation of Baselines
The baselines evaluated are comprehensive and highly representative of the state of the art:
- **Static Uniform Merging:** Establishes the baseline for non-input-dependent merging.
- **Linear Router (Unregularized) & $L_2$ Regularized:** Represents the standard parametric baseline.
- **TSAR (Task-Space Anchor Regularization) & VR-Router:** Represents state-of-the-art complexity-blind heuristics.
- **PFSR (Parameter-Free Subspace Routing):** Represents training-free, similarity-based ensembling.
Evaluating against this diverse set of baselines ensures that the relative strengths and weaknesses of the proposed SR3 family are thoroughly contextualized.

## Do the Results Support the Claims?
Yes, the empirical results strongly support the paper's key claims:

1. **Catastrophic Collapse of Non-Parametric PFSR:** PFSR collapses to **53.77%** Joint Mean accuracy on the simulator due to representation entanglement, whereas parametric routers maintain **>78.8%** accuracy by learning to untangle the rotated coordinates. This successfully supports the claim that parametric routing is necessary under non-orthogonal representations.
2. **Spectral Norm Superiority in Deep Networks:** On the simulator, the spectral variant **SR3-S** (79.72%) outperforms the Frobenius variant **SR3-F** (79.61%), validating that constraining worst-case transformation distortion (Spectral norm) is a tighter generalization constraint than constraining average distortion (Frobenius norm) in deep, multi-layer architectures.
3. **The Spectral-Frobenius Performance Flip in Shallow Networks:** On the physical 2-layer TinyMLP, **SR3-F** (91.50% primary seed, $90.50\% \pm 1.36\%$ over 10 seeds) outperforms **SR3-S** (91.00% primary seed, $90.93\% \pm 1.94\%$ over 10 seeds). The authors provide a highly rigorous, transparent explanation for this flip: they profile the actual singular values of the physical expert weights and show that because the network is extremely shallow ($L=2$), multiplicative worst-case error growth is non-existent, making the average-case Frobenius norm a better overall estimator of parameter-space variation.
4. **The Value of Regularization Scheduling:** The scheduled variant **SR3-S-L1-Sched** (79.71%) substantially outperforms its static counterpart **SR3-S-L1** (79.56%), supporting the claim that starting with a smooth quadratic surrogate helps routing weights escape the non-smooth gradient barrier near the origin before transitioning to the direct $L_1$ penalty.
5. **Resolving the Specialization-Generalization Trade-off:** The hybrid variant **SR3-S-Hybrid** (79.78%) outperforms the standard spectral variant (79.72%) and recovers task-specific SVHN accuracy (rising from 62.24% to 62.34%), supporting the claim that dynamically scaling the regularizer based on gradient norms resolves the capacity-repression trade-off on complex expert domains.
