# Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is described with commendable clarity. The authors provide:
- Clear mathematical formulations for each component (Equations 1--9).
- A detailed layout of how the early-layer routing paradox is resolved (restricting LoRA adapters to Layers 4--$L$).
- Explicit definitions of the physical PyTorch variants implemented (SPS-FP, SPS-SG, SPS-VSG, SPS-Compiled).
- Complete pseudo-code for the compiler-level Scatter-Gather loop in Appendix A.

## Appropriateness of Methods
The proposed methods are highly appropriate for the targeted systems-ML constraints:
- **SPS** leverages the linear additive nature of LoRA to blend activations, which is computationally elegant since the heavy base model weight execution is shared.
- **ZCA** utilizes early-layer representations to route, which is well-supported by representation theory (early layers capture domain-agnostic features, while mid-to-late layers specialize).
- **UNC, IDC, and GMM Shield** are standard, lightweight statistical tools that fit the low-resource constraint of edge devices.

## Potential Technical Flaws and Theoretical Gaps (Theorist Lens)
While the paper is highly complete from an engineering standpoint, a rigorous **theoretical evaluation** reveals several critical gaps, unproven assumptions, and methodological limitations:

1. **Lack of Rigorous Theoretical Underpinnings and Guarantees:**
   - The paper lacks formal proofs or mathematical guarantees for the recovery of the "Expert Ceiling" under SPS activation blending. Under what conditions on the representation spaces or expert parameters does sample-wise blending mathematically guarantee zero cross-task interference or perfect ceiling recovery?
   - The authors assume that task representations are "block-orthogonal normal distributions" in their simulation sandbox. While this represents a highly convenient assumption, there is no theoretical proof showing that real-world pre-trained representations actually satisfy such strict geometric properties, or how deviation from this assumption affects routing error bounds.
2. **Heuristic Nature of the Calibration Methods:**
   - **Intra-Task Dispersion Calibration (IDC):** The division of raw similarities by expected similarity scales ($u'_{k,b} = u_{k,b}/s_k$) is a heuristic scaling. What is the theoretical justification for this specific scaling factor? Does it guarantee optimal decision boundaries under multi-dimensional Gaussian assumptions?
   - **Adaptive Temperature Scaling:** The Shannon-entropy dependent temperature scaling ($\tau_b = \tau_0 \exp(\lambda H(u'_b))$) is introduced to handle borderline samples, but there is no proof or theoretical validation of its stability, convergence, or behavior under noise.
3. **Stated PAC Bound without Derivation or Context-Specific Proof:**
   - In Section 4.8, the authors state a standard PAC learning sample complexity bound: $N = \mathcal{O}\left( \frac{K + \log(1/\delta)}{\epsilon^2} \right)$ to justify the sample efficiency of Supervised Head Fine-Tuning (SHFT). However, they merely copy this standard formula from general learning theory literature without proving that it holds for their specific low-dimensional coordinate routing space, nor do they define the precise hypothesis class, loss function, or data-generation process associated with this bound.
4. **Simplistic GMM Assumption:**
   - The Coordinate GMM assumes a diagonal covariance structure. While this is computationally efficient and prevents singular covariance on small calibration splits, there is no theoretical analysis of how much information is lost by ignoring off-diagonal covariance terms, especially when expert task coordinates are correlated.
5. **No Theoretical Bounds on "Activation Bleeding":**
   - The authors identify "activation bleeding" as a boundary condition under overlapping task domains. However, they do not provide any theoretical bounds or error scaling analysis of how performance degrades as a function of task overlap (e.g., using mutual information, Wasserstein distance, or Fisher Separability Criterion).

## Reproducibility
The reproducibility of the work is high. The authors provide:
- Exact dataset specifications (MNIST, F-MNIST, CIFAR-10, SVHN).
- Precise model architectures (\texttt{vit\_tiny\_patch16\_224} and \texttt{gpt2}).
- Clear experimental hyperparameters (calibration split size $|\mathcal{C}_k| = 64$, temperature $\tau = 0.001$, mixture components $M=2$, covariance regularization $\gamma = 10^{-4}$).
- Complete pseudo-code for the fused loop layout.
- Detailed scaling, sensitivity, and ablation sweeps.
