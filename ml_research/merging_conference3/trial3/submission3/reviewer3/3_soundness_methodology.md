# 3. Soundness and Methodology

## Clarity and Completeness of Description
The methodology of FlatMerge is described with excellent mathematical clarity and structural completeness:
- The task vector merging formulation (Eq. 1 & 2) and the polynomial subspace projection (Eq. 3) are clearly defined.
- The unsupervised entropy minimization objective (Eq. 4) is standard and well-situated.
- The transition from point-wise minimization to flatness-aware randomized smoothing (Eq. 5) is theoretically sound and backed by established optimization literature.
- The zeroth-order randomized smoothing gradient estimator (Eq. 7) is mathematically correct and provides a clear, backpropagation-free path for parameter updates.
- Algorithm 1 outlines the complete ZO-FlatMerge TTA workflow with high specificity, listing all inputs, outputs, and loop sequences.

## Appropriateness of Methods
The choice of methods is highly appropriate for the target scenario (resource-constrained edge deployment under physical noise):
- **Polynomial Subspace:** Restricting the optimization to $D = K \times (d+1)$ parameters mathematically filters out high-frequency spatial variation across layers.
- **Zeroth-Order (ZO) Optimization:** Standard backpropagation is extremely expensive for on-device adaptation because it requires storing high-dimensional intermediate activation maps. Bypassing this via a gradient-free ZO approach is a highly pragmatic engineering decision that trades off a manageable latency overhead for complete protection against activation memory overflows.
- **Flatness-Aware Smoothing:** Optimizing a smoothed loss over random perturbations prevents the coefficients from drifting into sharp, noise-distorted entropy valleys, enhancing out-of-distribution robustness.

## Technical Flaws and Critical Trade-offs
While the methodology is mathematically sound, there are several critical technical trade-offs and unaddressed aspects that must be highlighted:
1. **Static DRAM Memory vs. Dynamic Activation Memory Trade-off:**
   The paper argues that FlatMerge is highly memory-efficient because it requires **exactly 0 MB of activation caching** (dynamic memory). However, to perform weight reconstruction dynamically on-the-fly, FlatMerge must simultaneously store in DRAM:
   - The pre-trained base model $\Theta_{\text{base}}$ ($340.11\text{ MB}$ for ViT-B/32)
   - All $K$ task-specific expert vectors $\{\mathbf{\Delta}_k\}_{k=1}^K$ ($4 \times 340.11\text{ MB} = 1360.44\text{ MB}$)
   - The active merged model weights $\Theta_{\text{merged}}$ ($340.11\text{ MB}$)
   This results in a static memory allocation of **$2040.66\text{ MB}$**. In contrast, standard weight-space TTA only needs to load the active weights, gradients, and optimizer moments, totaling **$1360.28\text{ MB}$**. Therefore, FlatMerge **saves dynamic activation memory at the expense of requiring $1.5\times$ more static DRAM allocation**. For edge devices with extremely tight DRAM budgets (e.g., microcontrollers or ultra-low-power accelerators), this static memory inflation could be a major bottleneck. The authors should explicitly discuss and clarify this static memory overhead.
2. **High Latency Overhead of ZO Forward Perturbations:**
   The ZO-FlatMerge gradient estimator (Eq. 7) requires evaluating the entropy loss for $B_{\text{zo}}$ positive and $B_{\text{zo}}$ negative perturbations, which translates to running $2 \times B_{\text{zo}}$ forward passes and weight reconstructions per step. For $B_{\text{zo}} = 10$, this increases step latency by $3.73\times$ compared to standard first-order weight-space TTA ($27716.21\text{ ms/step}$ vs. $7427.37\text{ ms/step}$). While the authors propose a highly sensible mitigation—**asynchronous, periodic adaptation**—which reduces amortized overhead to $0.027\times$, this mitigation relies on the assumption of a slow-evolving physical environment. Under rapidly changing non-stationary noise, periodic adaptation may fail to react in time, forcing a trade-off between latency and adaptation speed.
3. **Unevaluated Adaptive Perturbation Radius:**
   In Section 3.3 (Eq. 8), the authors present a mathematically elegant formulation for an "Adaptive Perturbation Radius" $\sigma(X)$ to handle non-stationary noise. However, they explicitly state: *"We leave the empirical exploration of this adaptive formulation to future work."* This means a core component of their conceptual robustness framework remains completely unevaluated. Introducing theoretical formulations without any empirical validation diminishes the scientific rigor of the methodology section.

## Reproducibility
The reproducibility of this paper is **excellent**. The authors provide:
- Precise model dimensions and parameter counts (85.02M ViT-B/32, 108K MLP, 250K CNN).
- Exact dataset split sizes, optimization steps ($T=100$), perturbation scale ($\sigma=0.05$), sample size ($B_{\text{zo}}=10$), and learning rates.
- The exact formulation of their continuous simulation environments (Model I and Model II), which allows researchers to easily replicate the surrogate loss landscapes and evaluate alternative merging algorithms.
- Fully detailed training budgets and fine-tuning epochs for their physical MLP and CNN experts.
- A baseline script (`run_mock_review.sh`) and detailed hardware profiling details (Section 3.5), allowing researchers to easily run and profile the adapter.
