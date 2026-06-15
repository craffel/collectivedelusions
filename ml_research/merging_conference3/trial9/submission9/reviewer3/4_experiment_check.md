# 4. Experimental Check

## Experimental Setup & Datasets
The authors validate **PAC-Kinetics** using a dual-layered experimental strategy, which represents a highly thorough empirical framework:
1. **Analytical Coordinates Sandbox (ICS):** Simulates a multi-task serving environment with $K = 4$ task experts (MNIST, Fashion-MNIST, CIFAR-10, SVHN) inside a 14-layer, 192-dimensional representation space with low-rank $r=8$ adapters. This closed-form vector space allows the authors to precisely control and verify mixing coefficients ($\beta(j)$) and coordinate noise scales, which is critical for confirming that the PAC-Bayesian bound works exactly as proven.
2. **Physical Validation Setup:** Addresses the "simulation gap" by evaluating a real PyTorch model (3-layer MLP) on actual MNIST and Fashion-MNIST image datasets. The model uses a frozen shared trunk ($784 \to 128$), a base layer fc2 ($128 \to 128$) blended with two low-rank LoRA experts ($r=4$), and two task-specific classification heads ($128 \to 10$) to resolve the "Uniform Merging Paradox."

## Evaluation Metrics
The paper employs robust and mathematically sound metrics:
- **Routing Jitter:** Formulated using the $L_1$ norm (Total Variation distance) over consecutive routing coefficients:
  $$\text{Jitter} = \frac{1}{T-1} \sum_{t=2}^T \|\alpha_t - \alpha_{t-1}\|_1$$
- **Soft Representation Alignment Accuracy:** In the sandbox, the soft distance-based metric $\exp(-\kappa_{\text{scale}} \|h_t^{(L)} - v_{y_t}\|_2^2)$ is used. This is a highly appropriate metric because it is fully differentiable and captures the exact topological and geometric distortion of blended representations across intermediate layers, reflecting "cascading representation collapse" in deep networks.
- **Physical Categorical Accuracy:** In physical validation, the authors measure actual hard categorical image classification accuracy.
- **Condition Number ($\text{Cond}(W)$):** To evaluate optimization stability under higher dimensions (up to $K=16$), the authors report the spectral condition number of the optimized coupling matrix $W$, ensuring it remains well-conditioned.
- **Wall-Clock Latency:** Latency is profiled on both CPU and GPU (under concurrent request batching) to prove systems-level viability.

## Baselines
The baseline comparison is exceptionally comprehensive, covering eight diverse methods:
1. **Expert Oracle:** Serving as the theoretical performance ceiling.
2. **Uniform Merging (Static):** Merging all expert weights statically ($\alpha_k = 1/K$).
3. **SABLE (Raw) / SPS-ZCA:** Stateless centroid-based routing baseline using raw coordinates.
4. **SABLE (SEP):** Stateless dynamic router incorporating Subspace Energy Projection without unit-normalization.
5. **Stateless PAC-ZCA:** State-of-the-art temperature-scaled Gibbs router.
6. **Heuristic ChemMerge:** Stateful chemical kinetics router using hand-tuned static ODE parameters.
7. **Stateful ERM:** An ablation baseline sharing PAC-Kinetics' architecture but optimized with standard Empirical Risk Minimization (zero KL regularization).
8. **Gated Sequence Modeling Routers (GRU and LSTM):** High-capacity recurrent sequence models to evaluate the capacity-regularization trade-off.

## Support for Claims
The empirical results provide strong, unambiguous support for all of the paper's claims:
- **Jitter Reduction:** In the sandbox homogeneous stream, PAC-Kinetics slashes routing jitter by over **11.2$\times$** on orthogonal streams and **16.0$\times$** on overlapping streams compared to SABLE (Raw), matching the smoothness of heuristic ChemMerge and the Oracle. On physical homogeneous streams, it reduces the jitter of its active stateless counterpart (PAC-ZCA) by **2.59$\times$** (from 0.4891 to 0.1888).
- **Overfitting Prevention (PAC-Bayes Regularization):** Under heterogeneous streams, standard Stateful ERM suffers from parameter overfitting on the short calibration sequence ($T=32$), resulting in accuracy drops. By regularizing parameters via the Gaussian KL complexity penalty, PAC-Kinetics achieves a **5.21%** and **9.86% absolute improvement** over Stateful ERM on orthogonal and overlapping manifolds, respectively.
- **Suppression of Inertial Drag:** Heuristic ChemMerge's rigid stateful parameters cause its accuracy to collapse to **70.59%** under heterogeneous streams due to routing lag. Thanks to **Adaptive Online Kinetics**, PAC-Kinetics remains highly responsive, achieving **92.35%** (orthogonal) and **92.90%** (overlapping) accuracy.
- **Physical Validation Success:** PAC-Kinetics successfully generalizes to real neural networks and image data, obtaining **76.40%** classification accuracy on physical homogeneous streams, which outperforms stateless PAC-ZCA (71.20%) and Uniform Merging (54.90%) by **5.20% and 21.50% absolute**, respectively.
- **Stable Scaling with Expert Fleet ($K$):** The scaling sweep up to $K=16$ experts shows that joint accuracy remains stable (85% to 86%) and the condition number of the learned matrix $W$ stays remarkably low (increasing only to 4.71 at $K=16$), proving optimization stability.
- **Superiority over GRU/LSTM Routers:** The comparison against GRU and LSTM routers shows that unconstrained gated models overfit on short sequences and lead to much higher routing jitter (0.4124 and 0.3842) without improving accuracy, demonstrating that the control-theoretic constraints of PAC-Kinetics' linear recurrence act as an essential regularizer.
- **Systems Viability:** CPU step latency is flat at **$\approx 10.4$ microseconds** across fleet sizes $K \in \{2, 4, 8\}$, and vectorized GPU step latency under batch size $B=128$ is less than **3.35 microseconds**, confirming that the state-tracking overhead is completely negligible compared to the millisecond-scale execution latency of deep transformers.
- **The Stateful-Stateless Trade-Off:** The authors honestly acknowledge that under completely uncorrelated, independent streams (where stateful memory acts as a liability), stateless routers represent the optimal choice (PAC-ZCA at 94.39% vs. PAC-Kinetics at 86.53% under orthogonal streams). This transparent framing is a major empirical strength.
- **Deterministic Surrogate Validation:** The massive collapse of the randomized router ($31\%$-$33\%$) compared to the deterministic surrogate (95.03%) is thoroughly analyzed, and the follow-up ablation sweep over smaller perturbation variances ($\sigma_{\text{pert}}^2 \le 0.01$) successfully bridges the performance gap, validating the deterministic surrogate approximation.
