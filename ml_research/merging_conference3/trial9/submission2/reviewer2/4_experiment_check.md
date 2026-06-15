# 4. Experiment Check

## Critical Evaluation of Experimental Setup
The paper evaluates RB-TopM on two main setups:
1. **14-layer Analytical Coordinate Sandbox (ICS):** A synthetic closed-loop simulation of model execution in a 192-dimensional representation space.
2. **TVM Compiler-Level NPU Simulation:** A pilot validation on a MobileNetV3-Large backbone running the DomainNet dataset, simulated on TVM v0.15 for an ARM Cortex-M7 core.

### Critique of the Synthetic ICS Sandbox
The "14-layer Analytical Coordinate Sandbox" is a highly simplified, synthetic simulation. It models a 14-layer model by projecting base samples onto 48-dimensional orthogonal task subspaces.
* **Critique:** In real-world deep neural networks, activations do not lie on clean, orthogonal, linear subspaces. They reside on highly non-linear, curved, and overlapping manifolds. By evaluating the routing algorithms on an artificially constructed orthogonal sandbox, the authors heavily inflate both the routing accuracy (e.g., near 100% on MNIST/Fashion-MNIST) and the GMM safety shield's performance. The sandbox fails to capture the true complexity of representation drift, manifold curvature, and domain overlap of physical deep networks.

### Critique of SVHN Noise Modeling
The SVHN task is modeled with extreme representation noise and head bias, resulting in a very low Expert Oracle accuracy of **21.68%** (barely above random guessing for a 10-class dataset).
* **Critique:** While the authors defend this as a deliberate "stress-test" simulating street-view sensor noise, serving an expert adapter with 21.68% accuracy is highly impractical. If the expert's performance is so poor, defaulting to the pre-trained base model (when OOD is flagged) is highly likely to perform similarly or better, which trivializes the "ensembling-regularization paradox." A more rigorous evaluation would use standard uncorrupted benchmarks where all experts achieve high standalone accuracy (which the authors briefly address in Section 4.3 point 6, but relegate to synthetic evaluation).

## Datasets and Baselines
* **Datasets:** MNIST, Fashion-MNIST, CIFAR-10, SVHN (for the sandbox), and DomainNet (Real, Clipart, Painting, Sketch) for the TVM pilot. These are standard and highly appropriate.
* **Baselines:** Expert Oracle, Uniform Merging, SABLE SOTA, SPS-ZCA, Q-SPS, and SOTA static merging (TIES, DARE). The baselines are strong, relevant, and well-contextualized.
* **Temperature Calibration Justification:** The authors provide a solid scientific justification for using different routing temperatures ($\tau=0.001$ vs. $\tau=0.05$) to match the literature-optimized settings of each baseline, ensuring a fair comparison.

## Do Results Support Claims?
* **Controllable Trade-off (Supported):** The sweeps across $C_{\text{budget}} \in [0, 1]$ successfully demonstrate a smooth, monotonic reduction in active experts and simulated latency.
* **Activation Dilution & Pruning Peak (Supported):** The non-monotonic accuracy peak at $C_{\text{budget}} = 0.4$ is empirically visible in both the sandbox (75.85% idealized accuracy vs. 75.37% at $C_{\text{budget}}=1.0$) and the TVM pilot (71.65% joint accuracy vs. 71.35% at $C_{\text{budget}}=1.0$), lending strong empirical support to the regularizing benefit of expert pruning.
* **OOD Protection (Partially Supported):** The GMM shield rejects 38.04% of high-noise OOD queries in the sandbox. However, the 13.75% test-set false-positive rate under unregularized calibration reveals a significant generalization gap that requires regularized calibration ($N=256$, 5-fold CV) to resolve, which is transparently documented.
* **Latency/Energy Savings (Partially Supported):** The 17.5% overall system latency reduction and 76.2% weight transfer reduction are reported on a *TVM compiler-level simulation*, not physical hardware. While Appendix F provides bare-metal physical measurements on an STM32 board to confirm these savings, the main text relies heavily on simulation.
