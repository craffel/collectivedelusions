# 4. Experimental Check and Empirical Evaluation

## Experimental Setup and Datasets
The authors evaluate their method using a ViT-Tiny backbone ($D=192$, $L=14$ layers) inside a **"Controlled Representation Sandbox."** They use representations from four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
While the sandbox is highly detailed, it is **entirely synthetic and toy-scale**:
1. **Gaussian-Simulated Expert Weights:** The expert task vectors $V_k$ are generated using independent Gaussian parameters ($\mathcal{N}(0, I)$) instead of utilizing real, fine-tuned expert weights from standard models. This makes the setup highly artificial.
2. **Prototype Evaluation:** Task classes are evaluated using simple class prototype vectors in feature space, rather than running complete forward/backward passes through a fully fine-tuned deep classification network.
3. **The SVHN Noise/Floor Effect Confounder:** The authors deliberately inject a high noise scale ($\sigma = 0.90$) to SVHN, forcing its expert ceiling down to an extremely low 16.8% (just 6.8% above random guessing). This creates a "floor effect" where all ensembling methods are compressed into a tiny band of 9.6% to 14.4% accuracy. Although scientifically honest to admit, this artificial degradation masks the true severity of EHPB's collapse. On the clean CIFAR-10 task, EHPB collapses from an 81.6% ceiling down to a near-random 12.0% (a catastrophic **-69.6% drop**), whereas on SVHN the drop appears minor only due to this floor effect.

## Baselines
The baselines evaluated include:
- **Uniform Merging:** Simply averaging the specialized weights.
- **Global Linear Router / Vectorized Direct Router (`vmap`-Linear-Router):** Direct sample-wise routing without holographic superposition.
- **QWS-Merge:** A wave-inspired quantum phase-interference ensembling method.
- **L3-Routers:** Layer-wise low-dimensional routers (Linear, Softmax, Tanh) with and without regularization.

While the selection of baselines is comprehensive, the empirical results **severely undermine the proposed method**:
- **Hadamard Dominance Paradox:** Static **Uniform Merging** achieves a Joint Mean accuracy of **52.3%** with **zero training, zero extra parameters, and zero latency overhead**. EHPB achieves only **25.4%** (a catastrophic **-26.9% absolute gap**). Even with the uncompressed 5% coordinates in **Residual-EHPB**, the performance (33.7%) is still far below a simple static average (52.3%).
- **Direct Vectorized Routing Dominance:** The vectorized direct router (`vmap`-Linear-Router) achieves **51.0% Joint Mean** (more than double EHPB's accuracy) and is also completely immune to heterogeneity collapse (0.0% delta). Although direct routing requires $O(K \times P)$ parameter storage, saving storage with EHPB is completely irrelevant if the resulting model is degraded to the point of being unusable.

## Support for Central Claims
1. **Claim: EHPB is immune to heterogeneity collapse.**
   *Supported?* **Yes**, EHPB and `vmap`-Linear-Router achieve 0.0% performance drop under mixed-task batches. However, `vmap`-Linear-Router also achieves this while maintaining far superior accuracy (51.0% vs. 25.4%).
2. **Claim: Dimension scaling sweep deconstructs Hadamard boundary.**
   *Supported?* **Yes**, the logarithmic dimension sweep successfully shows that relative reconstruction error is scale-invariant (~170%–179%) due to coordinate isolation.
3. **Claim: Continuous Cleanup Networks (CCN) can denoise activations and rescue performance.**
   *Supported?* **Only partially, and with major caveats.** While CCN reduces intermediate layer-wise MSE (e.g., 8.1$\times$ reduction at Layer 3) and rescues MNIST accuracy (from 61.2% to 81.2%), it actually **degrades performance on more complex tasks** (e.g., FashionMNIST collapses from 26.8% to 8.0%) due to projection distortions. 
4. **Claim: ReLU Post-Hoc Bias Correction stabilizes representation propagation.**
   *Supported?* **Only on a highly simplified toy simulation.** The authors demonstrate that analytic subtraction or learnable scale/shift improves representation fidelity in a simulated 5-layer deep ReLU vector propagation network. However, they do **not** implement or evaluate this bias correction within the actual ViT-Tiny multi-task visual classification backbone, leaving its practical benefit unverified.
5. **Claim: Structured Row-wise Residual-EHPB provides a hardware-friendly rescue.**
   *Supported?* **Yes, but only in terms of weight reconstruction error.** The authors show that structured row-wise masking incurs a small 7.77% absolute increase in relative weight reconstruction error compared to unstructured masking. However, they do not report the downstream classification accuracy of Structured Row-wise Residual-EHPB, leaving its actual task performance unconfirmed.
