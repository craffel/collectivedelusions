# 1. Summary of the Paper

## Topic and Scope
This paper addresses the challenge of **high-conflict parameter-space model merging** under highly constrained capacity regimes (specifically utilizing compact backbones like a 5.7M-parameter Vision Transformer, `vit_tiny_patch16_224`). Traditional model merging methods (e.g., Task Arithmetic, TIES-Merging, AdaMerging) produce a single static, input-independent set of weights, which leads to **catastrophic representational collapse** when combining experts trained on highly diverse and conflicting visual distributions.

## Proposed Approach: QWS-Merge
To resolve this limitation, the paper proposes **Quantum Wavefunction Superposition Merging (QWS-Merge)**, a dynamic, input-dependent merging framework inspired by quantum mechanics:
1. **Task Eigenstates:** Fine-tuned expert weights are modeled as task eigenstates $|\psi_k\rangle$ in a parameter Hilbert space.
2. **Input Phase State Extraction:** A global representation $z(x)_b$ is extracted from incoming batches via spatial average pooling of patch tokens, projected into a low-dimensional $d$-dimensional phase-state space via a frozen random projection matrix $P$, and normalized to the unit sphere to form the input phase state $\psi(x)_b$.
3. **Quantum Phase-Coherent Overlap:** A set of learned layer-wise phase basis vectors $\hat{\Phi}_k^{(l)}$, scaling amplitudes $R_k^{(l)}$, and phase biases $\phi_k^{(l)}$ are used to compute sample-level probability amplitudes (routing coefficients) via a wave-like cosine formulation:
   $$ \alpha_{k, b}(l) = R_k^{(l)} \cos\left( \omega \langle \psi(x)_b, \hat{\Phi}_k^{(l)} \rangle + \phi_k^{(l)} \right) $$
4. **Wavefunction Collapse:** To perform computationally efficient batch-wise inference, sample-level coefficients are averaged across the batch to project (collapse) the state into a single classical weight configuration:
   $$ \bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l) $$
5. **Parameter Assembly:** The final dynamically assembled weights for layer $l$ are:
   $$ W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k(l) V_k^{(l)} $$

## Key Findings & Empirical Evidence
- **Catastrophic Representational Collapse in Static Merging:** Static uniform merging degrades joint mean multi-task performance to $49.35\%$ compared to the specialized expert ceiling of $70.52\%$, particularly collapsing on highly conflicting datasets like SVHN ($21.90\%$ vs $34.50\%$) and FashionMNIST ($44.50\%$ vs $77.70\%$).
- **Success of QWS-Merge:** QWS-Merge achieves a joint multi-task average accuracy of $59.32\%$ on homogeneous test streams, completely bypassing representational collapse.
- **Wave-Like Subspace Regularization:** While a classical unconstrained *Linear Router* baseline achieves a higher overall joint mean accuracy of $61.23\%$ by over-fitting on simple tasks, it catastrophically collapses to $15.30\%$ on SVHN. Conversely, QWS-Merge maintains a high performance of **31.60\%** on SVHN ($91.5\%$ of the expert ceiling), demonstrating that the cosine-based bounded subspace acts as an exceptional regularizer.
- **Heterogeneity Collapse:** The paper systematically evaluates dynamic merging under mixed-task heterogeneous streams. At larger batch sizes ($B=256$), task-mixing causes the collapsed coefficients to average out back toward static compromises, degrading performance to $48.70\%$ for QWS-Merge. QWS-Merge shows stronger resilience to this collapse than the Linear Router ($47.70\%$).
- **Resource Efficiency:** The method optimizes only 336 parameters on a tiny validation set of 16 samples per task (64 total) for 100 steps in under 30 seconds.

## Explicitly Claimed Contributions
1. **A Visionary Paradigm Shift:** Challenging the static-parameter assumption via a quantum-inspired formulation where experts are superposed and collapsed dynamically via input phase-coherence.
2. **High-Conflict Model Merging Solution:** Complete resolution of catastrophic representational collapse on compact backbones in high-conflict scenarios.
3. **Wave-Like Subspace Regularization:** Proof that the non-monotonic cosine phase projections provide robust regularization under extreme task conflicts (SVHN), outperforming a classical soft-routing baseline by $+16.30\%$.
4. **Transparent Heterogeneity Benchmark:** First systematic investigation of batch size and task heterogeneity on dynamic merging, exposing "heterogeneity collapse".
5. **Extreme Resource Efficiency:** Optimizing only 336 parameters on a tiny 64-sample offline validation set, bypassing the Overfitting-Optimizer Paradox.
