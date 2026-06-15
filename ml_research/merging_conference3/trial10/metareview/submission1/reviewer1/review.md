# Peer Review: Markovian Path-Integral Ensembling (QPathMerge)

## 1. Summary of the Paper
This paper introduces **Markovian Path-Integral Ensembling (QPathMerge)**, a training-free serving controller designed to resolve the **accuracy-stability dilemma** encountered when serving Mixture-of-Experts (MoE) or dynamic adapter-merging systems on edge devices under rapid, heterogeneous task streams. 

To bypass the trade-off between the high-frequency spatial (layer-to-layer) oscillations of stateless routers (the **routing jitter paradox**) and the temporal lag (hysteresis) of stateful routers, the authors model network depth as a discrete 1D lattice. They formulate the routing trajectory as a discrete Euclidean path integral, mapping the problem to a 1D chain-structured **Markov Random Field (MRF)**. This allows them to execute the Forward-Backward sum-product algorithm (Belief Propagation) to compute mathematically exact, globally optimized marginal ensembling weights for each layer in $O(L K^2)$ linear time per sample. 

Because belief propagation is executed entirely within the depth lattice of a single forward pass, the model achieves profound spatial smoothing (acting as a spatial low-pass filter) while remaining absolutely stateless across sequential samples (eliminating temporal serving lag and hysteresis).

To make this formulation viable for low-power edge hardware, the authors introduce **QPathMerge-Single**, a single-pass candidate that recursively computes backward messages on-the-fly over a **Truncated Backward Horizon ($H \le 4$)** by speculatively assuming constant future potentials. They formally guarantee the exponential convergence of this truncation using **Dobrushin's contraction theorem**, and introduce linear extrapolation (\texttt{LinearExtrap}) to track non-monotonic representation trends over depth. 

Evaluation inside a high-fidelity 14-layer Coordinate Sandbox and physical validation on ResNet-18 using natural ImageNet-1K streams prove that QPathMerge slashes spatial layer-wise routing jitter by over $3\times$ compared to SABLE and ChemMerge, while completely avoiding representation hysteresis and maintaining top-tier classification accuracy.

---

## 2. Strengths of the Submission

### A. Exceptional Empirical Rigor and Statistical Soundness
The paper stands out for its outstanding empirical design and dedication to scientific validation. Rather than reporting single-run metrics, the authors execute multiple independent random seeds for all experimental configurations:
- **5 random seeds** with reported mean $\pm$ standard deviation for the high-fidelity Coordinate Sandbox evaluations.
- **3 random seeds** with mean $\pm$ standard deviation for the physical validation on pre-trained ResNet-18 models.
This provides a highly robust, statistically sound basis for comparing performance metrics.

### B. Comprehensive and Strong Baselines
The authors compare their method against a highly diverse, representative suite of **seven state-of-the-art baselines**, covering all major routing strategies:
- **SABLE-Static** and **SPS-ZCA-Static** (Static baseline copying anchor weights across depth).
- **SABLE-Dynamic** and **SPS-ZCA-Dynamic** (Stateless layer-by-layer dynamic routers).
- **SABLE-CausalFilter** and **SABLE-Gaussian** (Stateless spatial filtering baselines applying casual EMAs or post-hoc Gaussian smoothing).
- **Momentum-Merge**, **ChemMerge**, **PAC-Kinetics**, and **Stateful ERM** (Stateful kinetics and temporal smoothing baselines).
Crucially, the authors resolve previous comparison bugs by evaluating stateful models under a scientifically consistent protocol that preserves their historical state across sequential samples. This fair comparison directly reveals the severe temporal hysteresis and collapse of kinetics models under heterogeneous task switches.

### C. Exhaustive Ablation Studies
The empirical validation is thoroughly mapped out through multi-dimensional sweeps and ablations:
- **Truncated Horizon Sweep ($H$):** A systematic sweep of $H \in \{1, 2, 3, 4, 6, 8, 11\}$ empirically validates Dobrushin's convergence, proving that $H=4$ achieves near-identical spatial smoothness and accuracy to the full bidirectional model.
- **Extrapolation Ablations:** Evaluating standard constant potentials, Rolling average (\texttt{RollingExtrap}), and Linear trend (\texttt{LinearExtrap}) projections reveals the severe "spatial lag" of rolling averages under non-monotonic task switches and the superior tracking of linear projection.
- **Sensitivity Analyses:** Detailed sensitivity sweeps of the temperature parameter $\tau$, the calibration sample complexity (proving that 1 to 4 samples are highly sufficient due to cosine scale-invariance), and performance under clean vs. noisy inputs.

### D. Hardware-Level Validation
The authors bridge the gap between abstract machine learning and physical edge compiling by profiling computational overhead:
- Measuring FLOPs and empirical CPU latency (in microseconds) for experts registries up to $K=64$, proving that QPathMerge consumes less than 67.5k FLOPs and scales linearly.
- End-to-end model latency profiling showing that QPathMerge-Single adds only a minor **$1.35$ ms** ($5.35\%$) overhead on ResNet-18.
- Solid physical arguments showing how spatial smoothing actively prevents cache thrashing and energy-expensive DRAM memory transactions on NPUs.

---

## 3. Weaknesses of the Submission

### A. Scale of Physical Evaluation
While the physical validation on ResNet-18 is structurally isomorphic to PEFT adapter ensembling (e.g., dynamic LoRA ensembling) in Transformers, ResNet-18 is a lightweight 8-block CNN. Evaluating the QPathMerge-Single controller directly on a modern deep autoregressive Transformer backbone (such as LLaMA-3.2-3B or Mistral-7B) or a large-scale Vision Transformer (ViT) on the full ImageNet-1K validation set would further establish its generalizability and practical significance for production-scale generative models.

### B. Offline Centroid Dependency
Although the authors demonstrate that centroid calibration is extremely sample-efficient (requiring only 1 to 4 samples) and robust to distribution shifts due to cosine scale-invariance, the method remains dependent on an offline calibration prerequisite. Exploring completely training-free or online centroid adaptation would increase its versatility.

### C. Static Edge Potentials
The transition leakage matrix $\phi$ utilizes a static, globally set leakage parameter ($M = 0.10$), and the temperature $\tau = 0.5$ is manually tuned. Although the authors mathematically propose a layer-specific Scheduled Leakage $M_l$ and learned dynamic edge potentials, these are left as speculative extensions rather than being implemented and evaluated in the main results.

---

## 4. Evaluative Dimensions

### Soundness: Excellent
The mathematical formulation is elegant and mathematically sound. Mapping layer ensembling to a 1D chain MRF and solving for exact marginals using belief propagation is mathematically exact and highly appropriate. The proof of exponential convergence via Dobrushin's contraction theorem is rigorous, and the empirical setup—utilizing 5 seeds for synthetic streams and 3 seeds for natural ImageNet streams on pre-trained models—is designed with outstanding empirical integrity.

### Presentation: Excellent
The paper is exceptionally well-written, logically structured, and remarkably clear. Theoretical physical metaphors (path integrals, Boltzmann distribution) are grounded in classical probabilistic graphical model structures (MRFs, belief propagation). The authors are highly transparent about their design choices, mathematical power-iteration degeneracies, and the trade-offs of spatial smoothing. The inclusion of a self-contained PyTorch module in the appendix makes the paper immediately reproducible and accessible to practitioners.

### Significance: Excellent
Resolving the accuracy-stability dilemma is a critical bottleneck for deploying modular multi-task models (such as Mixture-of-Experts and PEFT registries) on low-power edge devices. By demonstrating that spatial trajectory smoothing actively reduces DRAM memory bandwidth consumption, the paper makes a highly significant contribution that spans ML theory, serving systems, and on-device hardware compiler optimization.

### Originality: Excellent
Formulating depth-wise ensembling as a discrete Euclidean path integral over a depth lattice and solving it exactly in linear time using sum-product message passing is highly novel. Bypassing temporal carryover states to achieve a zero-lag, state-free spatial low-pass filter is a creative paradigm shift that differs significantly from both stateless routers and stateful kinetics baselines.

---

## 5. Overall Recommendation

**Recommendation: 5 (Accept)**

**Justification:**
This is an exceptionally strong, technically flawless paper that makes a highly creative and original contribution to modular model serving on edge devices. By mapping network depth ensembling to a 1D chain Markov Random Field and solving it using exact belief propagation, the authors decouple spatial smoothing from temporal sample tracking. This successfully resolves the long-standing accuracy-stability trade-off under heterogeneous workloads, slashing spatial layer-wise routing jitter by over $3\times$ while completely bypassing temporal lag and hysteresis. 

The empirical evaluation is designed with outstanding scientific rigor, reporting mean and standard deviations across multiple random seeds, and evaluating against a highly comprehensive suite of seven baselines. The paper goes beyond theory, proving the exponential convergence of the truncated horizon using Dobrushin's contraction theorem, detailing a single-pass variant with linear extrapolation, profiling real CPU/NPU latencies, and providing a production-grade PyTorch implementation. The paper is highly recommended for acceptance.

---

## 6. Questions and Suggestions for the Authors

1. **Autoregressive Text Evaluation:** Have you considered evaluating QPathMerge on autoregressive sequence-generation tasks (such as translation or multi-task reasoning under heterogeneous query streams using pre-trained LoRA adapters)? It would be highly insightful to see if spatial trajectory smoothing stabilizes text generation stylistic coherence (e.g., preventing creative-logic expert swapping mid-generation) and if it reduces physical memory bandwidth overhead under autoregressive key-value cache constraints.
2. **Scheduled Leakage:** Can you provide a preliminary empirical validation of the proposed Scheduled Leakage schedule $M_l$ (e.g., early layers having small $M_l$ for flexibility, late layers having large $M_l$ for maximum semantic stability)? It would be highly valuable to see if this scheduled coupling shifts the accuracy-jitter Pareto frontier further upward.
3. **Centroid Adaptation:** How sensitive is the cosine similarity $S(h, \mu_k^{(l)})$ to extremely strong domain shifts (such as feeding sketch/cartoon images to a model calibrated on natural ImageNet images)? Would a simple online centroid running-average update break the stateless guarantee, or can it be managed purely inside local layers?
