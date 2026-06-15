# Comprehensive Summary of the Submission

## 1. Main Topic and Scope
The submission introduces **Lotka-Volterra Competitive Serving (LVCS)**, a stateful routing and ensembling paradigm for Parameter-Efficient Fine-Tuning (PEFT), specifically targeting Low-Rank Adaptation (LoRA) multi-expert serving environments. The focus is on sequential query streams where queries arrive from diverse and shifting task distributions. The paper addresses the fundamental trade-off between **responsiveness** (the speed of adapting to new tasks across stream boundaries) and **stability** (the ability to resist query-level activation noise and maintain coherent representation trajectories across network layers).

## 2. Proposed Approach and Methodology
LVCS models the layer-wise activation trajectories of task-specific experts as virtual population densities of competing biological species. Instead of using traditional linear state-space models (e.g., ChemMerge, Momentum-Merge, PAC-Kinetics), LVCS employs a non-linear, discrete-time ecological model. Its core components are:
- **Discrete-Time Lotka-Volterra Ricker Recurrence:** Governs the evolution of expert population densities $x_{k, t}^{(l)}$ across the depth of the network (layers $l_{\text{route}}+1$ to $L$).
- **Parametric Ecological Constraints:** We model self-limitation (diagonal carrying capacities $c_{kk} \ge 0.1$) and inter-species niche competition (off-diagonal competition coefficients $c_{kj} \in [0, 1]$ via a sigmoid function).
- **Adaptive Niche Plasticity:** Gated by the cosine similarity of resource coordinates between consecutive steps ($Sim_t$), this mechanism dynamically scales inter-species competition coefficients during sudden task transitions to eliminate "representational lag" (phase delay).
- **Systems-First Static Coordinate Approximation:** Extracts PCA coordinate projections at a single early layer ($l_{\text{route}} = 3$) rather than re-computing them dynamically at every subsequent layer.
- **Dynamical Stability Projection Operator:** Imposes bounds to guarantee that growth rates $r_{k, t}$ do not exceed the chaotic bifurcation threshold of $2.0$ (May's chaos).
- **Simplex Mapping:** Normalizes final population densities to yield expert blending weights $\alpha_{k, t}^{(l)} \in \Delta^{K-1}$.

## 3. Explicitly Claimed Contributions and Provided Evidence
The authors claim the following contributions:
1. **Introduction of LVCS:** A non-linear stateful router for PEFT serving, bridging mathematical ecology and representation learning.
2. **Formulation of Ricker Recurrence:** Proving that it guarantees strict population positivity and natural self-regulation, eliminating ad-hoc clamping hacks.
3. **Adaptive Niche Plasticity:** Eliminating phase delay under rapid task transitions by dynamically gating competition based on stream temporal homogeneity.
4. **Systems-First Static Coordinate Approximation:** Reducing serving latency by over 51% compared to a fully dynamic model while maintaining virtually identical ensembling accuracy in the synthetic sandbox.
5. **Introduction of Recurrent Baselines:** Formulating a spatially recurrent GRU Router to deconstruct trade-offs in dynamic ensembling.
6. **Empirical Evaluation on the Coordinates Sandbox:** Demonstration on Orthogonal and Overlapping manifold structures across 5 random seeds. The authors report that on overlapping manifolds, LVCS outperforms PAC-Kinetics by $+1.38\%$ absolute accuracy on homogeneous streams and $+1.34\%$ on heterogeneous streams, while maintaining stable spatial trajectories.
7. **Real-World Generalization on BERT-Tiny:** Evaluation of multi-task sequence classification (SST-2, MRPC, CoLA) using Hugging Face PEFT on a heterogeneous stream. The authors claim LVCS (Static) achieves $61.25\%$ downstream sequence accuracy, outperforming stateless SABLE ($60.25\%$) and PAC-Kinetics ($60.25\%$) while using up to $16\times$ fewer parameters than a GRU or MLP.

## 4. Summary of Key Findings
- **In the Coordinates Sandbox (Synthetic):** LVCS achieves superior performance on both orthogonal and overlapping task manifolds, outperforming traditional linear state-space models and SABLE.
- **In Systems Analysis:** Static coordinate extraction is shown to be highly efficient, reducing single-query latency on CPU compared to the dynamic variant (1.63 ms vs. 3.34 ms). Vectorized batch serving throughput scales from 703 to 86,933 QPS as batch size scales to 1024.
- **In BERT-Tiny (Real-World):** The authors report a downstream sequence accuracy of $61.25\%$ for LVCS (Static) on a heterogeneous stream of GLUE tasks, claiming a +1.0% improvement over SABLE and PAC-Kinetics.
