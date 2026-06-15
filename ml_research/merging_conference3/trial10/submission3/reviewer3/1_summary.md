# Evaluation Phase 1: Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of **dynamic model ensembling of task-specific Parameter-Efficient Fine-Tuning (PEFT) adapters** (such as Low-Rank Adaptation, or LoRA) in sequential serving streams. The key objective is to manage the trade-off between **responsiveness** (instantaneous adaptation to task boundaries in a query stream) and **stability** (robustness to query-level activation noise) without incurring representational lag, parameter co-dominance, or high latency.

## Proposed Approach: Lotka-Volterra Competitive Serving (LVCS)
The authors propose **Lotka-Volterra Competitive Serving (LVCS)**, a biologically-grounded, non-linear stateful routing framework. The core idea is to treat the network layers as an ecosystem where specialized task experts behave like biological species competing for "resources" (measured by PCA coordinate projections of normalized early-layer activations).

The framework includes the following key design components:
1. **Discrete-Time Lotka-Volterra Ricker Recurrence:** Instead of linear decay-and-injection mechanics, the virtual populations (routing states) of the experts evolve layer-by-layer across network depth using a discrete-time Ricker competition recurrence.
2. **Guaranteed Mathematical Positivity:** The exponential form of the Ricker model ensures that population states remain strictly positive, avoiding ad-hoc clamping methods.
3. **Adaptive Niche Plasticity (Disturbance-Gated Competition):** A stream-homogeneity-gated mechanism that scales down inter-species competition coefficients during rapid sequential task switches (when temporal coordinate similarity drops), reducing representational lag and enabling the new expert to establish dominance rapidly.
4. **Systems-First Static Coordinate Approximation (LVCS Static):** Extracts PCA projection resource coordinates once at an early layer ($l_{\text{route}} = 3$) and drives the spatial recurrence statically across subsequent layers, avoiding the high latency of re-projecting at every layer.
5. **Parametric Constraints and Stability Analysis:** Implements learnable diagonal carrying capacities, bounded inter-species niche competition coefficients, and a mathematical projection operator to prevent chaotic period-doubling oscillations.

## Key Findings and Claims
- **Superior Accuracy on Overlapping Manifolds:** In a synthetic 14-layer Coordinates Sandbox evaluation across 5 seeds, LVCS (Static) achieves **89.08%** (homogeneous) and **90.06%** (heterogeneous) accuracy on overlapping task manifolds. This outperforms the linear stateful baseline PAC-Kinetics by up to **+1.34%** absolute.
- **Negligible Latency Overhead:** The static coordinate approximation reduces single-query latency by over **51%** compared to the dynamic variant (1.63 ms vs. 3.34 ms on CPU) while maintaining virtually identical accuracy (within 0.1%), making it highly systems-efficient.
- **Robustness to Real-World Settings:** On a real-world multi-task sequence classification stream using BERT-Tiny fine-tuned on GLUE tasks (SST-2, MRPC, CoLA), LVCS (Static) achieves **61.25%** downstream accuracy, outperforming stateless SABLE (60.25%) and linear PAC-Kinetics (60.25%) with $5\times$ to $16\times$ fewer parameters than MLP or GRU alternatives.
- **High Multi-Batch Scalability:** Vectorized CPU scaling benchmarks show throughput scaling up to **86,933 QPS** at batch size 1024, with the Ricker recurrence overhead collapsing from $51.88\%$ to $20.37\%$ as batch size scales, showing no serialization bottlenecks.

## Evidence Supporting Claims
- Extensive tabular results are provided comparing LVCS against 9 baselines (Oracle, Uniform, SABLE, ChemMerge, Momentum-Merge, PAC-Kinetics Vanilla/Augmented, Softmax Static, MLP Static, and GRU Router).
- Evaluations are conducted across 5 random seeds with mean and standard deviation reported for both Orthogonal and Overlapping manifolds.
- Sensitivity analysis is performed for the baseline competition floor parameter $\delta$ showing its stabilizing effects.
- Real-world validation is provided via PEFT adapters on BERT-Tiny over a 1200-sample sequence classification stream.
- Execution latency, parameters, and throughput scaling are benchmarked on CPU to substantiate the systems-level claims.
