# Experimental Validation and Evaluation Check

## 1. Experimental Setup and Sandbox Environment
The evaluation is conducted inside the **Analytical Coordinate Sandbox (ICS)**, a 14-layer high-fidelity simulator of representational flow and expert ensembling:
- **Sandbox Appropriateness:** Testing on the ICS is highly appropriate for this type of research. It provides a controlled, mathematically clean environment to isolate the core representation alignment and stateful routing dynamics. By avoiding the massive computational overhead and confounding variables of full-scale LLM training/inference (e.g., KV cache management, continuous batching scheduling, token decoding strategies, vocabulary-level distortions), the authors can perform a precise, surgical scientific study of state contamination and temporal smoothing.
- **Manifold Geometries:** Evaluating across both **Orthogonal Manifolds** ($overlap=0$) and **Overlapping Manifolds** ($overlap=12$) is a major strength. It simulates varying degrees of task-space representation interference, which is a major challenge in real-world model merging.

---

## 2. Baseline Choices and Comparative Rigor
The paper compares TDSR against an exceptionally strong and comprehensive set of baselines:
1. **Static Uniform Merging:** Serves as a standard non-dynamic control.
2. **Stateless SABLE (Raw):** The primary baseline for stateless dynamic ensembling.
3. **Global PAC-Kinetics:** The state-of-the-art global stateful ensembling baseline, which is essential to demonstrate the severity of the state contamination bottleneck.
4. **Oracle / Isolated Clean-Stream Baseline:** Represents the theoretical upper bound where tenant streams are completely isolated and processed sequentially.

The choice of these baselines allows for a rigorous and complete mapping of the performance landscape, highlighting exactly what is lost under state contamination and what is recovered by TDSR.

---

## 3. Analysis of Quantitative Findings
The empirical results are highly compelling, statistically rigorous, and thoroughly analyzed across **5 independent random seeds**:
- **Resolving State Contamination (Orthogonal Manifolds):** standard Global PAC-Kinetics suffers from cross-talk, achieving only **68.70% ± 2.83%** accuracy (a drop of 3.90% compared to the Oracle ceiling). Our proposed **TDSR (Explicit, Local)** achieves **70.60% ± 2.81%** accuracy, representing a significant **+1.90%** absolute accuracy improvement over the contaminated Global baseline and closing the gap to the Oracle ceiling (**72.60% ± 5.22%**).
- **Overlapping Manifold Evaluation:** Under Overlapping Manifolds ($overlap=12$), **TDSR (Explicit, Local)** achieves **70.85% ± 3.02%** classification accuracy, completely outperforming stateless SABLE (**65.15% ± 3.36%**) and standard Global stateful routing (**69.10% ± 3.10%**), performing within 0.50% of the clean-stream ceiling (**71.35% ± 6.10%**).
- **The Intra-Session Jitter Breakthrough:** Slashing intra-session routing jitter by up to **2.4$\times$** compared to stateless SABLE (orthogonal: 0.552111 ± 0.035050 down to 0.232446 ± 0.014236; overlapping: 0.533918 ± 0.028162 down to 0.220341 ± 0.012543) demonstrates that TDSR successfully isolates the low-pass temporal filtering dynamics within virtual slots, matching the routing stability of the Oracle clean ceiling.

---

## 4. Gaps, Weaknesses, and Areas for Improvement
Despite the rigor of the evaluation, there are a few experimental limitations that could be addressed to strengthen the paper:
- **Lack of Real-World LLM Evaluation:** While the ICS sandbox is mathematically elegant and highly informative, the entire evaluation remains in simulation. Validating the framework on actual LLM routing streams (e.g., serving 4 LoRAs on a base LLaMA-3-8B model using Punica or S-LoRA on standard tasks like GSM8K, Alpaca, and HumanEval) would dramatically elevate the paper's empirical weight and appeal to a broader systems audience.
- **Scaling of the State Pool:** The experiments are currently restricted to $M=4$ tenants and $K=4$ tasks. Evaluating how the system behaves as $M$ scales to larger numbers (e.g., $M=16, 64, 100$) under varying degrees of interleaving sparsity would demonstrate the practical scalability of the Tenant-Specific Session-Step Decay policy under extremely sparse workloads.
- **Logical Inconsistency on Inactive Slot Decay:** The methodology states that inactive slots hold their state perfectly between active queries ($\Delta t_m = 0$) during the evaluated interleaved stream. While Section 3.6 presents a "Dual-Clock Decay" policy to resolve memory leaks in production, this dual-clock mechanism is not quantitatively evaluated. The tension between evaluated constant retention ($\Delta t_m=0$) and praised automatic passive decay should be explicitly clarified.
