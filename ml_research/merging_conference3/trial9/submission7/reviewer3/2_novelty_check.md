# 2. Novelty Check

## Key Novel Aspects
1. **Closed-Loop Active Feedback Control:** The primary novelty lies in formulating active representation coupling as a closed-loop control problem. Instead of relying on empirical constant feedback rates ($\eta$) or simple heuristic decay functions (e.g., Decay-ChemMerge), L-ARC derives the feedback rate $\eta^{(l)}$ dynamically at each layer. This is achieved by modeling representation similarity error as a candidate Lyapunov function $V$ and ensuring that each warping step is strictly dissipative (error-decreasing).
2. **Entropy-Gated Concentration Gating (ECG-Reset):** This mechanism represents a novel state-space shielding technique that freezes physical kinetics updates ($\Delta t^{(l)} = 0$) during high-entropy routing failures. It acts as a sample-and-hold circuit that bridges transient sensor/network dropouts without memory corruption.
3. **Representation-Agreement State Correction (RASC):** This is a novel dual-loop control mechanism designed to combat confident but systematically biased routing. By checking for agreement between feedforward predictions ($k_{\text{pred}}^{(l)}$) and unbiased feedback representation similarity tracking ($s_{\text{pred}}^{(l)}$), RASC dynamically overrides the kinetics rate to steer the state tracker away from systematic corruption.
4. **Entropy-Triggered Gating (ET-L-ARC):** This is a novel control-theoretic optimization that dynamically gates the evaluation of the Dissipation Guard itself. Under highly confident routing ($H^{(l)} < 0.15$) or failed routing ($H^{(l)} > 0.95$), the guard calculations are bypassed, which drastically reduces latency overhead under clean serving.

---

## The "Delta" From Prior Work
The paper explicitly contextualizes its contributions relative to two main state-of-the-art baselines:
* **Delta from SABLE (Stateless serving):** SABLE routes inputs on-the-fly but re-computes ensembling weights independently at each layer, ignoring representational depth-wise dependencies. This results in severe high-frequency routing weight fluctuations ("jitter") and a collapse in representation-space semantic similarity under realistic memory-constrained serving (Setting A). L-ARC introduces stateful continuous-time kinetics that act as a spatial low-pass filter to smooth trajectories, and utilizes a closed-loop Dissipation Guard to prevent the semantic corruption that SABLE suffers from.
* **Delta from ChemMerge (Stateful open-loop serving):** ChemMerge was the first to use physical reaction kinetics and continuous-depth ODEs to smooth routing trajectories. However, ChemMerge's active representation coupling is open-loop and heuristic. Setting a constant feedback step size ($\eta > 0$) causes a representational backward-shift, making the active feedback loop unusable under steady-state homogeneous or heterogeneous workloads. L-ARC resolves this fundamental limitation by introducing a closed-loop Lyapunov feedback controller that dynamically scales the coupling rate on-the-fly based on local dissipation, rendering active coupling highly effective and mathematically stable.

---

## Characterization of Novelty
The novelty of this work is characterized as **significant and conceptually robust**. 
* Rather than providing incremental parameter tweaks or minor empirical adjustments to existing serving architectures, L-ARC represents a fundamental paradigm shift: **from open-loop heuristic feedback to closed-loop control-theoretic stability**. 
* The mathematical rigor introduced (four theorems covering approximation bounds, linearization errors, unconditional non-negativity, and online centroid adaptation) is highly sophisticated and elevates the work far above typical empirical-heavy PEFT serving papers. 
* The dual-loop formulation of RASC, where representation-space tracking actively corrects kinetics-space tracking, is a beautiful and highly original conceptual solution to the persistent state-locking problem under systematic router bias.
* The paper's scientific transparency is refreshing: by explicitly identifying that feedback warping is statistically redundant under clean workloads ($p = 0.0969$) and advising edge practitioners to run decoupled kinetics under pristine serving, the authors build credibility and focus their novel control mechanisms strictly where they are mathematically and practically necessary (fault-tolerant and biased workloads).
