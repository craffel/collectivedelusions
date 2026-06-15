# 4. Experimental Setup, Baselines, and Results

## Evaluation of Experimental Setup
The experimental evaluation is exceptionally thorough and well-designed:
- **High-Fidelity Simulation:** The 14-layer, 192-dimensional coordinate simulation in PyTorch (Analytical Coordinate Sandbox) replicates realistic sequential, non-stationary serving workloads under non-i.i.d. task distributions and high-frequency activation fluctuations.
- **Robust Stream Configurations:** Evaluating both homogeneous (long blocks of tasks under high noise) and heterogeneous (rapid step-by-step task transitions) streams ensures that the models are stress-tested for both noise filtering and transition responsiveness.
- **Geometric Manifolds:** Testing both orthogonal and overlapping geometric task manifolds provides deep insights into the models' ability to handle ambiguous or shared task boundaries.
- **Robustness to Model Mismatch:** The high-dimensional, non-linear, heavy-tailed noise stress test in Section 4.3 directly addresses the "simulation-to-physical gap," demonstrating how the linear approximation behaves under non-linear, non-Gaussian conditions.

---

## Baselines
The paper compares AIR against a highly representative and comprehensive set of five baselines:
1. **Expert Oracle:** An omniscient router representing the theoretical upper bound.
2. **Uniform Merging:** A static baseline representing no adaptation.
3. **SABLE (Stateless):** A competitive modern nearest-centroid soft routing framework representing the stateless extreme.
4. **Momentum-Merge & ChemMerge (Stateful):** Classic low-pass exponential moving averages and biochemical ODE kinetics representing the stateful, rigid extreme.
5. **PAC-Kinetics (Recurrent/Optimized):** A sophisticated recurrent, optimization-based stateful router representing the complex adaptive state-of-the-art.

---

## Do the Results Support the Claims?
Yes, the empirical results fully support the authors' claims:
- **Resolution of Jitter-Lag Dilemma:** In Table 1, AIR simultaneously slashes SABLE's routing jitter by up to **2.49$\times$** under homogeneous streams (reducing jitter from 0.0860 to 0.0364) while matching its optimal alignment accuracy ($66.44\%$). Under rapid heterogeneous switches, AIR adapts near-instantaneously (1--2 steps), matching SABLE and the Oracle's accuracy ($66.23\%$) and outperforming PAC-Kinetics, while stateful ChemMerge and Momentum-Merge collapse due to severe lag.
- **Mechanistic Role of Active Inhibition:** The ablation study (Section 4.5 & Appendix P) clearly shows that restricting $\mathbf{W} \ge 0$ (no inhibitory pathways) results in a localized 15-step transient lag during task transitions under homogeneous streams, proving that negative feedback coupling is essential to suppress obsolete task beliefs.
- **Calibration Generalization & Scaling:** Section 4.6 (Appendix M & N) confirms that the learned precisions generalize perfectly across different sequence types (homogeneous vs. heterogeneous) and scale seamlessly to $K=16$ active experts.

---

## Experimental Critique (Minimalist Perspective)
1. **The Core Success is a Victory for Simplicity:** From a minimalist perspective, the outstanding performance of AIR is a major victory for simple, classical methods. Because the proposed closed-form solver is mathematically equivalent to a classical **linear state observer (Kalman filter)**, the experimental results are empirical proof that a simple, elegant classical control loop is vastly superior to both stateless feedforward heuristics and highly engineered, recurrent networks like PAC-Kinetics. The authors should have framed the results as celebrating the power and efficiency of a classic Kalman filter in neural networks, rather than attributing the success to a complex neuroscience narrative.
2. **The Triumph of AIR (Diagonal):** In the scaling experiments (Section 4.6 & Appendix M), the authors show that **AIR (Diagonal)**—which restricts $\mathbf{W}$ to a diagonal matrix and has only $5K$ parameters—performs exceptionally well. With a tiny calibration length of $T_{\text{cal}} = 32$, this extremely simple model achieves excellent accuracy and stability, completely bypassing the quadratic parameter scaling and potential overfitting risks of the dense model. From a minimalist standpoint, this diagonal model is the true highlight of the paper. It achieves maximum efficiency and simplicity, representing the ideal engineering solution. The authors should have featured this minimalist model prominently in the main text rather than burying it in the appendix.
3. **Speculative Speculations Contrast with Core Empirical Focus:** The experimental section is highly focused and empirical, yet the paper introduces speculative, highly complex, and completely un-evaluated extensions (e.g., Laplace approximations, Contractive Autoencoders for non-linear projections) in the appendix. From a minimalist perspective, these speculative extensions represent unnecessary theoretical over-engineering that distracts from the highly successful, simple linear-Gaussian core method.
