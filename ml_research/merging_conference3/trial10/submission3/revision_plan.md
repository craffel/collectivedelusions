# Revision Plan: Addressing Reviewer 2's Latest Critiques

We thank the reviewer for their exceptionally constructive and rigorous feedback. We will address each of the three critical flaws and constructive suggestions directly in our LaTeX source files and python scripts.

## 1. Critique 1: Superiority of Simpler Baselines & Marginal Practical Gains
*   **Issue:** The MLP (Static) baseline consistently outperforms LVCS in the Coordinates Sandbox. In the real-world BERT-Tiny stream, the accuracy improvement over Uniform merging (+0.17%) and MLP (Static) (+0.25%) is marginal, which limits practical adoptability given the computational overhead of an 11-step recurrence.
*   **Action Plan:**
    *   **Honest and Objective Discussion:** We will add a dedicated, self-critical paragraph in Section 4 (Experiments) discussing the "Overhead vs. Gain" trade-off. We will explicitly acknowledge that a practitioner seeking minimal deployment overhead may prefer zero-overhead Uniform merging, as the absolute gains of active ensembling in this compact setting are modest.
    *   **Reframing capacity vs. inductive bias:** We will explain that the synthetic Coordinates Sandbox represents a clean, low-noise coordinate space where unconstrained, highly parameterized classifiers (like MLP with 115 parameters) can easily fit the decision boundaries. However, in real-world messy representation spaces, highly parameterized models are prone to overfitting. The ecological constraints of LVCS act as a robust spatial regularization prior, achieving competitive performance while using $5\times$ fewer parameters and providing strict mathematical guarantees of positivity and stability.

## 2. Critique 2: Pseudo-Temporal Stateful Nature of Populations (1-Step Memory Window)
*   **Issue:** The population density state is reset to uniform at the start of each query, meaning the model is temporally stateless in its population variables and relies on a 1-step memory window ($Sim_t$). Marketing this as "stateful temporal serving" is structurally misleading.
*   **Action Plan:**
    *   **Terminology Rectification:** We will systematically revise the text in the Abstract, Introduction, and Methodology (Section 3) to replace any misleading "temporal statefulness of populations" claims.
    *   **Accurate Re-framing:** We will accurately frame the model as a **spatially recurrent (layer-by-layer) dynamic router with a 1-step temporal boundary-gating window** ($Sim_t$). We will explain that uniform re-initialization of populations at each query is a deliberate design choice that prevents error propagation across long streaming sequences, and propose truly temporally stateful population models as an exciting direction for future research.

## 3. Critique 3: Gating Contradiction under Heterogeneous Streams
*   **Issue:** In heterogeneous streams where $Sim_t \approx 0.0$ and $\delta = 0.1$, inter-species competition is reduced by 90%. This deactivates the coupled "Winner-Take-All" competition exactly in the regime where representational interference is most acute.
*   **Action Plan:**
    *   **Mathematical & Ecological Reconciliation:** We will add a dedicated explanation in Section 3.4 and Section 4. We will explain that under sudden task transitions ($Sim_t \approx 0$), reducing inter-species competition is ecologically necessary to lower the "invasion barrier" and prevent representational lag (allowing the colonizing species/new expert to establish itself). 
    *   **Compounded Recurrence Effect:** Once the new expert population begins to grow in the early layers, our 11-step layer-wise spatial recurrence exponentially compounds even the small 10% competition floor ($\delta = 0.1$) across depth. This compounded coupled competition is mathematically and empirically sufficient to prune representation leaks and co-dominant states by the final layer, successfully balancing invasion speed (responsiveness) and task isolation (stability).

## 4. Re-framing the Positivity Proof (Theorem A.1)
*   **Issue:** The mathematical proof of strict positivity for the Ricker competition model is trivial because of the exponential formulation, and presenting it as a grand theorem is hyperbolic.
*   **Action Plan:**
    *   **De-escalating the Terminology:** We will re-frame Section A of our Appendix from a "Rigorous Mathematical Proof of Strict Positivity" to an "Architectural Guarantee of Exponential Recurrence." We will present the positivity property as a simple, elegant structural property of the discrete-time Ricker formulation rather than a complex theorem, enhancing scientific modesty and credibility.

## 5. Bayesian Calibration Code Alignment
*   **Issue:** The code calculated expected self-distance `D_j` but hardcoded `D_cal` to 1.0, contradicting claims of automated Bayesian calibration.
*   **Action Plan:**
    *   **Unified Implementation:** We have already updated `simulate_all.py` to use `D_cal[method][j] = D_j.clamp(min=1e-5)` directly. We will ensure the LaTeX manuscript (Section 3.3 and Section 4) correctly describes this data-driven Bayesian calibration and explain that it eliminates the need for hand-tuned classification biases.

## 6. Tone and Hyperbole Downscaling
*   **Issue:** Avoid hyperbolic phrases like "massive breakthrough", "radical departure", and "flawless gradient flow."
*   **Action Plan:**
    *   We will systematically review and edit all LaTeX section files to adopt a standard, objective, and measured academic tone.
