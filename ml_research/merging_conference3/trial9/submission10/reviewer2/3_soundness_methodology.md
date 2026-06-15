# Intermediate Evaluation 3: Soundness and Methodology

## 1. Clarity of the Description
The description of the core Dirichlet-PAC framework is structured and uses formal notation. However, the methodology is overly verbose and suffers from "mathiness"—unnecessary mathematical excursions that are not empirically validated or relevant to the core paper. For example, Section 5.1 (quantization via Wedin-Davis perturbation) and Section 5.3 (sequential streaming via Azuma-Hoeffding martingales) are highly speculative, lack any empirical validation in the experiments, and serve mainly to bloat the theoretical density of the paper.

## 2. Appropriateness of the Methods
The choice of a Dirichlet policy is theoretically appropriate for simplex-constrained ensembling, as it naturally models probability vectors. However, several other methodological choices are highly questionable under closer inspection.

## 3. Major Technical Flaws and Hidden Assumptions

### A. The "Unsupervised" SEP Fallacy
- The authors repeatedly claim that Subspace Energy Projection (SEP) is "completely unsupervised and task-agnostic."
- However, as described in Section 3.2, during the prior calibration phase, the representation matrix $Z_k \in \mathbb{R}^{N_{\text{prior}, k} \times D}$ must be constructed **for each task $k$** from the prior split $\mathcal{S}_{\text{prior}}$.
- To group these activations by task $k$, the system **must have access to the task labels (or task memberships) of the samples in the prior split.** If the split were truly unsupervised and unlabeled, it would be impossible to separate the activations into $Z_1, \dots, Z_K$ to perform SVD per task.
- Thus, the claim that SEP is completely unsupervised is a mischaracterization. It requires task-grouped calibration data, which is a form of weak supervision. If task labels are unavailable at serve-time, the entire basis extraction protocol collapses.

### B. Severe Theory-Practice Mismatch
- **Stochastic Routing vs. Continuous Blending:** The authors derive their PAC-Bayesian generalization certificates under the assumption of **Stochastic Expert Routing** (discrete query routing, where a query is routed to a single expert with probability $\alpha_k$). Under this assumption, the expected loss of the stochastic model is exactly equal to the linear surrogate loss $\sum_k \alpha_k p_k(x_b)$, making the PAC bound theoretically exact (Section 3.5).
- **The Gap:** However, in their actual experiments (both inside the Analytical Coordinate Sandbox and on the BERT models), they evaluate **continuous activation-space blending** (Equation 5), where activations of all parallel experts are scaled and summed. As the authors admit: *"the true blended classification probability of the network on sample $b$ is not exactly equal to the linear combination of isolated expert predictions."*
- **Consequence:** Since continuous activation-space blending is non-linear and executes multiple expert paths simultaneously, the linear surrogate loss is merely a loose proxy with no rigorous theoretical connection. The "mathematically rigorous, watertight" generalization guarantees derived in the theory **do not apply** to the physical continuous-blending models evaluated. This constitutes a severe gap between the paper's theoretical claims and its empirical reality.

### C. Missing Mathematical Definition of PEM-Div
- As identified in the novelty check, the "diversity penalty" or "batch-averaged ensembling weight entropy maximization" that defines the top-performing **PEM-Div** variant is never mathematically formulated in the Methodology.
- The reader is left to guess how this penalty is formulated, what hyperparameter scales it, and how it is optimized within the PAC-Bayesian bound.

## 4. Reproducibility
- Replicating the exact results is highly difficult. The synthetic "Analytical Coordinate Sandbox (ICS)" is a custom, non-standard simulation environment described only conceptually. Without the exact initialization seeds, exact synthetic data-generation equations, and the missing mathematical formulation of the PEM-Div loss, reproducing the baseline results in Table 1 and Table 2 is virtually impossible.
- The omission of code links or repository details further degrades reproducibility.
