# Intermediate Evaluation 2: Novelty and Delta Check

## 1. Key Novel Aspects
The primary novelty of this paper lies in formulating a **simplex-constrained PAC-Bayesian complexity penalty** by modeling the routing ensembling weights directly as Dirichlet-distributed variables on the probability simplex $\Delta^{K-1}$. 
- Rather than optimizing unconstrained log-temperatures with Gaussian priors and posteriors (as done in PAC-ZCA), this work derives and minimizes the exact analytical KL divergence between Dirichlet distributions over the simplex itself.
- To drive these routing coefficients without relying on hand-tuned centroids or ground-truth labels, the paper introduces **Subspace Energy Projection (SEP)**, which extracts task coordinates via unsupervised SVD on early-layer activations.

## 2. 'Delta' from Prior Work
- **From PAC-ZCA (paczca2026):** PAC-ZCA applies PAC-Bayesian bounds to log-temperatures using Gaussian priors and posteriors. The "delta" is that Dirichlet-PAC operates directly on the probability simplex, resolving the "log-temperature explosion" and theoretical mismatch associated with Gaussian modeling of constrained spaces.
- **From SABLE (sable2025):** SABLE maps intermediate activations to ensembling weights using early-layer centroid distances with static, hand-tuned temperatures. The "delta" is that Dirichlet-PAC extracts coordinates using unsupervised SVD (SEP) instead of supervised centroids, and dynamically calibrates task-specific temperatures using a learning-theoretic bound instead of relying on static, hand-tuned scales.
- **From Weight-Space Merging (TIES, DARE):** While weight-space merging permanently averages or prunes weights, Dirichlet-PAC performs dynamic activation-space blending, avoiding parameter-clashing and representation collapse in heterogeneous task environments.

## 3. Characterization of Novelty
The novelty is characterized as **Incremental to Moderate**. 
- While the application of Dirichlet distributions and their analytical KL divergence to a PAC-Bayesian framework for test-time ensembling is mathematically sound and elegant, both the Dirichlet distribution properties and PAC-Bayes theorem are well-established. The integration is a natural progression of applying information-theoretic bounds to neural networks.
- SVD-based subspace projection is a standard dimensionality reduction technique in representation learning, though its deployment as a "scale-invariant task coordinate system" for test-time routing is a solid incremental step.

## 4. Critical Gaps and Unaddressed Similarities

### Conceptual Overlap and Standard Tools
- The mathematical formulation heavily relies on standard statistical identities (Dirichlet KL, SVD). The paper presents these with high narrative density (e.g., formalizing basic linear algebra as "Proposition 1") to maximize the perceived novelty.
- SVD projection coordinates have long been used for task identification and domain adaptation. The conceptual novelty of "discovering" that SVD is scale-invariant and basis-independent is overstated, as these are fundamental properties of orthogonal projections and SVD.

### The Undefined "PEM-Div" Router
- A major concern is that **PEM-Div**, which is presented as a state-of-the-art unsupervised baseline that achieves the highest accuracy in Table 1 (79.43% orthogonal, 78.73% overlapping) and performs exceptionally well on BERT models, is **never mathematically defined** in the Methodology section (Section 3).
- Section 3.5 only defines standard "Prediction Entropy Minimization (PEM)" in Equation 13. The "batch-averaged ensembling weight entropy maximization to preserve routing diversity" (which differentiates PEM-Div from PEM) is mentioned only conceptually in Section 4.3 and Section 4.5.3.
- Leaving a core, top-performing proposed variant mathematically undefined is a major gap that undermines the novelty claims of the unsupervised contribution and severely damages reproducibility.

### Lack of Simple Baseline Comparisons
- The authors justify the heavy PAC-Bayesian mathematical machinery by claiming it is necessary to prevent "temperature collapse" and "log-temperature explosion."
- However, they do not compare their method against standard, simpler engineering regularizers—such as L2 or L1 weight decay directly on the log-temperature parameters $\mathbf{w}$. If a simple L2 penalty on log-temperatures can stabilize optimization and prevent collapse similarly to the complex Dirichlet KL penalty, the practical necessity of the PAC-Bayesian formulation is highly questionable. The omission of this obvious baseline makes the claim of "necessity" unearned.
