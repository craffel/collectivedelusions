# 2. Novelty Check

## Key Novel Aspects and 'Delta' from Prior Work
The 'delta' between Dirichlet-PAC and prior work is substantial and represents a major conceptual shift:

1. **Simplex-Constrained vs. Unconstrained Parameter-Space Regularization:** 
   - *Prior Work (e.g., PAC-ZCA):* Attempted to regularize test-time routing parameters by optimizing Gaussian prior and posterior distributions over unconstrained log-temperature parameters. Because log-temperatures are unconstrained, this formulation is geometrically mismatched with ensembling coefficients (which must lie on the probability simplex $\Delta^{K-1}$). This misalignment leads to "log-temperature explosion" or sudden entropy collapse, numerical instability, and loose generalization bounds.
   - *Dirichlet-PAC:* Directly models the ensembling weight vector on the probability simplex using a Dirichlet distribution. By deriving and minimizing the exact analytical KL divergence between Dirichlet distributions over the simplex itself, it restricts the learned temperatures from exploding and acts as a natural, self-stabilizing routing-entropy regularizer.

2. **Unsupervised Subspace Energy Projection (SEP):**
   - *Prior Work (e.g., SABLE Raw Coords):* Relies on supervised, hand-tuned centroid similarities which require ground-truth task labels to group activations and locate task centroids. In real-world edge serving, such labels are rarely available.
   - *Dirichlet-PAC:* Formulates a completely unsupervised, task-agnostic feature coordinate extraction protocol using SVD on intermediate activations. In addition, Proposition 3.1 formally proves that this coordinate system is completely basis-independent (invariant to orthogonal changes of basis in the representation space) and scale-invariant (invariant to uniform scaling of activations), providing solid guarantees when deploying across different layers or model scales.

3. **Fully Unsupervised Prediction Entropy Minimization (PEM-Div):**
   - *Prior Work:* Relies heavily on weakly/few-shot supervised calibration datasets at test time.
   - *Dirichlet-PAC:* Introduces a fully unsupervised serving pathway that optimizes for maximum prediction confidence (minimum entropy) and batch-wide ensembling diversity under the PAC-Bayesian complexity penalty. This completely removes the dependency on labeled calibration data while achieving superior generalization accuracy.

4. **First-Principles Derivation of Representation Interference:**
   - *Prior Work:* Modeled representation interference using heuristic formulations.
   - *Dirichlet-PAC:* Formally proves from first principles that representation clashing/interference noise scales directly with the ensembling entropy (via Gini Impurity and the Simpson/Herfindahl concentration index), beautifully grounding their noise model in physical activation clashing.

## Characterization of Novelty
The novelty of this paper is **significant and highly original**. 

Instead of presenting an incremental or marginal modification to existing heuristic model merging/serving methods, Dirichlet-PAC introduces a paradigm-shifting learning-theoretic perspective to test-time model serving. The authors' conceptual leap—reformulating dynamic ensembling as a simplex-constrained statistical learning problem and deriving the exact analytical Dirichlet KL divergence within an input-dependent PAC-Bayesian bound—is a profound contribution. It successfully bridges theoretical learning theory with practical multi-task edge serving, showcasing that mathematical rigor can lead directly to stabilized and robust engineering solutions.
