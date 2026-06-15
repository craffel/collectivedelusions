# Novelty Check

## Key Novel Aspects
1. **Spatio-Temporal Bilinear Formulation for Expert Serving:** Unifying depth-wise (spatial) and sample-wise (temporal) ensembling weight smoothing into a single, direct, discrete-time 2D bilinear recursive filter. While spatial EMA (Momentum-Merge) and temporal state-space modeling (PAC-Kinetics) exist, 2D-STEM is the first to combine them into a simple, unified 2D discrete-time filter.
2. **Analytical Simplex-Preservation Proof:** Formulating and proving that a simple linear inequality constraint on the spatial and temporal momentum coefficients ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$) is sufficient to guarantee that the ensembling weights analytically remain on the probability simplex. This removes the need for active projection or re-normalization operations.
3. **Power-Law Gating (ATG-PL) for Upward Bias Resolution:** Recognizing that cosine similarity coordinates over non-negative probability vectors are strictly bounded below (creating a positive bias during transitions), and introducing a power-law exponent ($\gamma \ge 2$) to sharpen the transition response and collapse the temporal momentum to zero.

## Delta from Prior Work
- **SABLE (Stateless):** Evaluates layers and samples independently using nearest-centroid routing in raw feature space. 2D-STEM adds stateful propagation, reducing homogeneous routing jitter by up to $2.75\times$ on simulated manifolds and over $5.23\times$ on pre-trained ViT representation spaces.
- **Momentum-Merge (Spatial-only):** Uses depth-wise EMA but resets completely for each new sample. 2D-STEM propagates state across sequence samples, allowing it to smooth out temporal stream noise.
- **PAC-Kinetics (Temporal-only):** Propagates ensembling weights temporally using a learned first-order state-space model, but keeps weights constant across network depth. 2D-STEM performs local, layer-specific routing and state propagation across depth, preserving representational alignment.
- **ChemMerge (Spatio-Temporal):** Models routing trajectories as non-equilibrium biochemical kinetics integrated online via Euler discretization. 2D-STEM replaces continuous-time ODE solvers and five uninterpretable hand-tuned parameters with a single-line arithmetic 2D recurrence and a simple constraint.

## Characterization of Novelty
The novelty of this work is **moderate and highly practical**. The transition from continuous-time biochemical kinetics (ChemMerge) or complex learning-theoretic frameworks (PAC-Kinetics) to a direct, discrete-time 2D bilinear recursive filter represents an elegant application of Occam's razor. For a practitioner, this is a highly valuable simplification: it strips away unnecessary complexity and makes the system much easier to understand, implement, and compile for edge deployment. 

The introduction of Power-Law Gating (ATG-PL) is also a highly practical contribution. It directly addresses a geometric limitation of cosine similarity over non-negative coordinates, solving the transition latency issue without introducing any learnable parameters. However, the conceptual delta remains somewhat incremental, as it primarily combines and simplifies ideas from Momentum-Merge and first-order temporal filtering.
