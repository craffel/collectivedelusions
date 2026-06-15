# 5. Impact and Presentation

## Major Strengths
1. **Mathematical Rigor:** The paper provides a highly detailed mathematical formulation. The projection of acceleration and velocity onto tangent planes, the use of the exact spherical Exponential Map, and the closed-form parallel transport of velocity vectors are technically sound and demonstrate high geometric rigor.
2. **Comprehensive Appendix and Ablations:** The authors provide thorough ablations on key hyperparameters ($G$, $\epsilon$, $\gamma_{\text{drag}}$) and explore various conceptual extensions, such as "Coupled GraviMerge" (closed-loop feedback) and "Temporal State Carryover" for sequential streams.
3. **Clear Visualization:** The qualitative plots in Figure 2 are clean and effectively illustrate the layer-wise weight trajectories and the performance-stability trade-off.

## Areas for Improvement

### 1. Extreme Over-Engineering and Complexity
The entire framework is a classic example of excessive complexity. Introducing orbital mechanics, viscous drag, softened gravitational forces, spherical geodesic integration, and parallel transport simply to smooth out layer-wise routing weights represents massive over-engineering. The authors should strive for minimalist design principles, prioritizing simplicity and ease of deployment over metaphorical complexity.

### 2. Lack of Realistic, Large-Scale Evaluation
Evaluating a framework designed for deep model serving on scikit-learn's `load_digits` dataset (8x8 grayscale images) projected to 192 dimensions is a major weakness. The paper completely lacks real-world evaluation on modern, large-scale pre-trained models (e.g., LLMs like LLaMA/Mistral, or Vision Transformers) performing actual downstream tasks (e.g., MMLU, GSM8k, GLUE). Without these, the practical utility of GraviMerge remains completely unproven.

### 3. Clear Formulation of the "Decoupling Illusion"
The authors should be transparent about the fact that, in default "Decoupled" mode, the spacecraft's trajectory does not actually adapt to the intermediate representations propagating through deep layers. Since the spacecraft's motion depends only on the early-layer activation $\mathbf{h}^{(3)}$ and the static task centroids, the entire sequence of ensembling weights $\alpha_k^{(4)} \dots \alpha_k^{(14)}$ is completely deterministic and could be pre-computed at Layer 3. Describing this as a "dynamic stateful router" is highly misleading.

### 4. Removal of Redundant Empirical Columns
Table 1 reports three identical columns of accuracies for Homogeneous, Heterogeneous, and Real-Time ($B=1$) batching configurations because the mathematical operations are independent. The authors should remove these redundant columns and replace them with a genuine sequential evaluation where cross-query interference or temporal dependencies actually exist (e.g., using the temporal carryover mechanism).

### 5. Excessive Hyperparameter Burden
The method requires tuning at least five to seven hyperparameters ($G$, $\gamma_{\text{drag}}$, $\Delta t$, $\epsilon$, $\tau_{\text{grav}}$, and potentially $\eta_{\text{feedback}}$, $\lambda_{\text{temporal}}$), which exhibit high sensitivity. The authors should focus on simplifying the framework to reduce this hyperparameter space, making the method more robust and easier to calibrate.

## Overall Presentation Quality
The writing is professional, and the formatting of tables and figures is clean. However, the presentation is highly verbose and metaphorically dense. By wrapping standard interpolation and filtering concepts in heavy astrophysical analogies ("virtual spacecraft coordinate probe", "celestial stars", "auxiliary physical cosmology"), the paper obfuscates the actual operations and makes the method unnecessarily difficult to understand. A direct, clear, and minimalist presentation would significantly improve readability.

## Potential Impact and Significance
The potential impact of GraviMerge is extremely low. In real-world edge AI and multi-task serving, system engineers value simplicity, low latency, ease of debugging, and minimal memory footprint. Introducing a stateful, complex continuous-time physical simulation with tangent projections and parallel transport across every single neural layer is highly impractical. 

Furthermore, because the incredibly simple SPS-ZCA baseline achieves nearly identical serving accuracy (88.51% vs. 88.69%) with absolute zero layer-to-layer weight jitter and zero extra state/parameters, there is virtually no practical incentive for any researcher or practitioner to implement or deploy GraviMerge.
