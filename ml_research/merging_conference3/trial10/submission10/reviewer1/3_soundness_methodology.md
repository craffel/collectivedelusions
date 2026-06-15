# 3. Soundness and Methodology Evaluation

This evaluation focuses on the mathematical correctness, clarity of writing, methodological appropriateness, potential technical flaws, and reproducibility of the proposed framework.

## Clarity of the Description
The methodology is exceptionally well-written, structured, and easy to follow. The mathematical notation is clean and consistent across sections. The authors systematically lay out the problem setup, the 2D bilinear recurrence filter, the simplex-preservation proof, the boundary conditions, and the Adaptive Temporal Gating (ATG) mechanism. 
Crucially, the authors include a fully functional, self-contained PyTorch implementation in Appendix A (Listing 1), which details exactly how the Coordinate-Prior boundary condition and Power-Law ATG are computed in practice. This makes the method highly transparent and easy to understand.

## Methodological Appropriateness
For a sequential edge-serving environment, the proposed methodology is highly appropriate:
* **Discrete-Time Formulation:** Opting for a discrete-time bilinear recurrence rather than continuous-time biochemical ODEs (as in ChemMerge) or offline-learned state-space matrices (as in PAC-Kinetics) matches the discrete nature of digital neural network serving, where samples and layers arrive in discrete steps.
* **Simplex Preservation via Convex Combinations:** Exploiting the convexity of the probability simplex $\Delta^{K-1}$ to guarantee simplex preservation through a simple constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$) is a mathematically elegant and practically brilliant choice. It completely avoids the need for expensive online projection operators.
* **Power-Law Gating (ATG-PL):** The mathematical reasoning behind the upward bias of cosine similarity on non-negative probability coordinate spaces ($\mathbf{e}_t \in \mathbb{R}^K_{\ge 0}$) is highly solid. Applying a sharpening exponent ($\gamma = 3$) to collapse this residual momentum during task switches is an extremely effective, low-overhead way to resolve the classic smoothing-responsiveness trade-off.
* **Coordinate-Prior Boundary Condition:** The discovery of "spatial momentum cancellation" at the first adapted layer under standard raw-weight boundaries is a major technical find. The proposed Coordinate-Prior boundary derived from early task-coordinates successfully preserves active spatial smoothing at the entry layer without introducing task-agnostic accuracy drag.

## Potential Technical Flaws and Limitations
While the methodology is highly sound, we identify a few subtle areas and nuances that require careful consideration:
1. **Fine-Grained Manifold Fallbacks:** The authors acknowledge that in extremely fine-grained multi-task serving (e.g., fine-tuning experts on highly overlapping domains), early-layer activations may lack sufficient task-separability. They propose a 2-layer MLP coordinate mapper as a fallback (Appendix C). While this is a highly appropriate solution, it does mean the system is not *strictly* training-free in these extreme fine-grained settings. However, since the MLP trains in under 3 seconds on the microscopic $N_{\text{cal}} = 64$ split, the overhead remains negligible.
2. **Out-of-Distribution (OOD) Fallback:** The authors outline an elegant "OOD Fallback Policy" (Appendix B.1) involving uniform fallback or temporal state bypass. While the theoretical discussion is highly solid, these fallback policies are not integrated into the main implementation or empirically evaluated in Section 4. In practice, a robust deployment on physical edge hardware must implement these fallback strategies to prevent OOD noise from corrupting the active sequence history.
3. **Centroid Scaling on Massive LLMs:** For ultra-large LLMs with deep layers and large representational dimensions $D$, centroid storage scales as $K \times D \times L$ floats. For a model like LLaMA-7B ($L=32, D=4096$) with $K=10$ experts, this corresponds to approximately $10 \times 4096 \times 32 \times 4$ bytes $\approx 5.24$ MB. While this is extremely small compared to the model's 14 GB parameter budget, practitioners must be aware that centroid storage scale is linear, though practically negligible.

## Reproducibility
The reproducibility of the paper is **outstanding**:
* The authors provide a complete PyTorch implementation of the routing module in Listing 1.
* The paper lists the exact hyperparameter values used ($\beta_{\text{depth}} = 0.40, \beta_{\text{temp}, 0} = 0.40, \gamma = 3.0, \tau = 0.10$).
* The Analytical Coordinate Sandbox (ACS) setup is clearly parameterized (14 layers, representation dimension $D=192$, sequence length $T=1000$, etc.).
* The authors provide empirical results with mean and standard deviation across 5 independent evaluation seeds and report relative p-values from paired t-tests, confirming high scientific rigor and reproducibility.
