# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of the paper is exceptionally well-written, structured, and mathematically precise:
- **Formulation:** Section 3 clearly defines task vectors, the layer-wise weight merging parameterization, and the three distinct search spaces (GT-Merge, Poly-Val-Merge, and unconstrained layer-wise).
- **Optimization:** The paper clearly contrasts the supervised cross-entropy objective on scarce offline validation data with the unsupervised prediction entropy minimization objective of online TTA.
- **Algorithm:** Algorithm 1 (OFS-Tune) is straightforward and easy to understand.
- **Abstractions:** The authors are highly transparent about the mathematical formulation of their continuous simulation calibration (coupled Model II sensitivity landscape with non-convex cosine entropy surrogates) and explicitly defend these abstractions in Appendix B.

---

## Appropriateness of Methods
1. **Nelder-Mead & PyTorch Adam optimizers:** Using a derivative-free black-box optimizer (Nelder-Mead) for low-dimensional, non-differentiable spaces, and PyTorch Adam for scaling to high dimensions, is highly appropriate. The comparative analysis of random search, Nelder-Mead, and PyTorch Adam successfully exposes how different search behaviors interact with parameter capacity.
2. **Stress-Testing under Non-I.I.D. Stream Conditions:** To expose the fragility of online TTA, evaluating the models under Extreme Label Shift, Bursty Task Streams (temporal shifts), and Small Batch Sizes (gradient noise) is methodologically outstanding. These represent realistic target-stream shifts that are frequently neglected in the literature.
3. **Physical Neural Network Validation:** Adding a physical evaluation on actual Convolutional Neural Networks trained on MNIST and FashionMNIST is a highly commendable step. It grounds the theoretical claims of the simulation in real deep weight-space optimization dynamics.
4. **Analyzing Validation Selection Bias:** Sweeping systematic validation domain shifts (isotropic Gaussian bias and structured late-layer semantic shift) is a highly appropriate way to challenge the robustness of offline tuning under target mismatch.

---

## Potential Technical Flaws and Limitations (Empiricist Perspective)
While the methodology is highly rigorous, an **empiricist** reviewer must highlight several limitations and potential areas of concern:
1. **A fidelity gap in the simulation framework:** The simulation framework relies on a mathematically closed, continuous quadratic landscape with an artificial cosine wave penalty to model prediction entropy non-convexity. Although this is an elegant mathematical idealization that successfully replicates TTA literature performance under sterile conditions (Section 4.4), it is still an abstraction. Real-world deep networks have highly coupled, high-dimensional representational spaces with sharp discontinuous barriers and saddle points that may behave differently under local optimization.
2. **Toy-scale physical neural network validation:** The physical validation is limited to a 5-layer CNN on MNIST and FashionMNIST (input size $28 \times 28$, ~100k parameters). While it successfully validates the "Overfitting-Optimizer Paradox" and the "no-data" strawman, these datasets are highly simplified, and the model lacks the multi-head attention blocks and representational hierarchies of the Vision Transformers (ViT-B/32; 86M parameters) or LLMs where model merging is typically deployed. It remains empirically unproven whether the exact same trends (e.g., OFS-Tune outperforming PolyMerge) hold at scale on large-scale transformers.
3. **Information access boundaries ("Apples-to-Oranges" comparison):** The authors compare supervised few-shot validation tuning (OFS-Tune) directly against unsupervised zero-shot target adaptation (AdaMerging, RegCalMerge, PolyMerge). This is a fundamental discrepancy in the information access boundaries. If target-task labels are strictly impossible to acquire due to edge-deployment privacy, proprietary restrictions, or highly non-stationary domain generalization, OFS-Tune cannot be used. Although the authors argue that few-shot labels are easily procurable in most real-world software engineering applications, they should more clearly respect the target zero-shot boundaries and avoid completely dismissing TTA for setups where labeled validation data is physically unavailable.
4. **Uncertainty in Validation Set Selection:** The paper does not specify whether the 10 validation samples per task ($M=10$) are selected randomly and varied across the 5 independent random seeds in the physical CNN experiments, or if they are fixed. If validation sets are randomized, the standard deviation of OFS-Tune GT-Merge is quite high ($5.81\%$), which highlights that few-shot validation tuning is highly sensitive to the specific samples chosen. This is a critical empirical challenge in few-shot learning that deserves more explicit discussion and quantification.

---

## Reproducibility
The reproducibility of the work appears **excellent**:
- The paper details all hyperparameter settings, optimizers, learning rates, and epochs for both the simulation landscape and the physical CNN experiments (Appendix A).
- All experiments (both simulation and physical validation) are executed across multiple random seeds (30 seeds for simulation, 5 seeds for physical CNN), which is a high standard for scientific reproducibility and statistical soundness.
- Code blocks and mathematical formulations are provided, making the baseline implementation straightforward to replicate.
