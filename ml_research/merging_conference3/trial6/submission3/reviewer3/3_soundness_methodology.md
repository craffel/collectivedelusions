# Soundness and Methodology Evaluation

## Clarity of the Description
- The description of the methodology is **exceptionally clear, precise, and well-structured**.
- The architectural diagram (Figure 1) clearly maps out the flow from the input feature to the PCA projection, unit sphere normalization, shared block routing, sigmoidal gating, and weight blending.
- The mathematical formulation is mathematically complete:
  - Section 3.1 clearly defines the expert weights, task vectors, and feature compression.
  - Section 3.2 formalizes block-wise sharing, group indices, and batch-wise coefficient averaging.
  - Section 3.3 provides a mathematically rigorous model of Expected Ruggedness that goes beyond i.i.d. conditions to incorporate depth-dependent variance scales and adjacent block correlations.
  - Section 3.4 formally models the physical sequential weight-space blending process, detailing hidden activation propagation.
  - Section 3.5 specifies the calibration optimization objective, including loss functions and weight decay exclusions.
- Additionally, Appendix Section 13 (Setup details) and Section 14 (Vision Transformer pilot demonstration recipe) provide explicit, step-by-step guidance that removes any ambiguity.

---

## Appropriateness of Methods
- **Block-wise Weight Sharing:** Extremely appropriate. It acts as an inductive bias that constrains the optimization search space, preventing overfitting on small calibration splits (64 samples).
- **Unsupervised PCA Compression:** Very appropriate for dimensionality reduction. It filters out high-frequency noise and compresses features to a low-dimensional task space ($d=K=4$) before passing them to the routing network.
- **Unit Sphere Normalization:** Appropriate to ensure magnitude invariance across heterogeneous inputs, preventing dominant feature scales from distorting gating predictions.
- **Negative Bias Initialization ($B_{group} = -2.0$):** This "inhibitory default" trick is highly appropriate and effective. It keeps specialized task vectors inactive by default, preventing catastrophic interference from random initialization.
- **Independent Sigmoidal Gating:** Very well-justified for open-world settings. While Softmax performs better in closed classification (due to sum-to-one regularization), the authors demonstrate through rigorous open-world audits (Appendix Section 8) that Sigmoid is appropriate because it can deactivate experts under OOD inputs and co-activate multiple experts for non-exclusive tasks.
- **Sequential Smoothing Regularizer ($\mathcal{L}_{\text{smooth}}$):** This is a highly appropriate alternative to runtime residual routing links. It penalizes adjacent weight discrepancies during training, reducing physical sequential seed variance without degrading the dynamic capacity.

---

## Potential Technical Flaws or Limitations
As an **Empiricist** reviewer, I closely scrutinized the methodology for potential flaws:
1. **Virtual Sandbox vs. Physical Sequential Weight Blending:** 
   The virtual sandbox is a stylized, single-layer weight-space ensembling setup where routing coefficients are averaged across layers before being applied to the classification head. As the authors themselves intellectually acknowledge, "layer-averaging collapse" is an artifact of this design. However, they successfully address this limitation by evaluating a true **physical sequential weight-space model-merging framework** on multi-layer MLP experts, proving that their findings transfer.
2. **High Variance in Physical Sequential Blending:**
   In the physical sequential setup, there is a relatively high standard deviation across different seeds (e.g., $43.20 \pm 22.49\%$ for Heterogeneous BWS $M=3$). This variance represents a major open challenge. The authors are commendably honest about this, showing that sequential propagation suffers from compounding representation drift where early routing errors disrupt downstream features. They address this by proposing:
   - **Sequential Smoothing Regularization:** Reduces seed standard deviation from 21.28% to 13.41% while maintaining strong mean accuracy.
   - **Residual Gating Links:** Reduces seed standard deviation but at the cost of collapsing performance toward static uniform merging.
3. **Scale of Experts:**
   The physical MLP experts are relatively small (3-layer MLPs) and evaluated on synthetic inputs from the sandbox. While this is a valuable stepping stone, scaling to larger pre-trained backbones is critical. The authors address this by executing a PyTorch-level **pilot demonstration on an actual Vision Transformer backbone** (\texttt{vit\_tiny\_patch16\_224}) in the appendix, showing that their method executes in 382.93 ms on CPU with only 60 parameters.

---

## Reproducibility
- **Exceedingly High:** The paper is a model of scientific reproducibility.
- All training and evaluation details are completely specified in the main text and Appendix Section 13:
  - Expert architecture: 3-layer MLP (\texttt{Linear(192, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)}).
  - Expert training details: 200 samples, Adam, LR=0.01, WD=1e-4, 120 epochs.
  - Calibration split: 64 total samples (16 per task).
  - Router optimization details: AdamW, LR=0.05, WD=1e-4 (excluding biases), 120 epochs, gradient clipping norm=1.0.
  - Seed evaluation: All experiments are evaluated across 5 independent random seeds ($42, 43, 44, 45, 46$), reporting means and standard deviations.
- The detailed formulas and pseudocode instructions make reproduction highly straightforward.
