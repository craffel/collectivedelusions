# Soundness and Methodology

## 1. Clarity of Description
The technical description of the methodology is exceptionally clear, rigorous, and logically structured:
- **Dynamic Parameter Merging**: Clear definitions of the pretrained base model, the task vectors ($V_k$), and the sample-wise ($B=1$) versus batch-averaged coefficients.
- **Low-Dimensional Unit-State Feature Projection**: A mathematically solid projection to a low-dimensional space via a static normalized random projection matrix (Johnson-Lindenstrauss lemma), normalized onto a unit sphere. This is well-defined and includes numerical stabilizers ($\epsilon = 10^{-8}$).
- **Layer-wise Classical Routing**: Standard, clear linear projections followed by Softmax activation, explicitly contrasted against QWS-Merge's cosine activations.
- **Task-Variance Regularization ($\mathcal{L}_{VR}$)**: Detailed, clear formulation of intra-task sample variance. The authors include an explicit note on why the uncorrected population variance is chosen (to handle $|S_k|=1$ without division-by-zero), showing an impressive attention to numerical and mathematical detail.
- **Sequential Smoothness Regularization ($\mathcal{L}_{\text{smooth}}$)**: A well-formulated, consecutive layer consistency constraint.
- **Vectorized Assembly**: Explicitly explains how PyTorch's `torch.vmap` or `einsum` are used to compute sample-specific parameter assembly without batch-average compromises.

## 2. Appropriateness of Methods
The methods chosen are highly appropriate for the questions being investigated:
- **Zero-Initialized Softmax Routing**: Starting the routing weights at exact zero forces a maximum-entropy uniform prior, which is the exact mathematical compromise needed under high data-scarcity (64 samples).
- **Analytical Coordinate Sandbox**: Using a perfectly controlled 192-dimensional synthetic sandbox with a calibrated representation simulator is highly appropriate. It allows the authors to run exhaustive sweeps across 10 independent random seeds, sweeping over subspace overlap, projection dimension, and regularization penalties, which would be computationally noisy and expensive on massive models.
- **Real-World Validation**: The inclusion of real-world model merging on MNIST and FashionMNIST experts with a shared CNN backbone (Section 4.6) is an excellent choice. It validates that the observed phenomena (vectorization collapse, batch-average confounder, and prior-layer stabilization) generalize to real-world visual representations.

## 3. Potential Technical Flaws and Simplifications
- **Layer-Averaging Simplification in the Sandbox**: The sandbox models a 14-layer backbone, predicting independent routing coefficients per layer. However, the expert classifiers are represented by a single linear layer. To fit this, the authors average the predicted layer-wise coefficients over the layer dimension: $\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \alpha_k(l)$.
  - *Analysis*: This is a notable simplification. In actual deep sequential networks, representations are processed sequentially without layer-averaging, making consecutive parameter alignment and "routing jitter" a functional concern.
  - *Mitigation*: The authors are highly transparent about this simplification (Section 4.10) and successfully mitigate it by formulating and validating the Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$), which is shown to reduce sequential routing weight jitter by over 57.5%. They also provide a ready-to-run CLIP ViT-B/16 validation roadmap in Appendix A.
- **Low-Rank Parameter Assembly (Dynamic LoRA)**: While the systems-level analysis of full-parameter assembly bottlenecks is outstanding, the authors do not evaluate Dynamic LoRA's capacity in their main tables, instead dedicating an entire section (Section 4.8) to analyze LoRA's capacity as a function of the adapter rank $r$. This is a complete and excellent mitigation of any potential capacity concerns.

## 4. Reproducibility
The reproducibility of the paper is **excellent**:
- Every hyperparameter is clearly documented: calibration dataset size ($N=64$), learning rate ($10^{-3}$), Adam optimizer, weight decay ($\lambda_{wd} = 10^{-3}$), number of epochs (100), and specific training parameters.
- The 10 independent seeds (42 to 51) are explicitly stated.
- The mathematical formulations are complete, with no hidden variables, and are easily implementable.
- The datasets, baseline architectures (L3-Linear, L3-Softmax, QWS-Merge, Uniform), and calibration environments are described with extreme precision.
- The paper includes a complete real-world visual expert merging experiment on MNIST and FashionMNIST, providing exact classification test accuracies.
- Appendix A provides a detailed, comprehensive roadmap for reproducing the findings on real-world CLIP ViT-B/16 checkpoints, making it exceptionally easy for future researchers to build on this work.
