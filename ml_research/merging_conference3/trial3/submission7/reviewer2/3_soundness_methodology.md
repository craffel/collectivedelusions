# 3. Soundness and Methodology Evaluation

## Clarity of Description
The methodology is presented with mathematical clarity and structural rigor.
- Section 3.1 clearly formalizes the task-vector formulation and linear merging.
- Section 3.2 outlines the 5 nested structural levels of granularity with exact parameter counts, which is highly appreciated.
- Section 3.3 presents the prediction entropy minimization surrogate loss.
- Section 3.4 describes the update rules for Adam and 1+1 ES explicitly, including hyperparameters ($\eta=0.02, S=60$ steps for Adam; $\sigma=0.05, S=100$ steps and 1/5th success rule for ES).
- Section 3.5 provides mathematically precise definitions for Elastic Spatial Regularization (ESR) and Depth-wise Total Variation (TV) smoothness.

### Key Gaps in Description and Clarity:
1. **Handling of Classification Heads:**
   The paper evaluates 4 distinct classification tasks: MNIST (10 classes), FashionMNIST (10 classes), CIFAR-10 (10 classes), and SVHN (10 classes). The paper does not specify how the final classification heads are structured or merged.
   - Are there 4 independent classification heads, or do they share a single head?
   - If they are task-specific heads, how are they handled during test-time adaptation? Do we only adapt the shared ViT backbone while keeping the heads frozen?
   - During evaluation of the multi-task surrogate loss (Equation 5), does the model use the correct task-specific head for each respective task calibration batch $X_{\text{cal}, k}$? If so, this assumes that task labels (IDs) are known at test-time, which conflicts with the description of a fully "unlabeled calibration stream." This needs to be clarified.
2. **Other Model Parameters:**
   The paper states that Level 5 Tensor-wise merging defines 6 coefficients per layer per task scaling the projection components (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`).
   - What happens to the other parameters in the Vision Transformer, such as LayerNorm weights/biases, class tokens, positional embeddings, patch embeddings, and classification head weights?
   - Are they kept static at their pre-trained base values? Are they merged using a uniform weight? Or are they excluded from the task vectors entirely?

---

## Appropriateness of Methods
- **Nested Resolution Hierarchy:** The 5 levels of structural granularity are highly appropriate for testing the hypothesis.
- **Optimizer Diversity:** Comparing Adam (first-order) and 1+1 ES (zero-order) is a highly sound methodological choice, as it isolates how optimization trajectories affect overfitting.
- **Surrogate Loss Choice:** Minimizing prediction entropy is a standard practice in Test-Time Adaptation (TTA), but the paper's deep dive into its *misalignment* with classification accuracy is highly appropriate and rigorous.

---

## Technical Soundness and Potential Flaws
- **Rigorous Evaluation of 1+1 ES Robustness:**
  The authors' analysis of the "sluggishness hypothesis" (underfitting) is an excellent and highly sound technical contribution. It shows a deep understanding of optimization dynamics. Rather than naively claiming zero-order methods are inherently superior for representation learning, they prove that ES's apparent "robustness" at high dimensions is simply an artifact of its failure to optimize 288 parameters in 100 steps. This keeps the model near its static initialization, preserving baseline generalization.
- **Regularization Formulation:**
  The ESR and TV regularizers are mathematically sound and physically intuitive. Pulling fine-grained parameters to their layer-wise mean (ESR) and smoothing across depths (TV) matches the physical priors of transformer networks.

---

## Reproducibility
The methodology contains sufficient mathematical details and hyperparameter listings ($\eta, S, \beta, \gamma, \sigma, N$) for a researcher to implement a similar framework. However, reproducing the exact results would require the authors to:
1. Clarify the classification head routing mechanism.
2. Specify the status of non-projection parameters (LayerNorm, embeddings, etc.).
3. Provide the open-source code for `MergedViTTiny` and the dynamic on-the-fly coefficient-aware forward pass.
