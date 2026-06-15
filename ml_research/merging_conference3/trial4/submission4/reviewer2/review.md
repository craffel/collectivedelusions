# Peer Review

**Paper Title:** SpectralMerge: Frequency-Domain Parameter Consolidation for Post-hoc Model Merging

---

## 1. Summary of the Paper
This paper proposes **SpectralMerge**, an optimization search-space re-parameterization and regularization framework for post-hoc parameterized model merging. Instead of optimizing layer-wise task-combining coefficients $\vec{\alpha}_k$ directly in the spatial coordinate space across network depth, SpectralMerge maps this 1D sequence to the frequency domain using the Discrete Cosine Transform (DCT-II). 

The authors propose two frequency-domain formulations:
1. **SpectralMerge-LP (Low-Pass Hard Cutoff):** Restricts the trainable parameters to the first $F$ low-frequency spectral components, zeroing out higher frequencies to act as an analytical low-pass filter.
2. **SpectralMerge-Reg (Soft Spectral Regularization):** Optimizes all frequency components but adds a quadratic **Spectral Decay Penalty** ($\lambda_j = \mu \cdot j^2$) to the loss function to softly penalize high-frequency spatial oscillations.

Additionally, they propose **Block-wise Spectral Merging** to partition heterogeneous layer types and apply independent spectral transforms, and **LP-Adaptive** to dynamically expand the active bandwidth during optimization. The method is evaluated on a synthetic ViT-B/32 simulation landscape across standard/adversarial streams, and validated on physical PyTorch models (a 12-layer MLP and a pre-trained ResNet-18 on CIFAR-10 tasks).

---

## 2. Overall Recommendation

**Rating:** **3: Weak Reject**

**Justification:**
While the paper is mathematically elegant, exceptionally well-structured, and features highly professional visualizations, its critical weaknesses significantly outweigh its merits in its current form. 

1. **Failure to Cite and Discuss Crucial Concurrent Literature:** The paper claims to introduce a "paradigm shift" of operating model merging in the frequency domain, yet it completely fails to cite or discuss **FREE-Merging (Fourier Transform for Efficient Model Merging)** (published at **ICCV 2025**). This concurrent work already operates in the frequency domain of model merging (applying Fourier Transforms to filter model parameters directly). Failing to discuss this literature severely weakens the paper's claims of conceptual originality and novelty.
2. **Methodological Contradiction in Handling Architectural Heterogeneity:** The authors correctly argue that forcing a single smooth curve across extremely heterogeneous physical layers is architecturally restrictive and propose Block-wise Spectral Merging. However, in their real-world ResNet-18 evaluation, **they completely violate this argument by applying a global DCT-II globally across all 18 layers**—forcing convolutional filters, batch normalization parameters, and the classification head onto a single continuous trigonometric trajectory. This is a severe methodological contradiction.
3. **Critical Baseline Omission:** The paper compares SpectralMerge to unconstrained spatial search and polynomial smoothing (PolyMerge) but completely omits the most standard, simple spatial baseline: **direct spatial smoothness regularization** (adding a first-order or second-order finite difference penalty on layer coefficients directly in the spatial coordinate space). Without this baseline, the authors have not proven that performing optimization in the spectral domain is superior to simple spatial trajectory smoothing in the spatial domain.
4. **Poor Absolute Performance and Toy-Scale Evaluations:** In their physical ResNet-18 checkpoint evaluation on CIFAR-10, the proposed SpectralMerge-Reg only achieves **54.00%** multi-task accuracy. On a joint dataset consisting of two binary classification tasks, this is barely above random guessing ($50.00\%$) and represents a massive drop from the original experts ($86.00\%$ and $65.00\%$). Furthermore, all physical evaluations are restricted to toy-scale setups (120 training samples, 15 validation samples, two binary tasks), failing to demonstrate practical scalability on realistic multi-task benchmarks.
5. **Catastrophic Failure of Hard-Cutoff Variants under PEFT:** Under standard localized fine-tuning (PEFT), the hard-cutoff variant SpectralMerge-LP and LP-Adaptive fail completely, collapsing to majority-guessing ($29.00\%$). This exposes a severe vulnerability of their primary proposed method under standard fine-tuning workflows.

---

## 3. Strengths and Weaknesses

### Strengths
* **Elegant Mathematical Formulation:** The use of the orthonormal DCT-II basis to re-parameterize the merging coefficient trajectory is mathematically sound. The proof of flat spatial derivatives at virtual boundaries (zero slope) is elegant and provides a solid justification for using DCT-II over DFT or DST.
* **Effective Regularization against Noise:** The soft spectral decay penalty (SpectralMerge-Reg) is highly effective at preventing validation overfitting under extreme data scarcity, as shown by its performance on physical checkpoints compared to spatial search.
* **Deep Error Analysis:** The authors do not shy away from failure modes. Their detailed explanation of the **PEFT-Induced Step-Function Discontinuity** and why hard low-pass cutoffs fail on localized weights is a highly valuable, insightful addition.
* **Thorough Simulation Stress-Testing:** The simulation evaluations are highly comprehensive, covering 30 random seeds, multiple adversarial stream corruptions (extreme label shift, bursty streams, small batch noise), and validation selection biases.

### Weaknesses
* **Missing Key Related Work:** Complete omission of ICCV 2025 FREE-Merging and other frequency-domain parameter filtering techniques, overstating the "paradigm-shifting" novelty of the work.
* **Methodological Contradiction on Physical Networks:** Applying a global DCT-II across extremely heterogeneous layers in ResNet-18 (conv, batch norm, fully connected head) directly contradicts the authors' own block-wise architectural arguments.
* **Missing Standard Smoothing Baseline:** Failure to compare against simple spatial finite-difference regularization ($\sum (\alpha_k(l) - \alpha_k(l-1))^2$) in the spatial coordinate space.
* **Toy-Scale and Weak Physical Results:** Physical evaluations are run on tiny toy datasets (120 training samples). The absolute multi-task accuracy of SpectralMerge-Reg is extremely low ($54.00\%$), showing that post-hoc model merging still suffers from severe performance degradation.
* **Exaggerated Mathematical Claims:** Exaggerating the "ill-conditioning" of PolyMerge's Vandermonde matrix as a "catastrophic bottleneck" that requires "solving a system of equations" when PolyMerge is actually optimized via standard first-order gradient descent.
* **Overuse of Academic Hype:** Saturated with hyperbolic, non-sober language ("paradigm-shifting", "visionary", "spectacular blowout") that distracts from the objective science.

---

## 4. Detailed Evaluation

### Soundness
* **Rating:** **Fair**
* **Justification:**
  While the mathematical derivation of the DCT-II and its boundaries is correct and sound, the methodology suffers from serious conceptual flaws. Specifically, treating a highly heterogeneous network (ResNet-18) with wildly disjoint layer types as a single, continuous 1D spatial sequence is physically and architecturally nonsensical. Doing so globally across 18 layers (including batch norm parameters and convolutional filters) violates the authors' own architectural arguments regarding heterogeneity.
  
  Furthermore, the claim that polynomial ill-conditioning is a "catastrophic bottleneck" is technically flawed. Because PolyMerge uses low-degree polynomials ($d=2$), the condition number of the Vandermonde matrix is very small. Since coefficients are updated via standard first-order backpropagation (gradient descent) and not by solving a system of linear equations, this ill-conditioning has a negligible impact on optimization stability. Finally, the omission of direct spatial smoothness regularization (finite differences) as a baseline represents a major soundness gap.

### Presentation
* **Rating:** **Good**
* **Justification:**
  The manuscript is logically structured, and the mathematical notation is precise. The professional, highly descriptive figures (condition number curves, sample complexity sweeps, validation bias plots, and convergence trajectories) are exceptional and greatly enhance readability. 
  
  However, the writing style is overly verbose and highly repetitive (e.g., repeating the explanation of boundary symmetric extensions multiple times). Most importantly, the presentation is heavily saturated with excessive academic hype ("paradigm-shifting framework", "visionary question", "blowout improvement", "ultimate stability") and uses the fancy term "Overfitting-Optimizer Paradox" to describe basic statistical overfitting. A more sober, objective, and condensed presentation is highly recommended. Additionally, a critical reproducible detail—the **learning rates** used for optimizing spectral vs. spatial coordinates—is completely omitted.

### Significance
* **Rating:** **Fair**
* **Justification:**
  In its current state, the practical significance of SpectralMerge is limited. The paper over-relies on a synthetic quadratic simulation landscape (Model II) for almost all of its systematic findings.
  
  On actual physical checkpoints (ResNet-18), the absolute multi-task accuracy achieved by the best spectral variant is only $54.00\%$ (barely above a random/majority guessing rate of $50.00\%$, and a massive degradation from the task-specific experts which achieve $86.00\%$ and $65.00\%$). This shows that SpectralMerge merely acts as an optimization stabilizer rather than a practical solution to task interference. Furthermore, the physical evaluations are restricted to a tiny, toy-scale setup (120 training samples, 15 validation samples, two binary classification tasks). Because the authors fail to evaluate SpectralMerge on standard, large-scale model merging benchmarks (e.g., merging 5-8 vision models fine-tuned on full datasets, or merging LLMs on GLUE), its real-world scalability and utility remain unproven. Finally, the complete collapse of their primary hard-cutoff variant (SpectralMerge-LP) under standard localized fine-tuning (PEFT) severely limits its practical value.

### Originality
* **Rating:** **Fair**
* **Justification:**
  The originality is heavily compromised by the omission of concurrent frequency-domain model merging literature, specifically **FREE-Merging (ICCV 2025)**. The authors' claim of "introducing a novel frequency-domain parameter consolidation paradigm in model merging" is overstated given that Fourier filtering of model parameters has already been established. 
  
  While there is a clear conceptual distinction (FREE-Merging filters high-dimensional backbone weights, whereas SpectralMerge regularizes low-dimensional combining trajectories), the paper must explicitly discuss and cite this work to properly position itself. Stripped of the exaggerated claims, the transition from polynomial smoothing (PolyMerge) to a trigonometric cosine basis (SpectralMerge) is a highly logical, evolutionary step rather than a revolutionary paradigm shift. Re-parameterizing 1D spatial curves using orthogonal frequency bases is a standard and classical technique in signal processing and regression analysis.

---

## 5. Questions for the Authors / Suggestions for Improvement

To improve the paper's quality, the authors are encouraged to address the following issues in a future revision:

1. **Incorporate Missing Related Work:** Cite and discuss **FREE-Merging (ICCV 2025)** and other frequency-domain parameter-filtering techniques. Clearly articulate the conceptual difference between filtering model weights directly vs. re-parameterizing layer-wise blending coefficients.
2. **Add Direct Spatial Smoothness Regularization Baseline:** Implement and report results for a spatial smoothness baseline: optimizing spatial layer coefficients $\vec{\alpha}_k$ while penalizing the first-order finite differences $\gamma \sum ( \alpha_k(l) - \alpha_k(l-1) )^2$. Show whether performing optimization in the spectral domain (DCT-II) provides any fundamental benefits over simply regularizing the spatial trajectory directly in the spatial domain.
3. **Resolve the ResNet-18 Architectural Contradiction:** Instead of optimizing the 18 layers of ResNet-18 globally under a single DCT-II (which forces convolutional filters, batch norm parameters, and classification heads onto a single trajectory), implement and evaluate **Block-wise/Layer-type Spectral Merging** on the ResNet-18 architecture. Segment the network into homogeneous subsets (e.g., convolutional blocks, batch norm parameters, classification heads) and apply independent DCT-II transforms to each. Does this resolve the step-function discontinuity and prevent the collapse of SpectralMerge-LP?
4. **Scale Up Empirical Evaluations:** Evaluate SpectralMerge on a standard, full-scale multi-task model merging benchmark. For example, merge 5 to 8 diverse vision classifiers (e.g., ViT or ResNet-50 models) fine-tuned on full standard datasets (CIFAR-100, SVHN, DTD, RESISC45, EuroSAT) and compare against competitive baselines (TIES-Merging, DARE, PolyMerge). Show that the method scales to realistic, non-toy settings.
5. **Report Complete Optimization Details:** Provide the exact learning rates, weight decays, and hyperparameter grids used for optimizing both the spectral coordinates and all baseline methods.
6. **Tone Down the Hype:** Revise the manuscript to adopt a sober, scientific, and professional tone. Strip away hyperbolic terms such as "paradigm-shifting", "visionary", "spectacular blowout", and "ultimate stability," and describe the "Overfitting-Optimizer Paradox" simply as overfitting under extreme sample complexity.
