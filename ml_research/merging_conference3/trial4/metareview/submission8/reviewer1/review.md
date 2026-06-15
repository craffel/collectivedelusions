# Peer Review

## Summary of the Paper
This paper investigates the robustness of test-time adaptive model merging (TTA) frameworks under downstream post-training quantization (PTQ) for edge deployment. The authors identify a vulnerability termed **Quantization-Operator Overfitting**, where unregularized TTA (e.g., AdaMerging) converges to extremely sharp local minima that yield high floating-point (FP32) performance but collapse catastrophically under quantization noise (such as INT8 or INT4 symmetric/asymmetric schemas).

To address this, the authors propose **CR-PolySACM (Clipping-Regularized Sharpness-Aware Subspace Model Merging)**, a framework combining:
1. **Differentiable Polynomial Subspace Parameterization (PolyMerge):** Restricting $L \times K$ layer-wise blending coefficients to a low-degree polynomial of normalized network depth, reducing the optimization search space from 56 parameters to 12 parameters.
2. **Clipping-Regularized Sharpness-Aware Minimization (CR-SACM):** Minimizing local loss sharpness to improve robustness under discretization noise. This component addresses a "task-vector norm scale pathology" (where a massive discrepancy in task-vector norms across layers makes standard flatness optimizers blind to highly sensitive, low-norm layers such as the final layer norm) by scaling the sharpness perturbation inversely by the clipped task-vector norms.

The framework is evaluated across four visual domain datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using a Vision Transformer backbone, showing that while unconstrained adaptive model merging collapses under low precision, restricting the search space via depth-dependent polynomials (PolyMerge) provides exceptional stability. Under aggressive INT4 quantization, CR-PolySACM achieves a joint mean accuracy of **19.07%**, outperforming standard PolyMerge (**18.10%**) by **+0.97%**.

---

## Strengths and Weaknesses

### Strengths
1. **The Elegance of Polynomial Subspace Parameterization:** 
   The continuous polynomial parameterization of depth-dependent blending coefficients (the PolyMerge component) is an exceptionally elegant, simple, and effective technique. By reducing the number of optimization parameters from 56 to 12 based on the natural hierarchical depth of deep networks, it introduces minimal overhead, uses zero extra hyperparameters, and prevents transductive overfitting on small test-time calibration streams. Crucially, it provides a massive performance leap of **+8% to +9%** across all precision settings compared to unconstrained TTA methods. This represents a outstanding example of solving a complex, high-dimensional problem through elegant architectural simplification.
2. **Exhaustive and Rigorous Empirical Sweeps:**
   The paper features a highly comprehensive evaluation, validating the methods across six different post-training quantization schemas (FP32, four INT8 variants, and INT4), using multiple random seeds, and comparing against a solid collection of baselines (including RegCalMerge, Q-Merge, and unconstrained HessMerge).
3. **Scientific Candor and Transparency:**
   The authors are exceptionally honest and clear in Section 4.4 about the fundamental limitations of their work. They explicitly highlight the "Expert-to-Merge Drop" (a -31.27% performance gap between individual experts and the best merged model due to disjoint domain shifts) and acknowledge that their INT4 accuracy (19.07%), while setting a new state-of-the-art, is practically unusable for production systems, framing its value as purely scientific.

### Weaknesses
1. **Unjustified Complexity of CR-SACM with Detrimental Practical Returns:**
   While the polynomial constraint (PolyMerge) is extremely simple and effective, the primary proposed contribution—the addition of the CR-SACM flatness optimization loop—introduces significant algorithmic complexity. It requires calculating layer-wise norms, introducing a clipping threshold parameter ($\beta = 0.10$), scaling perturbations inversely by $(V_{\text{clipped}, k}^l)^2$, and clamping perturbed coefficients, requiring two forward-backward passes per iteration.
   
   Despite this high-complexity machinery, the empirical results show that in every single practical precision format (unquantized FP32 and all four INT8 formats), **adding CR-SACM consistently degrades performance** compared to the simpler PolyMerge baseline:
   - **FP32:** PolyMerge (**57.40%**) vs. CR-PolySACM (**57.00%**)
   - **INT8 Sym (Tensor):** PolyMerge (**57.62%**) vs. CR-PolySACM (**56.62%**)
   - **INT8 Sym (Channel):** PolyMerge (**58.15%**) vs. CR-PolySACM (**57.23%**)
   - **INT8 Asym (Tensor):** PolyMerge (**56.57%**) vs. CR-PolySACM (**56.48%**)
   - **INT8 Asym (Channel):** PolyMerge (**57.43%**) vs. CR-PolySACM (**56.93%**)
   
   This means that for any practitioner deploying a model in standard usable formats, the simpler, faster, and more elegant PolyMerge baseline is superior. The proposed CR-SACM loop is only beneficial under extreme INT4 quantization (+0.97% gain), but the absolute performance in INT4 is 19.07% (barely above the 10.00% random guessing floor). Over-engineering a system to get a minor fractional benefit in a completely broken, non-functional regime does not justify the added complexity.

2. **Artificial Creation of the Scale Pathology and Lack of a Direct Parameter-Space SAM Baseline:**
   The paper identifies the "task-vector norm scale pathology" as a major challenge in flatness optimization, requiring the complex CR-SACM clipping formulation to solve. However, this scale pathology is entirely an artificial consequence of the authors' design choice. In CR-PolySACM, the actual parameters being optimized are the 12 polynomial coefficients $\mathbf{p} \in \mathbb{R}^{3 \times K}$. Instead of perturbing these 12 parameters directly (which is the standard way to apply parameter-space Sharpness-Aware Minimization), the authors choose to perturb the intermediate 56 layer-wise coefficients $\Lambda$ and then backpropagate the perturbed loss to $\mathbf{p}$.
   
   If the authors had simply applied standard SAM directly to the 12-dimensional parameter space $\mathbf{p}$ (i.e., $\mathbf{p} \leftarrow \mathbf{p} + \Delta\mathbf{p}$), the entire task-vector norm scale pathology would have been completely bypassed. There would be no need to measure task-vector norms, no need for the clipping threshold $\beta$, and no risk of gradient explosion. The paper completely fails to evaluate or discuss this simpler, more direct parameter-space SAM baseline.

3. **Domain Disconnect and Practical Viability:**
   While the authors transparently acknowledge the "Expert-to-Merge Drop," the fact remains that a merged model achieving only 57.40% FP32 accuracy when individual experts achieve 88.67% average accuracy is extremely degraded. Merging models fine-tuned on highly disjoint domains (MNIST, CIFAR-10, SVHN) in weight space appears to be fundamentally flawed. While the polynomial subspace constraint significantly mitigates this compared to uniform blending, the massive performance gap raises doubts about the practical viability of post-hoc model merging under significant domain shifts compared to simple routing or mixture-of-experts (MoE) baselines.

---

## Evaluation on Reviewing Criteria

### Soundness
* **Rating:** Good  
* **Justification:** The paper's mathematical derivations (including the second-order Taylor expansion and polynomial curvature decomposition) are correct and internally consistent. The empirical evaluation is extensive and conducted across multiple random seeds, and the diagnostic analysis of layer-wise norms is accurate. However, the methodology is marked down due to a questionable design choice (perturbing intermediate coefficients instead of the actual optimization variables, which artificially created the norm scale pathology) and the lack of a direct parameter-space SAM baseline.

### Presentation
* **Rating:** Excellent  
* **Justification:** The submission is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is clean and consistent. Figures and tables are professionally formatted, and the authors are highly commendable for their outstanding transparency and honest discussion of their work's limitations.

### Significance
* **Rating:** Fair  
* **Justification:** The depth-dependent polynomial subspace constraint (PolyMerge component) is highly significant, offering a fast, simple, and elegant way to stabilize and regularize test-time model composition. However, the primary proposed contribution—the CR-SACM extension (CR-PolySACM)—has low significance. It adds significant algorithmic complexity and actively degrades performance in all practical deployment precision formats (FP32, INT8), only showing a minor gain in an unusable INT4 regime where the model is completely broken anyway.

### Originality
* **Rating:** Good  
* **Justification:** The paper presents a novel combination of depth-dependent polynomial constraints and weight-space flatness optimization. The analysis of layer-wise task-vector scale-blindness and the formulation of clipping-regularized perturbations are creative contributions to the model merging literature, though the complexity introduced is not justified by the empirical results.

---

## Overall Recommendation
* **Recommendation:** 3: Weak reject  
* **Justification:** The paper has clear merits, particularly the elegant and highly effective continuous polynomial subspace constraint (the PolyMerge component). However, the weaknesses currently outweigh these merits. The primary proposed method, CR-PolySACM, is an over-engineered approach that introduces substantial algorithmic complexity and actively degrades performance in all practical deployment precisions (FP32 and INT8) compared to standard PolyMerge. It only provides a marginal benefit in a completely broken INT4 regime where the model is non-functional anyway. Furthermore, the paper suffers from a key methodological omission by failing to evaluate a simpler, direct parameter-space SAM baseline on the 12-dimensional polynomial parameters, which would have bypassed the scale pathology altogether. 

To be suitable for acceptance, the authors should revise the work to address these concerns:
1. **Include a Direct Parameter-Space SAM Baseline:** Implement and evaluate standard SAM applied directly to the 12-dimensional polynomial parameters $\mathbf{p}$. Show whether this simpler approach can achieve competitive flatness and robustness without the need for the complex CR-SACM norm-clipping formulation.
2. **Demonstrate Practical Utility of CR-SACM in Usable Precisions:** Revise the formulation or hyperparameter tuning of the flatness regularizer to show that it can provide genuine robustness gains in usable deployment scenarios (such as INT8 or continuous FP32 under out-of-distribution noise) without degrading baseline representation quality.
3. **Expand on Alternative Architectures:** Provide a discussion or comparison with alternative paradigms (such as routing networks or mixture-of-experts) to contextualize the severe domain disconnect performance drop.
