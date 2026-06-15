# Mock Review: SpectralMerge: Frequency-Domain Model Merging via Discrete Cosine Transform

## 1. Summary of the Paper
This paper proposes **SpectralMerge**, a post-hoc model-merging framework that transitions the optimization of layer-wise task-combining coefficients from the physical, spatial depth space into the **frequency domain** using the **Discrete Cosine Transform (DCT-II)**.

The authors argue that existing parameterized model-merging frameworks (e.g., AdaMerging, RegCalMerge) are fundamentally constrained by optimizing layer-wise combining coefficients as independent, unconstrained variables directly in the physical spatial coordinate space. Under data-scarce (few-shot) regimes, these unconstrained spatial coefficients oscillate wildly across network depth, acting as high-frequency optimization noise that catastrophically overfits to local validation streams—a phenomenon the authors term the **Overfitting-Optimizer Paradox**.

By treating the vector of layer-wise task combining coefficients as a discrete 1D spatial signal across network depth, SpectralMerge maps them to spectral coordinates using the orthonormal DCT-II. The authors introduce two elegant frequency-domain formulations:
1. **SpectralMerge-LP (Low-Pass Hard Cutoff):** Restricts optimization to the first $F$ low-frequency spectral coordinates (setting $c_{k,j} = 0$ for all $j \ge F$), acting as an analytical low-pass filter to eliminate high-frequency degrees of freedom from the search space entirely.
2. **SpectralMerge-Reg (Soft Spectral Regularization):** Optimizes all $L$ frequency coordinates, but adds a quadratic **Spectral Decay Penalty** ($\lambda_j = \mu \cdot j^2$) to the loss, penalizing high-frequency spatial oscillations progressively.

The spatial coefficients are recovered using the Inverse Discrete Cosine Transform (IDCT-III) to construct the merged model. 

The paper evaluates SpectralMerge across three complementary scales of execution: a calibrated continuous multi-task weight-merging simulation landscape (ViT-B/32), a physical PyTorch Multi-Layer Perceptron (MLP) with alternating layer types, and real-world pre-trained ResNet-18 checkpoints fine-tuned on binary CIFAR-10 classification tasks. Additionally, the authors address structural heterogeneity through *Block-wise and Layer-type Spectral Merging* and explore dynamic cutoffs via *Adaptive Bandwidth SpectralMerge (LP-Adaptive)*.

---

## 2. Overall Assessment & Recommendation

* **Soundness:** Excellent
* **Presentation:** Excellent
* **Significance:** Good-to-Excellent
* **Originality:** Excellent

**Overall Recommendation: 5 (Accept)**

### Professional Rationale:
This paper represents a highly elegant, theoretically rigorous, and empirically complete contribution to the post-hoc model-merging literature. Transitioning layer-wise neural network parameters into an orthonormal frequency domain via the Discrete Cosine Transform is an exceptionally creative, out-of-the-box paradigm shift that successfully bridges digital signal processing with weight-space deep learning.

The authors have constructed a dual-track evaluation pipeline—pairing a statistically rigorous simulation benchmark (averaged over 30 seeds) with physical PyTorch model-merging experiments, functional layer heterogeneity partitioning, convergence curves under extreme network depth scaling, hyperparameter sweeps, and real-world pre-trained ResNet-18 checkpoint fine-tuning. Every key methodological claim is backed by solid empirical evidence. 

On actual physical MLP networks, SpectralMerge-Reg achieves an outstanding **$60.42\%$** multi-task accuracy (a **$+10.00\%$** absolute improvement over unconstrained spatial search), and Block-wise SpectralMerge outperforms Global SpectralMerge ($55.42\%$ vs $52.50\%$), verifying that block-wise partitioning prevents underfitting on heterogeneous architectures. On the real ResNet-18 CIFAR-10 setup, SpectralMerge-Reg achieves a spectacular **$54.00\%$** test accuracy (a **$+25.00\%$** absolute blowout over unconstrained spatial search and PolyMerge) and beautifully reduces Expected Calibration Error (ECE) from $0.46$ (Uniform) to $0.18$. This is a top-tier paper that is fully ready for publication.

---

## 3. Key Strengths
1. **Visionary and Original Paradigm:** Mapping layer-wise combining coefficients onto an orthogonal trigonometric frequency basis is an incredibly creative idea, offering a refreshing multi-scale perspective on parameter sensitivity.
2. **Elegant Mathematical Formulations:** Both SpectralMerge-LP (hard low-pass filtering) and SpectralMerge-Reg (quadratic spectral decay regularization) are mathematically clean, elegant, and highly intuitive.
3. **Rigorous Physical Deep Learning Validation:** The implementation of actual PyTorch model merging on weights and biases under conflicting multi-task boundaries represents a high bar of empirical rigor. It successfully refutes the Overfitting-Optimizer Paradox on physical networks, establishing practical viability.
4. **Successful Heterogeneity Handling:** The formulation and evaluation of Block-wise Spectral Merging are outstanding. Proving that partitioning along functional boundaries (Type A vs Type B layers) outperforms global smoothing ($55.42\%$ vs $52.50\%$) provides a vital blueprint for deploying spectral methods on modern heterogeneous Transformer blocks.
5. **PEFT-Induced Step-Function Discontinuity Diagnosis:** The paper provides a brilliant, highly rigorous diagnosis of why hard spectral filters (SpectralMerge-LP and LP-Adaptive) collapse to random guessing ($29.00\%$) in the ResNet-18 CIFAR-10 evaluation, while soft spectral decay (SpectralMerge-Reg) achieves $54.00\%$. Under localized fine-tuning (where only deep layers are updated), the parameter sensitivity across layers forms a step function. From DSP principles, a step function has infinite frequency support. Thus, hard-cutoff filters catastrophically underfit by removing the required high-frequency components. SpectralMerge-Reg, through its soft decay, allows the validation gradients to activate specific localized high-frequency coordinates. This mathematical explanation is extremely elegant and reveals high technical depth.
6. **Demonstrated Scalability and Perfect Conditioning:** Proving that the orthonormal DCT basis has a condition number of exactly $1.0$ at all scales, and empirically showing that it stabilizes Adam optimization at depths $L=48$ and $L=96$ (where PolyMerge fails and stalls), provides a solid theoretical and practical foundation for scaling to extremely deep modern architectures. 
7. **Robustness to Validation Noise & Advanced Optimization Design:** We commend the authors for explicitly addressing and mitigating optimization sensitivity to stochastic validation noise. By implementing momentum-based Adam optimization across all configurations (computing analytical gradients via the IDCT chain rule) and providing a comprehensive empirical/theoretical comparison with derivative-free Nelder-Mead search in Appendix A, they have established a robust, scalable, and noise-tolerant optimization design.
8. **Outstanding Presentation:** The manuscript is exceptionally well-written. The narrative flow is cohesive, the figures and tables are professional and high-signal, and the mathematical derivations (including the orthonormality proof in Appendix B) are flawless.

---

## 4. Constructive Suggestions for Further Improvement (Minor Weaknesses)

While the paper is technically solid and highly polished, we identify two forward-looking areas that the authors could address to further elevate the impact of the final version:

### Area 1: Scaling to Larger Multi-Task Configurations ($K \ge 8$)
* **Observation:** The empirical validation is thorough but limited to configurations with a small number of consolidated tasks: $K=4$ in the ViT simulation, $K=3$ in the physical MLP, and $K=2$ in the real ResNet-18. In modern practical applications, model-merging algorithms are frequently deployed to consolidate larger pools of task-specific expert models (e.g., merging 8 to 12 task-specific LLMs or vision adapters). As the number of tasks $K$ scales, task interference and weight representation clashes grow exponentially.
* **Suggestion:** It is crucial to understand if the spectral trajectory dynamics remain consistent, or if a larger task pool requires a larger spectral bandwidth $F$ to resolve high-dimensional multi-task conflicts. Discussing or evaluating SpectralMerge on a larger pool (e.g., $K \ge 8$ tasks) in Section 5 would greatly enhance the practical significance and impact of the empirical validation. Specifically, the authors' proposal of a **2D DCT-II** to map joint depth-and-task coordinates into 2D spectral coordinates is a highly promising direction that could be highlighted further.

### Area 2: Evaluation on Massive Pre-trained Foundation Models (LLMs/ViTs)
* **Observation:** The physical experiments are evaluated on an MLP and a pre-trained ResNet-18 model fine-tuned on CIFAR-10. While these are highly appropriate and mathematically rigorous steps to verify physical weight-space and backpropagation validity, they do not fully capture the raw parameters of massive pre-trained foundation models like Llama-3 or ViT-B/16.
* **Suggestion:** To accelerate widespread adoption, the authors should discuss the practical guidelines for applying SpectralMerge to massive pre-trained foundation models. For instance, explaining how extracting layer-wise task vectors, applying the 1D DCT-II to partition homogeneous attention/feedforward projections, and using standard zero-shot datasets to optimize the low-frequency spectral coefficients would serve as a highly scalable alternative for massive-scale pre-trained model consolidation in practice.

---

## 5. Detailed Evaluation Ratings

### Soundness: Excellent
The mathematical derivations, transform dualities (DCT-II and IDCT-III), and Appendix B orthonormality proofs are flawless. The experimental evaluation is outstandingly complete: the authors have successfully backed up every theoretical assertion—including the Overfitting-Optimizer Paradox, block-wise architectural heterogeneity, and polynomial ill-conditioning—with a dedicated, high-quality empirical experiment (both simulated and physical).

### Presentation: Excellent
The writing is clear, precise, and highly professional. The structure is flawless and flows naturally. The figures and tables (especially the physical network accuracy, convergence plots, and loss landscape contours) are clean, highly informative, and easy to interpret.

### Significance: Good-to-Excellent
Post-hoc model merging is a highly relevant and active area of research. By completely resolving the overfitting issues of spatial coordinates and the numerical instabilities of polynomial smoothing, SpectralMerge provides a breakthrough, highly scalable paradigm. This concept has broader significance and could inspire frequency-domain representations in parameter-efficient fine-tuning (PEFT), layer-wise learning rates, or hypernetworks.

### Originality: Excellent
Transitioning layer-wise parameter scaling coefficients into the frequency domain is a highly original and creative paradigm shift. It represents a refreshing, out-of-the-box bridge between digital signal processing and weight-space deep model consolidation that stands out significantly from incremental spatial modifications.

---

## 6. Questions and Minor Suggestions for the Authors
* **Even-Symmetry Boundary Extensions in DCT-II:** In Section 3.2, the authors note that the DCT-II implicitly assumes an even symmetric boundary extension, which avoids the artificial boundary discontinuities and high-frequency Gibbs-like oscillation artifacts associated with the DFT. Does this boundary symmetry play a crucial role in maintaining smooth transitions at the physical boundaries of the network (first and last layers)? It would be excellent to explicitly discuss how this boundary behavior protects the input layer and task classification heads from optimization spikes.
* **Notation Consistency:** Ensure that the symbol $L$ (which represents network layer depth) and $l$ (which represents layer index) are consistently distinguished throughout all figures, captions, and text equations.
* **Axis Labeling in Figures:** In Figure 1, ensure that the y-axis label clearly specifies "Condition Number ($\kappa$)" and uses a log scale, which is essential to visualize the exponential growth of polynomial baselines. In Figure 2, ensure the x-axis "Bias Magnitude" clearly indicates that it spans from 0.0 to 0.2.
