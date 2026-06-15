# Peer Review of Conference Submission

## Summary of the Paper
This paper addresses the storage and transmission bottlenecks of deploying specialized multi-task expert models on edge and IoT devices. The authors propose **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a training-free weight sparsification and merging framework. The approach extracts task-specific updates (task vectors) and prunes them using magnitude-based thresholding. Crucially, the authors incorporate **norm-preserving rescaling** (scaling the remaining active updates by the reciprocal of the retention rate, $1/p$ or $1/p_l$) to prevent update norm shrinkage, which they frame as a "signal-strength boost" to ensure that task vectors are not drowned out by the pre-trained base model weights during multi-task fusion.

The paper evaluates two main schemes: global Uniform Pruning (NP-BTVP-U) and layer-wise Adaptive Saliency-Based Pruning (NP-BTVP-S). The empirical evaluation is conducted across 3 seeds on a CLIP ViT-B/32 backbone using 4 image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) in a low-data regime (1024 samples per task). 

The key findings include:
1. Loss landscape flatness (via SAM experts) does not inherently provide an additional coordinate-aligned pruning buffer compared to standard AdamW experts under well-converged regimes.
2. NP-BTVP-U achieves highly competitive results, retaining up to 90-95% sparsity with minimal accuracy loss and outperforming TIES-Merging by 3.81% at half the parameter budget under SAM.
3. Saliency-Based Pruning (NP-BTVP-S) is trapped in a "Saliency Double-Bind" trade-off between scale distortion and local noise amplification, making the simpler Uniform method the superior and more stable choice.

---

## Strengths and Weaknesses

### Major Strengths
1. **Exceptional Empirical Rigor and Scientific Transparency:** The authors report means and standard deviations across 3 independent random seeds, sweep and optimize merging coefficients individually for all baselines to ensure fairness, and conduct a two-tailed paired $t$-test to verify the statistical significance of their findings. Furthermore, they are highly honest about their findings—clearly explaining why "Norm-Preserving" scaling actually amplifies the expected $L_1$ norm (Appendix A), reporting when the Saliency-Based method fails or behaves indistinguishably from the Uniform baseline, and highlighting the collapse of layer-wise scaling under quantization (Appendix E).
2. **Deep Diagnostic and Geometric Analysis:** Rather than just introducing a method, the paper provides insightful analysis, such as demonstrating that optimization-stage curvature flatness (SAM) does not inherently translate into coordinate-wise sparsification resilience, and dissecting the trade-offs of the "Saliency Double-Bind."
3. **High Practical Utility for Edge AI:** The proposed NP-BTVP-U is training-free, has extremely low computational complexity, and achieves up to 95% sparsification, translating directly to a 5-20$\times$ storage reduction in compressed formats (COO/CSR) on resource-constrained hardware.
4. **Well-Written and Highly Lucid Presentation:** The paper is exceptionally clear, structured, and easy to follow. The mathematical notation is clean and rigorous, and the tables and figures are of high quality.

### Major Weaknesses
1. **Highly Incremental Methodological Originalty:**
   - From an algorithmic standpoint, the core of the proposed framework (NP-BTVP-U) lacks conceptual novelty. It is a straightforward combination of two existing components: global magnitude pruning (which is identical to the "trimming" step of TIES-Merging) and a $1/p$ rescaling factor (which is the exact scaling factor popularized by DARE). Combining these two established operations represents an intuitive engineering refinement rather than a significant, paradigm-shifting conceptual leap.
   - The primary structurally unique contribution introduced in the paper, **Adaptive Saliency-Based Pruning (NP-BTVP-S)**, does not actually work in practice. The paper reports that NP-BTVP-S is slightly outperformed by or statistically indistinguishable from the simple global Uniform baseline (as confirmed by $p$-values of 0.96 and 0.68 under AdamW and SAM, respectively). Consequently, the paper is forced to recommend the simpler, highly standard global Uniform baseline (NP-BTVP-U) as the optimal choice. This means the paper's positive results rely entirely on the highly incremental NP-BTVP-U.
2. **Limited Scale and Dataset Complexity:**
   - The empirical evaluation is restricted to a small-scale visual encoder (CLIP ViT-B/32, 28.7M active parameters) on simple, toy classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). 
   - These classification tasks are extremely easy for a powerful pre-trained CLIP backbone to solve. Because the datasets are small (1024 samples), the task-specific parameter shifts (task vectors) are likely very small, which makes them naturally highly resilient to pruning. It is highly questionable whether these findings, such as the high resilience to 90-95% sparsity or the "Saliency Double-Bind" scale dynamics, would generalize to more complex, high-dimensional visual tasks (e.g., ImageNet classification, object detection) or to more challenging benchmarks in other modalities.
3. **Absence of Large Language Model (LLM) Validation:**
   - Model merging and task arithmetic are most critical and widely adopted in the context of Large Language Models (LLMs) like LLaMA, Mistral, or OPT, due to their immense parameter counts and storage requirements.
   - Both TIES-Merging and DARE-Merging were heavily validated on LLMs across diverse language benchmarks. By limiting the experimental validation to toy image classification datasets on a small ViT, the paper misses a crucial opportunity to prove its utility in the LLM domain where model merging is most vital. Without experimental verification on LLMs, the generalizability and academic significance of the paper's findings remain limited.
4. **Inconsistent Hyperparameter Sweeps in Comparative Baselines:**
   - In Table 3, TIES-Merging and DARE-Merging are evaluated at 80% sparsity ($p=0.20$ and $p_{\text{drop}}=0.80$, respectively), while the proposed Uniform and Saliency methods are evaluated at 90% sparsity ($p=0.10$).
   - While this shows that NP-BTVP-U at 90% sparsity can outperform TIES at 80% sparsity, it prevents a direct, scientifically rigorous head-to-head comparison at identical parameter budgets. All baselines should have been evaluated across the entire sweep of retention budgets ($p \in \{0.05, 0.10, 0.20\}$).

---

## Soundness
* **Rating:** **Good**
* **Justification:** The paper is technically solid, mathematically correct, and empirically rigorous within the scope of its experiments. The authors are highly honest about their methodology, theoretical trade-offs (such as the $L_2$ variance blowup derived in Appendix A.3), and empirical failures (the Saliency Double-Bind). However, the evaluation is limited to a small visual backbone and small classification splits, which prevents rating it as "excellent" due to the limited complexity of the evaluation regime.

## Presentation
* **Rating:** **Excellent**
* **Justification:** The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is clean, and the authors' explanations of their design decisions, limitations, and theoretical derivations are highly lucid and intellectually engaging.

## Significance
* **Rating:** **Fair**
* **Justification:** While the proposed framework offers immediate practical utility for edge vision deployments, its academic and theoretical significance is moderate. The lack of validation on Large Language Models (LLMs)—the primary domain where model merging and task arithmetic are critical—significantly limits the impact of this work on the broader machine learning community.

## Originality
* **Rating:** **Fair**
* **Justification:** The paper falls short of the conference standard for originality. The primary successful method (NP-BTVP-U) is an incremental combination of standard magnitude-based pruning (from TIES) and a reciprocal scaling factor (from DARE). The main structurally novel component proposed (NP-BTVP-S) does not consistently work and is slightly outperformed or statistically indistinguishable from the simple global Uniform baseline. While the diagnostic analysis is interesting, the work does not introduce a paradigm-shifting or highly original conceptual leap.

---

## Overall Recommendation
* **Rating:** **3: Weak reject**
* **Justification:** The paper has clear merits, including its outstanding writing quality, strong empirical rigor, and honest diagnostic insights. However, the weaknesses overall outweigh these merits. Specifically, the core algorithmic contribution is highly incremental (combining existing components from TIES and DARE), and the proposed novel layer-wise allocation scheme (NP-BTVP-S) fails to outperform the simpler baseline. Additionally, the experimental evaluation is limited to toy classification datasets on a small ViT backbone, completely lacking the Large Language Model (LLM) validation that is critical for modern model merging research. These limitations restrict the work's academic novelty and generalizability. The paper requires revisions—such as validating the framework on LLMs and introducing a more conceptually original pruning or scaling scheme—before it can be accepted.

---

## Constructive Feedback and Questions for the Authors
1. **Expand Validation to LLMs:** To significantly increase the significance and impact of this work, please evaluate NP-BTVP-U on Large Language Models (e.g., LLaMA-7B or OPT-1.3B) across standard language tasks (such as GSM8K, ARC, or MMLU). Verifying if the "Saliency Double-Bind" scale dynamics and high sparsification resilience hold in the language domain would provide a major conceptual validation.
2. **Standardize Comparative Sweeps:** Please update Table 3 to evaluate all baselines (TIES-Merging, DARE-Merging, and NP-BTVP-U) across the exact same parameter retention budgets (e.g., $p \in \{0.05, 0.10, 0.20\}$). This head-to-head comparison is essential for complete scientific rigor.
3. **Evaluate on More Challenging Vision Benchmarks:** Instead of simple classification tasks like MNIST and SVHN, please test the framework on more complex, high-dimensional vision datasets (e.g., ImageNet classification, CIFAR-100, or fine-grained classification) where the task vectors represent more substantial and complex parameter shifts.
4. **Develop a More Sophisticated Scaling Heuristic:** Since the $1/p$ scaling factor on the largest absolute updates amplifies the $L_1$ norm and acts as a beneficial signal boost, have you considered learning or dynamically adjusting a coordinate-wise scaling factor based on Hessian or gradient approximations, rather than using a flat $1/p$? This could potentially resolve the scale distortions in layer-wise budget allocation without suffering from the "Saliency Double-Bind."
