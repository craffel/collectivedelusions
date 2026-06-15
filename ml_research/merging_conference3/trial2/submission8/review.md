# Mock Peer Review

## Summary of the Paper
This paper addresses the critical bottleneck of deploying multiple specialized multi-task expert models on storage- and network-constrained edge and IoT devices. The authors propose **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a post-hoc weight sparsification and merging framework. It introduces two deterministic magnitude-based pruning schemes: global **Uniform Pruning (NP-BTVP-U)** and layer-wise **Adaptive Saliency-Based Pruning (NP-BTVP-S)**. To prevent the update norm shrinkage that typically causes sparse models to collapse back to the base model performance, both methods incorporate a **norm-preserving rescaling factor** (scaling active updates by $1/p$ or local $1/p_l$) as a deterministic signal-strength preservation heuristic.

The authors evaluate their framework using a pre-trained CLIP ViT-B/32 backbone across 4 diverse vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using 3 independent random seeds. Under a tight 90% sparsification budget ($p=0.10$), NP-BTVP-U achieves **90.34%** (AdamW) and **90.32%** (SAM) average accuracy, performing competitively with the advanced stochastic DARE-Merging baseline while outperforming TIES-Merging by **3.81%** at half the parameter budget. 

Crucially, the paper reports two highly valuable insights for the model-merging community:
1. **Geometric Separation of Flatness and Sparsification:** Contrary to the common hypothesis that training-stage loss-landscape flatness (via SAM) provides a coordinate-aligned sparsification buffer, standard AdamW and SAM experts show nearly identical and extraordinary levels of resilience to heavy post-hoc pruning when paired with norm-preserving rescaling.
2. **The Saliency Double-Bind:** Saliency-based budget allocation (NP-BTVP-S) is slightly outperformed by global Uniform Pruning (NP-BTVP-U) due to a trade-off between inter-layer scale distortion (under global scaling) and local noise amplification (under layer-wise scaling). The authors therefore recommend the simpler, global Uniform Pruning as the most stable, robust, and pragmatically superior edge-merging solution.

---

## Strengths
1. **High Practical Relevance & Actionability:** The paper focuses on direct, real-world deployment constraints on edge/IoT hardware. Compressing task vectors by 90% to 95% post-hoc allows storing delta updates in compressed sparse formats (like CSR) with zero runtime latency or FLOP overhead, translating to a massive **5x to 10x reduction in raw storage footprint** (reducing a 115 MB CLIP expert to ~23 MB).
2. **Training-Free & Zero-Overhead:** The proposed framework is completely post-hoc and runs in milliseconds (relying on simple magnitude sorting). It avoids the massive computational and memory overhead of calculating second-order curvature matrices (Hessian/Fisher) or fragile calibration optimization, making it incredibly easy to integrate into existing pipelines.
3. **Intellectual Honesty and Scientific Integrity:** The authors are highly commended for reporting "negative" results with exceptional scientific clarity. Discovering that SAM does not provide an additional coordinate-aligned pruning buffer compared to AdamW, and showing that their proposed Saliency Pruning is slightly outperformed by global Uniform Pruning due to the *Saliency Double-Bind*, represents a refreshing and high-signal contribution to the community.
4. **Rigorous Evaluation & Reproducibility:** Conducting all experiments across 3 independent random seeds on a diverse 4-dataset test suite, using state-of-the-art baselines (TIES, DARE) and fully disclosing all hyperparameters (epochs, learning rates, SAM perturbation radius), ensures outstanding scientific validity and ease of reproducibility.
5. **Strong Theoretical Underpinnings:** The analytical derivations of the expected L1 update norm under Laplace and Gaussian distribution assumptions in Appendix Section A are elegant and mathematically rigorous, clearly justifying the $1/p$ scale factor as a beneficial signal-strength boost.

---

## Weaknesses
1. **Misnomer of "Norm-Preserved" in the Title:** While the title and framework are named "Norm-Preserved", the authors' mathematical derivations in the Appendix show that applying the $1/p$ scale factor to deterministically pruned coordinates actually *amplifies* the expected $L_1$ update norm by $2.58\times$ (Gaussian) to $3.30\times$ (Laplace). The authors honestly characterize this as a "Signal-Strength Boost" that is empirically superior to strict $L_1$-preserving factors, but the term "Norm-Preserved" remains mathematically a bit of a misnomer.
2. **Scale of Backbones and Datasets:** The evaluation focuses on a CLIP ViT-B/32 backbone across 4 relatively small visual classification datasets. While these are diverse, the model-merging and weight sparsification community is heavily focused on Large Language Models (LLMs) and massive vision models (e.g., LLaMA, OPT, ViT-L). Demonstrating the generalizability of the deterministic rescaling framework on LLM instruction datasets would significantly increase the paper's reach, though the authors do provide a scalability discussion in Appendix Section C.
3. **Main-Text Integration of Quantization:** The authors mention preliminary results combining 10% sparse task vectors with INT8 quantization in Appendix Section E, showing an additional 4x storage saving with only a 0.12% accuracy drop. This is an incredibly powerful practical result that represents a massive win for edge deployment, but it is currently buried in the Appendix.

---

## Detailed Evaluation Ratings

### Soundness: Excellent
The paper's claims are fully backed by rigorous empirical evaluations (3 seeds, 4 datasets, state-of-the-art baselines) and sound mathematical derivations. The ablation study isolated and conclusively proved that the norm-preserving scale factor is the critical enabler of the framework's success (preventing a 9.4% to 9.8% accuracy drop). The choice of magnitude pruning is well-justified against second-order methods.

### Presentation: Excellent
The writing is exceptionally clear, logical, and concise. The structure is clean and easy to follow. The mathematical equations are fully defined, and the plots (Figures 1 and 2) are high-quality, readable, and directly support the text. The related work is thorough and positions the contributions accurately.

### Significance: Excellent
The paper addresses an important, highly relevant problem in machine learning deployment. Providing a training-free, robust, and zero-overhead post-hoc weight sparsification framework that achieves 90% to 95% compression with minimal performance degradation offers immediate, practical utility to edge AI practitioners.

### Originality: Good
The work offers a novel combination of existing techniques (magnitude pruning, task arithmetic, and reciprocal scale factors) and deepens our understanding of weight-space geometry. The formulation of the Saliency Double-Bind and the geometric separation of loss-landscape flatness and sparsification represent genuine, high-quality insights.

---

## Overall Recommendation

**Recommendation: 5: Accept**

This is a technically solid, highly practical, and scientifically honest paper. It offers actionable, low-overhead solutions to a major bottleneck in edge intelligence while providing deep, counter-intuitive geometric insights to the model-merging community. The paper meets the bar for publication and is ready for acceptance, subject to addressing the minor suggestions below.

---

## Actionable Feedback & Minor Suggestions

1. **Clarify the "Norm-Preserved" Terminology:** 
   Add a brief sentence in the Introduction or Section 3.3 explicitly acknowledging that while the framework is conceptually motivated by preventing norm shrinkage (hence the name "Norm-Preserved"), the $1/p$ scale factor actually amplifies the expected $L_1$ update norm mathematically (as derived in Appendix A). Framing this clearly as a "Signal-Strength Boost" in the main text will prevent any confusion.
2. **Move Quantization Highlights to the Main Text:**
   The synergy between the 10% sparse task vectors and INT8 quantization reported in Appendix Section E is a major highlight for real-world practitioners (achieving an additional 4x storage reduction with a negligible 0.12% accuracy drop). It is highly recommended to mention or summarize this joint sparsification-quantization result in the main text (e.g., in Section 4.3 or Section 4.5) to elevate the paper's practical impact.
3. **Discuss Potential LLM Generalizability in the Main Text:**
   While the scalability discussion in Appendix Section C is insightful, briefly highlighting in the main conclusion or introduction how the Saliency Double-Bind is expected to play out in Large Language Models (LLMs) would help contextualize the relevance of global Uniform Pruning (NP-BTVP-U) to a broader audience of NLP and LLM practitioners.
4. **Correction of Minor Typos:**
   * In `reviewing_plan.md` and some of the internal files, there is a minor typo ("glease use sparingly" instead of "please"). Ensure all LaTeX text is checked for minor typos before final publication.
