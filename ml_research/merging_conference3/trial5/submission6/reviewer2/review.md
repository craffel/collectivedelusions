# Conference Peer Review

## Strengths and Weaknesses

### Strengths
1. **Exceptional Practical Utility and Real-World Relevance:** The paper targets a critical, unresolved bottleneck in weight-space model merging: the batch-dependency and "heterogeneity collapse" of existing dynamic routers. Solving this is extremely important for edge computing, where real-time streaming inputs often arrive as highly heterogeneous, mixed-task batches. 
2. **Deterministic, Batch-Independent Inference:** By shifting from weight-reconstruction to sample-wise activation-space routing, SLD-Merge guarantees that a sample's downstream prediction remains completely deterministic and independent of other samples in the batch. This is a crucial safety and reliability requirement for mission-critical deployments.
3. **Outstanding Parameter and Compute Efficiency:** 
   * **92.5% reduction in extra task-specific parameter storage** (requiring only 0.295M additional parameters for 4 experts instead of 3.96M).
   * **37.9% overall memory reduction** (reducing total parameter footprint from 9.66M to 5.99M).
   * **Minimal computational overhead** (adding only **8.3% FLOPs** over a single static model via Top-1 hard gating), making it highly viable for hardware-constrained edge CPUs, NPUs, and microcontrollers.
4. **Pragmatic, Training-Free Integration:** The proposed **Activation-Space Mean Initialization** is a highlight. It achieves robust routing completely zero-shot, bypassing complex, unstable gradient-descent calibration (e.g., Gumbel-Softmax, RL-based routing) and allowing rapid, on-device adaptation to new local domains without any backpropagation overhead.
5. **Autonomy and Lack of Information Leakage:** Introducing **Autonomous Classification Head Selection** ensures that the model routes inputs to the correct classification head based entirely on internal activation statistics, eliminating the privileged oracle-routing assumption common in prior model merging studies.
6. **Scholarly Rigor and Analytical Depth:** The paper provides brilliant, highly original insights, such as empirically demonstrating that SVD low-rank truncation acts as a heavy implicit regularizer in data-scarce settings—actually outperforming full-rank models by filtering out training noise and overfitting artifacts in the base experts.

### Weaknesses
1. **Evaluations Restricted to Smaller, Classic Benchmarks:** While the authors successfully defend their dataset choice as a realistic low-shot streaming stress-test, the empirical evaluation is limited to subsampled, classic datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Evaluating on larger, more complex datasets (such as DomainNet subsets, VTAB, or NLP benchmarks) would strengthen the generalizability claims.
2. **Small Backbone Model Scale:** The experiments are conducted on a `vit_tiny_patch16_224` backbone (5.7M parameters). Modern industrial applications of model merging focus on much larger models (e.g., LLaMA, ViT-Giant). Empirical validation on a larger-scale model would increase the work's practical significance.
3. **Absence of Full-Network Merging Verification:** The paper's experiments freeze early layers (blocks 0--8) and specialize only late layers (blocks 9--11). It would be highly valuable to include empirical results for merging fully fine-tuned experts (with routers active across all 12 blocks) to demonstrate how routing accuracy and multi-layer representation shift scale when no layers are frozen.
4. **Deep Analysis of Jitter and Routing Disagreements:** The authors mention a high cross-layer routing agreement rate of **96.48%**. However, they do not explore the downstream consequences of the remaining 3.52% of samples where layer-wise routers disagree (e.g., layer 9 routes to expert A, but layer 10 routes to expert B).

---

## Soundness
**Rating: Excellent**

**Justification:**
The proposed methodology is technically sound and mathematically rigorous. Offline SVD factorization is a robust and optimal method for low-rank approximation under the Frobenius norm. Shifting the element-wise routing to the activation space in a parallelized, vectorized PyTorch forward pass is highly elegant and guarantees sample-wise isolation. Furthermore, the experimental results thoroughly support the core claims: the joint test accuracy of SLD-Merge remains perfectly flat (63.87%) across all batch sizes, verifying complete batch-independence. The authors are also highly transparent and honest about the "soft collapse" of baselines to the static uniform merging floor (attributing it to the pre-trained ViT capacity buffer), which indicates a high degree of academic integrity.

---

## Presentation
**Rating: Excellent**

**Justification:**
The submission is exceptionally well-written, clear, and logically structured. The narrative is cohesive, tracking from a strong, well-illustrated introduction of "heterogeneity collapse" (Figure 1) to a detailed mathematical formulation and a thorough discussion of results. The figures (especially Figure 2's pipeline overview) and tables are professional, descriptive, and seamlessly integrated into the text. All hyperparameters, datasets, and training details are exhaustively documented, making the work highly reproducible.

---

## Significance
**Rating: Excellent**

**Justification:**
From a deployment and systems engineering perspective, this paper addresses an extremely important and pressing problem in multi-task learning. Traditional dynamic merging methods are unviable in high-frequency production streaming systems because they suffer from performance collapse when processing heterogeneous batches and require rebuilding huge dense weight matrices in memory on the fly. SLD-Merge offers a highly practical, memory-efficient, and computationally cheap alternative that makes weight-space expert merging genuinely viable for real-world, on-device applications.

---

## Originality
**Rating: Excellent**

**Justification:**
The core contribution—shifting dynamic model merging from weight reconstruction to sample-wise activation-space routing of SVD adapters—is highly creative and original. While SVD and mixture-of-experts are independently known in parameter-efficient fine-tuning, their post-hoc synthesis to solve the specific issues of batch-dependency and heterogeneity collapse in model merging is highly innovative. Bounded cosine-similarity routing and zero-shot activation-space mean calibration are also elegant and original solutions to routing regularisation and non-differentiable gating.

---

## Overall Recommendation
**Rating: 5: Accept**

**Justification:**
This is a technically solid, highly practical paper with significant real-world utility. It resolves a major deployment hurdle in dynamic model merging, enabling deterministic, batch-independent multi-task inference with minimal RAM and storage footprint. The writing is outstanding, the methodology is mathematically sound, and the evaluation is highly comprehensive (featuring extensive ablations, variance characterizations, and autonomous selection tests). While expanding the experiments to larger models and datasets would elevate the submission further, the paper's highly useful engineering contributions, solid performance, and training-free calibration make it an exceptionally strong addition to the conference program.
