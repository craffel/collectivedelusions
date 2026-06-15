# Critical Evaluation of the Experimental Results

## Experimental Setup and Datasets
The authors set up a controlled evaluation environment:
- **Architecture:** Vision Transformer (\texttt{vit\_tiny\_patch16\_224}, 5.7M parameters), representing a modern but highly compact transformer backbone.
- **Datasets:** A 4-dataset multi-domain visual classification benchmark: MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Critical Review of Setup:** 
  The datasets selected are extremely basic, low-resolution toy/small-scale classification benchmarks. While they represent a highly controlled and computationally efficient "sandbox" for massive hyperparameter sweeps (which the authors leverage well), they are not representative of modern practical model-merging scenarios. In practice, weight-space merging is primarily applied to large-scale generative foundation models (such as LLMs with 7B+ parameters or CLIP-ViT-B/16) where parameter redundancy and feature representation dynamics are vastly different. The conclusions drawn from a tiny 5.7M parameter model on MNIST/CIFAR-10 may not generalize to billions of parameters, representing a significant limitation in the scope of the experimental setup.

---

## Evaluation of Baselines
The paper compares the proposed method against a comprehensive set of baselines:
1. **Naive Uniform TA & Optimized TA:** Foundational baselines that establish the performance of standard, unregularized Task Arithmetic.
2. **TIES-Merging & DARE-Merging:** Contemporary, state-of-the-art model merging baselines that also incorporate sparsification.
3. **Decoupled Prune-then-Merge (P-then-M):** An essential magnitude-based baseline.
4. **Layer-Group Scaling (L-Scale):** A baseline with 125 degrees of freedom to verify if flexible scaling alone can prevent representational collapse without sparsification.
5. **Fisher-Weighted Averaging:** A classic first-order curvature-based baseline.
6. **Joint Multi-Task Learning (MTL):** The ultimate multi-task training upper bound, and **Dense Experts (Ceiling)** which represents individual model performance.
- **Critical Review of Baselines:**
  The baseline coverage is exemplary. The inclusion of Layer-Group Scaling is highly effective as a control experiment, showing that purely adjusting scaling factors per layer group (achieving only $32.44\%$) fails completely without magnitude pruning. This strongly supports the hypothesis that weight-space sparsification is the primary driver of successful model merging. The inclusion of a physical Joint MTL baseline also provides a highly rigorous, honest upper-bound for the benchmark.

---

## Do the Results Support the Claims?

Let's critically evaluate each major claim against the reported results:

### Claim 1: "Sparsity-Guided Task Arithmetic (SG-TA) is a remarkably simple and deterministic weight-space regularization framework... filtering out harmful orthogonal parameter noise."
- **Support Status: Well-Supported.**
- **Evidence:** Table 1 shows that Naive Uniform TA suffers from catastrophic representational collapse ($46.32\%$ accuracy), while applying SG-TA (GQ) raises the Joint Mean Accuracy to $61.40\%$. The baseline comparisons and the cosine similarity analysis (showing pruned weights have extremely low pairwise similarities of 0.0099 to 0.0169) strongly support the hypothesis that pruning low-magnitude task vectors removes orthogonal noise and regularizes the weight space.

### Claim 2: "Global Budget Flexibility is Crucial... Global Quantile (GQ) masking consistently and substantially outperforms Layer-wise Quantile (LQ) masking."
- **Support Status: Well-Supported.**
- **Evidence:** Under identical calibration sweeps, SG-TA (GQ) achieves $61.40\% \pm 1.39\%$, while SG-TA (LQ) achieves $57.81\% \pm 2.52\%$ (a $+3.59\%$ absolute gap). The keep-ratio sensitivity analysis (Table 2 and Figure 1) shows that GQ consistently maintains a higher multi-task capability than LQ at all optimal keep-ratios (e.g., $60.11\%$ vs. $55.06\%$ at $k=0.3$). This confirms that letting key layers retain more active updates is superior to enforcing a rigid, homogeneous layer budget.

### Claim 3: "Superiority Over Complex Baselines... outperforming more complex baselines like TIES-Merging and DARE-Merging."
- **Support Status: Weakly Supported (No Statistical Significance).**
- **Evidence:** The reported Joint Mean Accuracies are:
  - SG-TA (GQ): $61.40\% \pm 1.39\%$
  - TIES-Merging: $60.64\% \pm 1.30\%$
  - DARE-Merging: $58.44\% \pm 3.02\%$
  While SG-TA (GQ) is numerically superior, the difference between SG-TA (GQ) and TIES-Merging is only **$0.76\%$ absolute**, which is well within their overlapping standard deviations ($\pm 1.39\%$ and $\pm 1.30\%$). 
  - *Analysis:* The authors are commended for their scientific honesty in explicitly stating: *"we note with scientific honesty that because of overlapping standard deviations, our method's superiority over TIES-Merging is not statistically significant."* However, from a peer-review perspective, this means the claim that SG-TA "outperforms" TIES-Merging is not rigorously supported by the evidence. It is more accurate to say that the proposed method performs comparably to TIES-Merging, which calls into question whether omitting sign-consensus actually "improves" performance or if it simply represents a comparable simplification.

### Claim 4: "Task Vector Magnitude Normalization (TV-Norm) successfully addresses magnitude imbalance."
- **Support Status: Well-Supported.**
- **Evidence:** Incorporating TV-Norm (SG-TA GQ-Norm) dramatically increases MNIST accuracy from $36.74\%$ to $53.70\%$ ($+16.96\%$ absolute increase) and CIFAR-10 from $67.82\%$ to $68.84\%$, while dampening the dominance of SVHN from $85.35\%$ to $70.18\%$. This strongly supports the claim that normalizing task vectors prior to masking balances the joint representation and prevents magnitude-dominant tasks from overwriting others.

### Claim 5: "Coordinate Search (CS) is a highly practical and scalable non-uniform calibration alternative."
- **Support Status: Well-Supported.**
- **Evidence:** Non-Uniform Coordinate Search optimizes task-specific keep-ratios $k_i$ and scaling factors $\alpha_i$ in linear time $\mathcal{O}(T)$, requiring only 100 model evaluations (43.61s) compared to the intractable exponential grid search ($\mathcal{O}(P^T) = 1.29 \times 10^7$ evaluations). It achieves $58.40\% \pm 2.32\%$ Joint Mean while dramatically rebalancing task representation (boosting MNIST to $50.38\%$ and CIFAR-10 to $75.04\%$), validating its utility as an efficient optimization strategy.
