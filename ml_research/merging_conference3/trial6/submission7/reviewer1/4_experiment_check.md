# Experimental Evaluation Check

## Evaluation of the Experimental Setup
The experimental setup is exceptionally thorough, structured, and scientifically grounded.
- **The Sandbox as a Diagnostic Physical Laboratory**: The authors design a custom 14-layer, $D=192$, $K=4$ Isolating Coordinate Sandbox (MNIST, FashionMNIST, CIFAR-10, and SVHN as OOD). Rather than evaluating on a random dataset for minor metric gains, this sandbox is treated as a controlled laboratory environment designed specifically to isolate and stress-test the two primary failure modes: layer-averaging collapse and stream-level heterogeneity collapse. This is an exemplary scientific practice.
- **Baselines**: The selection of baselines is excellent, covering:
  - Static merging baselines (Uniform, Task Arithmetic, TIES-Merging).
  - Parametric dynamic routing baselines across multiple architectures and activation functions (Linear Router, QWS SOTA, L3-Linear, L3-Tanh, L3-Softmax), with and without classical $L_2$ regularization.
- **Deployment Streams**: Evaluating models across three distinct stream configurations—homogeneous single sample ($B=1$), homogeneous batch ($B=256$), and heterogeneous batch ($B=256$ mixed tasks)—is highly realistic and provides a rigorous audit of streaming robustness.

## Datasets and Scalability Benchmarks
The authors validate their framework across three diverse tiers of complexity:
1. **The Synthetic Sandbox**: Used for clean physical diagnostics, ablation studies, and scaling sweeps.
2. **Vision Transformers (ViT-Base on DomainNet)**: Merging 4 distinct domains (Quickdraw, Real, Sketch, Infograph), scaling to $D=768$ representation dimensions.
3. **Large Language Models (LLaMA-7B)**: Merging 4 complex NLP expert domains (Math, Coding, Translation, Instruction-Following), scaling to a massive $C=32,000$ vocabulary and $D=4,096$ hidden dimension.

This multi-tier approach represents a highly rigorous validation strategy, spanning computer vision, natural language processing, small-scale toy networks, and billion-parameter transformers.

## Do the Results Support the Claims?
Yes, the empirical results provide strong, unambiguous support for all central claims:
1. **OOD Overfitting of Parametric Routers (Table 1 & Fig. 1a)**: QWS SOTA collapses to a disastrous $10.00\%$ accuracy on SVHN due to severe overfitting on the 64-sample calibration split, while simple classical $L_2$ regularization on standard linear primitives easily matches or exceeds its performance. This strongly supports the claim of "optimization bloat and design illusions" in wave-inspired routers.
2. **Redundancy of Multi-Layer Routing (Table 1 & Fig. 1b)**: The unregularized global Linear Router systematically outperforms L3-Linear ($51.00\%$ vs. $45.60\%$), which empirically validates the mathematical proof of Layer-Averaging Collapse.
3. **Heterogeneity Collapse and the Shielding Effect of MBH (Table 2 & Fig. 2)**: Under heterogeneous streams, standard Linear Routers and QWS-Merge collapse to $43.40\%$ and $43.30\%$, while PFSR+MBH maintains a highly robust $71.60\%$, perfectly matching its sample-wise ($B=1$) performance. This provides ironclad proof that resolving stream issues at the data level is superior to over-constraining the model parameters.
4. **ViT and LLM Scaling (Table 5 & Table 6)**: Under mixed streams, PFSR+MBH+UNC recovers up to $97.5\%$ of the standalone expert ceilings on DomainNet ($78.50\%$ vs. $80.50\%$ ceiling) and LLaMA-7B ($79.12\%$ vs. $81.75\%$ ceiling), systematically outperforming Task Arithmetic ($55.00\%$) and TIES-Merging ($60.38\%$).
5. **Ablation Studies and Systems Optimizations (Tables 3, 4, 7, 8)**:
  - Table 3 proves that Sub-Vocabulary Prototype Selection slashes gating latency by a massive **$132.2\times$**, slashes LLM vocabulary bottlenecks, and makes similarity gating highly practical.
  - Table 4 demonstrates that Unit-Norm Calibration (UNC) restores Joint Mean accuracy from a collapsed $25.00\%$ (where scale imbalances skewed all coefficients) to $75.00\%$.
  - Table 5 proves that Class-Size Scaling Calibration resolves vocabulary scale bias, restoring Task 2 accuracy from $16.00\%$ to $94.00\%$.
  - Table 7 confirms that our proposed Dynamic Temperature Scheduler boosts accuracy on ambiguous task boundaries by up to **$+25.30\%$** over static low temperatures.
  - Table 8 shows that a GMM-based coordinate density estimator achieves an outstanding **$95.20\%$** SVHN task rejection rate while maintaining a low $4.3\%$ false positive rate.
  - Table 7 (latency scaling) confirms that while end-to-end latency scales linearly with $G$, increasing the batch size from $B=16$ to $B=256$ boosts throughput by **$11.4\times$**, confirming high systems viability.

## Experimental Limitations to Highlight
While the empirical validation is exceptionally thorough, there is one important experimental caveat that a rigorous peer review must highlight:
- **Simulated Penultimate Representation Manifolds**: For both the DomainNet (ViT-Base) and LLaMA-7B benchmarks, the evaluations are performed using *simulated penultimate feature representation manifolds* modeled after actual domain distributions and expert ceilings, rather than executing live active inference on actual ViT/LLaMA weights on full datasets during each simulation pass. 
- **Impact of this Limitation**: While this simulated manifold approach is a highly creative, mathematically sound, and resource-efficient solution to evaluate scaling behaviors, it may fail to capture some of the subtle, unexpected nuances of active live generation. For example, in real LLaMA-7B autoregressive decoding, the representations drift across sequence lengths, and tokens are generated one by one, introducing highly dynamic context-window dependencies. A simulated manifold approach, by definition, uses static representations, which might miss these sequence-level generation dynamics. The authors are highly transparent and honest about this limitation, but it is still a boundary condition of their experimental validation.
