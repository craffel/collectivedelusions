# 4. Experiment Check

## Evaluation of the Experimental Setup
The experimental setup is comprehensive and logically designed, utilizing two distinct environments:
1. **Parameter-Space Representation Sandbox**: A synthetic, mathematically modeled proxy of a 14-layer Vision Transformer backbone (\texttt{vit\_tiny\_patch16\_224}) evaluating high-conflict vision domains (MNIST, FashionMNIST, CIFAR-10, SVHN).
2. **Physical CNN Experiments**: An end-to-end physical model merging experiment in PyTorch using standard vision datasets, training four independent SimpleCNN experts (25k parameters each, 4 layer groups) on 8,192 subsampled images per task.

Statistical rigor is ensured by reporting the mean and standard deviation across **3 independent calibration/sampling seeds** in all experiments. While 3 seeds are sufficient to show stability, a larger number of seeds (e.g., 5 or 10) would provide even greater statistical confidence.

## Datasets and Baselines
- **Datasets**: The choice of MNIST, FashionMNIST, CIFAR-10, and SVHN represents a highly challenging "high-conflict" multi-domain setup. Fusing these models directly in parameter space leads to extreme representation conflicts, making it an excellent stress-test for model merging.
- **Baselines**: The baselines are strong and highly appropriate:
  - **Uniform Merge / Task Arithmetic (TA)**: Represents the static baseline.
  - **AdaMerging (SOTA Static)**: Formulated directly on the sandbox features as a strong static competitor.
  - **Linear Router (Classical)**: Standard unregularized Softmax-based global linear routing.
  - **QWS-Merge (SOTA Cosine)**: State-of-the-art dynamic model ensembling.
  - **BL-Router**: Bounded Softmax baseline to isolate scaling ceilings.

## Analysis of Results and Claims Support
1. **Joint Multi-Task Capabilities (Table 1)**:
   - The results support the claim that dynamic routing models outperform static merging (96.20% for Linear Router vs. 72.31% for AdaMerging).
   - They reveal a significant performance gap between Softmax and Sigmoid routing under homogeneous conditions (Softmax-based BL-Router achieves 85.82% while BSigmoid-Router gets 84.70%).
   - The authors successfully support their claim that this gap is driven by conservative scaling ceilings rather than an architectural flaw: removing the scaling ceiling ($\lambda_{\text{max}} = 1.2$) immediately boosts BSigmoid's accuracy to 94.93%. This is a strong and convincing empirical analysis.

2. **Exhaustive Partition Depth Sweeps (Table 2)**:
   - The sweep demonstrates a highly favorable Pareto frontier: at $k=4$, Hybrid-Router achieves 76.75% (+4.44% over static AdaMerging) while saving 71.3% of ensembling latency and VRAM footprint.
   - The results support the "Overfitting-Optimizer Paradox" within the sandbox environment, showing that $k=12$ achieves 84.79% (+0.22% over $k=14$).
   - However, as noted in the soundness evaluation, this paradox is **not observed in the physical CNN experiments**, where performance increases monotonically with $k$. The paper's explanation of this discrepancy (model capacity and hierarchical depth) is convincing, but it highlights that the paradox remains an exploratory finding that is not yet physically demonstrated on real deep models (e.g., ViTs).

3. **Heterogeneous Streaming Benchmark (Table 3)**:
   - The streaming results show the severe impact of *Batch Style Blur* on standard routers as batch size increases (BSigmoid drops from 84.55% at $B=1$ to 66.63% at $B=256$).
   - The results provide outstanding support for the efficacy of **Dynamic Batch Filtering (DBF)**: activating DBF at $B=256$ climbs BSigmoid to 83.18% (+16.55% absolute increase) and Linear Router to 93.77% (+28.63% absolute increase). This is a highly impressive and convincing demonstration of DBF's systems-level value.
   - The physical CNN streaming experiments further validate DBF, showing enormous accuracy gains (+27.59% at $B=16$ and +30.56% at $B=64$) on real convolutional weights.

4. **Detailed Latency and serving comparison (Tables 4 & 5)**:
   - The wall-clock latency breakdown is highly detailed, profiling element-wise CPU parameter interpolation (755.55 $\mu$s per layer). It supports the claim that weight reconstruction is the primary bottleneck, while routing is extremely cheap (under 15 $\mu$s).
   - The quantitative comparison against PEFT runtimes (Punica, S-LoRA) highlights the hardware lock-in of adapters and the universal portability of Hybrid-Router, providing strong systems-level arguments.
   - The sensitivity analysis of $\eta$ (Table 6) shows that $k=12$ consistently outperforms $k=14$ across all penalty weights, proving the robustness of the structural regularization claim within the sandbox.
