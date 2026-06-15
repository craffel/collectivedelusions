# Evaluation Component 3: Soundness and Methodology

## Clarity of Description
The methodology of this paper is exceptionally well-written, clear, and mathematically rigorous. The problem formulation of task vectors, layer-wise merging coefficients, and search space parameterizations (GT-Merge, Poly-Val-Merge, and Unconstrained Layer-wise) is presented in a structured and transparent manner. The pseudo-code in Algorithm 1 outlines the OFS-Tune pipeline clearly.

## Appropriateness of Methods
1. **Calibrated Continuous Simulation Landscape:** The choice to utilize a continuous weight-merging simulation calibrated on empirical ViT-B/32 statistics is highly appropriate. It acts as an elegant "laboratory environment" that cleanly isolates and disentangles key variables (e.g., validation sample size $M$, systematic target bias $\sigma_{bias}$, optimization vs. generalization errors) that are heavily conflated in chaotic empirical settings. Running this over 30 independent seeds provides strong statistical confidence.
2. **Derivative-Free (Nelder-Mead) vs. Gradient-Based (Adam) Optimization:** Comparing Nelder-Mead and PyTorch Adam across different parameter dimensions is a brilliant methodological design. It allows the authors to empirically disentangle whether unconstrained layer-wise search fails due to optimization limits (simplex stalling in high dimensions) or generalization collapse (overfitting to noise).
3. **Physical Neural Network Validation:** Adding physical weight-merging experiments on actual deep convolutional neural networks (a 5-layer CNN evaluated on MNIST and FashionMNIST across 5 seeds) is a phenomenal methodological addition. It directly validates the simulation's findings in a non-trivial weight space.
4. **Stress-Testing under Distribution Shifts:** The selection of Extreme Label Shift, Bursty Temporal Streams, and Small Batch Sizes represents highly realistic, safety-critical real-world deployment challenges where online unsupervised objectives (like entropy minimization) are known to exhibit high variance and instability.

## Potential Technical Flaws and Practical Challenges (Practitioner's Critique)
While the paper is methodologically stellar, we identify a few practical bottlenecks and areas where further discussion is warranted:
1. **Weight Reconstruction and VRAM I/O Overhead during Tuning:**
   - From a practical deployment standpoint, weight-space model merging is highly valued for its parameter efficiency. However, because the prediction loss is non-linear with respect to the merging coefficients (due to activation layers), evaluating the validation loss $\mathcal{L}_{val}(\theta)$ at each optimization step *requires* reconstructing the merged weights $W_{merged}(\theta)$ and running a forward pass.
   - For massive models (e.g., 7B to 70B parameter LLMs or large ViTs), performing 100+ Nelder-Mead or Adam steps means the system must repeatedly perform weight additions in memory. This introduces severe memory bandwidth and disk/VRAM I/O bottlenecks. 
   - While OFS-Tune completely bypasses *test-time* compute (which is the most critical constraint), the *offline* validation tuning phase itself could be extremely resource-intensive for large foundation models. The paper would be stronger if it explicitly discussed this practical bottleneck and analyzed the memory/compute trade-offs of offline parameter reconstruction.
2. **Scale of Physical Validation:**
   - The physical validation is performed on a 5-layer CNN on MNIST/FashionMNIST datasets. While this successfully provides a proof-of-concept for the Overfitting-Optimizer Paradox, these datasets are relatively simple and homogeneous. 
   - Modern model merging is primarily applied to pre-trained foundation models (e.g., CLIP-ViT, LLaMA) with highly heterogeneous task spaces. While the authors argue that linear connectivity is even stronger in pre-trained networks, verifying OFS-Tune on a larger pre-trained backbone would further solidify its practical industry value.
3. **Differentiable Validation via functional API:**
   - The authors mention using PyTorch's `torch.func.functional_call` API to perform backpropagation directly through the merging coefficients. While extremely elegant, this functional forward pass is not native to all model architectures or deep learning frameworks, and can sometimes be complex to implement for practitioners using customized layers.

## Reproducibility
The reproducibility of the work is **excellent**. The authors provide precise mathematical formulations, explicit initialization parameters, learning rates, sample sizes ($M \in \{5, 10, 20, 50\}$), and seed sweeps (30 seeds for simulation, 5 seeds for physical CNN). All algorithms, baseline implementations, and simulation calibration factors are thoroughly detailed, ensuring that any expert practitioner can replicate the entire evaluation pipeline.
