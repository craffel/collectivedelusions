# Soundness and Methodology

## Clarity of Description
The paper is exceptionally well-written and structured. The mathematical formulations are clear and easy to follow:
- The problem setup, calibration, and raw routing weights are logically introduced in Equations 3–6.
- The 2D-STEM bilinear recurrence (Equation 10) is simple and transparent.
- The simplex-preservation proof for Theorem 3.1 is rigorous, complete, and mathematically sound.
- The transition from linear temporal gating to Power-Law Gating (ATG-PL) is clearly explained and geometrically justified.

## Appropriateness of Methods
From a practical deployment perspective, using a discrete-time bilinear recursive filter is highly appropriate. It avoids complex numerical integration and backpropagation loops, which are major bottlenecks for low-power edge hardware.

However, there is a **notable methodological contradiction** regarding the "training-free" and "zero-parameter" claims of the paper:
- The authors emphasize that 2D-STEM is "training-free, highly parameter-efficient... with zero extra parameters and zero online backpropagation or optimization overhead."
- Yet, in Appendix B (Section 7), they introduce a **2-layer MLP coordinate-prior mapper** to resolve representation overlaps in fine-grained domains. This MLP requires supervised training with task labels on the $N_{\text{cal}} = 64$ calibration samples and introduces approximately 7,000 parameters. While they argue it is trained offline and is small, it still violates the "training-free" and "parameter-free" claims.

## Potential Technical Flaws and Limitations
1. **Contrived "Physical Validation" Environment:** 
   - The authors present an "Activation-Space Trajectory Validation on Pre-Trained Vision Transformer Representations" (Section 4.4) as a bridge to real-world edge deployment.
   - However, this validation is highly simulated. Instead of fine-tuning physical LoRA experts on actual image datasets (like CIFAR-100 or DomainNet, which are deferred to "future work" in the conclusion), they programmatically generate four synthetic visual patterns (Checkerboard, Sinusoidal Waves, Fractal Noise, and Color Gradients) and extract CLS-token representations from a frozen pre-trained ViT.
   - They then measure a custom "relative alignment accuracy" based on CLS-token cosine similarities rather than actually merging LoRA weights and evaluating classification accuracy on downstream tasks. This is a highly indirect, surrogate-heavy evaluation that does not prove real-world utility.

2. **Severe Performance Discrepancy on Physical Representations:**
   - On the pre-trained ViT representation simulation, the baseline **PAC-Kinetics** (a temporal-only tracker) achieves an alignment accuracy of **$70.57\%$** and a routing jitter of **$0.0063$** under homogeneous streams.
   - In comparison, the proposed **2D-STEM** only achieves **$63.70\%$** alignment accuracy and a routing jitter of **$0.0675$** (which is over **10x higher** than PAC-Kinetics).
   - Even the constant-inertia **ChemMerge Proxy** baseline outperforms 2D-STEM on this task, achieving **$65.83\%$** alignment accuracy and **$0.0419$** jitter.
   - Under heterogeneous streams, PAC-Kinetics also outperforms 2D-STEM in accuracy ($67.08\%$ vs. $64.61\%$) and jitter ($0.0369$ vs. $0.0679$).
   - This represents a critical technical limitation. The authors claim 2D-STEM "surpasses" these highly parameterized frameworks, but their empirical superiority is restricted to their synthetic Analytical Coordinate Sandbox (ACS). When evaluated on a real pre-trained ViT model, 2D-STEM is actually outperformed by simpler temporal baselines like PAC-Kinetics. 

3. **Inability to Verify Real-World Latency and Hardware Gains:**
   - The paper argues that 2D-STEM's $O(K \cdot L)$ complexity avoids the $O(K \cdot L \cdot N_{\text{ODE}})$ overhead of ChemMerge. 
   - However, because they did not implement 2D-STEM on actual edge hardware or perform real-world end-to-end classification, they cannot verify whether these theoretical complexity savings translate to actual latency or energy reductions in a deployed multi-expert serving pipeline.

## Reproducibility
The mathematical formulation and PyTorch-like pseudocode in Listing 1 are highly detailed and clear, making the core routing logic straightforward to reproduce. However, reproducing the exact Analytical Coordinate Sandbox (ACS) and the synthetic ViT simulation setup would be difficult due to the lack of detailed parameter values for the manifold generation.
