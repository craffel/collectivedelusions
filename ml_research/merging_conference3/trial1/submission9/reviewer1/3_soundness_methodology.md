# 3. Soundness and Methodology

## Clarity of Description
The mathematical descriptions of both Standard-Deviation Scaling (SD-Scale) and Root-Mean-Square Scaling (RMS-Scale) are exceptionally clear. The progression from standard deviation to RMS is logical, highlighting standard deviation's translation-invariance vulnerability on small, low-variance tensors (like biases) and positioning RMS as a more stable, non-translation-invariant alternative. The inclusion of the exact PyTorch implementation pseudo-code (Section 3.7) further enhances the clarity and makes the methods highly accessible.

## Appropriateness of Methods
- **RMS vs. SD**: The choice of RMS over standard deviation is highly appropriate. By avoiding subtraction of the mean coordinate-wise shift, RMS-Scale naturally captures both the variance and the mean shift, ensuring absolute stability on low-dimensional bias vectors.
- **Parameter-Free Analytical Calibration (PF-RMS)**: The derivation of the alignment ratio $\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$ and the dynamic scaling factor $\lambda^l = 1 / \alpha^l$ is mathematically sound. It provides a clear, high-dimensional geometric explanation for the shrinkage that occurs when averaging orthogonal task vectors.
- **Clipping Safeguard**: Capping the dynamic scaling factor with a clipping threshold $\gamma(K) = C \cdot \sqrt{K}$ is an appropriate and necessary engineering safeguard to prevent division-by-zero or extreme noise amplification in pathological scenarios where task updates are perfectly opposing ($\alpha^l \to 0$).

## Potential Technical Flaws and Limitations
1. **Evaluation Scale and SimpleCNN Structure**: 
   While the mathematical formulation is solid, the accuracy evaluation is extremely limited. The authors evaluate classification accuracy solely on a custom, lightweight SimpleCNN backbone on MNIST, FashionMNIST, and KMNIST. Deep neural networks with billions of parameters (like Transformers or ResNets) have far more complex block-wise interactions, residual skip connections, attention maps, and feature representations. It is a major leap to assume that findings on a 3-layer toy CNN will translate directly to downstream performance on massive, complex architectures.
2. **Verification on CLIP ViT-B/32 is Limited to Activation Alignment**: 
   The authors attempt to bridge the scalability gap by evaluating on real-world CLIP ViT-B/32 weights. However, this evaluation is performed on isolated weight layers and only reports activation-space cosine alignments. While activation alignment is a useful proxy, it is not a substitute for evaluating zero-shot or fine-tuned downstream task accuracy. Without actual end-to-end downstream performance on CLIP benchmarks, the claim that RMS-Scale "scales to multi-billion parameter foundation models" remains empirically unverified.
3. **Noise Amplification in Conflict-Heavy Layers**: 
   In layers where task updates are highly conflicting, the alignment ratio $\alpha^l$ will be small, forcing a large scaling factor $\lambda^l$. If the remaining uncancelled parameter updates represent minor noise rather than task-relevant directions, scaling up the merged update will over-amplify noise and potentially corrupt the representations. Although the authors propose **PF-Ties-RMS** as a hybrid variant to resolve coordinate-wise sign conflicts first, they do not empirically evaluate this hybrid variant, leaving it purely as a theoretical recommendation.

## Reproducibility
The reproducibility of the proposed methods is **excellent**. The PyTorch implementation is incredibly simple (requiring only two lines of code per layer), and the authors provide clear descriptions of all datasets, learning rates, epochs, optimizer configurations, and baseline hyperparameter search spaces.
