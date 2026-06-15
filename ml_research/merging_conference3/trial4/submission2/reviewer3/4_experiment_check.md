# 4. Experimental Evaluation and Results Check

## Evaluation of Experimental Setup and Datasets
- **Over-Engineered and Ill-Suited Benchmark Suite:** The authors evaluate their framework using a pre-trained `ViT-Tiny` backbone (5.7M parameters) on four simple classification datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. This is a highly over-engineered and non-elegant setup:
  - Both MNIST and FashionMNIST consist of tiny $28 \times 28$ grayscale images, while SVHN and CIFAR-10 are $32 \times 32$ images.
  - To process these with `ViT-Tiny`, the authors upsample these low-resolution images to $224 \times 224$ pixels. 
  - Using a multi-million parameter Vision Transformer on MNIST/FashionMNIST is computationally wasteful and conceptually inappropriate. A simpler, more elegant setup would use standard, realistic transfer learning and domain adaptation benchmarks (e.g., Office-Home, DomainNet, or ImageNet subsets) where a Vision Transformer is actually necessary and standard, or use simple, classic convolutional networks (like ResNet-18) on these small datasets.

## Weak and Incomplete Expert Training
- **Poor Expert Validation Performance:** To simulate "genuine task experts", the authors train the experts on only 256 images per task for 3 epochs. This extremely restrictive training budget leads to very weak experts that are far from convergence:
  - **SVHN validation accuracy is only 28.91%** (barely above a random guess of 10% on a 10-class problem, whereas standard models easily achieve 95%+).
  - MNIST (82.03%), FashionMNIST (81.25%), and CIFAR-10 (74.22%) are also significantly under-trained.
  - Merging models that are barely functional (particularly on SVHN) makes the entire ensembling setup unrealistic. The "FP16 Task Arithmetic ceiling" of 38.67% average accuracy across tasks is extremely low. It is highly questionable whether conclusions drawn from merging such weakly adapted, poorly trained models are applicable to real-world deployment scenarios where experts are fully converged, highly specialized, and highly accurate.

## Ablation Study Contradicts the Main Claim
- **Unjustified Complexity of the Full OmniMerge Framework:** The core claim of the paper is that the *combination* of Stochastic Operator Sampling (SOS) and Scale/Zero-Point Noise Perturbation (SZNP) is necessary to achieve robust multi-schema co-optimization. However, Table 2 (Ablation study) directly contradicts this:
  - **SZNP Only (Baseline + TCR + SZNP)** achieves **50.45%** average cross-schema accuracy.
  - **Full OmniMerge (Ours)** achieves **50.33%** average cross-schema accuracy.
  - This means that stochastically sampling operators (SOS) actually **degrades** performance by 0.12% when combined with scale/zero-point noise perturbation. 
  - A simpler, more elegant method that simply applies scale/zero-point noise under a single static operator (SZNP Only) is performatively superior to the full, more complex "OmniMerge" framework.
  - The authors argue that Full OmniMerge is needed for out-of-pool schemas (Double Quantization), but they fail to report the performance of the "SZNP Only" configuration on the Double Quantization schema. Since the average in Table 2 is computed over all five schemas (including Double Quantization), and SZNP Only has a higher average, the simpler SZNP-only method remains superior on average overall. The authors have failed to prove that the added complexity of SOS is performatively justified.

## Over-Interpretation of Statistical Noise
- **Speculative "Weight Denoising" Claims:** The authors claim that 8-bit post-training quantization can act as a beneficial regularizer or "noise filter" because quantized OmniMerge (50.78% under Symmetric Per-Channel) outperforms the unquantized optimized FP16 ceiling of AdaMerging (46.68%). This is a misleading comparison:
  - The unquantized FP16 optimized ceiling of 46.68% is achieved under *AdaMerging* coefficients.
  - When the exact continuous coefficients optimized by OmniMerge are evaluated in unquantized FP16, the model achieves **50.39%** accuracy.
  - The quantized model under Symmetric Per-Channel achieves **50.78%** accuracy.
  - The difference between the quantized model (50.78%) and its unquantized counterpart (50.39%) under the exact same coefficients is exactly **0.39% absolute** (representing an increase of just 4 correct predictions out of 1024 total test images).
  - This tiny difference of 4 images is well within the binomial standard error of the evaluation stream ($\approx 1.56\%$) and is statistically insignificant. 
  - Promoting this negligible, random fluctuation into a scientific finding that "discrete rounding acts as a beneficial noise filter rather than a destructive lossy operation" is an over-interpretation of experimental noise. The real reason OmniMerge outperforms AdaMerging is simply that its optimization objective (prediction entropy + TCR with noise) finds better continuous blending coefficients, not because the subsequent quantization "regularizes" or "denoises" the weights.
