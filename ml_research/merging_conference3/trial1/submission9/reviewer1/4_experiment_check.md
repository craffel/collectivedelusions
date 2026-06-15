# 4. Experiment Check

## Experimental Setup, Datasets, and Baselines
- **Datasets**: MNIST, FashionMNIST, and KMNIST. These are standard, but extremely low-complexity, 10-class grayscale image classification datasets. 
- **Architecture**: A small custom SimpleCNN with 2 conv layers and 3 task heads, totaling around 500k parameters.
- **Baselines**: Includes relevant merging baselines like Task Arithmetic, Ties-Merging, DARE, AdaMerging, and SVD Isotropic Merging. However, the experimental setup **completely omits concurrent training-free layer-wise scaling baselines** (such as LARV, MAGIC, or LiNeS), which are the most direct competitors to their layer-wise calibration.
- **Seeds**: The authors conduct evaluations across 3 independent random seeds, providing statistical means and standard deviations. This represents good scientific practice.

## Support for Central Claims
1. **Claim: RMS-Scale resolves representation scale mismatches and prevents task dominance.**
   - *Support*: **Supported, but with a clear multi-task trade-off.** Tuned RMS-Scale increases MNIST accuracy from 87.00% (Task Arithmetic) to 91.37% (+4.37%) and KMNIST accuracy from 57.63% to 61.57% (+3.94%). However, it causes a regression on FashionMNIST, dropping from 72.87% to 66.73% (-6.14%). The authors are honest about this trade-off, explaining that Task Arithmetic allowed FashionMNIST to dominate the joint parameter space due to its larger update scale, and that balancing the scales necessarily downscales the dominant task to recover the other two.
2. **Claim: The parameter-free PF-RMS variant achieves robust performance out-of-the-box without any tuning.**
   - *Support*: **Supported.** PF-RMS achieves 72.23% average accuracy, significantly outperforming default un-tuned Task Arithmetic (71.68%) and un-tuned Ties-Merging (71.81%) while matching the performance of validation-tuned Ties-Merging (71.77%) and tuned DARE (71.49%).
3. **Claim: RMS-Scale is mathematically equivalent to Frobenius-norm scaling and matches SVD Isotropic Merging alignment with a 100x speedup.**
   - *Support*: **Supported.** The real-weight evaluation on CLIP ViT-B/32 projection layers demonstrates that RMS-Scale and SVD Isotropic Merging achieve the exact same average cosine alignment (57.74%) and isotropic balance (0.15% std), but RMS-Scale takes only 5.67ms per layer compared to SVD's 571.92ms (a 100$\times$ speedup).
4. **Claim: Minimalist scale calibration is robust, highly efficient, and sufficient.**
   - *Support*: **Supported on the toy setup**, but lacks empirical downstream accuracy support on modern, large-scale architectures.

## Key Experimental Weaknesses
- **Extreme Scale Gap**: Evaluating classification accuracy solely on a 500k parameter SimpleCNN on toy grayscale datasets (MNIST, Fashion, KMNIST) is highly insufficient for a conference like ICML 2026. Modern model-merging literature is judged on its ability to merge deep models like ResNet-50, ViT-B/16, CLIP, LLaMA-7B, or Mistral-7B on complex benchmarks (e.g., ImageNet, GLUE, GSM8k, or zero-shot vision tasks).
- **Missing Direct Baselines**: Fails to compare against or even cite contemporary training-free layer-wise scaling baselines (LARV, MAGIC, LiNeS, LOT Merging). This is a critical experimental and scholarly omission.
- **No Accuracy Evaluation on CLIP**: The CLIP ViT-B/32 experiment only measures activation cosine alignments on simulated updates. No end-to-end classification performance is reported for the merged model on real-world downstream datasets (e.g., Stanford Cars, Flowers102, EuroSAT), leaving the downstream viability of their CLIP merging claims unverified.
