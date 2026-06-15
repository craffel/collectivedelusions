# 4. Experimental Setup and Evaluation Check

## Critique of Experimental Setup and Datasets
1. **Low Baseline Expert Performances:**
   The paper reports that specialized expert models are fine-tuned on 500 training samples and reach diagonal test performance of:
   - MNIST: $81.00 \pm 0.82\%$
   - FashionMNIST: $74.67 \pm 1.25\%$
   - CIFAR-10: $71.67 \pm 4.03\%$
   - SVHN: $85.33 \pm 2.87\%$
   
   These accuracies are extremely low for standard deep neural networks (and specifically Vision Transformers) on these relatively simple classification datasets. For instance, a basic CNN easily achieves $>98\%$ on MNIST/FashionMNIST and $>90\%$ on SVHN. Under-trained experts can introduce high parameter noise, which may affect the generalizability of the merging results and distort the comparison.
   
2. **Subsampled Test Sets (High Evaluation Noise):**
   The evaluation test sets are subsampled to only 100 samples per task (total 400 test samples across 4 tasks). This extremely small test set explains the high standard deviations reported across 3 random seeds (e.g., up to $3.65\%$ standard deviation for U-PhaseMerge). Subsampling to 100 samples makes the empirical results highly susceptible to statistical noise and evaluation variance, calling into question the significance of small performance differences (such as the $0.83\%$ gap between U-PhaseMerge and AdaMerging in FP32).

## Evaluation of Baselines
The baselines selected are appropriate and cover both traditional real-space approaches (Uniform Task Arithmetic, AdaMerging, PolyMerge) and frequency-domain approaches (FREE-Merging). 
However, **PolyMerge** is the most powerful and direct baseline. It is a highly constrained adaptive merging method that parameterizes layer coefficients via a quadratic depth-wise polynomial. PolyMerge consistently and substantially outperforms the proposed PhaseMerge and U-PhaseMerge methods across all tables.

## Analysis of Claims vs. Empirical Results

| Paper Claim | Empirical Evidence in Paper | Supported? |
| :--- | :--- | :--- |
| **"PhaseMerge significantly outperforms static baselines under FP32 and PTQ regimes"** | **Table 1:** U-PhaseMerge ($42.83\%$) and PhaseMerge ($40.75\%$) outperform Uniform TA ($38.25\%$) and FREE-Merging ($27.17\%$) in FP32, and the trend holds for 8-bit and 4-bit PTQ. | **Yes.** The proposed methods are clearly superior to static baselines. |
| **"Crucially, the uniform $r=1$ variant exhibits exceptional robustness under extreme 4-bit quantization, outperforming traditional real-space layer-wise optimizations."** | **Table 1 (4-bit PTQ):** U-PhaseMerge ($37.42 \pm 1.94\%$) performs worse than AdaMerging ($37.50 \pm 1.22\%$, a traditional real-space layer-wise optimization) and significantly worse than PolyMerge ($43.42 \pm 1.30\%$). | **No.** It is empirically worse than the traditional layer-wise real-space optimization (AdaMerging) under 4-bit PTQ, and has higher standard deviation. |
| **"Operating in complex frequency-space... prevents the optimizer from fitting to sharp local rounding thresholds, providing excellent generalizability across diverse target post-training quantization schemas."** | **Table 3 (Calibrated 8-bit $\to$ Deployed 4-bit):** U-PhaseMerge achieves $37.42 \pm 1.94\%$, which is under AdaMerging's $37.50 \pm 1.22\%$ and far below PolyMerge's $43.42 \pm 1.30\%$. | **No.** Under schema shift, the proposed method does not outperform the unconstrained real-space baseline, and is beaten by the constrained real-space baseline (PolyMerge) by a large margin ($6.00\%$). |
| **"Uniform PhaseMerge (U-PhaseMerge) serves as a stable and robust matrix-basis regularizer, avoiding high-frequency overfitting and stabilizing optimization."** | **Table 2 (Sample Complexity):** When calibration data increases from $M=16$ to $M=32$, U-PhaseMerge's performance drops from $42.33 \pm 1.76\%$ to $40.67 \pm 3.65\%$ (a $-1.66\%$ decrease in mean, and a doubling of variance). | **No.** This represents optimization instability. A robust regularizer should exhibit monotonic or stable improvement as calibration data grows. U-PhaseMerge fails to scale stably to $M=32$ compared to AdaMerging (which improves from $41.67\%$ to $42.50\%$) and PhaseMerge $r=2$ (which improves from $40.83\%$ to $42.00\%$). |

## Summary
The empirical results do not support the paper's primary claims of superiority over traditional real-space layer-wise optimizations under low-bit quantization, schema shifts, or calibration scale-up. In fact, a simpler real-valued baseline, **PolyMerge**, consistently dominates PhaseMerge and U-PhaseMerge by a margin of 5% to 6% across all settings, while utilizing far fewer parameters (12 vs. 192/768) and requiring no complex Fourier/inverse Fourier transformations.
