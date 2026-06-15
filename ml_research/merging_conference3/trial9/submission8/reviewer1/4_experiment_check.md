# 4. Experimental Check

## Critical Evaluation of the Experimental Setup
The experimental setup is exceptionally weak and is a major limitation of this paper. The authors propose an incredibly heavy, mathematically dense, and highly complex continuous-time orbital mechanics framework to govern layer-wise expert adapter merging. However, they evaluate this framework on a **highly simplified, synthetic, and toy coordinate simulation proxy**:
1. **The Dataset:** They use scikit-learn's `load_digits` dataset, which consists of only **1,797 samples of 8x8 grayscale images** of handwritten digits. This is a tiny, toy dataset from decades ago that is completely unrepresentative of the complex, high-dimensional, and non-stationary workloads encountered in modern edge AI.
2. **The Simulation:** To mimic representation spaces, they project these 8x8 pixel features (64 dimensions) to 192 dimensions using a random orthogonal projection matrix.
3. **No Real Deep Models:** There is no actual pretrained transformer backbone, no active LoRA weights, and no real inference being run. The entire "backbone propagation and multi-expert blending are mathematically modeled as coordinate operations on these projected representation vectors."

Casting a simple vector-projection task on scikit-learn digits as a "Projected Digit Representation Space (RDS) Proxy benchmark" and using it as the primary evaluation of a complex manifold-integration method is highly inappropriate. A method designed for deep multi-task serving must be evaluated on real, large-scale deep models (e.g., LLaMA, Mistral, ViTs) on downstream tasks (e.g., GLUE, MMLU, GSM8k) to demonstrate any practical utility.

## Evaluation of Baselines
The paper compares GraviMerge to several baselines, including Uniform Merging, SABLE, SPS-ZCA, EMA, ChemMerge, and Kalman Filter. However, a close analysis of the empirical results in Table 1 reveals that the claims of GraviMerge's superiority are not supported in any meaningful way:

1. **SPS-ZCA is superior in design and equally effective:** SPS-ZCA (Single-Pass Model Merging with Zero-Shot Centroid Alignment) is an incredibly simple, zero-overhead baseline. It aligns inputs to task centroids once at the early layers and uses static routing weights throughout the rest of the network.
   - **SPS-ZCA Accuracy:** $88.51\% \pm 1.68\%$
   - **GraviMerge Accuracy:** $88.69\% \pm 1.68\%$
   - **Performance Delta:** A negligible **+0.18%** absolute accuracy difference.
   - **SPS-ZCA Jitter (MAD):** **0.00000** (Perfect stability by design).
   - **GraviMerge Jitter (MAD):** **0.00190** (Still has some layer-to-layer jitter).

SPS-ZCA completely resolves the layer-to-layer weight jitter problem by using static weights, achieving perfect stability (0.00000 MAD) with zero stateful tracking, zero velocity storage, zero parallel transport, and zero computational overhead. GraviMerge's massive mathematical machinery achieves a virtually identical accuracy (+0.18%) while actually introducing some weight jitter (0.00190 MAD). From a minimalist perspective, SPS-ZCA is vastly superior because it solves the problem in the simplest, most direct way possible with zero complexity.

2. **EMA and Standard Filters are Poorly Calibrated:** The paper claims first-order filters like EMA and ChemMerge suffer from "lag-induced control loop delays" that degrade performance. However, this is likely an artifact of poor hyperparameter calibration for the baselines or the artificial nature of the toy RDS benchmark, where lag is heavily penalized. In real-world applications, a simple moving average is extremely lightweight and easy to tune, whereas GraviMerge requires tuning 5-7 continuous physical constants.

## Redundant and Misleading Reporting
In Table 1, the columns for "Homo. Acc.", "Hetero. Acc.", and "Real-Time Acc." report **exactly identical accuracies** for every single method (e.g., $88.69\% \pm 1.68\%$ for GraviMerge across all three). 
The authors explain in the caption that this is because each sample in their default execution model is processed independently, meaning the batching configuration has no mathematical impact on inference. 

Reporting the same mathematical results in three separate columns to claim "total multi-stream robustness" and "high resilience" is highly deceptive. There is no actual dynamic streaming sequence or multi-stream interference being modeled; it is simply the exact same feed-forward operation run under different batch sizes, which naturally yields identical results in PyTorch. This redundancy artificially inflates the empirical portion of the paper and obscures the lack of real-world evaluation.
