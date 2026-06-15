# Evaluation Task 1: Paper Summary

## Main Topic and Goal
This paper addresses the challenges of **test-time model merging** (or dynamic ensembling of specialized expert adapters, such as LoRAs) without the need for intensive retraining. The primary goal is to perform dynamic, sample-specific weight merging over non-stationary, heterogeneous input streams in a way that is robust, optimization-free, and stable across varying batch sizes (from large batches down to sequential, single-sample streams where $B=1$).

## Proposed Approach: FIOSR
The authors propose **Fisher-Information Optimal Subspace Routing (FIOSR)**, a training-free and parameter-free dynamic ensembling framework. Instead of assuming a flat, isotropic Euclidean weight/representation space, FIOSR models the parameter and activation spaces as a Riemannian manifold. The key components of the framework are:
1. **Fisher-Weighted Cosine Similarity**: By computing a smoothed and power-scaled diagonal empirical Fisher Information Matrix (dFIM) over a tiny calibration split, the framework constructs a local Riemannian metric tensor. This metric tensor warps the coordinate space, naturally suppressing noisy/task-irrelevant activation dimensions (which have low Fisher values) while amplifying highly informative, discriminative coordinates.
2. **Class-Size Scaling Calibration (CSC)**: Bypasses statistical maximum bias introduced by varying class vocabulary sizes ($C_k$) among different specialized experts.
3. **Micro-Batch Homogenization (MBH)**: Bypasses *heterogeneity collapse* and enables high batch execution efficiency by partitioning a heterogeneous test stream into homogeneous micro-batches based on their dominant task coordinates, allowing a single merged model to process each micro-batch.

## Key Findings and Claims
- **Overcoming Overfitting (The Dynamic Routing Paradox)**: Parametric routers (e.g., Linear, QWS-Merge) that optimize routing coefficients at test-time on tiny support sets (e.g., $N=64$) suffer from severe overfitting. FIOSR, being parameter-free, is completely immune to this.
- **Overcoming Sequential Instability (Vectorization Collapse)**: On single-sample streams ($B=1$), parametric routers fluctuate wildly and collapse in accuracy. FIOSR maintains stable routing accuracy regardless of stream batch size.
- **SOTA Routing and Classification Accuracy**: Evaluated in an Analytical Coordinate Sandbox, FIOSR recovers near-perfect expert routing stability (~100% MNIST/FashionMNIST) and outperforms unweighted Cosine PFSR by 8.56% and parametric baselines by up to 40.7% under extreme non-stationary streaming.
- **Real-World and Physical Validation**: Evaluated on physical LoRA activation spaces and a pre-trained ResNet-18 backbone model, demonstrating its computational speed (estimating dFIM in 4.05ms) and its ability to outperform flat cosine similarity in real physical representation spaces.

## Explicitly Claimed Contributions
1. **Information-Geometric Perspective**: Introduces Riemannian manifold analysis and Fisher Information curvature to test-time model ensembling.
2. **Smoothed Fisher Regularizer**: Formulates an analytical coordinate filter that suppresses task-irrelevant noise in parameter-free routing.
3. **Optimization-Free Robustness**: Demonstrates complete immunity to overfitting and Vectorization Collapse under low-latency streaming.
4. **Rigorous Empirical Evaluation**: Compares FIOSR against five major baselines across 10 random seeds on synthetic, semi-synthetic (LoRA activations), and physical (ResNet-18 backbone) benchmarks.
