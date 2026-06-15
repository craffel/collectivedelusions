# Soundness and Methodology Evaluation: CAM-Router

## Clarity of Description
The proposed CAM-Router is described with commendable mathematical clarity. The equations in Section 3 outlining the Multi-Head Cross-Attention (MHCA) and independent bounded sigmoidal gating are precise. The "First-Block Paradox Resolution" is also well-explained.

## Appropriateness of Methods & Technical Flaws

### 1. Stateful Inference and Non-Determinism (A Major Conceptual Flaw)
The introduction of **Decoupled Historical Gating (DHG)** is mathematically clear but conceptually flawed for standard inference. By maintaining an Exponential Moving Average (EMA) of the merging coefficients over a sliding historical window, the weights of the merged model at step $t$ depend directly on the sequence of inputs processed in preceding steps ($t-1$, $t-2$, etc.). 
This stateful design introduces **non-determinism**: a single image $x$ will produce different merging coefficients and thus different classification predictions depending entirely on the historical stream of images that preceded it. This context-dependence is highly undesirable for production deployment, safety-critical applications, or standardized benchmarking, where predictions should be independent and identically distributed (i.i.d.).

### 2. First-Block Paradox and Ad-Hoc Hybrid Structure
To extract spatial features before the model's weights are merged, the authors keep the patch embedding and the first transformer block static (frozen to the base model weights). This creates an ad-hoc hybrid architecture where layer 1 is static, and layers 2-L are dynamically merged. When fine-tuning individual experts, all layers (including the first block) are typically updated. Forcing layer 1 of the merged model to use base model parameters could trigger a representation mismatch or distribution shift, which is not evaluated.

### 3. Eager Weight Summation and Latency Overhead
Naively sum-merging large weight tensors ($W_{base}^{(l)} + \sum \bar{\alpha}_k V_k^{(l)}$) on-the-fly for layers 2 to 14 during inference is extremely memory-bandwidth heavy. The authors acknowledge this in Section 3.4 (Latency Discussion) and propose hypothetical caching and custom Triton kernels. However, these are "future directions" and are not actually implemented or evaluated. Without them, the proposed dynamic model compilation adds a significant latency penalty, undermining the efficiency of parameter-space merging.

## Reproducibility
The architectural details (parameter counts, dimensions, layers) and hyperparameter settings (learning rate, steps, optimizer) are documented thoroughly, which is excellent for reproducibility. However, the experiments are conducted on a "14-layer compact Vision Transformer coordinate sandbox", which is a custom simulated environment. Because the simulation code is not standard or publicly available, reproducing the exact empirical results is challenging without access to the authors' custom training scripts.
