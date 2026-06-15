# Paper Summary: ELATI

## Core Goal
The paper introduces **ELATI** (**E**arly-**L**ayer **A**daptive **T**ask **I**dentification), a training-free and parameter-free "one-pass" dynamic model-merging system. It aims to eliminate the severe **two-pass latency penalty** of state-of-the-art dynamic weight-merging routers (such as Parameter-Free Subspace Routing, PFSR). Prior routers project activation vectors from the penultimate layer of the base model, requiring a full, throw-away forward pass of the backbone just to obtain routing coefficients, followed by a second forward pass through the dynamically merged network. ELATI shifts dynamic task identification to an early layer ($l_{\text{route}} = 2$), achieving a single-pass inference pipeline.

## Proposed Methodology
1. **Theoretically Modeled Target Architecture**: An $L$-layer ($L=14$ in the main simulation) sequential residual backbone. Each expert task is represented by a low-rank adapter (LoRA) update $V_k^{(l)}$.
2. **Early-Layer Representative Mapping (ELRM)**: Computes unsupervised, offline mean activation centroids $W'_k$ at the routing layer $l_{\text{route}}$ from a tiny calibration split (16 samples per task, 64 total). Test-time activations are projected against these centroids using cosine similarity to extract soft $K$-dimensional routing coordinates.
3. **Sequence Pooling for 3D Tensors**: Proposes various sequence pooling operators ($\Psi_{\text{mean}}$, $\Psi_{\text{cls}}$, $\Psi_{\text{final}}$, and $\Psi_{\text{attn}}$) to map 3D token sequences to 2D activation vectors at early layers.
4. **Downstream-Only Micro-Batch Homogenization (DO-MBH)**: Propagates the input batch once up to $l_{\text{route}}$, projects activations to get coefficients, groups samples by dominant task, dynamically interpolates downstream adapters ($l > l_{\text{route}}$) into a scratch VRAM buffer, dispatches grouped activations through the merged tail, and scatters outputs back to original sequence indices.

## Experimental Setup & Baselines
- **Hierarchical 14-Layer Sandbox**: A PyTorch-based simulator that mathematically models sequential multi-layer residual transformations with GeLU activations and LoRA adapters.
- **Simulated Task Subspaces**: Four simulated manifolds modeling MNIST, Fashion-MNIST, CIFAR-10, and SVHN, with task-specific noise scales ($\sigma_k$).
- **Physical ViT-Tiny Evaluation**: Pre-trained ViT-Tiny (\texttt{vit\_tiny\_patch16\_224}) on real-world test splits of MNIST, F-MNIST, CIFAR-10, and SVHN.
- **Physical GPT-2 NLP Evaluation**: Autoregressive GPT-2 model on four distinct language tasks.
- **Hardware-Level GPU Profiling**: A PyTorch CUDA-event profiling pipeline scaling CPU timing using GPU bandwidth limits (2.0 TB/s on NVIDIA A100) and scheduling overhead.
- **Baselines**: Expert Ceiling (Oracle), Static Uniform Merging, DARE-Merging, TIES-Merging, Linear Routers (Unreg and Reg), and PFSR+MBH (Penultimate).

## Key Empirical Findings
- **Sandbox Accuracy**: ELATI achieves a Joint Mean accuracy of **56.89% $\pm$ 1.66%**, outperforming Uniform Merging (48.27%), DARE (32.56%), and TIES (37.39%), while remaining highly competitive with penultimate PFSR (58.25%).
- **E2E Latency**: ELATI reduces CPU end-to-end execution latency from 36.90 ms (PFSR) to 26.43 ms (**1.40$\times$ physical speedup**).
- **Projection Latency**: Under vectorized batch execution on CPU, ELATI reduces projection latency from 1.31 ms to 0.39 ms (**3.33$\times$ speedup**).
- **Physical ViT Routing**: ELATI's unsupervised centroids achieve **79.25%** routing accuracy on entangled real-world ViT activations.
- **Physical ViT Downstream Accuracy**: ELATI achieves **21.50%** Joint Mean accuracy, a massive improvement over Static Uniform Merging (9.25%) and close to the Expert Oracle (26.00%).
- **Hybrid Online Adaptation**: Continuously tracks non-stationary domain drift, recovering routing accuracy to 99.50% vs. 63.00% for static centroids.
