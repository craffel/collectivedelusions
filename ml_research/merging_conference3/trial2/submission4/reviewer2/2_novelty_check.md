# Novelty and Literature Delta Assessment

## 1. Key Novel Aspects of the Submission
The paper's proposed core novelty lies in introducing **training-free, forward-only activation analysis** as a post-hoc weight-routing mechanism for model merging. Unlike previous adaptive model merging works, it tries to compute channel-wise scaling coefficients in closed-form without relying on backpropagation or iterative optimization.

The authors identify three main components of their framework:
- **Forward-Only Activation Sampling (FOAS):** Extracting activations from the base model and task experts in a single forward pass.
- **Scale-Normalized Delta Activation Salience (SNDAS):** Using Frobenius-norm scaled activation shifts to quantify a channel's functional importance across highly heterogeneous task-experts.
- **Channel-Wise Softmax Gating (CWSG):** Applying softmax over normalized saliency scores to assign output channels to specific experts.
- **Decoupled Scale Routing (DSR):** Separating the update scale of statically merged layers from that of the gated projection layers to resolve the scale-dampening effect of softmax routing.

## 2. Delta from Prior Work
The paper positions itself relative to two main classes of existing model merging techniques:
- **Static Model Merging (Task Arithmetic, TIES, Model Soups, Git Re-Basin, ZipIt!):** These methods consolidate parameters statically and uniformly across all layers, ignoring the context-specific or local significance of parameters. EdgeMerge attempts to introduce local, channel-wise adaptability.
- **Gradient-Based Adaptive Merging (AdaMerging, SyMerge, FoldMerge):** These methods optimize merging coefficients using test-time backpropagation (often 500 gradient steps), which requires significant compute, preparation latency, and high-end hardware. EdgeMerge completely removes the backward pass, executing in closed-form.

While this comparison highlights an open niche, the conceptual components of EdgeMerge are heavily drawn from existing research:
- **Activation-based saliency estimation** has been extensively explored in the deep learning compression literature, particularly in **channel pruning** (e.g., He et al., 2017; Molchanov et al., 2016) and **quantization** calibration. The formula for SNDAS is fundamentally a row-wise/channel-wise activation norm, a staple technique in pruning for over a decade.
- **Channel modulation / routing** via softmax or scaling is highly similar to **Squeeze-and-Excitation networks** (Hu et al., 2018) and other dynamic routing/attention layers, though EdgeMerge applies this post-hoc directly to weights without training new parameters.
- **Decoupled scaling** is an intuitive architectural hyperparameter tuning step (using different scaling coefficients for different layers), widely used in practice when fine-tuning or merging deep neural networks.

## 3. Characterization of Novelty
The novelty of this work must be characterized as **incremental and highly limited** due to several critical theoretical and empirical factors:

### A. The "Adaptive" Gating is Functionally Inert
The paper's core conceptual novelty is the use of activation statistics to perform dynamic, channel-wise adaptive gating (FOAS, SNDAS, CWSG). However, the paper's own ablation study (Table 5) reveals that:
- Bypassing Frobenius norm scaling (**No SNDAS**) achieves **69.58%** accuracy.
- Replacing fine-grained channel gating with a uniform average (**Uniform Gating**, $\alpha_k = 1/K$) achieves **69.58%** accuracy.
- Collapsing channel gating into a single layer-wise scalar (**Layer-wise Gating**) achieves **69.59%** accuracy.

Because setting $\alpha_k = 1/K$ (which is mathematically equivalent to static, uniform Task Arithmetic with a scaled-down projection update scale) yields the exact same performance as the full EdgeMerge pipeline, the actual adaptive gating mechanism does **absolutely nothing** to improve representation composition or resolve inter-task conflicts. The entire performance gain (+0.84%) is achieved by **Decoupled Scale Routing (DSR)**—specifically, setting $\lambda_{static} = 0.25$ and regularizing the visual projection layer with a small scale like $\lambda_{proj} = 0.025$ (which is $0.20 / 8$). Thus, the paper's primary proposed novelty is functionally redundant.

### B. High Overlap with Established Pruning and Scale Calibration Techniques
Rather than introducing a novel mathematical framework for weight routing, the paper merely translates well-known activation scale estimation metrics from the network pruning literature into the weight merging space. Because this post-hoc application of activation-average scales does not yield any performance benefits over static uniform averaging (as proven in their ablations), the scientific novelty of this translation is highly questionable.

### C. Summary of Novelty
In summary, while the paper identifies a highly practical engineering problem (on-device, gradient-free adaptation) and proposes an interesting direction, its core technical "novelty" is shown to be an over-engineered implementation of simple, static, decoupled hyperparameter scaling. The actual functional "delta" of their proposed adaptive routing over simple uniform scaling is **0.00% absolute accuracy**.
