# 3. Soundness and Methodology Check

## Clarity of the Description
The methodology is described with standard mathematical rigor (equations for MHCA, sigmoidal gating, and DHG). However, several critical operational and conceptual details are obscured:
- **First-Block execution details:** It is unclear if the first block's activation is passed directly to Layer 2 of the merged model, or if there is a representational mismatch. The base model first block outputs $H_0$, which is used to compute routing coefficients. Then, does Layer 2 of the merged model take $H_0$ as its input? If so, this assumes that the coordinate space of the base model's first block output is perfectly compatible with the merged model's Layer 2. Since Layer 2 weights are merged dynamically ($W_{merged}^{(2)}$), and they expect inputs from a task-specific Layer 1 (which was fine-tuned), passing $H_0$ (from base/static Layer 1) into $W_{merged}^{(2)}$ may cause severe representation misalignment. This potential mismatch is not discussed.

## Potential Technical Flaws and Conceptual Weaknesses

### 1. Stateful Inference and Temporal Dependencies (DHG)
The proposed **Decoupled Historical Gating (DHG)** is presented as a solution to "heterogeneity collapse" in batched environments. However, it introduces a severe, disqualifying flaw for production deployments:
- **Formula:** $\bar{\alpha}_k^{(t)} = \beta \bar{\alpha}_k^{(t-1)} + (1 - \beta) \left( \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(t)} \right)$
- By incorporating the historical average $\bar{\alpha}_k^{(t-1)}$ into the current merging coefficients, the model weights for the current batch become dependent on the inputs processed in previous batches.
- This introduces **statefulness and temporal dependency** during inference. The prediction on a given input image will vary depending on what other tasks or images were processed by the server minutes or hours prior. If a server receives a burst of MNIST images, the model weights will shift toward MNIST; if it then receives a CIFAR-10 image, it will be processed with weights heavily biased toward MNIST.
- This violates a core tenet of standard machine learning serving (statelessness and sample independence), making validation, safety monitoring, and debugging virtually impossible.

### 2. Discarding Task-Specific Fine-Tuning in early Layers
To resolve the "First-Block Paradox", the first block of the transformer is kept static and runs with base model parameters $W_{base}^{(1)}$.
- This assumes that task-specific fine-tuning did not alter low-level feature extraction in the first block, or that any alterations are unimportant.
- In reality, first-layer features often adapt to domain-specific statistics (e.g., color distribution in CIFAR-10/SVHN vs. grayscale in MNIST). By forcing the first block to be static, CAM-Router discards all task-specific, low-level adaptations developed by the experts during fine-tuning. No ablation study is conducted to evaluate the performance loss caused by freezing the first block.

### 3. Real Memory Footprint vs. Claimed Savings
The paper claims that dynamic model merging "carries zero additional memory... during inference since all expert parameters must remain resident... combining multiple expert models into a single standard model instance during inference".
- This is a logical contradiction. If the merging coefficients are computed **dynamically on-the-fly for each sample/batch**, then the system must perform the weight summation ($W_{merged} = W_{base} + \sum \alpha_k V_k$) *during the forward pass*.
- To do this, the weights of all $K$ expert models (or task vectors $V_k$) must be resident in GPU memory so they can be summed.
- Therefore, the VRAM footprint is exactly the same as keeping all $K$ expert models in memory, which completely invalidates any "zero additional memory" or "highly scalable memory" claims.
- If all experts must be kept in memory, one could simply run standard routing of activations to the active expert's forward pass. This would yield the upper-bound reference accuracy (85.85% Joint Mean Acc) without any of the representational collapse (53.07% Joint Mean Acc) or latency overhead associated with summing giant parameter tensors on-the-fly.

### 4. Overstated and Inconsistent Claims in the Abstract
There are major discrepancies between the numerical results claimed in the Abstract and those reported in the actual experiments:
- **Joint Mean Accuracy:** The Abstract claims **57.07%** (+15.10% improvement over the 41.97% static baseline). However, Table 4 and Section 4.2 show the Joint Mean Accuracy of CAM-Router is **53.07%** (+11.10% improvement). The authors appear to have copy-pasted the SVHN accuracy (which is exactly 57.07% in Table 4) as the Joint Mean Accuracy in the abstract.
- **Occlusion Accuracy:** The Abstract claims **53.63%** under up to 80% patch masking. Table 5 shows it is actually **50.57%**.
- **Batch Heterogeneity Accuracy:** The Abstract claims **55.47%** under $B=256$. Table 6 shows it is actually **54.30%**.
These inconsistencies represent a major failure in academic rigor and significantly overstate the performance of the proposed method in the most prominent section of the paper.

## Reproducibility
The authors state that they evaluate on a compact Vision Transformer (`vit_tiny_patch16_224`) coordinate-space sandbox. While they list architectural parameters and configuration choices, they do not provide links to source code or describe the exact implementation details of the "simulation sandbox". Since standard `vit_tiny_patch16_224` has 12 layers and they describe a 14-layer backbone, there is an unexplained discrepancy that makes exact replication difficult without access to the custom codebase.
