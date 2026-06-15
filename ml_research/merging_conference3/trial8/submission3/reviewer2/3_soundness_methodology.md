# Soundness and Methodology Evaluation

## Technical Flaws and Methodological Weaknesses

### 1. Misleading "Training-free" Claim vs. Reliance on QAT
The submission repeatedly characterizes **SA-QAB** as a "**training-free, forward-only**" framework in the Abstract, Section 1, and the start of Section 3. However, Section 4.3 (page 7) reveals a severe contradiction:
- Pure post-training quantized (PTQ) SA-QAB only achieves **50.00% joint accuracy**, which represents a massive **34.90% accuracy drop** compared to the unquantized blending baseline (84.90%).
- To obtain the highlighted headline result of **77.50% joint accuracy** (which is marketed in the Abstract and throughout the paper as a "+58.90% absolute improvement over post-merge quantization"), the authors must perform **Quantization-Aware Fine-Tuning (QAT)** for 5 epochs using Straight-Through Estimation (STE) over the adapters and classification heads.
- This is a fundamental flaw in the paper's narrative and claims. A method cannot be advertised as "training-free, forward-only" if its actual training-free variant suffers from a severe performance degradation (50.00% accuracy) and relies on active backpropagation (fine-tuning) to be competitive. The authors attempt to frame QAT as a "highly practical compromise," but this fails to address the fact that the primary selling point of the paper is contradicted by the experimental results.

### 2. Distribution Mismatch in Quantization Scale Recovery (QSR)
The proposed QSR mechanism (Equation 4) pre-computes calibration scaling factors $\beta_k^{(l)}$ as the expected ratio of unquantized to quantized adapter activations:
$$\beta_k^{(l)} = \frac{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_FP}_k(h_s^{(l-1)}) \|_2 \right]}{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_Quant}_k(h_s^{(l-1)}) \|_2 \right]}$$
Crucially, the authors disclose that during calibration, the input activation $h_s^{(l-1)}$ is extracted from the clean, **full-precision (FP16)** network stream rather than the quantized base network. They justify this by stating that calibrating over quantized features "corrupts the reference expectation with compounding noise, destabilizing the scale factors."
- This is a significant methodological flaw. If calibrating over quantized features is so fragile that compounding noise "destabilizes" the scale factors, it suggests the proposed scaling mechanism is highly unstable and mathematically brittle.
- At test-time, these scaling factors $\beta_k^{(l)}$ are applied to activations $h_b^{(l-1)}$ coming from the **INT4-quantized base backbone**. This introduces a fundamental distribution mismatch. There is no mathematical guarantee that scaling factors computed on clean, unperturbed FP16 activations will remain optimal, or even valid, when applied to noisy INT4-quantized activations. 

### 3. Artificial Subspace Orthogonality in the Coordinate Sandbox
In Section 4.1 (Idealized Subspace Orthogonality Limitation and Stress Test), the authors admit that the Coordinate Sandbox synthetically generates task representations in **completely disjoint (orthogonal) coordinate subspaces** (e.g., Task 0 in channels $[0:48]$, Task 1 in $[48:96]$, etc.).
- This artificial separation makes the dynamic routing task (Q-ZCA) trivial. Since the features are mathematically guaranteed to occupy non-overlapping channels, computing centroids at Layer 3 will naturally yield perfectly distinct, orthogonal vectors. 
- In real-world neural networks, task-specific features are highly entangled and share the same embedding channels. While the authors perform an overlap sweep ($\Omega$) in Appendix B, the reliance on an orthogonal coordinate-based simulation suite as the primary quantitative benchmark (Table 2) severely undermines the credibility and soundness of the reported numbers.

---

## Clarity and Reproducibility

### Clarity of Description
The mathematical descriptions of Decoupled Heterogeneous Quantization (DHQ), Quantized Zero-Shot Centroid Alignment (Q-ZCA), and Quantization Scale Recovery (QSR) are clearly presented in Section 3. However, the writing is heavily padded with convoluted and redundant jargon (e.g., "Spectacular +58.90% absolute accuracy improvement," "neutralize representation scale mismatches," "completely eliminates heterogeneity collapse"), which detracts from its scientific objectivity.

### Reproducibility Analysis
- **Calibration Set Details**: The calibration set size is specified as 64 samples per task, and the temperature $\tau$ is listed as $0.001$. However, the exact optimization parameters for the 5-epoch frozen-base QAT (such as learning rate, optimizer, weight decay) are not fully detailed in the main text.
- **Microcontroller Emulation**: The hardware profiling relies on a "cycle-accurate microcontroller profiling emulation" on an STM32H753XI. Since the emulator setup, CMSIS-NN compiler flags, and memory layout configurations are not open-sourced or fully described, reproducing the precise cycle counts, latencies (0.836 ms), and energy estimates (0.3035 mJ) in Table 3 is practically impossible for an outside researcher.
- **Real-Pixel Feasibility Study**: Section 4.3 briefly describes evaluations on ViT-Tiny and ResNet-18 but lacks a detailed breakdown of hyperparameters, training schedules for adapters, and centroid extraction details.
