# Summary of the Paper

## Main Topic
The paper systematically investigates how model merging and Low-Rank Adaptation (LoRA/QLoRA) interact with downstream Post-Training Quantization (PTQ) constraints (specifically 8-bit and 4-bit formats) required for hardware deployment on resource-constrained edge devices. It exposes a methodological blindspot in existing model-merging literature—which typically evaluates models only in full precision (FP16/FP32)—and deconstructs the **"Re-Quantization Silence"** phenomenon, where low-bit quantization of merged models can round task-specific adapter updates to zero due to a severe magnitude mismatch between base weights and adapter updates.

## Approach
To investigate and address this problem, the authors introduce:
1. **The Re-Quantization Auditing (RQA) Framework:** A systematic, multi-axial auditing protocol evaluating merged model performance across different quantization bit-widths (4-bit vs. 8-bit), granularities (Per-Tensor vs. Per-Channel), and formats (Symmetric vs. Asymmetric).
2. **Two Proposed Mitigations:**
   - **Scale-Adaptive Weight Shifting (SAWS):** A data-free, closed-form scaling method that pre-scales the low-rank updates using a layer-wise norm ratio $\gamma^l = \alpha \frac{\|\tilde{W}_0\|_F}{\|\Delta W_{\text{merged}}\|_F}$ before merging to help them survive rounding. It applies a closed-form scalar weight alignment factor $c^l$ to correct output scaling at inference.
   - **Quantization-Aware Adapter Coefficient Search (QA-ACS):** An optimization-based test-time adaptation (TTA) method that searches for optimal layer-wise blending coefficients $\Lambda$ directly through the quantization operator by minimizing prediction entropy on a tiny calibration set ($N=16$), backpropagating gradients via the Straight-Through Estimator (STE).
3. **Rigorous Self-Critical Analysis:**
   - Proving the **Representation Scale Preservation Dilemma** in SAWS: demonstrating that attempting to mathematically scale back the entire output dynamically to preserve activation scale collapses the pre-trained base representations because $\gamma^l$ is very large (scaling down base representations by a factor of 10 to 100). Thus, SAWS operates via selective task-vector boosting.
   - Exposing the risk of **unsupervised entropy collapse** in QA-ACS under high quantization noise, where prediction entropy minimization collapses the network's capacity.
   - Exposing the **Straight-Through Estimator (STE) gradient mismatch** under low-bit constraints, and showing that the choice of optimizer (Adam vs. SGD) acts as a critical mitigation for gradient noise.
   - Deconstructing the **double quantization format-shift noise** confounding variable.
   - Proposing a **Zero-Interference RQA Protocol** to decouple task interference from downstream quantization noise.

## Key Findings
1. **Quantization Granularity Bifurcation:** The "Re-Quantization Silence" is highly dependent on the quantization grid granularity. Standard per-channel grids provide inherent protection, rendering naive, unmitigated re-quantization nearly lossless (dropping only 0.15% to 0.30% in INT8, and 1.80% in INT4 symmetric). catastrophic collapse only occurs under aggressive, sub-optimal per-tensor configurations (dropping up to 8.6% accuracy).
2. **Double Quantization Noise Confounder:** Transitioning the pre-trained base model weights from 4-bit NormalFloat (NF4) during training to a target linear format (e.g., INT8/INT4) for deployment introduces a systematic representation error (Absolute Frobenius error increase of 16.5% to 29.4% on ViT-Base), completely independent of any adapter updates.
3. **Robustness of Test-Time Optimization (TTA):** TTA methods (AdaMerging and QA-ACS) are highly robust under standard per-channel configurations, achieving the best overall performance and outperforming data-free scaling. However, they run a risk of minor local performance decay (unsupervised entropy decay) under severe per-tensor noise, which can be stabilized via supervised tuning or basic coefficient regularization.
4. **Decoupling Quantization Noise from Task Interference:** By auditing individual unmerged quantized experts, the authors find that under per-channel grids, the individual experts suffer almost zero degradation (0.70%-0.85% drop). This reveals that the low performance of the merged model under per-channel formats is driven entirely by pre-existing weight-space task interference, and NOT by quantization erasure. Conversely, under per-tensor grids, individual experts lose 10.90%, showing that per-tensor grids are highly destructive even without task interference.

## Explicitly Claimed Contributions (with Evidence)
1. **Exposing and mathematically deconstructing "Re-Quantization Silence":** Shown through magnitude mismatch equations (Section 3.2, Eq. 8-9) and empirical evaluations showing a catastrophic drop in unmitigated per-tensor configurations (Table 6).
2. **The Multi-Axial RQA Framework:** Evaluated on a Vision Transformer backbone (\texttt{vit\_tiny}) across four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) under 4 quantization configurations (Tables 2-6).
3. **Characterization of the Quantization Granularity Bifurcation:** Proving that per-channel grids naturally preserve adapter updates while per-tensor grids erase them (Section 4.3).
4. **Formulation and critical analysis of SAWS:** Detail of the closed-form formulation (Section 3.3, Eq. 10-15) and empirical results showing it is robust in per-channel settings but degrades slightly under per-tensor grids (Tables 3-6). Proof of the Representation Scale Preservation Dilemma (Section 3.3.1, Eq. 16-17).
5. **Formulation and critical analysis of QA-ACS:** Detail of the prediction entropy optimization and STE formulation (Section 3.4), documenting the risk of unsupervised entropy collapse under high noise (Section 3.4.1) and STE gradient mismatch (Section 3.4.2).
6. **Double Quantization Audit:** Showing Relative Frobenius error of base weights under format shifts (Table 1, Section 3.2.1).
7. **Alternative Baseline Analysis (Co-existence):** Highlighting the latency and memory overheads of dual-path inference baselines (Section 3.2.2).
8. **Decoupling Audit:** Proving that per-channel quantization does not erase task updates on individual unmerged experts (Table 7, Section 4.3.6).
