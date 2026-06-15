# 4. Empirical Evaluation Check

## 4.1 Experimental Design and Settings
The experimental framework is well-structured and designed to test model merging under extreme task conflicts and hardware deployment constraints.

- **Backbone Architecture**: Vision Transformer Tiny (`vit_tiny_patch16_224`) containing 12 transformer blocks, with 48 targeted dense weight matrices across the self-attention (`qkv`, `proj`) and MLP (`fc1`, `fc2`) modules.
- **Expert Diagnostics**: The paper reports diagonal expert performance ($81.00 \pm 0.82\%$ on MNIST, $74.67 \pm 1.25\%$ on FashionMNIST, $71.67 \pm 4.03\%$ on CIFAR-10, and $85.33 \pm 2.87\%$ on SVHN) and collapses to near-random guessing ($\approx 10\%$) on off-diagonal evaluations. This confirms a highly conflicting multi-task setup, posing a severe challenge for naive parameter merging.
- **Robustness Reporting**: All metrics represent the mean and standard deviation computed over **3 independent random seeds** (seeds 42, 100, and 2026), providing excellent statistical confidence.

## 4.2 Critique of Key Results

### 4.2.1 The PolyMerge Empirical Gap
The most obvious experimental outcome is that **PolyMerge** remains the strongest empirical baseline, outperforming PhaseMerge variants across all configurations:
- **FP32**: PolyMerge achieves $48.00 \pm 1.62\%$, whereas U-PhaseMerge ($r=1$) yields $42.83 \pm 1.76\%$ and PhaseMerge ($r=2$) yields $40.75 \pm 1.43\%$.
- **8-bit PTQ**: PolyMerge achieves $48.00 \pm 1.47\%$, whereas U-PhaseMerge achieves $42.33 \pm 1.76\%$.
- **4-bit PTQ**: PolyMerge achieves $43.42 \pm 1.30\%$, whereas U-PhaseMerge achieves $37.42 \pm 1.94\%$.

This performance gap of approximately $5\%$ to $6\%$ absolute accuracy indicates that, in this setup, depth-wise polynomial scaling (PolyMerge) provides a much stronger macroscopic prior than unconstrained frequency-domain phase shifts.

### 4.2.2 The Defeat of Static Frequency Filtering
A massive positive result is the complete failure of the static **FREE-Merging** baseline, which achieves only $27.17 \pm 1.96\%$ in FP32. U-PhaseMerge ($r=1$) outperforms FREE-Merging by **$+15.66\%$ absolute accuracy**. This provides convincing proof that static, non-adaptive frequency filtering collapses because it cannot adjust to the alignment dynamics of conflicting experts. Differentiable, adaptive phase synchronization is essential.

### 4.2.3 Analysis of the Overfitting-Optimizer Paradox (Table 2)
In the sample complexity sweep:
- At extreme data scarcity ($M=4$), unconstrained AdaMerging achieves $40.75 \pm 1.24\%$, while U-PhaseMerge ($r=1$) achieves a highly competitive $42.42 \pm 1.64\%$. This shows that the conjugate symmetry mask and the low parameter footprint (192-D) successfully act as a regularizer.
- At $M=32$ without $L_2$ decay, both U-PhaseMerge and PhaseMerge ($r=2$) exhibit significant optimization instability, dropping to $39.23 \pm 4.92\%$ and $38.96 \pm 4.13\%$ respectively, with extremely high standard deviations across seeds.
- However, once the soft $L_2$ phase decay penalty is applied ($\gamma=10^{-4}$), it dramatically reduces optimization variance (slashing the standard deviation of PhaseMerge from $4.13\%$ to a highly stable $1.34\%$) and improves accuracy to $40.67 \pm 3.65\%$ and $42.00 \pm 1.34\%$, respectively. This confirms that proximity-constrained wave superposition is essential for stabilizing frequency-space test-time optimization.

### 4.2.4 Robustness under Schema Shift (Table 3)
When calibrated under 8-bit but evaluated under a 4-bit deployment schema:
- AdaMerging drops by $4.17\%$ (from $41.67\%$ to $37.50\%$).
- PolyMerge drops by $4.58\%$ (from $48.00\%$ to $43.42\%$).
- U-PhaseMerge ($r=1$) drops by $4.91\%$ (from $42.33\%$ to $37.42\%$).
- PhaseMerge ($r=2$) drops by $3.91\%$ (from $40.83\%$ to $36.92\%$).

While the PhaseMerge variants do not outperform PolyMerge on absolute terms, they demonstrate highly competitive generalizability to target schema shifts, confirming that frequency-space phase transformations remain robust to discretization boundaries.

## 4.3 Key Evaluation Limitations
We identify two major limitations in the empirical evaluation:
1. **Scale of Sandbox Evaluation**: Downstream experts are fine-tuned on only 500 samples, and the test set evaluations are subsampled to 100 samples per task. While this is appropriate for fast prototyping and CPU validation, evaluating over the full test sets is necessary to claim complete statistical reliability.
2. **Evaluation on Modern Foundation Models**: The experiments are confined to a tiny Vision Transformer (`vit_tiny`, 5.7M parameters). Demonstrating PhaseMerge on multi-billion parameter LLMs (e.g., LLaMA, Mistral) or large-scale generative diffusion experts is essential to prove the scalability claims made in Appendix A.

## 4.4 Verdict on Empirical Evaluation
The empirical evaluation is rated as **good**. The comparative baseline selection is excellent, the statistical reporting with standard deviations over 3 seeds is highly rigorous, and the analysis of the PolyMerge gap is remarkably honest and theoretically insightful. However, the evaluation's limited scale and reliance on tiny subsampled datasets restrict the strength of the empirical conclusions.
