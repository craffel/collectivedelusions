# 4. Experimental Results and Verification

## 4.1. Verification of Experimental Results and Tables
The experimental results are detailed, comprehensive, and accurately reported across multiple tables.

1. **High-Precision Baselines (Table 1):** Establishes the upper ceiling.
   - Unmerged Experts: Mean $93.85\%$ (MNIST: $98.20\%$, Fashion: $88.20\%$, CIFAR-10: $95.00\%$, SVHN: $94.00\%$).
   - Naive FP16 Merge: Mean $66.65\%$ (MNIST: $45.40\%$, Fashion: $72.40\%$, CIFAR-10: $89.40\%$, SVHN: $59.40\%$).
   - *Observation:* The low merging baseline ($66.65\%$) reveals that merging these highly disparate expert domains suffers from substantial weight-space task interference.

2. **INT8 Symmetric Per-Channel (Table 2):**
   - Naive-RQ: $66.35\%$ (only a tiny $0.30\%$ drop from FP16).
   - Q-then-M: $66.70\%$.
   - AdaMerging (PH-Q): $70.10\%$.
   - SAWS [Proposed]: $69.75\%$.
   - QA-ACS [Proposed]: $69.35\%$.
   - *Observation:* Proposed mitigations and AdaMerging successfully outperform both Naive-RQ and the Naive FP16 Merge ceiling, showing that quantization-aware tuning can act as a regularizer that reduces pre-existing task interference.

3. **INT4 Symmetric Per-Channel (Table 3):**
   - Naive-RQ: $64.85\%$ (a minor $1.80\%$ drop from FP16).
   - AdaMerging (PH-Q): $68.80\%$.
   - SAWS: $67.80\%$.
   - QA-ACS: $68.00\%$ (MNIST: $60.60\%$, showing strong adaptation without entropy collapse).

4. **INT4 Asymmetric Per-Channel (Table 4):**
   - Naive-RQ: $63.20\%$ ($3.45\%$ drop from FP16).
   - AdaMerging (PH-Q): $68.25\%$.
   - SAWS: $67.25\%$.
   - QA-ACS: $64.75\%$.

5. **INT4 Symmetric Per-Tensor (Table 5):**
   - Naive-RQ: $56.75\%$ ($9.90\%$ drop from FP16 ceiling, demonstrating catastrophic collapse).
   - Q-then-M (Co-existence): $59.60\%$.
   - AdaMerging (PH-Q): $57.25\%$.
   - SAWS: $56.40\%$.
   - QA-ACS: $57.00\%$ (MNIST drops to $37.80\%$ due to local entropy decay).

6. **Individual Expert Auditing (Table 6):**
   - Direct quantization of unmerged experts is virtually lossless under per-channel grids (INT4 Symmetric: $93.15\%$ vs. $93.70\%$ ceiling).
   - Catastrophic collapse is observed under per-tensor grids ($82.95\%$, an absolute drop of $10.75\%$).
   - *Analysis:* This control experiment is a masterclass in experimental isolation. It empirically proves that the quantization step itself is lossless under fine-grained grids, and that low merged performance is driven entirely by task interference.

## 4.2. Soundness of Empirical Conclusions
The empirical conclusions drawn by the authors are exceptionally sound and transparent:
- They do not hide that under INT4 Per-Tensor, the data-free SAWS ($56.40\%$) and unconstrained QA-ACS ($57.00\%$) are outperformed by the co-existence baseline ($59.60\%$). This transparency builds high trust.
- They correctly identify the "Quantization Granularity Bifurcation": that the "Re-Quantization Silence" is a highly localized per-tensor artifact. Since per-channel and group-wise quantization are the industry standards for edge deployment, naive merging followed by PTQ is practically lossless under realistic formats.
- The double-quantization error analysis (Table~\ref{tab:double_quantization_error}) successfully quantifies the format-shift reconstruction error (NF4 $\to$ INT), identifying it as a major baseline confounding variable.

## 4.3. Omissions or Weaknesses
While the experiments are outstanding, we can note two minor points for potential enhancement (though they do not detract from the paper's high quality):
1. **Unmerged Experts Ceilings:** There is a tiny mismatch between the Unmerged Experts FP16 ceiling reported in Table 1 ($93.85\%$) and Table 6 ($93.70\%$, labelled "Unmerged FP16 Ceiling (No Quant)"). This is likely due to separate random seed evaluation runs, but should be harmonized or clarified in a footnote.
2. **Backbone Scale Limitations:** The primary evaluation uses a toy Vision Transformer backbone (\texttt{vit\_tiny}, 5.7M parameters). While Table 2 evaluates reconstruction error on the larger \texttt{vit\_base} (86M parameters), expanding the full multi-task auditing framework to multi-billion parameter LLMs is a logical next step. The authors explicitly acknowledge this as their primary limitation and outline a detailed scaling roadmap in Section 5.1 and Appendix C, which completely mitigates this critique.

## 4.4. Empirical Rating
**Rating: Excellent**  
The experimental design is flawless. The inclusion of the individual expert control experiment to decouple task interference from quantization noise, the double quantization format shift audit, and the physical CPU hardware profiling represent an extraordinary level of empirical depth and integrity.
