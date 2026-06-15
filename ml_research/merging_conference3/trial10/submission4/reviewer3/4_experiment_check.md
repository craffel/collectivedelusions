# Experimental Check: QA-Merge (Quantization-Aware Merge)

## Evaluation of the Experimental Setup and Datasets
The experimental evaluations are conducted inside the **Coordinate Sandbox (ICS)**, a controlled, 14-layer coordinate-space simulator ($D=192$). Orthogonal synthetic task signatures corresponding to four classic visual datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN) are projected into the sandbox. 
- While using synthetic task signatures in a simulated sandbox is a highly valuable tool for isolating and analyzing mathematical and representational coordinate-space dynamics, it lacks the architectural noise and structural complexity of real-world deep neural networks (e.g., Attention heads, LayerNorm scales, autoregressive generation cascades).
- The task expert accuracies are set to standard values (MNIST/Fashion-MNIST: 100%, CIFAR-10: 92.40%, and SVHN: 22.80%). Setting the SVHN accuracy to 22.80% is a clever design choice, as it functions as a "weak expert representational distractor," allowing the authors to verify if the routing layers can successfully isolate and bypass noisy representational pathways.

## Evaluation of the Baselines
The baselines are comprehensive and rigorous:
1. **Expert Oracle (Float32)** provides the absolute representational ceiling.
2. **Uniform Merging (Float32 & Quantized)** represents static ensembling, which is extremely robust but non-adaptive.
3. **SABLE, ChemMerge, Momentum-Merge, and Parametric Router** are evaluated in both their standard **Float32** (full precision) and **Quantized-Naive** (naive INT8/INT4 quantization) formats.
This allows for a direct, side-by-side comparison of: (a) the performance drop due to naive quantization, and (b) the performance recovery achieved by QA-Merge.

## Critique of Results and Supporting Evidence
The empirical results strongly support the paper's core claims:
- **Quantization Collapse Verification:** Tables 1 and 2 demonstrate a clear "Quantization Collapse." Naive low-precision quantization collapses all adaptive ensembling baselines directly to static uniform merging levels (e.g., SABLE naive drops from 76.20% to 73.70% at $\rho=0.2$, matching Uniform's 73.90%). 
- **Near-100% Accuracy Recovery:** Across both small-sample ($N_{\text{cal}} = 64$) and large-sample ($N_{\text{cal}} = 4000$) regimes and various entanglement levels ($\rho$), QA-Merge successfully recovers the Float32 ceilings within 0.1–0.3% absolute accuracy. For instance, for Momentum-Merge (QA-Merge) at $\rho=0.5$ (Table 2), QA-Merge achieves **90.50%**, matching the unquantized ceiling and beating static Uniform by **5.30%** absolute.
- **Weak Expert Isolation:** Non-SVHN queries allocate negligible weight ($\le 0.02$) to the weak SVHN expert, confirming that the scale-invariant cosine similarity gating successfully isolates noisy representational pathways.
- **Sample Complexity Curves (Figure 2):** Tracking performance from $N_{\text{cal}} = 32$ to $4000$ shows that SABLE (QA-Merge) and ChemMerge (QA-Merge) are highly sample-efficient, maintaining stable accuracies under extreme data scarcity, whereas the standard parametric router overfits severely.
- **Trajectory Jitter Comparison (Figure 3):** The bar chart confirms that EF-Smooth successfully suppresses trajectory discretization noise, keeping jitter levels near-zero.
- **On-Device Physical Speedup:** The physical Cortex-M7 (STM32H753XI) benchmark is highly compelling. Running the integer loop in **0.18 ms** versus the Float32 loop in **0.95 ms** represents a massive **5.2x latency speedup** and a **42% power reduction** (to 18 mW), demonstrating the real-world efficiency of QA-Merge.

## Potential Weaknesses and Gaps
1. **No Evaluation on Large Language Models (LLMs):** While the PyTorch toy LoRA experiment (`toy_qamerge_lora.py`) is highly valuable for demonstrating implementation compatibility in deep learning frameworks, there are no end-to-end downstream evaluations of QA-Merge on large language models (such as LLaMA-3 or Mistral) or Vision Transformers. A full downstream evaluation on standard NLP tasks (e.g., GLUE or MMLU benchmarks) would make the empirical claims much stronger.
2. **SmoothQuant Isolation:** The Dynamic Outlier-Aware Activation Scaling (SmoothQuant parameter sweep, Table 4) is validated on a separate, simulated outlier dataset rather than being integrated end-to-end within the main Coordinate Sandbox experiments. While this is justified by the lack of extreme outliers in the sandbox itself, evaluating SmoothQuant on a real transformer layer that naturally contains heavy-tailed outliers (like LLM attention sinks) remains an open empirical gap.
