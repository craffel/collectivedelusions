# Presentation and Impact Evaluation

This document critically evaluates the presentation quality, major strengths, areas for improvement, and overall significance/impact of the paper.

## 1. Major Strengths of the Paper
*   **Exemplary Self-Criticism and Honesty:** 
    The authors are exceptionally transparent, self-critical, and rigorous about their own limitations. They actively deconstruct and expose:
    - The "representation scale preservation dilemma" of SAWS, clarifying that it does not actually preserve scale but succeeds through selective task-vector boosting.
    - The fragility of QA-ACS (entropy collapse) under severe noise.
    - The fact that naive re-quantization is virtually lossless under standard per-channel configurations.
    - The toy nature of their backbone (`vit_tiny`, 5.7M parameters).
    - The severe task interference in their full-precision merging baseline.
*   **Meticulous and Rigorous Scientific Control Experiments:**
    - The **Individual Expert Auditing** control experiment (Table 6) is outstanding. By applying quantization directly to unmerged experts, the authors decoupled quantization-induced degradation from pre-existing task-interference conflicts. This is a brilliant and highly commendable scientific design.
    - The **Double Quantization Noise** format-shift analysis (Table 1) provides a very interesting and valuable characterization of how moving from non-linear NF4 to uniform INT4/8 grids introduces severe representation-space errors.
*   **Sophisticated and Polished Presentation:**
    The paper is exceptionally well-written, clearly structured, and logically organized. The mathematical definitions are formal, and the captions/tables are detailed and descriptive.

## 2. Areas for Improvement (Constructive Critique)
*   **Tone Down Sensationalized Claims and Rebrandings:**
    The paper introduces highly sensationalized terminology (e.g., "dangerous methodological blindspot," "Re-Quantization Silence") to describe a phenomenon that, in practical deployable settings (per-channel), is virtually lossless (dropping only 1.80%). The authors should frame their work more modestly as a study of "hyperparameter scaling and coefficient tuning under quantization" rather than an audit of a "catastrophic blindspot."
*   **Resolve the Utility of the Proposed Methods:**
    - **QA-ACS:** The paper proposes QA-ACS as a core contribution, but the experiments show it is strictly worse than the simpler AdaMerging (PH-Q) baseline. Proposing a more complex method that underperforms existing simple methods is highly questionable. The authors should either find a configuration where QA-ACS outperforms AdaMerging (PH-Q) or de-emphasize QA-ACS as a main contribution and focus on why FP16 optimization is superior.
    - **SAWS:** Since SAWS is functionally just scaling up the adapter updates (since $c^l \approx 1$ and $1/\gamma^l$ is not applied), the authors should simplify the mathematical framing. They should compare SAWS directly against a simple full-precision task-vector scaling baseline to prove that the computed $\gamma^l$ provides an advantage over manual tuning of the LoRA scaling factor.
*   **Upgrade the Quantization Pipeline:**
    The "Double Quantization" reconstruction error of 30.40% in INT8 (Table 1) is a direct artifact of using a naive, unclipped symmetric quantizer. The authors should evaluate their models using standard post-training quantization techniques that incorporate percentile clipping or optimal step-size search. Evaluating against a sub-optimal quantizer limits the validity of their conclusions.
*   **Scale Beyond Toy-Scale Architectures:**
    Evaluating a 5.7M parameter ViT on MNIST, FashionMNIST, CIFAR-10, and SVHN is extremely small-scale. Since model merging and QLoRA are primarily deployed on Large Language Models (7B+ parameters) or large diffusion models, the authors must validate their hypotheses on at least a small LLM (e.g., LLaMA-1B or Pythia-1.4B) across standard NLP/instruction-following tasks.

## 3. Presentation Quality Rating
*   **Rating: Excellent**
    The overall presentation, clarity of writing, mathematical rigor, and structured narrative are excellent. The authors have done a fantastic job of organizing the paper, presenting tables, and writing detailed, self-critical descriptions of their methods.

## 4. Significance and Impact Rating
*   **Rating: Fair**
    Despite the high quality of writing and excellent scientific control experiments, the practical significance of the contribution is quite modest:
    1.  **Standard per-channel quantization is lossless:** The paper reveals that under standard per-channel/group-wise configurations (which are universally used in actual deployments), the "Re-Quantization Silence" only causes a minor 1.80% drop in INT4. 
    2.  **Proposed methods fail in the only collapse regime:** In the per-tensor INT4 regime where catastrophic collapse actually occurs, the proposed SAWS method is completely ineffective (performing worse than naive re-quantization), and QA-ACS performs worse than AdaMerging.
    3.  **Toy-scale limitation:** The experiments are restricted to a toy vision setup with severe baseline task interference, making it unclear if the observed dynamics generalize to modern LLM or Diffusion deployments.
    Consequently, the paper's actual impact on practical machine learning deployment is quite limited.
