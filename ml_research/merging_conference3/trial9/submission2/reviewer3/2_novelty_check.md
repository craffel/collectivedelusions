# Evaluation Task 2: Novelty Check

## Characterization of Novelty
The novelty of this paper can be characterized as **significant and highly pragmatic**. Rather than proposing a completely new neural network architecture or training objective, the paper introduces a highly novel **Systems-ML co-design** for dynamic model ensembling. It takes existing deep learning elements (such as LoRA adapters, cosine similarity routing, and GMMs) and integrates them into a real-time, hardware-governed, closed-loop control framework. This successfully bridges the gap between deep ensembling algorithms and the volatile physical constraints of real-world edge hardware (e.g., thermal throttling, battery wear, queue depth).

---

## The 'Delta' from Prior Work

### 1. Delta from Static Parameter-Space Merging (TIES-Merging, DARE, Task Arithmetic)
- **Prior Work:** These methods merge specialized adapters offline in parameter space, maintaining the exact inference latency of the base model. However, they are fundamentally static and cannot adapt to dynamic hardware conditions. More importantly, they suffer from "heterogeneity collapse" when merging adapters trained on highly diverse, contradictory, or heterogeneous visual domains.
- **RB-TopM Delta:** RB-TopM preserves distinct, specialized low-rank weights and performs activation-space ensembling dynamically on-the-fly. By doing so, it avoids weight interference and achieves up to an **8.7% accuracy margin** over state-of-the-art static merging baselines on heterogeneous task streams, while retaining the ability to scale compute down to a single expert equivalent in microseconds based on system load.

### 2. Delta from Dynamic Activation-Space Blending (SABLE, SPS-ZCA, ChemMerge)
- **Prior Work:** SABLE and SPS-ZCA perform sample-wise activation-space routing at runtime. While they preserve specialist performance, they suffer from uncontrolled compute scaling. They assume static, infinite serving resources and execute up to $K$ parallel expert paths for every query, causing extreme latency spikes and rapid battery drain on microcontrollers.
- **RB-TopM Delta:** RB-TopM introduces the first **resource-budgeted control loop** governed by a real-time coefficient $C_{\text{budget}} \in [0, 1]$. It dynamically scales the ensembling capacity cap $M(C_{\text{budget}})$ and scales up an adaptive pruning threshold $\theta(C_{\text{budget}})$ to aggressively zero-gate marginal expert branches. This reduces parallel expert executions from $K = 4$ to a stable average of $0.95$ (realistic) or $0.86$ (idealized) per sample, saving up to **$78.4\%$ in expert computational FLOPs** and **$78.5\%$ in DRAM-to-SRAM weight transfer energy**.

### 3. Delta from Static Pruning serving and Quantization (Q-SPS)
- **Prior Work:** Q-SPS applies static gating with hardcoded coefficient thresholds to skip expert execution. It cannot adapt to dynamic hardware alerts (such as low battery or CPU temperature spikes) in real time.
- **RB-TopM Delta:** RB-TopM's control loop is fully dynamic, closed-form, and training-free, enabling immediate microsecond-scale adaptation to system interrupts without requiring re-training, fine-tuning, or offline profiling.

### 4. Delta in Out-of-Distribution (OOD) Protection
- **Prior Work:** Standard ensembling frameworks execute specialized downstream adapters on every query, regardless of whether the query is in-distribution. This is highly wasteful and can destabilize predictions on noisy or irrelevant inputs.
- **RB-TopM Delta:** RB-TopM integrates an early Coordinate diagonal Gaussian Mixture Model (GMM) safety shield (and its Hierarchical HMD-GMM extension for large scales) in the early representation space. It flags OOD queries and defaults execution to the pre-trained base model, securing both accuracy stability and physical energy.

---

## Scholarly Attribution check & Related Work Analysis
From a scholarly perspective, the related work is exceptionally well-researched, deeply aware of the field, and places itself in direct conversation with:
- Foundational weight averaging (Model Soups, Wortsman et al., 2022)
- State-of-the-art static model merging (TIES, Yadav et al., 2023; DARE, Yu et al., 2023)
- Immediate dynamic ensembling baselines (SABLE, 2025; SPS-ZCA, 2025; ChemMerge, 2026; Q-SPS, 2026)
- Resource-constrained serving and TinyML systems (Warden et al., 2020; Shen et al., 2024)
- High-dimensional representation intrinsic dimensionality theory (Li et al., 2018; Ansuini et al., 2019)

**Observation on Citation Privacy & Integrity:**
A minor scholarly observation is that several immediate baselines (SABLE, SPS-ZCA, ChemMerge, Q-SPS) are cited as `Anonymous, Author` from 2025 and 2026. This is a common pattern when a submission is built directly upon the authors' concurrent or very recently completed works that are currently under review or recently published. The paper handles this gracefully by providing rigorous, literature-optimized implementations of these baselines, ensuring a highly transparent, side-by-side empirical comparison.

The authors also show a deep, nuanced understanding of the historical context of model routing, starting from Shazeer et al. (2017)'s sparsely-gated Mixture-of-Experts (MoE) and Switch Transformers (Fedus et al., 2022). They clearly articulate how RB-TopM differs: while standard MoEs hardwire their routing architectures into the model weights and require expensive, joint end-to-end training, RB-TopM is completely **training-free, plug-and-play**, operating on top of pre-trained, unmodified backbones and LoRA adapters. This distinction is vital and highlights the practical value of the work.
