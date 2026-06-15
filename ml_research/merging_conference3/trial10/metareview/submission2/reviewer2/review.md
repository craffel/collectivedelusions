# Peer Review: Layer-Decoupled Stateful Kinetics (LDS-Kinetics) for Dynamic Model Merging

## Summary of the Paper
The submission introduces **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**, a framework designed to improve test-time ensembling of parameter-efficient low-rank adapters (LoRAs) during sequential multi-task workloads. While prior stateful ensembling routers smooth out sample-level coordinate noise and routing jitter by modeling routing trajectories as a temporal kinetics process, they enforce **spatial homogeneity**—using a single global ensembling weight vector across all layers. LDS-Kinetics challenges this assumption by partitioning ensembling layers into depth-decoupled blocks (or layers), each maintaining separate, independent concentration states that evolve according to block-specific parameters.

To prevent transductive overfitting on short calibration streams ($T=32$), the authors derive a unified learning-theoretic complexity penalty from Catoni's $\beta$-mixing PAC-Bayesian bound. Adhering to an empirical philosophy, the authors evaluate LDS-Kinetics inside a 14-layer coordinate sandbox and on a physical 6-layer sequence model, deconstructing temporal-spatial ensembling dynamics. Their analysis reveals a **"tempo-gradient"** along the network's depth (early layers learn fast adaptation tempos, and deeper layers learn slow, low-pass filtering tempos). They also identify an Adam-specific sign-symmetry optimization pathology that traps unregularized decoupled models in a degenerate lockstep path, and demonstrate that under non-linear activation propagation (GELU + LN), temporal kinetics are mathematically required to prevent compounding representational drift.

---

## Strengths and Weaknesses

### Major Strengths
1. **Compelling Conceptual Motivation**: Breaking the spatial homogeneity assumption is a logical and elegant step forward for stateful dynamic model merging. It aligns perfectly with the established deep learning paradigm that representation spaces at different depths evolve and require different ensembling tempos.
2. **Deep Optimization and Mathematical Insights**: The paper's deconstruction of why unregularized decoupled routing fails (Section 4.3.1) is brilliant. The authors identify that identical SABLE-grounded initializations and correlated gradients trap blocks in a permanent, degenerate lockstep path under the Adam optimizer due to its sign-based first update step. They show that Catoni's PAC-Bayesian complexity penalty elegantly breaks this symmetry during optimization while simultaneously preventing transductive overfitting.
3. **Exhaustive Empirical Validation**: The paper performs extensive sweeps inside a 14-layer coordinate sandbox and on a physical 6-layer sequence model. It includes 9 baseline configurations (including new spatial-only stateless baselines to isolate temporal kinetics), multi-dimensional sweeps of noise, overlapping manifolds, expert pool scaling ($K$), and calibration sequence lengths ($T$).
4. **Systems-Grounded Relevance**: The authors address crucial engineering details, demonstrating that running independent block-wise recurrences can be executed in parallel as batched tensor updates, making LDS-Kinetics statistically latency-neutral compared to a single global router. They also address KV-cache coherence during autoregressive text generation.

### Major Weaknesses
1. **Critical and Unacceptable Citation/Attribution Negligence**: The manuscript contains severe, historically inaccurate, and inappropriate attributions of key machine learning concepts:
   * **Federated Learning (Section 2.1)**: The authors state: *"Early methods relied on simple parameter averaging (e.g., in federated learning~\cite{mitchell80, langley00})."* Neither Tom Mitchell (1980, Rutgers Technical Report on learning generalizations) nor Pat Langley (2000, ICML editorial on paper writing) is remotely related to federated learning or parameter averaging! Federated learning was introduced decades later by McMahan et al. (AISTATS 2017). Citing Mitchell and Langley for this is a shocking scholarly error.
   * **Low-Rank Adaptation (Section 1)**: The authors cite Richard O. Duda, Peter E. Hart, and David G. Stork's classic textbook *Pattern Classification* (2nd Edition, 2000) for Low-Rank Adaptation (LoRA) instead of Edward Hu et al. (ICLR 2022). While `lora` is in the bibliography, it is never cited in the text.
   * **Deep Learning Layer Dynamics (Section 2.4)**: The authors cite 1983 and 2000 books (`MachineLearningI`, `DudaHart2nd`) for modern deep neural network layer representation behaviors (such as layers forming modular semantic concepts or capturing low-level features). Modern deep learning representation dynamics were obviously not studied in 1983 or 2000. Correct seminal representation analysis works like SVCCA (Raghu et al., 2017) or CKA (Kornblith et al., 2019) are in the bibliography but are completely ignored in the text.
2. **Careless formatting / Compilation Artifacts**: In Section 1, the authors cite SABLE, ChemMerge, and PAC-Kinetics using `\cite{anonymous}` (e.g., *"SABLE~\cite{anonymous} projects intermediate..."* and *"ChemMerge~\cite{anonymous} and PAC-Kinetics~\cite{anonymous} were introduced"*). However, in later sections, they cite these exact works correctly as `sable_2024`, `chemmerge_2026`, and `pac_kinetics_2026`. Leaving `anonymous` placeholders in the final draft demonstrates a severe lack of proofreading care.
3. **Theoretical Assumption Violation**: The PAC-Bayesian complexity penalty is derived from Catoni’s bound, which strictly assumes a stationary $\beta$-mixing stochastic process. However, the sequential serving workloads evaluated in the paper are highly non-stationary, featuring sudden task switches. While the authors openly acknowledge this limitation and provide an elegant systems-level justification (using $Sim_t$ as a dynamic flush mechanism during serving to handle non-stationarity while retaining the mathematical regularization benefits of the bound), a formal theoretical extension to non-stationary environments is missing.

---

## Dimension Ratings and Justifications

### 1. Soundness: Excellent
* **Justification**: The technical claims and experimental methodology are exceptionally sound and rigorous. The math is correct, the baseline coverage is outstanding, and the paired $t$-tests across 10 seeds provide strong statistical backing for the results. The deconstruction of Adam's lockstep update path and the physical sequence model validation are brilliant.

### 2. Presentation: Fair
* **Justification**: While the text is beautiful and easy to read, the overall scholarly polish is significantly compromised by the severe, historically incorrect citation blunders (FL, LoRA, and Deep Layer Dynamics) and the careless `\cite{anonymous}` placeholders in the introduction.

### 3. Significance: Good
* **Justification**: Dynamic model merging and multi-task serving are highly active and relevant areas in machine learning. By showing that network depth plays an active, decoupled role in ensembling kinetics, the paper opens up a new avenue of temporal-spatial routing and offers valuable interpretability insights (the "tempo-gradient").

### 4. Originality: Fair
* **Justification**: Conceptually, decoupling stateful kinetics is a logical extension of prior stateful routers (PAC-Kinetics) and well-known layer dynamics. However, the authors fail to accurately map and describe the landscape of their field in the related work, misattributing several foundational concepts and ignoring correct works listed in their own bibliography.

---

## Overall Recommendation

**Rating: 3: Weak Reject**

**Justification**: 
The core technical contribution of LDS-Kinetics is exceptionally strong, rigorous, and empirically sound. The deconstruction of Adam's sign-symmetry pathology, the deconstruction of the "tempo-gradient", and the physical LoRA validation are outstanding scientific achievements. 

However, from a scholarly standpoint, the submission in its current state cannot be accepted. A machine learning paper that fails to accurately describe the landscape of its own field, leaves broken `\cite{anonymous}` placeholders in the introduction, and makes preposterous, historically inaccurate attributions—citing a 1980 Rutgers TR and 2000 ICML editorial for Federated Learning, and a 2000 pattern classification textbook for LoRA—fails to meet the standard of academic rigor expected at a premier conference.

These errors are particularly frustrating because the correct seminal papers (`fedavg2017`, `lora`, `svcca2017`, `cka2019`) are already listed in the authors' bibliography file (`references.bib`) but were completely ignored in the text. I strongly encourage the authors to perform a thorough, rigorous revision to completely correct their literature contextualization and citation mappings. If these critical scholarly flaws are completely resolved, this paper would be a strong candidate for acceptance.

---

## Constructive Questions and Feedback for the Authors

1. **Correction of Citations**: Please completely replace the historically inaccurate citations in the text:
   - In Section 2.1, replace the citation of `mitchell80` and `langley00` for federated learning with the correct reference `fedavg2017`.
   - In Section 1, replace the citation of `DudaHart2nd` for LoRA with the correct reference `lora`.
   - In Section 2.4, replace the citation of `DudaHart2nd` and `MachineLearningI` for deep learning layer dynamics with the correct references `svcca2017` and `cka2019`.
   - In Section 1, resolve the inconsistent `anonymous` placeholders by replacing them with the correct citations (`sable_2024`, `chemmerge_2026`, `pac_kinetics_2026`).
2. **Non-Stationarity**: Could you expand on how the PAC-Bayesian bound's generalization guarantees might degrade under frequent, non-stationary task switches? Providing a brief mathematical characterization of this degradation in the appendix would make the theoretical narrative significantly stronger.
3. **KV-Cache Analysis**: The discussion regarding KV-cache coherence under autoregressive generation is highly insightful. Do you have any plans to conduct an empirical evaluation of KV-cache cosine similarity or perplexity degradation across sequential text generation steps?
