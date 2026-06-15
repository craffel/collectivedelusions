# Evaluation 2: Novelty and Literature Check

## Key Novel Aspects and "Delta" from Prior Work
The core conceptual novelty of this work is **breaking the spatial homogeneity assumption** in stateful dynamic model merging. 
* **State of the Art (Homogeneous Stateful Routing)**: Existing stateful routing frameworks (such as ChemMerge and PAC-Kinetics) apply a single global ensembling weight vector uniformly across all network depths. This assumes that the optimal blending trajectory is identical at every layer, forcing a sub-optimal trade-off between the fast adaptation needed at early layers (to track workload switches) and the stability needed at late layers (to prevent logit-level jitter).
* **Proposed Delta (Layer-Decoupled Stateful Routing)**: LDS-Kinetics introduces independent state recurrences across disjoint blocks of layers. This allows separate temporal scales to emerge at different network depths. Early layers can adapt instantly to task transitions, while deeper layers can act as stable, low-pass filters to keep decision logits steady.
* **Empirical Delta**: The paper provides the first systematic deconstruction of the temporal-spatial dynamics of ensembling. By analyzing the learned parameters, it discovers a clear **"tempo-gradient"**—early blocks learn high decay (short memory / high responsiveness), and late blocks learn low decay (high memory / high stability).

---

## Scholar-Lens Critique: Citation, Attribution, and Literature Analysis
From a rigorous, scholarly perspective, the submission exhibits **exceptionally severe and unacceptable errors in the attribution of key machine learning concepts**, as well as careless formatting inconsistencies. These errors indicate a profound lack of historical understanding and a sloppy literature review, which severely damages the academic authority of the manuscript.

### 1. Preposterous Attribution of Federated Learning
* **The Error**: In Section 2.1, the authors state:
  > *"Early methods relied on simple parameter averaging (e.g., in federated learning~\cite{mitchell80, langley00})."*
* **The Reality**: 
  - `mitchell80` is Tom Mitchell's Rutgers Technical Report, *"The Need for Biases in Learning Generalizations"* (1980).
  - `langley00` is Pat Langley's editorial paper, *"Crafting Papers on Machine Learning"* (ICML 2000).
  - Neither of these papers is remotely related to federated learning or parameter averaging! Federated learning was introduced decades later by McMahan et al. (AISTATS 2017). The authors' own bibliography contains the correct reference (`fedavg2017`), but they fail to cite it in the text.

### 2. Bizarre Attribution of Low-Rank Adaptation (LoRA)
* **The Error**: In Section 1, the authors state:
  > *"...where task-specific parameter-efficient adapters (such as low-rank adapters or LoRAs~\cite{DudaHart2nd}) are dynamically scaled..."*
* **The Reality**:
  - `DudaHart2nd` is Richard O. Duda, Peter E. Hart, and David G. Stork's classic textbook, *Pattern Classification* (2nd Edition, 2000).
  - LoRA (Low-Rank Adaptation) was proposed by Hu et al. in 2021/2022 (`lora`). While `lora` is present in the bibliography, the authors fail to cite it, erroneously attributing LoRA to a pattern classification textbook from 2000.

### 3. Anachronistic Citation of Deep Learning Layer Dynamics
* **The Error**: In Section 2.4, the authors state:
  > *"Early layers capture low-level, generic structural features, middle layers form modular semantic concepts, and late layers represent class-specific logit statistics~\cite{DudaHart2nd, MachineLearningI}."*
* **The Reality**:
  - `DudaHart2nd` is from 2000. `MachineLearningI` is a volume edited by Michalski, Carbonell, and Mitchell in 1983.
  - Deep learning representation dynamics, feature extraction layers, and modular semantic concept formation were not studied in 1983 or 2000! Citing these books for modern deep neural network layer behaviors is completely inappropriate.
  - The correct literature includes seminal deep representation analysis works like Maithra Raghu et al. (SVCCA, NeurIPS 2017, `svcca2017`) or Simon Kornblith et al. (CKA, ICML 2019, `cka2019`). Both are in the bibliography but completely uncited in the text.

### 4. Sloppy Blind-Submission Residuals
* **The Error**: In Section 1, lines 6 and 8, the authors cite SABLE, ChemMerge, and PAC-Kinetics using `\cite{anonymous}` (e.g., *"SABLE~\cite{anonymous} projects intermediate..."* and *"ChemMerge~\cite{anonymous} and PAC-Kinetics~\cite{anonymous} were introduced"*).
* **The Reality**: 
  - `anonymous` is not defined in `references.bib`, which would cause compilation warnings or errors.
  - In Section 2 (Related Work) and other sections, the authors cite these exact works correctly as `sable_2024`, `chemmerge_2026`, and `pac_kinetics_2026`. This inconsistency demonstrates severe carelessness in proofreading and formatting.

---

## Characterization of Novelty
* **Conceptual Novelty**: **Incremental to Moderate**. Decoupling states across layers is a natural extension of existing stateful routing frameworks (e.g., PAC-Kinetics) and well-known layer heterogeneity in deep learning. However, combining them is logical and elegant.
* **Empirical Novelty**: **Significant**. The empirical discovery and thorough deconstruction of the "tempo-gradient" (quantifying that early layers have short memory and deeper layers have stable, long-term memory) is highly novel and provides valuable insights into sequential multi-task serving dynamics.
* **Overall Rating on Novelty**: While the empirical insights are strong, the novelty is overshadowed by **inexcusable negligence in literature contextualization**. A paper that fails to accurately describe the landscape of its own field and misattributes fundamental concepts (FL, LoRA, Deep Layer Dynamics) fails to meet the scholarly standard expected of a machine learning conference.
