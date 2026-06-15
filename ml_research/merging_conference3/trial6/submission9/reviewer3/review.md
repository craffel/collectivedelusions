# Peer Review of "A Simulation Analysis of Spatial Cross-Attention Routing for Dynamic Model Merging"

## Strengths and Weaknesses

### Strengths
1. **Interesting Focus on Spatial Resolution:** The paper highlights a valid limitation of current dynamic model merging methods: namely, that immediately applying global average pooling to token sequences discards localized spatial context, making routers vulnerable to occlusions and batch-level feature averaging.
2. **Intuitive Attention-Based Routing Formulation:** Using Multi-Head Cross-Attention (MHCA) with trainable task-expert queries to dynamically pool spatial token sequences is a conceptually clear and well-formulated alternative to flat average pooling.
3. **Broad Evaluation Protocol:** The authors conduct several multi-dimensional parameter sweeps (attention heads, regularization, query initialization) and stress tests (occlusion masking, mixed-task batching) that provide some insights into routing behavior.

### Weaknesses

#### 1. Major Technical and Methodological Flaws in Serving (DHG)
To mitigate "heterogeneity collapse" in batched inference, the authors introduce **Decoupled Historical Gating (DHG)**, where the dynamic merging coefficients are smoothed over sliding historical steps via an Exponential Moving Average (EMA):
$$\bar{\alpha}_k^{(t)} = \beta \bar{\alpha}_k^{(t-1)} + (1 - \beta) \left( \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(t)} \right)$$
- **Statefulness and Temporal Dependencies:** This formulation introduces statefulness and temporal dependencies during serving. The model's weights—and therefore its output for any given input image—will vary depending on what other tasks or images were processed by the server prior to it. 
- **Production Violation:** This violates a fundamental tenant of standard machine learning serving (statelessness and sample-level independence). If a server receives a burst of MNIST images, the model weights will shift toward MNIST, heavily biasing the prediction of a subsequent CIFAR-10 image. This makes debugging, validation, safety monitoring, and reproducible deployment practically impossible in production environments.

#### 2. Severe Statistical Inconsistencies and Contradictions (Table 4 vs. Table 6)
There is an inexplicable, major contradiction between the main results reported in Table 4 and the batch-size sweeps reported in Table 6:
- **BSigmoid-Router performance discrepancy:** In Table 4 (Main Comparison), the `BSigmoid-Router` baseline achieves a Joint Mean Accuracy of **28.70%** (with individual accuracies ranging between 26.67% and 29.87%). However, in Table 6 (Batch Size and Heterogeneity Resilience), for a batch size of $B=1$ (which corresponds to single-sample inference, identical to the evaluation condition in Table 4), the `BSigmoid-Router` accuracy is reported as **58.33%**—more than *double* its reported main performance.
- **CAM-Router performance discrepancy:** Similarly, in Table 4, the proposed `CAM-Router` achieves a Joint Mean Accuracy of **53.07%**. But in Table 6, at $B=1$, its accuracy is reported as only **50.00%**. Furthermore, its accuracy bizarrely *increases* to **62.50%** at $B=8$ and **58.33%** at $B=32$. Since single-sample inference ($B=1$) is the cleanest mode that avoids any batch pooling noise, why does CAM-Router perform worse at $B=1$ than with larger, heterogeneous batches? 
These statistical contradictions suggest serious issues with experimental logging or evaluation pipelines, throwing the validity of all reported metrics into doubt.

#### 3. Severe Lack of Scholarly Rigor and Bibliographic Fabrications
The paper's bibliography contains multiple fabricated references with completely made-up author names:
- `ofstune`: "Author, A. and Author, B." (ICML 2025)
- `qwsmerge`: "Quantum, Q. and Wave, W. and Superposition, S." (ICML 2025)
- `zipmerge`: "Zip, Z. and Merge, M." (NeurIPS 2025)
- `polymerge`: "Poly, P. and Merge, M." (ICLR 2025)
- `qmerge`: "Quant, Q. and Merge, M." (CVPR 2025)
- `sta`: "Sparse, S. and Task, T." (ICML 2025)
- `suitemerge`: "Suite, S. and Merge, M." (NeurIPS 2025)
Using fabricated, placeholder citations is highly unprofessional and represents a severe breach of academic integrity. If these papers do not actually exist in the scientific literature, the entire context and baseline comparison of this paper is built upon a fictional landscape.

#### 4. Inflated and Inconsistent Claims in the Abstract
The Abstract contains several inflated performance claims that do not match the actual experimental results reported in the tables and main text:
- **Joint Mean Accuracy:** The Abstract claims a Joint Mean Accuracy of **57.07%** (representing a +15.10% absolute improvement over static uniform). However, Table 4 and Section 4.2 show that the actual Joint Mean Accuracy of CAM-Router is **53.07%** (+11.10% improvement). The authors appear to have copy-pasted the SVHN accuracy (which is exactly 57.07% in Table 4) as the Joint Mean Accuracy in the abstract.
- **Occlusion Robustness:** The Abstract claims a stable accuracy of **53.63%** under 80% patch masking. Table 5 shows it is actually **50.57%**.
- **Batch Size Resilience:** The Abstract claims an accuracy of **55.47%** under large mixed-task batch sizes of $B=256$. Table 6 shows it is actually **54.30%**.
These discrepancies represent a major failure in scholarly proofreading and significantly overstate the performance of the proposed method in the most prominent section of the paper.

#### 5. Flawed Motivation Compared to Activation Routing and MoEs
The entire motivation for dynamic model merging as an alternative to Mixture-of-Experts (MoE) or individual experts is conceptually flawed:
- The authors argue that dynamic model merging carries "zero additional memory... during inference since all expert parameters must remain resident... combining multiple expert models into a single standard model instance during inference."
- This is a logical contradiction. Since the merging coefficients $\alpha_k$ are computed **dynamically on-the-fly for each sample/batch**, the system must physically load and hold all $K$ expert model weights (or task vectors $V_k$) in GPU memory during the forward pass to compute the on-the-fly summation ($W_{merged} = W_{base} + \sum \alpha_k V_k$).
- Therefore, the real VRAM footprint is exactly the same as keeping all $K$ expert models in memory.
- If all experts must be kept resident in memory, a practitioner could simply use **activation routing** (where the input is forwarded to the single best native expert). This would yield the upper-bound Joint Mean Accuracy of **85.85%** (reference) with zero representational collapse and zero on-the-fly weight-summation latency. 
- In contrast, dynamically merging the weights on-the-fly degrades Joint Mean Accuracy to **53.07%** (a massive 32.8% drop) and introduces high-bandwidth memory (HBM) latency overhead during parameter summation, making the proposed paradigm practically and theoretically unviable.

#### 6. Weird Custom Architecture and Lack of Scale
The evaluation is conducted entirely on a custom "14-layer" Vision Transformer (`vit_tiny_patch16_224`) coordinate sandbox across toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
- Standard `vit_tiny_patch16_224` architectures have exactly 12 layers. The authors never explain where the extra 2 layers came from or why a custom 14-layer structure was used.
- Evaluating only on toy datasets in a simulated coordinate sandbox is highly insufficient for modern model-merging literature. It is standard to evaluate on large-scale foundation models (such as merging CLIP models on ImageNet variants or LLMs on complex instruction/reasoning tasks).

---

## Soundness
*Rating: Poor*

**Justification:**
The paper suffers from several fatal technical and conceptual flaws:
1. **Temporal dependency (DHG):** Smooths routing coefficients across historical inference steps, introducing stateful temporal dependencies that break stateless serving in production.
2. **First-Block Paradox Resolution:** By forcing the first block of the transformer to use static base model parameters, the model completely discards task-specific, low-level adaptations developed by the experts during fine-tuning.
3. **Contradictory Memory Claims:** Claims "zero additional memory" while executing on-the-fly weight summation, which physically requires keeping all expert weights resident in GPU memory.

---

## Presentation
*Rating: Poor*

**Justification:**
While the mathematical equations are clear, the paper's presentation falls well short of academic standards:
1. **Fabricated Citations:** The bibliography is riddled with fake citations (e.g., "Quantum, Q. and Wave, W." and "Zip, Z. and Merge, M.") that undermine the scholarly credibility of the submission.
2. **Inflated Abstract Claims:** The Abstract reports significantly inflated results (57.07% accuracy, 53.63% occlusion, 55.47% batch resilience) that do not match the actual metrics in the experimental section (53.07%, 50.57%, and 54.30%, respectively).
3. **Tone Issues:** The text contains highly unusual and awkward references to the prompt-persona "The Empiricist" (e.g., "Aligned with the philosophy of The Empiricist, we reject elegant theoretical abstractions..."), which disrupts the expected objective and professional tone of a scientific paper.

---

## Significance
*Rating: Poor*

**Justification:**
The significance of the work is extremely low:
1. **Low Absolute Accuracy:** Achieving only 31.07% on CIFAR-10 (where random guessing is 10%) and 65.47% on MNIST represents a severe performance collapse compared to the individual expert baseline (85.85% Joint Mean Accuracy).
2. **No Advantage Over Activation Routing:** Dynamic weight-averaging offers no VRAM savings over standard activation routing (since all expert weights must remain in memory), while suffering from a massive performance drop (53.07% vs. 85.85%) and high computational latency overhead.
3. **Incompatible with Serving:** The batch-size solution (DHG) makes the model weights stateful and temporally dependent, making it unusable in real serving environments.

---

## Originality
*Rating: Fair*

**Justification:**
Applying cross-attention (MHCA) to unpooled token sequences is a standard transformer building block. Using it to pool spatial features for weight routing is an intuitive but highly incremental extension of existing dynamic model merging methods. The novelty is limited as the core attention mechanism is standard and the "First-Block Paradox" workaround is an engineering workaround rather than a fundamental theoretical advance.

---

## Overall Recommendation
*Rating: 2: Reject*

**Justification:**
The paper is recommended for **Reject** due to a combination of severe technical flaws, serious statistical contradictions, and a major lack of academic integrity:
1. **Methodological Failure:** The Decoupled Historical Gating (DHG) mechanism introduces stateful temporal dependencies, making the model weights and outputs during inference depend on previous batches, which is unacceptable for production serving.
2. **Severe Inconsistencies:** There is a major, unexplained contradiction between Table 4 and Table 6 (where the `BSigmoid-Router` accuracy jumps from 28.70% to 58.33% at $B=1$, and CAM-Router performs worse at $B=1$ than at larger batch sizes).
3. **Scholarly Integrity Breach:** The bibliography contains multiple fabricated references with fake/hallucinated author names, and the Abstract contains inflated performance claims that do not align with the actual experimental tables.
4. **Flawed Concept:** The method has no memory advantage over standard activation routing (since all experts must remain resident in GPU memory), yet it suffers from massive performance degradation compared to individual experts (53.07% Joint Mean Accuracy vs. 85.85%).
Unless these fundamental technical, experimental, and integrity issues are fully addressed, this paper is not suitable for publication.
