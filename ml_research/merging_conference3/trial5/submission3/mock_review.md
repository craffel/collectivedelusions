# Peer Review

**Paper Title:** Robust Linear Routing (RLR) for Dynamic Model Merging  
**Recommendation:** 6 (Strong Accept)  
**Soundness:** Excellent  
**Presentation:** Excellent  
**Significance:** Excellent  
**Originality:** Excellent  

---

## 1. Summary of the Paper
This paper presents a critical deconstruction of recent trends in *dynamic model merging* (parameter fusion) through the lens of **Occam's razor**. Specifically, it investigates *Quantum Wavefunction Superposition Merging (QWS-Merge)*, which claimed that classical linear routing is structurally limited and prone to catastrophic representation collapse on challenging, high-variance datasets like SVHN (collapsing to a reported $15.30\%$). 

The authors demonstrate that this reported collapse is not a structural limitation of linear gating, but rather a standard, preventable overfitting and high-variance routing logit issue caused by unregularized calibration on tiny datasets. They propose **Robust Linear Routing (RLR)**, which retains a simple 768-parameter classical gating layer but stabilizes its optimization using two standard regularizations: $L_2$ weight decay and softmax temperature scaling. 

Evaluated on a 4-task Vision Transformer benchmark, the classical unregularized Linear Router (properly configured) achieves an outstanding **$91.53\% \pm 0.41\%$** Joint Mean accuracy across 5 random calibration seeds, completely avoiding SVHN collapse. RLR delivers a statistically identical homogeneous performance ($91.46\% \pm 0.42\%$ Joint Mean) but acts as a specialized stabilizer, securing superior resilience and maintaining a significant accuracy buffer over the unregularized classical baseline under mixed-task heterogeneous evaluation streams (e.g., $+1.88\%$ absolute accuracy benefit at batch size $B=256$).

---

## 2. Key Strengths
- **Rigorous Scientific Deconstruction:** Instead of accepting published cross-paper numbers, the authors locally re-implemented the QWS-Merge baseline under identical conditions on the exact same expert weights. This local evaluation showed that both classical unregularized routing ($95.46\%$ Joint Mean on seed 42) and RLR ($94.68\%$ Joint Mean) significantly outperform QWS-Merge ($90.03\%$ Joint Mean), definitively debunking its core thesis and showing that its wavefunction projection does not offer any empirical benefit.
- **Deep Technical Diagnosis (Table 2):** To guide future researchers, the paper provides a systematic diagnostic comparison of the sub-optimal configuration choices (such as extracting representations from deep task-warped layers, using excessive learning rates, and over-optimizing for thousands of steps) that likely triggered the baseline collapse in prior work, establishing an exceptionally stable setup.
- **Excellent Methodology and Evaluation Rigor:** The proposed RLR is supported by multi-seed sweeps (5 seeds), 2D hyperparameter sensitivity sweeps ($\alpha \times T$), and representation source layer ablations, demonstrating high stability and proving that its performance is not reliant on sensitive hyperparameter tuning.
- **Intellectual Honesty and Nuance:** The authors openly acknowledge that under homogeneous settings, regularized and unregularized classical routers are statistically indistinguishable, framing RLR clearly as a specialized stabilizer for OOD shifts and heterogeneous environments. They also detail the trade-offs between static supervised merging (OFS-Tune) and dynamic routing (RLR) under mixed heterogeneous streams, providing invaluable engineering guidelines for practitioners.
- **Actionable Scaling Pathways to LLMs:** The paper addresses generalizability early by formulating three concrete, highly promising pathways for scaling RLR's regularized gating to multi-billion parameter Large Language Models (sequence-level pooled representations, routing over lightweight LoRA experts, and exploiting linear mode connectivity).
- **Outstanding Presentation:** The paper is exceptionally well-written, clearly formatted, and flows logically. Equations are elegant and beautiful, and the figures are highly polished and directly support the text.

---

## 3. Key Weaknesses
While the paper is extremely solid and represents an exemplary submission, there are a few minor limitations that the authors should consider:
- **Evaluated on Smaller-Scale Models and Tasks:** The empirical validation is restricted to compact Vision Transformers ($\mathtt{vit\_tiny\_patch16\_224}$, 5.7M parameters) and standard vision classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). While this choice is fully justified as a direct replication and deconstruction of QWS-Merge (which used the same setup), verifying the findings on slightly larger models (e.g., ViT-Base, ResNet-50) or more complex datasets (such as ImageNet tasks) would further strengthen the paper's claims of generalizability.
- **Degradation under Extreme Heterogeneity:** Despite RLR's stabilizing effects, both dynamic routers still suffer from severe accuracy drops under mixed heterogeneous streams as the batch size increases ($B=256$ accuracy drops to $\approx 75\%$, compared to $\approx 92\%$ at $B=1$). While the authors address this trade-off honestly and show that static methods like OFS-Tune ($86.23\%$) are superior in this regime, it highlights a fundamental, unresolved limitation of weight-space dynamic routing.

---

## 4. Detailed Feedback and Suggestions for Improvement
The paper is highly complete, technically sound, and ready for publication. I have only a few minor suggestions to further polish the manuscript:
1. **Elaborate on LLM Scaling Paths in Future Work:** In Section 5, you discuss scaling pathways to LLMs (such as using sequence-level pooled representations and routing over LoRA experts). If space permits, it would be highly valuable to include a brief equation or diagram illustrating how the sequence-level pooled hidden state of a generative LLM (e.g., at the first layer or [BOS] token) would be mapped to LoRA blending coefficients, as this would provide a concrete blueprint for generative AI researchers.
2. **Dynamic Sorting of Heterogeneous Streams:** In Section 4.4, you mention that dynamic models are optimal for dedicated task routing servers where inference queries can be batched homogeneously. It would be interesting to discuss if a lightweight pre-sorting or routing layer could be placed in front of the dynamic model to group incoming heterogeneous queries into homogeneous batches, thereby circumventing the heterogeneity collapse entirely.
3. **Typographical Note:** In Section 2.3, "The authors applied regularization to the underlying model landscapes... surprisingly, while gating network regularizations... are foundational pillars... regularization of the *routing network itself* has been almost entirely overlooked..." Ensure the asterisk in "*routing network itself*" is formatted correctly (it currently has an unescaped asterisk before and after).

---

## 5. Overall Verdict
This is an outstanding, scientifically rigorous, and highly refreshing paper. It champions **Occam's razor** in deep learning by demonstrating that a simple, 768-parameter classically regularized linear router can match or outperform convoluted, over-engineered quantum-inspired frameworks. The paper's empirical evaluation, local baseline re-implementations, multi-seed statistical sweeps, hyperparameter sensitivity heatmaps, and intellectual honesty set a gold standard for empirical machine learning research. I strongly recommend **Strong Accept (Score: 6)**.
