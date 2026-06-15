# Peer Review

## Summary of the Paper
The paper addresses a major systems-level latency bottleneck in **multi-tenant Parameter-Efficient Fine-Tuning (PEFT)** serving: the "two-pass latency penalty" of penultimate-layer dynamic weight-merging routers (such as Parameter-Free Subspace Routing, PFSR). Because these existing dynamic routers project representations at the penultimate layer of the backbone, they require a full, throw-away first forward pass of the model to compute routing coefficients, followed by a second pass to produce the actual output.

To resolve this, the authors propose **ELATI** (**E**early-**L**ayer **A**daptive **T**ask **I**dentification), a training-free and parameter-free "one-pass" dynamic merging framework. ELATI shifts dynamic task identification to Layer 2 of a 14-layer network, avoiding the redundant deep backbone propagation in Pass 1. To route early without final semantic heads, ELATI introduces **Early-Layer Representative Mapping (ELRM)**, which projects intermediate activations against unsupervised task centroids computed from a hyper-sparse 64-sample calibration split (16 samples per task). For online serving, **Downstream-Only Micro-Batch Homogenization (DO-MBH)** dynamically runs early shared layers once, projects activations, partitions the batch on-the-fly, and propagates the micro-batches through dynamically ensembled downstream layers (Layers 3–14). The authors evaluate ELATI using a synthetic **Hierarchical 14-Layer Sandbox** and present secondary evaluations using a physical pre-trained **Vision Transformer (ViT-Tiny)** on real datasets, as well as several analytical sweeps (ablation on routing depth, sequence pooling under noise, and concept drift tracking).

---

## Strengths and Weaknesses

### Strengths:
1. **Compelling and Practical Motivation:** The paper identifies a genuine systems-level latency bottleneck in dynamic model-merging servers (the two-pass forward execution of the backbone). Solving this bottleneck is of paramount importance for low-latency multi-tenant edge and cloud deployments.
2. **Clear and Well-Formulated Methodology:** The proposed pipeline—encompassing early-layer centroid profiling (ELRM), similarity-based routing (OPSR), and downstream-only batch partitioning (DO-MBH)—is mathematically well-defined, transparent, and structured. Algorithm 1 is highly detailed and helpful.
3. **Thorough Analytical Sweeps:** The deep-dive section (Section 4.6) is exceptionally extensive, featuring insightful ablations of the routing layer index ($l_{\text{route}}$), sequence-token pooling choices, scaling benchmarks, ensembling pruning thresholds, and concept drift tracking.
4. **High Scientific Transparency:** The authors are highly honest about their experimental setup, explicitly disclosing the simulated nature of their sandbox task subspaces, CPU-bound timings, and the scaled model for the LLaMA-7B micro-benchmarks.

### Weaknesses:
1. **Catastrophically Low Classification Accuracy in Real Experiments:** To validate ELATI under physical representational flows, the authors deploy a pre-trained Vision Transformer (ViT-Tiny) on real-world datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN). However, the reported "Expert Ceiling (Oracle)" accuracy is incredibly weak—achieving a Joint Mean of only **26.00%** (MNIST: 39%, F-MNIST: 20%, CIFAR-10: 29%, SVHN: 16%). An accuracy of 16-39% on standard classification benchmarks like MNIST or CIFAR-10 indicates a severely flawed, under-trained, or unoptimized training setup (e.g., incorrect learning rates, bad head initialization, or lack of proper convergence). Because the physical model is operating at near-random classification performance (where random guess is 10%), the claim that "ELATI successfully guides downstream dynamic merging without disrupting representational flows" is highly suspect. It is unverified whether these ensembling properties hold in a high-performance, realistic regime (e.g., >80-90% accuracy) where feature representations are highly co-adapted, fragile, and sensitive to parameter-space interpolation.
2. **The "Simulation" Crutch:** The paper relies overwhelmingly on toy simulations. The primary evaluations (Table 1) use a synthetic "Hierarchical 14-Layer Sandbox" where task manifolds are modeled as disjoint orthogonal coordinates with added isotropic Gaussian noise. Real deep network activations do not exhibit orthogonal coordinates. Furthermore, the LLaMA-7B systems scaling benchmarks (Figure 8) and the hardware-level GPU profiling (Figure 12) are purely mathematical extrapolations and scaled simulations. For a paper focused on resolving latency in high-throughput cloud and production servers, the complete lack of physical GPU benchmarks and physical LLM serving evaluation is a major structural deficit.
3. **Suspiciously Poor Performance of Static Merging Baselines:** In Table 1, the authors report that advanced static model-merging methods like **DARE-Merging** (32.56% ± 2.66%) and **TIES-Merging** (37.39% ± 3.03%) perform drastically worse than standard **Uniform Merging** (48.27% ± 2.23%). In the established model-merging literature, DARE and TIES-Merging consistently and significantly outperform standard Uniform averaging by pruning redundant parameters and resolving sign interference. This pathological performance suggests a flawed or unoptimized implementation of these baselines in the authors' sandbox, which undermines the integrity of the comparison.
4. **Risk of Runaway Drift under Hybrid Adaptation:** The proposed "Hybrid Online Centroid Adaptation" updates centroids based on the model's own routing predictions. This introduces a severe risk of **confirmation bias** and runaway drift in the wild. If the early-layer router confidently misclassifies an out-of-distribution or noisy sample, it will integrate the corrupted activation into the centroid, eventually causing catastrophic routing collapse. While the authors propose stabilizers (Centroid Anchoring, Periodic Recalibration), these introduce multiple heuristic hyperparameters ($\nu, \lambda_{\text{anchor}}, \delta_{\text{margin}}$) that would be highly difficult to tune dynamically in a production setting.

---

## Soundness
### Rating: Fair
**Justification:** While the mathematical description is clear and logical, the soundness of the empirical validation is severely compromised. 
- Heavy reliance on a toy 14-layer sandbox with orthogonal task subspaces does not capture the highly non-linear, overlapping, and co-adapted representational manifolds of real deep networks.
- In the real-world Vision Transformer experiments, the "Expert Oracle" performs at a catastrophically low Joint Mean of **26.00%** (MNIST is only 39.00%), proving that the model is severely under-trained or bug-ridden. Drawing conclusions about downstream dynamic ensembling on-the-fly under a near-random classification regime is methodologically flawed.
- There are zero physical GPU wall-clock benchmarks; all speedup claims are either simulated or evaluated on CPU, which has completely different memory-bus, cache-locality, and execution thread characteristics compared to industrial GPU servers.

---

## Presentation
### Rating: Good
**Justification:** The paper is clearly written, well-structured, and easy to follow. The authors formulate their methodology rigorously and describe each systems step in detail. However, the overall narrative suffers from overstating simulated results as physical validations (e.g., LLaMA-7B scaling timing extrapolations are presented as micro-benchmarks before disclosing they are CPU-bound matrix multiplications). The presentation would be significantly improved by segregating simulated results from physical hardware results more clearly.

---

## Significance
### Rating: Fair
**Justification:** Shifting dynamic model ensembling to early layers and ensembling only downstream weights is a highly significant systems idea that could influence multi-tenant edge and cloud frameworks. However, because the evaluation is confined to toy sandboxes, CPU wall-clock timings, and an unoptimized ViT-Tiny model, the immediate significance of these findings to practitioners is limited. Without physical GPU validation on a real-world serving engine (like vLLM or S-LoRA) or physical LLM experiments, the practical utility of ELATI remains unverified.

---

## Originality
### Rating: Good
**Justification:** The concept of downstream-only model merging driven by early-layer, unsupervised centroid projection (ELRM) is a clever combination of early-probing networks and weight-space ensembling. While the components are derived from classical prototype-based learning and existing dynamic routing frameworks (PFSR), the integration and structural configuration represent a distinct and original contribution to the parameter-space serving literature.

---

## Overall Recommendation
### Rating: 3: Weak Reject
**Justification:** The paper addresses an important systems challenge and proposes an elegant, mathematically sound approach (ELATI) with significant systems potential. However, the execution and empirical validation are not yet ready for publication:
1. The physical Vision Transformer experiments operate at an incredibly poor classification performance (26% Oracle), indicating a buggy fine-tuning pipeline. The authors must optimize this pipeline to reach realistic accuracy scales to prove downstream ensembling does not cause catastrophic representation mismatch.
2. The lack of any actual physical GPU execution timings or real causal LLM routing benchmarks (despite heavy textual serving terminology) restricts the paper to a simulation study.
3. The pathological failure of the DARE and TIES-Merging baselines needs to be addressed or explained.

For these reasons, the weaknesses currently outweigh the merits. The submission requires a thorough revision of its training pipeline and real-world hardware profiling before it can be accepted.

---

## Questions and Constructive Feedback for the Authors

1. **Physical ViT Fine-Tuning Setup:** Why is the Expert Oracle's classification performance so extremely low (e.g., 39.00% on MNIST, 29.00% on CIFAR-10)? Please audit your image preprocessing, normalization, learning rates, and fine-tuning epochs. Re-running these experiments to achieve realistic performance ceilings (e.g., >80%) is critical to prove that early-layer routing successfully guides downstream parameter ensembling under highly developed and fragile representation flows.
2. **Static Merging Baselines:** Why do DARE and TIES-Merging consistently perform much worse than standard Uniform Merging in your Hierarchical Sandbox? Did you optimize their drop rates, scaling factors, or magnitude pruning thresholds? Standard literature indicates these methods should significantly outperform Uniform Merging.
3. **Physical GPU Validation:** Given that ELATI targets high-throughput production servers and large-scale models, please provide actual, physical wall-clock execution timings on an active GPU. Compiling a basic PyTorch pipeline and measuring actual CUDA Event timings (rather than CPU timing mathematical scaling) would significantly strengthen your systems claims.
4. **Causal LLM Routing:** Since the paper extensively discusses causal masks, autoregressive sequence pooling ($\Psi_{\text{attn}}$, $\Psi_{\text{final}}$), and LLaMA models, can you evaluate ELATI's routing accuracy on a real, small causal LLM (e.g., GPT-2 or LLaMA-3-8B-Instruct) across textual task domains (e.g., translation, sentiment, math) to validate these proposed sequence operators under real linguistic attention sinks?
5. **Hyperparameter Stability under Drift:** In the Hybrid Online Centroid Adaptation, how sensitive is the system to the anchoring coefficient ($\lambda_{\text{anchor}}$) and learning rate ($\nu$) under varying levels of stream noise? Is there a risk of the anchor pulling the centroids too strongly, preventing adaptation to legitimate non-stationary drift?
