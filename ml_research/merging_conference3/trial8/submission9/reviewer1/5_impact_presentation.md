# Evaluation Phase 5: Impact and Presentation

## 1. Major Strengths
- **Academic Honesty and Transparency:** 
  The paper stands out for its exemplary honesty. Rather than hiding or downplaying the failures of its proposed zero-shot methods (EER and EPL-OCA) on real-world ResNet-18 embeddings, the authors analyze and report these collapses in detail. They explain *why* the methods fail (Entropy Calibration Discrepancy and the self-referential pseudo-label corruption loop), which provides immense scientific value.
- **Comprehensive and Diverse Baselines:** 
  The study compares its methods against an impressive variety of baselines, including static weight merging, non-parametric projection routing, supervised SOTA (SPS-ZCA), unsupervised streaming K-Means, and gradient-based test-time adaptation (TENT).
- **Practical Systems-Level Analysis:** 
  The paper doesn't just focus on accuracy; it provides a thorough analysis of edge hardware constraints. It mathematically formulates the FLOP serving complexity under post-activation divergence and proposes a practical optimization (Amortized Pseudo-Labeling) which is backed by physical CPU profiling and theoretical energy/bandwidth analysis.
- **Exceptional Presentation and Structure:** 
  The paper is excellently structured, with a clear narrative flow. The methodology and experimental results are presented with rigorous detail, including numerous valuable ablation studies (softmax temperature, vocabulary-size normalization, warm-up window sizes, and temporal task locality).

## 2. Areas for Improvement
- **Address the Lack of Formal Mathematical Rigor:**
  The authors frequently use grandiose language to describe their analysis (e.g., "technically rigorous," "formally demonstrate," "mathematically motivating"). However, there are no actual theorems, lemmas, or formal mathematical proofs in the paper. The authors should tone down this phrasing and honestly acknowledge that their proposed methods are empirical heuristics.
- **Formally Analyze the Online Update Dynamics:**
  Instead of merely describing the *self-referential pseudo-label corruption loop* and the *Representational Sparsity Paradox* qualitatively, a theoretically rigorous study would model these phenomena mathematically. For example, the online centroid update could be formulated as a dynamical system, and stability or convergence analyses could be provided under various levels of domain shift and calibration discrepancy.
- **Mitigate the Synthetic-to-Real Disconnect:**
  The synthetic sandbox's assumptions (perfect subspace and class orthogonality, isotropic Gaussian noise) are highly idealized and do not translate to real-world representation spaces. Since both zero-shot methods (EER and EPL-OCA) collapse on real features, the synthetic sandbox results are of limited practical value. The paper should focus more on why these assumptions fail in real spaces and how they can be modified to better reflect real-world representation manifolds.
- **Scale of Evaluation:**
  To broaden the significance of the paper, the authors should evaluate their methods on larger models (e.g., Vision Transformers like ViT-B/16 or Language Models like BERT/Llama) and a wider variety of datasets (such as NLP or multi-modal benchmarks), as model merging is most commonly applied in these contexts.

## 3. Overall Presentation Quality
The presentation quality is **excellent**. The writing is clear, precise, and highly professional. The mathematical equations are well-formatted, and the tables and descriptions are incredibly detailed.

## 4. Potential Impact and Significance
The potential impact of the paper is **moderate**. 
- On-device serving of multiple LoRA experts is a highly relevant, active research area. Eliminating the need for offline labeled calibration splits is a major step toward practical on-device autonomy.
- The systems-level profiling and the Amortized Pseudo-Labeling strategy are highly valuable for practitioners deploying ensembling models on edge devices.
- However, the fact that the proposed "completely calibration-free zero-shot" paradigms (EER and EPL-OCA) **completely collapse on real features** means they are not immediately viable for real-world applications. The only successful real-world method proposed, CG-EER, is semi-supervised and relies on pre-computed offline centroids, which violates the "calibration-free" promise.
- Nonetheless, by clearly diagnosing the *Entropy Calibration Discrepancy* and the *self-referential corruption loop*, the paper provides an excellent, scientifically rigorous foundation for future researchers to build upon to solve these fundamental limits of unsupervised test-time adaptation.
