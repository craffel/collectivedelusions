# Intermediate Evaluation: 5. Impact and Presentation

## Major Strengths
1. **Clarity and Mathematical Notation:** The paper is exceptionally well-written and structured. The mathematical notation is rigorous, consistent, and easy to follow. The decoupling of the standardization filter from unstandardized parameter integration is clearly explained.
2. **Rigorous Dissection of Baselines:** The authors provide a highly detailed discussion of why high-dimensional test-time adaptation baselines (AdaMerging and ZipMerge) fail under zero-order search, separating optimization failure from transductive overfitting through a systematic 500-step optimization study.
3. **Honest Discussion of Limitations:** The authors are transparent about the physical trade-offs of weight-space merging compared to joint fine-tuning (MTL) and LoRA adapters. They explicitly acknowledge the performance drops, the zero-sum trade-offs of the minimax objective, and the limitations of testing on a compact ViT-Tiny backbone.
4. **Detailed Implementation Blueprint:** The inclusion of an explicit pseudo-code algorithm (Algorithm 1) showing how to adapt EPM directly to decoder-only LLM architectures (self-attention projections and MLP layers) provides high practical utility for scaling the work.

---

## Areas for Improvement
1. **Provide Formal Mathematical Guarantees:** The paper lacks any theoretical proofs, theorems, or bounds. The authors should introduce formal mathematical analysis to justify why Soft-EPA preserves representational coherence and why TLC-Tune is guaranteed to generalize. For instance, they could derive generalization bounds (Rademacher complexity) for the $K$-dimensional global scaling space compared to high-dimensional layer-wise spaces to theoretically prove the "Overfitting-Optimizer Paradox."
2. **Derive a Principled Dynamic Coherence Schedule (DCS):** The current quadratic schedule is based on a loose, heuristic assumption of independent random pruning masks. The authors should mathematically model the correlation between fine-tuned task vectors sharing an initialization and derive a more rigorous scheduling rule that accounts for coordinate-level overlap.
3. **Include the Missing Activation Manifold Data:** The detailed empirical assertions regarding Centered Kernel Alignment (CKA) decay and t-SNE clustering in Section 4.3 must be backed up with actual quantitative tables or plots. If these experiments were not conducted or cannot be visualized, these claims should be removed to maintain scientific integrity.
4. **Resolve the Baseline Optimizer Mismatch:** To ensure a truly fair comparison, the authors should evaluate AdaMerging and ZipMerge under their native first-order gradient descent pipelines on differentiable validation cross-entropy losses, rather than forcing them into a zero-order accuracy-aligned search where they are mathematically expected to fail.
5. **Establish Scale Generality:** The authors should evaluate EPM empirically on modern Large Language Models (e.g., merging specialized Llama-3 or Mistral models on code and mathematics) to verify that the coordinate-level routing mechanics scale and that the performance gap with expert ceilings shrinks in highly overparameterized regimes.

---

## Overall Presentation Quality
The presentation quality is **good to excellent**. The narrative flows logically, the figures are clean and informative (specifically the conceptual overview and the optimization study trajectory), and the tables are comprehensive. The mathematical framing of Soft-EPA as a convex combination (Equation 9) is elegant. However, the presentation is "theory-heavy in style, but heuristic in substance"—it uses advanced-sounding mathematical terminology ("activation manifolds," "gradient physics," "convex combination") to describe what are ultimately simple empirical heuristics, without providing any actual proofs or theoretical depth.

---

## Potential Impact and Significance
The potential impact of this paper is **fair to moderate**:
- **For Practitioners:** The simplicity of EPM (fewer than 10 lines of PyTorch, zero-overhead deployment) and the detailed LLM algorithm make it highly attractive for resource-constrained multi-task serving.
- **For Researchers:** The critique of highly parameterized test-time optimization and the demonstration of the Overfitting-Optimizer Paradox are valuable and could shift focus toward simpler global scaling frameworks like TLC-Tune.
- **Theoretical Significance:** Low. Because the routing protocol and standardization are built entirely on empirical heuristics without formal mathematical foundations or guarantees, the paper does not advance the theoretical understanding of model weight spaces or optimization theory in a significant way.
