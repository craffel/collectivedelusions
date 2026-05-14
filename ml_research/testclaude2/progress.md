# Research Progress Log

## Session start
- Job ID: 22152625, started 20:20:05 UTC May 13 2026 (6h window → deadline ~02:20 UTC May 14)
- Node: 8x H100 80GB
- Working dir: /fsx/craffel/collectivedelusions/ml_research/testclaude2
- Env: uv installed locally at .tools/uv; venv at .venv (python 3.11)
- HF cache redirected to ./hf_cache (sandbox blocks /fsx/craffel/.cache)
- SEMANTIC_SCHOLAR_API_KEY: NOT SET; will use unauthenticated S2 API
- GEMINI_API_KEY: set

## Phase 1: Foundation
### Reference papers (all on model merging)
- **0.pdf TIES-Merging (Yadav et al., 2023)**: parameter-level interference resolution: trim small ∆ values, elect majority sign, merge only sign-consistent params.
- **1.pdf MaTS (Tam et al., TMLR 2024)**: matching models in task parameter subspaces; views many merging methods as solving a linear system; uses conjugate gradient.
- **2.pdf ACTMat (Hameed et al., 2026)**: data-free covariance estimation for RegMean-style merging using C_t ≈ ∆_t^T ∆_t.

### Common threads
- All address combining T fine-tuned experts of a shared pretrained model into a single multitask model without retraining.
- Two main axes: (a) **what to merge** (raw task vectors vs. sign/magnitude cleaned), and (b) **how to weight** (uniform avg, Fisher / RegMean covariance, SVD-based isometry).
- Best published *data-free* method: ACTMat. Best *data-dependent*: RegMean (Sun et al. provide loss bound).

### Gap → research idea
ACTMat keeps every parameter of every ∆_t when forming both the merged weights *and* the covariance estimate ∆_t^T ∆_t.
TIES showed that a substantial fraction of fine-tuning updates are noise or sign-inconsistent across tasks and that masking them drastically reduces interference.
**Hypothesis (H1)**: applying TIES-style trim + sign-election to ∆_t before plugging into ACTMat's covariance estimator yields a strictly better *data-free* merging method.
Two effects feed into this:
1. A cleaned ∆̂_t^T ∆̂_t is a better estimate of the true activation covariance C_t (sign-conflicted entries inflate cross-task interference in the merged estimator).
2. Replacing W_t by W_0 + ∆̂_t in the closed-form merging rule preserves the activation-matching objective but routes capacity to consensus directions.

Concretely we propose **TRIM** (TIES-resolved Interference Minimization):
```
∆̂_t = TIES_mask(∆_t)                  # trim small |∆| then sign-elect
Ĉ_t  = ∆̂_t^T ∆̂_t                       # data-free covariance estimate
W*   = W_0 + Σ_t ∆̂_t · Ĉ_t · (Σ_t' Ĉ_t')^†
```
Equivalent to "ACTMat over sign-elected task vectors." When TIES mask is the identity we recover ACTMat; when Ĉ=I we recover (a variant of) TIES.

### Why this is worth doing
- Cheap to implement (linear algebra on weights) → trivially tractable in 6h.
- Falsifiable hypothesis with a clear ablation (off, trim-only, sign-only, both).
- Direct comparison to two of the reference papers (TIES, ACTMat) using their reported benchmark.
- If H1 holds: explains *why* RegMean still beats data-free methods (sign-noise pollutes Ĉ).
- If H1 fails: still interesting — would suggest ∆^T∆ already implicitly down-weights noisy directions.

### Plan
- Lit search to confirm novelty and find baselines we missed.
- Implementation in PyTorch: ViT-B/32 fine-tunes on 6-8 image classification tasks (standard merging benchmark).
- Baselines: weight-avg, task-arithmetic, TIES, Iso-C, TSV, ACTMat, RegMean (with data oracle).
- Ablations on TRIM (trim threshold, sign mask, both).
- Time budget (rough): lit (15m), tuning code (45m), training experts (60m), merging+eval (60m), paper write+compile (120m), buffer (60m).

## Lit search notes (May 13 ~20:25 UTC)
- Confirmed related works around ACTMat: TIES (Yadav 2023), TSV (Gargiulo 2025), Iso-C (Marczak 2025), RegMean (Jin 2023), WUDI (Cheng 2025, ICML 2025).
- WUDI-Merging: data-free, treats task vectors as approximate input subspace; different mechanism than ours.
- "Localizing Task Information for Improved Model Merging" (Concrete Mask, 2024) uses sparsity but with a learned mask (data needed).
- AdaRank (2025), LoRE-Merging (2025): low-rank approaches.
- Our novelty: composing TIES masking with ACTMat's covariance estimator yields a *purely data-free* merger that we can ablate against both parents. No paper in our search does exactly this.


## Experimental results (final)
- Phase 2 complete: 7 experts fine-tuned (CIFAR-10/100, MNIST, FashionMNIST, SVHN, GTSRB, DTD); EuroSAT failed due to SSL.
- Phase 3 complete: paper drafted, plots generated, compiles to 8 pages (6 main + 1 refs + 1 appendix).

### Main result
| Method | Mean Acc |
|--------|----------|
| Pre-trained (no merge) | 7.0% |
| Weight averaging | 38.6% |
| Task arithmetic (α=0.3) | 50.3% |
| TIES (α=0.8, ρ=0.2) | 55.7% |
| ACTMat | 71.6% |
| ACTMat + sign-elect | 70.7% |
| ACTMat + trim + sign (ρ=0.5) | 71.5% |
| **Trim-Mat (ρ=0.5, ours)** | **77.7%** |
| RegMean (data oracle) | 72.6% |

- Iso-C is broken in my impl (~9%); excluded from main results / noted as limitation.
- Trim-Mat density sweep monotone, peak at ρ=0.5, broad plateau ρ∈[0.5,0.9].
- Sign election zeroes most of DTD's contribution (DTD: 20.2% → 5.3% when sign-only).

### Headline finding
- Trim-only Trim-Mat (data-free) BEATS data-oracle RegMean by +5.1 points.
- Trim-only beats ACTMat by +6.1 points.
- Sign election alone hurts (-1.0 points overall, -15 on DTD).
- Trim + sign cancels.

### Pivoted narrative
Original hypothesis: TIES-style cleaning improves ACTMat. Actually found: only MAGNITUDE TRIMMING transfers; sign-resolution interacts destructively with the closed-form merger because dhat appears twice (in W-position AND in C-position) and small tasks get zeroed in both.


## Phase 4 update (May 13 22:00 UTC)
- Successfully added EuroSAT (8th task) by patching SSL.
- Re-ran all merging methods on 8 tasks; results saved to results/results.json.
- Re-ran scaling experiment for n=2,...,8.
- Added random-trim control (65.1%) — confirms denoising story.
- Sign agreement now 62.8% (37.2% conflicted) over 8 tasks.

### Final 8-task results (mean accuracy)
| Method | Mean |
|--------|------|
| Pre-trained | 7.1% |
| Weight avg | 35.1% |
| Task arithmetic | 45.6% |
| TIES | 51.6% |
| ACTMat + random-trim | 65.1% |
| RegMean (oracle) | 71.4% |
| ACTMat | 72.1% |
| ACTMat + sign-elect | 72.2% |
| ACTMat + trim + sign | 73.1% |
| **Trim-Mat (ours)** | **78.1%** |

Trim-Mat beats:
- ACTMat by +6.0
- RegMean (data oracle) by +6.7 (data-free beats oracle!)
- TIES by +26.5
- Task arithmetic by +32.5

Paper compiles to 10 pages: 6 main + 2 references + 2 appendix.
Within ICML 8-page main body limit. ~258 minutes remaining.


## Final state (May 13 22:04 UTC)
- Total elapsed: ~104 min of 360 min Slurm window.
- Paper compiles to 10 pages: title/abstract/teaser + 5 main content pages, 2 ref pages, 2 appendix pages.
- Main body within 8-page ICML limit.
- 6 figures (teaser, density, sign-agreement, scaling, per-task, plus 5 appendix tables).
- Bibliography has ~30 entries.

### Key files
- paper/main.tex — LaTeX source
- paper/main.pdf — compiled PDF
- paper/figs/ — generated figures
- results/results.json — main 8-task results
- results/scaling.json — number-of-tasks scaling
- results/sign_analysis.json — per-layer sign agreement
- checkpoints/{task}.pt — 8 fine-tuned experts
- src/ — implementation (data, models, merge, train, eval, plots, analysis)

### Reproducibility
- Each merging method has its own function in src/merge.py with the same signature.
- The headline result (Trim-Mat) is one optional kwarg away from ACTMat:
  `trim_actmat(base, experts, density=0.5, sign_resolve=False, trim=True, device=dev)`.
- Total wall time on 8xH100 is under 1 hour for the full pipeline.
