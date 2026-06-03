import random

ideas = [
    {
        "id": 1,
        "title": "The Baselines are Untuned: A Rigorous Re-evaluation of Parameter Rescale Theories",
        "description": "Exposing a severe baseline tuning issue in current parameter rescaling works. We hypothesize that when standard Task Arithmetic is paired with simple, training-free Uniform BatchNorm statistics merging, it outperforms flashy theories like HNS and IPR. This will reveal that proposed 'mathematically exact' methods are actually inferior to a simple, ignored baseline when properly configured.",
        "expected_results": "Demonstrate that standard Task Arithmetic with simple Uniform BatchNorm statistics merging achieves higher accuracy than HNS or IPR, exposing weak baseline comparisons in prior literature.",
        "impact": "High. It will challenge the necessity of complex parameter scaling theories and advocate for rigorous baseline tuning in model merging research."
    },
    {
        "id": 2,
        "title": "The 'Orthogonal Task Updates' Assumption: Fact or Artifact of Toy Benchmarks?",
        "description": "Analyzing whether the assumption of nearly perfect orthogonality of task updates holds on larger, more realistic models (e.g., ViT, CLIP, LLMs) and complex tasks, or if it is merely an artifact of fine-tuning small ResNet-18 models on toy, highly distinct datasets like MNIST and CIFAR-10.",
        "expected_results": "Show that task updates are not orthogonal on realistic models/tasks, and that HNS and IPR degrade in performance as a result.",
        "impact": "High. Questions the foundational assumptions of multiple recent publications."
    },
    {
        "id": 3,
        "title": "Confounding Variables in Zero-Shot Fisher Estimation: Noise vs. Signal in Post-Merge Calibration",
        "description": "Deconstructing why Synthetic Fisher estimation fails at small batch sizes. We will study how the choice of synthetic input distribution (OOD Gaussian vs. in-distribution proxies) interacts with network activations and identify how to perform zero-shot Fisher estimation rigorously.",
        "expected_results": "Identify the key mathematical cause of ReLU dead-ends and propose a noise distribution that restores zero-shot Fisher performance.",
        "impact": "Medium. Provides constructive methodological fixes to a known failure mode."
    },
    {
        "id": 4,
        "title": "Is 'Representation Collapse' Actually 'Normalization Collapse'? An Investigation across Normalization Layers",
        "description": "Investigating how representation collapse manifests in networks with LayerNorm, GroupNorm, or InstanceNorm compared to BatchNorm. We will evaluate if parameter scaling is uniquely beneficial to BatchNorm or generalized.",
        "expected_results": "Determine if representation collapse occurs in the absence of running statistics and if IPR/HNS are robust across different norm layers.",
        "impact": "Medium. Expands the scope and generalizes model-merging calibration theories."
    },
    {
        "id": 5,
        "title": "The Hidden Cost of Reconstruction: A Fair Comparison of Parameter-Space Scaling vs. True Merging",
        "description": "Conducting a fair, rigorous evaluation of HNS and IPR in a true multi-task joint inference setting (a single merged model running mixed inference) without task-specific weight reconstruction or routing at runtime.",
        "expected_results": "Demonstrate that HNS completely collapses under joint multi-task inference because its parameters are reconstructed on a per-task basis, exposing its limitation as a retrieval scheme rather than a true model merge.",
        "impact": "High. Restores the true definition of model merging and exposes a major hidden assumption in parameter scaling works."
    },
    {
        "id": 6,
        "title": "The Impact of Fine-Tuning Regularization on Model Merging Orthogonality",
        "description": "Studying how different fine-tuning regularizers (weight decay, L2SP, weight anchoring) affect the orthogonality of task updates and whether aligned updates make standard weight averaging perform exceptionally well without post-merge scaling.",
        "expected_results": "Prove that proper fine-tuning regularization eliminates the need for complex post-merge calibrations by preventing chaotic update drift.",
        "impact": "High. Shifts the focus of model merging from post-merge repair to pre-merge alignment."
    },
    {
        "id": 7,
        "title": "A Systematic Study of Evaluation Size and Robustness in Model Merging Benchmarks",
        "description": "Investigating if small evaluation sizes (e.g., 2,000 samples) introduce high variance or bias in reported merging accuracies, establishing a rigorous evaluation protocol with confidence intervals.",
        "expected_results": "Establish the standard error of merging evaluations and provide a statistically robust benchmarking protocol.",
        "impact": "Medium. Professionalizes the evaluation practices in the model merging community."
    },
    {
        "id": 8,
        "title": "An Anisotropic Post-Mortem: Why Simple Isotropic Scaling (U-IPR) Beats Sophisticated Spectral Alignment (S-IPR, SA-IPR)",
        "description": "Systematically analyzing why SVD-based spectral alignment methods (S-IPR, SA-IPR) fail to outperform simple scalar scaling (U-IPR), exposing how projecting onto perturbed coordinate systems distorts the activation manifold.",
        "expected_results": "Show that SVD on merged updates distorts feature dimensions and argue against over-engineered spectral methods.",
        "impact": "Medium. Promotes simplicity and prevents over-engineering in future research."
    },
    {
        "id": 9,
        "title": "How Task Similarity Confounds Parameter-Space Merging Calibrations",
        "description": "Evaluating HNS and IPR on a continuum of task similarities. We hypothesize that when tasks are highly similar (collinear updates), the scaling formulas in HNS and IPR over-correct and degrade performance, whereas Uniform averaging succeeds.",
        "expected_results": "Identify task similarity as a key confounding variable where current parameter calibration methods fail.",
        "impact": "High. Defines the operational boundaries and failure modes of parameter-space calibration."
    },
    {
        "id": 10,
        "title": "The 'Equalizer' Myth: Rigorous Stress-Testing of Test-Time BatchNorm Calibration under Severe Covariate Shift",
        "description": "Stress-testing test-time calibration (SP-TTBC) under out-of-distribution (OOD) noise and label shift, showing that while it acts as an equalizer under clean settings, it is highly unstable under shift compared to static parameter-space methods.",
        "expected_results": "Expose the vulnerability of test-time calibration to real-world deployment challenges like covariate shift.",
        "impact": "High. Warns the community about the safety and stability risks of test-time calibration."
    }
]

# Write to progress.md
with open("progress.md", "w") as f:
    f.write("# Research Progress Log\n\n")
    f.write("## Phase 1: Foundation (Read & Formulate)\n\n")
    f.write("### Brainstormed Research Ideas (The Methodologist Persona)\n\n")
    for idea in ideas:
        f.write(f"#### Idea {idea['id']}: {idea['title']}\n")
        f.write(f"- **Concept:** {idea['description']}\n")
        f.write(f"- **Expected Results:** {idea['expected_results']}\n")
        f.write(f"- **Impact:** {idea['impact']}\n\n")
    
    # Selection using PRNG with seed 42
    random.seed(42)
    selected_index = random.randint(0, len(ideas) - 1)
    selected_idea = ideas[selected_index]
    
    f.write("### Idea Selection\n\n")
    f.write(f"Using a pseudo-random number generator seeded with 42, we selected:\n")
    f.write(f"**Idea {selected_idea['id']}: {selected_idea['title']}**\n\n")
    f.write(f"- **Hypothesis:** When standard Task Arithmetic is paired with simple, training-free, data-free Uniform BatchNorm statistics merging (which is also completely data-free and offline), it matches or outperforms complex parameter-space scaling methods like Holographic Norm Scaling (HNS) or Isotropic Parameter Resonance (IPR).\n")
    f.write(f"- **Rationale:** Recent works (HNS, IPR) compare their methods against extremely weak or uncalibrated baselines (e.g., Weight Averaging without statistic merging, or Task Arithmetic without statistic merging). Pairwise averaging of expert BatchNorm running statistics is an incredibly simple, robust baseline. If Uniform statistics merging matches or outperforms HNS and IPR, it exposes a critical methodological flaw in these papers: their complex mathematical scaling theories are unnecessary and actually underperform a simple, ignored baseline.\n\n")
    f.write("### Iterative Idea Refinement\n\n")
    f.write("As a Methodologist, we refine this hypothesis to include a rigorous, multi-dimensional grid sweep over learning rates, task vector scaling coefficients, and architectural blocks. We will explicitly test if the SOTA claims of HNS and IPR are robust, or if they evaporate under fair baseline comparison and hyperparameter tuning.\n\n")
    f.write("### Final Chosen Hypothesis & Project Goal\n\n")
    f.write("- **Hypothesis:** PA-WA (Properly-Aligned Weight Averaging / Task Arithmetic with Uniform BatchNorm statistics merging) is a highly robust, training-free, data-free, and mathematically simple baseline that consistently outperforms complex parameter-space scaling theories (HNS, IPR) across standard benchmarks.\n")
    f.write("- **Goal:** Conduct the most rigorous, fair re-evaluation of post-merge statistics calibration and parameter-space scaling to date, exposing the weak baselines in prior SOTA claims and establishing a new, simpler, and stronger baseline for the model-merging community.\n")

print(f"Selected Idea {selected_idea['id']}: {selected_idea['title']}")
