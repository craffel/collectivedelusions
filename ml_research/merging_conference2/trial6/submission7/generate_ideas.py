import random

ideas = [
    {
        "id": 1,
        "title": "A Rigorous Stress-Test of Model Merging Calibration Under Distribution Shift",
        "description": "Existing calibration methods (like SLR-WBC, WRSA, and SP-TAAC) assume the calibration set and test set are from the same distribution. We analyze how distribution shift (e.g. CIFAR-10-C, MNIST-C) between calibration and test sets affects calibration robustness, exposing that spatial and frequency calibration can overfit to clean calibration sets and collapse under shift.",
        "expected_results": "We expect to find that complex calibration methods overfit to the clean calibration distribution and their performance degrades significantly more under shift compared to simple weight averaging or joint uncalibrated baselines.",
        "impact": "Highlights a major hidden vulnerability in existing calibration methods and pushes the community to evaluate model merging under more realistic, non-i.i.d. scenarios."
    },
    {
        "id": 2,
        "title": "Deconstructing the Localization Illusion: Do Early Layers Remain Task-Agnostic Under Large Task Counts?",
        "description": "MSPR and SRAC rely on the 'Localization Illusion' (early layers are intact and task-agnostic). We stress-test this assumption by merging an increasing number of tasks (from 3 to 10). We evaluate if task interference leaks into early layers (Layer 2) as task count grows, which would degrade early-layer routing performance.",
        "expected_results": "We expect that as the number of merged tasks increases, the CKA of early layers between the merged model and experts drops significantly, and task routing based on Layer 2 cosine similarity becomes highly inaccurate.",
        "impact": "Exposes the limits of early-layer task routing and the 'Localization Illusion' hypothesis, guiding future research toward more robust routing locations."
    },
    {
        "id": 3,
        "title": "Are Calibration Benefits an Artifact of Poorly Tuned Merging Hyperparameters?",
        "description": "We investigate whether the reported improvements of calibration methods (like SLR-WBC or WRSA) vanish when the merging hyperparameters (e.g., task vector scaling lambda in Task Arithmetic, or individual expert weights) are exhaustively and rigorously tuned using hyperparameter optimization.",
        "expected_results": "We expect that a significant portion of the calibration gain (e.g., >50%) is simply correcting for sub-optimal scaling, and that a highly tuned uncalibrated baseline is much stronger than previously reported.",
        "impact": "Pushes for more rigorous baseline tuning in model merging research, showing that many complex methods are overperforming against untuned strawman baselines."
    },
    {
        "id": 4,
        "title": "Sparsity-Utility Trade-offs in Post-Merge Activation Calibration",
        "description": "Calibration methods like SLR-WBC modify weights/activations to restore variance, but they might inadvertently destroy the native activation sparsity of ReLU networks, leading to a massive increase in activation density. This study evaluates the trade-off between task performance and activation sparsity/efficiency.",
        "expected_results": "We expect to find that SLR-WBC and other calibration methods drastically increase the percentage of active (non-zero) neurons, showing a hidden cost of calibration in terms of inference-time efficiency.",
        "impact": "Exposes a hidden trade-off in calibration methods and advocates for sparsity-preserving calibration techniques."
    },
    {
        "id": 5,
        "title": "A Critical Analysis of Calibration Set Selection: Size, Bias, and Class Imbalance",
        "description": "We rigorously evaluate how the composition of the calibration set (sample size, class imbalance, and selection bias) affects post-merge calibration. We show that existing calibration methods are highly sensitive to these factors, which is often hidden by using clean, balanced calibration sets.",
        "expected_results": "We expect to find that class-imbalanced calibration sets severely bias the calibration parameters, causing catastrophic accuracy drops on underrepresented classes, exposing a major robustness issue.",
        "impact": "Establishes guidelines for robust calibration set selection and highlights the necessity of evaluating calibration under class imbalance."
    },
    {
        "id": 6,
        "title": "Is Frequency-Domain Calibration Actually Superior to Properly Regularized Spatial Calibration?",
        "description": "WRSA claims frequency-domain is superior because of spectral properties. We implement a highly regularized spatial calibration baseline (e.g., with weight decay, ridge, or dropout) and compare it to WRSA, testing if the frequency-domain benefits are simply due to a lack of proper regularization in spatial baselines.",
        "expected_results": "We expect that properly regularized spatial-domain calibration (using ridge or Lasso on channel-wise scaling) matches or exceeds WRSA performance, showing that frequency-domain operations are not strictly necessary.",
        "impact": "Deconstructs the necessity of complex frequency-domain calibration, favoring simpler, well-regularized spatial-domain alternatives."
    },
    {
        "id": 7,
        "title": "Robustness of Model Merging Routers Under Mixed-Task Test Streams",
        "description": "MSPR and SRAC evaluate on batches containing a single task. We evaluate how these routers perform under realistic mixed-task streams (where inputs from different tasks are interleaved), evaluating the impact of batch size, stream composition, and class overlap on routing accuracy.",
        "expected_results": "We expect that when samples are interleaved, batch-level routing collapses, and sample-level routing is highly sensitive to class overlap and representation shift.",
        "impact": "Highlights a major practical limitation of batch-based model merging routers, guiding the design of stream-robust routing methods."
    },
    {
        "id": 8,
        "title": "A Multi-Seed and Statistical Significance Audit of Model Merging Calibration",
        "description": "We perform a large-scale statistical audit of model merging calibration. By running multiple seeds for training, merging, and calibration set sampling, we analyze if the performance gains of SLR-WBC or WRSA are statistically significant or within the margin of seed-level variance.",
        "expected_results": "We expect to find that some calibration improvements are statistically marginal compared to the high variance introduced by different fine-tuning seeds and calibration samples.",
        "impact": "Advocates for more rigorous statistical reporting and significance testing in the model merging literature."
    },
    {
        "id": 9,
        "title": "Can Simple BatchNorm Re-calibration Obsoletize Complex Weight Correction?",
        "description": "SLR-WBC proposes low-rank weight corrections. We investigate if simply running a few steps of BatchNorm statistics re-estimation (with proper momentum and/or affine parameter tuning) on the calibration set can match SLR-WBC, exposing that complex weight correction is largely redundant.",
        "expected_results": "We expect that a carefully tuned BatchNorm statistics update (e.g. running statistics recalculation + slight scaling) recovers most of the performance, making SVD weight corrections unnecessary.",
        "impact": "Simplifies the model merging calibration pipeline, proving that simple, standard deep learning operations are sufficient."
    },
    {
        "id": 10,
        "title": "The Spatial Phase Deconstruction: Why Frequency-Domain Calibration Fails on Out-of-Distribution Structure",
        "description": "Frequency-domain methods like WRSA modify magnitudes but keep phases intact. We investigate if this assumption holds under out-of-distribution structural changes (e.g. rotation, scaling, or blurring), exposing how phase-magnitude misalignment under OOD inputs degrades WRSA compared to spatial-domain methods.",
        "expected_results": "We expect that under OOD structural corruptions, keeping the expert phase with merged magnitude (or vice-versa) leads to severe visual and representation distortion, causing WRSA to underperform spatial methods.",
        "impact": "Identifies a structural limitation of frequency-domain merging calibration and emphasizes the importance of phase alignment."
    }
]

# Use a PRNG to select one idea
random.seed(2026) # Seed with current year
selected_idx = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_idx]

print(f"Selected Idea Index: {selected_idx}")
print(f"Selected Idea: {selected_idea['title']}")

# Now let's write to progress.md
with open("progress.md", "w") as f:
    f.write("# Research Progress Log\n\n")
    f.write("## Phase 1: Foundation (Read & Formulate)\n\n")
    f.write("### Literature Synthesis\n")
    f.write("- **Submission 1 (SLR-WBC):** In-place, SVD-based Low-Rank Weight Correction and BatchNorm alignment to resolve representation collapse without online inference hooks.\n")
    f.write("- **Submission 7 (WRSA):** Frequency-domain calibration modeling spectral alignment as a Wiener deconvolution problem to resolve noise amplification (Spectral Sparsity Trap) under pointwise division.\n")
    f.write("- **Submission 9 (MSPR):** Minimalist Static Prototype Routing. Extracts task prototypes from early layers (Layer 2) and performs cosine-similarity hard routing once at test-time to route samples/heads, proving dynamic online routing is redundant.\n\n")
    
    f.write("### Generated Research Ideas\n")
    for idea in ideas:
        f.write(f"#### Idea {idea['id']}: {idea['title']}\n")
        f.write(f"- **Description:** {idea['description']}\n")
        f.write(f"- **Expected Results:** {idea['expected_results']}\n")
        f.write(f"- **Impact:** {idea['impact']}\n\n")
        
    f.write("### Selection\n")
    f.write(f"Using a PRNG (seed=2026), we selected **Idea {selected_idea['id']}**: *{selected_idea['title']}*.\n\n")
    f.write(f"- **Hypothesis:** {selected_idea['description']}\n")
    f.write(f"- **Rationale:** As a Methodologist, we are highly skeptical of SOTA claims and complex architectures. Evaluating whether these calibration benefits are merely artifacts of sub-optimal merging hyperparameters directly addresses a fundamental methodological flaw in the model merging literature: the use of weak, untuned baselines.\n\n")
    
    f.write("### Iteration & Refinement of Selected Idea\n")
    f.write("We will improve the selected idea by formulation of a concrete research question and experimental setup.\n")
    f.write("- **Research Question:** Does exhaustive hyperparameter tuning of the uncalibrated merged model (e.g. tuning task vector scaling coefficients and merging weights per layer/task) close the gap with post-merge calibration methods? Are post-merge calibration methods simply acting as an expensive patch for poorly-tuned merging parameters?\n")
    f.write("- **Experimental Protocol:** We will train experts on MNIST, Fashion-MNIST, and CIFAR-10, merge them using Weight Averaging and Task Arithmetic, and compare (1) standard untuned merging, (2) exhaustively tuned merging, and (3) untuned merging + post-merge calibration (SLR-WBC/WRSA). This will expose whether calibration benefits are indeed a byproduct of sub-optimal hyperparameters.\n")
