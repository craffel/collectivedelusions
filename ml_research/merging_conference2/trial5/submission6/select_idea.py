import random

ideas = [
    {
        "id": 1,
        "title": "Analytical BatchNorm Scaling (ABS)",
        "description": "Instead of using a calibration dataset to find scaling factors, mathematically scale down the merged running variance of each BatchNorm layer by a factor of 1/M (where M is the number of merged experts) or dynamically based on the cosine similarity of the expert weights to combat variance collapse in a data-free manner.",
        "expected_impact": "Completely eliminates the need for any calibration dataset and online hooks, achieving near-parity with SP-TAAC with zero data and zero-shot parameter adjustments."
    },
    {
        "id": 2,
        "title": "Frequency-Flat Spatial Scaling (FFSS)",
        "description": "Since FDSA uses complex 2D FFTs on activations to find frequency-domain scaling factors, show that we can achieve 90%+ of the benefits by simply applying a single spatial scalar to the convolutional weights, proving that complex frequency-domain calibration is needlessly complex.",
        "expected_impact": "Bypasses all Fourier transforms, simplifying FDSA to a simple static weight scaling method that works on non-image data too."
    },
    {
        "id": 3,
        "title": "Task-Agnostic Prototype-Free Routing (TAPFR)",
        "description": "Instead of using SRAC's complex routing against task prototypes using early layers, use a simple activation norm threshold (e.g., L2 norm of the first layer) to route activations, bypassing prototype and similarity computations.",
        "expected_impact": "Strips away clustering, prototype storage, and cosine similarity calculations from SRAC, leading to a much cleaner and faster routing system."
    },
    {
        "id": 4,
        "title": "Data-Free Covariance-Aware Merging (DF-CAM)",
        "description": "Estimate the covariance between expert models' weights to analytically adjust the merged model's weights and BatchNorm scales, achieving representation alignment without any forward passes on calibration data.",
        "expected_impact": "Enables high-performance calibration without needing any sample images, enhancing privacy and ease of deployment."
    },
    {
        "id": 5,
        "title": "Linear SVD-Free Subspace Merging (LSF-SM)",
        "description": "Instead of complex Singular Value Decomposition (SVD) on weights to resolve interference, use simple coordinate-wise weight alignment before averaging.",
        "expected_impact": "Simplifies weight-space alignment methods by eliminating computationally expensive and complex SVD decompositions."
    },
    {
        "id": 6,
        "title": "In-Place Output-Head Centering (IP-OHC)",
        "description": "Show that instead of doing head adaptation (SFT) on a dataset, we can simply align the mean/variance of the final classification heads of the experts in weight space, restoring accuracy with zero training.",
        "expected_impact": "Replaces post-merge supervised fine-tuning (SFT) with a zero-shot parameter translation, achieving high performance with zero gradient steps."
    },
    {
        "id": 7,
        "title": "Uniform Layer-wise Weight Rescaling (ULWR)",
        "description": "Prove that the complex activation calibration of SP-TAAC / TAAC can be matched by simply applying a single uniform scale factor to the weights of the deeper layers of the merged network, eliminating all calibration hooks.",
        "expected_impact": "Bypasses activation-space hooks entirely by adjusting static parameter weights with a single global coefficient."
    },
    {
        "id": 8,
        "title": "Elegantly Pruned Expert Merging (EPEM)",
        "description": "Before merging, aggressively prune low-magnitude weights from each expert, showing that sparser experts have less interference and require no activation calibration at all.",
        "expected_impact": "Shows that simple parameter-space sparsity can replace complex post-merge calibration pipelines, leading to faster inference and smaller models."
    },
    {
        "id": 9,
        "title": "Zero-Shot Orthogonal Alignment (ZSOA)",
        "description": "Orthogonally project the weight matrices of different experts to align them before simple weight averaging, which minimizes phase interference in activations.",
        "expected_impact": "Aligns representations in parameter space before merging, avoiding the need for activation-space corrections entirely."
    },
    {
        "id": 10,
        "title": "Global Activation-Free Variance Injection (GAF-VI)",
        "description": "Add a constant small epsilon or scaling factor directly to the network's residual connections to combat representation collapse without layer-specific calibration.",
        "expected_impact": "Provides a dead-simple, network-wide architectural tweak that automatically prevents variance decay."
    }
]

# Using a fixed seed for reproducible PRNG selection as instructed
random.seed(42)
selected_index = random.randint(0, 9)
selected_idea = ideas[selected_index]

print("--- ALL IDEAS ---")
for idea in ideas:
    print(f"{idea['id']}. {idea['title']}")
    print(f"   Description: {idea['description']}")
    print(f"   Expected Impact: {idea['expected_impact']}\n")

print("--- SELECTED IDEA ---")
print(f"Selected Idea {selected_idea['id']}: {selected_idea['title']}")
print(f"Description: {selected_idea['description']}")
print(f"Expected Impact: {selected_idea['expected_impact']}")
