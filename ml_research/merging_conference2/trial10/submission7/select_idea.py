import random

# Seeding for reproducibility
random.seed(2026)

ideas = [
    {
        "id": 1,
        "title": "Dynamic Task-Specific Weighting under Fluctuating Edge Resources",
        "description": "Adjusting expert merging coefficients on-the-fly to meet changing memory and latency constraints during deployment.",
        "expected_results": "Maintains optimal accuracy-latency trade-offs across different compute budgets without retraining.",
        "impact": "High utility for edge devices with dynamic workloads or battery-saving modes."
    },
    {
        "id": 2,
        "title": "Activation-Aware Channel Pruning for Quantized Merged Models (ACP-QMM)",
        "description": "Pruning convolutional channels based on the activation magnitudes obtained during the tiny-batch DE-BN calibration, followed by PTQ.",
        "expected_results": "Reduces model size and latency by 30-50% with minimal accuracy loss under 8-bit and 4-bit quantization.",
        "impact": "Directly addresses inference speed and memory constraints on edge hardware."
    },
    {
        "id": 3,
        "title": "Robust Data-Efficient BatchNorm Calibration (R-DEBN) under Noisy and OOD Deployments",
        "description": "Calibrating BatchNorm statistics using a robust estimator (e.g., huber-mean or trimmed variance) or regularizing them with the progenitor's stats to withstand environmental noise and Out-Of-Distribution (OOD) test-time shifts.",
        "expected_results": "Significantly outperforms standard DE-BN on corrupted (blur, noise) datasets and minor OOD tasks.",
        "impact": "Crucial for reliable real-world deployment in unpredictable environments (e.g., autonomous driving, outdoor cameras)."
    },
    {
        "id": 4,
        "title": "Task-Conditioned BatchNorm (TC-BN) for Mixed-Batch Multi-Task Serving",
        "description": "Eliminating the routing and latency overhead of task-specific BN switching by predicting BN statistics dynamically from a lightweight task-embedding or routing network, allowing mixed-task batches to be processed in a single forward pass.",
        "expected_results": "Enables zero-overhead mixed-batch serving with identical accuracy to task-specific DE-BN.",
        "impact": "Critical for high-throughput cloud and edge serving where inputs are not pre-grouped by task."
    },
    {
        "id": 5,
        "title": "Outlier-Aware Weight-Activation Scaler (OWA-Scale) for Ultra-Low Bit Quantization",
        "description": "A very simple, data-free or data-efficient channel-wise weight and activation rescaling method designed to clamp and smooth outlier values before uniform 2-bit or 4-bit quantization.",
        "expected_results": "Restores performance of merged models to >70% under 4-bit and >50% under 2-bit quantization with minimal computation.",
        "impact": "Facilitates deployment on extreme microcontrollers and low-power IoT devices."
    },
    {
        "id": 6,
        "title": "Task-Specific Feature-Sharing Merging (TS-FSM) with Dynamic Gating",
        "description": "Merging only the general early feature extraction layers of the backbone and keeping late layers/heads separate, guided by a simple training-free dynamic routing/gating mechanism.",
        "expected_results": "Mitigates cross-task interference while keeping the parameter count low and serving speed high.",
        "impact": "Provides a practical alternative to full model merging when task interference is high."
    },
    {
        "id": 7,
        "title": "Mixed-Precision Quantization of Merged Models (MP-QMM) via Activation MSE Optimization",
        "description": "A fast, data-efficient layer sensitivity analysis using the calibration set to automatically assign optimal bit-widths (e.g., 4, 8, or 16-bit) to different layers of the merged model.",
        "expected_results": "Achieves the best trade-off between model size, inference speed, and accuracy compared to uniform quantization.",
        "impact": "Allows practitioners to optimize merged models for specific hardware architectures with mixed-precision support."
    },
    {
        "id": 8,
        "title": "On-Device Continuous Calibration of BatchNorm (OD-CC) for Streaming Data Drifts",
        "description": "Updating task-specific BN statistics continuously on-device using a running exponential moving average on streaming test inputs, without requiring labels, storing data, or offline calibration.",
        "expected_results": "Maintains or improves model performance over time as the deployment environment's data distribution drifts.",
        "impact": "Perfect for continuous deployment scenarios like robotic vision, weathering sensors, or wearable devices."
    },
    {
        "id": 9,
        "title": "Sparsity-Driven Merging with Zero-Shot Weight Compression (SD-ZWC)",
        "description": "Combining sparse merging (like DARE/TIES) with post-training quantization to aggressively compress expert weights and reduce memory bandwidth bottlenecks.",
        "expected_results": "Produces highly sparse, low-bit models that run extremely fast on sparse-aware edge hardware.",
        "impact": "Unlocks execution speedups on hardware with native sparse matrix-multiplication support."
    },
    {
        "id": 10,
        "title": "Unified Serving of Merged Models with Agnostic Task Routing (ATR-Merge)",
        "description": "Developing a training-free or extremely cheap task routing mechanism (e.g., a tiny classifier on top of intermediate progenitor features) that automatically predicts which task an input belongs to, routing it to the correct head and applying the correct BN statistics.",
        "expected_results": "Enables completely task-agnostic serving of a merged multi-task model at test time, with performance close to the task-labeled oracle.",
        "impact": "Solves the massive practical limitation where the test-time task label is unknown."
    }
]

selected_idx = random.randint(0, len(ideas) - 1)
selected_idea = ideas[selected_idx]

print(f"Total ideas: {len(ideas)}")
print(f"Selected index: {selected_idx}")
print(f"Selected idea: {selected_idea['title']}")
print(f"Description: {selected_idea['description']}")
