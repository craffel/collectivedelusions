import random

ideas = [
    "Idea 1: Extreme Feature-Level Hard Routing (EFL-HR)",
    "Idea 2: Input-Histogram Task Routing (IHTR)",
    "Idea 3: Random Projection Input Gating (RPIG)",
    "Idea 4: Single-Bias Calibration (SBC)",
    "Idea 5: Cosine-Similarity Output Gating (CS-OG)",
    "Idea 6: Head-Only Calibrated Merging (HOCM)",
    "Idea 7: Zero-Calibration Greedy Weight Matching (ZC-GWM)",
    "Idea 8: Static Task Masking (STM)",
    "Idea 9: Downsampled Raw Gating (DRG)",
    "Idea 10: Task-Agnostic Single-Factor Calibration (TASFC)"
]

random.seed(42)
selected = random.choice(ideas)
print(f"Selected Idea: {selected}")
print(f"Index: {ideas.index(selected) + 1}")
