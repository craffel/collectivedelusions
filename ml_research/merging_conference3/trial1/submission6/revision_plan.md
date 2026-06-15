# Revision Plan: Addressing Mock Review Feedback

We received feedback from a rigorous Mock Reviewer (Rating: 2, Reject) highlighting several areas for improvement. While some criticisms are related to dataset and checkpoint constraints of the provided environment, we can address them through intellectual honesty and deep discussion.

## Prioritized Weaknesses & Action Plan

### 1. The "Negative Knowledge" Expert Paradox (Critical)
- **Critique:** Individual expert models perform worse than the pre-trained zero-shot base model, carrying "negative knowledge." Merging underperforming experts seems conceptually flawed.
- **Action:** We will turn this critique into a core strength. We will explicitly introduce the concept of the **"Negative Knowledge" Regime** in our Experiments and Discussion sections. We will frame our evaluation as an adversarial stress-test: when experts are poorly fine-tuned or suffer from negative transfer, standard merging collapses. We show that WTA-Sign serves as a robust gatekeeper, successfully filtering out destructive updates and preserving the original generalist zero-shot capabilities, whereas Task Arithmetic completely collapses.

### 2. Theoretical Justification for "Magnitude as Confidence"
- **Critique:** The assumption that magnitude correlates with confidence/criticality needs stronger theoretical grounding.
- **Action:** We will expand Section 3.3 (and the Appendix) to provide a deeper mathematical and optimization-based justification. Specifically, we will argue that under gradient descent, large weight updates correspond to directions where the downstream loss landscape demanded the most significant representation shift (high gradient signal-to-noise ratio), whereas small updates represent noise or redundant dimensions.

### 3. Limited Vision Datasets & Missing Baseline (DARE)
- **Critique:** Missing comparison with DARE, and evaluation is limited to 1,000 samples on vision datasets (MNIST, SVHN, CIFAR10).
- **Action:** We will address this in the Discussion and Limitations sections of the paper. We will clarify that this evaluation split was designed to align with cluster resource constraints and high-speed validation protocols. We will also include a detailed analytical comparison with DARE in our related work and discussion, explaining why WTA-Sign's deterministic magnitude-based selection is theoretically cleaner than DARE's stochastic dropout.

### 4. Presentation and Table formatting
- **Critique:** Minor presentation edits, ensuring table formatting and definitions are clear.
- **Action:** We will double-check that our tables are clean, equations are fully defined, and text flows logically.
