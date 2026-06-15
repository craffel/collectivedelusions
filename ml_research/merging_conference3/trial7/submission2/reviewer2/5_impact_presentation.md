# 5. Impact and Presentation

## Major Strengths
1. **Mathematically Rigorous and Elegant Foundation:** The paper successfully unifies information-geometric curvature (Riemannian geometry) with test-time model merging. Modeling parameter and representation spaces as Riemannian manifolds and using diagonal Fisher Information to warp local geometries is a beautiful and sound mathematical concept.
2. **Training-Free & Parameter-Free Test-Time Adaptation:** By completely bypassing test-time optimization of routing parameters, the framework is immune to severe few-shot overfitting ("The Dynamic Routing Paradox") and sequential stream instability ("Vectorization Collapse") that plague parametric routers. This is highly attractive for real-world low-latency edge deployment.
3. **Honest Self-Critique and Transparency:** The authors provide an exceptionally thorough and transparent discussion of limitations in Appendix A.1, explicitly outlining seven key weaknesses, systems-level trade-offs, and practical constraints. This level of self-awareness is refreshing and highly valuable.
4. **Concrete Engineering Safeguards:** The introduction of Class-Size Scaling Calibration (CSC) and Micro-Batch Homogenization (MBH) directly addresses real-world, practical deployment challenges (asymmetric class vocabularies and heterogeneous streams), bridging the gap between theory and application.
5. **Excellent Experimental Breadth:** Evaluating the proposed framework across a synthetic sandbox, a realistic LoRA activation simulation, and a physical end-to-end ResNet-18 backbone demonstrates a commendable commitment to empirical validation.

## Areas for Improvement (Practitioner's Critical Lens)

From the perspective of real-world utility, ease of deployment, and systems scalability, the paper has several critical weaknesses:

### 1. The Systems-Level Bottleneck of Micro-Batch Homogenization (MBH)
In heterogeneous stream settings, MBH partitions a mixed batch of size $B$ into $G \le K$ homogeneous micro-batches, executing $G$ separate sequential forward passes with dynamically assembled weights.
- **Worst-case Computational Redundancy:** If the stream is highly diverse, we have $G = K$ micro-batches. Running $K$ sequential forward passes with $K$ distinct dynamically merged models is computationally equivalent (in terms of FLOPs) to simply executing the original, unmerged specialized expert models on their respective samples. 
- **Defeating the Merge Purpose:** Dynamic weight merging's primary goal is to run a *single* merged model to handle multiple tasks. MBH defeats this purpose in the worst-case, adding significant systems latency, dynamic weight assembly overhead, and memory bandwidth bottleneck.
- **Ensembling Trade-off:** Bounding this via Top-$1$ expert gating completely eliminates sequential micro-batching overhead, but replaces true weight ensembling with a hard task-routing selection. This represents a major conceptual compromise that practitioners must make.

### 2. Underwhelming Gains on Real-World Networks (The Physical Gap)
While FIOSR achieves dramatic improvements in the synthetic sandbox ($+8.56\%$), the end-to-end physical ResNet-18 validation shows a very modest performance gain:
- Routing accuracy improves by only **+2.67%** (59.00% vs. 56.33%).
- Joint ensembling accuracy improves by an insignificant **+1.33%** (52.00% vs. 50.67%).
- Both methods fall **17.67%** short of the Direct Expert Routing Oracle (69.67%).
- This suggests that in actual non-Gaussian, highly correlated, and spatial representation spaces of physical neural networks, the diagonal Fisher-weighted coordinate warping provides very limited practical utility. The substantial mathematical overhead of FIM estimation and smoothing yields negligible real-world returns.

### 3. Calibration Dependency (Not Truly Zero-Shot)
To estimate the coordinate variances and dFIM, the method requires a calibration split of $N_c \ge 8$ samples per task. This dependency means that practitioners cannot deploy FIOSR immediately out-of-the-box for a new, uncalibrated task without first collecting a small calibration set and passing it through the network, which adds operational complexity in dynamic streaming environments.

### 4. Purely Theoretical LLM Scaling Strategies
While the authors propose "Class-Grouped Pooling" and "Low-Rank FIM Factorization" to handle storage and memory overhead for massive LLM vocabularies ($32\text{K}$ to $128\text{K}$ tokens), these strategies are purely theoretical and lack any empirical validation.

## Overall Presentation Quality
The presentation quality is **Excellent**.
- The writing style is highly professional, precise, and logical.
- The authors utilize rigorous mathematical language and provide detailed proofs and derivations for every claim.
- Hyperparameters and experimental setups are exhaustively documented, making the work highly reproducible.
- Figures and tables are clean, informative, and well-designed.

## Potential Impact and Significance
The paper has **Moderate** potential impact:
- **High Theoretical Significance:** The information-geometric perspective and formal dual-space proof will likely inspire future researchers in the modular deep learning and model merging fields to move away from flat Euclidean ensembling heuristics and explore Riemannian geometries.
- **Limited Practical Impact:** For real-world engineering and industry practitioners, the severe sequential latency bottlenecks of MBH, the calibration split dependency, and the very small empirical gains on physical networks (+1.33%) mean that this method is unlikely to be deployed in production streams over simpler static merging or hard routing alternatives in its current form.
