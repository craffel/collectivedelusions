# Research Agent Operating Plan

## Objective
Execute a complete research cycle—from literature review and idea formulation to experimentation and paper writing—within a 24-hour window on an 8x H100 node.

## Runtime Instructions
This agent is invoked every 10 minutes. On each start:
1. **Check Remaining Time:** Run `squeue -h -j $SLURM_JOB_ID -O TimeLeft` to monitor the 24-hour deadline.
2. **Load State:** Read `progress.md` to determine the current status and the next pending task.
3. **Execute & Persist:** Perform the next step in the workflow and immediately record progress, experimental results, and any state changes to `progress.md`.

## Workflow Phases

### Phase 1: Foundation (Read & Formulate)
- **Input:** Read the three PDF papers located in the `papers/` directory.
- **Synthesis:** Identify core contributions, limitations, and potential extensions.
- **Idea Generation:** Formulate a novel research idea that builds directly on these three papers. Record the hypothesis and rationale in `progress.md`.

### Phase 2: Literature Search
- **Tool:** Use the Semantic Scholar API as described in `semantic_scholar.md`. Use the API key store in the `SEMANTIC_SCHOLAR_API_KEY` environment variable.
- **Action:** Search for papers related to the new research idea to ensure novelty and find supporting/competing evidence.
- **Outcome:** Download relevant PDFs if necessary and refine the research idea based on findings.

### Phase 3: Experimentation
- **Formulation:** Design a set of experiments to test the hypothesis.
- **Implementation:** Write the necessary Python code.
- **Environment:** All code, models, and datasets should be written to the current working directory. Do not access any files outside of /fsx/craffel/collectivedelusions/ml_research/. Create and use a local environment with `uv` to manage any new dependencies.
- **Execution:** Run experiments using the available 8x H100 GPUs. Optimize for the 24-hour total time limit.
- **AI:** Optionally use the Gemini API with the `GEMINI_API_KEY` environment variable.
- **Analysis:** Collect and analyze results. Save key plots and metrics.

### Phase 4: Paper Writing
- **Template:** Use the LaTeX template in the `template/` directory.
- **Constraints:** Maximum 6 pages.
- **Content:** Include Abstract, Introduction, Related Work (using Semantic Scholar results), Methodology, Experiments, Results, and Conclusion.
- **Compilation:** Ensure the LaTeX document compiles correctly to a PDF.

## Critical Requirements
- **Persistence:** Every significant action and decision MUST be recorded in `progress.md`. If the agent is interrupted, it must be able to resume exactly where it left off.
- **Deadlines:** All phases—including final paper compilation—must be completed within the 24-hour Slurm job allocation.
- **Efficiency:** Parallelize experiment implementation and execution where possible to maximize GPU utilization.
