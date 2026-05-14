# Research Agent Operating Plan

## Objective
Execute a complete research cycle—from literature review and idea formulation to experimentation and paper writing—within a 6-hour window on an 8x H100 node.
The final paper should be of the same length, quality, and level of contribution as the papers in the `papers/` directory.
Closely follow the general guidelines on doing effective research provided in `research_tips.md`.

## Runtime Instructions
This agent is invoked every 10 minutes. On each start:
1. **Check Remaining Time:** Run `squeue -h -j $SLURM_JOB_ID -O TimeLeft` to monitor the 6-hour deadline.
2. **Load State:** Read `progress.md` to determine the current status and the next pending task. If all phases (1-3) are marked as complete, transition to Phase 4.
3. **Execute & Persist:** Perform the next step in the workflow and immediately record progress, experimental results, and any state changes to `progress.md`. Retain all notes in progress.md by appending updates.

## Workflow Phases

### Phase 1: Foundation (Read & Formulate)
- **Input:** Read the three PDF papers located in the `papers/` directory.
- **Synthesis:** Identify general themes, core contributions, limitations, and potential extensions.
- **Literature search:** Search for related papers covering related themes and methods, including background work.
- **Tool:** Use Google Search or the Semantic Scholar API as described in `semantic_scholar.md`. Use the API key stored in the `SEMANTIC_SCHOLAR_API_KEY` environment variable.
- **Outcome:** Download relevant PDFs if necessary and refine the research idea based on findings.
- **Idea Generation:** Formulate a novel research idea on the identified theme.
- **Iterate:** If possible, improve upon the novelty, feasibility, and importance of the proposed research idea by reconsidering prior work.
- **Commit:** Record the hypothesis and rationale in `progress.md`.

### Phase 2: Experimentation
- **Formulation:** Design a set of experiments to test the hypothesis.
- **Implementation:** Write the necessary Python code.
- **Environment:** All code, models, and datasets should be written to the current working directory. Do not access any files outside of /fsx/craffel/collectivedelusions/ml_research/. Create and use a local environment with `uv` to manage any new dependencies.
- **Execution:** Run experiments using the available 8x H100 GPUs. Optimize for the 6-hour total time limit.
- **LLM API:** Optionally use the Gemini API with the `GEMINI_API_KEY` environment variable.
- **Analysis:** Collect and analyze results. Save key plots and metrics.

### Phase 3: Paper Writing
- **Template:** Use the LaTeX template in the `template/` directory.
- **Constraints:** Exactly 8 pages for the main paper, plus unlimited pages for references and appendix. Follow the formatting instructions in `template/example_paper.pdf`.
- **Sections:** Include an abstract, introduction, and related work section. Describe the proposed method, the experimental setup, and the results.
- **Visuals:** Include diagrams illustrating key ideas. Create plots to visualize results.
- **References:** Typical papers will have 50+ references, covering the body of related work in detail.
- **Evaluation:** The paper will be judged by the criteria specified in `reviewing_criteria.md`.
- **Compilation:** Ensure the LaTeX document compiles correctly to a PDF.

### Phase 4: Iterative Refinement
- **Condition:** If Phases 1-3 are complete and time remains.
- **Context:** Re-read `research_tips.md` and `revewing_criteria.md` to understand what makes a good paper and how to improve the project.
- **Action:** Restart the research cycle to refine and strengthen the work.
- **Refinement Strategy:**
  - **Literature:** Build a larger body of related work and expand the set of baselines.
  - **Core hypothesis:** Reconsider, refine, and expand the main hypothesis.
  - **Theory:** Add or expand theoretical justification if it improves the project's core arguments.
  - **Methodology:** Improve the proposed methodology in light of initial experimental results.
  - **Experimentation:** Expand the range of experimental settings, hyperparameters, and ablation studies.
  - **Writing:** Substantially improve the quality, clarity, and technical depth of the written paper.
- **State Management:** Update `progress.md` to begin again at Phase 1, ensuring that previous insights are preserved and used to inform the next iteration.
- **Requirements:** DO NOT declare the paper "finished" if time remains. There are always ways to improve the paper. Find them and take action.

## Critical Requirements
- **Persistence:** Every action and decision MUST be recorded in `progress.md`. If the agent is interrupted, it must be able to resume exactly where it left off. All updates should be appended to progress.md.
- **Deadlines:** All phases—including final paper compilation—must be completed within the 6-hour Slurm job allocation.
- **Efficiency:** Parallelize experiment implementation and execution where possible to maximize GPU utilization.
