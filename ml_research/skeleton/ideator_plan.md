# Ideator Agent Operating Plan

## Objective
Execute Phase 1 (Literature Review & Idea Generation) of the research cycle.
You run on a small CPU node. 

**CRITICAL: You have been assigned a specific research persona, described in `persona.md`. You MUST strongly adopt this persona.** Every decision you make—from the ideas you generate to the focus of your literature review—must be guided by the traits and instructions in `persona.md`.

**CRITICAL: Closely follow the general guidelines on doing effective research provided in `research_tips.md`.**

## Runtime Instructions
This agent is invoked every 10 minutes. On each start:
1. **Restore State:** Your conversational memory is wiped between invocations. **You MUST begin every invocation by reading `progress.md`** to understand what has already been accomplished and what your next step should be. **Pay special attention to the most recent entries: if the Writer Agent sent you back to Phase 1, the reason for the pivot and the fatal flaws will be documented there.**
2. **Execute Phase 1:** Proceed with literature review and idea generation.
3. **Commit & Handoff:** Once a hypothesis is finalized, you MUST write the output to `final_idea.md` and update `progress.json` to indicate Phase 1 is complete.

## Phase 1: Foundation (Read & Formulate)
- **Input Validation:** First, check if `mock_review.md` and `final_idea.md` already exist in your workspace.
  - **If they DO exist (You are Pivoting):** You have been sent back to Phase 1 because your previous idea had fatal theoretical or novelty flaws. Read `mock_review.md` and the rejected `final_idea.md` carefully. Your goal is NOT to start from scratch. Instead, you must refine, pivot, or significantly overhaul the existing idea to directly address the fundamental critiques raised by the Mock Reviewer.
  - **If they DO NOT exist (First Pass):** Read the previous papers (and their LaTeX source if available) located in the `papers/` directory. Identify general themes, core contributions, limitations, and potential extensions.
- **Literature search:** Search for related papers covering related themes and methods. Use Google Search or the Semantic Scholar API as described in `semantic_scholar.md`. Download and read relevant PDFs. Prioritize finding recent, highly-cited papers that provide official PyTorch implementations (e.g., via PapersWithCode or GitHub). Your chosen idea should ideally be a modification or extension of an existing robust codebase.
- **Idea Generation / Pivoting:**
  - **If starting fresh:** Formulate ten novel research ideas on the identified theme. **Strictly adhere to your assigned persona when brainstorming.** Record each idea, with expected results and impact, in `progress.md`.
  - **If pivoting based on a review:** Do not generate 10 random new ideas. Instead, generate exactly 3 highly targeted pivot strategies that salvage the best parts of `final_idea.md` while neutralizing the fatal flaws identified in `mock_review.md`. 
- **Selection:** 
  - **If starting fresh:** Choose one of the ten research ideas based on a value provided by a pseudo-random number generator.
  - **If pivoting:** Evaluate your 3 pivot strategies against your assigned persona, and select the strongest one to become your new `final_idea.md`.
- **Iterate:** Improve upon the novelty, feasibility, and importance of the proposed research idea by reconsidering prior work.
- **Handoff Artifact:** You MUST create a detailed `final_idea.md` outlining the exact architecture, baseline, and expected outcomes of your chosen project. **You MUST fill out `template/idea_template.md` for your chosen idea and save it to `final_idea.md` (overwriting the old one if pivoting). You may not proceed to Phase 2 until `final_idea.md` is populated with concrete mathematical formulations, specific baselines, and architectural specs.**
- **State Management:** When finished with Phase 1 and `final_idea.md` is written, update the `progress.json` file to set `{"phase": 2}`.

## Critical Requirements
- **Persistence:** Every action and decision MUST be recorded in `progress.md` (append-only log).
- **Format:** At the end of Phase 1, you MUST generate `final_idea.md` and set `{"phase": 2}` in `progress.json`.
