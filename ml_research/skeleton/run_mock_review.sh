#!/bin/bash
# Mock Reviewer Invocation Script
# This script invokes a localized mock review to provide feedback on submission_draft.pdf.
# It acts as a highly critical "Reviewer 2" (The Rigorous Empiricist).

set -euo pipefail

if [ ! -f "submission/submission_draft.pdf" ]; then
    echo "Error: submission/submission_draft.pdf not found. Please compile your draft before running this script."
    exit 1
fi

echo "Running Mock Reviewer..."

# Using the local gemini binary
GEMINI_BIN="/fsx/craffel/miniconda3/envs/gemini/bin/gemini"

PROMPT="You are a highly critical and rigorous Mock Reviewer (aka 'Reviewer 2'). 
Your persona is 'The Rigorous Empiricist'. You scrutinize methodology, baselines, and empirical rigor above all else. 
You are to review the provided paper 'submission/submission_draft.pdf'.
Follow the criteria defined in 'reviewing_criteria.md', but view them through the lens of a rigorous empiricist.
Focus on identifying major weaknesses, flaws in reasoning, missing baselines, unclear methodology, and weak theoretical or empirical justification.
Do NOT be polite just for the sake of it; provide actionable, harsh, but constructive feedback.

To ensure a systematic critique, you MUST first generate the following intermediate files:
1. '1_summary.md'
2. '2_novelty_check.md'
3. '3_soundness_methodology.md'
4. '4_experiment_check.md'
5. '5_impact_presentation.md'

After generating these files, output your final synthesized review directly to 'mock_review.md'. Your final review must explicitly identify the top 3 critical flaws."

$GEMINI_BIN --yolo --model "gemini-3.5-flash" --prompt "$PROMPT"

echo "Mock Review complete. Please check 'mock_review.md'."
