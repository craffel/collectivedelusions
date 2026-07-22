#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-genai",
#   "numpy",
#   "scipy",
# ]
# ///

import logging
import os
import re
import sys
import time

from google import genai
import numpy as np
import scipy.integrate

# Configure isolated local logging to completely suppress third-party package noise
logger = logging.getLogger("optimization")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Initialize client lazily or if API key is present, to prevent crash on --help
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

GENERATOR_PROMPT_TEMPLATE = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function (sin(pi*x/2) + 1)^(x + 1)^1.5.
{past_iterates}
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose at this iteration."""

ITERATES_DESCRIPTION = """Below are the values found in past iterations.
{past_iterates}
Come up with a new set of values that improve upon past iterations."""

JUDGE_PROMPT_TEMPLATE = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function (sin(pi*x/2) + 1)^(x + 1)^1.5.
Below are {n_guesses} possible values for m and b, provided as tuples.
{guesses}
Choose the pair of values that provide the best fit.
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose."""


def approximate_mse(m: float, b: float) -> float:
    """
    Approximates the Mean Squared Error (MSE) between the linear function
    y = mx + b and the target function y = (sin(pi*x/2) + 1)^((x + 1)^1.5)
    over the interval [0, 1].

    Parameters:
    m (float): The slope of the linear model.
    b (float): The y-intercept of the linear model.

    Returns:
    float: The approximated MSE.
    """
    def target_function(x):
        return (np.sin(np.pi * x / 2) + 1) ** ((x + 1) ** 1.5)

    def squared_error(x):
        return (m * x + b - target_function(x)) ** 2

    return scipy.integrate.quad(squared_error, 0, 1)[0]


def extract_answer(response: str) -> tuple[float, float] | None:
    # Captures:
    # - Optional negative/positive signs (- or +)
    # - Integers and floats (including leading dot floats like -.5)
    # - Scientific notation (e.g., 1.2e-3)
    # - Robust/optional whitespace inside \boxed{...} and (...)
    # - Optional LaTeX '$' surrounding characters
    pattern = r"\\boxed\s*\{\s*\(\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*,\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\)\s*\}"
    
    matches = re.findall(pattern, response)
    if matches:
        # Extract the last match, which is typically the final answer
        str_tuple = matches[-1]
        try:
            return float(str_tuple[0]), float(str_tuple[1])
        except ValueError:
            return None
    return None


def wrapped_generate(client, model, prompt):
    while True:
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
            )
        except genai.errors.ServerError:
            time.sleep(1)


def run_experiment(generator_model: str, judge_model: str, n_steps: int=10, n_guesses=10, max_past_iterates="all"):
    global client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set to run the experiment.")
        client = genai.Client(api_key=api_key)

    logger.info(f"Generator Prompt Template:\n{GENERATOR_PROMPT_TEMPLATE}\n")
    logger.info(f"Judge Prompt Template:\n{JUDGE_PROMPT_TEMPLATE}\n")

    iterates = []
    for i in range(n_steps):
        logger.info(f"--- Round {i + 1} ---")
        guesses = []
        
        # Slices the list of past iterates if configured
        if max_past_iterates == "all":
            visible_iterates = iterates
        else:
            visible_iterates = iterates[-max_past_iterates:] if max_past_iterates > 0 else []

        generator_prompt = GENERATOR_PROMPT_TEMPLATE.format(
            past_iterates=(
                ITERATES_DESCRIPTION.format(past_iterates="\n".join(str(t) for t in visible_iterates)) if visible_iterates else ""
            )
        )
        for j in range(n_guesses):
            while True:
                response = wrapped_generate(client, generator_model, generator_prompt)
                ans = extract_answer(response.text)
                if ans is not None:
                    guesses.append(ans)
                    break
                logger.warning(f"No match found in generator response (guess {j + 1}/{n_guesses}). Retrying generation...")
        
        logger.info(f"Generated guesses for Round {i + 1}: {guesses}")
        
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            n_guesses=n_guesses,
            guesses="\n".join(str(g) for g in guesses)
        )
        response = wrapped_generate(client, judge_model, judge_prompt)
        chosen = extract_answer(response.text)
        logger.info(f"Judge chose for Round {i + 1}: {chosen}")
        iterates.append(chosen)
    return iterates


def parse_max_past_iterates(value):
    if value.lower() == "all":
        return "all"
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError()
        return ivalue
    except ValueError:
        raise ValueError(f"Invalid value: {value}. Must be 'all' or a non-negative integer.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM-based optimization of a linear fit of a target function."
    )
    parser.add_argument(
        "--generator_model", "--generator-model",
        type=str,
        required=True,
        help="The Gemini model to use for generating guesses.",
    )
    parser.add_argument(
        "--judge_model", "--judge-model",
        type=str,
        required=True,
        help="The Gemini model to use for judging guesses.",
    )
    parser.add_argument(
        "--n_steps", "--n-steps",
        type=int,
        default=10,
        help="The number of optimization steps to run (default: 10).",
    )
    parser.add_argument(
        "--n_guesses", "--n-guesses",
        type=int,
        default=10,
        help="The number of guesses to generate per step (default: 10).",
    )
    parser.add_argument(
        "--max_past_iterates", "--max-past-iterates",
        type=parse_max_past_iterates,
        default="all",
        help="The maximum number of past iterates sent to the generator. Can be 'all' or a non-negative integer (default: 'all').",
    )

    args = parser.parse_args()

    logger.info(f"Starting experiment with:\n"
                f"  Generator Model: {args.generator_model}\n"
                f"  Judge Model: {args.judge_model}\n"
                f"  Steps: {args.n_steps}\n"
                f"  Guesses per step: {args.n_guesses}\n"
                f"  Max past iterates: {args.max_past_iterates}")

    results = run_experiment(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        n_steps=args.n_steps,
        n_guesses=args.n_guesses,
        max_past_iterates=args.max_past_iterates,
    )

    logger.info("Experiment complete. Optimization iterates:")
    for idx, iterate in enumerate(results):
        if iterate is not None:
            m, b = iterate
            mse = approximate_mse(m, b)
            logger.info(f"Step {idx + 1:02d}: m = {m:.6f}, b = {b:.6f} | Approx MSE = {mse:.6f}")
        else:
            logger.error(f"Step {idx + 1:02d}: Failed to extract values from model response.")
