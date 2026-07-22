#!/usr/bin/env python3
# /// script
# dependencies = [
#   "google-genai",
#   "numpy",
#   "scipy",
# ]
# ///

import os
import re
import time

from google import genai
import numpy as np
import scipy.integrate

# Initialize client lazily or if API key is present, to prevent crash on --help
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


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
    pattern = r"\$\\boxed\{\(([0-9]+\.?[0-9]*),\s*([0-9]+\.?[0-9]*)\)\}\$"
    matches = re.findall(pattern, response)
    if len(matches) == 1:
        str_tuple = matches[0]
        return float(str_tuple[0]), float(str_tuple[1])
    else:
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


def run_experiment(generator_model: str, judge_model: str, n_steps: int=10, n_guesses=10):
    global client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set to run the experiment.")
        client = genai.Client(api_key=api_key)

    generator_prompt_template = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function (sin(pi*x/2) + 1)^(x + 1)^1.5.
{past_iterates}
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose at this iteration."""
    iterates_description = """Below are the values found in past iterations.
{past_iterates}
Come up with a new set of values that improve upon past iterations."""
    judge_prompt_template = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function (sin(pi*x/2) + 1)^(x + 1)^1.5.
Below are {n_guesses} possible values for m and b, provided as tuples.
{guesses}
Choose the pair of values that provide the best fit.
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose."""

    iterates = []
    for i in range(n_steps):
        guesses = []
        generator_prompt = generator_prompt_template.format(
            past_iterates=(
                iterates_description.format(past_iterates="\n".join(str(t) for t in iterates)) if iterates else ""
            )
        )
        print(generator_prompt)
        for j in range(n_guesses):
            response = wrapped_generate(client, generator_model, generator_prompt)
            guesses.append(extract_answer(response.text))
        judge_prompt = judge_prompt_template.format(
            n_guesses=n_guesses,
            guesses="\n".join(str(g) for g in guesses)
        )
        print(judge_prompt)
        response = wrapped_generate(client, judge_model, judge_prompt)
        iterates.append(extract_answer(response.text))
    return iterates


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

    args = parser.parse_args()

    print(f"Starting experiment with:\n"
          f"  Generator Model: {args.generator_model}\n"
          f"  Judge Model: {args.judge_model}\n"
          f"  Steps: {args.n_steps}\n"
          f"  Guesses per step: {args.n_guesses}\n")

    results = run_experiment(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        n_steps=args.n_steps,
        n_guesses=args.n_guesses,
    )

    print("\nExperiment complete. Optimization iterates:")
    for idx, iterate in enumerate(results):
        if iterate is not None:
            m, b = iterate
            mse = approximate_mse(m, b)
            print(f"Step {idx + 1:02d}: m = {m:.6f}, b = {b:.6f} | Approx MSE = {mse:.6f}")
        else:
            print(f"Step {idx + 1:02d}: Failed to extract values from model response.")
