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
import random
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

EQUATIONS = {
    1: {
        "text": "(sin(pi*x/2) + 1)^(x + 1)^1.5",
        "func": lambda x: (np.sin(np.pi * x / 2) + 1) ** ((x + 1) ** 1.5)
    },
    2: {
        "text": "exp((x + 1)^(3/2) ln(sin(pi*x/2) + 1))",
        "func": lambda x: np.exp(((x + 1) ** 1.5) * np.log(np.sin(np.pi * x / 2) + 1))
    },
    3: {
        "text": "(sin(pi*x/4) + cos(pi*x/4))^(2*(x + 1)^(3/2))",
        "func": lambda x: (np.sin(np.pi * x / 4) + np.cos(np.pi * x / 4)) ** (2 * (x + 1) ** 1.5)
    },
    4: {
        "text": "(2*cos^2(pi*(1 - x)/4))^((x + 1)*sqrt(x + 1))",
        "func": lambda x: (2 * (np.cos(np.pi * (1 - x) / 4) ** 2)) ** ((x + 1) * np.sqrt(x + 1))
    }
}

GENERATOR_PROMPT_TEMPLATE = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function {equation_text}.
{past_iterates}
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose at this iteration."""

ITERATES_DESCRIPTION = """Below are the values found in past iterations.
{past_iterates}
Come up with a new set of values that improve upon past iterations."""

JUDGE_PROMPT_TEMPLATE = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function {equation_text}.
Below are {n_guesses} possible values for m and b, provided as tuples.
{guesses}
{judge_instruction}
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose."""


def approximate_mse(m: float, b: float, equation_option: int = 1) -> float:
    """
    Approximates the Mean Squared Error (MSE) between the linear function
    y = mx + b and the target function over the interval [0, 1].

    Parameters:
    m (float): The slope of the linear model.
    b (float): The y-intercept of the linear model.
    equation_option (int): The equation option ID (1 to 4).

    Returns:
    float: The approximated MSE.
    """
    target_function = EQUATIONS[equation_option]["func"]

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


def run_experiment(generator_model: str, judge_model: str, n_steps: int=10, n_guesses=10, max_past_iterates="all", equation_option: int = 1, judge_approximate_mse: bool = False):
    global client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set to run the experiment.")
        client = genai.Client(api_key=api_key)

    equation_text = EQUATIONS[equation_option]["text"]
    
    if judge_approximate_mse:
        judge_instruction = "Choose the pair of values that achieves the lowest mean-squared error. You should explicitly approximate the mean-squared error for each option."
    else:
        judge_instruction = "Choose the pair of values that provide the best fit."

    # Log formatted templates once on startup
    logger.info(f"Generator Prompt Template:\n{GENERATOR_PROMPT_TEMPLATE.format(equation_text=equation_text, past_iterates='{past_iterates}')}\n")
    
    is_llm_judge = judge_model not in ["mse", "random"]
    if is_llm_judge:
        logger.info(f"Judge Prompt Template:\n{JUDGE_PROMPT_TEMPLATE.format(equation_text=equation_text, n_guesses='{n_guesses}', guesses='{guesses}', judge_instruction=judge_instruction)}\n")

    iterates = []
    rounds_data = []

    for i in range(n_steps):
        logger.info(f"--- Round {i + 1} ---")
        guesses = []
        
        # Slices the list of past iterates if configured
        if max_past_iterates == "all":
            visible_iterates = iterates
        else:
            visible_iterates = iterates[-max_past_iterates:] if max_past_iterates > 0 else []

        generator_prompt = GENERATOR_PROMPT_TEMPLATE.format(
            equation_text=equation_text,
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
        
        # Convert guesses list to JSON-serializable list
        round_guesses_json = []
        for g in guesses:
            if g is not None:
                round_guesses_json.append({"m": g[0], "b": g[1]})
            else:
                round_guesses_json.append(None)
        
        if is_llm_judge:
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                equation_text=equation_text,
                n_guesses=n_guesses,
                guesses="\n".join(str(g) for g in guesses),
                judge_instruction=judge_instruction
            )
            response = wrapped_generate(client, judge_model, judge_prompt)
            chosen = extract_answer(response.text)
        elif judge_model == "mse":
            valid_guesses = [g for g in guesses if g is not None]
            if valid_guesses:
                chosen = min(valid_guesses, key=lambda g: approximate_mse(g[0], g[1], equation_option=equation_option))
            else:
                chosen = None
        elif judge_model == "random":
            valid_guesses = [g for g in guesses if g is not None]
            chosen = random.choice(valid_guesses) if valid_guesses else None

        if chosen is not None:
            mse = approximate_mse(chosen[0], chosen[1], equation_option=equation_option)
            chosen_json = {"m": chosen[0], "b": chosen[1], "approx_mse": mse}
        else:
            chosen_json = None

        rounds_data.append({
            "round": i + 1,
            "guesses": round_guesses_json,
            "chosen": chosen_json
        })

        iterates.append(chosen)
        logger.info(f"Round {i + 1} completed.")

    # Format final iterates data for JSON serialization
    final_iterates_json = []
    for idx, iterate in enumerate(iterates):
        if iterate is not None:
            mse = approximate_mse(iterate[0], iterate[1], equation_option=equation_option)
            final_iterates_json.append({
                "step": idx + 1,
                "m": iterate[0],
                "b": iterate[1],
                "approx_mse": mse
            })
        else:
            final_iterates_json.append({
                "step": idx + 1,
                "error": "Failed to extract values"
            })

    return {
        "rounds": rounds_data,
        "final_iterates": final_iterates_json
    }


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
    import json

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
        help="The Gemini model to use for judging guesses, or 'mse' to choose the guess with the lowest MSE directly, or 'random' to choose a random guess.",
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
    parser.add_argument(
        "--equation_option", "--equation-option",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="The mathematical equation option to optimize (1, 2, 3, or 4) (default: 1).",
    )
    parser.add_argument(
        "--judge_approximate_mse", "--judge-approximate-mse",
        action="store_true",
        help="Ask the judge model to explicitly approximate and choose the lowest MSE.",
    )
    parser.add_argument(
        "--output_dir", "--output-dir",
        type=str,
        required=True,
        help="The directory path where results.json, config.json, and the log file will be saved.",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure isolated local logging FileHandler to save a transcript of the log by default
    log_file_path = os.path.join(args.output_dir, "optimization.log")
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Save parsed configuration to config.json
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f"Starting experiment with:\n"
                f"  Generator Model: {args.generator_model}\n"
                f"  Judge Model: {args.judge_model}\n"
                f"  Steps: {args.n_steps}\n"
                f"  Guesses per step: {args.n_guesses}\n"
                f"  Max past iterates: {args.max_past_iterates}\n"
                f"  Equation Option: {args.equation_option} ({EQUATIONS[args.equation_option]['text']})\n"
                f"  Judge Approx MSE: {args.judge_approximate_mse}\n"
                f"  Output Directory: {args.output_dir}")

    results = run_experiment(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        n_steps=args.n_steps,
        n_guesses=args.n_guesses,
        max_past_iterates=args.max_past_iterates,
        equation_option=args.equation_option,
        judge_approximate_mse=args.judge_approximate_mse,
    )

    # Save results to results.json
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Experiment complete. Results saved to {args.output_dir}")
