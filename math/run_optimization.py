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

EQUATIONS = {
    1: "(sin(pi*x/2) + 1)^(x + 1)^1.5",
    2: "exp((x + 1)^(3/2) ln(sin(pi*x/2) + 1))",
    3: "(sin(pi*x/4) + cos(pi*x/4))^(2*(x + 1)^(3/2))",
    4: "(2*cos^2(pi*(1 - x)/4))^((x + 1)*sqrt(x + 1))"
}

GENERATOR_PROMPT_TEMPLATE = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function {equation_text}.
Your goal is to minimize the mean squared error (MSE) of the linear approximation.
{past_iterates}
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose at this iteration."""

ITERATES_DESCRIPTION = """Below are the values found in past iterations.
{past_iterates}
Come up with a new set of values that improve upon past iterations."""

JUDGE_PROMPT_TEMPLATE = r"""You are working to iteratively find a line of best fit (y = mx + b) over the interval [0, 1] for the function {equation_text}.
Your goal is to minimize the mean squared error (MSE) of the linear approximation.
Below are {n_guesses} possible values for m and b, provided as tuples.
{guesses}
Choose the pair of values that provide the best fit.
Provide your final response in the format $\boxed{{(m, b)}}$, replacing m and b with the values you choose."""

APPROXIMATE_MSE_PROMPT_TEMPLATE = r"""If the function {equation_text} is approximated by the line y = mx + b with m = {m} and b = {b} over the interval [0, 1], what is the mean squared error (MSE) of this approximation?
Carefully derive an estimate of the MSE and provide your final approximated value in the format $\boxed{{val}}$, replacing val with your approximated MSE."""

TUPLE_EXTRACTION_PROMPT = """Below is a model response choosing or generating a linear fit (m, b) from a list of options or calculations.
Read the response and extract the chosen or generated pair as a tuple (m, b).

Instructions:
1. If the response contains a specific chosen or generated pair of values for m and b, output EXACTLY $\\boxed{{(m, b)}}$, replacing m and b with those exact values.
2. If the response does NOT contain any clear, specific chosen or generated values for m and b, output EXACTLY the word 'NONE'. Do not guess or hallucinate any values.

Model response:
{response_text}"""

SINGLE_EXTRACTION_PROMPT = """Below is a model response approximating the Mean Squared Error (MSE) of a line.
Read the response and extract the final approximated value.
Format the final response EXACTLY as $\\boxed{{val}}$, replacing val with the approximated value. If no approximation is present, output EXACTLY the word 'NONE'.

Model response:
{response_text}"""


def approximate_mse(m: float, b: float) -> float:
    """
    Approximates the Mean Squared Error (MSE) between the linear function
    y = mx + b and the target function over the interval [0, 1].

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


def extract_single_value(response: str) -> float | None:
    # Captures a single boxed float, supporting scientific notation and negative values
    pattern = r"\\boxed\s*\{\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\}"
    matches = re.findall(pattern, response)
    if matches:
        try:
            return float(matches[-1])
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
            logger.warning("Google GenAI Server Error encountered. Retrying in 1 second...")
            time.sleep(1)
        except genai.errors.APIError as e:
            # Code 429 indicates rate-limiting (Resource Exhausted)
            if e.code == 429:
                logger.warning("Google GenAI Rate limit reached (429). Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise e


def fallback_extract(client, response_text: str, prompt_template: str, parser):
    extraction_prompt = prompt_template.format(response_text=response_text)
    try:
        res = wrapped_generate(client, "gemini-3.5-flash", extraction_prompt)
        return parser(res.text)
    except Exception as e:
        logger.warning(f"Fallback extraction failed: {e}")
        return None


def generate_and_parse_with_fallback(
    client,
    model: str,
    prompt: str,
    primary_parser,
    fallback_parser=None,
    label: str = "generation"
):
    """
    Unified LLM prompting and extraction function.
    Handles rate-limiting/API retries internally, primary regex extraction,
    and optional LLM fallback extraction. If extraction completely fails,
    it retries the full LLM invocation loop.
    """
    while True:
        response = wrapped_generate(client, model, prompt)
        ans = primary_parser(response.text)
        if ans is not None:
            return ans
            
        if fallback_parser:
            logger.info(f"Attempting fallback extraction for {label} response...")
            ans = fallback_parser(client, response.text)
            if ans is not None:
                return ans
                
        logger.warning(f"Failed to extract a valid answer for {label}. Retrying full generation...")


def run_experiment(
    generator_model: str,
    judge_model: str,
    n_steps: int = 10,
    n_guesses: int = 10,
    max_past_iterates: str | int = "all",
    equation_option: int = 1,
    early_stopping_mse: float = 0.18,
    initial_results: dict = None,
    output_dir: str = None,
    generator_show_mse: bool = False,
    judge_show_mse: bool = False,
    generator_approximate_mse: bool = False,
    judge_approximate_mse: bool = False
) -> dict:
    client = genai.Client()

    equation_text = EQUATIONS[equation_option]

    # Log formatted templates once on startup
    logger.info(f"Generator Prompt Template:\n{GENERATOR_PROMPT_TEMPLATE.format(equation_text=equation_text, past_iterates='{past_iterates}')}\n")
    
    is_llm_judge = judge_model not in ["mse", "random"]
    if is_llm_judge:
        logger.info(f"Judge Prompt Template:\n{JUDGE_PROMPT_TEMPLATE.format(equation_text=equation_text, n_guesses='{n_guesses}', guesses='{guesses}')}\n")

    rounds_data = []
    consecutive_low_mse = 0

    if initial_results:
        rounds_data = list(initial_results.get("rounds", []))
        
        # Recalculate consecutive low MSE so far from resumed history
        for r_data in rounds_data:
            chosen_data = r_data["chosen"]
            if chosen_data.get("approx_mse", float("inf")) < early_stopping_mse:
                consecutive_low_mse += 1
            else:
                consecutive_low_mse = 0

    start_round = len(rounds_data)

    for i in range(start_round, n_steps):
        logger.info(f"--- Round {i + 1} ---")
        guesses = []
        
        # Slices the list of past completed rounds if configured
        visible_rounds = rounds_data[-max_past_iterates:] if max_past_iterates != "all" and max_past_iterates > 0 else rounds_data[:i]

        # Represent visible past iterates as a list of dictionaries, reading directly from history!
        visible_iterates_dicts = []
        for r_past in visible_rounds:
            chosen_past = r_past["chosen"]
            d = {"m": chosen_past["m"], "b": chosen_past["b"]}
            if generator_approximate_mse:
                d["mse"] = chosen_past.get("approximated_mse_by_generator")
            elif generator_show_mse:
                d["mse"] = chosen_past["approx_mse"]
            visible_iterates_dicts.append(d)

        # Format past iterates strings cleanly and dynamically
        past_iterates_strings = []
        for d in visible_iterates_dicts:
            if "mse" in d and d["mse"] is not None:
                past_iterates_strings.append(f"({d['m']}, {d['b']}) with MSE = {d['mse']:.6f}")
            else:
                past_iterates_strings.append(f"({d['m']}, {d['b']})")

        generator_prompt = GENERATOR_PROMPT_TEMPLATE.format(
            equation_text=equation_text,
            past_iterates=(
                ITERATES_DESCRIPTION.format(past_iterates="\n".join(past_iterates_strings)) if visible_rounds else ""
            )
        )
        for j in range(n_guesses):
            ans = generate_and_parse_with_fallback(
                client,
                model=generator_model,
                prompt=generator_prompt,
                primary_parser=extract_answer,
                fallback_parser=lambda c, r: fallback_extract(c, r, TUPLE_EXTRACTION_PROMPT, extract_answer),
                label=f"generator guess {j + 1}/{n_guesses}"
            )
            
            # Proactively approximate MSE for this generator guess as soon as it is generated
            guess_entry = {"m": ans[0], "b": ans[1]}
            if generator_approximate_mse:
                approx_val = generate_and_parse_with_fallback(
                    client,
                    model=generator_model,
                    prompt=APPROXIMATE_MSE_PROMPT_TEMPLATE.format(equation_text=equation_text, m=ans[0], b=ans[1]),
                    primary_parser=extract_single_value,
                    fallback_parser=lambda c, r: fallback_extract(c, r, SINGLE_EXTRACTION_PROMPT, extract_single_value),
                    label=f"generator MSE approximation for guess ({ans[0]}, {ans[1]})"
                )
                guess_entry["approximated_mse_by_generator"] = approx_val
            guesses.append(guess_entry)

        # Conditionally prompt the judge to approximate the MSE of each proposed guess
        if judge_approximate_mse and is_llm_judge:
            for d in guesses:
                approx_val = generate_and_parse_with_fallback(
                    client,
                    model=judge_model,
                    prompt=APPROXIMATE_MSE_PROMPT_TEMPLATE.format(equation_text=equation_text, m=d["m"], b=d["b"]),
                    primary_parser=extract_single_value,
                    fallback_parser=lambda c, r: fallback_extract(c, r, SINGLE_EXTRACTION_PROMPT, extract_single_value),
                    label=f"judge MSE approximation for proposed guess ({d['m']}, {d['b']})"
                )
                d["mse"] = approx_val
                # Save approximated MSE under custom key inside guesses for results.json backwards compatibility
                d["approximated_mse_by_judge"] = approx_val
        elif judge_show_mse:
            for d in guesses:
                d["mse"] = approximate_mse(d["m"], d["b"])

        # Format proposed guesses strings cleanly and dynamically
        guesses_strings = []
        for d in guesses:
            if "mse" in d and d["mse"] is not None:
                guesses_strings.append(f"({d['m']}, {d['b']}) with MSE = {d['mse']:.6f}")
            else:
                guesses_strings.append(f"({d['m']}, {d['b']})")

        if is_llm_judge:
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                equation_text=equation_text,
                n_guesses=n_guesses,
                guesses="\n".join(guesses_strings)
            )
            chosen = generate_and_parse_with_fallback(
                client,
                model=judge_model,
                prompt=judge_prompt,
                primary_parser=extract_answer,
                fallback_parser=lambda c, r: fallback_extract(c, r, TUPLE_EXTRACTION_PROMPT, extract_answer),
                label="judge final selection"
            )
            
            # Find chosen guess dictionary directly
            chosen_dict = None
            for d in guesses:
                if (d["m"], d["b"]) == chosen:
                    chosen_dict = dict(d)
                    break
            if chosen_dict is None:
                chosen_dict = {"m": chosen[0], "b": chosen[1]}
        elif judge_model == "mse":
            chosen_dict = dict(min(guesses, key=lambda g: approximate_mse(g["m"], g["b"])))
        elif judge_model == "random":
            chosen_dict = dict(random.choice(guesses))

        mse = approximate_mse(chosen_dict["m"], chosen_dict["b"])
        chosen_dict["approx_mse"] = mse
        
        # Remove temporary 'mse' key from the selection dict before round logging/saving
        chosen_dict.pop("mse", None)

        round_record = {
            "round": i + 1,
            "guesses": guesses,
            "chosen": chosen_dict
        }

        rounds_data.append(round_record)

        logger.info(f"Round {i + 1} completed.")

        # Check early stopping condition
        if mse < early_stopping_mse:
            consecutive_low_mse += 1
        else:
            consecutive_low_mse = 0

        # Progressively write results.json after each completed round
        if output_dir:
            progressive_results = {
                "rounds": rounds_data,
                "final_iterates": [
                    {
                        "step": r["round"],
                        "m": r["chosen"]["m"],
                        "b": r["chosen"]["b"],
                        "approx_mse": r["chosen"]["approx_mse"]
                    }
                    for r in rounds_data
                ]
            }
            results_path = os.path.join(output_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(progressive_results, f, indent=4)

        if consecutive_low_mse >= 3:
            logger.info(f"Early stopping triggered: attained MSE < {early_stopping_mse} for {consecutive_low_mse} consecutive rounds.")
            break

    # Build final_iterates on the fly from rounds_data dynamically to ensure zero-duplication in memory
    final_iterates_json = [
        {
            "step": r["round"],
            "m": r["chosen"]["m"],
            "b": r["chosen"]["b"],
            "approx_mse": r["chosen"]["approx_mse"]
        }
        for r in rounds_data
    ]

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
        default=20,
        help="The number of optimization steps to run (default: 10).",
    )
    parser.add_argument(
        "--n_guesses", "--n-guesses",
        type=int,
        default=5,
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
        "--early_stopping_mse", "--early-stopping-mse",
        type=float,
        default=0.18,
        help="The MSE threshold for early stopping. If MSE is below this value for 3 consecutive rounds, the experiment stops (default: 0.18).",
    )
    parser.add_argument(
        "--generator_show_mse", "--generator-show-mse",
        action="store_true",
        help="Provide the generator model with the MSE of every past iterate in its prompt.",
    )
    parser.add_argument(
        "--judge_show_mse", "--judge-show-mse",
        action="store_true",
        help="Provide the judge model with the MSE of every proposed guess in its prompt.",
    )
    parser.add_argument(
        "--generator_approximate_mse", "--generator-approximate-mse",
        action="store_true",
        help="Ask the generator model to independently approximate the MSE of past iterates.",
    )
    parser.add_argument(
        "--judge_approximate_mse", "--judge-approximate-mse",
        action="store_true",
        help="Ask the judge model to independently approximate the MSE of proposed guesses.",
    )
    parser.add_argument(
        "--output_dir", "--output-dir",
        type=str,
        required=True,
        help="The directory path where results.json, config.json, and the log file will be saved.",
    )

    args = parser.parse_args()

    # Fail fast if GEMINI_API_KEY is not set
    if "GEMINI_API_KEY" not in os.environ:
        logger.error("Error: GEMINI_API_KEY environment variable is not set. Please set it before running the script.")
        sys.exit(1)

    # Check for existing partial results.json to support resuming
    results_path = os.path.join(args.output_dir, "results.json")
    initial_results = None
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                initial_results = json.load(f)
            completed_rounds = len(initial_results.get("rounds", []))
            if completed_rounds >= args.n_steps:
                logger.info(f"Experiment in {args.output_dir} is already completed ({completed_rounds}/{args.n_steps} rounds). Exiting.")
                sys.exit(0)
            elif completed_rounds > 0:
                logger.info(f"Resuming experiment in {args.output_dir} from round {completed_rounds + 1} ({completed_rounds}/{args.n_steps} rounds already completed).")
        except Exception as e:
            logger.warning(f"Could not parse existing results.json, starting from scratch. Error: {e}")

    # Create output directory if starting fresh
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure isolated local logging FileHandler to save/append a transcript of the log by default
    log_file_path = os.path.join(args.output_dir, "optimization.log")
    file_handler = logging.FileHandler(log_file_path, mode="a") # Use append mode to preserve logs on resume
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
                f"  Equation Option: {args.equation_option} ({EQUATIONS[args.equation_option]})\n"
                f"  Early Stopping MSE: {args.early_stopping_mse}\n"
                f"  Generator Show MSE: {args.generator_show_mse}\n"
                f"  Judge Show MSE: {args.judge_show_mse}\n"
                f"  Generator Approx MSE: {args.generator_approximate_mse}\n"
                f"  Judge Approx MSE: {args.judge_approximate_mse}\n"
                f"  Output Directory: {args.output_dir}")

    results = run_experiment(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        n_steps=args.n_steps,
        n_guesses=args.n_guesses,
        max_past_iterates=args.max_past_iterates,
        equation_option=args.equation_option,
        early_stopping_mse=args.early_stopping_mse,
        initial_results=initial_results,
        output_dir=args.output_dir,
        generator_show_mse=args.generator_show_mse,
        judge_show_mse=args.judge_show_mse,
        generator_approximate_mse=args.generator_approximate_mse,
        judge_approximate_mse=args.judge_approximate_mse
    )

    # Save final results to results.json (completely validated and outputted)
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Experiment complete. Results saved to {args.output_dir}")
