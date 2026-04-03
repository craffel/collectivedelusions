import os
import sys
import random
import re
import time
import argparse
from google import genai

def extract_svg(text):
    # Try to find content between <svg and </svg>
    match = re.search(r'<svg.*?</svg>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate and iteratively improve SVGs using Gemini.")
    parser.add_argument("directory", help="Target directory to save SVGs")
    parser.add_argument("iterations_n", type=int, help="Number of iterations (n)")
    parser.add_argument("improvements_m", type=int, help="Number of improvements per iteration (m)")
    parser.add_argument("--prompts", default="/fsx/craffel/collectivedelusions/svg/prompts.md", help="Path to the prompts file (default: /fsx/craffel/collectivedelusions/svg/prompts.md)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Gemini model to use (default: gemini-3.1-pro-preview)")
    
    args = parser.parse_args()

    target_dir = args.directory
    n = args.iterations_n
    m = args.improvements_m
    prompts_file = args.prompts
    model_name = args.model

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    # Use the new SDK client
    client = genai.Client(api_key=api_key)

    # 1) Pick random prompt from prompts.md
    if not os.path.exists(prompts_file):
        print(f"Prompts file not found at {prompts_file}")
        sys.exit(1)

    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        print(f"No prompts found in {prompts_file}")
        sys.exit(1)

    prompt = random.choice(prompts)
    print(f"Selected prompt: {prompt}")

    # Initial generation: save as 0.svg
    print("Generating initial SVG (0.svg)...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=f"Generate an SVG of: {prompt}"
        )
        svg_content = extract_svg(response.text)
        if not svg_content:
            print("Failed to extract SVG from initial response. Using raw response.")
            svg_content = response.text
        
        with open(os.path.join(target_dir, "0.svg"), "w") as f:
            f.write(svg_content)
    except Exception as e:
        print(f"Error during initial generation: {e}")
        sys.exit(1)
    
    # Iterative improvement
    for k in range(1, n + 1):
        print(f"\nIteration {k}/{n}...")
        prev_svg_path = os.path.join(target_dir, f"{k-1}.svg")
        with open(prev_svg_path, "r") as f:
            prev_svg_content = f.read()
        
        improved_svgs = []
        for i in range(1, m + 1):
            print(f"  Generating improvement {i}/{m}...")
            improve_prompt = f"Attached is an SVG of {prompt}. Please improve it.\n\n{prev_svg_content}"
            try:
                # Adding a small sleep to avoid hitting rate limits too hard if applicable
                time.sleep(1)
                response = client.models.generate_content(
                    model=model_name,
                    contents=improve_prompt
                )
                svg_improved = extract_svg(response.text)
                if not svg_improved:
                    svg_improved = response.text
                
                improved_svgs.append(svg_improved)
                with open(os.path.join(target_dir, f"{k-1}-{i}.svg"), "w") as f:
                    f.write(svg_improved)
            except Exception as e:
                print(f"    Error during improvement {i}: {e}")
        
        if not improved_svgs:
            print(f"No improved SVGs generated in iteration {k}. Stopping.")
            break
            
        # Pick favorite
        print(f"  Picking favorite for iteration {k}...")
        pick_prompt = f"Attached are different SVGs of {prompt}. Pick your favorite. Please output the full SVG code of your favorite.\n\n"
        for i, svg in enumerate(improved_svgs):
            pick_prompt += f"SVG {i+1}:\n{svg}\n\n"
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=pick_prompt
            )
            chosen_svg = extract_svg(response.text)
            
            if not chosen_svg:
                # Try to see if it mentioned a number
                match = re.search(r'SVG (\d+)', response.text, re.IGNORECASE)
                if match:
                    idx = int(match.group(1)) - 1
                    if 0 <= idx < len(improved_svgs):
                        chosen_svg = improved_svgs[idx]
                
                if not chosen_svg:
                    print("    Could not determine favorite, defaulting to first improvement.")
                    chosen_svg = improved_svgs[0]
            
            with open(os.path.join(target_dir, f"{k}.svg"), "w") as f:
                f.write(chosen_svg)
        except Exception as e:
            print(f"    Error picking favorite: {e}")
            with open(os.path.join(target_dir, f"{k}.svg"), "w") as f:
                f.write(improved_svgs[0])

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
