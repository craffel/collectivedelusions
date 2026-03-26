import os
import sys
import random
import pathlib
import shutil
import time
import subprocess
import argparse
from io import BytesIO
from PIL import Image
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: 'google-genai' library not found. Please install it using: pip install google-genai pillow")
    sys.exit(1)

import generate_readme

# Configuration
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: Neither GOOGLE_API_KEY nor GEMINI_API_KEY environment variable is set.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3-flash-preview" # Updated from 2.0 to 3.0 flash preview

BASE_DIR = None
GEN_PROMPT_FILE = None
JUDGE_PROMPT_FILE = None
ORIGINAL_WINNERS_DIR = None

def setup_directories(base_dir_path):
    global BASE_DIR, GEN_PROMPT_FILE, JUDGE_PROMPT_FILE, ORIGINAL_WINNERS_DIR
    BASE_DIR = pathlib.Path(base_dir_path).resolve()
    GEN_PROMPT_FILE = BASE_DIR / "generation_prompt.md"
    JUDGE_PROMPT_FILE = BASE_DIR / "judge_prompt.md"
    ORIGINAL_WINNERS_DIR = BASE_DIR / "original_winners"

def get_images_from_folder(folder):
    return sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))

def generate_round_submissions(round_n):
    submission_dir = BASE_DIR / f"submissions_round_{round_n}"
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    if round_n == 1:
        # In round 1, pick 3 random images from original_winners
        all_prev_winners = get_images_from_folder(ORIGINAL_WINNERS_DIR)
        prev_winners = random.sample(all_prev_winners, min(3, len(all_prev_winners)))
        print(f"Round 1: Randomly selected {len(prev_winners)} seeds from original_winners.")
        
        # Save seeds for README visualization
        winners_0_dir = BASE_DIR / "winners_round_0"
        winners_0_dir.mkdir(parents=True, exist_ok=True)
        for w in prev_winners:
            shutil.copy(w, winners_0_dir / w.name)
    else:
        # In subsequent rounds, pick from the previous round's winners
        prev_winners_dir = BASE_DIR / f"winners_round_{round_n - 1}"
        prev_winners = get_images_from_folder(prev_winners_dir)[:3]
    
    with open(GEN_PROMPT_FILE, "r") as f:
        full_prompt = f.read()
    
    desc_file = BASE_DIR / "contest_description.md"
    if desc_file.exists():
        with open(desc_file, "r") as f:
            full_prompt += "\n\n" + f.read()
    
    print(f"--- Round {round_n}: Generating 10 submissions ---")
    
    # Load previous winner images for multimodal context
    image_parts = []
    for img_path in prev_winners:
        img = Image.open(img_path)
        image_parts.append(img)

    for i in range(1, 11):
        print(f"Generating image {i}/10...", end="\r")
        try:
            # Using generate_content for native image generation with gemini-3.1-flash-image-preview
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[full_prompt] + image_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                )
            )
            
            found_image = False
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    img = Image.open(BytesIO(image_bytes))
                    img.save(submission_dir / f"submission_{i}.jpg")
                    found_image = True
                    break
            
            if not found_image:
                print(f"\nWarning: No image generated for submission {i}. Response: {response.text}")
            
            # Rate limit backoff
            time.sleep(1)
        except Exception as e:
            print(f"\nError generating image {i}: {e}")
            if "429" in str(e):
                print("Rate limit reached. Sleeping for 5 seconds...")
                time.sleep(5)
    print("\nGeneration complete.")

def judge_round_winners(round_n):
    submission_dir = BASE_DIR / f"submissions_round_{round_n}"
    winners_dir = BASE_DIR / f"winners_round_{round_n}"
    winners_dir.mkdir(parents=True, exist_ok=True)
    
    submissions = get_images_from_folder(submission_dir)
    if not submissions:
        print(f"Error: No submissions found in {submission_dir}")
        return

    with open(JUDGE_PROMPT_FILE, "r") as f:
        judge_prompt = f.read()
        
    desc_file = BASE_DIR / "contest_description.md"
    if desc_file.exists():
        with open(desc_file, "r") as f:
            judge_prompt += "\n\n" + f.read()
    
    # Add explicit instructions for parsing
    parse_instruction = "\n\nPlease identify exactly three winners. List only the numbers of the winners as a comma-separated list, e.g., 'Winners: 1, 4, 7'."
    
    content_parts = [judge_prompt + parse_instruction]
    for i, img_path in enumerate(submissions):
        img = Image.open(img_path)
        content_parts.append(f"Submission {i+1}:")
        content_parts.append(img)

    print(f"--- Round {round_n}: Judging winners ---")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content_parts
        )
        
        response_text = response.text
        print(f"Judge response: {response_text}")
        
        # Simple parsing for numbers
        import re
        numbers = re.findall(r'\b(10|[1-9])\b', response_text)
        # Try to find unique numbers in case Gemini repeated them
        unique_winners = []
        for n in numbers:
            if n not in unique_winners:
                unique_winners.append(n)
        
        selected_indices = [int(n) for n in unique_winners[:3]]
        
        if len(selected_indices) < 3:
            print("Warning: Judge did not return 3 clear winners. Picking first 3 as fallback.")
            # Fallback to the first available if not enough winners were chosen
            selected_indices += [i for i in [1, 2, 3] if i not in selected_indices]
            selected_indices = selected_indices[:3]

        for idx in selected_indices:
            if 1 <= idx <= len(submissions):
                src = submissions[idx-1]
                shutil.copy(src, winners_dir / src.name)
                print(f"Winner selected: {src.name}")
            else:
                print(f"Warning: Judge picked invalid index {idx}")

    except Exception as e:
        print(f"Error during judging: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run the image generation contest.")
    parser.add_argument("base_dir", type=str, help="The base images directory (e.g., images_v2)")
    parser.add_argument("num_rounds", type=int, nargs="?", default=1, help="Number of rounds to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initial image selection")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    setup_directories(args.base_dir)

    start_round = 1
    while True:
        winners_dir = BASE_DIR / f"winners_round_{start_round}"
        if winners_dir.exists() and len(get_images_from_folder(winners_dir)) >= 3:
            start_round += 1
        else:
            break

    if start_round <= args.num_rounds:
        print(f"Resuming from round {start_round}...")
        partial_sub_dir = BASE_DIR / f"submissions_round_{start_round}"
        if partial_sub_dir.exists():
            shutil.rmtree(partial_sub_dir)
        partial_win_dir = BASE_DIR / f"winners_round_{start_round}"
        if partial_win_dir.exists():
            shutil.rmtree(partial_win_dir)

    for n in range(start_round, args.num_rounds + 1):
        print(f"\\n=== STARTING ROUND {n} ===")
        generate_round_submissions(n)
        judge_round_winners(n)
        print(f"=== COMPLETED ROUND {n} ===")
        
        # Generate README and push to Git
        print(f"Updating README and pushing to GitHub for Round {n}...")
        try:
            generate_readme.generate_readme(args.base_dir)
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Add results for Round {n}"], check=True, capture_output=True)
            subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
            print("Successfully pushed to GitHub!")
        except subprocess.CalledProcessError as e:
            print(f"Git push failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        except Exception as e:
            print(f"Failed to update README or push: {e}")
            
        print("\\n")

if __name__ == "__main__":
    main()
