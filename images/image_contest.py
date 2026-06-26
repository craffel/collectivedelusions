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
MODEL_NAME = "gemini-3-flash-preview" 

class ImageGenerator:
    def generate(self, full_prompt: str, seed_images: list[pathlib.Path], output_path: pathlib.Path, attempt: int = 0):
        raise NotImplementedError

class GeminiImageGenerator(ImageGenerator):
    def generate(self, full_prompt: str, seed_images: list[pathlib.Path], output_path: pathlib.Path, attempt: int = 0):
        image_parts = []
        for img_path in seed_images:
            img = Image.open(img_path)
            image_parts.append(img)
            
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[full_prompt] + image_parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    img = Image.open(BytesIO(image_bytes))
                    img.save(output_path)
                    return
                    
        raise Exception("No image generated in response")

class FluxImageGenerator(ImageGenerator):
    def __init__(self):
        print("Loading Flux 2 dev pipeline...")
        import torch
        from diffusers import Flux2Pipeline
        self.pipe = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        print("Flux 2 dev pipeline loaded.")

    def generate(self, full_prompt: str, seed_images: list[pathlib.Path], output_path: pathlib.Path, attempt: int = 0):
        import torch
        
        # Load and combine the seed images into a single horizontal composite image
        opened_images = [Image.open(img_path).convert("RGB") for img_path in seed_images]
        
        target_size = 512
        resized_images = [img.resize((target_size, target_size)) for img in opened_images]
        
        total_w = target_size * len(resized_images)
        grid_image = Image.new('RGB', (total_w, target_size))
        
        x_offset = 0
        for img in resized_images:
            grid_image.paste(img, (x_offset, 0))
            x_offset += target_size
            
        print(f"  Attempt {attempt+1} feeding prompt and composite image directly to Flux...")
        
        image = self.pipe(
            prompt=full_prompt,
            image=grid_image,
            guidance_scale=3.5,
            num_inference_steps=28,
            height=target_size,
            width=total_w,
        ).images[0]
        
        image.save(output_path)
        torch.cuda.empty_cache()

class ImageJudge:
    def judge(self, submissions: list[pathlib.Path], judge_prompt: str, winners_dir: pathlib.Path):
        raise NotImplementedError

class GeminiImageJudge(ImageJudge):
    def judge(self, submissions: list[pathlib.Path], judge_prompt: str, winners_dir: pathlib.Path):
        parse_instruction = "\n\nPlease identify exactly three winners. List only the numbers of the winners as a comma-separated list, e.g., 'Winners: 1, 4, 7'."
        
        content_parts = [judge_prompt + parse_instruction]
        for i, img_path in enumerate(submissions):
            img = Image.open(img_path)
            content_parts.append(f"Submission {i+1}:")
            content_parts.append(img)
            
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content_parts
        )
        
        response_text = response.text
        print(f"Judge response: {response_text}")
        
        import re
        numbers = re.findall(r'\b(10|[1-9])\b', response_text)
        unique_winners = []
        for n in numbers:
            if n not in unique_winners:
                unique_winners.append(n)
        
        selected_indices = [int(n) for n in unique_winners[:3]]
        
        if len(selected_indices) < 3:
            print("Warning: Judge did not return 3 clear winners. Picking first 3 as fallback.")
            selected_indices += [i for i in [1, 2, 3] if i not in selected_indices]
            selected_indices = selected_indices[:3]

        for idx in selected_indices:
            if 1 <= idx <= len(submissions):
                src = submissions[idx-1]
                shutil.copy(src, winners_dir / src.name)
                print(f"Winner selected: {src.name}")
            else:
                print(f"Warning: Judge picked invalid index {idx}")

class ContestRunner:
    def __init__(self, base_dir_path: str, generator: ImageGenerator, judge: ImageJudge):
        self.base_dir = pathlib.Path(base_dir_path).resolve()
        self.gen_prompt_file = self.base_dir / "generation_prompt.md"
        self.judge_prompt_file = self.base_dir / "judge_prompt.md"
        self.original_winners_dir = self.base_dir / "original_winners"
        self.generator = generator
        self.judge = judge
        
    def get_images_from_folder(self, folder):
        return sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))

    def get_full_gen_prompt(self):
        with open(self.gen_prompt_file, "r") as f:
            full_prompt = f.read()
        desc_file = self.base_dir / "contest_description.md"
        if desc_file.exists():
            with open(desc_file, "r") as f:
                full_prompt += "\n\n" + f.read()
        return full_prompt

    def get_full_judge_prompt(self):
        with open(self.judge_prompt_file, "r") as f:
            judge_prompt = f.read()
        desc_file = self.base_dir / "contest_description.md"
        if desc_file.exists():
            with open(desc_file, "r") as f:
                judge_prompt += "\n\n" + f.read()
        return judge_prompt

    def generate_round(self, round_n):
        submission_dir = self.base_dir / f"submissions_round_{round_n}"
        submission_dir.mkdir(parents=True, exist_ok=True)
        
        if round_n == 1:
            all_prev_winners = self.get_images_from_folder(self.original_winners_dir)
            prev_winners = random.sample(all_prev_winners, min(3, len(all_prev_winners)))
            print(f"Round 1: Randomly selected {len(prev_winners)} seeds from original_winners.")
            winners_0_dir = self.base_dir / "winners_round_0"
            winners_0_dir.mkdir(parents=True, exist_ok=True)
            for w in prev_winners:
                shutil.copy(w, winners_0_dir / w.name)
        else:
            prev_winners_dir = self.base_dir / f"winners_round_{round_n - 1}"
            prev_winners = self.get_images_from_folder(prev_winners_dir)[:3]
            
        full_prompt = self.get_full_gen_prompt()
        print(f"--- Round {round_n}: Generating 10 submissions ---")
        
        for i in range(1, 11):
            print(f"Generating image {i}/10...")
            attempt = 0
            while True:
                try:
                    output_path = submission_dir / f"submission_{i}.jpg"
                    self.generator.generate(full_prompt, prev_winners, output_path, attempt=attempt)
                    time.sleep(2)
                    break
                except Exception as e:
                    attempt += 1
                    print(f"\nError generating image {i} (Attempt {attempt}): {e}")
                    sleep_time = min((2 ** attempt) * 5, 120)
                    print(f"Sleeping for {sleep_time} seconds before retrying...")
                    time.sleep(sleep_time)
        print("\nGeneration complete.")

    def judge_round(self, round_n):
        submission_dir = self.base_dir / f"submissions_round_{round_n}"
        winners_dir = self.base_dir / f"winners_round_{round_n}"
        winners_dir.mkdir(parents=True, exist_ok=True)
        
        submissions = self.get_images_from_folder(submission_dir)
        if not submissions:
            print(f"Error: No submissions found in {submission_dir}")
            return
            
        judge_prompt = self.get_full_judge_prompt()
        print(f"--- Round {round_n}: Judging winners ---")
        
        attempt = 0
        while True:
            try:
                self.judge.judge(submissions, judge_prompt, winners_dir)
                break
            except Exception as e:
                attempt += 1
                print(f"Error during judging (Attempt {attempt}): {e}")
                sleep_time = min((2 ** attempt) * 5, 120)
                print(f"Sleeping for {sleep_time} seconds before retrying judging...")
                time.sleep(sleep_time)

    def run(self, num_rounds):
        start_round = 1
        while True:
            winners_dir = self.base_dir / f"winners_round_{start_round}"
            if winners_dir.exists() and len(self.get_images_from_folder(winners_dir)) >= 3:
                start_round += 1
            else:
                break

        if start_round <= num_rounds:
            print(f"Resuming from round {start_round}...")
            partial_sub_dir = self.base_dir / f"submissions_round_{start_round}"
            if partial_sub_dir.exists():
                shutil.rmtree(partial_sub_dir)
            partial_win_dir = self.base_dir / f"winners_round_{start_round}"
            if partial_win_dir.exists():
                shutil.rmtree(partial_win_dir)

        for n in range(start_round, num_rounds + 1):
            print(f"\n=== STARTING ROUND {n} ===")
            self.generate_round(n)
            self.judge_round(n)
            print(f"=== COMPLETED ROUND {n} ===")
            
            print(f"Updating README and pushing to GitHub for Round {n}...")
            try:
                generate_readme.generate_readme(str(self.base_dir))
                subprocess.run(["git", "add", "."], check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"Add results for Round {n}"], check=True, capture_output=True)
                subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
                print("Successfully pushed to GitHub!")
            except subprocess.CalledProcessError as e:
                print(f"Git push failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
            except Exception as e:
                print(f"Failed to update README or push: {e}")
            print("\n")

def main():
    parser = argparse.ArgumentParser(description="Run the image generation contest.")
    parser.add_argument("base_dir", type=str, help="The base images directory (e.g., images_v2)")
    parser.add_argument("num_rounds", type=int, nargs="?", default=1, help="Number of rounds to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initial image selection")
    parser.add_argument("--generator", type=str, default="flux", choices=["gemini", "flux"], help="Image generator model to use")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generator = FluxImageGenerator() if args.generator == "flux" else GeminiImageGenerator()
    judge = GeminiImageJudge()
    
    runner = ContestRunner(args.base_dir, generator, judge)
    runner.run(args.num_rounds)

if __name__ == "__main__":
    main()
