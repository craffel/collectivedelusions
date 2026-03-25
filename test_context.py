import os
import sys
import random
import pathlib
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: 'google-genai' library not found.")
    sys.exit(1)

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
GEN_MODEL = "gemini-3.1-flash-image-preview"
JUDGE_MODEL = "gemini-3-flash-preview"

OUTPUT_DIR = pathlib.Path("/fsx/craffel/collectivedelusions/test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper for drawing synthetic images
def create_image_with_text(text, color="white", bg_color="black"):
    img = Image.new('RGB', (400, 400), color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        # Pillow 12+ supports size in load_default
        font = ImageFont.load_default(size=150)
    except TypeError:
        font = ImageFont.load_default()
    
    # Simple centering approximation
    draw.text((100, 100), text, fill=color, font=font)
    return img

def test_generation_context():
    print("--- Test 1: Generation Context ---")
    digits = [2, 3, 4]
    expected_sum = sum(digits)
    print(f"Input digits: {digits}. Expected output image should show: {expected_sum}")
    
    images = []
    for idx, d in enumerate(digits):
        img = create_image_with_text(str(d), bg_color="blue")
        img_path = OUTPUT_DIR / f"input_digit_{idx}.jpg"
        img.save(img_path)
        images.append(img)
    
    prompt = (
        "Attached are three images. Each image contains a single digit. "
        "Calculate the sum of these three digits. "
        "Generate a new image that displays ONLY the resulting sum as a number."
    )
    
    print("Calling Gemini generation model...")
    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=[prompt] + images,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
        )
        
        found = False
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                img = Image.open(BytesIO(part.inline_data.data))
                out_path = OUTPUT_DIR / "output_sum.jpg"
                img.save(out_path)
                print(f"Success! Generated image saved to {out_path}.")
                print(f"Please review {out_path} to see if it shows '{expected_sum}'.")
                found = True
                break
        if not found:
            print("Model failed to return an image.")
            print(response.text)
    except Exception as e:
        print(f"Generation test failed: {e}")

def test_judging_context():
    print("\n--- Test 2: Judging Context ---")
    
    # Pick 3 random indices (1-based) to be the "winners"
    winners = sorted(random.sample(range(1, 11), 3))
    print(f"Ground truth winners (have marker): {winners}")
    
    images = []
    for i in range(1, 11):
        if i in winners:
            img = create_image_with_text(f"{i}\nSTAR", color="green", bg_color="black")
        else:
            img = create_image_with_text(f"{i}\nNO", color="red", bg_color="black")
        
        img_path = OUTPUT_DIR / f"judge_input_{i}.jpg"
        img.save(img_path)
        images.append(img)
    
    prompt = (
        "Attached are ten images, labeled 1 to 10 in the prompt. "
        "Three of these images contain the word 'STAR' in green text. "
        "The rest contain the word 'NO' in red text. "
        "Please identify exactly which three images contain the word 'STAR'. "
        "List ONLY the numbers of those three images as a comma-separated list (e.g., '1, 4, 7')."
    )
    
    contents = [prompt]
    for i, img in enumerate(images):
        contents.append(f"Image {i+1}:")
        contents.append(img)
        
    print("Calling Gemini judging model...")
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=contents
        )
        print(f"Model response: {response.text.strip()}")
        print("Does this match the ground truth?")
    except Exception as e:
        print(f"Judging test failed: {e}")

if __name__ == "__main__":
    test_generation_context()
    test_judging_context()
