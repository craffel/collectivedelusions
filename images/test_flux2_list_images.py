import torch
from diffusers import Flux2Pipeline
from PIL import Image

print("Loading pipeline...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

# Load 3 test images
img_paths = [
    "images_v4/winners_round_0/PCWINNERS-Baby-Animals-1st-WINTER26-960x686.jpg",
    "images_v4/winners_round_0/PCWINNERS-Birds-2nd-WINTER26-960x642.jpg",
    "images_v4/winners_round_0/PCWINNERS-Other-Wildlife-1st-WINTER26_960x678.jpg"
]
images = [Image.open(p).convert("RGB").resize((512, 512)) for p in img_paths]

print("Running pipeline with a list of 3 images...")
try:
    out = pipe(
        prompt="A stylized version of the image",
        image=images,
        guidance_scale=3.5,
        num_inference_steps=10
    ).images
    print(f"Success! Number of images returned: {len(out)}")
except Exception as e:
    print(f"Error: {e}")
