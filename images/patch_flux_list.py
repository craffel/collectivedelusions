with open('image_contest.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("class FluxImageGenerator(ImageGenerator):"):
        start_idx = i
    if line.startswith("class ImageJudge:"):
        end_idx = i
        break

new_class = """class FluxImageGenerator(ImageGenerator):
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
        
        # Open seed images directly as a list of PIL Images
        opened_images = [Image.open(img_path).convert("RGB") for img_path in seed_images]
        
        # Pre-resize all images to 1024x1024 (Flux's native resolution) before passing to pipeline
        target_size = 1024
        resized_images = [img.resize((target_size, target_size)) for img in opened_images]
            
        print(f"  Attempt {attempt+1} feeding prompt and {len(resized_images)} images directly to Flux...")
        
        image = self.pipe(
            prompt=full_prompt,
            image=resized_images,
            guidance_scale=3.5,
            num_inference_steps=28,
            height=target_size,
            width=target_size,
        ).images[0]
        
        image.save(output_path)
        torch.cuda.empty_cache()

"""

lines = lines[:start_idx] + [new_class] + lines[end_idx:]

with open('image_contest.py', 'w') as f:
    f.writelines(lines)
