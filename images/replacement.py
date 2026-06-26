import pathlib
from PIL import Image

class FluxImageGenerator:
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
