import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

pipe = AutoPipelineForImage2Image.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

init_image = Image.open("images_v4/winners_round_0/PCWINNERS-Baby-Animals-1st-WINTER26-960x686.jpg").convert("RGB").resize((1024, 1024))

out = pipe(
    prompt="A stylized version of the image",
    image=init_image,
    strength=0.7,
    guidance_scale=3.5,
    num_inference_steps=20
).images[0]
out.save("test_flux2_img2img.jpg")
print("Done!")
