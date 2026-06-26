import torch
from diffusers import AutoPipelineForImage2Image
print("Loading AutoPipelineForImage2Image...")
pipe = AutoPipelineForImage2Image.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)
print("Pipeline loaded.")
