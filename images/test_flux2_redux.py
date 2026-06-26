import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from PIL import Image

device = "cuda"
dtype = torch.bfloat16

print("Loading Redux...")
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=dtype
).to(device)

print("Loading Flux.2-dev...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", 
    text_encoder=None, text_encoder_2=None, 
    torch_dtype=dtype,
    device_map="balanced"
)

image_paths = [
    "images_v4/winners_round_0/PCWINNERS-Baby-Animals-1st-WINTER26-960x686.jpg",
    "images_v4/winners_round_0/PCWINNERS-Birds-2nd-WINTER26-960x642.jpg",
    "images_v4/winners_round_0/PCWINNERS-Other-Wildlife-1st-WINTER26_960x678.jpg"
]
images = [Image.open(p).convert("RGB") for p in image_paths]

print("Running Redux...")
# We pass the prompt directly into Redux if we loaded text encoders, but we didn't. 
# Wait, if we want text conditioning as well, we should load text encoders in the prior pipeline!
