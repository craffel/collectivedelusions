import torch
from diffusers import DiffusionPipeline
print("Loading pipe")
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
try:
    from accelerate import infer_auto_device_map, dispatch_model
    # For a pipeline, we might not dispatch easily, let's see if diffusers supports device_map natively for this pipeline
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, device_map="balanced")
    print("Loaded with device_map balanced")
except Exception as e:
    print("Error:", e)
