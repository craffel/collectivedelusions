import sys
sys.modules['flash_attn'] = None

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
    print("Successfully imported CLIPModel with flash attention bypass!")
except Exception as e:
    print("Bypass failed:", e)
