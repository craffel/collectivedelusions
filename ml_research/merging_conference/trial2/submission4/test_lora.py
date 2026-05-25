import torch
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

print("Loading pre-trained ViT...")
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)
print("Base model loaded.")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none"
)

peft_model = get_peft_model(model, config)
print("PEFT model created.")
peft_model.print_trainable_parameters()

# Inspect some modules
for name, module in peft_model.named_modules():
    if "query" in name and "lora_A" in name:
        print("Found lora module:", name)
        break
