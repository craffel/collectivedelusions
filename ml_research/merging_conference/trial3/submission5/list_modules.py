from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

print("First 40 module names:")
for i, (name, module) in enumerate(model.named_modules()):
    if i < 40:
        print(f"{i}: {name} (type: {type(module).__name__})")
