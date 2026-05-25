import torch
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig, TaskType

try:
    print("Loading model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    print("Model loaded successfully.")
    
    # Configure LoRA
    # For HF ViT, the query and value projections are q_proj and v_proj
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"] # we save the classification head separately for each expert
    )
    
    print("Applying LoRA...")
    peft_model = get_peft_model(model, peft_config)
    print("PEFT model applied successfully.")
    peft_model.print_trainable_parameters()
    
except Exception as e:
    print("Error:", e)
