import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from merge_df_bfm import merge_models_df_bfm, simple_average, task_arithmetic, get_task_vector
import os
from tqdm import tqdm

# Constants
TASKS = ["mnist", "eurosat", "gtsrb", "stanford-cars", "resisc45", "dtd", "svhn", "sun397"]
MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_MAPPING = {
    "mnist": ("mnist", "test"),
    "eurosat": ("erosat", "test"), # Note: might need exact HF name
    "gtsrb": ("mwts/gtsrb", "test"),
    "stanford-cars": ("cars196", "test"),
    "resisc45": ("resisc45", "test"),
    "dtd": ("dtd", "test"),
    "svhn": ("svhn", "test"),
    "sun397": ("sun397", "test")
}

def evaluate_model(model, processor, task, device):
    """
    Simplified zero-shot evaluation for CLIP on specific datasets.
    In a real research setting, we'd use proper class names for prompts.
    """
    model.eval()
    # Placeholder for actual CLIP zero-shot logic
    # For now, return a dummy score to test pipeline
    import random
    return 0.5 + 0.4 * random.random()

def run_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    finetuned_models = []
    task_vectors = []
    
    for task in TASKS:
        ft_id = f"tanganke/clip-vit-base-patch32_{task}"
        print(f"Loading {ft_id}...")
        try:
            ft_model = CLIPModel.from_pretrained(ft_id).to(device)
            finetuned_models.append(ft_model)
            task_vectors.append(get_task_vector(pretrained_model, ft_model))
        except Exception as e:
            print(f"Skipping {task} due to error: {e}")

    results = {}
    
    # 1. Simple Average
    print("Merging: Simple Average")
    avg_sd = simple_average([m.state_dict() for m in finetuned_models])
    # ... evaluate ...
    
    # 2. Task Arithmetic
    print("Merging: Task Arithmetic")
    ta_sd = task_arithmetic(pretrained_model.state_dict(), task_vectors)
    # ... evaluate ...

    # 3. DF-BFM (Ours)
    print("Merging: DF-BFM (Ours)")
    df_bfm_sd = merge_models_df_bfm(pretrained_model, finetuned_models)
    
    # Save results
    torch.save(df_bfm_sd, "df_bfm_final.pt")
    print("Experiments completed.")

if __name__ == "__main__":
    run_all()
