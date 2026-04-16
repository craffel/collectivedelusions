import torch
from transformers import CLIPModel, CLIPProcessor
from merge_df_bfm import (
    get_task_vector, merge_models_df_bfm, merge_models_df_bfm_sq, simple_average, 
    task_arithmetic, ties_merging, actmat_merging
)
from evaluate_clip import evaluate_clip_task
import os
import json

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "openai/clip-vit-base-patch32"
    TASKS = ["mnist", "gtsrb", "eurosat"]
    
    print(f"Loading pretrained model {MODEL_ID}...", flush=True)
    pretrained_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    finetuned_models = []
    task_vectors = []
    
    for task in TASKS:
        ft_id = f"tanganke/clip-vit-base-patch32_{task}"
        print(f"Loading {ft_id}...", flush=True)
        try:
            # Add timeout/retry logic or just more prints
            print(f"Calling CLIPModel.from_pretrained for {task}...", flush=True)
            ft_model = CLIPModel.from_pretrained(ft_id).to(device)
            print(f"Successfully loaded {task}", flush=True)
            finetuned_models.append(ft_model)
            task_vectors.append(get_task_vector(pretrained_model, ft_model))
        except Exception as e:
            print(f"Error loading {task}: {e}", flush=True)

    methods = {
        "Simple Average": lambda: simple_average([m.state_dict() for m in finetuned_models]),
        "Task Arithmetic": lambda: task_arithmetic(pretrained_model.state_dict(), task_vectors),
        "TIES-Merging": lambda: ties_merging(pretrained_model.state_dict(), task_vectors),
        "ACTMat (Data-Free)": lambda: actmat_merging(pretrained_model, finetuned_models),
        "DF-BFM (v1 - Abs)": lambda: merge_models_df_bfm(pretrained_model, finetuned_models),
        "DF-BFM (v2 - Sq)": lambda: merge_models_df_bfm_sq(pretrained_model, finetuned_models)
    }

    all_results = {}

    for name, merge_fn in methods.items():
        print(f"\n--- Testing Method: {name} ---")
        merged_sd = merge_fn()
        
        # Load merged state dict into a temporary model
        test_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        test_model.load_state_dict(merged_sd)
        
        method_results = {}
        for task in TASKS:
            acc = evaluate_clip_task(test_model, processor, task, device, num_samples=200)
            print(f"{name} - {task}: {acc:.4f}")
            method_results[task] = acc
        
        avg_acc = sum(method_results.values()) / len(method_results)
        print(f"{name} - Average: {avg_acc:.4f}")
        method_results["Average"] = avg_acc
        all_results[name] = method_results

    # Save results
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    print("\nAll experiments complete. Results saved to results.json")

if __name__ == "__main__":
    main()
