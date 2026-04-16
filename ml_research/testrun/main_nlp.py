import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from merge_df_bfm import (
    get_task_vector, merge_models_df_bfm, simple_average, 
    task_arithmetic, ties_merging, actmat_merging
)
from datasets import load_dataset
from tqdm import tqdm
import json

def evaluate_t5(model, tokenizer, task, device, num_samples=100):
    model.eval()
    if task == "paws":
        ds = load_dataset("paws", "labeled_final", split="test")
        input_fmt = "paws sentence1: {sentence1} sentence2: {sentence2}"
        label_map = {"0": "different", "1": "same"}
    elif task == "qasc":
        ds = load_dataset("qasc", split="test")
        input_fmt = "qasc question: {question} choice: {choices[text]}"
        # Simplified for rank eval
    else:
        return 0.5
    
    correct = 0
    total = 0
    indices = list(range(min(num_samples, len(ds))))
    
    for i in tqdm(indices, desc=f"Eval {task}"):
        item = ds[i]
        if task == "paws":
            inp = input_fmt.format(**item)
            target = label_map[str(item['label'])]
            candidates = ["different", "same"]
        else:
            continue

        inputs = tokenizer(inp, return_tensors="pt").to(device)
        
        # Rank classification
        best_score = -float('inf')
        best_cand = None
        
        for cand in candidates:
            with torch.no_grad():
                labels = tokenizer(cand, return_tensors="pt").input_ids.to(device)
                outputs = model(**inputs, labels=labels)
                log_prob = -outputs.loss.item() # Approximation
                if log_prob > best_score:
                    best_score = log_prob
                    best_cand = cand
        
        if best_cand == target:
            correct += 1
        total += 1
        
    return correct / total if total > 0 else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "google-t5/t5-base"
    pretrained_model = T5ForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    
    # Using found models
    TASKS = {
        "paws": "SeongwooKim/T5-base-paws",
        "storycloze": "SeongwooKim/T5-base-story_cloze",
        "winogrande": "SeongwooKim/T5-base-winogrande"
    }
    
    finetuned_models = []
    task_vectors = []
    task_names = []
    
    for name, path in TASKS.items():
        print(f"Loading {path}...")
        try:
            m = T5ForConditionalGeneration.from_pretrained(path).to(device)
            finetuned_models.append(m)
            task_vectors.append(get_task_vector(pretrained_model, m))
            task_names.append(name)
        except Exception as e:
            print(f"Error: {e}")

    methods = {
        "Simple Average": lambda: simple_average([m.state_dict() for m in finetuned_models]),
        "Task Arithmetic": lambda: task_arithmetic(pretrained_model.state_dict(), task_vectors),
        "TIES-Merging": lambda: ties_merging(pretrained_model.state_dict(), task_vectors),
        "ACTMat (Data-Free)": lambda: actmat_merging(pretrained_model, finetuned_models),
        "DF-BFM (Ours)": lambda: merge_models_df_bfm(pretrained_model, finetuned_models)
    }

    results = {}
    for name, merge_fn in methods.items():
        print(f"\n--- {name} ---")
        merged_sd = merge_fn()
        test_model = T5ForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
        test_model.load_state_dict(merged_sd)
        
        # Just evaluate on paws for speed/demo
        acc = evaluate_t5(test_model, tokenizer, "paws", device, num_samples=100)
        print(f"{name} PAWS: {acc:.4f}")
        results[name] = acc

    with open("results_nlp.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
