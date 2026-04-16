import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import numpy as np
from merger import sr_actmat_merge, task_arithmetic_merge, ties_merge, actmat_merge, asr_actmat_merge, psr_actmat_merge, ace_merging, sr_ace_merging, asr_ace_merging, spectral_ace_merging, base_prior_ace_merging, adaptive_ace_merging
from train_experts import preprocess_function, TASKS, MODEL_ID
import evaluate
import os

def evaluate_model(model, tokenizer, task):
    dataset = load_dataset("glue", task, split="validation")
    
    # Simple manual evaluation for T5 conditional generation
    model.eval()
    model.to("cuda")
    
    correct = 0
    total = 0
    
    if task == "cola":
        label_map = {0: "unacceptable", 1: "acceptable"}
        prefix = "cola sentence: "
    elif task == "mrpc":
        label_map = {0: "no", 1: "yes"}
        prefix = "mrpc sentence1: "
    elif task == "rte":
        label_map = {0: "entailment", 1: "not_entailment"}
        prefix = "rte sentence1: "
    elif task == "qnli":
        label_map = {0: "entailment", 1: "not_entailment"}
        prefix = "qnli sentence1: "

    inv_label_map = {v: k for k, v in label_map.items()}

    print(f"Evaluating {task}...")
    for i, example in enumerate(dataset):
        if i >= 200: break # Limit eval for speed
        
        if task == "cola":
            input_text = prefix + example["sentence"]
        elif task == "qnli":
            input_text = f"qnli sentence1: {example['question']} sentence2: {example['sentence']}"
        else:
            input_text = prefix + example["sentence1"] + " sentence2: " + example["sentence2"]
            
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=10)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        gt_label = label_map[example["label"]]
        if prediction == gt_label:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    base_sd = base_model.state_dict()
    
    expert_sds = []
    for task in TASKS:
        expert_path = f"./experts/{task}_final"
        if not os.path.exists(expert_path):
            print(f"Expert for {task} not found at {expert_path}. Skipping.")
            continue
        expert_model = T5ForConditionalGeneration.from_pretrained(expert_path)
        expert_sds.append(expert_model.state_dict())
        
    if len(expert_sds) < len(TASKS):
        print(f"Found {len(expert_sds)} experts out of {len(TASKS)} expected.")

    methods = {
        "Base Model": lambda: base_sd,
        "ACTMat": lambda: actmat_merge(base_sd, expert_sds),
        "ACE-Merging (G=0.05)": lambda: ace_merging(base_sd, expert_sds, gamma=0.05),
        "ACE-Merging (G=0.1)": lambda: ace_merging(base_sd, expert_sds, gamma=0.1),
        "ACE-Merging (G=0.15)": lambda: ace_merging(base_sd, expert_sds, gamma=0.15),
        "ACE-Merging (G=0.2)": lambda: ace_merging(base_sd, expert_sds, gamma=0.2),
        "Base-Prior ACE (B=0.1)": lambda: base_prior_ace_merging(base_sd, expert_sds, alpha=0.1, beta=0.1),
    }
    
    # Add individual experts for reference
    for i, task in enumerate(TASKS):
        methods[f"Expert-{task}"] = lambda i=i: expert_sds[i]
    
    results = {}
    for method_name, merge_fn in methods.items():
        print(f"--- Testing {method_name} ---")
        merged_sd = merge_fn()
        base_model.load_state_dict(merged_sd)
        
        method_results = {}
        for task in TASKS:
            acc = evaluate_model(base_model, tokenizer, task)
            method_results[task] = acc
            print(f"{method_name} - {task}: {acc:.4f}")
        
        results[method_name] = method_results

    # Print summary table
    header = "Method".ljust(20) + "".join([t.rjust(10) for t in TASKS])
    print("\n" + header)
    print("-" * len(header))
    for method_name, method_results in results.items():
        row = method_name.ljust(20) + "".join([f"{method_results[t]:.4f}".rjust(10) for t in TASKS])
        print(row)

if __name__ == "__main__":
    main()
