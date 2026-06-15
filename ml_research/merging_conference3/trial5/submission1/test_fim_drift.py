import sys
import os
sys.path.insert(0, os.path.abspath("./custom_libs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.func import functional_call
from torch.optim import AdamW
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # 1. Generate synthetic text data for 2 distinct tasks
    texts_task1 = [
        "this movie was absolutely fantastic and wonderful",
        "i loved the acting and the beautiful direction",
        "great story line and highly entertaining plot",
        "terrible acting and extremely boring screenplay",
        "worst film i have ever seen in my life",
        "complete waste of time and money do not watch"
    ] * 10
    labels_task1 = [1, 1, 1, 0, 0, 0] * 10

    texts_task2 = [
        "the team scored an amazing goal in the final minute",
        "championship match was exciting with great basketball players",
        "football tournament began yesterday with several matches",
        "quantum physics experiments reveal new properties of atoms",
        "astronomical telescope observed a distant supernova galaxy",
        "scientific researchers published a breakthrough paper on chemistry"
    ] * 10
    labels_task2 = [0, 0, 0, 1, 1, 1] * 10

    def tokenize_texts(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
        dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
        return dataset

    ds1 = tokenize_texts(texts_task1, labels_task1)
    ds2 = tokenize_texts(texts_task2, labels_task2)

    dl1 = DataLoader(ds1, batch_size=8, shuffle=True)
    dl2 = DataLoader(ds2, batch_size=8, shuffle=True)

    # Train expert models
    print("Fine-tuning Experts...")
    expert1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    expert2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    opt1 = AdamW(expert1.parameters(), lr=1e-4)
    opt2 = AdamW(expert2.parameters(), lr=1e-4)
    
    expert1.train()
    for epoch in range(5):
        for batch in dl1:
            opt1.zero_grad()
            input_ids, attn_mask, labels = [b.to(device) for b in batch]
            outputs = expert1(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            opt1.step()

    expert2.train()
    for epoch in range(5):
        for batch in dl2:
            opt2.zero_grad()
            input_ids, attn_mask, labels = [b.to(device) for b in batch]
            outputs = expert2(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            opt2.step()

    expert1.eval()
    expert2.eval()
    base_model.eval()

    L = 12
    v1_layers, v2_layers, base_layers = [], [], []
    for l in range(L):
        layer_base = getattr(base_model.bert.encoder.layer, str(l))
        layer_exp1 = getattr(expert1.bert.encoder.layer, str(l))
        layer_exp2 = getattr(expert2.bert.encoder.layer, str(l))
        
        v1_layers.append({n: p_exp1 - p_base for (n, p_base), p_exp1 in zip(layer_base.named_parameters(), layer_exp1.parameters())})
        v2_layers.append({n: p_exp2 - p_base for (n, p_base), p_exp2 in zip(layer_base.named_parameters(), layer_exp2.parameters())})
        base_layers.append({n: p_base for n, p_base in layer_base.named_parameters()})

    # Prepare calibration data
    cal_texts = texts_task1[:8] + texts_task2[:8]
    cal_enc = tokenizer(cal_texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
    input_ids = cal_enc["input_ids"].to(device)
    attn_mask = cal_enc["attention_mask"].to(device)

    # 2. Function to compute FIM trace for a given state dict
    def compute_fim_trace(params_dict, model_obj):
        c_raw = torch.zeros(L, device=device)
        grad_params = {n: p.clone().detach().requires_grad_(True) for n, p in params_dict.items() if "bert.encoder.layer" in n}
        
        eval_params = {n: p for n, p in params_dict.items()}
        for n, p in grad_params.items():
            eval_params[n] = p
            
        outputs = functional_call(model_obj, eval_params, kwargs={"input_ids": input_ids, "attention_mask": attn_mask})
        probs = torch.softmax(outputs.logits, dim=-1)
        
        for i in range(len(cal_texts)):
            y_sample = torch.multinomial(probs[i], 1).item()
            loss = -torch.log(probs[i, y_sample] + 1e-8)
            
            grads = torch.autograd.grad(loss, grad_params.values(), retain_graph=True, allow_unused=True)
            
            for l in range(L):
                prefix = f"bert.encoder.layer.{l}."
                for (name, param), g in zip(grad_params.items(), grads):
                    if name.startswith(prefix) and g is not None:
                        c_raw[l] += g.pow(2).sum()
        return c_raw

    # Compute FIM trace at Base (step 0)
    base_params = {n: p for n, p in base_model.named_parameters()}
    c_base = compute_fim_trace(base_params, base_model)
    print(f"Base FIM Trace: {c_base.tolist()}")

    # Compute FIM trace at RCR-merged model (step 50) using the final coeffs of RCR-Merge
    coeffs_rcr = torch.tensor([[0.4387, 0.4810, 0.5140, 0.5441, 0.4811, 0.4773, 0.4797, 0.4942, 0.5263, 0.5642, 0.5299, 0.5274],
                                [0.5119, 0.5357, 0.5114, 0.4996, 0.5235, 0.5442, 0.5098, 0.5231, 0.5507, 0.5207, 0.5500, 0.5203]], device=device)
    
    # Construct merged params dictionary
    merged_params = {}
    for name, param in expert1.named_parameters():
        is_merged = False
        for l in range(L):
            prefix = f"bert.encoder.layer.{l}."
            if name.startswith(prefix):
                is_merged = True
                p_name = name[len(prefix):]
                merged_p = base_layers[l][p_name] + coeffs_rcr[0, l]*v1_layers[l][p_name] + coeffs_rcr[1, l]*v2_layers[l][p_name]
                merged_params[name] = merged_p
                break
        if not is_merged:
            merged_params[name] = param

    c_online = compute_fim_trace(merged_params, expert1)
    print(f"Online FIM Trace (Step 50): {c_online.tolist()}")

    # Cosine Similarity
    dot_prod = (c_base * c_online).sum()
    norm_base = torch.linalg.norm(c_base)
    norm_online = torch.linalg.norm(c_online)
    cos_sim = dot_prod / (norm_base * norm_online)
    print(f"Cosine Similarity between base and online FIM traces: {cos_sim.item():.6f}")

if __name__ == "__main__":
    main()
