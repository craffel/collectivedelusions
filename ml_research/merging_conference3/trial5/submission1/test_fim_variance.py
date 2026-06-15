import sys
import os
sys.path.insert(0, os.path.abspath("./custom_libs"))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.func import functional_call
import numpy as np

torch.manual_seed(42)

def main():
    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    L = 12
    # Prepare calibration data
    cal_texts = [
        "this movie was absolutely fantastic and wonderful",
        "i loved the acting and the beautiful direction",
        "great story line and highly entertaining plot",
        "terrible acting and extremely boring screenplay",
        "worst film i have ever seen in my life",
        "complete waste of time and money do not watch",
        "the team scored an amazing goal in the final minute",
        "championship match was exciting with great basketball players",
        "football tournament began yesterday with several matches",
        "quantum physics experiments reveal new properties of atoms",
        "astronomical telescope observed a distant supernova galaxy",
        "scientific researchers published a breakthrough paper on chemistry"
    ]
    cal_enc = tokenizer(cal_texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
    input_ids = cal_enc["input_ids"].to(device)
    attn_mask = cal_enc["attention_mask"].to(device)

    # We need to compute gradients on the parameters of the base model
    base_params = {n: p for n, p in base_model.named_parameters()}
    grad_params = {n: p.clone().detach().requires_grad_(True) for n, p in base_params.items() if "bert.encoder.layer" in n}
    
    eval_params = {n: p for n, p in base_params.items()}
    for n, p in grad_params.items():
        eval_params[n] = p
        
    outputs = functional_call(base_model, eval_params, kwargs={"input_ids": input_ids, "attention_mask": attn_mask})
    probs = torch.softmax(outputs.logits, dim=-1)
    
    # Store squared gradients for each parameter
    param_squared_grads = {n: torch.zeros_like(p) for n, p in grad_params.items()}
    
    for i in range(len(cal_texts)):
        y_sample = torch.multinomial(probs[i], 1).item()
        loss = -torch.log(probs[i, y_sample] + 1e-8)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, grad_params.values(), retain_graph=True, allow_unused=True)
        
        # Accumulate squared gradients
        for n, g in zip(grad_params.keys(), grads):
            if g is not None:
                param_squared_grads[n] += g.pow(2)

    # Now let's group by transformer layer and components
    print("="*60)
    print("ANISOTROPY ANALYSIS OF FIM TRACES IN BERT-BASE-UNCASED (L=12)")
    print("="*60)
    
    for l in range(L):
        print(f"\n--- Layer {l} ---")
        layer_prefix = f"bert.encoder.layer.{l}."
        
        # We will collect the FIM traces of different functional components:
        # 1. Attention Query, Key, Value
        # 2. Attention Output projection
        # 3. MLP Intermediate Dense
        # 4. MLP Output Dense
        components = {
            "Attention QKV": ["attention.self.query", "attention.self.key", "attention.self.value"],
            "Attention Out": ["attention.output.dense"],
            "MLP Intermediate": ["intermediate.dense"],
            "MLP Output": ["output.dense"]
        }
        
        component_traces = {}
        for comp_name, patterns in components.items():
            comp_trace_sum = 0.0
            num_params = 0
            for n, sq_grad in param_squared_grads.items():
                if n.startswith(layer_prefix):
                    short_name = n[len(layer_prefix):]
                    if any(patt in short_name for patt in patterns):
                        comp_trace_sum += sq_grad.sum().item()
                        num_params += sq_grad.numel()
            component_traces[comp_name] = (comp_trace_sum, num_params)
            
        total_layer_trace = sum(t[0] for t in component_traces.values())
        print(f"Total Layer {l} FIM Trace: {total_layer_trace:.6e}")
        
        traces_list = []
        for comp_name, (trace_sum, num_params) in component_traces.items():
            percentage = (trace_sum / total_layer_trace) * 100 if total_layer_trace > 0 else 0
            mean_intensity = trace_sum / num_params if num_params > 0 else 0
            traces_list.append(trace_sum)
            print(f"  {comp_name:<18} | Trace: {trace_sum:.6e} ({percentage:5.2f}%) | Params: {num_params:6d} | Intensity: {mean_intensity:.6e}")
            
        variance = np.var(traces_list)
        std_dev = np.std(traces_list)
        print(f"  Component-wise Trace Variance: {variance:.6e} (Std Dev: {std_dev:.6e})")

if __name__ == "__main__":
    main()
