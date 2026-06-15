import sys
import os
sys.path.insert(0, os.path.abspath("./custom_libs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.func import functional_call
from torch.optim import AdamW, Adam
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("="*60)
    print("REAL-WORLD BERT-BASE-UNCASED MODEL MERGING PILOT STUDY (L=12)")
    print("="*60)

    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.to(device)

    # 1. Generate synthetic text data for 2 distinct tasks
    # Task 1: Sentiment Classification (Positive vs Negative)
    # Task 2: Topic Classification (Sports vs Science)
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

    # Tokenize data
    def tokenize_texts(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
        dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
        return dataset

    ds1 = tokenize_texts(texts_task1, labels_task1)
    ds2 = tokenize_texts(texts_task2, labels_task2)

    dl1 = DataLoader(ds1, batch_size=8, shuffle=True)
    dl2 = DataLoader(ds2, batch_size=8, shuffle=True)

    # 2. Train two expert models from the shared base model
    print("Training Expert 1 (Sentiment) and Expert 2 (Topic)...")
    
    expert1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    expert2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    # Training Loop for 5 epochs
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

    # Define task vectors for the 12 encoder layers (L=12)
    L = 12
    K = 2

    v1_layers = []
    v2_layers = []
    base_layers = []

    for l in range(L):
        layer_base = getattr(base_model.bert.encoder.layer, str(l))
        layer_exp1 = getattr(expert1.bert.encoder.layer, str(l))
        layer_exp2 = getattr(expert2.bert.encoder.layer, str(l))
        
        # Collect parameters
        v1_p = {n: p_exp1 - p_base for (n, p_base), p_exp1 in zip(layer_base.named_parameters(), layer_exp1.parameters())}
        v2_p = {n: p_exp2 - p_base for (n, p_base), p_exp2 in zip(layer_base.named_parameters(), layer_exp2.parameters())}
        base_p = {n: p_base for n, p_base in layer_base.named_parameters()}
        
        v1_layers.append(v1_p)
        v2_layers.append(v2_p)
        base_layers.append(base_p)

    # 3. Estimate base FIM trace (base curvature) for the 12 layers
    print("Estimating pre-trained base model curvature (FIM diagonal trace)...")
    c_raw = torch.zeros(L, device=device)
    base_model.zero_grad()
    
    # Use calibration inputs from both tasks
    cal_texts = texts_task1[:8] + texts_task2[:8]
    cal_encodings = tokenizer(cal_texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
    cal_input_ids = cal_encodings["input_ids"].to(device)
    cal_attn_mask = cal_encodings["attention_mask"].to(device)
    
    outputs = base_model(input_ids=cal_input_ids, attention_mask=cal_attn_mask)
    probs = torch.softmax(outputs.logits, dim=-1)
    
    for i in range(len(cal_texts)):
        y_sample = torch.multinomial(probs[i], 1).item()
        loss = -torch.log(probs[i, y_sample] + 1e-8)
        grads = torch.autograd.grad(loss, base_model.parameters(), retain_graph=True)
        
        for l in range(L):
            layer_base = getattr(base_model.bert.encoder.layer, str(l))
            layer_params = list(layer_base.parameters())
            for p, g in zip(layer_params, grads):
                if g is not None:
                    c_raw[l] += g.pow(2).sum()

    # Normalize FIM trace to get curvature scale
    c = c_raw / c_raw.mean()
    print("Normalized layer-wise curvatures:")
    for l in range(L):
        print(f"  Layer {l:2d} = {c[l]:.4f}")

    # Helper function to construct merged state dict for functional_call
    def get_merged_params(coeffs, target_expert):
        merged_params = {}
        for name, param in target_expert.named_parameters():
            is_merged = False
            for l in range(L):
                prefix = f"bert.encoder.layer.{l}."
                if name.startswith(prefix):
                    is_merged = True
                    p_name = name[len(prefix):]
                    merged_p = base_layers[l][p_name] + coeffs[0, l]*v1_layers[l][p_name] + coeffs[1, l]*v2_layers[l][p_name]
                    merged_params[name] = merged_p
                    break
            if not is_merged:
                merged_params[name] = param
        return merged_params

    # 4. Evaluate accuracies of merged model
    def evaluate_model(coeffs):
        accs = []
        # Evaluate Expert 1 on Task 1 with merged backbone
        correct = 0
        total = 0
        with torch.no_grad():
            m_p1 = get_merged_params(coeffs, expert1)
            for batch in dl1:
                input_ids, attn_mask, labels = [b.to(device) for b in batch]
                outputs = functional_call(expert1, m_p1, kwargs={"input_ids": input_ids, "attention_mask": attn_mask})
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accs.append(correct / total)
        
        # Evaluate Expert 2 on Task 2 with merged backbone
        correct = 0
        total = 0
        with torch.no_grad():
            m_p2 = get_merged_params(coeffs, expert2)
            for batch in dl2:
                input_ids, attn_mask, labels = [b.to(device) for b in batch]
                outputs = functional_call(expert2, m_p2, kwargs={"input_ids": input_ids, "attention_mask": attn_mask})
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accs.append(correct / total)
        return accs

    # Initialize coefficients to Uniform (0.5, 0.5)
    coeffs_uniform = torch.ones((K, L), device=device) * 0.5
    uniform_accs = evaluate_model(coeffs_uniform)
    print(f"Uniform Baseline (0.5) - Task 1 Acc: {uniform_accs[0]*100:.2f}%, Task 2 Acc: {uniform_accs[1]*100:.2f}%, Avg: {np.mean(uniform_accs)*100:.2f}%")

    # 5. Test-Time Adaptation under Local Transductive Noise
    print("\nRunning Test-Time Adaptation on Local Stream...")
    
    # Highly biased local stream containing Task 1 text inputs (Positive Sentiment)
    tta_texts = ["this movie was absolutely fantastic and wonderful"] * 16
    tta_encodings = tokenizer(tta_texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
    tta_input_ids = tta_encodings["input_ids"].to(device)
    tta_attn_mask = tta_encodings["attention_mask"].to(device)

    # Method A: Unconstrained AdaMerging
    print("Optimizing Unconstrained AdaMerging (no regularization)...")
    coeffs_ada = (torch.ones((K, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_ada], lr=0.5)

    for step in range(50):
        optimizer.zero_grad()
        # Compute TTA loss on local batch using expert1 (Task 1)
        m_p1 = get_merged_params(coeffs_ada, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"input_ids": tta_input_ids, "attention_mask": tta_attn_mask})
        probs = torch.softmax(outputs.logits, dim=-1)
        # Unsupervised entropy minimization
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        loss_tta.backward()
        optimizer.step()
        
    ada_accs = evaluate_model(coeffs_ada.detach())
    print(f"Unconstrained AdaMerging final coeffs:\n{coeffs_ada.detach()}")
    print(f"Unconstrained AdaMerging - Task 1 Acc: {ada_accs[0]*100:.2f}%, Task 2 Acc: {ada_accs[1]*100:.2f}%, Avg: {np.mean(ada_accs)*100:.2f}%")

    # Method B: PolyMerge (d=2)
    print("\nOptimizing PolyMerge (d=2)...")
    poly_params = torch.zeros((K, 3), device=device, requires_grad=True)
    with torch.no_grad():
        poly_params[:, 0] = 0.5
    optimizer = torch.optim.Adam([poly_params], lr=0.1)

    for step in range(50):
        optimizer.zero_grad()
        coeffs_poly = torch.zeros((K, L), device=device)
        for l in range(L):
            x_l = l / (L - 1)
            coeffs_poly[:, l] = poly_params[:, 0] + poly_params[:, 1] * x_l + poly_params[:, 2] * (x_l ** 2)
            
        m_p1 = get_merged_params(coeffs_poly, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"input_ids": tta_input_ids, "attention_mask": tta_attn_mask})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        loss_tta.backward()
        optimizer.step()
        
    final_coeffs_poly = torch.zeros((K, L), device=device)
    for l in range(L):
        x_l = l / (L - 1)
        final_coeffs_poly[:, l] = poly_params[:, 0] + poly_params[:, 1] * x_l + poly_params[:, 2] * (x_l ** 2)
        
    poly_accs = evaluate_model(final_coeffs_poly.detach())
    print(f"PolyMerge final coeffs:\n{final_coeffs_poly.detach()}")
    print(f"PolyMerge - Task 1 Acc: {poly_accs[0]*100:.2f}%, Task 2 Acc: {poly_accs[1]*100:.2f}%, Avg: {np.mean(poly_accs)*100:.2f}%")

    # Method C: Flat TV-Regularized AdaMerging
    print("\nOptimizing Flat TV-Regularized AdaMerging...")
    coeffs_tv = (torch.ones((K, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_tv], lr=0.5)
    beta = 2.0

    for step in range(50):
        optimizer.zero_grad()
        m_p1 = get_merged_params(coeffs_tv, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"input_ids": tta_input_ids, "attention_mask": tta_attn_mask})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        loss_reg = 0.0
        for l in range(1, L):
            loss_reg += ((coeffs_tv[:, l] - coeffs_tv[:, l-1])**2).sum()
        loss_reg += 0.1 * ((coeffs_tv - 0.5)**2).sum()
        
        loss_joint = loss_tta + beta * loss_reg
        loss_joint.backward()
        optimizer.step()

    tv_accs = evaluate_model(coeffs_tv.detach())
    print(f"Flat TV-Regularized final coeffs:\n{coeffs_tv.detach()}")
    print(f"Flat TV-Regularized - Task 1 Acc: {tv_accs[0]*100:.2f}%, Task 2 Acc: {tv_accs[1]*100:.2f}%, Avg: {np.mean(tv_accs)*100:.2f}%")

    # Method D: RCR-Merge (Ours)
    print("\nOptimizing RCR-Merge (with spatial curvature-weighted TV)...")
    coeffs_rcr = (torch.ones((K, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_rcr], lr=0.5)
    beta = 2.0  # Regularization strength

    for step in range(50):
        optimizer.zero_grad()
        m_p1 = get_merged_params(coeffs_rcr, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"input_ids": tta_input_ids, "attention_mask": tta_attn_mask})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        # Riemannian Curvature-Weighted TV regularizer + anchoring penalty
        loss_reg = 0.0
        for l in range(1, L):
            geom_mean = torch.sqrt(c[l] * c[l-1])
            loss_reg += geom_mean * ((coeffs_rcr[:, l] - coeffs_rcr[:, l-1])**2).sum()
        
        # Anchoring penalty
        loss_reg += 0.1 * ((coeffs_rcr - 0.5)**2).sum()
        
        loss_joint = loss_tta + beta * loss_reg
        loss_joint.backward()
        optimizer.step()

    rcr_accs = evaluate_model(coeffs_rcr.detach())
    print(f"RCR-Merge final coeffs:\n{coeffs_rcr.detach()}")
    print(f"RCR-Merge (Ours) - Task 1 Acc: {rcr_accs[0]*100:.2f}%, Task 2 Acc: {rcr_accs[1]*100:.2f}%, Avg: {np.mean(rcr_accs)*100:.2f}%")

    # Print summary table
    print("\n" + "="*60)
    print("PILOT STUDY SUMMARY:")
    print("="*60)
    print(f"Method                     | Task 1 Acc | Task 2 Acc | Average Acc")
    print(f"Uniform Baseline (0.5)     | {uniform_accs[0]*100:9.2f}% | {uniform_accs[1]*100:9.2f}% | {np.mean(uniform_accs)*100:9.2f}%")
    print(f"Unconstrained AdaMerging   | {ada_accs[0]*100:9.2f}% | {ada_accs[1]*100:9.2f}% | {np.mean(ada_accs)*100:9.2f}%")
    print(f"PolyMerge (d=2)            | {poly_accs[0]*100:9.2f}% | {poly_accs[1]*100:9.2f}% | {np.mean(poly_accs)*100:9.2f}%")
    print(f"Flat TV-Regularized        | {tv_accs[0]*100:9.2f}% | {tv_accs[1]*100:9.2f}% | {np.mean(tv_accs)*100:9.2f}%")
    print(f"RCR-Merge (Ours)           | {rcr_accs[0]*100:9.2f}% | {rcr_accs[1]*100:9.2f}% | {np.mean(rcr_accs)*100:9.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
