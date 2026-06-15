import torch
import numpy as np
from run_experiments import model, test_x, test_y, test_task, get_quantized_zca_coefficients, K, num_classes

# Set scale alignment factors to 1.0
scale_alignment_ones = {l: [1.0]*K for l in range(1, 13)}

model.eval()
correct = 0
total = 0
B_size = 256

with torch.no_grad():
    num_batches = int(np.ceil(test_x.shape[0] / B_size))
    for b in range(num_batches):
        start = b * B_size
        end = min((b + 1) * B_size, test_x.shape[0])
        bx = test_x[start:end]
        by = test_y[start:end]
        btask = test_task[start:end]
        
        # Extact Layer 3 features with 4-bit quantization
        h = bx
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h = h @ W_dequant
            
        alpha = get_quantized_zca_coefficients(h, tau=0.001)
        
        # Forward pass with beta = 1.0
        logits = model(bx, task_idx=btask, alpha=alpha, scale_alignment=scale_alignment_ones, fake_quant_base_bit=4)
        preds = logits.argmax(dim=1)
        correct += (preds == by).sum().item()
        total += bx.shape[0]

accuracy_ones = correct / total * 100.0
print(f"SA-QAB Joint Accuracy with beta = 1.0: {accuracy_ones:.2f}%")
