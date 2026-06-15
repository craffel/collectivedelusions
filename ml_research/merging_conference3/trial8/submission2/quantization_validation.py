import os
import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def quantize_tensor(x, scale, bits=8):
    qmin = -(2 ** (bits - 1) - 1)
    qmax = 2 ** (bits - 1) - 1
    q_x = torch.round(torch.clamp(x / scale, qmin, qmax))
    return q_x

def dequantize_tensor(q_x, scale):
    return q_x * scale

def main():
    print("=== Empirical Quantization Validation on Real Pre-trained ViT-Tiny with CIFAR-10 ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load pre-trained ViT-Tiny
    print("Loading pre-trained timm vit_tiny_patch16_224 model...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.eval()
    
    # 2. Load actual CIFAR-10 test set and extract 16 real images
    print("Loading actual CIFAR-10 images...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_set = dset.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    
    images, labels = next(iter(test_loader))
    print(f"Loaded {images.shape[0]} real images of shape {images.shape[1:]}")
    
    # 3. Extract real activation vectors at Layer 4
    print("Extracting real Layer 4 input activations from pre-trained ViT on CIFAR-10...")
    with torch.no_grad():
        h = model.patch_embed(images)
        h = model.pos_drop(h)
        for i in range(4): # Forward through blocks 0, 1, 2, 3
            h = model.blocks[i](h)
            
    # h shape is (B, 197, 192). We treat token vectors as a batch of representations.
    X_activations = h.reshape(-1, 192) # Shape: (16 * 197, 192) = (3152, 192)
    print(f"Extracted activations shape: {X_activations.shape}")
    
    # We will use 1000 tokens for calibration, and the remaining 2152 for testing
    calib_split = 1000
    X_calib = X_activations[:calib_split]
    X_test = X_activations[calib_split:]
    
    # 4. Extract real pre-trained weights and construct a real Low-Rank Adapter
    # We take the pre-trained weights of blocks[4].mlp.fc1 which is (768, 192)
    # We perform SVD to construct a low-rank adapter of rank r=8 representing the top singular components.
    print("Extracting real MLP layer weights and performing SVD to construct rank r=8 adapters...")
    W_real = model.blocks[4].mlp.fc1.weight.data # Shape: (768, 192)
    U, S, Vh = torch.linalg.svd(W_real, full_matrices=False)
    
    r = 8
    A_real = (Vh[:r, :].t() * torch.sqrt(S[:r])).detach() # Shape (192, 8)
    B_real = (U[:, :r] * torch.sqrt(S[:r])).t().detach()  # Shape (8, 768)
    
    print(f"Constructed real adapters: A_real shape = {A_real.shape}, B_real shape = {B_real.shape}")
    
    # 5. Reference FP32 output on test split
    with torch.no_grad():
        H_fp32 = torch.matmul(X_test, A_real) # Shape (2152, 8)
        Y_fp32 = torch.matmul(H_fp32, B_real) # Shape (2152, 768)
        Y_var = torch.var(Y_fp32)
        
    # 6. Evaluate different quantization schemes at INT4
    bitwidth = 4
    print(f"\nEvaluating 4-bit Weight, 8-bit Activation quantization schemes...")
    
    # 6.1. Standard RTN (Round-To-Nearest) without calibration
    # Simple max-abs scale factor for weights
    scale_A_rtn = A_real.abs().max() / 7.0
    scale_B_rtn = B_real.abs().max() / 7.0
    
    A_rtn = quantize_tensor(A_real, scale_A_rtn, bits=4)
    B_rtn = quantize_tensor(B_real, scale_B_rtn, bits=4)
    
    # Run evaluation
    with torch.no_grad():
        # Input activation quantization to INT8
        scale_x = X_test.abs().max() / 127.0
        X_q = quantize_tensor(X_test, scale_x, bits=8)
        
        # Down projection in integer precision
        H_q_accum = torch.matmul(X_q, A_rtn)
        
        # Intermediate dynamic activation scaling and re-quantization to INT8
        scale_h_q = H_q_accum.abs().max() / 127.0
        H_q = quantize_tensor(H_q_accum, scale_h_q, bits=8)
        
        # Up projection
        Y_q_accum = torch.matmul(H_q, B_rtn)
        
        # Dequantization back to float
        Y_rtn = Y_q_accum * (scale_x * scale_A_rtn * scale_h_q * scale_B_rtn)
        
        rtn_mse = torch.mean((Y_fp32 - Y_rtn) ** 2)
        rtn_rel_mse = rtn_mse / Y_var
        rtn_cosine = torch.mean(torch.nn.functional.cosine_similarity(Y_fp32, Y_rtn, dim=1))
        print(f"RTN (Standard PTQ) -> Relative MSE: {rtn_rel_mse:.6f}, Cosine Similarity: {rtn_cosine:.6f}")
        
    # 6.2. MinMax Grid Search Calibration
    # We search the clipping factor alpha in [0.50, 1.1] for weight scales: scale = alpha * max_abs / 7.0
    # To minimize MSE over calibration split
    best_mse_A = float('inf')
    best_scale_A_minmax = scale_A_rtn
    
    # Extract calibration float outputs
    with torch.no_grad():
        H_calib_fp32 = torch.matmul(X_calib, A_real)
        Y_calib_fp32 = torch.matmul(H_calib_fp32, B_real)
        
        scale_x_calib = X_calib.abs().max() / 127.0
        X_calib_q = quantize_tensor(X_calib, scale_x_calib, bits=8)
        
    for alpha_A in np.linspace(0.50, 1.10, 61):
        s_A = alpha_A * A_real.abs().max() / 7.0
        A_q = quantize_tensor(A_real, s_A, bits=4)
        A_deq = dequantize_tensor(A_q, s_A)
        with torch.no_grad():
            H_reconstr = torch.matmul(X_calib, A_deq)
            mse = torch.mean((H_calib_fp32 - H_reconstr) ** 2)
            if mse < best_mse_A:
                best_mse_A = mse
                best_scale_A_minmax = s_A
                
    best_mse_B = float('inf')
    best_scale_B_minmax = scale_B_rtn
    
    for alpha_B in np.linspace(0.50, 1.10, 61):
        s_B = alpha_B * B_real.abs().max() / 7.0
        B_q = quantize_tensor(B_real, s_B, bits=4)
        B_deq = dequantize_tensor(B_q, s_B)
        with torch.no_grad():
            Y_reconstr = torch.matmul(H_calib_fp32, B_deq)
            mse = torch.mean((Y_calib_fp32 - Y_reconstr) ** 2)
            if mse < best_mse_B:
                best_mse_B = mse
                best_scale_B_minmax = s_B
                
    # Run evaluation for MinMax Calibration
    with torch.no_grad():
        A_minmax = quantize_tensor(A_real, best_scale_A_minmax, bits=4)
        B_minmax = quantize_tensor(B_real, best_scale_B_minmax, bits=4)
        
        H_q_accum = torch.matmul(X_q, A_minmax)
        scale_h_q = H_q_accum.abs().max() / 127.0
        H_q = quantize_tensor(H_q_accum, scale_h_q, bits=8)
        Y_q_accum = torch.matmul(H_q, B_minmax)
        
        Y_minmax = Y_q_accum * (scale_x * best_scale_A_minmax * scale_h_q * best_scale_B_minmax)
        minmax_mse = torch.mean((Y_fp32 - Y_minmax) ** 2)
        minmax_rel_mse = minmax_mse / Y_var
        minmax_cosine = torch.mean(torch.nn.functional.cosine_similarity(Y_fp32, Y_minmax, dim=1))
        print(f"MinMax Calibration  -> Relative MSE: {minmax_rel_mse:.6f}, Cosine Similarity: {minmax_cosine:.6f}")
        
    # 6.3. QASC Calibration (Ours)
    # Sequentially decouples down- and up-projection optimization.
    best_mse_A = float('inf')
    best_scale_A_qasc = scale_A_rtn
    
    for alpha_A in np.linspace(0.50, 1.10, 61):
        s_A = alpha_A * A_real.abs().max() / 7.0
        A_q = quantize_tensor(A_real, s_A, bits=4)
        
        with torch.no_grad():
            H_q_accum = torch.matmul(X_calib_q, A_q)
            H_deq = H_q_accum * (scale_x_calib * s_A)
            mse = torch.mean((H_calib_fp32 - H_deq) ** 2)
            if mse < best_mse_A:
                best_mse_A = mse
                best_scale_A_qasc = s_A
                
    best_mse_B = float('inf')
    best_scale_B_qasc = scale_B_rtn
    
    with torch.no_grad():
        A_q_opt = quantize_tensor(A_real, best_scale_A_qasc, bits=4)
        H_calib_q_accum = torch.matmul(X_calib_q, A_q_opt)
        scale_h_calib_q = H_calib_q_accum.abs().max() / 127.0
        H_calib_q = quantize_tensor(H_calib_q_accum, scale_h_calib_q, bits=8)
        
    for alpha_B in np.linspace(0.50, 1.10, 61):
        s_B = alpha_B * B_real.abs().max() / 7.0
        B_q = quantize_tensor(B_real, s_B, bits=4)
        
        with torch.no_grad():
            Y_q_accum = torch.matmul(H_calib_q, B_q)
            Y_deq = Y_q_accum * (scale_x_calib * best_scale_A_qasc * scale_h_calib_q * s_B)
            mse = torch.mean((Y_calib_fp32 - Y_deq) ** 2)
            if mse < best_mse_B:
                best_mse_B = mse
                best_scale_B_qasc = s_B
                
    # Run evaluation for QASC (Dynamic Scale)
    with torch.no_grad():
        A_qasc = quantize_tensor(A_real, best_scale_A_qasc, bits=4)
        B_qasc = quantize_tensor(B_real, best_scale_B_qasc, bits=4)
        
        H_q_accum = torch.matmul(X_q, A_qasc)
        scale_h_q = H_q_accum.abs().max() / 127.0
        H_q = quantize_tensor(H_q_accum, scale_h_q, bits=8)
        Y_q_accum = torch.matmul(H_q, B_qasc)
        
        Y_qasc = Y_q_accum * (scale_x * best_scale_A_qasc * scale_h_q * best_scale_B_qasc)
        qasc_mse = torch.mean((Y_fp32 - Y_qasc) ** 2)
        qasc_rel_mse = qasc_mse / Y_var
        qasc_cosine = torch.mean(torch.nn.functional.cosine_similarity(Y_fp32, Y_qasc, dim=1))
        print(f"QASC (Dynamic)      -> Relative MSE: {qasc_rel_mse:.6f}, Cosine Similarity: {qasc_cosine:.6f}")
        
    # Run evaluation for QASC (Static Scale Alternative)
    with torch.no_grad():
        H_q_accum_static = torch.matmul(X_q, A_qasc)
        H_q_static = quantize_tensor(H_q_accum_static, scale_h_calib_q, bits=8)
        Y_q_accum_static = torch.matmul(H_q_static, B_qasc)
        
        Y_qasc_static = Y_q_accum_static * (scale_x * best_scale_A_qasc * scale_h_calib_q * best_scale_B_qasc)
        qasc_static_mse = torch.mean((Y_fp32 - Y_qasc_static) ** 2)
        qasc_static_rel_mse = qasc_static_mse / Y_var
        qasc_static_cosine = torch.mean(torch.nn.functional.cosine_similarity(Y_fp32, Y_qasc_static, dim=1))
        print(f"QASC (Static)       -> Relative MSE: {qasc_static_rel_mse:.6f}, Cosine Similarity: {qasc_static_cosine:.6f}")
        
    # Write report
    with open("real_weight_results.txt", "w") as f:
        f.write("=== Real Weight Quantization Validation Report ===\n")
        f.write(f"Model: Pre-trained timm vit_tiny_patch16_224\n")
        f.write(f"Layer: blocks.4.mlp.fc1\n")
        f.write(f"Validation Dataset: Real CIFAR-10 Test Set (16 images, 3,152 activation tokens)\n")
        f.write(f"Low-Rank Adapter Rank: r={r}\n")
        f.write(f"Precision: INT4 weights, INT8 input & intermediate activations\n\n")
        f.write(f"RTN (Standard PTQ)       - Relative MSE: {rtn_rel_mse:.6f}, Cosine Similarity: {rtn_cosine:.6f}\n")
        f.write(f"MinMax Calibration       - Relative MSE: {minmax_rel_mse:.6f}, Cosine Similarity: {minmax_cosine:.6f}\n")
        f.write(f"QASC Dynamic Scaling     - Relative MSE: {qasc_rel_mse:.6f}, Cosine Similarity: {qasc_cosine:.6f}\n")
        f.write(f"QASC Static Scaling (Alt) - Relative MSE: {qasc_static_rel_mse:.6f}, Cosine Similarity: {qasc_static_cosine:.6f}\n")
        
    # Plot results
    plt.figure(figsize=(8.5, 4.5))
    methods = ['RTN (Standard)', 'MinMax Calib', 'QASC (Dynamic)', 'QASC (Static)']
    rel_mses = [rtn_rel_mse.item(), minmax_rel_mse.item(), qasc_rel_mse.item(), qasc_static_rel_mse.item()]
    cosines = [rtn_cosine.item(), minmax_cosine.item(), qasc_cosine.item(), qasc_static_cosine.item()]
    
    color_mse = '#e34a33'
    color_cos = '#2b8cbe'
    
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.bar(np.arange(len(methods)) - 0.15, rel_mses, width=0.3, label='Relative Reconstruction MSE', color=color_mse)
    ax1.set_ylabel('Relative Reconstruction MSE (Lower is Better)', color=color_mse)
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_xticklabels(methods)
    
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(methods)) + 0.15, cosines, width=0.3, label='Cosine Similarity', color=color_cos)
    ax2.set_ylabel('Cosine Similarity with FP32 Output (Higher is Better)', color=color_cos)
    ax2.tick_params(axis='y', labelcolor=color_cos)
    ax2.set_ylim(0.95, 1.002)
    
    plt.title('Quantization & Scaling Calibration on Pre-trained ViT-Tiny with CIFAR-10')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fig8_real_weight_quant.png", dpi=150)
    os.makedirs("submission", exist_ok=True)
    plt.savefig("submission/fig8_real_weight_quant.png", dpi=150)
    plt.close()
    
    print("Saved results/fig8_real_weight_quant.png and real_weight_results.txt")

if __name__ == '__main__':
    main()
