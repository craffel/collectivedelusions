import torch
import timm
import numpy as np

# Standard NF4 quantization values from Dettmers et al. (QLoRA)
NF4_POINTS = torch.tensor([
    -1.0000, -0.6962, -0.5155, -0.3854, -0.2742, -0.1701, -0.0664,  0.0000,
     0.0811,  0.1701,  0.2585,  0.3552,  0.4682,  0.6105,  0.8105,  1.0000
])

def nf4_quantize_dequantize(W):
    # To simulate block-wise or tensor-wise NF4: we will do it per-tensor for simplicity, or per-row
    max_abs = torch.max(torch.abs(W))
    if max_abs < 1e-8:
        return W.clone()
    
    W_norm = W / max_abs
    
    # Reshape to 1D to find nearest points
    orig_shape = W_norm.shape
    W_flat = W_norm.view(-1)
    
    # Compute distances to all 16 NF4 points
    distances = torch.abs(W_flat.unsqueeze(1) - NF4_POINTS.to(W.device).unsqueeze(0)) # [N, 16]
    nearest_idx = torch.argmin(distances, dim=1)
    
    # Dequantize
    W_dq_norm = NF4_POINTS.to(W.device)[nearest_idx].view(orig_shape)
    return W_dq_norm * max_abs

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def round_ste(x):
    return RoundSTE.apply(x)

def quantize_dequantize_weight(W, bits, symmetric, per_channel):
    qmin = -(2**(bits - 1))
    qmax = (2**(bits - 1)) - 1
    
    if per_channel:
        dim = 1
        if symmetric:
            max_val = torch.max(torch.abs(W), dim=dim, keepdim=True)[0]
            max_val = torch.clamp(max_val, min=1e-8)
            scale = max_val / qmax
            q = torch.clamp(round_ste(W / scale), qmin, qmax)
            W_dq = q * scale
        else:
            min_val = torch.min(W, dim=dim, keepdim=True)[0]
            max_val = torch.max(W, dim=dim, keepdim=True)[0]
            diff = torch.clamp(max_val - min_val, min=1e-8)
            scale = diff / (qmax - qmin)
            zp = round_ste(-min_val / scale) + qmin
            zp = torch.clamp(zp, qmin, qmax)
            q = torch.clamp(round_ste(W / scale) + zp, qmin, qmax)
            W_dq = scale * (q - zp)
    else:
        if symmetric:
            max_val = torch.max(torch.abs(W))
            max_val = torch.clamp(max_val, min=1e-8)
            scale = max_val / qmax
            q = torch.clamp(round_ste(W / scale), qmin, qmax)
            W_dq = q * scale
        else:
            min_val = torch.min(W)
            max_val = torch.max(W)
            diff = torch.clamp(max_val - min_val, min=1e-8)
            scale = diff / (qmax - qmin)
            zp = round_ste(-min_val / scale) + qmin
            zp = torch.clamp(zp, qmin, qmax)
            q = torch.clamp(round_ste(W / scale) + zp, qmin, qmax)
            W_dq = scale * (q - zp)
            
    return W_dq

def measure_model(model_name):
    print(f"\nLoading model {model_name}...")
    model = timm.create_model(model_name, pretrained=True)
    
    # Collect all attention qkv layers
    qkv_weights = []
    # ViT has 12 blocks
    for l in range(12):
        qkv_weights.append(model.blocks[l].attn.qkv.weight.detach().clone())
        
    print(f"Loaded {len(qkv_weights)} layers of shape {qkv_weights[0].shape}")
    
    # We will test configurations:
    configs = [
        (8, True, True, "INT8 Symmetric Per-Channel"),
        (4, True, True, "INT4 Symmetric Per-Channel"),
        (4, False, True, "INT4 Asymmetric Per-Channel"),
        (4, True, False, "INT4 Symmetric Per-Tensor")
    ]
    
    print(f"=== RESULTS: {model_name} WEIGHT RECONSTRUCTION ERROR ===")
    print("| Configuration | Direct Quantization Error | Double Quantization Error (NF4 -> INT) | Increase |")
    print("|---|---|---|---|")
    
    for bits, sym, pc, config_name in configs:
        direct_errors = []
        double_errors = []
        for W in qkv_weights:
            # 1. Direct Quantization error
            W_direct = quantize_dequantize_weight(W, bits, sym, pc)
            err_direct = torch.norm(W - W_direct, p="fro") / torch.norm(W, p="fro")
            direct_errors.append(err_direct.item())
            
            # 2. Double Quantization error: FP32 -> NF4 -> INT
            W_nf4 = nf4_quantize_dequantize(W)
            W_double = quantize_dequantize_weight(W_nf4, bits, sym, pc)
            err_double = torch.norm(W - W_double, p="fro") / torch.norm(W, p="fro")
            double_errors.append(err_double.item())
            
        mean_direct = np.mean(direct_errors) * 100
        mean_double = np.mean(double_errors) * 100
        increase = mean_double - mean_direct
        print(f"| {config_name} | {mean_direct:.3f}% | {mean_double:.3f}% | +{increase:.3f}% |")

def main():
    measure_model("vit_tiny_patch16_224")
    measure_model("vit_base_patch16_224")

if __name__ == "__main__":
    main()
