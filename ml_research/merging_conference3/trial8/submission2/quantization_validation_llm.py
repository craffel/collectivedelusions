import os
import numpy as np
import torch
import torch.nn as nn

def quantize_tensor(x, scale, bits=8):
    qmin = -(2 ** (bits - 1) - 1)
    qmax = 2 ** (bits - 1) - 1
    q_x = torch.round(torch.clamp(x / scale, qmin, qmax))
    return q_x

def dequantize_tensor(q_x, scale):
    return q_x * scale

class QuantizedLLMAdapterWrapper(nn.Module):
    def __init__(self, in_features=3072, out_features=3072, r=16, bits_w=4, bits_a=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.bits_w = bits_w
        self.bits_a = bits_a
        
        # Simulate a real pre-trained linear weight matrix using a normal distribution with decay
        # representing a realistic trained parameter spectrum
        torch.manual_seed(42)
        np.random.seed(42)
        
        W = torch.randn(out_features, in_features) * 0.02
        # Apply singular component decay to mimic realistic low-rank structure
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S_decay = S * torch.exp(-torch.arange(len(S)) / 500.0)
        W_decayed = torch.matmul(U, torch.matmul(torch.diag(S_decay), Vh))
        
        # Perform SVD to construct rank r low-rank adapters
        U_d, S_d, Vh_d = torch.linalg.svd(W_decayed, full_matrices=False)
        self.A = nn.Parameter(Vh_d[:r, :].t() * torch.sqrt(S_d[:r]), requires_grad=False) # (in_features, r)
        self.B = nn.Parameter((U_d[:, :r] * torch.sqrt(S_d[:r])).t(), requires_grad=False)  # (r, out_features)
        
        # Default RTN scales
        self.scale_A_rtn = self.A.abs().max() / 7.0
        self.scale_B_rtn = self.B.abs().max() / 7.0
        
        # Calibrated scales (filled during calibration)
        self.scale_A_calib = self.scale_A_rtn
        self.scale_B_calib = self.scale_B_rtn
        self.scale_h_calib = None
        
    def calibrate_qasc(self, X_calib):
        # 1. Optimize scale_A using decoupling on calibration data
        with torch.no_grad():
            H_calib_fp32 = torch.matmul(X_calib, self.A)
            scale_x_calib = X_calib.abs().max() / 127.0
            X_calib_q = quantize_tensor(X_calib, scale_x_calib, bits=self.bits_a)
            
        best_mse_A = float('inf')
        best_scale_A = self.scale_A_rtn
        
        for alpha_A in np.linspace(0.50, 1.10, 61):
            s_A = alpha_A * self.A.abs().max() / 7.0
            A_q = quantize_tensor(self.A, s_A, bits=self.bits_w)
            with torch.no_grad():
                H_q_accum = torch.matmul(X_calib_q, A_q)
                H_deq = H_q_accum * (scale_x_calib * s_A)
                mse = torch.mean((H_calib_fp32 - H_deq) ** 2)
                if mse < best_mse_A:
                    best_mse_A = mse
                    best_scale_A = s_A
                    
        self.scale_A_calib = best_scale_A
        
        # 2. Extract calibrated intermediate scale
        with torch.no_grad():
            A_q_opt = quantize_tensor(self.A, self.scale_A_calib, bits=self.bits_w)
            H_calib_q_accum = torch.matmul(X_calib_q, A_q_opt)
            self.scale_h_calib = H_calib_q_accum.abs().max() / 127.0
            H_calib_q = quantize_tensor(H_calib_q_accum, self.scale_h_calib, bits=self.bits_a)
            Y_calib_fp32 = torch.matmul(H_calib_fp32, self.B)
            
        # 3. Optimize scale_B
        best_mse_B = float('inf')
        best_scale_B = self.scale_B_rtn
        
        for alpha_B in np.linspace(0.50, 1.10, 61):
            s_B = alpha_B * self.B.abs().max() / 7.0
            B_q = quantize_tensor(self.B, s_B, bits=self.bits_w)
            with torch.no_grad():
                Y_q_accum = torch.matmul(H_calib_q, B_q)
                Y_deq = Y_q_accum * (scale_x_calib * self.scale_A_calib * self.scale_h_calib * s_B)
                mse = torch.mean((Y_calib_fp32 - Y_deq) ** 2)
                if mse < best_mse_B:
                    best_mse_B = mse
                    best_scale_B = s_B
                    
        self.scale_B_calib = best_scale_B

    def forward_eval(self, X, mode="float"):
        with torch.no_grad():
            # True Float32 Output
            H_fp32 = torch.matmul(X, self.A)
            Y_fp32 = torch.matmul(H_fp32, self.B)
            
            if mode == "float":
                return Y_fp32
                
            scale_x = X.abs().max() / 127.0
            X_q = quantize_tensor(X, scale_x, bits=self.bits_a)
            
            if mode == "rtn":
                # Standard uncalibrated RTN PTQ
                A_q = quantize_tensor(self.A, self.scale_A_rtn, bits=self.bits_w)
                H_q = torch.matmul(X_q, A_q)
                scale_h = H_q.abs().max() / 127.0
                H_q_quant = quantize_tensor(H_q, scale_h, bits=self.bits_a)
                B_q = quantize_tensor(self.B, self.scale_B_rtn, bits=self.bits_w)
                Y_q = torch.matmul(H_q_quant, B_q)
                Y_deq = Y_q * (scale_x * self.scale_A_rtn * scale_h * self.scale_B_rtn)
                return Y_deq
                
            elif mode == "minmax":
                # MinMax Static weight calibration
                # Optimized weights, but uncalibrated intermediate scale
                scale_A_mm = 0.8 * self.A.abs().max() / 7.0
                scale_B_mm = 0.8 * self.B.abs().max() / 7.0
                A_q = quantize_tensor(self.A, scale_A_mm, bits=self.bits_w)
                H_q = torch.matmul(X_q, A_q)
                scale_h = H_q.abs().max() / 127.0
                H_q_quant = quantize_tensor(H_q, scale_h, bits=self.bits_a)
                B_q = quantize_tensor(self.B, scale_B_mm, bits=self.bits_w)
                Y_q = torch.matmul(H_q_quant, B_q)
                Y_deq = Y_q * (scale_x * scale_A_mm * scale_h * scale_B_mm)
                return Y_deq
                
            elif mode == "qasc_dynamic":
                # Our QASC with dynamic intermediate scale
                A_q = quantize_tensor(self.A, self.scale_A_calib, bits=self.bits_w)
                H_q = torch.matmul(X_q, A_q)
                scale_h_dyn = H_q.abs().max() / 127.0
                H_q_quant = quantize_tensor(H_q, scale_h_dyn, bits=self.bits_a)
                B_q = quantize_tensor(self.B, self.scale_B_calib, bits=self.bits_w)
                Y_q = torch.matmul(H_q_quant, B_q)
                Y_deq = Y_q * (scale_x * self.scale_A_calib * scale_h_dyn * self.scale_B_calib)
                return Y_deq
                
            elif mode == "qasc_static":
                # Our QASC with hardcoded offline scale_h
                A_q = quantize_tensor(self.A, self.scale_A_calib, bits=self.bits_w)
                H_q = torch.matmul(X_q, A_q)
                H_q_quant = quantize_tensor(H_q, self.scale_h_calib, bits=self.bits_a)
                B_q = quantize_tensor(self.B, self.scale_B_calib, bits=self.bits_w)
                Y_q = torch.matmul(H_q_quant, B_q)
                Y_deq = Y_q * (scale_x * self.scale_A_calib * self.scale_h_calib * self.scale_B_calib)
                return Y_deq

def simulate_llm_activations(num_tokens=50000, features=3072, outlier_factor=40.0, num_outlier_channels=3):
    # Standard Gaussian activations
    torch.manual_seed(100)
    X = torch.randn(num_tokens, features) * 0.1
    
    # Inject heavy-tailed systematic outlier channels representing LLM "attention sinks" or "heavy channels"
    outlier_channels = torch.randperm(features)[:num_outlier_channels]
    for ch in outlier_channels:
        X[:, ch] = torch.randn(num_tokens) * outlier_factor * 0.1
        
    return X, outlier_channels

def main():
    print("=========================================================================")
    print("Executing Quantization-Aware Scale Calibration on Simulated LLM Weights")
    print("Layer Dimension: 3072 x 3072, LoRA Expert Rank: r = 16, Precision: INT4/INT8")
    print("=========================================================================")
    
    # 1. Create wrapper
    model = QuantizedLLMAdapterWrapper(in_features=3072, out_features=3072, r=16, bits_w=4, bits_a=8)
    
    # 2. Simulate heavy-tailed LLM token activations
    X, outlier_ch = simulate_llm_activations(num_tokens=50000, features=3072)
    print(f"Generated 50,000 tokens with {len(outlier_ch)} systemic outlier channels (outlier factor = 40.0).")
    print(f"Outlier Channels: {outlier_ch.tolist()}")
    print(f"Max absolute input value: {X.abs().max().item():.4f} (compared to typical channel std of 0.1)")
    
    # 3. Split into Calibration and Test
    X_calib = X[:10000] # 10,000 calibration tokens
    X_test = X[10000:]  # 40,000 test tokens
    
    # 4. Calibrate QASC
    print("Calibrating QASC offline over 10,000 tokens...")
    model.calibrate_qasc(X_calib)
    print("QASC Calibration complete.")
    print(f"Calibrated scale_A: {model.scale_A_calib.item():.6f}")
    print(f"Calibrated scale_B: {model.scale_B_calib.item():.6f}")
    print(f"Calibrated scale_h (static): {model.scale_h_calib.item():.6f}")
    
    # 5. Evaluate configurations on test split
    print("\nEvaluating on 40,000 test tokens...")
    Y_fp32 = model.forward_eval(X_test, mode="float")
    
    modes = ["rtn", "minmax", "qasc_dynamic", "qasc_static"]
    names = {
        "rtn": "RTN Uniform PTQ (Baseline)",
        "minmax": "MinMax Calibration",
        "qasc_dynamic": "QASC Dynamic Scaling (Ours)",
        "qasc_static": "QASC Static Scaling (Ours)"
    }
    
    results = {}
    for m in modes:
        Y_q = model.forward_eval(X_test, mode=m)
        relative_mse = torch.mean((Y_fp32 - Y_q) ** 2) / torch.mean(Y_fp32 ** 2)
        
        # Output Cosine Similarity
        numerator = torch.sum(Y_fp32 * Y_q, dim=-1)
        denominator = torch.sqrt(torch.sum(Y_fp32 ** 2, dim=-1)) * torch.sqrt(torch.sum(Y_q ** 2, dim=-1))
        cosine_sim = torch.mean(numerator / (denominator + 1e-8))
        
        results[m] = {
            "rel_mse": relative_mse.item() * 100.0,
            "cos_sim": cosine_sim.item()
        }
        print(f"{names[m]:32s} | Relative MSE: {results[m]['rel_mse']:6.4f}% | Cosine Similarity: {results[m]['cos_sim']:8.6f}")
        
    # Write results to disk
    with open("llm_weight_results.txt", "w") as f:
        f.write("=== Quantized LLM Adapter Validation Results ===\n")
        f.write(f"Layer Dimensions: 3072 x 3072, Rank: r = 16\n")
        f.write(f"Weights: INT4, Activations: INT8\n")
        f.write(f"Tokens: 40,000 test tokens with {len(outlier_ch)} systematic outlier channels\n")
        f.write("-------------------------------------------------------------------------\n")
        for m in modes:
            f.write(f"{names[m]:32s} | Relative MSE: {results[m]['rel_mse']:6.4f}% | Cosine Similarity: {results[m]['cos_sim']:8.6f}\n")
            
    print("\nResults successfully saved to llm_weight_results.txt.")

if __name__ == "__main__":
    main()
