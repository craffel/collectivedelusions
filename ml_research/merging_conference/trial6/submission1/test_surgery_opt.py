import torch
import time

def original_surgery(lambda_params, class_gradients, G, use_iggs, classes_present):
    projected_gradients = {c: {name: grad.clone() for name, grad in class_gradients[c].items()} for c in classes_present}
    if use_iggs:
        for i in range(len(classes_present)):
            for j in range(len(classes_present)):
                if i == j:
                    continue
                ca = classes_present[i]
                cb = classes_present[j]
                
                # Compute Riemannian inner product
                inner_prod = 0.0
                norm_b = 0.0
                for name in lambda_params.keys():
                    g_a_w = class_gradients[ca][name]
                    g_b_w = class_gradients[cb][name]
                    
                    dot_prod = torch.dot(g_a_w, g_b_w).item()
                    inner_prod += G[name] * dot_prod
                    norm_b += G[name] * torch.dot(g_b_w, g_b_w).item()
                    
                # If conflict, project
                if inner_prod < 0:
                    for name in lambda_params.keys():
                        g_a_w = projected_gradients[ca][name]
                        g_b_w = class_gradients[cb][name]
                        projected_gradients[ca][name] = g_a_w - (inner_prod / (norm_b + 1e-8)) * g_b_w
    return projected_gradients

def tensorized_surgery(lambda_params, class_gradients, G, use_iggs, classes_present, device):
    projected_gradients = {c: {name: grad.clone() for name, grad in class_gradients[c].items()} for c in classes_present}
    if use_iggs:
        # Pre-convert G dict to tensor
        keys = list(lambda_params.keys())
        G_tensor = torch.tensor([G[name] for name in keys], device=device)
        
        # Pre-stack gradients for each class present
        stacked_grads = {}
        for c in classes_present:
            stacked_grads[c] = torch.stack([class_gradients[c][name] for name in keys]) # shape: (num_layers, K)
            
        for i in range(len(classes_present)):
            for j in range(len(classes_present)):
                if i == j:
                    continue
                ca = classes_present[i]
                cb = classes_present[j]
                
                # Tensorized Riemannian inner product
                g_a = stacked_grads[ca]
                g_b = stacked_grads[cb]
                
                inner_prod = torch.sum(G_tensor * torch.sum(g_a * g_b, dim=1))
                norm_b = torch.sum(G_tensor * torch.sum(g_b * g_b, dim=1))
                
                if inner_prod.item() < 0:
                    for name in keys:
                        g_a_w = projected_gradients[ca][name]
                        g_b_w = class_gradients[cb][name]
                        projected_gradients[ca][name] = g_a_w - (inner_prod / (norm_b + 1e-8)) * g_b_w
    return projected_gradients

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")
    
    # Simulate ResNet-18 layer structure: 110 layers
    num_layers = 110
    K = 3
    lambda_params = {f"layer_{i}": torch.randn(K, device=device) for i in range(num_layers)}
    G = {f"layer_{i}": float(torch.rand(1).item()) for i in range(num_layers)}
    
    classes_present = list(range(10))
    class_gradients = {c: {name: torch.randn(K, device=device) for name in lambda_params.keys()} for c in classes_present}
    
    # Warmup
    _ = original_surgery(lambda_params, class_gradients, G, True, classes_present)
    _ = tensorized_surgery(lambda_params, class_gradients, G, True, classes_present, device)
    
    # Benchmark original
    t0 = time.time()
    for _ in range(20):
        orig_proj = original_surgery(lambda_params, class_gradients, G, True, classes_present)
    t1 = time.time()
    orig_time = (t1 - t0) / 20
    print(f"Original surgery time per call: {orig_time*1000:.2f} ms")
    
    # Benchmark tensorized
    t0 = time.time()
    for _ in range(20):
        tens_proj = tensorized_surgery(lambda_params, class_gradients, G, True, classes_present, device)
    t1 = time.time()
    tens_time = (t1 - t0) / 20
    print(f"Tensorized surgery time per call: {tens_time*1000:.2f} ms")
    print(f"Speedup: {orig_time / tens_time:.1f}x")
    
    # Verify correctness
    all_close = True
    for c in classes_present:
        for name in lambda_params.keys():
            diff = torch.max(torch.abs(orig_proj[c][name] - tens_proj[c][name])).item()
            if diff > 1e-5:
                print(f"Mismatch at class {c}, layer {name}: diff = {diff}")
                all_close = False
                break
    if all_close:
        print("Success! Both implementations produce mathematically identical results.")

if __name__ == "__main__":
    main()
