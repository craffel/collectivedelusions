import os
import sys
import json
import torch

from experiment import (
    set_seed,
    get_dataloaders,
    get_base_model_and_params,
    extract_lora_and_head,
    run_tta,
    evaluate_merged_model
)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    _, cifar_test_loader, _, svhn_test_loader = get_dataloaders(
        subset_size=10000, batch_size=64, seed=42
    )
    
    if not os.path.exists("cifar10_best_model.pt") or not os.path.exists("svhn_best_model.pt"):
        print("Error: Trained expert models not found!")
        return
        
    cifar_state = torch.load("cifar10_best_model.pt", map_location="cpu")
    svhn_state = torch.load("svhn_best_model.pt", map_location="cpu")
    
    lora1, head1 = extract_lora_and_head(cifar_state)
    lora2, head2 = extract_lora_and_head(svhn_state)
    
    base_model, base_params, base_buffers = get_base_model_and_params(device=device)
    
    results = {}
    steps_tta = 100
    lr_tta = 0.02
    rho_sam = 0.05
    sosr_weight = 0.1
    
    # We will test three stream types
    stream_types = ["alternating", "sequential", "skewed"]
    
    for stream in stream_types:
        print(f"\n================== EVALUATING STREAM TYPE: {stream.upper()} ==================")
        results[stream] = {}
        
        # 1. Static Merged Baseline (same for all streams, as there is no adaptation)
        static_coeff = torch.tensor([0.5] * 24, device=device)
        static_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, static_coeff, cifar_test_loader, task_idx=1, device=device)
        static_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, static_coeff, svhn_test_loader, task_idx=2, device=device)
        static_avg = (static_c10 + static_svhn) / 2.0
        results[stream]["Static"] = {
            "cifar10": static_c10,
            "svhn": static_svhn,
            "avg": static_avg
        }
        print(f"Static | CIFAR-10: {static_c10:.2f}% | SVHN: {static_svhn:.2f}% | Avg: {static_avg:.2f}%")
        
        # 2. Standard TTA
        print("\nRunning Standard TTA...")
        std_coeff, std_h1, std_h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=steps_tta, batch_size=64, lr=lr_tta,
            use_sam=False, use_sosr=False, stream_type=stream, device=device
        )
        std_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, std_h1, lora2, std_h2, std_coeff, cifar_test_loader, task_idx=1, device=device)
        std_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, std_h1, lora2, std_h2, std_coeff, svhn_test_loader, task_idx=2, device=device)
        std_avg = (std_c10 + std_svhn) / 2.0
        results[stream]["Standard_TTA"] = {
            "cifar10": std_c10,
            "svhn": std_svhn,
            "avg": std_avg
        }
        print(f"Standard TTA | CIFAR-10: {std_c10:.2f}% | SVHN: {std_svhn:.2f}% | Avg: {std_avg:.2f}%")
        
        # 3. SAM-Only TTA
        print("\nRunning SAM-Only TTA...")
        sam_coeff, sam_h1, sam_h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=steps_tta, batch_size=64, lr=lr_tta,
            rho=rho_sam, use_sam=True, use_sosr=False, stream_type=stream, device=device
        )
        sam_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sam_h1, lora2, sam_h2, sam_coeff, cifar_test_loader, task_idx=1, device=device)
        sam_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sam_h1, lora2, sam_h2, sam_coeff, svhn_test_loader, task_idx=2, device=device)
        sam_avg = (sam_c10 + sam_svhn) / 2.0
        results[stream]["SAM_Only_TTA"] = {
            "cifar10": sam_c10,
            "svhn": sam_svhn,
            "avg": sam_avg
        }
        print(f"SAM-Only TTA | CIFAR-10: {sam_c10:.2f}% | SVHN: {sam_svhn:.2f}% | Avg: {sam_avg:.2f}%")
        
        # 4. SATA-TTA
        print("\nRunning SATA-TTA...")
        sata_coeff, sata_h1, sata_h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=steps_tta, batch_size=64, lr=lr_tta,
            rho=rho_sam, use_sam=True, use_sosr=True, sosr_weight=sosr_weight, stream_type=stream, device=device
        )
        sata_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sata_h1, lora2, sata_h2, sata_coeff, cifar_test_loader, task_idx=1, device=device)
        sata_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sata_h1, lora2, sata_h2, sata_coeff, svhn_test_loader, task_idx=2, device=device)
        sata_avg = (sata_c10 + sata_svhn) / 2.0
        results[stream]["SATA_TTA"] = {
            "cifar10": sata_c10,
            "svhn": sata_svhn,
            "avg": sata_avg
        }
        print(f"SATA-TTA | CIFAR-10: {sata_c10:.2f}% | SVHN: {sata_svhn:.2f}% | Avg: {sata_avg:.2f}%")

    with open("stream_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nStream evaluation completed! Results saved to stream_results.json.")

if __name__ == "__main__":
    main()
