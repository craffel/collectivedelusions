import torch
import torch.nn.functional as F
from models import get_resnet18_32x32, merge_models_weight_space
from utils import build_test_stream, get_calibration_loader, compute_expert_prototypes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    expert_cifar = get_resnet18_32x32().to(device)
    expert_svhn = get_resnet18_32x32().to(device)
    expert_fmnist = get_resnet18_32x32().to(device)
    
    def load_expert_weights(model, path, device):
        sd = torch.load(path, map_location=device)
        adapted_sd = {f"resnet.{k}": v for k, v in sd.items()}
        model.load_state_dict(adapted_sd)
        
    load_expert_weights(expert_cifar, "./checkpoints/expert_cifar10.pth", device)
    load_expert_weights(expert_svhn, "./checkpoints/expert_svhn.pth", device)
    load_expert_weights(expert_fmnist, "./checkpoints/expert_fmnist.pth", device)
    
    experts = [expert_cifar, expert_svhn, expert_fmnist]
    
    cal_loader_cifar = get_calibration_loader("cifar10", num_samples=256)
    cal_loader_svhn = get_calibration_loader("svhn", num_samples=256)
    proto_cifar = compute_expert_prototypes(expert_cifar, cal_loader_cifar, device)
    proto_svhn = compute_expert_prototypes(expert_svhn, cal_loader_svhn, device)
    known_prototypes = [proto_cifar, proto_svhn]
    
    stream = build_test_stream(batch_size=64, num_batches_per_domain=30)
    
    # 1. SVHN expert running on FashionMNIST inputs (Simulates Batch 61 arrival!)
    inputs, targets, domain = stream[60] # Batch 61 (FashionMNIST)
    inputs = inputs.to(device)
    
    # Merged model is Expert 1 (SVHN)
    lambda_svhn = torch.tensor([0.0, 1.0, 0.0], device=device)
    merged_model = merge_models_weight_space(experts, lambda_svhn)
    merged_model.eval()
    
    # Running with different confidence thresholds
    for tau in [0.0, 0.50, 0.80, 0.90]:
        print(f"\n--- Evaluation with tau_conf = {tau} ---")
        with torch.no_grad():
            outputs = merged_model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            
            mask = conf > tau
            num_selected = mask.sum().item()
            print(f"Num samples selected: {num_selected} / {len(conf)}")
            
            if num_selected > 0:
                # Extract features of selected samples
                feats = merged_model.extract_features(inputs)
                
                # Update running mean of features (simulated running mean)
                v_bias = feats.mean(dim=0)
                feats_centered = feats - v_bias
                feats_norm = feats_centered / (feats_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
                filtered_feats = feats_norm[mask]
                filtered_pseudo = pseudo_labels[mask]
                
                # Check cohesion scores
                for k in range(2):
                    proto_k = known_prototypes[k]
                    proto_samples = proto_k[filtered_pseudo]
                    cos_sim = torch.sum(filtered_feats * proto_samples, dim=1)
                    print(f"Expert {k} Cohesion Score: {cos_sim.mean().item():.4f}")
            else:
                print("No samples passed the threshold!")

if __name__ == "__main__":
    main()
