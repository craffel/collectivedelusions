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
    
    # Test on a few batches
    for t in [0, 30, 60]: # CIFAR, SVHN, FMNIST
        inputs, targets, domain = stream[t]
        inputs = inputs.to(device)
        
        print(f"\n--- Batch {t+1} (Domain: {domain}) ---")
        
        # 1. Uniform Merged Model
        lambda_uniform = torch.tensor([1/3, 1/3, 1/3], device=device)
        merged_model = merge_models_weight_space(experts, lambda_uniform)
        merged_model.eval()
        
        with torch.no_grad():
            outputs = merged_model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            print(f"Uniform Merged Model: Avg Max Confidence: {conf.mean().item():.4f}")
            print(f"Uniform Merged Model: Num samples > 0.90 conf: {(conf > 0.90).sum().item()} / {len(conf)}")
            print(f"Uniform Merged Model: Num samples > 0.40 conf: {(conf > 0.40).sum().item()} / {len(conf)}")
            
            # Extract features and center
            feats = merged_model.extract_features(inputs)
            feats_norm = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # Check cohesion scores
            for k, proto_k in enumerate(known_prototypes):
                proto_samples = proto_k[pseudo_labels]
                cos_sim = torch.sum(feats_norm * proto_samples, dim=1)
                print(f"Expert {k} Cohesion Score: {cos_sim.mean().item():.4f}")
                
        # 2. Domain-Specific Experts
        for k, expert in enumerate(experts):
            expert.eval()
            with torch.no_grad():
                outputs = expert(inputs)
                probs = F.softmax(outputs, dim=1)
                conf, _ = probs.max(dim=1)
                print(f"Expert {k}: Avg Max Confidence: {conf.mean().item():.4f}")

if __name__ == "__main__":
    main()
