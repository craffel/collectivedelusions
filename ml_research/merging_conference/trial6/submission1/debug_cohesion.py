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
    
    # Track v_bias running mean
    v_bias = torch.zeros(512, device=device)
    
    # Test across the stream
    for t in [0, 15, 30, 45, 60, 75]: # Tasks A, B, C
        inputs, targets, domain = stream[t]
        inputs = inputs.to(device)
        
        # Merge experts based on domain to see the "ideal" vs "uniform" states
        if domain == "cifar10":
            lambda_t = torch.tensor([1.0, 0.0, 0.0], device=device)
        elif domain == "svhn":
            lambda_t = torch.tensor([0.0, 1.0, 0.0], device=device)
        else: # fmnist
            lambda_t = torch.tensor([0.0, 0.0, 1.0], device=device)
            
        merged_model = merge_models_weight_space(experts, lambda_t)
        merged_model.eval()
        
        with torch.no_grad():
            outputs = merged_model(inputs)
            feats = merged_model.extract_features(inputs)
            
            # Update v_bias
            if t == 0:
                v_bias = feats.mean(dim=0)
            else:
                v_bias = 0.9 * v_bias + 0.1 * feats.mean(dim=0)
                
            feats_centered = feats - v_bias
            feats_norm = feats_centered / (feats_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            
            print(f"\n--- Batch {t+1} (Domain: {domain}) ---")
            print(f"Ideal lambda: {lambda_t.cpu().numpy()}")
            print(f"Avg Max Confidence: {conf.mean().item():.4f}")
            
            # Compute cohesion with known prototypes
            for k in range(2):
                proto_k = known_prototypes[k]
                proto_samples = proto_k[pseudo_labels]
                cos_sim = torch.sum(feats_norm * proto_samples, dim=1)
                print(f"Cohesion with Expert {k} Prototypes: {cos_sim.mean().item():.4f}")
                
if __name__ == "__main__":
    main()
