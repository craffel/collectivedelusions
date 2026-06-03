import torch
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models, calibrate_running_stats

def main():
    device = torch.device('cpu')
    
    # Load expert
    chk = torch.load("./checkpoints/expert_mnist.pt", map_location=device)
    expert_state_dict = chk['state_dict']
    
    print("--- MNIST Expert bn1 stats ---")
    print("running_mean (first 5 channels):", expert_state_dict["backbone.bn1.running_mean"][:5])
    print("running_var (first 5 channels):", expert_state_dict["backbone.bn1.running_var"][:5])
    
    # Load progenitor
    progenitor = MultiTaskResNet18()
    progenitor_state_dict = {f"backbone.{k}": v.clone() for k, v in progenitor.backbone.state_dict().items()}
    
    # Let's see what happens to the merged backbone
    expert_state_dicts = {'mnist': expert_state_dict, 'fmnist': torch.load("./checkpoints/expert_fmnist.pt", map_location=device)['state_dict'], 'cifar10': torch.load("./checkpoints/expert_cifar10.pt", map_location=device)['state_dict']}
    
    merged_backbone = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type='WA', lam=0.5)
    
    print("\n--- Merged Model bn1 stats (Uncalibrated) ---")
    print("running_mean (first 5 channels):", merged_backbone["backbone.bn1.running_mean"][:5])
    print("running_var (first 5 channels):", merged_backbone["backbone.bn1.running_var"][:5])
    
    # Now let's calibrate the merged model on MNIST and check stats
    model = MultiTaskResNet18()
    model.load_state_dict(merged_backbone, strict=False)
    
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    train_mnist, _ = datasets_dict['mnist']
    indices = torch.randperm(len(train_mnist))[:256]
    real_samples = torch.stack([train_mnist[idx][0] for idx in indices], dim=0)
    
    # Print statistics of the input samples to make sure they are correct
    print("\n--- MNIST Input Sample stats ---")
    print("Mean of input samples:", real_samples.mean().item())
    print("Std of input samples:", real_samples.std().item())
    print("Shape:", real_samples.shape)
    
    # Run 1 forward pass in train mode to see batch stats
    model.train()
    with torch.no_grad():
        features = model.backbone(real_samples[:64])
    
    print("\n--- After 1 forward pass in train() mode on model (without resetting stats) ---")
    print("bn1 running_mean (first 5 channels):", model.backbone.bn1.running_mean[:5])
    print("bn1 running_var (first 5 channels):", model.backbone.bn1.running_var[:5])
    
    # Now reset and calibrate
    calibrate_running_stats(model, real_samples, epochs=10, device=device)
    
    print("\n--- After calibrate_running_stats ---")
    print("bn1 running_mean (first 5 channels):", model.backbone.bn1.running_mean[:5])
    print("bn1 running_var (first 5 channels):", model.backbone.bn1.running_var[:5])

if __name__ == '__main__':
    main()
