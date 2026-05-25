import torch
import torchvision.models as models
import numpy as np

def analyze_checkpoints():
    print("Loading checkpoints...")
    state_0 = torch.load("checkpoints/model_base.pt", map_location="cpu")
    state_a_std = torch.load("checkpoints/model_a_standard.pt", map_location="cpu")
    state_b_std = torch.load("checkpoints/model_b_standard.pt", map_location="cpu")
    state_a_sam = torch.load("checkpoints/model_a_sam.pt", map_location="cpu")
    state_b_sam = torch.load("checkpoints/model_b_sam.pt", map_location="cpu")

    # Load ResNet-18 structure
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 10) # CIFAR-10

    std_distances_from_I = []
    std_distances_between_tasks = []
    
    sam_distances_from_I = []
    sam_distances_between_tasks = []

    layer_details = []

    for name, p_0 in model.named_parameters():
        if "weight" not in name or p_0.dim() < 2:
            continue
        
        w_0 = state_0[name]
        w_a_std = state_a_std[name]
        w_b_std = state_b_std[name]
        w_a_sam = state_a_sam[name]
        w_b_sam = state_b_sam[name]

        orig_shape = w_0.shape
        d_out = orig_shape[0]
        d_in = torch.numel(w_0) // d_out

        w_0_2d = w_0.view(d_out, d_in)
        w_a_std_2d = w_a_std.view(d_out, d_in)
        w_b_std_2d = w_b_std.view(d_out, d_in)
        w_a_sam_2d = w_a_sam.view(d_out, d_in)
        w_b_sam_2d = w_b_sam.view(d_out, d_in)

        I = torch.eye(d_out)

        # Helper to compute rotation R
        def get_R(w_i_2d, w_0_2d):
            M = w_i_2d @ w_0_2d.T
            U, S, V_T = torch.linalg.svd(M)
            return U @ V_T

        # Standard
        R_a_std = get_R(w_a_std_2d, w_0_2d)
        R_b_std = get_R(w_b_std_2d, w_0_2d)
        dist_a_I_std = torch.norm(R_a_std - I, p="fro").item()
        dist_b_I_std = torch.norm(R_b_std - I, p="fro").item()
        dist_ab_std = torch.norm(R_a_std - R_b_std, p="fro").item()

        # SAM
        R_a_sam = get_R(w_a_sam_2d, w_0_2d)
        R_b_sam = get_R(w_b_sam_2d, w_0_2d)
        dist_a_I_sam = torch.norm(R_a_sam - I, p="fro").item()
        dist_b_I_sam = torch.norm(R_b_sam - I, p="fro").item()
        dist_ab_sam = torch.norm(R_a_sam - R_b_sam, p="fro").item()

        std_distances_from_I.extend([dist_a_I_std, dist_b_I_std])
        std_distances_between_tasks.append(dist_ab_std)
        
        sam_distances_from_I.extend([dist_a_I_sam, dist_b_I_sam])
        sam_distances_between_tasks.append(dist_ab_sam)

        layer_details.append({
            "name": name,
            "shape": list(orig_shape),
            "std_dist_I": 0.5 * (dist_a_I_std + dist_b_I_std),
            "std_dist_ab": dist_ab_std,
            "sam_dist_I": 0.5 * (dist_a_I_sam + dist_b_I_sam),
            "sam_dist_ab": dist_ab_sam
        })

    print("\n=== SUMMARY GEOMETRICS ===")
    print(f"Standard SGD: Average R_i distance from I: {np.mean(std_distances_from_I):.6f}")
    print(f"Standard SGD: Average distance between Task A and Task B Rotations: {np.mean(std_distances_between_tasks):.6f}")
    print(f"SAM:          Average R_i distance from I: {np.mean(sam_distances_from_I):.6f}")
    print(f"SAM:          Average distance between Task A and Task B Rotations: {np.mean(sam_distances_between_tasks):.6f}")
    print("\n=== LAYER DETAILS ===")
    for details in layer_details:
        print(f"Layer: {details['name']} ({details['shape']})")
        print(f"  Standard: Dist-I={details['std_dist_I']:.4f}, Dist-AB={details['std_dist_ab']:.4f}")
        print(f"  SAM:      Dist-I={details['sam_dist_I']:.4f}, Dist-AB={details['sam_dist_ab']:.4f}")

if __name__ == "__main__":
    analyze_checkpoints()
