import torch
import torch.nn as nn

def test_deep_hns():
    torch.manual_seed(42)
    layers = 10
    channels = 64
    batch_size = 32
    size = 32

    # We will simulate 10 conv layers in sequence
    # Each expert has its own sequence of conv layers
    experts_seq_1 = []
    experts_seq_2 = []
    experts_seq_3 = []
    merged_seq = []
    hns_seq = []

    # Initialize shared base conv layers
    base_convs = [nn.Conv2d(channels, channels, 3, padding=1, bias=False) for _ in range(layers)]

    for i in range(layers):
        conv = base_convs[i]
        # Fine-tune with orthogonal drift
        drift1 = torch.randn_like(conv.weight) * 0.3
        drift2 = torch.randn_like(conv.weight) * 0.3
        drift3 = torch.randn_like(conv.weight) * 0.3

        w1 = conv.weight + drift1
        w2 = conv.weight + drift2
        w3 = conv.weight + drift3

        # Store expert 1 layers
        c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        c1.weight.data = w1
        experts_seq_1.append(c1)

        # Weight Averaging
        w_merged = (w1 + w2 + w3) / 3.0
        cm = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        cm.weight.data = w_merged
        merged_seq.append(cm)

        # Holographic Norm Scaling (HNS) for Expert 1
        # Compute channel-wise scaling factors
        norm_w1 = torch.norm(w1, p=2, dim=(1, 2, 3))
        norm_merged = torch.norm(w_merged, p=2, dim=(1, 2, 3))
        gamma1 = norm_w1 / (norm_merged + 1e-8)

        w1_hns = w_merged * gamma1.view(-1, 1, 1, 1)
        ch = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        ch.weight.data = w1_hns
        hns_seq.append(ch)

    # Let's run a random input batch through the layers and measure activation variance
    x_init = torch.randn(batch_size, channels, size, size)

    print("Propagating activations through 10 layers...")

    # Expert 1
    x = x_init.clone()
    for layer in experts_seq_1:
        x = torch.relu(layer(x))
    var_expert = torch.var(x).item()

    # Merged (WA)
    x = x_init.clone()
    for layer in merged_seq:
        x = torch.relu(layer(x))
    var_merged = torch.var(x).item()

    # HNS (Ours)
    x = x_init.clone()
    for layer in hns_seq:
        x = torch.relu(layer(x))
    var_hns = torch.var(x).item()

    print(f"Activation Variance at output (Layer {layers}):")
    print(f"Expert 1: {var_expert:.6f}")
    print(f"Merged (WA): {var_merged:.6f}  (Ratio of expert: {var_merged/var_expert*100:.2f}%)")
    print(f"HNS (Ours): {var_hns:.6f}  (Ratio of expert: {var_hns/var_expert*100:.2f}%)")

test_deep_hns()
