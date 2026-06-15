import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import timm

# We can import the classes from run_experiments directly!
import run_experiments

print("Testing physical ViT experiment with larger fine-tuning data size...")

# Let's write a targeted function to run with custom train sizes and epochs
def run_custom_vit_experiment(train_size=128, epochs=15):
    print(f"\nRunning custom physical ViT experiment with train_size={train_size}, epochs={epochs}")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.eval()

    def to_rgb(img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    transform_gray = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_color = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mnist = dset.MNIST('./data', train=True, transform=transform_gray, download=True)
    fmnist = dset.FashionMNIST('./data', train=True, transform=transform_gray, download=True)
    cifar = dset.CIFAR10('./data', train=True, transform=transform_color, download=True)
    svhn = dset.SVHN('./data', split='train', transform=transform_color, download=True)

    # Subsets (custom train_size, 100 test per task)
    mnist_cal = torch.utils.data.Subset(mnist, range(train_size))
    mnist_test = torch.utils.data.Subset(mnist, range(train_size, train_size + 100))

    fmnist_cal = torch.utils.data.Subset(fmnist, range(train_size))
    fmnist_test = torch.utils.data.Subset(fmnist, range(train_size, train_size + 100))

    cifar_cal = torch.utils.data.Subset(cifar, range(train_size))
    cifar_test = torch.utils.data.Subset(cifar, range(train_size, train_size + 100))

    svhn_cal = torch.utils.data.Subset(svhn, range(train_size))
    svhn_test = torch.utils.data.Subset(svhn, range(train_size, train_size + 100))

    # Loaders
    mnist_cal_loader = torch.utils.data.DataLoader(mnist_cal, batch_size=32, shuffle=False)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

    fmnist_cal_loader = torch.utils.data.DataLoader(fmnist_cal, batch_size=32, shuffle=False)
    fmnist_test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=100, shuffle=False)

    cifar_cal_loader = torch.utils.data.DataLoader(cifar_cal, batch_size=32, shuffle=False)
    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=100, shuffle=False)

    svhn_cal_loader = torch.utils.data.DataLoader(svhn_cal, batch_size=32, shuffle=False)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=100, shuffle=False)

    # Extract centroids
    def extract_activations_local(loader):
        all_z = []
        with torch.no_grad():
            for imgs, labels in loader:
                x = model.patch_embed(imgs)
                x = model._pos_embed(x)
                x = model.norm_pre(x)
                x = model.blocks[0](x)
                x = model.blocks[1](x)
                z = x[:, 1:].mean(dim=1)
                all_z.append(z)
        return torch.cat(all_z, dim=0)

    print("Extracting calibration activations...")
    m_cal_z = extract_activations_local(mnist_cal_loader)
    f_cal_z = extract_activations_local(fmnist_cal_loader)
    c_cal_z = extract_activations_local(cifar_cal_loader)
    s_cal_z = extract_activations_local(svhn_cal_loader)

    centroids = torch.stack([
        m_cal_z.mean(dim=0),
        f_cal_z.mean(dim=0),
        c_cal_z.mean(dim=0),
        s_cal_z.mean(dim=0)
    ])
    centroids_norm = centroids / (torch.norm(centroids, dim=1, keepdim=True) + 1e-8)

    # Freeze all base parameters
    for p in model.parameters():
        p.requires_grad = False

    task_heads = run_experiments.TaskHeads()
    for l in range(2, 12):
        model.blocks[l].attn.qkv = run_experiments.LoRALinear(model.blocks[l].attn.qkv)

    lora_params = []
    for l in range(2, 12):
        lora_params.append(model.blocks[l].attn.qkv.lora_A)
        lora_params.append(model.blocks[l].attn.qkv.lora_B)

    optimizer = optim.AdamW([
        {'params': task_heads.parameters(), 'lr': 1.0e-2, 'weight_decay': 1.0e-2},
        {'params': lora_params, 'lr': 2.0e-3, 'weight_decay': 1.0e-2}
    ])
    criterion = nn.CrossEntropyLoss()
    loaders_cal = [mnist_cal_loader, fmnist_cal_loader, cifar_cal_loader, svhn_cal_loader]

    print("Fine-tuning LoRA and Heads on CPU...")
    t_start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        for k in range(4):
            loader = loaders_cal[k]
            run_experiments.set_active_expert(model, k=k)
            for imgs, labels in loader:
                feats = model.forward_features(imgs)
                pooled = model.forward_head(feats, pre_logits=True)
                logits = task_heads(pooled, k)
                loss = criterion(logits, labels)
                loss.backward()
                total_loss += loss.item()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    print(f"Fine-tuning completed in {time.time() - t_start:.2f} seconds.")

    # Evaluate Expert Ceiling, Uniform, and ELATI
    model.eval()
    task_heads.eval()
    loaders_test = [mnist_test_loader, fmnist_test_loader, cifar_test_loader, svhn_test_loader]

    expert_accs = []
    uniform_accs = []
    elati_accs = []

    with torch.no_grad():
        # Expert Ceiling
        for k in range(4):
            loader = loaders_test[k]
            run_experiments.set_active_expert(model, k=k)
            correct, total = 0, 0
            for imgs, labels in loader:
                feats = model.forward_features(imgs)
                pooled = model.forward_head(feats, pre_logits=True)
                logits = task_heads(pooled, k)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            expert_accs.append((correct / total) * 100.0)

        # Uniform
        for k in range(4):
            loader = loaders_test[k]
            run_experiments.set_active_expert(model, alphas=torch.tensor([0.25, 0.25, 0.25, 0.25]))
            correct, total = 0, 0
            for imgs, labels in loader:
                feats = model.forward_features(imgs)
                pooled = model.forward_head(feats, pre_logits=True)
                W_merged = sum(0.25 * task_heads.heads[j].weight for j in range(4))
                b_merged = sum(0.25 * task_heads.heads[j].bias for j in range(4))
                logits = F.linear(pooled, W_merged, b_merged)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            uniform_accs.append((correct / total) * 100.0)

        # ELATI
        for k in range(4):
            loader = loaders_test[k]
            correct, total = 0, 0
            for imgs, labels in loader:
                x = model.patch_embed(imgs)
                x = model._pos_embed(x)
                x = model.norm_pre(x)
                x = model.blocks[0](x)
                x = model.blocks[1](x)
                z2 = x[:, 1:].mean(dim=1)
                z2_norm = z2 / (torch.norm(z2, dim=1, keepdim=True) + 1e-8)
                u_b = z2_norm @ centroids_norm.t()
                alphas_b = torch.softmax(u_b / 0.05, dim=1)
                k_stars = torch.argmax(u_b, dim=1)

                final_logits = torch.zeros(imgs.size(0), 10, device=imgs.device)
                for g in range(4):
                    mask = (k_stars == g)
                    indices = torch.where(mask)[0]
                    if len(indices) == 0:
                        continue
                    alphas_g = alphas_b[indices].mean(dim=0)
                    run_experiments.set_active_expert(model, alphas=alphas_g)
                    x_g = x[indices]
                    for layer_idx in range(2, 12):
                        x_g = model.blocks[layer_idx](x_g)
                    x_g = model.norm(x_g)
                    pooled_g = model.forward_head(x_g, pre_logits=True)
                    W_merged_g = sum(alphas_g[j] * task_heads.heads[j].weight for j in range(4))
                    b_merged_g = sum(alphas_g[j] * task_heads.heads[j].bias for j in range(4))
                    logits_g = F.linear(pooled_g, W_merged_g, b_merged_g)
                    final_logits[indices] = logits_g

                preds = torch.argmax(final_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            elati_accs.append((correct / total) * 100.0)

    print("\nCustom experiment results:")
    print(f"Expert Ceiling: {expert_accs} -> Mean: {np.mean(expert_accs):.2f}%")
    print(f"Uniform Merging: {uniform_accs} -> Mean: {np.mean(uniform_accs):.2f}%")
    print(f"ELATI (Ours): {elati_accs} -> Mean: {np.mean(elati_accs):.2f}%")

run_custom_vit_experiment(train_size=256, epochs=10)
