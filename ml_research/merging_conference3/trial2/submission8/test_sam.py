import torch
import torchvision
import open_clip
import copy
from run_experiments import SAM, train_expert, get_cached_dataset, evaluate_model, get_text_head, TASK_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
base_model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

text_head = get_text_head(base_model, tokenizer, TASK_CLASSES["MNIST"])
train_dataset = get_cached_dataset("MNIST", "train", preprocess, 512, 42)
test_dataset = get_cached_dataset("MNIST", "test", preprocess, 512, 42)

acc_zs = evaluate_model(base_model, test_dataset, text_head)
print(f"Zero-shot MNIST Accuracy: {acc_zs*100:.2f}%")

for rho in [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005]:
    print(f"\nTraining with SAM, rho={rho}...")
    # Temporarily monkeypatch rho in train_expert by changing how SAM is initialized
    # We can write a custom train loop here or modify SAM initialization inside run_experiments.py
    # Let's write a targeted train expert here with the specific rho
    model = copy.deepcopy(base_model)
    target_params = []
    for name, param in model.named_parameters():
        if "visual.proj" in name or ("visual.transformer.resblocks" in name and ".attn." in name):
            param.requires_grad = True
            target_params.append(param)
        else:
            param.requires_grad = False
            
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(target_params, base_optimizer, rho=rho, lr=2e-5, weight_decay=1e-4)
    
    for epoch in range(3):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            # First step
            model.train()
            image_features = model.encode_image(images)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
            logits = 100.0 * image_features @ text_head.T
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second step
            image_features_2 = model.encode_image(images)
            image_features_2 = image_features_2 / (image_features_2.norm(dim=-1, keepdim=True) + 1e-12)
            logits_2 = 100.0 * image_features_2 @ text_head.T
            loss_2 = criterion(logits_2, targets)
            loss_2.backward()
            optimizer.second_step(zero_grad=True)
            
    # Evaluate
    temp_model = model
    acc = evaluate_model(temp_model, test_dataset, text_head)
    print(f"MNIST Accuracy with SAM (rho={rho}): {acc*100:.2f}%")
