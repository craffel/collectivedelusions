import os
import sys
from huggingface_hub import hf_hub_download

# Mapping of dataset names in the AdaMerging codebase to nik-dim/tall_masks repo dataset names
datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

base_dir = os.path.abspath('.')
checkpoint_dir = os.path.join(base_dir, 'checkpoints', 'ViT-B-32')

print("Starting asset downloads...")

# 1. Download pretrained zero-shot model
print("Downloading zero-shot base model...")
try:
    path = hf_hub_download(
        repo_id="nik-dim/tall_masks",
        filename="single_task/ViT-B-32/MNISTVal/nonlinear_zeroshot.pt",
        local_dir=base_dir,
        local_dir_use_symlinks=False
    )
    # Move to checkpoints/ViT-B-32/zeroshot.pt
    dest = os.path.join(checkpoint_dir, 'zeroshot.pt')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.rename(path, dest)
    print(f"Zero-shot model saved to {dest}")
except Exception as e:
    print(f"Error downloading zero-shot model: {e}")

# 2. Download task checkpoints and heads
for dataset in datasets:
    # Fine-tuned model checkpoint
    print(f"Downloading fine-tuned model for {dataset}...")
    try:
        path = hf_hub_download(
            repo_id="nik-dim/tall_masks",
            filename=f"single_task/ViT-B-32/{dataset}Val/nonlinear_finetuned.pt",
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        dest = os.path.join(checkpoint_dir, dataset, 'finetuned.pt')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(path, dest)
        print(f"Fine-tuned model for {dataset} saved to {dest}")
    except Exception as e:
        print(f"Error downloading {dataset} fine-tuned model: {e}")

    # Classification head
    print(f"Downloading classification head for {dataset}...")
    try:
        path = hf_hub_download(
            repo_id="nik-dim/tall_masks",
            filename=f"single_task/ViT-B-32/head_{dataset}Val.pt",
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        dest = os.path.join(checkpoint_dir, f'head_{dataset}.pt')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(path, dest)
        print(f"Classification head for {dataset} saved to {dest}")
    except Exception as e:
        print(f"Error downloading {dataset} classification head: {e}")

# Cleanup empty downloaded directories if any
import shutil
for folder in ['single_task', 'tall_masks']:
    if os.path.exists(folder):
        shutil.rmtree(folder)

print("All asset downloads completed successfully!")
