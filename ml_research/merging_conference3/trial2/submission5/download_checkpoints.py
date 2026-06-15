import os
from huggingface_hub import hf_hub_download

def download_file(repo_id, filename, local_path):
    print(f"Downloading {filename} to {local_path}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(local_path),
        local_dir_use_symlinks=False
    )
    # Rename downloaded file if the structure doesn't match exactly
    downloaded_name = os.path.basename(filename)
    dest_name = os.path.basename(local_path)
    if downloaded_name != dest_name:
        current_path = os.path.join(os.path.dirname(local_path), downloaded_name)
        target_path = os.path.join(os.path.dirname(local_path), dest_name)
        if os.path.exists(current_path):
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(current_path, target_path)

if __name__ == "__main__":
    repo_id = "nik-dim/tall_masks"
    
    # We want to download zeroshot base, and 4 tasks: MNIST, FashionMNIST, CIFAR10, SVHN
    downloads = [
        # Pretrained base
        ("single_task/ViT-B-32/MNISTVal/nonlinear_zeroshot.pt", "checkpoints/ViT-B-32/zeroshot.pt"),
        # MNIST
        ("single_task/ViT-B-32/MNISTVal/nonlinear_finetuned.pt", "checkpoints/ViT-B-32/MNIST/finetuned.pt"),
        ("single_task/ViT-B-32/head_MNISTVal.pt", "checkpoints/ViT-B-32/head_MNIST.pt"),
        # FashionMNIST
        ("single_task/ViT-B-32/FashionMNISTVal/nonlinear_finetuned.pt", "checkpoints/ViT-B-32/FashionMNIST/finetuned.pt"),
        ("single_task/ViT-B-32/head_FashionMNISTVal.pt", "checkpoints/ViT-B-32/head_FashionMNIST.pt"),
        # CIFAR10
        ("single_task/ViT-B-32/CIFAR10Val/nonlinear_finetuned.pt", "checkpoints/ViT-B-32/CIFAR10/finetuned.pt"),
        ("single_task/ViT-B-32/head_CIFAR10Val.pt", "checkpoints/ViT-B-32/head_CIFAR10.pt"),
        # SVHN
        ("single_task/ViT-B-32/SVHNVal/nonlinear_finetuned.pt", "checkpoints/ViT-B-32/SVHN/finetuned.pt"),
        ("single_task/ViT-B-32/head_SVHNVal.pt", "checkpoints/ViT-B-32/head_SVHN.pt"),
    ]
    
    for filename, local_path in downloads:
        download_file(repo_id, filename, local_path)
    
    print("All downloads complete!")
