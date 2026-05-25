import torch
import torchvision
import torchvision.transforms as transforms

def test_contrast_shift_preserves_signal():
    """
    Asserts that our safe un-normalization/re-normalization contrast shift pipeline
    successfully preserves structural signal (retaining variance/std dev),
    whereas the naive direct contrast shift collapses the tensor and destroys the signal.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load actual MNIST test images from the workspace
    dataset = torchvision.datasets.MNIST(root="data", train=False, transform=transform, download=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    images_normalized, _ = next(iter(loader))
    
    # Standard normalization constants used in transform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    # Naive adjustment: Apply contrast adjustment directly on normalized images
    images_naive = transforms.functional.adjust_contrast(images_normalized, 0.3)
    
    # Safe adjustment: Un-normalize, adjust contrast, and re-normalize
    images_unnorm = images_normalized * std + mean
    images_unnorm_corr = transforms.functional.adjust_contrast(images_unnorm, 0.3)
    images_safe = (images_unnorm_corr - mean) / std
    
    # Compute correlation with the original normalized image
    flat_orig = images_normalized.reshape(-1)
    flat_naive = images_naive.reshape(-1)
    flat_safe = images_safe.reshape(-1)
    
    corr_naive = torch.corrcoef(torch.stack([flat_orig, flat_naive]))[0, 1].item()
    corr_safe = torch.corrcoef(torch.stack([flat_orig, flat_safe]))[0, 1].item()
    
    print(f"Naive Min: {images_naive.min().item():.4f} | Max: {images_naive.max().item():.4f}")
    print(f"Safe Min: {images_safe.min().item():.4f} | Max: {images_safe.max().item():.4f}")
    print(f"Correlation (Original vs Naive): {corr_naive:.6f}")
    print(f"Correlation (Original vs Safe): {corr_safe:.6f}")
    
    # Assertions
    assert images_naive.min().item() == 0.0, "Naive adjustment should clamp negative values to exactly 0.0"
    assert images_safe.min().item() < -1.0, "Safe adjustment should preserve the real normalized negative background (typically around -2.1)"
    assert corr_safe > 0.90, f"Safe contrast shift must maintain high correlation (actual: {corr_safe:.4f}) with original image structure."
    assert corr_naive < 0.30, f"Naive contrast shift should have extremely low correlation (actual: {corr_naive:.4f}) due to severe clamping distortion."
    
    print("All mathematical assertions passed successfully!")

if __name__ == "__main__":
    test_contrast_shift_preserves_signal()
