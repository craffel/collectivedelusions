import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import io

def apply_gaussian_noise(x, severity):
    # x: [C, H, W] tensor in [0, 1] (or normalized, we assume normalized or unnormalized)
    # We will work on unnormalized tensors (in [0, 1]) and normalize afterwards, or handle normalized
    # Let's assume input x is a tensor in [0, 1] range.
    std = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)

def apply_speckle_noise(x, severity):
    std = [0.10, 0.20, 0.35, 0.50, 0.75][severity - 1]
    noise = torch.randn_like(x) * std
    return torch.clamp(x * (1.0 + noise), 0.0, 1.0)

def apply_impulse_noise(x, severity):
    # Salt and pepper
    p = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    corrupted = x.clone()
    mask = torch.rand_like(x[0]) # 2D mask
    
    # Salt (1.0)
    salt_mask = mask < (p / 2.0)
    corrupted[:, salt_mask] = 1.0
    
    # Pepper (0.0)
    pepper_mask = (mask >= (p / 2.0)) & (mask < p)
    corrupted[:, pepper_mask] = 0.0
    
    return corrupted

def apply_gaussian_blur(x, severity):
    ksize = [3, 5, 5, 7, 9][severity - 1]
    sigma = [0.5, 1.0, 1.5, 2.0, 3.0][severity - 1]
    # Ensure ksize is odd
    if ksize % 2 == 0:
        ksize += 1
    return TF.gaussian_blur(x, [ksize, ksize], [sigma, sigma])

def apply_motion_blur(x, severity):
    # Simple linear motion blur using 2D convolution
    size = [3, 5, 7, 9, 11][severity - 1]
    kernel = torch.zeros((size, size))
    # Fill diagonal
    for i in range(size):
        kernel[i, i] = 1.0
    kernel = kernel / size
    # Reshape for conv2d: [out_channels, in_channels/groups, h, w]
    kernel = kernel.view(1, 1, size, size).repeat(x.size(0), 1, 1, 1)
    
    # Pad input
    padding = size // 2
    x_padded = F.pad(x.unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
    # Apply depthwise conv2d
    blurred = F.conv2d(x_padded, kernel, groups=x.size(0))
    return torch.clamp(blurred.squeeze(0), 0.0, 1.0)

def apply_contrast(x, severity):
    factor = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = torch.mean(x, dim=[1, 2], keepdim=True)
    return torch.clamp(mean + factor * (x - mean), 0.0, 1.0)

def apply_brightness(x, severity):
    shift = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    # We can randomly add or subtract, but let's subtract to simulate dimming
    return torch.clamp(x - shift, 0.0, 1.0)

def apply_pixelation(x, severity):
    size = [24, 18, 14, 10, 6][severity - 1]
    orig_shape = x.shape[1:] # (H, W)
    # Downsample
    downsampled = F.interpolate(x.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
    # Upsample
    upsampled = F.interpolate(downsampled, size=orig_shape, mode='nearest')
    return upsampled.squeeze(0)

def apply_glass_blur(x, severity):
    # Local pixel shuffling
    dist = [1, 1, 2, 2, 3][severity - 1]
    C, H, W = x.shape
    corrupted = x.clone()
    for _ in range(severity):
        for h in range(H):
            for w in range(W):
                # Random shift
                dh = np.random.randint(-dist, dist + 1)
                dw = np.random.randint(-dist, dist + 1)
                nh = min(max(h + dh, 0), H - 1)
                nw = min(max(w + dw, 0), W - 1)
                # Swap
                temp = corrupted[:, h, w].clone()
                corrupted[:, h, w] = corrupted[:, nh, nw]
                corrupted[:, nh, nw] = temp
    return corrupted

def apply_jpeg_compression(x, severity):
    quality = [80, 50, 30, 15, 5][severity - 1]
    # Convert PyTorch tensor to PIL Image
    pil_img = TF.to_pil_image(x)
    # Compress in-memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    # Convert back to tensor
    return TF.to_tensor(compressed_pil)

CORRUPTIONS = {
    "gaussian_noise": apply_gaussian_noise,
    "speckle_noise": apply_speckle_noise,
    "impulse_noise": apply_impulse_noise,
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "contrast": apply_contrast,
    "brightness": apply_brightness,
    "pixelation": apply_pixelation,
    "glass_blur": apply_glass_blur,
    "jpeg_compression": apply_jpeg_compression
}

def corrupt_dataset_batch(images, corruption_name, severity):
    # images is a PyTorch tensor of shape [B, C, H, W] in [0, 1] or normalized.
    # It is safer to un-normalize images, apply corruption, and re-normalize.
    # Let's assume input images are already normalized to ImageNet mean/std.
    # We will un-normalize them: x = x * std + mean
    # Then apply corruption, then re-normalize: x = (x - mean) / std
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    
    # Unnormalize to [0, 1] range
    unnormalized = torch.clamp(images * std + mean, 0.0, 1.0)
    
    # Apply corruption to each image in batch
    corrupted_list = []
    corruption_fn = CORRUPTIONS[corruption_name]
    for img in unnormalized:
        corrupted_img = corruption_fn(img, severity)
        corrupted_list.append(corrupted_img)
    
    corrupted_batch = torch.stack(corrupted_list).to(images.device)
    
    # Re-normalize
    normalized = (corrupted_batch - mean) / std
    return normalized
