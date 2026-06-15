import torch
import torchvision.models as models

# Load ViT-B/16
print("Loading ViT-B/16...")
model = models.vit_b_16(pretrained=True)
model.eval()

# Create dummy input of shape (B, C, H, W)
dummy_input = torch.randn(2, 3, 224, 224)

# Process inputs to get initial patch projection and CLS token
x = model._process_input(dummy_input)
print("Processed input shape (with patch + CLS token):", x.shape)

# Run through encoder layers
# ViT has 12 layers: 0 to 11
# Let's extract CLS token at layer 4 (after 4th encoder layer, i.e., index 3)
with torch.no_grad():
    # Layer 1
    x = model.encoder.layers[0](x)
    # Layer 2
    x = model.encoder.layers[1](x)
    # Layer 3
    x = model.encoder.layers[2](x)
    # Layer 4 (index 3)
    x = model.encoder.layers[3](x)
    layer4_cls = x[:, 0, :] # Extract [CLS] token representation
    print("Layer 4 CLS token shape:", layer4_cls.shape)
    
    # Run remaining layers (5..12, i.e., index 4..11)
    for i in range(4, 12):
        x = model.encoder.layers[i](x)
    
    # Final layer norm and head
    x = model.encoder.ln(x)
    final_cls = x[:, 0, :]
    out = model.heads(final_cls)
    print("Final output logits shape:", out.shape)
print("SUCCESS!")
