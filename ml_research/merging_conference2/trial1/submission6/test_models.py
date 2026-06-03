import torch
from transformers import CLIPModel, CLIPVisionModel

def test():
    print("Loading base CLIP model...")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading fine-tuned MNIST vision model...")
    mnist_vision = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_mnist")
    
    print("Verifying state dict keys and shape...")
    base_sd = base_model.vision_model.state_dict()
    mnist_sd = mnist_vision.vision_model.state_dict()
    
    # Calculate task vector
    task_vector = {}
    for k in base_sd.keys():
        if k in mnist_sd:
            diff = mnist_sd[k] - base_sd[k]
            norm = torch.norm(diff.float())
            if norm > 0:
                print(f"Key: {k}, Diff Norm: {norm.item():.4f}")
                task_vector[k] = diff
                break # Just print one
                
    print("Successfully tested model loading!")

if __name__ == "__main__":
    test()
