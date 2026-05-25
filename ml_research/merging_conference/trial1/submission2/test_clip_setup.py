import torch
import open_clip
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)

# Class names for MNIST
mnist_classes = [str(i) for i in range(10)]
templates = [f"a photo of the number {c}" for c in mnist_classes]

# Tokenize and encode
tokenizer = open_clip.get_tokenizer('ViT-B-32')
text_tokens = tokenizer(templates).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print("Text features shape:", text_features.shape)

# Load a single MNIST image
transform = Compose([
    Resize(224),
    lambda img: img.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in loader:
    images = images.to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("Predictions:", similarity.argmax(dim=-1))
        print("Labels:", labels)
    break
