import torch
import timm
from download_datasets import main as pre_download

def test():
    print("Testing dataloader...")
    import torchvision.transforms as transforms
    import torchvision
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    images, labels = next(iter(loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    
    print("Testing model...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(192, 10)
    outputs = model(images)
    print("Outputs shape:", outputs.shape)
    
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    print("Backward pass successful. Grad of head weight:", model.head.weight.grad is not None)
    
    print("Test passed successfully!")

if __name__ == '__main__':
    test()
