import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights

print("Testing dataset and model downloads...")

try:
    # Try downloading standard ResNet-18 weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("Successfully downloaded ResNet-18 pre-trained weights!")
except Exception as e:
    print(f"Error downloading ResNet-18 weights: {e}")

try:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # For MNIST, standard transforms might need 3 channels (Grayscale to RGB)
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading MNIST...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    print("Downloading FashionMNIST...")
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
    print("Downloading CIFAR10...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print("Successfully downloaded all datasets!")
except Exception as e:
    print(f"Error downloading datasets: {e}")
