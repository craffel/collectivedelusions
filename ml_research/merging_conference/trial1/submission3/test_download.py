import torchvision
import torchvision.transforms as transforms

print("Attempting to download CIFAR-10 dataset...")
try:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    print(f"CIFAR-10 downloaded successfully! Size: {len(trainset)}")
except Exception as e:
    print("Error downloading CIFAR-10:", e)
