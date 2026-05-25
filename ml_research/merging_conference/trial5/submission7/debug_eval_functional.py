import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from torch.func import functional_call

# Disable cuDNN
torch.backends.cudnn.enabled = False

def load_base_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    pretrained_conv1 = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(pretrained_conv1.mean(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def debug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load expert
    experts = {}
    experts['mnist'] = torch.load('experts/mnist_expert.pt', map_location='cpu')
    experts['fashion'] = torch.load('experts/fashion_expert.pt', map_location='cpu')
    experts['kmnist'] = torch.load('experts/kmnist_expert.pt', map_location='cpu')
    
    # Load clean MNIST test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    subset = Subset(mnist_test, list(range(32)))
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    merged_model = load_base_model().to(device)
    base_weights = load_base_model().state_dict()
    
    # Pre-move weights to device
    base_weights_dev = {k: v.to(device) for k, v in base_weights.items()}
    experts_dev = {
        'mnist': {k: v.to(device) for k, v in experts['mnist'].items()},
        'fashion': {k: v.to(device) for k, v in experts['fashion'].items()},
        'kmnist': {k: v.to(device) for k, v in experts['kmnist'].items()}
    }
    
    # parameters only (for gradient tracking)
    backbone_layers = [name for name, param in merged_model.named_parameters() if 'fc' not in name]
    # all float state dict keys (parameters + buffers)
    all_backbone_keys = [name for name, param in merged_model.state_dict().items() if 'fc' not in name and param.is_floating_point()]
    
    # Initialize coefficients lambdas
    lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in all_backbone_keys}
    
    # Run a forward pass
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    # Build blended params for the functional call
    blended_params = {}
    for name, param in base_weights_dev.items():
        if 'fc' not in name:
            if name in lambdas and param.is_floating_point():
                # Detach if it is a buffer (not in parameters backbone_layers)
                l = lambdas[name] if name in backbone_layers else lambdas[name].detach()
                v1 = experts_dev['mnist'][name] - base_weights_dev[name]
                v2 = experts_dev['fashion'][name] - base_weights_dev[name]
                v3 = experts_dev['kmnist'][name] - base_weights_dev[name]
                blended_params[name] = base_weights_dev[name] + l[0] * v1 + l[1] * v2 + l[2] * v3
            else:
                blended_params[name] = param
        else:
            blended_params[name] = experts_dev['mnist'][name]
            
    # Differentiable forward pass
    outputs = functional_call(merged_model, blended_params, images)
    
    probs = torch.softmax(outputs, dim=1)
    loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    
    # Compute gradients with respect to lambdas
    grads = torch.autograd.grad(loss, lambdas.values(), allow_unused=True)
    
    print("Functional call forward pass accuracy:")
    _, predicted = torch.max(outputs, 1)
    acc = 100 * (predicted == labels).sum().item() / labels.size(0)
    print(f"Accuracy: {acc:.2f}%")
    
    # Check if gradients are non-None
    non_none_grads = [name for name, grad in zip(backbone_layers, grads) if grad is not None]
    print(f"\nTotal backbone layers with gradients: {len(non_none_grads)} out of {len(backbone_layers)}")
    if len(non_none_grads) > 0:
        print("Sample gradient for conv1.weight:", grads[backbone_layers.index('conv1.weight')])

if __name__ == '__main__':
    debug()
