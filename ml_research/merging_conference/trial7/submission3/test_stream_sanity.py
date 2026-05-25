import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.func import functional_call

def get_resnet18_grayscale():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    batches = [next(iter(loader)) for _ in range(5)]
    
    model1 = get_resnet18_grayscale().to(device)
    model1.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
    
    model1.eval()
    
    # Check direct accuracy on these batches
    for idx, (inputs, targets) in enumerate(batches):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model1(inputs)
        acc = out.max(1)[1].eq(targets).sum().item() / inputs.size(0)
        print(f"Batch {idx+1} - direct acc: {acc*100:.2f}%")
        
    # Let's inspect what happens in run_t_adaptation.py
    # How are coefficients initialized?
    coefficients = {}
    model_template = get_resnet18_grayscale()
    for name, param in model_template.named_parameters():
        if param.requires_grad:
            # MNIST expert coefficients [1.0, 0.0]
            coefficients[name] = torch.tensor([1.0, 0.0], device=device)
            
    model1_params = {name: param.clone() for name, param in model1.named_parameters()}
    model2 = get_resnet18_grayscale().to(device)
    model2.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
    model2_params = {name: param.clone() for name, param in model2.named_parameters()}
    
    model1_buffers = {name: buf.clone() for name, buf in model1.named_buffers()}
    
    merged_model = get_resnet18_grayscale().to(device)
    
    # Simulate first batch step of run_t_adaptation.py
    inputs, targets = batches[0]
    inputs, targets = inputs.to(device), targets.to(device)
    
    merged_params = {}
    for name in model1_params:
        if name in coefficients:
            l1, l2 = coefficients[name][0], coefficients[name][1]
            merged_params[name] = l1 * model1_params[name] + l2 * model2_params[name]
        else:
            merged_params[name] = model1_params[name]
            
    merged_buffers = {}
    for name in model1_buffers:
        merged_buffers[name] = model1_buffers[name]
        
    features_list = []
    def hook_fn(module, input, output):
        features_list.append(torch.flatten(output, 1))
        
    hook = merged_model.avgpool.register_forward_hook(hook_fn)
    
    params_and_buffers = {**merged_params, **merged_buffers}
    logits = functional_call(merged_model, params_and_buffers, inputs)
    feats = features_list[0]
    hook.remove()
    
    acc = logits.max(1)[1].eq(targets).sum().item() / inputs.size(0)
    print(f"Simulated merged model (MNIST coefficients) batch acc: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
