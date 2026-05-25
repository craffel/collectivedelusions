import torch
from torch.func import functional_call
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

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
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    model1 = get_resnet18_grayscale().to(device)
    model1.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
    
    # Let's test direct forward on model1
    model1.eval()
    with torch.no_grad():
        out1 = model1(inputs)
        acc1 = out1.max(1)[1].eq(targets).sum().item() / inputs.size(0)
        print(f"Direct model1 batch acc: {acc1*100:.2f}%")
        
    # Let's test functional_call on model1
    params = {name: param.clone() for name, param in model1.named_parameters()}
    buffers = {name: buf.clone() for name, buf in model1.named_buffers()}
    params_and_buffers = {**params, **buffers}
    
    with torch.no_grad():
        out2 = functional_call(model1, params_and_buffers, inputs)
        acc2 = out2.max(1)[1].eq(targets).sum().item() / inputs.size(0)
        print(f"Functional model1 batch acc: {acc2*100:.2f}%")
        
    # Let's test functional_call on a newly created template model
    model_template = get_resnet18_grayscale().to(device)
    with torch.no_grad():
        out3 = functional_call(model_template, params_and_buffers, inputs)
        acc3 = out3.max(1)[1].eq(targets).sum().item() / inputs.size(0)
        print(f"Functional model_template batch acc: {acc3*100:.2f}%")

    # TEST: Autograd-detached BN buffer merging verification
    print("\n--- Verifying Autograd-detached BN buffer merging ---")
    model2 = get_resnet18_grayscale().to(device)
    # create dummy parameters and buffers
    buf_name = "bn1.running_mean"
    model1_buf = torch.ones(64, device=device) * 2.0
    model2_buf = torch.ones(64, device=device) * 5.0
    
    # Coefficients with requires_grad=True
    l1 = torch.tensor(0.4, device=device, requires_grad=True)
    l2 = torch.tensor(0.6, device=device, requires_grad=True)
    
    # Perform autograd-detached merging
    l1_det, l2_det = l1.detach(), l2.detach()
    merged_buf = l1_det * model1_buf + l2_det * model2_buf
    
    # Assertions
    expected_val = 0.4 * 2.0 + 0.6 * 5.0 # 3.8
    assert torch.allclose(merged_buf, torch.ones(64, device=device) * expected_val), "Merged BN buffer values mismatch!"
    assert merged_buf.requires_grad == False, "Merged BN buffer should not have gradients!"
    print(f"SUCCESS: BN buffer merging correctly computed {merged_buf[0].item():.4f} and has requires_grad={merged_buf.requires_grad}")

if __name__ == "__main__":
    main()
