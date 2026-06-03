import torch
import torch.nn as nn
from torchvision.models import resnet18
from cpos_merging import CPOSResNet, QCPOSResNet, HCPOSResNet, GeneralizedCPOSResNet, ChannelWiseCPOSResNet, DPRCPOSResNet

def test_cpos_forward():
    print("Testing CPOS ResNet-18 compilation and forward pass...")
    # Initialize two dummy expert models
    model_A = resnet18(weights=None)
    model_A.fc = nn.Linear(512, 10)
    
    model_B = resnet18(weights=None)
    model_B.fc = nn.Linear(512, 10)
    
    # Initialize CPOS wrapper
    cpos_model = CPOSResNet(model_A, model_B, alpha=1.0, beta=1.0)
    cpos_model.eval()
    
    # Create random batch of inputs (batch size 4, 3 channels, 32x32)
    x = torch.randn(4, 3, 32, 32)
    
    # Test task 0 forward pass
    cpos_model.set_task(0)
    outputs_A = cpos_model(x)
    assert outputs_A.shape == (4, 10), f"Expected shape (4, 10), got {outputs_A.shape}"
    assert not torch.isnan(outputs_A).any(), "Found NaNs in outputs_A!"
    print("Task 0 forward pass: Success!")
    
    # Test task 1 forward pass
    cpos_model.set_task(1)
    outputs_B = cpos_model(x)
    assert outputs_B.shape == (4, 10), f"Expected shape (4, 10), got {outputs_B.shape}"
    assert not torch.isnan(outputs_B).any(), "Found NaNs in outputs_B!"
    print("Task 1 forward pass: Success!")
    
    print("All CPOS ResNet-18 compilation tests passed successfully!")


def test_qcpos_forward():
    print("Testing Q-CPOS ResNet-18 compilation and forward pass...")
    # Initialize three dummy expert models
    model_A = resnet18(weights=None)
    model_A.fc = nn.Linear(512, 10)
    
    model_B = resnet18(weights=None)
    model_B.fc = nn.Linear(512, 10)
    
    model_C = resnet18(weights=None)
    model_C.fc = nn.Linear(512, 10)
    
    # Initialize Q-CPOS wrapper
    qcpos_model = QCPOSResNet(model_A, model_B, model_C, w1=1.0, w2=1.0, w3=1.0)
    qcpos_model.eval()
    
    # Create random batch of inputs
    x = torch.randn(4, 3, 32, 32)
    
    for task_idx in [0, 1, 2]:
        qcpos_model.set_task(task_idx)
        outputs = qcpos_model(x)
        assert outputs.shape == (4, 10), f"Expected shape (4, 10) for task {task_idx}, got {outputs.shape}"
        assert not torch.isnan(outputs).any(), f"Found NaNs in outputs for task {task_idx}!"
        print(f"Task {task_idx} forward pass: Success!")
        
    print("All Q-CPOS ResNet-18 compilation tests passed successfully!")


def test_hcpos_forward():
    print("Testing H-CPOS ResNet-18 (Generalized) compilation and forward pass...")
    # Initialize 4 dummy expert models
    models = []
    for i in range(4):
        m = resnet18(weights=None)
        m.fc = nn.Linear(512, 10)
        models.append(m)
        
    # Test H-CPOS with 4 experts
    hcpos_model = HCPOSResNet(models, weights=[1.0, 1.0, 1.0, 1.0])
    hcpos_model.eval()
    
    # Create random batch of inputs
    x = torch.randn(4, 3, 32, 32)
    
    for task_idx in range(4):
        hcpos_model.set_task(task_idx)
        outputs = hcpos_model(x)
        assert outputs.shape == (4, 10), f"Expected shape (4, 10) for task {task_idx}, got {outputs.shape}"
        assert not torch.isnan(outputs).any(), f"Found NaNs in outputs for task {task_idx}!"
        print(f"Task {task_idx} forward pass: Success!")
        
    print("All H-CPOS ResNet-18 compilation tests passed successfully!")


def test_generalized_cpos_forward():
    print("Testing Generalized CPOS ResNet-18 compilation and forward pass...")
    model_A = resnet18(weights=None)
    model_A.fc = nn.Linear(512, 10)
    
    model_B = resnet18(weights=None)
    model_B.fc = nn.Linear(512, 10)
    
    # Test for multiple phase angles (0, pi/4, pi/2)
    for theta in [0.0, 0.7853, 1.5708]:
        g_model = GeneralizedCPOSResNet(model_A, model_B, alpha=0.707, beta=0.707, theta=theta)
        g_model.eval()
        
        x = torch.randn(4, 3, 32, 32)
        g_model.set_task(0)
        out_A = g_model(x)
        assert out_A.shape == (4, 10)
        assert not torch.isnan(out_A).any()
        
        g_model.set_task(1)
        out_B = g_model(x)
        assert out_B.shape == (4, 10)
        assert not torch.isnan(out_B).any()
        print(f"Forward pass for theta={theta:.4f}: Success!")
        
    print("All Generalized CPOS ResNet-18 compilation tests passed successfully!")


def test_channel_wise_cpos_forward():
    print("Testing Channel-Wise CPOS ResNet-18 compilation and forward pass...")
    model_A = resnet18(weights=None)
    model_A.fc = nn.Linear(512, 10)
    
    model_B = resnet18(weights=None)
    model_B.fc = nn.Linear(512, 10)
    
    # Test for multiple channel phase distributions
    for dist in ["binary", "linear", "sinusoidal", "random"]:
        cw_model = ChannelWiseCPOSResNet(model_A, model_B, alpha=0.707, beta=0.707, distribution=dist)
        cw_model.eval()
        
        x = torch.randn(4, 3, 32, 32)
        cw_model.set_task(0)
        out_A = cw_model(x)
        assert out_A.shape == (4, 10), f"Expected shape (4, 10), got {out_A.shape}"
        assert not torch.isnan(out_A).any(), f"NaNs in out_A for dist={dist}!"
        
        cw_model.set_task(1)
        out_B = cw_model(x)
        assert out_B.shape == (4, 10), f"Expected shape (4, 10), got {out_B.shape}"
        assert not torch.isnan(out_B).any(), f"NaNs in out_B for dist={dist}!"
        print(f"Forward pass for distribution={dist}: Success!")
        
    print("All Channel-Wise CPOS ResNet-18 compilation tests passed successfully!")


def test_dpr_cpos_forward():
    print("Testing DPR-CPOS ResNet-18 compilation and forward pass...")
    model_A = resnet18(weights=None)
    model_A.fc = nn.Linear(512, 10)
    
    model_B = resnet18(weights=None)
    model_B.fc = nn.Linear(512, 10)
    
    dpr_model = DPRCPOSResNet(model_A, model_B, alpha=0.707, beta=0.707)
    dpr_model.eval()
    
    x = torch.randn(4, 3, 32, 32)
    
    dpr_model.set_task(0)
    out_A = dpr_model(x)
    assert out_A.shape == (4, 10)
    assert not torch.isnan(out_A).any()
    
    dpr_model.set_task(1)
    out_B = dpr_model(x)
    assert out_B.shape == (4, 10)
    assert not torch.isnan(out_B).any()
    print("DPR-CPOS forward pass: Success!")


if __name__ == "__main__":
    test_cpos_forward()
    test_qcpos_forward()
    test_hcpos_forward()
    test_generalized_cpos_forward()
    test_channel_wise_cpos_forward()
    test_dpr_cpos_forward()
