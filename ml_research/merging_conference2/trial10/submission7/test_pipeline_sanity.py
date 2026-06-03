import torch
import torch.nn as nn
import torchvision.models as models
from utils import get_target_layers_mapping, apply_structured_pruning_mask, apply_weight_quantization

def test_target_layers_mapping():
    """
    Test that the target layer mapping returns a dict with expected keys and values.
    """
    mapping = get_target_layers_mapping()
    assert isinstance(mapping, dict), "Mapping must be a dictionary"
    assert "conv1" in mapping, "conv1 must be in mapping"
    assert "layer1.0.conv1" in mapping, "layer1.0.conv1 must be in mapping"
    
    # Check structure
    for layer, info in mapping.items():
        assert "bn" in info, f"bn missing in mapping for {layer}"
        assert "next_conv" in info, f"next_conv missing in mapping for {layer}"

def test_pruning_mask_application():
    """
    Test that applying a structured pruning mask zeroes out correct weight elements
    in the convolutional layer, the batchnorm layer, and the subsequent convolutional layer.
    """
    device = torch.device("cpu")
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    
    # Get a target layer, e.g., 'layer1.0.conv1'
    layer_name = 'layer1.0.conv1'
    mapping = get_target_layers_mapping()
    info = mapping[layer_name]
    
    conv_module = dict(model.named_modules())[layer_name]
    bn_module = dict(model.named_modules())[info['bn']]
    next_conv_module = dict(model.named_modules())[info['next_conv']]
    
    # Initialize weights to non-zero values
    nn.init.constant_(conv_module.weight, 1.0)
    if conv_module.bias is not None:
        nn.init.constant_(conv_module.bias, 1.0)
    nn.init.constant_(bn_module.weight, 1.0)
    nn.init.constant_(bn_module.bias, 1.0)
    nn.init.constant_(next_conv_module.weight, 1.0)
    
    # Apply a mask that prunes the first 16 channels out of 64
    out_channels = conv_module.weight.shape[0]
    mask = torch.ones(out_channels)
    mask[:16] = 0.0  # Prune first 16 channels
    
    apply_structured_pruning_mask(model, layer_name, mask)
    
    # Check that conv weights are zeroed for pruned channels
    assert (conv_module.weight[:16] == 0.0).all(), "Pruned output channels in conv weights not zeroed"
    assert (conv_module.weight[16:] == 1.0).all(), "Active output channels in conv weights were altered"
    
    # Check that BN weights and biases are zeroed for pruned channels
    assert (bn_module.weight[:16] == 0.0).all(), "Pruned channels in BN weights not zeroed"
    assert (bn_module.bias[:16] == 0.0).all(), "Pruned channels in BN biases not zeroed"
    
    # Check that next conv input channels are zeroed for pruned channels
    assert (next_conv_module.weight[:, :16] == 0.0).all(), "Pruned input channels in next conv weights not zeroed"
    assert (next_conv_module.weight[:, 16:] == 1.0).all(), "Active input channels in next conv weights were altered"

def test_quantization_application():
    """
    Test that fake weight quantization maps weight values onto simulated quantized steps.
    """
    device = torch.device("cpu")
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    
    # Let's initialize some weights in a layer
    conv_module = dict(model.named_modules())['layer1.0.conv1']
    with torch.no_grad():
        conv_module.weight.copy_(torch.linspace(-0.5, 0.5, conv_module.weight.numel()).view(conv_module.weight.shape))
        
    # Apply 8-bit weight quantization
    apply_weight_quantization(model, num_bits=8, per_channel=True)
    
    # Quantized weights should be within the original range but discrete
    max_val = conv_module.weight.max().item()
    min_val = conv_module.weight.min().item()
    assert abs(max_val) <= 0.501, "Quantization inflated range too much"
    assert abs(min_val) <= 0.501, "Quantization deflated range too much"

if __name__ == "__main__":
    test_target_layers_mapping()
    test_pruning_mask_application()
    test_quantization_application()
    print("All unit tests passed successfully!")
