import os
from safetensors.torch import load_file

checkpoint_path = "checkpoints/cifar10_best"
weights_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
state_dict = load_file(weights_file)

print("Classifier keys:")
for k in state_dict.keys():
    if "classifier" in k:
        print(k)

print("\nFirst 5 other keys:")
other_keys = [k for k in state_dict.keys() if "classifier" not in k]
for k in other_keys[:5]:
    print(k)
