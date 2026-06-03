import torch
import time
import copy
from models import ResNet18CIFAR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")
    
    model = ResNet18CIFAR().to(device)
    
    # Benchmark copy.deepcopy
    t0 = time.time()
    for _ in range(50):
        m_copy = copy.deepcopy(model).to(device)
    t1 = time.time()
    print(f"copy.deepcopy (50 runs): {t1 - t0:.4f} seconds")
    
    # Benchmark type(model)().to(device) + load_state_dict
    t2 = time.time()
    for _ in range(50):
        m_copy = type(model)().to(device)
        m_copy.load_state_dict(model.state_dict())
    t3 = time.time()
    print(f"type(model)().to(device) + load_state_dict (50 runs): {t3 - t2:.4f} seconds")
    
    # Compare correctness
    m_copy = type(model)().to(device)
    m_copy.load_state_dict(model.state_dict())
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), m_copy.state_dict().items()):
        assert torch.allclose(v1, v2), f"Weights differ for key {k1}!"
    print("Correctness verified!")

if __name__ == "__main__":
    main()
