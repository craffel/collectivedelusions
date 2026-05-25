import torch
import torch.nn as nn
from torchvision.models import resnet18
import time

torch.backends.cudnn.enabled = False

def profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    
    # Batch size
    batch_size = 64
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # 1. Static Merging Profile
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load single model
    model = resnet18()
    model.fc = nn.Identity()
    model.eval().to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(inputs)
        
    start_time = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = model(inputs)
    static_latency = (time.time() - start_time) / 20 * 1000  # ms
    static_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    print(f"Static Merging: Memory={static_memory:.2f} MB, Latency={static_latency:.2f} ms")
    
    # 2. Teacher-Free TTA (Standard AdaMerging / LFWA) Profile
    # Requires student model, forward, backward of entropy loss, and optimization step.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    student = resnet18()
    student.fc = nn.Identity()
    student.train().to(device) # Needs gradients
    
    # We optimize 24 layer coefficients
    lam = [torch.tensor(0.5, requires_grad=True, device=device) for _ in range(24)]
    optimizer = torch.optim.Adam(lam, lr=0.01)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = student(inputs)
        # Dummy entropy loss
        probs = torch.softmax(out, dim=-1)
        loss = - (probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        
    start_time = time.time()
    for _ in range(20):
        optimizer.zero_grad()
        out = student(inputs)
        probs = torch.softmax(out, dim=-1)
        loss = - (probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
    tta_latency = (time.time() - start_time) / 20 * 1000  # ms
    tta_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    print(f"Teacher-Free TTA (LFWA): Memory={tta_memory:.2f} MB, Latency={tta_latency:.2f} ms")
    
    # 3. Teacher-Guided TTA Profile (Emulating SyMerge/SATA-SBF)
    # Requires student model + K=2 expert models in VRAM, forward pass on all 3, backward on student, optimization step.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    student = resnet18()
    student.fc = nn.Identity()
    student.train().to(device)
    
    expert1 = resnet18()
    expert1.fc = nn.Identity()
    expert1.eval().to(device)
    
    expert2 = resnet18()
    expert2.fc = nn.Identity()
    expert2.eval().to(device)
    
    lam = [torch.tensor(0.5, requires_grad=True, device=device) for _ in range(24)]
    optimizer = torch.optim.Adam(lam, lr=0.01)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out_student = student(inputs)
        with torch.no_grad():
            out_e1 = expert1(inputs)
            out_e2 = expert2(inputs)
        probs = torch.softmax(out_student, dim=-1)
        loss = - (probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        # Emulate joint loss with experts
        # e.g. KL divergence or representation alignment
        loss = loss + 0.1 * ((out_student - out_e1)**2).mean() + 0.1 * ((out_student - out_e2)**2).mean()
        loss.backward()
        optimizer.step()
        
    start_time = time.time()
    for _ in range(20):
        optimizer.zero_grad()
        out_student = student(inputs)
        with torch.no_grad():
            out_e1 = expert1(inputs)
            out_e2 = expert2(inputs)
        probs = torch.softmax(out_student, dim=-1)
        loss = - (probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        loss = loss + 0.1 * ((out_student - out_e1)**2).mean() + 0.1 * ((out_student - out_e2)**2).mean()
        loss.backward()
        optimizer.step()
    guided_latency = (time.time() - start_time) / 20 * 1000  # ms
    guided_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    print(f"Teacher-Guided TTA (SATA/SyMerge): Memory={guided_memory:.2f} MB, Latency={guided_latency:.2f} ms")
    
    # Print LaTeX ready Table entries
    print("\nLaTeX Table formatting:")
    print(f"Static Merging & {static_memory:.1f} MB & {static_latency:.1f} ms & 0 \\\\")
    print(f"Teacher-Free TTA (LFWA) & {tta_memory:.1f} MB & {tta_latency:.1f} ms & 1 \\\\")
    print(f"Teacher-Guided TTA & {guided_memory:.1f} MB & {guided_latency:.1f} ms & {guided_latency/tta_latency:.1f}\\times \\\\")
    
    results = {
        "static_memory": static_memory,
        "static_latency": static_latency,
        "tta_memory": tta_memory,
        "tta_latency": tta_latency,
        "guided_memory": guided_memory,
        "guided_latency": guided_latency
    }
    torch.save(results, "checkpoints/results_efficiency.pt")

if __name__ == "__main__":
    profile()
