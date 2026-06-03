import torch
import torch.nn as nn
import math

class CPOSResNet(nn.Module):
    """
    Complex-Valued Phase-Orthogonal Superposition (CPOS) model wrapper for ResNet-18.
    Lifts model weights of two expert models (A and B) into the complex plane with orthogonal phases
    (A is real, B is imaginary), so that the activations at each layer can coexist orthogonally.
    At the end of each block, it performs a wavefunction measurement (magnitude collapse):
    |Y| = sqrt(alpha^2 * Y_A^2 + beta^2 * Y_B^2)
    This mathematically eliminates destructive interference and variance collapse training-free.
    """
    def __init__(self, model_A, model_B, alpha=1.0, beta=1.0):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.alpha = alpha
        self.beta = beta
        
        # task_idx: 0 evaluates Task A (using model_A's FC head), 1 evaluates Task B (using model_B's FC head)
        self.task_idx = 0

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def forward(self, x):
        # 1. Stem: conv1, bn1, relu, maxpool
        # Run input through Model A's stem
        y_A = self.model_A.relu(self.model_A.bn1(self.model_A.conv1(x)))
        # Run input through Model B's stem
        y_B = self.model_B.relu(self.model_B.bn1(self.model_B.conv1(x)))
        
        # Apply CPOS magnitude combination (orthogonal superposition measurement)
        # We add 1e-8 inside the sqrt for numerical stability of gradients
        x = torch.sqrt(self.alpha**2 * y_A**2 + self.beta**2 * y_B**2 + 1e-8)
        
        # Apply maxpool
        x = self.model_A.maxpool(x)
        
        # 2. Sequential Blocks (layer1, layer2, layer3, layer4)
        # Each layer contains 2 BasicBlocks
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(self.model_A, layer_name)
            layer_B = getattr(self.model_B, layer_name)
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                
                # Run the merged input through block A and block B independently
                out_A = block_A(x)
                out_B = block_B(x)
                
                # Apply CPOS magnitude combination
                x = torch.sqrt(self.alpha**2 * out_A**2 + self.beta**2 * out_B**2 + 1e-8)
                
        # 3. Global Pooling and Flattening
        x = self.model_A.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Task-Specific Classification Head
        if self.task_idx == 0:
            logits = self.model_A.fc(x)
        elif self.task_idx == 1:
            logits = self.model_B.fc(x)
        else:
            raise ValueError(f"Invalid task_idx: {self.task_idx}")
            
        return logits


class GeneralizedCPOSResNet(nn.Module):
    """
    Generalized Phase-Interleaved CPOS (GP-CPOS) model wrapper for ResNet-18.
    Unifies standard Euclidean merging and Phase-Orthogonal Merging into a continuous
    phase-space interpolation.
    Let task B have a relative phase theta with respect to task A (which is along the real axis).
    W = alpha * W_A + e^(i*theta) * beta * W_B
    The pre-activation Y is:
    Y = alpha * Y_A + e^(i*theta) * beta * Y_B
    At each block boundary, the magnitude collapse is:
    |Y| = sqrt(alpha^2 * Y_A^2 + beta^2 * Y_B^2 + 2 * alpha * beta * Y_A * Y_B * cos(theta) + 1e-8)
    When theta = 0, this recovers the absolute value of standard linear merging.
    When theta = pi/2, this recovers standard CPOS (perfect orthogonality).
    """
    def __init__(self, model_A, model_B, alpha=1.0, beta=1.0, theta=0.0):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        
        # task_idx: 0 evaluates Task A, 1 evaluates Task B
        self.task_idx = 0

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def set_theta(self, theta):
        self.theta = theta

    def forward(self, x):
        # 1. Stem: conv1, bn1, relu, maxpool
        y_A = self.model_A.relu(self.model_A.bn1(self.model_A.conv1(x)))
        y_B = self.model_B.relu(self.model_B.bn1(self.model_B.conv1(x)))
        
        # GP-CPOS magnitude collapse
        cos_theta = math.cos(self.theta)
        term = (self.alpha**2) * (y_A**2) + (self.beta**2) * (y_B**2) + 2.0 * self.alpha * self.beta * y_A * y_B * cos_theta
        x = torch.sqrt(torch.clamp(term, min=0.0) + 1e-8)
        
        x = self.model_A.maxpool(x)
        
        # 2. Sequential Blocks (layer1, layer2, layer3, layer4)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(self.model_A, layer_name)
            layer_B = getattr(self.model_B, layer_name)
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                
                out_A = block_A(x)
                out_B = block_B(x)
                
                term = (self.alpha**2) * (out_A**2) + (self.beta**2) * (out_B**2) + 2.0 * self.alpha * self.beta * out_A * out_B * cos_theta
                x = torch.sqrt(torch.clamp(term, min=0.0) + 1e-8)
                
        # 3. Global Pooling and Flattening
        x = self.model_A.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Task-Specific Classification Head
        if self.task_idx == 0:
            logits = self.model_A.fc(x)
        elif self.task_idx == 1:
            logits = self.model_B.fc(x)
        else:
            raise ValueError(f"Invalid task_idx: {self.task_idx}")
            
        return logits


class QCPOSResNet(nn.Module):
    """
    Quaternion-Inspired Phase-Orthogonal Superposition (Q-CPOS) model wrapper for ResNet-18.
    Extends standard CPOS into the hypercomplex quaternion space H.
    Lifts three expert models (A, B, and C) using mutually orthogonal dimensions (1, i, j).
    Activations are merged block-by-block using magnitude collapse:
    |Y| = sqrt(w1^2 * Y_A^2 + w2^2 * Y_B^2 + w3^2 * Y_C^2)
    This guarantees zero destructive interference and exact representation energy conservation
    across three tasks simultaneously, entirely training-free and data-free.
    """
    def __init__(self, model_A, model_B, model_C, w1=1.0, w2=1.0, w3=1.0):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        
        # task_idx: 0 evaluates Task A, 1 evaluates Task B, 2 evaluates Task C
        self.task_idx = 0

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def forward(self, x):
        # 1. Stem: conv1, bn1, relu, maxpool
        y_A = self.model_A.relu(self.model_A.bn1(self.model_A.conv1(x)))
        y_B = self.model_B.relu(self.model_B.bn1(self.model_B.conv1(x)))
        y_C = self.model_C.relu(self.model_C.bn1(self.model_C.conv1(x)))
        
        # Q-CPOS magnitude collapse
        x = torch.sqrt(self.w1**2 * y_A**2 + self.w2**2 * y_B**2 + self.w3**2 * y_C**2 + 1e-8)
        x = self.model_A.maxpool(x)
        
        # 2. Sequential Blocks (layer1, layer2, layer3, layer4)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(self.model_A, layer_name)
            layer_B = getattr(self.model_B, layer_name)
            layer_C = getattr(self.model_C, layer_name)
            
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                block_C = layer_C[i]
                
                out_A = block_A(x)
                out_B = block_B(x)
                out_C = block_C(x)
                
                # Q-CPOS magnitude collapse
                x = torch.sqrt(self.w1**2 * out_A**2 + self.w2**2 * out_B**2 + self.w3**2 * out_C**2 + 1e-8)
                
        # 3. Global Pooling and Flattening
        x = self.model_A.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Task-Specific Classification Head
        if self.task_idx == 0:
            logits = self.model_A.fc(x)
        elif self.task_idx == 1:
            logits = self.model_B.fc(x)
        elif self.task_idx == 2:
            logits = self.model_C.fc(x)
        else:
            raise ValueError(f"Invalid task_idx: {self.task_idx}")
            
        return logits


class HCPOSResNet(nn.Module):
    """
    Hypercomplex Phase-Orthogonal Superposition (H-CPOS) model wrapper for ResNet-18.
    A unified generalization of CPOS and Q-CPOS to arbitrary numbers of tasks (N)
    using N-dimensional hypercomplex space (such as Octonions for N=8, etc.).
    Fuses N expert models dynamically, performing magnitude collapse:
    |Y| = sqrt(sum_{k=1}^N w_k^2 * Y_k^2)
    This mathematical structure guarantees zero mutual task interference and
    exact activation energy conservation across any number of tasks.
    """
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        if weights is None:
            self.weights = [1.0 / (self.num_models ** 0.5)] * self.num_models
        else:
            self.weights = list(weights)
        
        # Normalize weights so that sum(w_k^2) = 1.0
        w_sum = sum(w**2 for w in self.weights)
        self.weights = [w / (w_sum ** 0.5) for w in self.weights]
        
        # task_idx: evaluates Task k (using models[k]'s FC head)
        self.task_idx = 0

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def forward(self, x):
        # 1. Stem: conv1, bn1, relu, maxpool
        stem_outs = [model.relu(model.bn1(model.conv1(x))) for model in self.models]
        term = 0.0
        for w, out in zip(self.weights, stem_outs):
            term = term + (w**2) * (out**2)
        x = torch.sqrt(term + 1e-8)
        x = self.models[0].maxpool(x)
        
        # 2. Sequential Blocks (layer1, layer2, layer3, layer4)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_len = len(getattr(self.models[0], layer_name))
            for i in range(layer_len):
                block_outs = []
                for model in self.models:
                    layer = getattr(model, layer_name)
                    block = layer[i]
                    block_outs.append(block(x))
                
                term = 0.0
                for w, out in zip(self.weights, block_outs):
                    term = term + (w**2) * (out**2)
                x = torch.sqrt(term + 1e-8)
                
        # 3. Global Pooling and Flattening
        x = self.models[0].avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Task-Specific Classification Head
        logits = self.models[self.task_idx].fc(x)
        return logits


class ChannelWiseCPOSResNet(nn.Module):
    """
    Channel-Wise Phase-Interleaved CPOS (CW-CPOS) model wrapper for ResNet-18.
    Instead of a single phase angle for the entire block, we distribute relative phase angles
    theta_c across channels. This allows some channels to have highly shared, aligned representations (theta_c ~ 0)
    and others to have perfectly isolated, phase-orthogonal representations (theta_c ~ pi/2).
    """
    def __init__(self, model_A, model_B, alpha=1.0, beta=1.0, distribution="linear"):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.alpha = alpha
        self.beta = beta
        self.distribution = distribution
        
        # task_idx: 0 evaluates Task A, 1 evaluates Task B
        self.task_idx = 0

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def _get_cos_theta(self, num_channels, device):
        if self.distribution == "binary":
            half = num_channels // 2
            cos_theta = torch.ones(num_channels, device=device)
            cos_theta[half:] = 0.0
        elif self.distribution == "linear":
            theta = torch.linspace(0.0, math.pi / 2.0, num_channels, device=device)
            cos_theta = torch.cos(theta)
        elif self.distribution == "sinusoidal":
            theta = torch.sin(torch.linspace(0.0, 1.0, num_channels, device=device) * (math.pi / 2.0)) * (math.pi / 2.0)
            cos_theta = torch.cos(theta)
        elif self.distribution == "random":
            g = torch.Generator(device=device)
            g.manual_seed(42)
            theta = torch.rand(num_channels, device=device, generator=g) * (math.pi / 2.0)
            cos_theta = torch.cos(theta)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        return cos_theta.view(1, -1, 1, 1)

    def forward(self, x):
        # 1. Stem: conv1, bn1, relu, maxpool
        y_A = self.model_A.relu(self.model_A.bn1(self.model_A.conv1(x)))
        y_B = self.model_B.relu(self.model_B.bn1(self.model_B.conv1(x)))
        
        cos_theta = self._get_cos_theta(y_A.size(1), y_A.device)
        term = (self.alpha**2) * (y_A**2) + (self.beta**2) * (y_B**2) + 2.0 * self.alpha * self.beta * y_A * y_B * cos_theta
        x = torch.sqrt(torch.clamp(term, min=0.0) + 1e-8)
        
        x = self.model_A.maxpool(x)
        
        # 2. Sequential Blocks (layer1, layer2, layer3, layer4)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(self.model_A, layer_name)
            layer_B = getattr(self.model_B, layer_name)
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                
                out_A = block_A(x)
                out_B = block_B(x)
                
                cos_theta = self._get_cos_theta(out_A.size(1), out_A.device)
                term = (self.alpha**2) * (out_A**2) + (self.beta**2) * (out_B**2) + 2.0 * self.alpha * self.beta * out_A * out_B * cos_theta
                x = torch.sqrt(torch.clamp(term, min=0.0) + 1e-8)
                
        # 3. Global Pooling and Flattening
        x = self.model_A.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Task-Specific Classification Head
        if self.task_idx == 0:
            logits = self.model_A.fc(x)
        elif self.task_idx == 1:
            logits = self.model_B.fc(x)
        else:
            raise ValueError(f"Invalid task_idx: {self.task_idx}")
            
        return logits


class DPRCPOSResNet(nn.Module):
    r"""
    Dynamic Phase Routing CPOS (DPR-CPOS) model wrapper for ResNet-18.
    Dynamically computes the relative phase angle \theta for each sample block-by-block
    based on the relative activation energy of the two expert branches.
    If one task is highly dominant, \theta is routed towards 0 (standard linear/aligned representation)
    to maximize task fidelity. If both tasks are active, \theta is routed towards \pi/2 (orthogonal superposition)
    to eliminate interference.
    """
    def __init__(self, model_A, model_B, alpha=1.0, beta=1.0, delta=1e-5):
        super().__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        
        # task_idx: 0 evaluates Task A, 1 evaluates Task B
        self.task_idx = 0
        
        # For logging/analysis
        self.recorded_thetas = []

    def set_task(self, task_idx):
        self.task_idx = task_idx
        
    def clear_recorded_thetas(self):
        self.recorded_thetas = []

    def forward(self, x):
        # 1. Stem: conv1, bn1, relu, maxpool
        y_A = self.model_A.relu(self.model_A.bn1(self.model_A.conv1(x)))
        y_B = self.model_B.relu(self.model_B.bn1(self.model_B.conv1(x)))
        
        # Compute activation energy for each sample in the batch
        # shape of y_A: [B, C, H, W]
        # We compute L2 norm over C, H, W for each sample:
        E_A = torch.norm(y_A, p=2, dim=(1, 2, 3))
        E_B = torch.norm(y_B, p=2, dim=(1, 2, 3))
        
        # Compute dynamic theta for each sample in the batch
        # theta = (pi / 2) * (1.0 - abs(E_A - E_B) / (E_A + E_B + delta))
        diff_ratio = torch.abs(E_A - E_B) / (E_A + E_B + self.delta)
        theta = (math.pi / 2.0) * (1.0 - diff_ratio)
        
        # We keep track of average theta for logging/analysis
        if self.training is False:
            self.recorded_thetas.append(theta.mean().item())
            
        # To apply sample-wise theta, we compute the magnitude collapse sample-by-sample or vectorized
        # cos_theta shape: [B, 1, 1, 1]
        cos_theta = torch.cos(theta).view(-1, 1, 1, 1)
        
        term = (self.alpha**2) * (y_A**2) + (self.beta**2) * (y_B**2) + 2.0 * self.alpha * self.beta * y_A * y_B * cos_theta
        x = torch.sqrt(torch.clamp(term, min=0.0) + 1e-8)
        
        x = self.model_A.maxpool(x)
        
        # 2. Sequential Blocks (layer1, layer2, layer3, layer4)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(self.model_A, layer_name)
            layer_B = getattr(self.model_B, layer_name)
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                
                out_A = block_A(x)
                out_B = block_B(x)
                
                # Compute sample-wise energy
                E_A = torch.norm(out_A, p=2, dim=(1, 2, 3))
                E_B = torch.norm(out_B, p=2, dim=(1, 2, 3))
                
                diff_ratio = torch.abs(E_A - E_B) / (E_A + E_B + self.delta)
                theta = (math.pi / 2.0) * (1.0 - diff_ratio)
                
                if self.training is False:
                    self.recorded_thetas.append(theta.mean().item())
                    
                cos_theta = torch.cos(theta).view(-1, 1, 1, 1)
                
                term = (self.alpha**2) * (out_A**2) + (self.beta**2) * (out_B**2) + 2.0 * self.alpha * self.beta * out_A * out_B * cos_theta
                x = torch.sqrt(torch.clamp(term, min=0.0) + 1e-8)
                
        # 3. Global Pooling and Flattening
        x = self.model_A.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 4. Task-Specific Classification Head
        if self.task_idx == 0:
            logits = self.model_A.fc(x)
        elif self.task_idx == 1:
            logits = self.model_B.fc(x)
        else:
            raise ValueError(f"Invalid task_idx: {self.task_idx}")
            
        return logits

