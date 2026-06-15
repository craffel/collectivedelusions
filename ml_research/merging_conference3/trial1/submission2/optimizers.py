import torch
import math

class SAM(torch.optim.Optimizer):
    """
    Standard Sharpness-Aware Minimization (SAM) with AdamW base optimizer.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2, rho=0.05):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, rho=rho)
        super(SAM, self).__init__(params, defaults)
        
        # Initialize AdamW moments
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)
                state["step"] = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Compute L2 norm of gradients
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm += p.grad.data.pow(2).sum()
        grad_norm = torch.sqrt(grad_norm)
        
        # Apply perturbation and backup weights
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad.data * scale
                p.data.add_(e_w)
                
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_prime = p.grad.data
                
                # Restore original weights
                p.data.copy_(state["old_p"])
                
                # Apply weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                
                # Update Adam moments using perturbed gradient g_prime (standard SAM-Adam approach)
                state["step"] += 1
                state["m"].mul_(beta1).add_(g_prime, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(g_prime, g_prime, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                m_hat = state["m"] / bias_correction1
                v_hat = state["v"] / bias_correction2
                
                # Step update
                p.data.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)
                
        if zero_grad:
            self.zero_grad()


class SABCD(torch.optim.Optimizer):
    """
    Sharpness-Aware Block Coordinate Descent (SA-BCD) Optimizer.
    Supports three update modes:
      1. 'literal': Follows the exact paper formula literally:
         step = lr * (m_hat / (sqrt(v_hat) + eps)) * g_prime
      2. 'standard_adam': Standard AdamW step using g_prime, restricted to Omega_t.
         Moments m and v are updated using g_prime.
      3. 'adam_gt': AdamW step restricted to Omega_t, but using unperturbed moments (gt).
         Moments m and v are updated using gt, and step is lr * m_hat / (sqrt(v_hat) + eps).
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2, rho=0.05, p_ratio=0.3, mode='adam_gt'):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, rho=rho, p_ratio=p_ratio, mode=mode)
        super(SABCD, self).__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)
                state["step"] = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # 1. Update unperturbed momentum estimates
        all_abs_m = []
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] += 1
                gt = p.grad.data
                
                # Update m and v with gt
                state["m"].mul_(beta1).add_(gt, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(gt, gt, value=1 - beta2)
                
                all_abs_m.append(state["m"].abs().view(-1))
                
        if not all_abs_m:
            return
            
        all_abs_m = torch.cat(all_abs_m)
        group = self.param_groups[0]
        k = int(len(all_abs_m) * group["p_ratio"])
        k = max(1, min(k, len(all_abs_m)))
        
        # Find global threshold for top-p% largest absolute momentum
        threshold = torch.topk(all_abs_m, k).values[-1]
        
        # 2. Compute L2 norm of gradient restricted to Omega
        grad_norm_omega = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                mask = state["m"].abs() >= threshold
                p_grad_omega = p.grad.data * mask
                grad_norm_omega += p_grad_omega.pow(2).sum()
        grad_norm_omega = torch.sqrt(grad_norm_omega)
        
        # 3. Apply perturbation to Omega and backup original parameters
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm_omega + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                mask = state["m"].abs() >= threshold
                
                state["old_p"] = p.data.clone()
                state["mask"] = mask
                
                # Perturb parameters in Omega only
                e_w = (p.grad.data * mask) * scale
                p.data.add_(e_w)
                
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            mode = group["mode"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_prime = p.grad.data
                mask = state["mask"]
                
                # Restore original parameters
                p.data.copy_(state["old_p"])
                
                # Apply weight decay (only to parameters in Omega_t to be consistent with BCD)
                if wd != 0:
                    p_decay = torch.where(mask, p.data * (1 - lr * wd), p.data)
                    p.data.copy_(p_decay)
                
                if mode == 'literal':
                    # Exact paper formula: \Theta_{t, i} = \Theta_{t-1, i} - \eta * (\hat{m}_{t, i} / (\sqrt{\hat{v}_{t, i}} + \epsilon)) * g'_{t, i} (only for i \in \Omega_t)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    m_hat = state["m"] / bias_correction1
                    v_hat = state["v"] / bias_correction2
                    
                    step_val = (m_hat / (torch.sqrt(v_hat) + eps)) * g_prime
                    p_new = torch.where(mask, p.data - lr * step_val, p.data)
                    p.data.copy_(p_new)
                    
                elif mode == 'standard_adam':
                    # Standard AdamW update but using g_prime, restricted to Omega_t.
                    # We re-update m and v with g_prime for the coordinates in Omega_t.
                    m_omega = torch.where(mask, state["m"].mul(beta1).add(g_prime, alpha=1 - beta1), state["m"])
                    v_omega = torch.where(mask, state["v"].mul(beta2).addcmul(g_prime, g_prime, value=1 - beta2), state["v"])
                    
                    state["m"].copy_(m_omega)
                    state["v"].copy_(v_omega)
                    
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    m_hat = state["m"] / bias_correction1
                    v_hat = state["v"] / bias_correction2
                    
                    step_val = m_hat / (torch.sqrt(v_hat) + eps)
                    p_new = torch.where(mask, p.data - lr * step_val, p.data)
                    p.data.copy_(p_new)
                    
                elif mode == 'adam_gt':
                    # Standard AdamW step using moments computed with gt, restricted to Omega_t.
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    m_hat = state["m"] / bias_correction1
                    v_hat = state["v"] / bias_correction2
                    
                    step_val = m_hat / (torch.sqrt(v_hat) + eps)
                    p_new = torch.where(mask, p.data - lr * step_val, p.data)
                    p.data.copy_(p_new)
                    
        if zero_grad:
            self.zero_grad()
