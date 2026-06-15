import torch
import torch.nn.functional as F
import numpy as np
from run_experiments import RepresentationSandbox, get_oracle_experts, PFSRRouter, OTSPRouter, evaluate_router_detailed, get_projection_matrix

def run_sweep(rho_val):
    seeds = list(range(42, 52))
    taus = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    
    print(f"\n=================== SWEEP FOR RHO = {rho_val} ===================")
    
    # 1. Evaluate Uniform Merging first across 10 seeds
    uniform_accs = []
    for seed in seeds:
        sandbox = RepresentationSandbox(seed, rho=rho_val)
        experts = get_oracle_experts(sandbox)
        X_test, Y_test, task_test = sandbox.generate_split(250)
        P = get_projection_matrix(sandbox.D, sandbox.K)
        
        class UniformRouter:
            def __call__(self, X):
                return torch.ones(len(X), 4) * 0.25
        uniform_router = UniformRouter()
        h, h256, h1, r = evaluate_router_detailed(uniform_router, sandbox, experts, X_test, Y_test, task_test, P)
        uniform_accs.append(h1)
    print(f"Uniform Merging Heterogeneous (B=1): {np.mean(uniform_accs)*100:.2f}% ± {np.std(uniform_accs)*100:.2f}%")
    
    # 2. Evaluate PFSR and OTSP for each tau
    for tau in taus:
        p_accs = []
        p_routes = []
        o_accs = []
        o_routes = []
        
        for seed in seeds:
            sandbox = RepresentationSandbox(seed, rho=rho_val)
            experts = get_oracle_experts(sandbox)
            X_test, Y_test, task_test = sandbox.generate_split(250)
            P = get_projection_matrix(sandbox.D, sandbox.K)
            
            pfsr = PFSRRouter(experts, tau=tau)
            otsp = OTSPRouter(experts, tau=tau)
            
            h_p, h256_p, h1_p, r_p = evaluate_router_detailed(pfsr, sandbox, experts, X_test, Y_test, task_test, P)
            h_o, h256_o, h1_o, r_o = evaluate_router_detailed(otsp, sandbox, experts, X_test, Y_test, task_test, P)
            
            p_accs.append(h1_p)
            p_routes.append(r_p)
            o_accs.append(h1_o)
            o_routes.append(r_o)
            
        print(f"Tau: {tau:5.3f} | PFSR: Het(B=1)={np.mean(p_accs)*100:.2f}% ± {np.std(p_accs)*100:.2f}%, Route={np.mean(p_routes)*100:.2f}% | OTSP: Het(B=1)={np.mean(o_accs)*100:.2f}% ± {np.std(o_accs)*100:.2f}%, Route={np.mean(o_routes)*100:.2f}%")

if __name__ == "__main__":
    run_sweep(0.0)
    run_sweep(0.33)
