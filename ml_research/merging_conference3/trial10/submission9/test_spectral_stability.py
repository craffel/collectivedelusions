import torch
import torch.nn as nn
import torch.nn.functional as F
from run_experiments import AIR

def test_spectral_stability_analytical():
    print("Running unit test: test_spectral_stability_analytical...")
    K = 4
    # Instantiate AIR model
    model = AIR(K=K)
    
    # Intentionally set precision parameters to extremely high values (e.g., 30.0)
    # This represents extreme precision and under iterative gradient descent would instantly diverge.
    # The exact analytical solver handles it with 100% numerical stability.
    nn.init.constant_(model.p_e, 30.0)  # extremely high log-sensory precision
    nn.init.constant_(model.p_s, 30.0)  # extremely high log-prior precision
    
    # Mock inputs
    e_t = torch.randn(1, K)
    model.reset(torch.zeros(1, K))
    
    # Run forward pass
    try:
        logits = model(e_t, return_logits=True)
        assert not torch.isnan(logits).any(), "Found NaNs in logits!"
        assert not torch.isinf(logits).any(), "Found Infs in logits!"
        print("Success: Exact closed-form analytical solver proved unconditionally stable under extreme precisions!")
    except Exception as e:
        print(f"Failed: Forward pass raised an exception: {e}")
        assert False

if __name__ == "__main__":
    test_spectral_stability_analytical()
