import test_improved_sable as t
import torch
import numpy as np

print("Running SABLE Routing Temperature (tau) Sensitivity Sweep...")
# Evaluate SABLE with different routing temperatures tau
tau_values = [0.01, 0.05, 0.1, 0.2, 0.5]
for tau in tau_values:
    # We temporarily patch pfsr_coefficients to use the current tau
    original_coefs = t.pfsr_coefficients
    t.pfsr_coefficients = lambda features: original_coefs(features, gamma_OOD=0.2, tau=tau)
    
    acc = t.compute_acc(t.evaluate_sable_single_pass_non_oracle(t.test_x), t.test_y)
    print(f"  tau = {tau:<4}: SABLE Accuracy = {acc:.2f}%")
    
    # Restore
    t.pfsr_coefficients = original_coefs

print("\nRunning SABLE OOD Rejection Threshold (gamma_OOD) Sensitivity Sweep...")
# Evaluate SABLE with different OOD rejection thresholds
gamma_values = [0.0, 0.1, 0.2, 0.4, 0.6]
for gamma in gamma_values:
    original_coefs = t.pfsr_coefficients
    t.pfsr_coefficients = lambda features: original_coefs(features, gamma_OOD=gamma, tau=0.05)
    
    acc = t.compute_acc(t.evaluate_sable_single_pass_non_oracle(t.test_x), t.test_y)
    print(f"  gamma_OOD = {gamma:<4}: SABLE Accuracy = {acc:.2f}%")
    
    # Restore
    t.pfsr_coefficients = original_coefs
