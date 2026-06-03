import torch
from train_and_merge import MultiTaskResNet18, merge_wa, merge_ta, merge_wcpr, merge_qcot, merge_qcsw, quantize_backbone_, evaluate_merged

def test_all():
    print("Testing model initialization...")
    device = torch.device('cpu')
    model = MultiTaskResNet18()
    state_dict = model.state_dict()
    print("Model initialized successfully.")
    
    # Create mock expert states with small perturbations
    print("\nCreating mock expert states...")
    expert_states = []
    for i in range(3):
        expert_state = {}
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                expert_state[k] = v + torch.randn_like(v) * 0.01
            else:
                expert_state[k] = v.clone()
        expert_states.append(expert_state)
    print("Mock expert states created.")
    
    # Test WA
    print("\nTesting WA Merging...")
    wa_backbone = merge_wa(expert_states)
    assert len(wa_backbone) > 0
    print("WA Merging passed.")
    
    # Test TA
    print("\nTesting TA Merging...")
    ta_backbone = merge_ta(state_dict, expert_states, lambd=0.4)
    assert len(ta_backbone) > 0
    print("TA Merging passed.")
    
    # Test WCPR
    print("\nTesting WCPR Merging...")
    wcpr_backbone = merge_wcpr(state_dict, expert_states)
    assert len(wcpr_backbone) > 0
    print("WCPR Merging passed.")
    
    # Test QCOT
    print("\nTesting QCOT Merging...")
    qcot_backbone = merge_qcot(state_dict, expert_states, C=0.5)
    assert len(qcot_backbone) > 0
    print("QCOT Merging passed.")
    
    # Test QCSW
    print("\nTesting QCSW Merging...")
    qcsw_backbone = merge_qcsw(state_dict, expert_states, C=0.5, num_projections=10)
    assert len(qcsw_backbone) > 0
    print("QCSW Merging passed.")
    
    # Test Quantization
    print("\nTesting Backbone Quantization...")
    backbone_loaded = MultiTaskResNet18().backbone
    backbone_loaded.load_state_dict(wa_backbone)
    
    quant_per_tensor = quantize_backbone_(backbone_loaded, bits=8, quant_type='per_tensor')
    assert len(quant_per_tensor) > 0
    print("Per-tensor quantization passed.")
    
    quant_per_channel = quantize_backbone_(backbone_loaded, bits=8, quant_type='per_channel')
    assert len(quant_per_channel) > 0
    print("Per-channel quantization passed.")
    
    print("\nALL CODE UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_all()
