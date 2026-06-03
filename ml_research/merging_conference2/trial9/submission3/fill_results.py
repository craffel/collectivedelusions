import json
import os
import re

def main():
    resnet_json_path = 'results/results_resnet18.json'
    mlp_json_path = 'results/results_mlp.json'
    tex_path = 'submission.tex'
    
    if not os.path.exists(resnet_json_path) or not os.path.exists(mlp_json_path):
        print("Results JSONs not found yet!")
        return
        
    with open(resnet_json_path, 'r') as f:
        resnet = json.load(f)
    with open(mlp_json_path, 'r') as f:
        mlp = json.load(f)
        
    # Read LaTeX template
    with open(tex_path, 'r') as f:
        tex = f.read()
        
    # Define mappings for ResNet-18 (Table 1)
    # Since ResNet-18 has BatchNorm, we report the calibrated values (with _DEBN)
    # for WCPR, Ours, WA, and others if available, or without if _DEBN is not evaluated/not present.
    def get_resnet_val(merge_key, cond_key):
        # Find actual key in JSON (to handle Tuned TA lam=... dynamically)
        target_key = None
        for k in resnet.keys():
            if k.startswith(merge_key):
                target_key = k
                break
        if not target_key:
            return "0.00"
            
        # Try with _DEBN first, then fallback
        cond_debn = cond_key + "_DEBN"
        if cond_debn in resnet[target_key]:
            val = resnet[target_key][cond_debn]['avg']
        elif cond_key in resnet[target_key]:
            val = resnet[target_key][cond_key]['avg']
        else:
            val = 0.0
        return f"{val:.2f}"

    def get_mlp_val(merge_key, cond_key):
        target_key = None
        for k in mlp.keys():
            if k.startswith(merge_key):
                target_key = k
                break
        if not target_key:
            return "0.00"
            
        if cond_key in mlp[target_key]:
            val = mlp[target_key][cond_key]['avg']
        else:
            val = 0.0
        return f"{val:.2f}"

    # ResNet-18 replacements
    replacements = {
        '[WA_FP32]': get_resnet_val('Weight Averaging (WA)', 'FP32_Clean'),
        '[WA_Tensor]': get_resnet_val('Weight Averaging (WA)', 'INT8_Tensor_Clean'),
        '[WA_Channel]': get_resnet_val('Weight Averaging (WA)', 'INT8_Channel_Clean'),
        '[WA_Noise]': get_resnet_val('Weight Averaging (WA)', 'FP32_Noise'),
        '[WA_Blur]': get_resnet_val('Weight Averaging (WA)', 'FP32_Blur'),
        
        '[TA_FP32]': get_resnet_val('Tuned TA', 'FP32_Clean'),
        '[TA_Tensor]': get_resnet_val('Tuned TA', 'INT8_Tensor_Clean'),
        '[TA_Channel]': get_resnet_val('Tuned TA', 'INT8_Channel_Clean'),
        '[TA_Noise]': get_resnet_val('Tuned TA', 'FP32_Noise'),
        '[TA_Blur]': get_resnet_val('Tuned TA', 'FP32_Blur'),
        
        '[TIES_FP32]': get_resnet_val('TIES-Merging', 'FP32_Clean'),
        '[TIES_Tensor]': get_resnet_val('TIES-Merging', 'INT8_Tensor_Clean'),
        '[TIES_Channel]': get_resnet_val('TIES-Merging', 'INT8_Channel_Clean'),
        '[TIES_Noise]': get_resnet_val('TIES-Merging', 'FP32_Noise'),
        '[TIES_Blur]': get_resnet_val('TIES-Merging', 'FP32_Blur'),
        
        '[DARE_FP32]': get_resnet_val('DARE-Merging', 'FP32_Clean'),
        '[DARE_Tensor]': get_resnet_val('DARE-Merging', 'INT8_Tensor_Clean'),
        '[DARE_Channel]': get_resnet_val('DARE-Merging', 'INT8_Channel_Clean'),
        '[DARE_Noise]': get_resnet_val('DARE-Merging', 'FP32_Noise'),
        '[DARE_Blur]': get_resnet_val('DARE-Merging', 'FP32_Blur'),
        
        '[WCPR_FP32]': get_resnet_val('WCPR', 'FP32_Clean'),
        '[WCPR_Tensor]': get_resnet_val('WCPR', 'INT8_Tensor_Clean'),
        '[WCPR_Channel]': get_resnet_val('WCPR', 'INT8_Channel_Clean'),
        '[WCPR_Noise]': get_resnet_val('WCPR', 'FP32_Noise'),
        '[WCPR_Blur]': get_resnet_val('WCPR', 'FP32_Blur'),
        
        '[QRIPR_FP32]': get_resnet_val('QR-IPR', 'FP32_Clean'),
        '[QRIPR_Tensor]': get_resnet_val('QR-IPR', 'INT8_Tensor_Clean'),
        '[QRIPR_Channel]': get_resnet_val('QR-IPR', 'INT8_Channel_Clean'),
        '[QRIPR_Noise]': get_resnet_val('QR-IPR', 'FP32_Noise'),
        '[QRIPR_Blur]': get_resnet_val('QR-IPR', 'FP32_Blur'),
        
        '[Ours_FP32]': get_resnet_val('QR-SP-WCPR', 'FP32_Clean'),
        '[Ours_Tensor]': get_resnet_val('QR-SP-WCPR', 'INT8_Tensor_Clean'),
        '[Ours_Channel]': get_resnet_val('QR-SP-WCPR', 'INT8_Channel_Clean'),
        '[Ours_Noise]': get_resnet_val('QR-SP-WCPR', 'FP32_Noise'),
        '[Ours_Blur]': get_resnet_val('QR-SP-WCPR', 'FP32_Blur'),
    }
    
    # MLP replacements (Table 2)
    replacements.update({
        '[MLP_WA_FP32]': get_mlp_val('Weight Averaging (WA)', 'FP32_Clean'),
        '[MLP_WA_Tensor]': get_mlp_val('Weight Averaging (WA)', 'INT8_Tensor_Clean'),
        '[MLP_WA_Channel]': get_mlp_val('Weight Averaging (WA)', 'INT8_Channel_Clean'),
        '[MLP_WA_Noise]': get_mlp_val('Weight Averaging (WA)', 'FP32_Noise'),
        '[MLP_WA_Blur]': get_mlp_val('Weight Averaging (WA)', 'FP32_Blur'),
        
        '[MLP_TA_FP32]': get_mlp_val('Tuned TA', 'FP32_Clean'),
        '[MLP_TA_Tensor]': get_mlp_val('Tuned TA', 'INT8_Tensor_Clean'),
        '[MLP_TA_Channel]': get_mlp_val('Tuned TA', 'INT8_Channel_Clean'),
        '[MLP_TA_Noise]': get_mlp_val('Tuned TA', 'FP32_Noise'),
        '[MLP_TA_Blur]': get_mlp_val('Tuned TA', 'FP32_Blur'),
        
        '[MLP_TIES_FP32]': get_mlp_val('TIES-Merging', 'FP32_Clean'),
        '[MLP_TIES_Tensor]': get_mlp_val('TIES-Merging', 'INT8_Tensor_Clean'),
        '[MLP_TIES_Channel]': get_mlp_val('TIES-Merging', 'INT8_Channel_Clean'),
        '[MLP_TIES_Noise]': get_mlp_val('TIES-Merging', 'FP32_Noise'),
        '[MLP_TIES_Blur]': get_mlp_val('TIES-Merging', 'FP32_Blur'),
        
        '[MLP_DARE_FP32]': get_mlp_val('DARE-Merging', 'FP32_Clean'),
        '[MLP_DARE_Tensor]': get_mlp_val('DARE-Merging', 'INT8_Tensor_Clean'),
        '[MLP_DARE_Channel]': get_mlp_val('DARE-Merging', 'INT8_Channel_Clean'),
        '[MLP_DARE_Noise]': get_mlp_val('DARE-Merging', 'FP32_Noise'),
        '[MLP_DARE_Blur]': get_mlp_val('DARE-Merging', 'FP32_Blur'),
        
        '[MLP_WCPR_FP32]': get_mlp_val('WCPR', 'FP32_Clean'),
        '[MLP_WCPR_Tensor]': get_mlp_val('WCPR', 'INT8_Tensor_Clean'),
        '[MLP_WCPR_Channel]': get_mlp_val('WCPR', 'INT8_Channel_Clean'),
        '[MLP_WCPR_Noise]': get_mlp_val('WCPR', 'FP32_Noise'),
        '[MLP_WCPR_Blur]': get_mlp_val('WCPR', 'FP32_Blur'),
        
        '[MLP_QRIPR_FP32]': get_mlp_val('QR-IPR', 'FP32_Clean'),
        '[MLP_QRIPR_Tensor]': get_mlp_val('QR-IPR', 'INT8_Tensor_Clean'),
        '[MLP_QRIPR_Channel]': get_mlp_val('QR-IPR', 'INT8_Channel_Clean'),
        '[MLP_QRIPR_Noise]': get_mlp_val('QR-IPR', 'FP32_Noise'),
        '[MLP_QRIPR_Blur]': get_mlp_val('QR-IPR', 'FP32_Blur'),
        
        '[MLP_Ours_FP32]': get_mlp_val('QR-SP-WCPR', 'FP32_Clean'),
        '[MLP_Ours_Tensor]': get_mlp_val('QR-SP-WCPR', 'INT8_Tensor_Clean'),
        '[MLP_Ours_Channel]': get_mlp_val('QR-SP-WCPR', 'INT8_Channel_Clean'),
        '[MLP_Ours_Noise]': get_mlp_val('QR-SP-WCPR', 'FP32_Noise'),
        '[MLP_Ours_Blur]': get_mlp_val('QR-SP-WCPR', 'FP32_Blur'),
    })
    
    # Replace in TeX
    for placeholder, val in replacements.items():
        tex = tex.replace(placeholder, val)
        
    with open(tex_path, 'w') as f:
        f.write(tex)
        
    print("Successfully injected all results into submission.tex!")

if __name__ == '__main__':
    main()
