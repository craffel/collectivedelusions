import json
import numpy as np

with open("sweep_results.json", "r") as f:
    results = json.load(f)

print(f"Total results loaded: {len(results)}")

# Let's inspect some sample configurations
print("\nUnique Merge Types:", set(r['merge_type'] for r in results))
print("Unique Calibrations:", set(r['calibration'] for r in results))
print("Unique Quant Bits:", set(r['quantization_bits'] for r in results))
print("Unique Quant Modes:", set(r['quantization_mode'] for r in results))

print("\n--- DETAILED RESULTS (Clean, Task Arithmetic lambda=0.5) ---")
print(f"{'Cal':<8} | {'FP32':<8} | {'INT8-Tensor':<12} | {'INT8-Channel':<13} | {'INT4-Tensor':<12} | {'INT4-Channel':<13}")
print("-" * 80)
for cal in ['none', 'u-ipr', 'hns', 'qr-ipr']:
    fp32 = [r['avg_acc'] for r in results if r['merge_type']=='ta' and r['lambda']==0.5 and r['calibration']==cal and r['quantization_bits']=='FP32' and r['corruption']=='clean'][0]
    i8_t = [r['avg_acc'] for r in results if r['merge_type']=='ta' and r['lambda']==0.5 and r['calibration']==cal and r['quantization_bits']==8 and r['quantization_mode']=='per_tensor' and r['corruption']=='clean'][0]
    i8_c = [r['avg_acc'] for r in results if r['merge_type']=='ta' and r['lambda']==0.5 and r['calibration']==cal and r['quantization_bits']==8 and r['quantization_mode']=='per_channel' and r['corruption']=='clean'][0]
    i4_t = [r['avg_acc'] for r in results if r['merge_type']=='ta' and r['lambda']==0.5 and r['calibration']==cal and r['quantization_bits']==4 and r['quantization_mode']=='per_tensor' and r['corruption']=='clean'][0]
    i4_c = [r['avg_acc'] for r in results if r['merge_type']=='ta' and r['lambda']==0.5 and r['calibration']==cal and r['quantization_bits']==4 and r['quantization_mode']=='per_channel' and r['corruption']=='clean'][0]
    print(f"{cal:<8} | {fp32:8.2f}% | {i8_t:12.2f}% | {i8_c:13.2f}% | {i4_t:12.2f}% | {i4_c:13.2f}%")

print("\n--- MERGING ALGORITHMS COMPARISON (Clean, FP32, INT8 per-tensor) ---")
print(f"{'Merge Type':<12} | {'None FP32':<10} | {'QR-IPR FP32':<12} | {'None INT8-T':<12} | {'QR-IPR INT8-T':<14}")
print("-" * 70)
for m_type in ['wa', 'ta', 'ties', 'dare']:
    lam = 0.5 if m_type != 'wa' else 0.5 # wa doesn't use lam but let's query lambda=0.5 config
    acc_none_fp32 = [r['avg_acc'] for r in results if r['merge_type']==m_type and r['calibration']=='none' and r['quantization_bits']=='FP32' and r['corruption']=='clean'][0]
    acc_qr_fp32 = [r['avg_acc'] for r in results if r['merge_type']==m_type and r['calibration']=='qr-ipr' and r['quantization_bits']=='FP32' and r['corruption']=='clean'][0]
    acc_none_i8t = [r['avg_acc'] for r in results if r['merge_type']==m_type and r['calibration']=='none' and r['quantization_bits']==8 and r['quantization_mode']=='per_tensor' and r['corruption']=='clean'][0]
    acc_qr_i8t = [r['avg_acc'] for r in results if r['merge_type']==m_type and r['calibration']=='qr-ipr' and r['quantization_bits']==8 and r['quantization_mode']=='per_tensor' and r['corruption']=='clean'][0]
    print(f"{m_type:<12} | {acc_none_fp32:9.2f}% | {acc_qr_fp32:11.2f}% | {acc_none_i8t:11.2f}% | {acc_qr_i8t:13.2f}%")
