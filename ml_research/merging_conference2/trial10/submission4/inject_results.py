import os
import json

def main():
    if not os.path.exists("results.json"):
        print("results.json not found. Run evaluation first.")
        return
        
    with open("results.json", "r") as f:
        data = json.load(f)
        
    runs = data["runs"]
    
    # Let's find the best QCOT C value on INT4_Channel, BN=32, Clean
    qcot_clean_int4 = [r for r in runs if "QCOT" in r["method"] and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"]
    best_qcot_run = max(qcot_clean_int4, key=lambda x: x["average"])
    best_qcot_method = best_qcot_run["method"]
    best_qcot_acc = best_qcot_run["average"]
    
    # Let's find the best QWC q value on INT4_Channel, BN=32, Clean
    qwc_clean_int4 = [r for r in runs if "QWC" in r["method"] and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"]
    best_qwc_run = max(qwc_clean_int4, key=lambda x: x["average"])
    best_qwc_method = best_qwc_run["method"]
    best_qwc_acc = best_qwc_run["average"]
    
    # CWSS Clean under INT4, BN=32
    cwss_clean_int4 = [r for r in runs if r["method"] == "CWSS" and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
    cwss_acc = cwss_clean_int4["average"]
    
    # CWSS-QC Noise under INT4, BN=32
    cwss_qc_noise = [r for r in runs if r["method"] == "CWSS-QC (q=0.9999)" and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "noise"][0]
    cwss_qc_noise_acc = cwss_qc_noise["average"]
    
    print(f"Best QCOT: {best_qcot_method} | Acc: {best_qcot_acc:.2f}%")
    print(f"Best QWC: {best_qwc_method} | Acc: {best_qwc_acc:.2f}%")
    print(f"CWSS Clean Acc: {cwss_acc:.2f}%")
    print(f"CWSS-QC Noise Acc: {cwss_qc_noise_acc:.2f}%")
    
    # Load latex template
    with open("submission_template.tex", "r") as f:
        template = f.read()
        
    # Load latex table
    with open("results/latex_table.tex", "r") as f:
        table_content = f.read()
        
    # Replacements
    output = template.replace("__INSERT_LATEX_TABLE__", table_content)
    output = output.replace("__QWC_INT4_CLEAN_BN32__", f"{best_qwc_acc:.2f}")
    output = output.replace("__QCOT_INT4_CLEAN_BN32__", f"{best_qcot_acc:.2f}")
    output = output.replace("__CWSS_INT4_CLEAN_BN32__", f"{cwss_acc:.2f}")
    output = output.replace("__CWSS_QC_INT4_NOISE_BN32__", f"{cwss_qc_noise_acc:.2f}")
    
    # Save to submission.tex
    with open("submission.tex", "w") as f:
        f.write(output)
        
    print("Injected results successfully into submission.tex.")

if __name__ == "__main__":
    main()
