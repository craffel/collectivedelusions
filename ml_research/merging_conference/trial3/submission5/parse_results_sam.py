import os

tracker_file = "experiments_tracker_sam.txt"
if not os.path.exists(tracker_file):
    print("Tracker file not found.")
    exit()

with open(tracker_file, "r") as f:
    lines = f.readlines()

results = []
incomplete = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    name, job_id = line.split(",")
    out_file = f"{name}_{job_id}.out"
    
    if os.path.exists(out_file):
        with open(out_file, "r") as f_out:
            content = f_out.read()
            
        if "Final Results" in content:
            final_line = ""
            for l in content.split("\n"):
                if "Final Results" in l:
                    final_line = l
                    break
            
            try:
                parts = final_line.split("|")
                cifar = parts[1].split(":")[1].strip().replace("%", "")
                svhn = parts[2].split(":")[1].strip().replace("%", "")
                avg = parts[3].split(":")[1].strip().replace("%", "")
                results.append({
                    "name": name,
                    "cifar": float(cifar),
                    "svhn": float(svhn),
                    "avg": float(avg),
                    "status": "Completed"
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "cifar": "-",
                    "svhn": "-",
                    "avg": "-",
                    "status": f"Error parsing: {e}"
                })
        elif "Traceback" in content:
            results.append({
                "name": name,
                "cifar": "-",
                "svhn": "-",
                "avg": "-",
                "status": "Failed"
            })
        else:
            incomplete.append(name)
    else:
        incomplete.append(name)

print("\n### S2C-SAM Experimental Results Summary ###\n")
print(f"{'Job Name':<32} | {'CIFAR-10 (%)':<12} | {'SVHN (%)':<10} | {'Average (%)':<12} | {'Status':<10}")
print("-" * 80)
for r in sorted(results, key=lambda x: x["name"]):
    print(f"{r['name']:<32} | {r['cifar']:<12} | {r['svhn']:<10} | {r['avg']:<12} | {r['status']:<10}")

print(f"\nIncomplete jobs ({len(incomplete)}): {', '.join(incomplete)}")
