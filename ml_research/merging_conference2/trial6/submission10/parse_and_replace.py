import re

def parse_results(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse Oracle
    oracle_mnist = float(re.search(r"Expert MNIST Accuracy:\s+([\d.]+)", content).group(1))
    oracle_fmnist = float(re.search(r"Expert FMNIST Accuracy:\s+([\d.]+)", content).group(1))
    oracle_cifar = float(re.search(r"Expert CIFAR10 Accuracy:\s+([\d.]+)", content).group(1))
    oracle_avg = (oracle_mnist + oracle_fmnist + oracle_cifar) / 3.0

    print(f"Parsed Oracle: {oracle_mnist:.2f}, {oracle_fmnist:.2f}, {oracle_cifar:.2f}, {oracle_avg:.2f}")

    # Helper to parse a block for a specific merge method and calibration method
    def get_stats(merge_name, calib_name):
        # Find the block for the merge method using lookahead
        merge_pattern = rf"Merging Method:\s+{re.escape(merge_name)}.*?(?=(?:Merging Method:|$))"
        merge_match = re.search(merge_pattern, content, re.DOTALL)
        if not merge_match:
            # Let's try matching with looser regex
            merge_pattern = rf"Merging Method:\s+{re.escape(merge_name)}.*"
            merge_match = re.search(merge_pattern, content, re.DOTALL)
            
        merge_block = merge_match.group(0)

        # Now find the calibration block within this merge block
        calib_pattern = rf"--- Calibration:\s+{re.escape(calib_name)} ---\n(?:Task MNIST Accuracy:\s+([\d.]+))?.*?\n(?:Task FMNIST Accuracy:\s+([\d.]+))?.*?\n(?:Task CIFAR10 Accuracy:\s+([\d.]+))?.*?\nAverage Multi-Task Accuracy:\s+([\d.]+)"
        calib_match = re.search(calib_pattern, merge_block, re.DOTALL)
        if not calib_match:
            # Try a softer match
            calib_pattern = rf"--- Calibration:\s+{re.escape(calib_name)} ---.*?Average Multi-Task Accuracy:\s+([\d.]+)"
            calib_match = re.search(calib_pattern, merge_block, re.DOTALL)
            avg = float(calib_match.group(1))
            # Get individual tasks
            mnist_match = re.search(rf"--- Calibration:\s+{re.escape(calib_name)} ---.*?Task MNIST Accuracy:\s+([\d.]+)", merge_block, re.DOTALL)
            fmnist_match = re.search(rf"--- Calibration:\s+{re.escape(calib_name)} ---.*?Task FMNIST Accuracy:\s+([\d.]+)", merge_block, re.DOTALL)
            cifar_match = re.search(rf"--- Calibration:\s+{re.escape(calib_name)} ---.*?Task CIFAR10 Accuracy:\s+([\d.]+)", merge_block, re.DOTALL)
            mnist = float(mnist_match.group(1)) if mnist_match else 0.0
            fmnist = float(fmnist_match.group(1)) if fmnist_match else 0.0
            cifar = float(cifar_match.group(1)) if cifar_match else 0.0
            return mnist, fmnist, cifar, avg

        mnist = float(calib_match.group(1)) if calib_match.group(1) else 0.0
        fmnist = float(calib_match.group(2)) if calib_match.group(2) else 0.0
        cifar = float(calib_match.group(3)) if calib_match.group(3) else 0.0
        avg = float(calib_match.group(4))
        return mnist, fmnist, cifar, avg

    # Store results in a dictionary mapping placeholder suffix to value
    replacements = {
        "MNIST-ORACLE": f"{oracle_mnist:.2f}",
        "FMNIST-ORACLE": f"{oracle_fmnist:.2f}",
        "CIFAR-ORACLE": f"{oracle_cifar:.2f}",
        "AVG-ORACLE": f"{oracle_avg:.2f}",
    }

    # Define mappings of merge method names and calibration names to suffixes
    merges = {
        "Weight Averaging (WA)": "WA",
        "Task Arithmetic (TA, lambda=0.5)": "TA"
    }
    
    calibs = {
        "No Calibration": "NONE",
        "Real Task-Specific Calib (Oracle)": "REAL",
        "Real Joint Multi-Task Calib": "JOINT",
        "White-Noise Calib (Data-Free)": "WN",
        "Pink-Noise Calib (Data-Free)": "PN",
        "OOD Calib: CIFAR for MNIST/FMNIST, MNIST for CIFAR (Data-Free)": "OOD",
        "Generative BN-Matching Calib (Our DF-Calib)": "GEN"
    }

    for m_full, m_suf in merges.items():
        for c_full, c_suf in calibs.items():
            try:
                mnist, fmnist, cifar, avg = get_stats(m_full, c_full)
                replacements[f"{m_suf}-{c_suf}-MNIST"] = f"{mnist:.2f}"
                replacements[f"{m_suf}-{c_suf}-FMNIST"] = f"{fmnist:.2f}"
                replacements[f"{m_suf}-{c_suf}-CIFAR"] = f"{cifar:.2f}"
                replacements[f"{m_suf}-{c_suf}-AVG"] = f"{avg:.2f}"
                print(f"Parsed {m_suf}-{c_suf}: {mnist:.2f}, {fmnist:.2f}, {cifar:.2f}, {avg:.2f}")
            except Exception as e:
                print(f"Could not parse {m_full} / {c_full}: {e}")

    # Now open submission.tex and do the replacements
    with open("submission.tex", "r") as f:
        tex_content = f.read()

    for placeholder, val in replacements.items():
        tex_content = tex_content.replace(f"[{placeholder}]", val)

    with open("submission.tex", "w") as f:
        f.write(tex_content)

    print("Successfully replaced all placeholders in submission.tex!")

if __name__ == "__main__":
    parse_results("results.txt")
