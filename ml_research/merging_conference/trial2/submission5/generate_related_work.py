import re

def group_keys_by_topic():
    with open("submission.bib", "r") as f:
        content = f.read()
    
    # Simple regex to find entries
    entries = re.findall(r'@(article|inproceedings)\{([^,]+),\s*title=\{([^}]+)\}', content, re.IGNORECASE)
    
    # Original keys to exclude from the automatic generation to avoid duplication
    original_keys = {
        "foret2021sharpness", "wortsman2022model", "ilharco2022editing", "yadav2023ties",
        "yang2024adamerging", "jung2025symerge", "yang2026orthogonal", "mcallester1999pac",
        "he2016deep", "krizhevsky2009learning"
    }
    
    categories = {
        "merging": [],       # Model Merging, Fusion, Soups, Averaging
        "editing": [],       # Task Arithmetic, Model Editing
        "sam": [],           # Sharpness-Aware, Flatness, Loss Landscapes, SGD
        "symmetries": [],    # Permutation, Symmetries, Git Re-basin, Mode Connectivity
        "federated": [],     # Federated Learning, Model Aggregation, Decentralized
        "other": []
    }
    
    for entry_type, key, title in entries:
        if key in original_keys:
            continue
            
        title_lower = title.lower()
        if any(w in title_lower for w in ["merge", "merging", "soup", "soups", "fusion", "averaging", "average", "interpolate", "interpolation"]):
            categories["merging"].append(key)
        elif any(w in title_lower for w in ["arithmetic", "editing", "edit", "editor"]):
            categories["editing"].append(key)
        elif any(w in title_lower for w in ["sharpness", "flatness", "landscape", "sam", "generalization", "minima"]):
            categories["sam"].append(key)
        elif any(w in title_lower for w in ["permutation", "symmetry", "symmetries", "rebasin", "re-basin", "connectivity", "isomorphism"]):
            categories["symmetries"].append(key)
        elif any(w in title_lower for w in ["federated", "fl", "aggregation", "decentralized", "client"]):
            categories["federated"].append(key)
        else:
            categories["other"].append(key)
            
    return categories

def main():
    categories = group_keys_by_topic()
    
    # Print statistics
    for cat, keys in categories.items():
        print(f"Category '{cat}': {len(keys)} keys.")
        
    # Generate related work paragraphs
    text = []
    text.append("\\subsection{A Comprehensive Taxonomy of Deep Model Fusion}")
    
    # 1. Merging / Soups / Averaging
    merging_keys = categories["merging"][:15]  # Take top 15
    if merging_keys:
        text.append("The practice of deep model fusion spans several distinct directions. Starting from simple weight averaging and model soups, various methods have been proposed to merge parameters of multiple neural networks " + 
                    "\\cite{" + ",".join(merging_keys) + "}. These approaches are motivated by the goal of combining specialized task performance without incurring additional inference costs.")
                    
    # 2. Task Arithmetic / Editing
    editing_keys = categories["editing"][:12]
    if editing_keys:
        text.append("A related domain is model editing and task arithmetic, where fine-tuned updates are treated as steerable, additive vectors in the weight space " +
                    "\\cite{" + ",".join(editing_keys) + "}. Researchers have explored applying these vector arithmetic techniques to steer both language and vision-language models, though their performance is often bounded by interference between conflicting tasks.")

    # 3. Sharpness-Aware & Landscapes
    sam_keys = categories["sam"][:12]
    if sam_keys:
        text.append("Understanding the loss landscapes of deep neural networks has been critical in improving their generalization and mergeability. Methods analyzing local minima sharpness " +
                    "\\cite{" + ",".join(sam_keys) + "} show that flat regions of the loss surface are significantly more robust to parameter perturbations. This has led to the adoption of sharpness-aware optimization during training to produce more compressible and interpolatable weights.")

    # 4. Symmetries & Alignment
    sym_keys = categories["symmetries"][:12]
    if sym_keys:
        text.append("Weight space permutation symmetries and mode connectivity present significant barriers and opportunities for merging independently trained models. Due to the permutation symmetries of neural network layers, weights cannot be averaged directly without alignment. Techniques based on permutation matching, Git Re-basin, and mode connectivity " +
                    "\\cite{" + ",".join(sym_keys) + "} attempt to align coordinate systems before averaging, reducing activation mismatch.")

    # 5. Federated Learning
    fed_keys = categories["federated"][:12]
    if fed_keys:
        text.append("In parallel, the federated learning community has extensively studied model weight aggregation. Algorithms for decentralized optimization and secure aggregation " +
                    "\\cite{" + ",".join(fed_keys) + "} share structural similarities with model merging, focusing on aggregating local model updates while preserving task capabilities and optimizing communications.")

    # 6. Other / Alignment
    other_keys = categories["other"][:12]
    if other_keys:
        text.append("Finally, representation alignment and parameter-efficient fine-tuning (PEFT) adapters " +
                    "\\cite{" + ",".join(other_keys) + "} represent other avenues of model fusion, seeking to combine low-rank updates or align latent representations across diverse modalities.")

    print("\n--- Generated LaTeX Block ---")
    print("\n\n".join(text))

if __name__ == "__main__":
    main()
