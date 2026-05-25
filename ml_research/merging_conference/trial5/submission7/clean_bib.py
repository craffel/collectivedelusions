import re

exclude_keywords = [
    "freeway", "remote sensing", "segmentation", "fake news", "crash", "alzheimer", 
    "diabetes", "gravitational-wave", "physics-informed", "i-vector", "ctr ", "ctr_prediction", 
    "graph neural", "speaker verification", "crash risk", "remote-sensing", "black hole",
    "inspirals", "binary evolution", "gravitational waves"
]

include_keywords = [
    "merg", "adapt", "task vector", "gradient", "fisher", "curvature", "weight", 
    "ensemble", "parameter-efficient", "lora", "peft", "ewc", "surgery", "tent", 
    "cotta", "entropy", "multi-task", "multitask", "fine-tuning", "finetuning",
    "optim"
]

def main():
    with open("example_paper.bib") as f:
        text = f.read()
    
    # Simple regex parser for bibtex entries
    entries = re.findall(r'(@[a-zA-Z]+\{[^,]+,\s+title=\{([^}]+)\}.*?\n\})', text, re.DOTALL)
    
    cleaned_entries = []
    seen_keys = set()
    
    # Always keep the known core citations
    core_keys = {"wang2021tent", "yang2024adamerging", "anonymous2026s2c", "anonymous2026pc", "anonymous2026lfwa"}
    
    for entry_text, title in entries:
        # Extract key
        key_match = re.match(r'@[a-zA-Z]+\{([^,]+),', entry_text)
        if not key_match:
            continue
        key = key_match.group(1)
        
        if key in seen_keys:
            continue
            
        # Core keys are always kept
        if key in core_keys:
            seen_keys.add(key)
            cleaned_entries.append(entry_text)
            continue
            
        title_lower = title.lower()
        
        # Check exclusions
        excluded = False
        for kw in exclude_keywords:
            if kw in title_lower:
                excluded = True
                break
        if excluded:
            continue
            
        # Check inclusions
        included = False
        for kw in include_keywords:
            if kw in title_lower:
                included = True
                break
        
        # Also check key just in case
        key_lower = key.lower()
        for kw in include_keywords:
            if kw in key_lower:
                included = True
                break
                
        if included:
            seen_keys.add(key)
            cleaned_entries.append(entry_text)
            
    print(f"Original entries: {len(entries)}")
    print(f"Cleaned entries: {len(cleaned_entries)}")
    
    with open("example_paper.bib", "w") as f:
        f.write("\n\n".join(cleaned_entries))
    print("Wrote cleaned entries to example_paper.bib!")
    
    # Print the keys for easy citation
    print("Cleaned keys:")
    print(", ".join(sorted(list(seen_keys))))

if __name__ == "__main__":
    main()
