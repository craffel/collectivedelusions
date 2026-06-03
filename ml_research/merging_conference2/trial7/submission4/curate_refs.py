import re

def parse_keys_from_bib(filepath):
    keys = set()
    with open(filepath, "r") as f:
        content = f.read()
    matches = re.findall(r"@\w+\{([^,]+),", content)
    for m in matches:
        keys.add(m.strip())
    return keys

def curate():
    existing_keys = parse_keys_from_bib("submission.bib")
    print(f"Existing keys: {len(existing_keys)}")
    
    exclude_keywords = [
        "wood density", "Gas Sensors", "Raman", "concrete bridges", 
        "NIR", "Spectroscopy", "EMG-based", "SSVEP-BCIs", "SSVEP-Based",
        "spectrometer", "Spectrometers", "eye gaze", "Wavelength Selective"
    ]
    
    with open("fetched_bibtexs.txt", "r") as f:
        text = f.read()
        
    entries = text.split("\n\n")
    curated_entries = []
    added_keys = []
    
    for entry in entries:
        if not entry.strip():
            continue
        # Check exclusion
        skip = False
        for kw in exclude_keywords:
            if kw.lower() in entry.lower():
                skip = True
                break
        if skip:
            continue
            
        # Extract key
        m = re.search(r"@(\w+)\{([^,]+),", entry)
        if m:
            entry_type = m.group(1)
            key = m.group(2).strip()
            
            # De-duplicate
            if key in existing_keys or key in added_keys:
                continue
                
            # Keep it
            added_keys.append(key)
            curated_entries.append(entry.strip())
            
    print(f"Curated {len(curated_entries)} new entries.")
    
    # Append to submission.bib
    with open("submission.bib", "a") as f:
        f.write("\n\n% Curated Refined References\n")
        for entry in curated_entries:
            f.write(entry + "\n\n")
            
    # Write added keys to a file
    with open("added_keys.txt", "w") as f:
        for k in added_keys:
            f.write(k + "\n")
            
if __name__ == "__main__":
    curate()
