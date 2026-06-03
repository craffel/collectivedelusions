def show_sections(txt_path, sections):
    print(f"\n==================== {txt_path} ====================")
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    for section in sections:
        # Simple search for the section heading
        idx = content.lower().find(section.lower())
        if idx != -1:
            print(f"\n--- Found section: '{section}' at char {idx} ---")
            print(content[idx:idx+2000])
        else:
            print(f"\n--- Section '{section}' not found ---")

show_sections("sub3_text.txt", ["1. Introduction", "3. Methodology", "4. Experimental Setup"])
show_sections("sub9_text.txt", ["1. Introduction", "3. Deconstructing", "5. Experimental Evaluation"])
show_sections("sub10_text.txt", ["1. Introduction", "3. Rescales the Fourier", "5. Experimental Evaluation"])
