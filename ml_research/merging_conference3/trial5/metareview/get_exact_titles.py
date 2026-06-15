import os
import re

base_dir = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/metareview"

def extract_title_from_tex(tex_path):
    if not os.path.exists(tex_path):
        return None
    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Search for \title{...}
    # Handing potential newlines and nested braces
    match = re.search(r'\\title\{([^}]+)\}', content)
    if match:
        title = match.group(1).strip()
        # Clean up some latex commands if any
        title = re.sub(r'\s+', ' ', title)
        title = title.replace('\\\\', ' ')
        title = title.replace('\\icmltitle', '')
        return title
    
    # Sometimes it is \icmltitle{...}
    match = re.search(r'\\icmltitle\{([^}]+)\}', content)
    if match:
        title = match.group(1).strip()
        title = re.sub(r'\s+', ' ', title)
        title = title.replace('\\\\', ' ')
        return title
        
    return None

for n in range(1, 11):
    sub_dir = os.path.join(base_dir, f"submission{n}", "submission")
    if os.path.exists(sub_dir):
        # Find any .tex files
        tex_files = [f for f in os.listdir(sub_dir) if f.endswith('.tex')]
        found_title = None
        for tf in tex_files:
            title = extract_title_from_tex(os.path.join(sub_dir, tf))
            if title:
                found_title = title
                break
        print(f"Submission {n}: {found_title or 'Not Found in Tex'}")
