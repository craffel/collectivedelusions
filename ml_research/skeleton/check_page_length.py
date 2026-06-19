import sys
import os
import re
import subprocess

def check_page_length(tex_file):
    with open(tex_file, 'r') as f:
        content = f.read()
    
    hook_code = """
\\AddToHook{cmd/appendix/before}{\\label{checkpagelength_lastpage}}
\\AddToHook{cmd/bibliography/before}{\\label{checkpagelength_lastpage}}
\\AddToHook{cmd/printbibliography/before}{\\label{checkpagelength_lastpage}}
\\AddToHook{env/thebibliography/before}{\\label{checkpagelength_lastpage}}
\\AtEndDocument{\\label{checkpagelength_lastpage}}
"""

    match = re.search(r'\\begin\{document\}', content)
    if match:
        insert_pos = match.start()
    else:
        insert_pos = 0
            
    modified_content = content[:insert_pos] + hook_code + content[insert_pos:]
    
    tmp_tex = tex_file.replace('.tex', '_check_length.tex')
    with open(tmp_tex, 'w') as f:
        f.write(modified_content)
        
    print(f"Compiling {tmp_tex} with tectonic...")
    # Run from the same directory as tex_file
    work_dir = os.path.dirname(os.path.abspath(tex_file))
    subprocess.run(['tectonic', '-k', os.path.basename(tmp_tex)], cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    aux_file = tmp_tex.replace('.tex', '.aux')
    page_length = -1
    if os.path.exists(aux_file):
        with open(aux_file, 'r') as f:
            for line in f:
                match = re.search(r'\\newlabel{checkpagelength_lastpage}{{.*?}{([^}]+)', line)
                if match:
                    try:
                        page_length = int(match.group(1))
                        # The first match is the earliest hook that fired
                        break
                    except ValueError:
                        pass
    
    for ext in ['.tex', '.aux', '.pdf', '.log', '.out', '.bbl', '.blg', '.synctex.gz', '.fls', '.fdb_latexmk']:
        f = tmp_tex.replace('.tex', ext)
        if os.path.exists(f):
            os.remove(f)
            
    if page_length != -1:
        print(f"Main text page length: {page_length}")
        if page_length > 8:
            print("Status: FAILED (Exceeds 8 pages)")
            sys.exit(1)
        else:
            print("Status: PASSED (Within 8 pages)")
            sys.exit(0)
    else:
        print("Error: Could not determine page length.")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python check_page_length.py <main_tex_file>")
        sys.exit(1)
    check_page_length(sys.argv[1])
