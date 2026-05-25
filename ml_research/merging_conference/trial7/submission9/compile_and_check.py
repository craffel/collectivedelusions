import subprocess
import pypdf
import sys

def run_compile():
    print("Compiling LaTeX document...")
    res = subprocess.run(["local_env/bin/tectonic", "example_paper.tex"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation FAILED!")
        print("STDOUT:")
        print(res.stdout)
        print("STDERR:")
        print(res.stderr)
        return False
    print("Compilation successful.")
    return True

def check_pdf():
    try:
        reader = pypdf.PdfReader("example_paper.pdf")
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        return False

    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    ref_pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if "References" in text:
            ref_pages.append(idx + 1)
            
    print(f"References found on pages: {ref_pages}")
    
    if not ref_pages:
        print("WARNING: 'References' section not found in PDF!")
        return False
        
    first_ref_page = ref_pages[0]
    print(f"References start on Page: {first_ref_page}")
    
    # We want exactly 8 pages for the main paper, meaning page 1 to 8 have NO references,
    # and page 9 is the FIRST page that contains "References".
    if first_ref_page == 9:
        print("SUCCESS: The main paper is exactly 8 pages, and references start on page 9!")
        return True
    elif first_ref_page < 9:
        print(f"FAILURE: References start too early (Page {first_ref_page}). The main paper needs more content to fill up 8 pages.")
        return False
    else:
        print(f"FAILURE: References start too late (Page {first_ref_page}). The main paper is too long and exceeds 8 pages.")
        return False

if __name__ == "__main__":
    if run_compile():
        success = check_pdf()
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
