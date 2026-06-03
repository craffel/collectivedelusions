import os
import subprocess
import pypdf

def compile_and_get_pages():
    print("Compiling LaTeX...")
    try:
        res = subprocess.run(["tectonic", "submission.tex"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        print(e.stderr)
        return None, None
        
    if not os.path.exists("submission.pdf"):
        print("submission.pdf not found!")
        return None, None
        
    reader = pypdf.PdfReader("submission.pdf")
    total_pages = len(reader.pages)
    
    # Let's find which page contains the bibliography
    references_page = None
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if "References" in text or "thebibliography" in text:
            # We want the page index (1-based)
            references_page = idx + 1
            break
            
    print(f"Total Pages: {total_pages}")
    print(f"References start on page: {references_page}")
    return total_pages, references_page

if __name__ == "__main__":
    compile_and_get_pages()
