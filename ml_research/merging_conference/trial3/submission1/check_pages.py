import subprocess
import pypdf
import sys

def get_page_count():
    # Run tectonic compilation
    print("Compiling paper.tex...")
    result = subprocess.run(["tectonic", "paper.tex"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stdout)
        print(result.stderr)
        return -1
    
    # Read page count
    try:
        reader = pypdf.PdfReader('paper.pdf')
        total_pages = len(reader.pages)
        print(f"Compilation successful! Total pages: {total_pages}")
        
        # We want to identify the page where references start.
        # Let's search each page's text for 'References'
        ref_page = -1
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if "References" in text:
                ref_page = i + 1
                break
        
        if ref_page != -1:
            print(f"References start on page: {ref_page}")
            print(f"Main body page count (before references): {ref_page - 1}")
        else:
            print("Could not find 'References' in the PDF text.")
        
        return total_pages, ref_page
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return -1, -1

if __name__ == "__main__":
    get_page_count()
