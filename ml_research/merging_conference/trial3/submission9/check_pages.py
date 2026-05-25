import subprocess
import os

def check_pages():
    print("Compiling LaTeX document with Tectonic...")
    result = subprocess.run(["tectonic", "submission.tex"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stdout)
        print(result.stderr)
        return None
    
    try:
        import pypdf
        reader = pypdf.PdfReader("submission.pdf")
        page_count = len(reader.pages)
        print(f"Compilation successful. Total PDF page count: {page_count}")
        
        # Search for "References" section
        references_page = -1
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if "References" in text:
                references_page = i + 1
                break
                
        if references_page != -1:
            print(f"References section starts on page: {references_page}")
            main_text_pages = references_page - 1
            print(f"Main text length: {main_text_pages} pages")
            if main_text_pages == 8:
                print("SUCCESS: Main text is exactly 8 pages long!")
            elif main_text_pages < 8:
                print(f"Main text is too short: {main_text_pages} pages (needs to be exactly 8)")
            else:
                print(f"Main text is too long: {main_text_pages} pages (needs to be exactly 8)")
        else:
            print("Could not find 'References' in the PDF text.")
            
        return page_count
    except Exception as e:
        print(f"Error checking pages: {e}")
        return None

if __name__ == "__main__":
    check_pages()
