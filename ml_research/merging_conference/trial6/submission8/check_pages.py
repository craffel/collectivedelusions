import subprocess
import pypdf

def compile_and_check():
    print("Compiling LaTeX paper...")
    res = subprocess.run(["tectonic", "submission.tex"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation failed!")
        print(res.stderr)
        return False
    
    print("Reading PDF...")
    try:
        reader = pypdf.PdfReader('submission.pdf')
        total_pages = len(reader.pages)
        ref_page = -1
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if 'References' in text:
                ref_page = i + 1
                break
        print(f"Total pages: {total_pages}")
        if ref_page != -1:
            print(f"References start on page: {ref_page}")
            print(f"Main paper pages: {ref_page - 1}")
        else:
            print("References section not found in PDF!")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return False
    return True

if __name__ == "__main__":
    compile_and_check()
