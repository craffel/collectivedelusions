import subprocess
import os
import shutil

def compile():
    print("Compiling LaTeX paper...")
    orig_dir = os.getcwd()
    os.chdir("template")
    
    try:
        # Try running tectonic first (extremely robust, auto-manages bibtex and packages)
        print("Attempting compilation with tectonic...")
        subprocess.run(["tectonic", "example_paper.tex"], check=True)
        print("Tectonic compilation successful!")
        shutil.copy("example_paper.pdf", "../submission.pdf")
        print("Successfully copied compiled paper to root as submission.pdf!")
        return
    except Exception as e_tectonic:
        print(f"Tectonic failed or not available: {e_tectonic}")
        print("Falling back to pdflatex + bibtex...")
        
    try:
        # Fallback to pdflatex
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "example_paper.tex"], check=True)
        subprocess.run(["bibtex", "example_paper"], check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "example_paper.tex"], check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "example_paper.tex"], check=True)
        
        print("pdflatex compilation successful!")
        shutil.copy("example_paper.pdf", "../submission.pdf")
        print("Successfully copied compiled paper to root as submission.pdf!")
    except Exception as e_pdf:
        print(f"Error during fallback compilation: {e_pdf}")
    finally:
        os.chdir(orig_dir)

if __name__ == "__main__":
    compile()
