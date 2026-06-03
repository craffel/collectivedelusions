import os
import subprocess
import shutil

def compile_paper():
    print("Compiling LaTeX paper using Tectonic...")
    # Run tectonic on template/example_paper.tex
    cmd = ["/fsx/craffel/miniconda3/bin/tectonic", "template/example_paper.tex"]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Compilation successful!")
            # Copy generated PDF to the root as submission.pdf
            src_pdf = "template/example_paper.pdf"
            dest_pdf = "submission.pdf"
            if os.path.exists(src_pdf):
                shutil.copy(src_pdf, dest_pdf)
                print(f"Copied {src_pdf} to {dest_pdf}")
            else:
                print(f"Error: Compiled PDF not found at {src_pdf}")
        else:
            print("Compilation failed!")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
    except Exception as e:
        print(f"Error executing Tectonic: {e}")

if __name__ == "__main__":
    compile_paper()
