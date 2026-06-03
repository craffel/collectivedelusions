import subprocess
import os

def compile_paper():
    print("=== Compiling submission.tex using Tectonic ===")
    cmd = ["/fsx/craffel/miniconda3/bin/tectonic", "submission.tex"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Tectonic stdout:")
        print(res.stdout)
        print("Tectonic stderr:")
        print(res.stderr)
        print("Compilation successful!")
    except subprocess.CalledProcessError as e:
        print("Compilation failed with exit code", e.returncode)
        print("Stdout:")
        print(e.stdout)
        print("Stderr:")
        print(e.stderr)
        return False

    if os.path.exists("submission.pdf"):
        print("Success! submission.pdf generated in current directory.")
        print("File size:", os.path.getsize("submission.pdf"), "bytes")
        return True
    else:
        print("Error: submission.pdf not found after successful compilation!")
        return False

if __name__ == "__main__":
    compile_paper()
