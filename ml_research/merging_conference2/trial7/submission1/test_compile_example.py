import subprocess
import shutil
import os

# Copy sty and bst files to template folder so tectonic can find them if needed, or compile in root
print("Compiling template/example_paper.tex...")
# Copy example_paper.tex to root to make it easy to find style sheets
shutil.copy("template/example_paper.tex", "test_example.tex")
shutil.copy("template/example_paper.bib", "test_example.bib")

res = subprocess.run(["tectonic", "test_example.tex"], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Exit Code:", res.returncode)
