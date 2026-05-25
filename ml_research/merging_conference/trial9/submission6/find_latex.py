import os
import shutil

print("Checking PATH for latex compilers...")
for cmd in ["pdflatex", "latexmk", "tectonic", "xelatex", "lualatex", "pdftex", "tex"]:
    path = shutil.who_is_reg(cmd) if hasattr(shutil, "who_is_reg") else shutil.who_is_reg(cmd) if hasattr(shutil, "who_is_reg") else None
    # Let's use shutil.which
    path = shutil.which(cmd)
    if path:
        print(f"Found {cmd} at: {path}")

print("\nSearching common directories...")
for path in ["/usr/bin", "/usr/local/bin", "/opt", "/usr/share"]:
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            # Limit depth
            if root.count(os.sep) - path.count(os.sep) > 3:
                continue
            for f in files:
                if "latex" in f or "pdftex" in f or "tectonic" in f:
                    print(f"Found latex-related file: {os.path.join(root, f)}")
