import re

def get_cited_keys(tex_path):
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Find all \cite{...} or \cite[...]{...} etc.
    # Pattern to match \cite{key1,key2}
    cites = re.findall(r"\\cite(?:\[.*?\])?\{(.*?)\}", content)
    keys = set()
    for c in cites:
        for k in c.split(","):
            keys.add(k.strip())
    return keys

def get_bib_keys(bib_path):
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Find all @type{key,
    keys = re.findall(r"@\w+\s*\{\s*([\w\-]+)\s*,", content)
    return set(keys)

def main():
    tex_keys = get_cited_keys("submission.tex")
    bib_keys = get_bib_keys("submission.bib")
    
    print("Cited keys in TeX:", tex_keys)
    print(f"Number of keys in Bib: {len(bib_keys)}")
    
    missing = tex_keys - bib_keys
    print("Missing keys:", missing)

if __name__ == "__main__":
    main()
