import re

def get_cited_keys(tex_file):
    cited_keys = set()
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Match \cite{key1, key2} etc.
    # regex matches \cite{...} or \citet{...} or \citep{...}
    matches = re.findall(r'\\cite[a-z]*\{([^}]+)\}', content)
    for match in matches:
        # Split by comma and clean whitespace
        keys = [k.strip() for k in match.split(',')]
        for k in keys:
            if k:
                cited_keys.add(k)
    return cited_keys

def get_bib_keys(bib_file):
    bib_keys = set()
    with open(bib_file, 'r') as f:
        content = f.read()
    
    # Match @article{key, or @inproceedings{key, etc.
    matches = re.findall(r'@[a-zA-Z]+\s*\{\s*([^,\s]+)', content)
    for match in matches:
        bib_keys.add(match.strip())
    return bib_keys

def verify():
    cited = get_cited_keys('submission.tex')
    bib = get_bib_keys('submission.bib')
    
    print(f"Total unique keys cited in .tex: {len(cited)}")
    print(f"Total unique keys defined in .bib: {len(bib)}")
    
    missing = cited - bib
    if missing:
        print("\n[WARNING] The following keys are cited in .tex but missing in .bib:")
        for k in sorted(missing):
            print(f"  - {k}")
    else:
        print("\n[SUCCESS] All cited keys are present in .bib!")
        
    unused = bib - cited
    print(f"\nTotal unused keys in .bib: {len(unused)}")

if __name__ == '__main__':
    verify()
