import re

def fix_bibtex_file(filename="submission.bib"):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Replace unescaped & with \&
    # We want to match '&' only if it's not preceded by '\'
    fixed_content = re.sub(r'(?<!\\)&', r'\\&', content)
    
    # Also, replace % characters inside title or journal with \% if not escaped
    fixed_content = re.sub(r'(?<!\\)%', r'\\%', fixed_content)
    
    # Let's clean any non-ASCII characters that can cause issues, or replace them
    # For example, curly quotes or other unicode chars
    # We can use unicode representation or replace with simple LaTeX equivalents if possible,
    # or just keep them as ASCII if they are special characters.
    # Actually, let's keep it simple: replace common non-ASCII characters
    replacements = {
        '“': "``",
        '”': "''",
        '‘': "`",
        '’': "'",
        '–': "--",
        '—': "---",
    }
    for orig, rep in replacements.items():
        fixed_content = fixed_content.replace(orig, rep)
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write(fixed_content)
    print("Successfully fixed & and other characters in bibtex!")

if __name__ == "__main__":
    fix_bibtex_file()
