with open("example_paper.bib", "r", encoding="utf-8") as f:
    text = f.read()

# Replace special characters
replacements = {
    "‐": "-",
    "•. ": "",
    "•": "",
    "’": "'",
    "ü": '\\"{u}',
    "š": "\\v{s}",
    "á": "\\'{a}",
    "é": "\\'{e}",
    "í": "\\'{i}",
    "ó": "\\'{o}",
    "ú": "\\'{u}",
    "ã": "\\~{a}",
    "ñ": "\\~{n}",
    "ö": '\\"{o}',
    "Ä": '\\"{A}',
    "Ö": '\\"{O}',
    "Ü": '\\"{U}',
    "ć": "\\'{c}",
    "č": "\\v{c}",
    "ł": "\\l{}",
    "ř": "\\v{r}",
    "ž": "\\v{z}",
}

for src, dst in replacements.items():
    text = text.replace(src, dst)

with open("example_paper.bib", "w", encoding="utf-8") as f:
    f.write(text)

print("Cleaned up special characters in example_paper.bib!")
