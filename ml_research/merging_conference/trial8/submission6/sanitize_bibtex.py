import re

with open("example_paper.bib", "r") as f:
    content = f.read()

# Let's find any '&' that are not preceded by '\'
# A negative lookbehind assertion (?<!\\)& matches '&' if not preceded by '\'
pattern = r"(?<!\\)&"
matches = re.findall(pattern, content)
print(f"Found {len(matches)} unescaped '&' characters.")

# Let's replace them
sanitized_content = re.sub(pattern, r"\&", content)

# Also check for other special characters like '%' if unescaped
# (though '%' is usually rare in bibtex field values, let's look for (?<!\\)%)
percent_pattern = r"(?<!\\)%"
percent_matches = re.findall(percent_pattern, sanitized_content)
print(f"Found {len(percent_matches)} unescaped '%' characters.")

with open("example_paper.bib", "w") as f:
    f.write(sanitized_content)

print("Sanitization of example_paper.bib complete!")
