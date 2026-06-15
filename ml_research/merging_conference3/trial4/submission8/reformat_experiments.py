import re

file_path = "submission/sections/04_experiments.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Locate the start and end of the paragraph
paragraph_start_marker = r"\paragraph{Unconstrained Flatness Regularization Limitations.}"
section_end_marker = r"\subsection{Quantization Schema Sensitivity Analysis}"

start_idx = content.find(paragraph_start_marker)
end_idx = content.find(section_end_marker)

print("Start index:", start_idx)
print("End index:", end_idx)

if start_idx != -1 and end_idx != -1:
    extracted_paragraph = content[start_idx:end_idx]
    print("Extracted paragraph length:", len(extracted_paragraph))
    print("Extracted paragraph preview:", extracted_paragraph[:200])
    print("Extracted paragraph end preview:", extracted_paragraph[-200:])
else:
    print("Markers not found!")
