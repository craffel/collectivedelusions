import os
import textwrap

def wrap_file(file_path):
    print(f"Wrapping {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        # Keep newline character at the end separately
        stripped_line = line.rstrip("\r\n")
        
        # If line length is <= 100 characters or starts with specific commands we want to keep together
        if len(stripped_line) <= 100:
            new_lines.append(line)
        else:
            # We wrap the line at 80 characters
            # Keep indentation
            leading_spaces = len(stripped_line) - len(stripped_line.lstrip())
            indent = " " * leading_spaces
            
            wrapped = textwrap.wrap(
                stripped_line.strip(),
                width=80 - len(indent),
                break_long_words=False,
                break_on_hyphens=False
            )
            
            if not wrapped:
                new_lines.append(line)
            else:
                for wl in wrapped:
                    new_lines.append(indent + wl + "\n")
                    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    sections_dir = "submission/sections"
    for filename in os.listdir(sections_dir):
        if filename.endswith(".tex"):
            wrap_file(os.path.join(sections_dir, filename))
    print("Done wrapping long lines.")
