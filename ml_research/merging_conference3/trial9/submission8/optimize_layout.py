import os
import subprocess

def optimize_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace vskips and table spacing
    content = content.replace(r'\vskip 0.15in', r'\vskip 0.05in')
    content = content.replace(r'\vskip 0.1in', r'\vskip 0.02in')
    
    # Add negative vspace before subsections to pull text up
    # We do this by searching for \subsection{
    # But be careful not to double add if we run it multiple times
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith(r'\subsection{') and not (new_lines and r'\vspace' in new_lines[-1]):
            new_lines.append(r'\vspace{-5pt}')
            new_lines.append(line)
        elif line.strip().startswith(r'\begin{equation}') and not (new_lines and r'\vspace' in new_lines[-1]):
            new_lines.append(r'\vspace{-3pt}')
            new_lines.append(line)
        elif line.strip().startswith(r'\end{equation}') and not (new_lines and r'\vspace' in new_lines[-1]):
            new_lines.append(line)
            new_lines.append(r'\vspace{-3pt}')
        elif line.strip().startswith(r'\begin{figure*}'):
            new_lines.append(line)
            new_lines.append(r'\vspace{-5pt}')
        elif line.strip().startswith(r'\end{figure*}'):
            new_lines.append(r'\vspace{-10pt}')
            new_lines.append(line)
        elif line.strip().startswith(r'\begin{table*}'):
            new_lines.append(line)
            new_lines.append(r'\vspace{-5pt}')
        elif line.strip().startswith(r'\end{table*}'):
            new_lines.append(r'\vspace{-10pt}')
            new_lines.append(line)
        else:
            new_lines.append(line)
            
    content = '\n'.join(new_lines)
    
    with open(filepath, 'w') as f:
        f.write(content)

# Apply optimization to the section files
sections_dir = 'submission/sections'
for filename in os.listdir(sections_dir):
    if filename.endswith('.tex') and filename != 'appendix.tex':
        optimize_file(os.path.join(sections_dir, filename))

print("Layout optimization applied successfully.")
