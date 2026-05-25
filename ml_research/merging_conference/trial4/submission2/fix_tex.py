import re

def fix_tex():
    with open('submission.tex', 'r') as f:
        content = f.read()

    # 1. Fix ampersands in titles
    content = content.replace(
        r'\subsection{Benchmark Datasets & Architectures}',
        r'\subsection{Benchmark Datasets \& Architectures}'
    )
    content = content.replace(
        r'\subsection{Hyperparameter Sensitivity & Ablations}',
        r'\subsection{Hyperparameter Sensitivity \& Ablations}'
    )

    # 2. Fix duplicate figure blocks
    # We want to replace the sequence of two figure* environments with a single one.
    # Let's find all occurrences of the figure* environment.
    fig_pattern = r'\\begin{figure\*\}\[t\].*?\\end{figure\*\}'
    figures = re.findall(fig_pattern, content, flags=re.DOTALL)
    print(f"Found {len(figures)} figure* environments.")
    
    if len(figures) >= 2:
        # We can use lambda to avoid escape issues in replacement string
        double_fig_pattern = r'\\begin{figure\*\}\[t\].*?\\end{figure\*\}\s*\\begin{figure\*\}\[t\].*?\\end{figure\*\}'
        content, count = re.subn(double_fig_pattern, lambda m: figures[0], content, flags=re.DOTALL)
        print(f"Replaced {count} duplicate figure sequences with a single one.")

    with open('submission.tex', 'w') as f:
        f.write(content)
    print("submission.tex fixed successfully!")

if __name__ == '__main__':
    fix_tex()
