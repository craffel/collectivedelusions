import re
import subprocess

def format_tex():
    with open("example_paper.tex", "r") as f:
        content = f.read()

    # 1. Replace all markdown bold formatting **text** with \textbf{text}
    formatted = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
    
    # 2. Fix the manual item list in Section 5.2 (Discovery 2)
    old_list_section = r"""By sweeping the sparsity threshold $\tau$ in SMACS, we map out a classic U-shaped performance curve (visualized in Table~\ref{tab:results}). 
* As we lower $\tau$ from $1.1$ (LSC) to $0.50$, the average accuracy increases from \textbf{34.16\%} to a peak of \textbf{39.84\%} (an absolute improvement of \textbf{+5.68\%}). This is driven by the fact that stable active channels are given higher-capacity channel-wise scaling, correcting localized variance collapse.
* As we lower $\tau$ below $0.50$ (e.g., $0.30$ and $0.10$), the accuracy drops rapidly to \textbf{32.04\%} and then \textbf{26.79\%}, eventually merging with the collapsed TCAC baseline. This happens because the threshold is too permissive, allowing unstable, highly sparse channels to receive catastrophic scaling.
This confirms"""

    new_list_section = r"""By sweeping the sparsity threshold $\tau$ in SMACS, we map out a classic U-shaped performance curve (visualized in Table~\ref{tab:results}). 
\begin{itemize}
    \item As we lower $\tau$ from $1.1$ (LSC) to $0.50$, the average accuracy increases from \textbf{34.16\%} to a peak of \textbf{39.84\%} (an absolute improvement of \textbf{+5.68\%}). This is driven by the fact that stable active channels are given higher-capacity channel-wise scaling, correcting localized variance collapse.
    \item As we lower $\tau$ below $0.50$ (e.g., $0.30$ and $0.10$), the accuracy drops rapidly to \textbf{32.04\%} and then \textbf{26.79\%}, eventually merging with the collapsed TCAC baseline. This happens because the threshold is too permissive, allowing unstable, highly sparse channels to receive catastrophic scaling.
\end{itemize}
This confirms"""

    # Do the replacement if found
    if old_list_section in formatted:
        formatted = formatted.replace(old_list_section, new_list_section)
        print("Successfully replaced Section 5.2 list with proper latex itemize.")
    else:
        # Try a slightly more relaxed match if exact string matching fails
        print("Warning: Section 5.2 list exact replacement did not match. Trying regex-based list replacement.")
        # Let's replace the lists starting with '* ' inside results section
        formatted = re.sub(
            r'\*\s+As we lower(.*?)\.\n\*\s+As we lower(.*?)\.\nThis confirms',
            r'\\begin{itemize}\n    \\item As we lower\1.\n    \\item As we lower\2.\n\\end{itemize}\nThis confirms',
            formatted
        )

    with open("example_paper.tex", "w") as f:
        f.write(formatted)
    print("example_paper.tex has been formatted and saved.")

if __name__ == "__main__":
    format_tex()
