import subprocess

# The list of lines in the preamble
preamble_lines = [
    r"\documentclass{article}",
    r"\usepackage{microtype}",
    r"\usepackage{graphicx}",
    r"\usepackage{subcaption}",
    r"\usepackage{booktabs}",
    r"\usepackage{hyperref}",
    r"\newcommand{\theHalgorithm}{\arabic{algorithm}}",
    r"\usepackage{icml2026}",
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{mathtools}",
    r"\usepackage{amsthm}",
    r"\usepackage[capitalize,noabbrev]{cleveref}",
    r"\theoremstyle{plain}",
    r"\newtheorem{theorem}{Theorem}[section]",
    r"\newtheorem{proposition}[theorem]{Proposition}",
    r"\newtheorem{lemma}[theorem]{Lemma}",
    r"\newtheorem{corollary}[theorem]{Corollary}",
    r"\theoremstyle{definition}",
    r"\newtheorem{definition}[theorem]{Definition}",
    r"\newtheorem{assumption}[theorem]{Assumption}",
    r"\theoremstyle{remark}",
    r"\newtheorem{remark}[theorem]{Remark}",
    r"\icmltitlerunning{The Optimizer Confounder: Deconstructing SGD vs. AdamW in Model Merging}",
]

def test_compile(lines):
    latex_code = "\n".join(lines) + "\n\\begin{document}\nHello\n\\end{document}\n"
    with open("test_sub.tex", "w") as f:
        f.write(latex_code)
    res = subprocess.run(["tectonic", "test_sub.tex"], capture_output=True, text=True)
    return res.returncode == 0, res.stderr

# Test empty list (no packages at all)
ok, err = test_compile([r"\documentclass{article}"])
print("Only documentclass compile:", ok)

# Let's add them one by one and find where it breaks!
active_lines = [r"\documentclass{article}"]
for line in preamble_lines[1:]:
    active_lines.append(line)
    ok, err = test_compile(active_lines)
    print(f"Added '{line}': compile =", ok)
    if not ok:
        print("Breaking error:", err.split("\n")[-5:])
        break
