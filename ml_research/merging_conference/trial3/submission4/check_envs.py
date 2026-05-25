import re

with open("submission.tex") as f:
    text = f.read()

# Find all \begin{...} and \end{...}
pattern = r"\\(begin|end)\{([a-zA-Z*]+)\}"

stack = []
for m in re.finditer(pattern, text):
    tag_type, env = m.groups()
    if tag_type == "begin":
        stack.append((env, m.start()))
    else:
        if stack:
            open_env, pos = stack.pop()
            if open_env != env:
                print(f"Mismatched environment: began '{open_env}' at position {pos}, but ended '{env}' at position {m.start()}")
                line_num = text.count('\n', 0, pos) + 1
                print(f"  Began on line {line_num}: ... {text[pos:pos+40]} ...")
        else:
            print(f"Extra end environment '{env}' at position {m.start()}")
            line_num = text.count('\n', 0, m.start()) + 1
            print(f"  Ended on line {line_num}")

if stack:
    print("\nUnclosed environments:")
    for env, pos in stack:
        line_num = text.count('\n', 0, pos) + 1
        print(f"  '{env}' starting on line {line_num}: ... {text[pos:pos+40]} ...")
else:
    print("\nAll environments are perfectly balanced and nested!")
