with open("submission.tex", "r") as f:
    text = f.read()

stack = []
pairs = {
    "}": "{",
    "]": "[",
    ")": "("
}

for idx, char in enumerate(text):
    if char in "{[(":
        stack.append((char, idx))
    elif char in "}])":
        expected = pairs[char]
        if not stack:
            print(f"Unmatched closing '{char}' at char {idx}")
        else:
            top_char, top_idx = stack.pop()
            if top_char != expected:
                line_num = text[:idx].count("\n") + 1
                top_line_num = text[:top_idx].count("\n") + 1
                print(f"Mismatched nesting at line {line_num}: closed '{char}' but expected '{top_char}' from line {top_line_num}")
                print(f"  Opening context: {text[top_idx:top_idx+30].replace('\n', ' ')}")
                print(f"  Closing context: {text[idx-15:idx+15].replace('\n', ' ')}")

if stack:
    for char, idx in stack:
        line_num = text[:idx].count("\n") + 1
        print(f"Unclosed '{char}' at line {line_num}: {text[idx:idx+30].replace('\n', ' ')}")
