with open("submission.bib", "r") as f:
    text = f.read()

print("Bib Braces: { count:", text.count("{"), "} count:", text.count("}"))

stack = []
for i, char in enumerate(text):
    if char == "{":
        stack.append(i)
    elif char == "}":
        if stack:
            stack.pop()
        else:
            print(f"Unmatched Bib closing brace at char {i}")
if stack:
    for idx in stack:
        print(f"Unmatched Bib open brace at char {idx}: {text[idx:idx+30]}")
