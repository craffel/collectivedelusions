with open("submission.tex", "r") as f:
    text = f.read()

# Let's count occurrences
print("Braces: { count:", text.count("{"), "} count:", text.count("}"))
print("Brackets: [ count:", text.count("["), "] count:", text.count("]"))
print("Parentheses: ( count:", text.count("("), ") count:", text.count(")"))

# Let's track line by line where braces might be unmatched
stack = []
for i, char in enumerate(text):
    if char == "{":
        stack.append((i, "braces"))
    elif char == "}":
        if stack and stack[-1][1] == "braces":
            stack.pop()
        else:
            print(f"Unmatched closing brace at char {i}")
            
# Print unmatched open braces
if stack:
    for idx, t in stack:
        line_num = text[:idx].count("\n") + 1
        print(f"Unmatched open {t} at line {line_num}: {text[idx:idx+30].replace('\n', ' ')}")
