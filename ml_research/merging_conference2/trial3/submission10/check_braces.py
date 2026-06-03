def check_braces():
    with open("example_paper.tex", "r") as f:
        lines = f.readlines()
        
    stack = []
    for i, line in enumerate(lines):
        line_num = i + 1
        # Strip comments
        if '%' in line:
            # Find uncommented %
            # Simple check
            uncommented = ""
            escaped = False
            for char in line:
                if char == '\\':
                    escaped = not escaped
                    uncommented += char
                elif char == '%':
                    if escaped:
                        uncommented += char
                        escaped = False
                    else:
                        break
                else:
                    escaped = False
                    uncommented += char
            line = uncommented
            
        for pos, char in enumerate(line):
            if char == '{':
                stack.append((line_num, pos + 1))
            elif char == '}':
                if not stack:
                    print(f"Error: Unmatched closing brace '}}' at line {line_num}, col {pos + 1}")
                else:
                    stack.pop()
                    
    if stack:
        print(f"Error: {len(stack)} unmatched opening braces '{{' left on stack:")
        for line_num, col in stack[:10]:
            print(f"  Line {line_num}, col {col}")
    else:
        print("Success: All curly braces are balanced!")

if __name__ == "__main__":
    check_braces()
