import re

def check_environments():
    with open("example_paper.tex", "r") as f:
        content = f.read()
        
    begins = re.findall(r'\\begin\{(.*?)\}', content)
    ends = re.findall(r'\\end\{(.*?)\}', content)
    
    print(f"Begins: {begins}")
    print(f"Ends: {ends}")
    
    stack = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line_num = i + 1
        matches = re.finditer(r'\\(begin|end)\{(.*?)\}', line)
        for m in matches:
            cmd = m.group(1)
            env = m.group(2)
            if cmd == 'begin':
                stack.append((env, line_num))
            else:
                if not stack:
                    print(f"Error: Unmatched \\end{{{env}}} at line {line_num}")
                else:
                    top_env, top_line = stack.pop()
                    if top_env != env:
                        print(f"Error: Mismatched environment. Expected \\end{{{top_env}}} (from line {top_line}), but found \\end{{{env}}} at line {line_num}")
                        
    if stack:
        print(f"Error: Unclosed environments remaining:")
        for env, line_num in stack:
            print(f"  \\begin{{{env}}} at line {line_num}")
    else:
        print("Success: All environments are properly closed!")

if __name__ == "__main__":
    check_environments()
