def check_braces(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    stack = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Ignore comments
        clean_line = ""
        in_comment = False
        j = 0
        while j < len(line):
            if line[j] == '%':
                if j == 0 or line[j-1] != '\\':
                    break
            clean_line += line[j]
            j += 1
            
        for char in clean_line:
            if char == '{':
                stack.append(('{', i+1))
            elif char == '}':
                if not stack:
                    print(f"Error: Unmatched '}}' on line {i+1}")
                else:
                    stack.pop()
                    
    for char, line_num in stack:
        print(f"Error: Unmatched '{char}' on line {line_num}")
    
    if not stack:
        print("All braces are balanced!")

if __name__ == "__main__":
    check_braces("submission.tex")
