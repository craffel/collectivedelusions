import re

def parse_txt_headings(txt_path):
    print(f"\n==================== Headings in {txt_path} ====================")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    heading_pattern = re.compile(r"^(?:\d+\.|\d+\.\d+\.)\s+[A-Z].*")
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if heading_pattern.match(clean_line):
            print(f"Line {i+1:04d}: {clean_line}")

parse_txt_headings("papers/submission2.txt")
parse_txt_headings("papers/submission7.txt")
parse_txt_headings("papers/submission8.txt")
