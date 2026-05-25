import re

files = ["submission6.txt", "submission8.txt", "submission9.txt"]

def search_patterns(filepath, patterns):
    print(f"=== Searching in {filepath} ===")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by double newlines to find paragraphs
    paragraphs = content.split("\n\n")
    for pattern in patterns:
        print(f"\n--- Matches for pattern: {pattern} ---")
        matches = []
        for p in paragraphs:
            if re.search(pattern, p, re.IGNORECASE):
                # Clean up multiple whitespaces/newlines for display
                clean_p = " ".join(p.split())
                if len(clean_p) > 500:
                    clean_p = clean_p[:500] + "..."
                matches.append(clean_p)
        for i, m in enumerate(matches[:5]): # show top 5 matches
            print(f"{i+1}. {m}")

patterns = [
    r"AdaMerging",
    r"entropy minimization",
    r"equation|formula|eq\.|\b\d\.\d\b",
    r"Fisher",
    r"routing|prototype",
    r"projection|reset"
]

for f in files:
    search_patterns(f, patterns)
