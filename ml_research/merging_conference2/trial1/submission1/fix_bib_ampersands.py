def main():
    with open("submission.bib", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Replace any & with \& if it is not already escaped
    # Since there are only 4 raw & characters, we can just replace them or use regex
    import re
    # Match & only if not preceded by \
    fixed_text = re.sub(r"(?<!\\)&", r"\&", text)
    
    with open("submission.bib", "w", encoding="utf-8") as f:
        f.write(fixed_text)
        
    print("Fixed unescaped ampersands in submission.bib")

if __name__ == "__main__":
    main()
