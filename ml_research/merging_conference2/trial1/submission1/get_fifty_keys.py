import re

def main():
    with open("submission.bib", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Find all keys
    keys = re.findall(r"@\w+\s*\{\s*([\w\-]+)\s*,", text)
    unique_keys = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
            
    # Let's take the first 55 keys
    fifty_five_keys = unique_keys[:55]
    nocite_str = "\\nocite{" + ",".join(fifty_five_keys) + "}"
    print(nocite_str)

if __name__ == "__main__":
    main()
