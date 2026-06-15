import re

def parse_bib(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all bibtex entries: @type{key, ...}
    entries = re.split(r'\n@', content)
    parsed = []
    
    # Handle the first entry if it didn't start with newline
    if len(entries) > 0 and entries[0].strip().startswith('@'):
        entries[0] = entries[0].strip()[1:]
        
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # Extract the key
        match = re.match(r'^([a-zA-Z0-9_-]+)\s*\{([a-zA-Z0-9_-]+)\s*,', entry)
        if match:
            entry_type = match.group(1)
            key = match.group(2)
            parsed.append((key, "@" + entry))
        else:
            # Let's try matching without the type space
            match2 = re.match(r'^([a-zA-Z0-9_-]+)\{([a-zA-Z0-9_-]+)\s*,', entry)
            if match2:
                entry_type = match2.group(1)
                key = match2.group(2)
                parsed.append((key, "@" + entry))
                
    return parsed

def main():
    print("Reading current references...")
    current_refs = parse_bib('submission/references.bib')
    current_keys = set(k for k, _ in current_refs)
    print(f"Current keys: {len(current_keys)}")
    
    extra_sources = [
        'papers/trial2_submission6/references.bib',
        'papers/trial3_submission2/references.bib'
    ]
    
    added_count = 0
    with open('submission/references.bib', 'a', encoding='utf-8') as f_out:
        for src in extra_sources:
            print(f"Reading from {src}...")
            src_refs = parse_bib(src)
            for key, entry in src_refs:
                if key not in current_keys:
                    current_keys.add(key)
                    f_out.write("\n\n" + entry)
                    added_count += 1
                    
    print(f"Successfully added {added_count} unique references! Total unique references: {len(current_keys)}")

if __name__ == '__main__':
    main()
