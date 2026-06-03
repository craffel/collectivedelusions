import pypdf

def verify():
    reader = pypdf.PdfReader("submission.pdf")
    found_issues = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        page_num = idx + 1
        # Check for [?]
        if "[?]" in text:
            found_issues.append(f"Page {page_num}: Found '[?]'" )
        # Check for ?? (often indicates a broken cross-reference)
        # We need to be careful with things like double question marks in actual text,
        # but in research papers, "?? " or " ??" usually indicates a broken reference.
        matches = list(re_find_all_positions(r"\?\?", text))
        if matches:
            found_issues.append(f"Page {page_num}: Found '??' at {matches}")
            
    if found_issues:
        print("Verification FAILED:")
        for issue in found_issues:
            print("  ", issue)
    else:
        print("Verification PASSED: No [?] or ?? found in submission.pdf!")

def re_find_all_positions(pattern, text):
    import re
    return [m.start() for m in re.finditer(pattern, text)]

if __name__ == "__main__":
    verify()
