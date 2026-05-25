import pypdf

def verify():
    reader = pypdf.PdfReader('paper.pdf')
    num_pages = len(reader.pages)
    print(f"Total Pages: {num_pages}")
    
    ref_page = None
    app_page = None
    unresolved_refs = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        page_num = i + 1
        
        # Check for unresolved references
        if "??" in text:
            unresolved_refs.append(page_num)
            
        # Check where References start
        if "References" in text and ref_page is None:
            # Let's check if it is the section title
            if "REFERENCES" in text.upper() or "References" in text:
                ref_page = page_num
                
        # Check where Appendix starts
        if "Appendix" in text and app_page is None:
            if "APPENDIX" in text.upper() or "Appendix" in text:
                app_page = page_num
                
    print(f"References start on Page: {ref_page}")
    print(f"Appendix starts on Page: {app_page}")
    if unresolved_refs:
        print(f"Unresolved references found on pages: {unresolved_refs}")
    else:
        print("No unresolved references found!")

if __name__ == "__main__":
    verify()
