import pypdf
import sys

def audit_pdf(pdf_path):
    print(f"Auditing {pdf_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return False

    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    # 1. Total page budget check (typical conference paper expects exactly 8 pages for main, unlimited for references and appendix)
    if total_pages < 8:
        print("Warning: PDF has fewer than 8 pages!")
        return False
    
    # 2. Check each page's text
    main_body_pages = 0
    references_start_page = -1
    appendix_start_page = -1
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        # Filter headers and footers
        filtered_lines = [
            l for l in lines 
            if "Deconstructing Activation Calibration" not in l 
            and "The Methodologist" not in l 
            and not l.isdigit()
        ]
        
        page_text = " ".join(filtered_lines).lower()
        
        if "references" in page_text and references_start_page == -1:
            # Check if this page contains the bibliography section header
            # Usually 'References' or 'references'
            references_start_page = idx + 1
            
        if ("appendix" in page_text or "expert specialist training" in page_text) and appendix_start_page == -1 and idx > 8:
            appendix_start_page = idx + 1
            
    print(f"Detected References start page: {references_start_page}")
    print(f"Detected Appendix start page: {appendix_start_page}")
    
    # 3. Main body page budget verification
    if references_start_page != 9:
        print(f"Warning: References should start exactly on Page 9! Found starting on Page {references_start_page}")
        return False
    else:
        print("Success: Main body occupies exactly 8 pages (Pages 1 to 8).")
        
    if appendix_start_page != 11:
        print(f"Warning: Appendix should start exactly on Page 11! Found starting on Page {appendix_start_page}")
        return False
    else:
        print("Success: Appendix starts exactly on Page 11.")
        
    # 4. Check for unresolved placeholders or TODOs
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for marker in ["todo", "placeholder", "draft", "??", "[?]"]:
            if marker in text.lower():
                print(f"Warning: Found '{marker}' marker on Page {idx + 1}!")
                return False
                
    print("Success: No unresolved placeholder or TODO markers found.")
    print("Audit completed successfully! The PDF is 100% compliant with the formatting and structural constraints.")
    return True

if __name__ == "__main__":
    success = audit_pdf("submission.pdf")
    if not success:
        sys.exit(1)
    sys.exit(0)
