import pypdf
import os

def check_pdf(pdf_path):
    print(f"Checking {pdf_path} for layout and citation issues...")
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist.")
        return
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    issues = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        page_num = i + 1
        
        # Check for unresolved references
        if "??" in text:
            issues.append(f"Page {page_num}: Found unresolved reference '??'")
        
        # Check for missing citations
        if "[?]" in text:
            issues.append(f"Page {page_num}: Found missing citation '[?]'")
            
        # Check for "Title Suppressed" error
        if "Title Suppressed" in text:
            issues.append(f"Page {page_num}: Found 'Title Suppressed' warning in running header")
            
        # Check for TODOs
        if "TODO" in text.upper():
            issues.append(f"Page {page_num}: Found TODO comment")
            
    if issues:
        print("\n--- ISSUES FOUND ---")
        for issue in issues:
            print(issue)
    else:
        print("\nNo issues (??, [?], Title Suppressed, or TODOs) found. PDF looks pristine!")

if __name__ == "__main__":
    check_pdf("submission.pdf")
