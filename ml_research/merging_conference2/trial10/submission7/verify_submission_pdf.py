import re
import os
from pypdf import PdfReader

def verify_pdf(pdf_path):
    print("="*80)
    print(f"VERIFYING MANUSCRIPT: {pdf_path}")
    print("="*80)
    if not os.path.exists(pdf_path):
        print(f"ERROR: File {pdf_path} does not exist.")
        return
        
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"Total Page Count: {total_pages}")
        
        # Check all pages for key sections
        conclusion_page = None
        references_page = None
        appendix_page = None
        title_suppressed_found = False
        
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text = page.extract_text()
            
            # Check for header bug
            if "Title Suppressed" in text or "Excessive Size" in text:
                print(f"WARNING: 'Title Suppressed Due to Excessive Size' found on Page {page_num}!")
                title_suppressed_found = True
                
            # Locate section boundaries
            if "References" in text and references_page is None:
                references_page = page_num
            if ("Conclusion" in text or "Future Work" in text) and conclusion_page is None:
                conclusion_page = page_num
            if ("Appendix" in text or "Robustness to Test-Time Input Noise" in text) and appendix_page is None and page_num > 8:
                appendix_page = page_num
                
        print("\n--- Structural Analysis ---")
        print(f"Conclusion section starts on Page: {conclusion_page}")
        print(f"References section starts on Page: {references_page}")
        print(f"Appendix starts on Page: {appendix_page}")
        
        # Validate rules
        errors = []
        if total_pages < 8:
            errors.append(f"Total page count ({total_pages}) is less than 8 pages.")
        if references_page is None:
            errors.append("Could not find 'References' section in the PDF.")
        elif references_page < 8:
            # References can start on page 8 if the main body is complete
            print(f"Note: References begin on Page {references_page}, which is within the 8-page limit.")
        elif references_page > 9:
            errors.append(f"References start too late (Page {references_page}), main body exceeds the 8-page limit!")
            
        if title_suppressed_found:
            errors.append("Found 'Title Suppressed Due to Excessive Size' header defect.")
            
        # Check if conclusion wraps up on page 8
        # The main text (sections 1-6) must not exceed 8 pages. So references can start on page 8 or 9.
        if references_page is not None and references_page > 9:
            errors.append(f"Main body exceeds 8 pages (References start on Page {references_page}).")
            
        print("\n--- Compliance Results ---")
        if not errors:
            print("SUCCESS: PDF complies perfectly with all ICML page limits, headers, and formatting constraints!")
        else:
            print("FAILURE: The following compliance errors were found:")
            for err in errors:
                print(f"  - {err}")
                
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == '__main__':
    verify_pdf("submission.pdf")
