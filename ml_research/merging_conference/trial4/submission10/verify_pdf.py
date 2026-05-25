import pypdf

def main():
    try:
        reader = pypdf.PdfReader("submission.pdf")
        pages = len(reader.pages)
        print(f"Total pages: {pages}")
        
        ref_page = None
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if "References" in text:
                print(f"Found 'References' on page {i + 1}")
                ref_page = i + 1
                break
                
        if ref_page is None:
            print("ERROR: 'References' section not found.")
        elif ref_page != 9:
            print(f"WARNING: References start on page {ref_page}, expected page 9.")
        else:
            print("SUCCESS: References start exactly on page 9.")
            
        # Let's also check for any occurrence of "Title Suppressed"
        suppressed_found = False
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if "Title Suppressed" in text or "excessive size" in text:
                print(f"WARNING: Title suppressed message found on page {i + 1}!")
                suppressed_found = True
        if not suppressed_found:
            print("SUCCESS: No title suppression messages found.")
            
    except Exception as e:
        print(f"Error checking PDF: {e}")

if __name__ == "__main__":
    main()
