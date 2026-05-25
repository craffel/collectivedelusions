import pypdf

def main():
    try:
        reader = pypdf.PdfReader('submission.pdf')
        print(f"Total pages: {len(reader.pages)}")
        
        ref_pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if "References" in text:
                ref_pages.append(idx + 1)
                
        print(f"References found on page(s): {ref_pages}")
        if ref_pages:
            main_body_pages = ref_pages[0] - 1
            print(f"Main body page count: {main_body_pages}")
            if main_body_pages == 8:
                print("SUCCESS: Main body is EXACTLY 8 pages!")
            elif main_body_pages < 8:
                print(f"WARNING: Main body is too short ({main_body_pages} pages). Needs expansion.")
            else:
                print(f"WARNING: Main body is too long ({main_body_pages} pages). Needs compression.")
        else:
            print("WARNING: 'References' section not found in PDF text.")
    except Exception as e:
        print(f"Error reading submission.pdf: {e}")

if __name__ == "__main__":
    main()
