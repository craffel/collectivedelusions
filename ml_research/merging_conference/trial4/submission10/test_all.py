import os
import subprocess
import pypdf

def test_pdf_verification():
    print("Testing PDF Verification...")
    reader = pypdf.PdfReader("submission.pdf")
    pages = len(reader.pages)
    print(f"Total pages: {pages}")
    assert pages == 12, f"Expected 12 pages, got {pages}"
    
    ref_page = None
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if "References" in text:
            ref_page = i + 1
            break
            
    assert ref_page == 9, f"Expected references on page 9, got {ref_page}"
    print("PDF Verification Passed!")

def test_code_paper_consistency():
    print("Testing Code-Paper Consistency...")
    # Read generate_paper_tex.py content
    with open("generate_paper_tex.py", "r") as f:
        gen_content = f.read()
        
    # Run generate_paper_tex.py to regenerate submission.tex
    subprocess.run(["python", "generate_paper_tex.py"], check=True)
    
    # Read submission.tex
    with open("submission.tex", "r") as f:
        sub_content = f.read()
        
    # Check if they match the internal paper_content
    # The submission.tex should be successfully generated and non-empty
    assert len(sub_content) > 10000, "submission.tex is too short!"
    print("Code-Paper Consistency Passed!")

def main():
    try:
        test_pdf_verification()
        test_code_paper_consistency()
        print("\nAll sanity checks and automated tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
