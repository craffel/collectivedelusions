import pypdf
try:
    reader = pypdf.PdfReader("submission.pdf")
    print("Total pages:", len(reader.pages))
    page_9_text = reader.pages[8].extract_text()
    print("--- Page 9 Beginning ---")
    print(page_9_text[:500])
    print("------------------------")
except Exception as e:
    print("Error:", e)
