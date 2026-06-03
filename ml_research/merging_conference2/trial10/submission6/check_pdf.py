import pypdf
try:
    reader = pypdf.PdfReader("submission.pdf")
    print("Page count:", len(reader.pages))
except Exception as e:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader("submission.pdf")
        print("Page count:", len(reader.pages))
    except Exception as e2:
        print("Error reading PDF:", e, e2)
