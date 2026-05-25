import pypdf

def main():
    try:
        reader = pypdf.PdfReader("submission.pdf")
        text = reader.pages[8].extract_text() # Page 9 (index 8)
        print("--- CONTENT OF PAGE 9 ---")
        print(text[:2000])
        print("-------------------------")
    except Exception as e:
        print(f"Error reading page 9: {e}")

if __name__ == "__main__":
    main()
