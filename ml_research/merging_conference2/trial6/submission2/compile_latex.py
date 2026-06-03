import requests
import os
import sys

def main():
    api_url = "https://texlive.net/cgi-bin/latexcgi"
    
    # Files to upload
    upload_files = [
        ("submission.tex", "document.tex"),
        ("submission.bib", "submission.bib"),
        ("icml2026.bst", "icml2026.bst"),
        ("icml2026.sty", "icml2026.sty"),
        ("algorithm.sty", "algorithm.sty"),
        ("algorithmic.sty", "algorithmic.sty"),
        ("fancyhdr.sty", "fancyhdr.sty")
    ]
    
    print("Preparing files for LaTeX compilation via TeXLive.net...")
    
    # Prepare multipart form data
    fields = [
        ("engine", "pdflatex"),
        ("bibcmd", "bibtex"),
        ("return", "pdf")
    ]
    
    # Keep file handles open during request
    file_handles = []
    try:
        files_data = []
        for local_name, remote_name in upload_files:
            if not os.path.exists(local_name):
                print(f"Error: Required file '{local_name}' not found!")
                sys.exit(1)
            
            # Read file contents and normalize line endings if tex
            if local_name.endswith(".tex") or local_name.endswith(".bib"):
                with open(local_name, "rb") as f:
                    content = f.read()
                # Normalize line endings to Windows-style CRLF for TeXLive.net compatibility
                content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                files_data.append(("filecontents[]", (remote_name, content)))
            else:
                f = open(local_name, "rb")
                file_handles.append(f)
                files_data.append(("filecontents[]", (remote_name, f)))
            
            fields.append(("filename[]", remote_name))
            
        print("Sending compilation request to TeXLive.net...")
        response = requests.post(api_url, data=fields, files=files_data, allow_redirects=False)
        
        print(f"Server responded with status code: {response.status_code}")
        
        # Capture redirect link
        if response.status_code in (301, 302, 303, 307, 308) and "Location" in response.headers:
            redirect_url = response.headers["Location"]
            if redirect_url.startswith("/"):
                redirect_url = "https://texlive.net" + redirect_url
            print(f"Redirect URL: {redirect_url}")
            
            # Download compiled PDF or error log
            print("Downloading compilation output...")
            pdf_response = requests.get(redirect_url)
            
            # Check content type or headers to determine if it's a PDF or error log
            content_type = pdf_response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                with open("submission.pdf", "wb") as f:
                    f.write(pdf_response.content)
                print(f"Success! Final compiled PDF saved as 'submission.pdf' ({len(pdf_response.content) / 1024:.1f} KB)")
            else:
                print("Compilation might have failed. Let's see the returned response:")
                try:
                    text_content = pdf_response.content.decode("utf-8", errors="ignore")
                    print(text_content[:2000])  # Show first 2000 characters of log
                    with open("compilation_errors.log", "w") as f:
                        f.write(text_content)
                    print("\nFull error log saved as 'compilation_errors.log'")
                except Exception as e:
                    print(f"Could not decode error response: {e}")
                sys.exit(1)
        else:
            # If allow_redirects=False wasn't respected or directly returned content
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                with open("submission.pdf", "wb") as f:
                    f.write(response.content)
                print(f"Success! Compiled PDF saved directly as 'submission.pdf' ({len(response.content) / 1024:.1f} KB)")
            else:
                print("Received non-PDF response direct from server. Log:")
                print(response.text[:2000])
                sys.exit(1)
                
    finally:
        for f in file_handles:
            f.close()

if __name__ == "__main__":
    main()
