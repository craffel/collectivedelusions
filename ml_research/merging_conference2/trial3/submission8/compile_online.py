import os
import requests

def main():
    url = "https://texlive.net/cgi-bin/latexcgi"
    
    # We must use lists of tuples because requests supports duplicate keys
    # for multipart form-data array arrays.
    data = [
        ('engine', 'pdflatex'),
        ('return', 'pdf')
    ]
    files = []
    
    # 1. Main Document
    print("Adding main document...")
    data.append(('filename[]', 'document.tex'))
    files.append(('filecontents[]', ('document.tex', open('submission.tex', 'rb'), 'text/plain')))
    
    # 2. Styles and Bib
    extra_files = [
        'icml2026.sty', 'icml2026.bst', 'fancyhdr.sty',
        'algorithm.sty', 'algorithmic.sty', 'example_paper.bib'
    ]
    for f in extra_files:
        if os.path.exists(f):
            print(f"Adding extra file {f}...")
            data.append(('filename[]', f))
            files.append(('filecontents[]', (f, open(f, 'rb'), 'text/plain')))
            
    # 3. Figures (Flat filenames)
    figures = [
        ('main_benchmark.pdf', 'main_benchmark.pdf'),
        ('sample_efficiency.pdf', 'sample_efficiency.pdf'),
        ('distributional_robustness.pdf', 'distributional_robustness.pdf'),
        ('rtaac_shrinkage.pdf', 'rtaac_shrinkage.pdf'),
        ('ablation_depth.pdf', 'ablation_depth.pdf')
    ]
    for flat_name, real_path in figures:
        if os.path.exists(real_path):
            print(f"Adding flat figure {flat_name} from {real_path}...")
            data.append(('filename[]', flat_name))
            files.append(('filecontents[]', (flat_name, open(real_path, 'rb'), 'application/pdf')))
            
    print("\nSending POST request to TeXLive.net online compiler...")
    response = requests.post(url, data=data, files=files, headers={'User-Agent': 'curl/7.68.0'})
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 301 or response.headers.get('Location'):
        loc = response.headers.get('Location')
        print(f"\nRedirected to log file: https://texlive.net{loc}")
        log_res = requests.get(f"https://texlive.net{loc}")
        print("\n=== LATEX COMPILATION LOG PREVIEW ===")
        print(log_res.text[-2000:])
    elif response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' in content_type:
            with open('submission.pdf', 'wb') as f:
                f.write(response.content)
            print("\nSuccessfully compiled and downloaded submission.pdf!")
        else:
            print("\nReceived response but content is not PDF (showing end of log for errors):")
            print(response.text[-3000:])
    else:
        print(f"\nFailed with status: {response.status_code}")
        print(response.text[:2000])

if __name__ == '__main__':
    main()
