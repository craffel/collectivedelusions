import os
import sys
import json
import random
import requests
from pypdf import PdfReader
import google.generativeai as genai

# Configure API Keys
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

genai.configure(api_key=GEMINI_API_KEY)

def search_semantic_scholar(query, limit=5):
    print(f"Searching Semantic Scholar for: '{query}'...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    params = {
        "query": query,
        "fields": "paperId,title,abstract,year,openAccessPdf,citationCount",
        "openAccessPdf": "",
        "limit": limit
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Search failed with status code {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
        return []

def download_pdf(url, dest_path):
    print(f"Downloading PDF from {url} to {dest_path}...")
    try:
        response = requests.get(url, timeout=30, stream=True)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
            return True
        else:
            print(f"Failed to download PDF, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Read up to first 8 pages to avoid excessive context length
        pages_to_read = min(8, len(reader.pages))
        for page_num in range(pages_to_read):
            text += f"--- Page {page_num + 1} ---\n"
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def load_local_papers():
    papers_dir = "papers"
    papers_data = {}
    for filename in os.listdir(papers_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(papers_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                papers_data[filename] = f.read()
    return papers_data

def main():
    print("=== Starting Phase 1: Foundation ===")
    
    # 1. Search Semantic Scholar for related papers
    query = "test-time model merging"
    search_results = search_semantic_scholar(query, limit=5)
    
    downloaded_papers_context = ""
    # Download top 2 open-access papers if available and extract text
    download_count = 0
    for idx, paper in enumerate(search_results):
        title = paper.get("title")
        pdf_info = paper.get("openAccessPdf")
        abstract = paper.get("abstract")
        year = paper.get("year")
        citation_count = paper.get("citationCount", 0)
        
        print(f"\nFound Paper {idx+1}: {title} ({year}) - Citations: {citation_count}")
        if pdf_info and pdf_info.get("url") and download_count < 2:
            pdf_url = pdf_info.get("url")
            pdf_name = f"downloaded_paper_{download_count + 1}.pdf"
            pdf_path = os.path.join("papers", pdf_name)
            if download_pdf(pdf_url, pdf_path):
                txt = extract_text_from_pdf(pdf_path)
                if txt:
                    downloaded_papers_context += f"\n=========================\n"
                    downloaded_papers_context += f"DOWNLOADED RELATED PAPER {download_count + 1}:\n"
                    downloaded_papers_context += f"Title: {title}\n"
                    downloaded_papers_context += f"Year: {year}\n"
                    downloaded_papers_context += f"Abstract: {abstract}\n"
                    downloaded_papers_context += f"Extracted Text Snippet:\n{txt[:12000]}\n" # limit snippet size
                    download_count += 1
        else:
            if abstract:
                downloaded_papers_context += f"\n=========================\n"
                downloaded_papers_context += f"METADATA ONLY RELATED PAPER:\n"
                downloaded_papers_context += f"Title: {title}\n"
                downloaded_papers_context += f"Year: {year}\n"
                downloaded_papers_context += f"Abstract: {abstract}\n"

    # 2. Load the 3 core submission papers
    local_papers = load_local_papers()
    local_papers_context = ""
    for name, content in local_papers.items():
        # Keep first 15000 chars of each local paper to fit easily in context
        local_papers_context += f"\n=========================\n"
        local_papers_context += f"SUBMISSION PAPER: {name}\n"
        local_papers_context += f"Content:\n{content[:15000]}\n"

    # 3. Call Gemini to generate 10 novel ideas
    print("\nCalling Gemini to synthesize papers and generate 10 novel ideas...")
    
    prompt = f"""
You are an expert machine learning researcher specialized in Model Merging, Test-Time Adaptation (TTA), and Test-Time Model Merging (TTMM).
We are working on developing a highly novel, high-impact research paper in this space to submit to a top conference (ICML).

Here are the 3 submission papers currently under review or development in our workspace:
{local_papers_context}

Here are some additional related papers we found in the literature search:
{downloaded_papers_context}

Based on the above, please:
1. Synthesize the general themes, core contributions, limitations, and potential extensions of the 3 submission papers (CPA-Merge, LFWA, and PC-Merge).
2. Formulate 10 novel, high-impact research ideas on Test-Time Model Merging. For each idea, provide:
   - Title
   - Description and core mechanics (how it works)
   - Hypothesis and Expected results
   - Significance and Novelty (why it is a major contribution)
   - Feasibility and potential risks
3. Format the 10 ideas clearly as a JSON list (or structured block) within your markdown response.
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    analysis_text = response.text
    print("Gemini analysis and idea generation completed.")
    
    # Save the analysis to progress.md
    with open("progress.md", "a", encoding="utf-8") as f:
        f.write("\n## Phase 1: Foundation & Idea Generation\n")
        f.write("\n### Literature Synthesis & 10 Novel Ideas\n")
        f.write(analysis_text)
        f.write("\n")

    # 4. Use Gemini to select one of the ideas using a PRNG and refine it
    # We will parse the output to select one, or have Gemini select one using a random seed.
    # To be strictly procedural and use a PRNG in Python:
    # We can ask Gemini to extract the 10 ideas, and then we will choose one in Python with a seed, 
    # and then prompt Gemini again to refine that specific selected idea.
    
    # Let's write a prompt to extract the titles of the 10 ideas to make selection clean
    extract_prompt = f"""
Based on your previous response, please list the titles of the 10 ideas you generated.
Return them as a simple numbered list, one per line, like this:
1. Title 1
2. Title 2
...
10. Title 10

Do not include any other text.
Here is your previous response:
{analysis_text}
"""
    titles_response = model.generate_content(extract_prompt).text
    print("\nGenerated Idea Titles:")
    print(titles_response)
    
    # Parse titles
    titles = []
    for line in titles_response.strip().split("\n"):
        line = line.strip()
        if line and any(line.startswith(str(i) + ".") for i in range(1, 11)):
            parts = line.split(".", 1)
            if len(parts) > 1:
                titles.append(parts[1].strip())
    
    if len(titles) < 10:
        # Fallback if parsing failed
        print("Warning: could not parse all 10 titles. Using dummy indexing.")
        titles = [f"Idea {i+1}" for i in range(10)]
    
    # Select using PRNG
    seed_value = 42 # Chosen seed
    random.seed(seed_value)
    selected_idx = random.randint(0, 9)
    selected_title = titles[selected_idx]
    print(f"\nPRNG selected index: {selected_idx} (Seed: {seed_value})")
    print(f"Selected Idea: {selected_title}")

    # 5. Refine the chosen idea
    print(f"\nCalling Gemini to refine the selected idea: '{selected_title}'...")
    refine_prompt = f"""
You are an expert machine learning researcher.
We have selected the following research idea to execute using a pseudo-random number generator:
Index: {selected_idx + 1}
Title: {selected_title}

From our full list of ideas:
{analysis_text}

Please significantly improve upon the novelty, feasibility, and importance of this proposed research idea by reconsidering prior work (especially CPA-Merge, LFWA, and PC-Merge, and the wider literature). 
Formulate a rigorous project hypothesis and rationale.
Provide:
1. Improved Title
2. Core Hypothesis
3. Mathematical/Algorithmic Formulation (detailed equations, loss functions, optimization steps)
4. Concrete Experimental Design (Datasets, Baselines, Architecture, Evaluation Metrics)
5. Refined Rationale and Why it will be Accepted at ICML

Format your response as markdown.
"""
    refine_response = model.generate_content(refine_prompt)
    refined_text = refine_response.text
    print("Refinement completed.")

    # Save to progress.md
    with open("progress.md", "a", encoding="utf-8") as f:
        f.write("\n### PRNG Selection & Selection Rationale\n")
        f.write(f"- **PRNG Seed:** {seed_value}\n")
        f.write(f"- **Selected Index:** {selected_idx + 1} (out of 10)\n")
        f.write(f"- **Selected Original Title:** {selected_title}\n\n")
        f.write("### Refined Project Hypothesis and Rationale\n")
        f.write(refined_text)
        f.write("\n\n=== Phase 1 Complete ===\n")

    print("\nPhase 1 completed and written to progress.md successfully.")

if __name__ == "__main__":
    main()
