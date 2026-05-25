import json

with open("search_results.json", "r") as f:
    data = json.load(f)

for query, papers in data.items():
    print(f"\n================= QUERY: {query} =================")
    for i, paper in enumerate(papers):
        title = paper.get("title", "No Title")
        year = paper.get("year", "No Year")
        pdf_info = paper.get("openAccessPdf")
        pdf_url = pdf_info.get("url") if pdf_info else "No PDF"
        print(f"{i+1}. {title} ({year})")
        print(f"   PDF: {pdf_url}")
        abstract = paper.get("abstract", "No Abstract")
        if abstract:
            print(f"   Abstract: {abstract[:300]}...")
        else:
            print("   No Abstract")
        print()
