# Semantic Scholar API Skill

Expert guidance for interacting with the Semantic Scholar Academic Graph API to search for relevant research papers and retrieve their metadata, specifically focusing on open-access PDFs.

## Context
Semantic Scholar provides a REST API for accessing its database of over 200 million papers. This skill focuses on the `/graph/v1/paper/search` and `/graph/v1/paper/{paper_id}` endpoints.

## Instructions

### 1. Searching for Papers
Use the search endpoint to find papers relevant to a specific query.

- **Endpoint:** `GET https://api.semanticscholar.org/graph/v1/paper/search`
- **Required Parameters:**
    - `query`: The search string (e.g., "transformer architectures").
- **Recommended Parameters:**
    - `fields`: Comma-separated list of fields to return. Always include `paperId`, `title`, and `openAccessPdf`.
    - `limit`: Number of results (default 10, max 100).
    - `openAccessPdf`: Include this parameter (no value needed) to filter for papers that have a publicly available PDF.
- **Example Request:**
  `https://api.semanticscholar.org/graph/v1/paper/search?query=large+language+models&fields=title,authors,year,openAccessPdf&openAccessPdf&limit=5`

### 2. Retrieving Paper Details
Use the paper detail endpoint when you have a specific `paperId`.

- **Endpoint:** `GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}`
- **Parameters:**
    - `fields`: Explicitly request `openAccessPdf` to get the URL.
- **Example Request:**
  `https://api.semanticscholar.org/graph/v1/paper/7c124310d32c9497e8e50b91e9f1a044d08e5c1e?fields=title,abstract,openAccessPdf`

### 3. Handling the `openAccessPdf` Field
The response will contain an `openAccessPdf` object:
```json
"openAccessPdf": {
  "url": "https://example.com/paper.pdf",
  "status": "GOLD",
  "license": "CC-BY"
}
```
If `openAccessPdf` is `null`, no public PDF is available via Semantic Scholar.

### 4. Rate Limiting & Authentication
- **Free Tier:** Limited rate. Use an API key if available to increase limits.
- **Header:** If using a key, include `x-api-key: <YOUR_API_KEY>`.
- **Strategy:** If you encounter a 429 error, implement a brief backoff before retrying.

## Workflow Patterns

### Pattern: Search and Download
1. **Search:** Query the search endpoint with `openAccessPdf` filter and `fields=paperId,title,openAccessPdf`.
2. **Select:** Identify the most relevant paper(s) from the results.
3. **Fetch PDF:** Extract the `url` from `openAccessPdf`.
4. **Download:** Use `curl` or a similar tool to save the PDF.

### Pattern: Detail Enrichment
1. **Get ID:** Start with a `paperId` (e.g., from a bibliography).
2. **Query:** Call the `/paper/{paper_id}` endpoint requesting `abstract`, `citationCount`, and `openAccessPdf`.
3. **Analyze:** Use the abstract and metadata to determine relevance before downloading.
