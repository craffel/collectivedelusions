import glob
import os
import re

files = sorted(glob.glob("submission*/reviewer*/review.md"))

output_file = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial9/metareview/full_reviews_summary.txt"

with open(output_file, 'w', encoding='utf-8') as out:
    for file_path in files:
        parts = file_path.split('/')
        sub_num = parts[0].replace('submission', '')
        rev_num = parts[1].replace('reviewer', '')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        title_m = re.search(r'^#\s*(.*)', content)
        title = title_m.group(1).strip() if title_m else "Unknown Title"
        
        # Extract Recommendation/Rating/Score
        rating_lines = []
        for line in content.split('\n'):
            if any(term in line.lower() for term in ["rating:", "score:", "recommendation:", "overall rating:", "recommendation score:"]):
                # Clean up bolding and headers
                cleaned = line.replace('**', '').replace('__', '').strip()
                if cleaned and len(cleaned) < 200:
                    rating_lines.append(cleaned)
                    
        ratings_str = " | ".join(rating_lines) if rating_lines else "None explicitly found"
        
        # Try to find specific sections
        # 1. Summary of the paper
        summary = ""
        sum_m = re.search(r'##\s*(?:\d\.\s*)?Summary of the Paper\s*\n*(.*?)\n*(?:##|#|$)', content, re.DOTALL | re.IGNORECASE)
        if sum_m:
            summary = sum_m.group(1).strip()
            
        # 2. Strengths
        strengths = ""
        str_m = re.search(r'###\s*(?:Major\s*)?Strengths\s*:?\s*\n*(.*?)\n*(?:###|##|$)', content, re.DOTALL | re.IGNORECASE)
        if not str_m:
            str_m = re.search(r'Strengths\s*\n*(.*?)\n*(?:Weaknesses|##|$)', content, re.DOTALL | re.IGNORECASE)
        if str_m:
            strengths = str_m.group(1).strip()
            
        # 3. Weaknesses
        weaknesses = ""
        weak_m = re.search(r'###\s*(?:Major\s*)?Weaknesses\s*:?\s*\n*(.*?)\n*(?:###|##|$)', content, re.DOTALL | re.IGNORECASE)
        if not weak_m:
            weak_m = re.search(r'Weaknesses\s*\n*(.*?)\n*(?:##|$)', content, re.DOTALL | re.IGNORECASE)
        if weak_m:
            weaknesses = weak_m.group(1).strip()
            
        # 4. Overall Recommendation / Justification
        rec_text = ""
        rec_m = re.search(r'##\s*(?:\d\.\s*)?(?:Overall Recommendation|Recommendation|Ratings|Overall Rating)\s*\n*(.*?)\n*(?:##|#|$)', content, re.DOTALL | re.IGNORECASE)
        if rec_m:
            rec_text = rec_m.group(1).strip()

        out.write(f"============================================================\n")
        out.write(f"SUBMISSION: {sub_num} | REVIEWER: {rev_num}\n")
        out.write(f"TITLE: {title}\n")
        out.write(f"RATINGS DETECTED: {ratings_str}\n")
        out.write(f"------------------------------------------------------------\n")
        out.write(f"SUMMARY OF PAPER:\n{summary[:400]}...\n\n")
        out.write(f"STRENGTHS:\n{strengths[:1500]}\n\n")
        out.write(f"WEAKNESSES:\n{weaknesses[:1500]}\n\n")
        out.write(f"OVERALL RECOMMENDATION / JUSTIFICATION:\n{rec_text[:1200]}\n")
        out.write(f"============================================================\n\n\n")

print(f"Extraction complete! Structured summaries written to {output_file}")
