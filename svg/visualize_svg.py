import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_svg.py <output_directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    html_path = os.path.join(target_dir, "index.html")

    # Detect the number of iterations (n)
    n = 0
    while os.path.exists(os.path.join(target_dir, f"{n+1}.svg")):
        n += 1

    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <title>SVG Evolution</title>",
        "    <style>",
        "        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; padding: 20px; }",
        "        h1 { text-align: center; color: #2c3e50; }",
        "        .round { background: #fff; margin-bottom: 40px; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }",
        "        .round h2 { margin-top: 0; color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 10px; }",
        "        .grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }",
        "        .card { border: 3px solid transparent; padding: 15px; border-radius: 10px; background: #fafafa; text-align: center; width: 300px; display: flex; flex-direction: column; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }",
        "        .card.chosen { border-color: #2ecc71; background: #eafaf1; box-shadow: 0 0 15px rgba(46, 204, 113, 0.4); transform: scale(1.02); transition: transform 0.2s; }",
        "        .card img { max-width: 100%; max-height: 250px; border: 1px dashed #ccc; background: #fff; margin-top: 10px; padding: 5px; border-radius: 5px; }",
        "        .badge { display: inline-block; background: #2ecc71; color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }",
        "        .filename { font-family: monospace; font-size: 14px; color: #555; background: #eee; padding: 3px 8px; border-radius: 4px; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>SVG Generation and Improvement Process</h1>"
    ]

    # Round 0 (Initial Generation)
    if os.path.exists(os.path.join(target_dir, "0.svg")):
        html_content.append("    <div class='round'>")
        html_content.append("        <h2>Initial Generation (Round 0)</h2>")
        html_content.append("        <div class='grid'>")
        html_content.append("            <div class='card chosen'>")
        html_content.append("                <div class='badge'>Base Image</div>")
        html_content.append("                <div class='filename'>0.svg</div>")
        html_content.append("                <img src='0.svg' alt='0.svg'>")
        html_content.append("            </div>")
        html_content.append("        </div>")
        html_content.append("    </div>")

    # Iterative improvements
    for k in range(1, n + 1):
        html_content.append(f"    <div class='round'>")
        html_content.append(f"        <h2>Iteration {k}</h2>")
        html_content.append("        <div class='grid'>")

        chosen_svg_path = os.path.join(target_dir, f"{k}.svg")
        chosen_content = ""
        if os.path.exists(chosen_svg_path):
            with open(chosen_svg_path, 'r', encoding='utf-8') as f:
                chosen_content = f.read().strip()

        m = 1
        match_found = False
        while os.path.exists(os.path.join(target_dir, f"{k-1}-{m}.svg")):
            cand_path = os.path.join(target_dir, f"{k-1}-{m}.svg")
            with open(cand_path, 'r', encoding='utf-8') as f:
                cand_content = f.read().strip()

            # Check if this candidate is the chosen one
            is_chosen = (cand_content == chosen_content) and (cand_content != "")
            if is_chosen:
                match_found = True

            css_class = "card chosen" if is_chosen else "card"
            
            html_content.append(f"            <div class='{css_class}'>")
            if is_chosen:
                html_content.append("                <div class='badge'>Winner</div>")
            html_content.append(f"                <div class='filename'>{k-1}-{m}.svg</div>")
            html_content.append(f"                <img src='{k-1}-{m}.svg' alt='{k-1}-{m}.svg'>")
            html_content.append("            </div>")
            
            m += 1
            
        # Fallback if somehow the content doesn't perfectly match any candidate
        # but the k.svg file exists
        if not match_found and os.path.exists(chosen_svg_path):
            html_content.append(f"            <div class='card chosen'>")
            html_content.append("                <div class='badge'>Winner (Standalone)</div>")
            html_content.append(f"                <div class='filename'>{k}.svg</div>")
            html_content.append(f"                <img src='{k}.svg' alt='{k}.svg'>")
            html_content.append("            </div>")

        html_content.append("        </div>")
        html_content.append("    </div>")

    html_content.append("</body>")
    html_content.append("</html>")

    with open(html_path, "w", encoding='utf-8') as f:
        f.write("\n".join(html_content))

    print(f"Visualization successfully generated at: {html_path}")

if __name__ == "__main__":
    main()