import os
import sys
import subprocess

def get_git_info():
    """Get GitHub repository owner/name and current branch."""
    try:
        remote_url = subprocess.check_output(["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL).decode().strip()
        # Handle both SSH and HTTPS formats
        if remote_url.startswith("git@github.com:"):
            repo_path = remote_url.split("git@github.com:")[1].replace(".git", "")
        elif remote_url.startswith("https://github.com/"):
            repo_path = remote_url.split("https://github.com/")[1].replace(".git", "")
        else:
            repo_path = "USER/REPO" # Fallback

        branch = subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL).decode().strip()
        if not branch:
            branch = "main"

        # Get repo root for relative path calculations
        repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL).decode().strip()
        
        return repo_path, branch, repo_root
    except Exception:
        return "USER/REPO", "main", os.getcwd()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_svg.py <output_directory>")
        sys.exit(1)

    target_dir = os.path.abspath(sys.argv[1])
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    repo_path, branch, repo_root = get_git_info()
    
    # Calculate relative path from repo root to target_dir
    try:
        rel_path = os.path.relpath(target_dir, repo_root)
    except ValueError:
        rel_path = target_dir

    md_path = os.path.join(target_dir, "RESULTS.md")

    # Detect the number of iterations (n)
    n = 0
    while os.path.exists(os.path.join(target_dir, f"{n+1}.svg")):
        n += 1

    md_content = [
        "# SVG Generation and Improvement Process",
        ""
    ]

    def get_raw_url(filename):
        return f"https://raw.githubusercontent.com/{repo_path}/{branch}/{rel_path}/{filename}?sanitize=true"

    # Round 0 (Initial Generation)
    if os.path.exists(os.path.join(target_dir, "0.svg")):
        md_content.append("## Initial Generation (Round 0)")
        md_content.append("")
        md_content.append(f"**Base Image (0.svg):**")
        md_content.append(f"![0.svg]({get_raw_url('0.svg')})")
        md_content.append("")

    # Iterative improvements
    for k in range(1, n + 1):
        md_content.append(f"## Iteration {k}")
        md_content.append("")

        chosen_svg_path = os.path.join(target_dir, f"{k}.svg")
        chosen_content = ""
        if os.path.exists(chosen_svg_path):
            with open(chosen_svg_path, 'r', encoding='utf-8') as f:
                chosen_content = f.read().strip()

        m = 1
        md_content.append("### Candidates")
        md_content.append("")
        
        while os.path.exists(os.path.join(target_dir, f"{k-1}-{m}.svg")):
            cand_path = os.path.join(target_dir, f"{k-1}-{m}.svg")
            filename = f"{k-1}-{m}.svg"
            with open(cand_path, 'r', encoding='utf-8') as f:
                cand_content = f.read().strip()

            is_chosen = (cand_content == chosen_content) and (cand_content != "")
            
            status = " ✅ **(WINNER)**" if is_chosen else ""
            md_content.append(f"#### {filename}{status}")
            md_content.append(f"![{filename}]({get_raw_url(filename)})")
            md_content.append("")
            
            m += 1
            
        # Fallback if somehow the content doesn't perfectly match any candidate
        if not any(open(os.path.join(target_dir, f"{k-1}-{x}.svg")).read().strip() == chosen_content 
                   for x in range(1, m) if os.path.exists(os.path.join(target_dir, f"{k-1}-{x}.svg"))):
            if os.path.exists(chosen_svg_path):
                md_content.append(f"#### {k}.svg ✅ **(WINNER)**")
                md_content.append(f"![{k}.svg]({get_raw_url(f'{k}.svg')})")
                md_content.append("")

    with open(md_path, "w", encoding='utf-8') as f:
        f.write("\n".join(md_content))

    print(f"Markdown visualization successfully generated at: {md_path}")

if __name__ == "__main__":
    main()
