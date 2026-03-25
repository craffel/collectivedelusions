import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).parent.resolve()

def generate_readme(images_dir_str):
    IMAGES_DIR = pathlib.Path(images_dir_str).resolve()
    README_PATH = IMAGES_DIR / "README.md"
    
    with open(README_PATH, "w") as f:
        f.write("# Collective Delusions: Nano Banana Contest\n\n")
        f.write("An evolutionary image generation contest where Gemini models generate submissions and judge the winners.\n\n")
        
        # Add Prompts Section
        f.write("## Contest Configuration\n\n")
        
        gen_prompt_path = IMAGES_DIR / "generation_prompt.md"
        judge_prompt_path = IMAGES_DIR / "judge_prompt.md"
        desc_path = IMAGES_DIR / "contest_description.md"
        
        if gen_prompt_path.exists():
            f.write("### Generation Prompt\n")
            f.write("```markdown\n")
            f.write(gen_prompt_path.read_text().strip())
            f.write("\n```\n\n")
            
        if judge_prompt_path.exists():
            f.write("### Judge Prompt\n")
            f.write("```markdown\n")
            f.write(judge_prompt_path.read_text().strip())
            f.write("\n```\n\n")
            
        if desc_path.exists():
            f.write("### Contest Description\n")
            f.write("```markdown\n")
            f.write(desc_path.read_text().strip())
            f.write("\n```\n\n")
            
        f.write("---\n\n")
        
        # Find all round folders
        sub_dirs = sorted([d for d in IMAGES_DIR.glob("submissions_round_*") if d.is_dir()], key=lambda x: int(x.name.split("_")[-1]))
        
        if not sub_dirs:
            f.write("No rounds completed yet.\n")
            return
            
        for sub_dir in sub_dirs:
            round_n = int(sub_dir.name.split("_")[-1])
            win_dir = IMAGES_DIR / f"winners_round_{round_n}"
            
            f.write(f"## Round {round_n}\n\n")
            
            winners = set()
            if win_dir.exists():
                winners = {w.name for w in win_dir.glob("*.jpg")}
            
            # Sort submissions by their number
            submissions = sorted(list(sub_dir.glob("*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
            
            if not submissions:
                f.write("*No images found for this round.*\n\n")
                continue
                
            # Create a markdown table for the grid (3 columns for better viewing)
            f.write("| Image 1 | Image 2 | Image 3 |\n")
            f.write("| :---: | :---: | :---: |\n")
            
            row = []
            for sub in submissions:
                try:
                    rel_path = sub.relative_to(IMAGES_DIR)
                except ValueError:
                    rel_path = sub # Fallback if not relative
                
                is_winner = sub.name in winners
                status = "🏆 **WINNER**" if is_winner else "Submission"
                
                # Format: Image followed by status
                cell = f"<img src='{rel_path}' width='250'><br>{status}<br>`{sub.name}`"
                row.append(cell)
                
                if len(row) == 3:
                    f.write(f"| {' | '.join(row)} |\n")
                    row = []
            
            # Write remaining cells in the last row
            if row:
                while len(row) < 3:
                    row.append("")
                f.write(f"| {' | '.join(row)} |\n")
            
            f.write("\n---\n\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_readme(sys.argv[1])
    else:
        print("Please provide the images directory as an argument.")
