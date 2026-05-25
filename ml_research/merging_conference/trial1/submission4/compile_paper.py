import json
import subprocess

def compile_paper():
    # Read results
    with open('experiment_results.json', 'r') as f:
        results = json.load(f)
        
    ta = results['Task_Arithmetic']
    saim = results['SAIM']
    ortho = results['OrthoMerge']
    smm = results['SMM']
    
    # Read submission.tex
    with open('submission.tex', 'r') as f:
        tex = f.read()
        
    # Replace placeholders with backslashed underscores
    tex = tex.replace('\\_TA\\_ACC1\\_', f"{ta['task1']:.2f}")
    tex = tex.replace('\\_TA\\_ACC2\\_', f"{ta['task2']:.2f}")
    tex = tex.replace('\\_TA\\_AVG\\_', f"{ta['avg']:.2f}")
    
    tex = tex.replace('\\_SAIM\\_ACC1\\_', f"{saim['task1']:.2f}")
    tex = tex.replace('\\_SAIM\\_ACC2\\_', f"{saim['task2']:.2f}")
    tex = tex.replace('\\_SAIM\\_AVG\\_', f"{saim['avg']:.2f}")
    
    tex = tex.replace('\\_ORTHO\\_ACC1\\_', f"{ortho['task1']:.2f}")
    tex = tex.replace('\\_ORTHO\\_ACC2\\_', f"{ortho['task2']:.2f}")
    tex = tex.replace('\\_ORTHO\\_AVG\\_', f"{ortho['avg']:.2f}")
    
    tex = tex.replace('\\_SMM\\_ACC1\\_', f"{smm['task1']:.2f}")
    tex = tex.replace('\\_SMM\\_ACC2\\_', f"{smm['task2']:.2f}")
    tex = tex.replace('\\_SMM\\_AVG\\_', f"{smm['avg']:.2f}")
    
    # Write updated tex
    with open('submission.tex', 'w') as f:
        f.write(tex)
        
    print("Placeholders replaced in submission.tex successfully.")
    
    # Compile with tectonic
    print("Compiling submission.tex with tectonic...")
    res = subprocess.run(['/fsx/craffel/miniconda3/bin/tectonic', 'submission.tex'], capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    if res.returncode == 0:
        print("Paper compiled successfully! submission.pdf generated.")
    else:
        print("Paper compilation failed with code:", res.returncode)

if __name__ == '__main__':
    compile_paper()
