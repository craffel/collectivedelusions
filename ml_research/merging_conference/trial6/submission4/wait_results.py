import os
import time

def main():
    start = time.time()
    timeout = 180
    found = False
    
    while time.time() - start < timeout:
        if os.path.exists("calib_ablation_results.json"):
            found = True
            break
        time.sleep(5)
        
    if found:
        print("SUCCESS: Found results file!")
        with open("calib_ablation_results.json") as f:
            print(f.read())
    else:
        print("TIMEOUT: Results file not found.")
        # Let's read the latest out/err files to see if there was an error
        out_files = [f for f in os.listdir(".") if f.startswith("calib-ablation_") and f.endswith(".out")]
        err_files = [f for f in os.listdir(".") if f.startswith("calib-ablation_") and f.endswith(".err")]
        if out_files:
            print("=== LATEST OUT LOG ===")
            with open(out_files[0]) as f:
                print(f.read())
        if err_files:
            print("=== LATEST ERR LOG ===")
            with open(err_files[0]) as f:
                print(f.read())

if __name__ == "__main__":
    main()
