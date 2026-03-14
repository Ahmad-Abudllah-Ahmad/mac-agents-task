import json
from datasets import load_dataset
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("Fixing ground truth...")
    ds = load_dataset("CADCODER/GenCAD-Code", split="test", streaming=True)
    ds_iter = iter(ds)
    gt_codes = {}
    for _ in range(50):
        try:
            item = next(ds_iter)
            gt_codes[item["deepcad_id"]] = item["cadquery"]
        except Exception as e:
            print("Error:", e)
            break
            
    with open("results/ground_truth.json", "w") as f:
        json.dump(gt_codes, f, indent=2)
    print("Done. Saved to results/ground_truth.json")

if __name__ == "__main__":
    main()
