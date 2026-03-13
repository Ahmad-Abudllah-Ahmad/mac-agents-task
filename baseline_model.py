import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from mlx_vlm import load, generate

def load_data():
    print("Loading datasets...")
    # Using the test split, but we'll cap it at 50 to evaluate in time
    try:
        ds = load_dataset(
            "CADCODER/GenCAD-Code", 
            split="test", 
            cache_dir="/Volumes/BIG-DATA/HUGGINGFACE_CACHE"
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # fallback without custom cache dir if it fails
        ds = load_dataset(
            "CADCODER/GenCAD-Code", 
            split="test"
        )
        
    # Set seed for reproducibility and select 50 samples
    ds = ds.shuffle(seed=42)
    return ds.select(range(50))

def run_baseline():
    print("Loading MLX VLM model (Baseline)...")
    # Using 4-bit Qwen2-VL to be GPU-poor friendly (runs on Apple Silicon unified memory)
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    
    # Load model and processor
    model, processor = load(model_path)
    
    ds = load_data()
    
    results = {}
    ground_truth = {}
    
    print("Generating baseline predictions...")
    for idx, item in enumerate(tqdm(ds)):
        image = item["image"]
        deepcad_id = item["deepcad_id"]
        gt_cadquery = item["cadquery"]
        
        # Store GT for evaluation
        ground_truth[deepcad_id] = gt_cadquery
        
        # Zero-shot baseline prompt
        prompt = "Write CadQuery Python code to generate the 3D shape shown in this image. Only output the Python code, with the final shape assigned to a variable named 'result'."
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, 
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Generate code 
        # MLX processor handles image automatically
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        output = generate(model, processor, formatted_prompt, [image], max_tokens=1024, temperature=0.1)
        
        # Simple extraction logic: extract python block if any
        code = output.strip()
        if "```python" in code:
            parts = code.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
        elif "```" in code:
            parts = code.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                
        results[deepcad_id] = code
        
    # Save the generated code
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open("results/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
        
    print("Baseline generation complete. Saved to results/")

if __name__ == "__main__":
    run_baseline()
