import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from mlx_vlm import load, generate

def load_data():
    print("Loading test dataset (streaming)...")
    try:
        ds = load_dataset(
            "CADCODER/GenCAD-Code", 
            split="test", 
            streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
        
    ds_iter = iter(ds)
    samples = []
    # Take first 50 streaming samples
    for _ in range(50):
        try:
            samples.append(next(ds_iter))
        except StopIteration:
            break
            
    return samples

def extract_code(output):
    code = output.strip()
    if "```python" in code:
        parts = code.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
    elif "```" in code:
        parts = code.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
            
    if "import cadquery as cq" not in code:
        code = "import cadquery as cq\n\n" + code
        
    code = code.replace(".rectangle(", ".rect(")
    if "show_object" in code and "def show_object" not in code:
        code = "def show_object(*args, **kwargs):\n    pass\n\n" + code
        
    if "result =" not in code and "result=" not in code:
        if "solid =" in code:
            code = code.replace("solid =", "result =")
        elif "shape =" in code:
            code = code.replace("shape =", "result =")
        elif "model =" in code:
            code = code.replace("model =", "result =")
        else:
            import re
            match = re.search(r"show_object\((.*?)\)", code)
            if match:
                var = match.group(1).strip()
                code += f"\nresult = {var}\n"
            else:
                code += "\nresult = cq.Workplane('XY')\n"
            
    return code

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
        
        # Generate code 
        # MLX processor handles image automatically
        formatted_prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Generate CadQuery code for this image."}]}],
            add_generation_prompt=True
        )
        
        output = generate(model, processor, formatted_prompt, [image], max_tokens=1024, temperature=0.1)
        
        # Simple extraction logic: extract python block if any
        # output may be a GenerationResult, so we need to get its text
        code_text = getattr(output, 'text', str(output))
        code = extract_code(code_text)
                
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
