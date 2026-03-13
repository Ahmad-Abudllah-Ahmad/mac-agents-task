import os
import json
import textwrap
from tqdm import tqdm
from datasets import load_dataset
from mlx_vlm import load, generate

def load_data():
    print("Loading test dataset...")
    try:
        ds = load_dataset(
            "CADCODER/GenCAD-Code", 
            split="test", 
            cache_dir="/Volumes/BIG-DATA/HUGGINGFACE_CACHE"
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # fallback without custom cache dir if it fails
        ds = load_dataset("CADCODER/GenCAD-Code", split="test")
        
    ds = ds.shuffle(seed=42)
    return ds.select(range(50))

SYSTEM_PROMPT = """You are an expert in 3D CAD modeling using Python's CadQuery library.
Your task is to write CadQuery Python code to reproduce the 3D shape shown in the provided image.

CRITICAL RULES:
1. ONLY write valid Python code. Do not include markdown formatting or explanations.
2. You MUST define all variables before using them.
3. Your final solid object must be assigned to a variable named exactly 'result'.
4. Import 'cadquery as cq' at the top of your script.
5. All geometric parameters must be defined as variables with numerical values.

Example 1:
import cadquery as cq

height = 60.0
width = 80.0
thickness = 10.0

result = cq.Workplane("XY").box(height, width, thickness)

Example 2:
import cadquery as cq

height = 60.0
width = 80.0
thickness = 10.0
diameter = 22.0
padding = 12.0

result = (
    cq.Workplane("XY")
    .box(height, width, thickness)
    .faces(">Z")
    .workplane()
    .hole(diameter)
    .faces(">Z")
    .workplane()
    .rect(height - padding, width - padding, forConstruction=True)
    .vertices()
    .cboreHole(2.4, 4.4, 2.1)
)
"""

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
            
    # Quick auto-healing for common LLM mistakes
    if "import cadquery as cq" not in code:
        code = "import cadquery as cq\n\n" + code
        
    if "result =" not in code and "result=" not in code:
        if "solid =" in code:
            code = code.replace("solid =", "result =")
        elif "shape =" in code:
            code = code.replace("shape =", "result =")
            
    return code

def run_enhanced():
    print("Loading MLX VLM model (Enhanced)...")
    # Using 4-bit Qwen2-VL 
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    
    model, processor = load(model_path)
    
    ds = load_data()
    
    results = {}
    
    print("Generating enhanced predictions...")
    for idx, item in enumerate(tqdm(ds)):
        image = item["image"]
        deepcad_id = item["deepcad_id"]
        
        prompt = "Write CadQuery Python code to generate the 3D shape shown in this image. Remember to assign the final object to a variable named 'result'."
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"}, 
                {"type": "text", "text": prompt}
            ]}
        ]
        
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        output = generate(model, processor, formatted_prompt, [image], max_tokens=1024, temperature=0.2)
        
        code = extract_code(output)
        results[deepcad_id] = code
        
    os.makedirs("results", exist_ok=True)
    with open("results/enhanced_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Enhanced generation complete. Saved to results/enhanced_predictions.json")

if __name__ == "__main__":
    run_enhanced()
