import os
import json
import textwrap
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
        
    # Hallucination fixes
    code = code.replace(".rectangle(", ".rect(")
    # Remove or mock out show_object to prevent execution failure if the model includes it
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

def check_syntax(code):
    """Checks if the code is valid CadQuery by executing it in a local scope."""
    import cadquery as cq
    # Standard mocks
    def show_object(*args, **kwargs): pass
    
    local_vars = {"cq": cq, "show_object": show_object}
    try:
        exec(code, {}, local_vars)
        if "result" in local_vars:
            return True, None
        return False, "Variable 'result' not found in execution scope."
    except Exception as e:
        return False, str(e)

def run_enhanced():
    print("Loading MLX VLM model (Enhanced with Reflexion)...")
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, processor = load(model_path)
    ds = load_data()
    results = {}
    
    print("Generating hyper-optimized predictions...")
    for idx, item in enumerate(tqdm(ds)):
        image = item["image"]
        deepcad_id = item["deepcad_id"]
        
        prompt = "Write CadQuery Python code to generate the 3D shape shown in this image. Remember to assign the final object to a variable named 'result'."
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        
        # Round 1: Initial Generation
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        output = generate(model, processor, formatted_prompt, [image], max_tokens=1024, temperature=0.1)
        code_text = getattr(output, 'text', str(output))
        code = extract_code(code_text)
        
        # Round 2: Reflexion (Self-Correction)
        is_valid, error_msg = check_syntax(code)
        if not is_valid:
            # Feed back the error to the model
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"CODE:\n{code}"}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": f"Your code failed with error: {error_msg}. Please fix the code and provide only the corrected Python block."}]})
            
            formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            output = generate(model, processor, formatted_prompt, [image], max_tokens=1024, temperature=0.1)
            code_text = getattr(output, 'text', str(output))
            code = extract_code(code_text)
            
        results[deepcad_id] = code
        
    os.makedirs("results", exist_ok=True)
    with open("results/enhanced_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Enhanced generation complete (with Reflexion). Saved to results/enhanced_predictions.json")

if __name__ == "__main__":
    run_enhanced()
