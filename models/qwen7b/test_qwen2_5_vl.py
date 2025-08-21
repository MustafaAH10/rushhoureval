#!/usr/bin/env python3
"""
QwenVL 2.5 End-to-End Test for Rush Hour Puzzles
Using the correct Qwen2.5-VL model and API
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
import re
from typing import List, Dict, Optional

def test_qwen25_availability():
    """Test if Qwen2.5-VL can be imported and loaded"""
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        print("✅ Transformers available")
        
        # Try to install qwen_vl_utils if needed
        try:
            from qwen_vl_utils import process_vision_info
            print("✅ qwen_vl_utils available")
        except ImportError:
            print("⚠️ qwen_vl_utils not found, installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "qwen-vl-utils"])
            from qwen_vl_utils import process_vision_info
            print("✅ qwen_vl_utils installed and imported")
        
        # Try to load Qwen2.5-VL
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"  # Using non-AWQ version for compatibility
        print(f"📥 Loading {model_id}...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")
        
        if device == "cuda":
            # Try to use GPU with CPU offloading for large model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_buffers=True,
                max_memory={0: "4GiB", "cpu": "8GiB"}
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"✅ Qwen2.5-VL loaded successfully")
        print(f"🔧 Device: {model.device if hasattr(model, 'device') else 'unknown'}")
        
        return processor, model, process_vision_info
        
    except Exception as e:
        print(f"❌ Error loading Qwen2.5-VL: {e}")
        return None, None, None

def load_sample_puzzle(data_path: Path = Path("../../data")) -> Dict:
    """Load first puzzle for testing"""
    
    puzzle_folders = sorted([f for f in data_path.glob("puzzle*") if f.is_dir()])
    
    if not puzzle_folders:
        print(f"❌ No puzzles found in {data_path}")
        return None
    
    puzzle_folder = puzzle_folders[0]  # Use first puzzle
    print(f"📁 Using puzzle: {puzzle_folder.name}")
    
    puzzle_data = {}
    
    # Load prompt
    prompt_file = puzzle_folder / "prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            puzzle_data['prompt'] = f.read().strip()
        print(f"✅ Loaded prompt ({len(puzzle_data['prompt'])} chars)")
    else:
        print("❌ No prompt.txt found")
        return None
    
    # Load image
    image_file = puzzle_folder / "initial_state.png"
    if image_file.exists():
        puzzle_data['image'] = Image.open(image_file)
        print(f"✅ Loaded image {puzzle_data['image'].size}")
    else:
        print("❌ No initial_state.png found")
        return None
    
    # Load reference solution
    solution_file = puzzle_folder / "solution.txt"
    if solution_file.exists():
        with open(solution_file, 'r') as f:
            puzzle_data['reference'] = f.read().strip()
        print(f"✅ Loaded reference solution")
    
    puzzle_data['name'] = puzzle_folder.name
    return puzzle_data

def query_qwen25_model(processor, model, process_vision_info, prompt: str, image: Image.Image) -> str:
    """Query Qwen2.5-VL model using the correct API"""
    
    try:
        print("🤖 Querying Qwen2.5-VL...")
        start_time = time.time()
        
        # Create messages in the correct format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # Pass PIL image directly
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0] if output_text else ""
        inference_time = time.time() - start_time
        print(f"✅ Response generated ({inference_time:.2f}s)")
        
        return response
        
    except Exception as e:
        print(f"❌ Error querying model: {e}")
        return f"ERROR: {str(e)}"

def parse_solution(response: str) -> Optional[List[Dict]]:
    """Parse solution from response"""
    
    print("🔍 Parsing solution...")
    
    # Look for <solution> tags first
    solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
    if solution_match:
        solution_text = solution_match.group(1).strip()
        print("✅ Found <solution> tags")
    else:
        solution_text = response
        print("⚠️ No <solution> tags, parsing full response")
    
    # Extract moves
    step_pattern = r'Step\s+\d+:\s*([A-Z]\d*)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
    matches = re.findall(step_pattern, solution_text, re.IGNORECASE)
    
    if not matches:
        print("❌ No valid moves found")
        return None
    
    moves = []
    for match in matches:
        piece, start_row, start_col, end_row, end_col = match
        moves.append({
            'piece': piece.upper(),
            'start': (int(start_row), int(start_col)),
            'end': (int(end_row), int(end_col))
        })
    
    print(f"✅ Found {len(moves)} moves")
    return moves

def save_results(puzzle_data: Dict, response: str, moves: List[Dict], inference_time: float):
    """Save test results"""
    
    results = {
        'puzzle_name': puzzle_data['name'],
        'model': 'qwen2.5_vl_3b',
        'response': response,
        'parsed_moves': moves,
        'inference_time': inference_time,
        'timestamp': time.time()
    }
    
    # Create results directory
    results_dir = Path("../../test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save individual result
    result_file = results_dir / f"{puzzle_data['name']}_qwen25_test.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {result_file}")
    return results

def main():
    """Main test function"""
    
    print("🚗 Qwen2.5-VL Rush Hour End-to-End Test")
    print("=" * 50)
    
    # Test model availability
    processor, model, process_vision_info = test_qwen25_availability()
    if processor is None or model is None:
        print("❌ Cannot proceed without Qwen2.5-VL model")
        return 1
    
    # Load sample puzzle
    puzzle_data = load_sample_puzzle()
    if puzzle_data is None:
        print("❌ Cannot proceed without puzzle data")
        return 1
    
    # Show puzzle info
    print(f"\n📋 Puzzle Information:")
    print(f"  Name: {puzzle_data['name']}")
    print(f"  Image size: {puzzle_data['image'].size}")
    print(f"  Prompt length: {len(puzzle_data['prompt'])} characters")
    
    # Query model
    print(f"\n🤖 Querying Qwen2.5-VL...")
    start_time = time.time()
    response = query_qwen25_model(processor, model, process_vision_info, puzzle_data['prompt'], puzzle_data['image'])
    inference_time = time.time() - start_time
    
    if response.startswith("ERROR:"):
        print(f"❌ Model query failed: {response}")
        return 1
    
    print(f"✅ Response received ({len(response)} chars)")
    
    # Parse solution
    moves = parse_solution(response)
    
    # Save results
    results = save_results(puzzle_data, response, moves, inference_time)
    
    # Display results
    print(f"\n📊 Results Summary:")
    print(f"  Inference time: {inference_time:.2f} seconds")
    print(f"  Response length: {len(response)} characters")
    print(f"  Moves parsed: {len(moves) if moves else 0}")
    
    print(f"\n💭 Model Response:")
    print("-" * 40)
    print(response[:500] + "..." if len(response) > 500 else response)
    print("-" * 40)
    
    if moves:
        print(f"\n🎯 Parsed Moves:")
        for i, move in enumerate(moves, 1):
            print(f"  Step {i}: {move['piece']} {move['start']} -> {move['end']}")
    
    # Compare with reference if available
    if 'reference' in puzzle_data:
        print(f"\n📚 Reference Solution:")
        ref_lines = puzzle_data['reference'].split('\n')
        solution_started = False
        for line in ref_lines:
            if line.strip() == "Solution:":
                solution_started = True
                continue
            if solution_started and line.strip().startswith("Step"):
                print(f"  {line.strip()}")
    
    print(f"\n🎉 End-to-end test completed successfully!")
    print(f"📁 Detailed results saved in: test_results/")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())