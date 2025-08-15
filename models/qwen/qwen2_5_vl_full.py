#!/usr/bin/env python3
"""
QwenVL 2.5 Full Evaluation for All 300 Rush Hour Puzzles
Using the correct Qwen2.5-VL model and API
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
import re
from typing import List, Dict, Optional
import traceback

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

def load_all_puzzles(data_path: Path = Path("../../data")) -> List[Dict]:
    """Load all puzzles from the data directory"""
    
    puzzle_folders = sorted([f for f in data_path.glob("puzzle*") if f.is_dir()])
    
    if not puzzle_folders:
        print(f"❌ No puzzles found in {data_path}")
        return []
    
    print(f"📁 Found {len(puzzle_folders)} puzzle folders")
    
    puzzles = []
    failed_puzzles = []
    
    for puzzle_folder in puzzle_folders:
        try:
            puzzle_data = {}
            
            # Load prompt
            prompt_file = puzzle_folder / "prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    puzzle_data['prompt'] = f.read().strip()
            else:
                print(f"⚠️ No prompt.txt found in {puzzle_folder.name}")
                failed_puzzles.append(puzzle_folder.name)
                continue
            
            # Load image
            image_file = puzzle_folder / "initial_state.png"
            if image_file.exists():
                puzzle_data['image'] = Image.open(image_file)
            else:
                print(f"⚠️ No initial_state.png found in {puzzle_folder.name}")
                failed_puzzles.append(puzzle_folder.name)
                continue
            
            # Load reference solution (optional)
            solution_file = puzzle_folder / "solution.txt"
            if solution_file.exists():
                with open(solution_file, 'r') as f:
                    puzzle_data['reference'] = f.read().strip()
            
            puzzle_data['name'] = puzzle_folder.name
            puzzle_data['folder_path'] = puzzle_folder
            puzzles.append(puzzle_data)
            
        except Exception as e:
            print(f"❌ Error loading puzzle {puzzle_folder.name}: {e}")
            failed_puzzles.append(puzzle_folder.name)
    
    print(f"✅ Successfully loaded {len(puzzles)} puzzles")
    if failed_puzzles:
        print(f"⚠️ Failed to load {len(failed_puzzles)} puzzles: {failed_puzzles[:5]}{'...' if len(failed_puzzles) > 5 else ''}")
    
    return puzzles

def query_qwen25_model(processor, model, process_vision_info, prompt: str, image: Image.Image) -> str:
    """Query Qwen2.5-VL model using the correct API"""
    
    try:
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
        return response
        
    except Exception as e:
        print(f"❌ Error querying model: {e}")
        return f"ERROR: {str(e)}"

def parse_solution(response: str) -> Optional[List[Dict]]:
    """Parse solution from response"""
    
    # Look for <solution> tags first
    solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
    if solution_match:
        solution_text = solution_match.group(1).strip()
    else:
        solution_text = response
    
    # Extract moves
    step_pattern = r'Step\s+\d+:\s*([A-Z]\d*)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
    matches = re.findall(step_pattern, solution_text, re.IGNORECASE)
    
    if not matches:
        return None
    
    moves = []
    for match in matches:
        piece, start_row, start_col, end_row, end_col = match
        moves.append({
            'piece': piece.upper(),
            'start': (int(start_row), int(start_col)),
            'end': (int(end_row), int(end_col))
        })
    
    return moves

def save_individual_result(puzzle_data: Dict, response: str, moves: List[Dict], inference_time: float, output_dir: Path):
    """Save individual puzzle result"""
    
    results = {
        'puzzle_name': puzzle_data['name'],
        'model': 'qwen2.5_vl_3b',
        'response': response,
        'parsed_moves': moves,
        'inference_time': inference_time,
        'timestamp': time.time(),
        'image_size': puzzle_data['image'].size if 'image' in puzzle_data else None,
        'prompt_length': len(puzzle_data['prompt']) if 'prompt' in puzzle_data else 0
    }
    
    # Save individual result
    result_file = output_dir / f"{puzzle_data['name']}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

def save_summary_results(all_results: List[Dict], output_dir: Path, total_time: float):
    """Save summary of all results"""
    
    summary = {
        'total_puzzles': len(all_results),
        'total_time': total_time,
        'average_time_per_puzzle': total_time / len(all_results) if all_results else 0,
        'successful_parses': sum(1 for r in all_results if r['parsed_moves'] is not None),
        'failed_parses': sum(1 for r in all_results if r['parsed_moves'] is None),
        'error_responses': sum(1 for r in all_results if r['response'].startswith('ERROR:')),
        'model': 'qwen2.5_vl_3b',
        'timestamp': time.time(),
        'puzzle_names': [r['puzzle_name'] for r in all_results]
    }
    
    # Save summary
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save detailed results
    detailed_file = output_dir / "all_results.json"
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"💾 Summary saved to {summary_file}")
    print(f"💾 Detailed results saved to {detailed_file}")
    
    return summary

def main():
    """Main evaluation function"""
    
    print("🚗 Qwen2.5-VL Full Rush Hour Evaluation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Test model availability
    print("\n🔧 Loading Qwen2.5-VL model...")
    processor, model, process_vision_info = test_qwen25_availability()
    if processor is None or model is None:
        print("❌ Cannot proceed without Qwen2.5-VL model")
        return 1
    
    # Load all puzzles
    print("\n📋 Loading all puzzles...")
    puzzles = load_all_puzzles()
    if not puzzles:
        print("❌ Cannot proceed without puzzle data")
        return 1
    
    print(f"✅ Loaded {len(puzzles)} puzzles for evaluation")
    
    # Process all puzzles
    all_results = []
    failed_puzzles = []
    start_time = time.time()
    
    for i, puzzle_data in enumerate(puzzles, 1):
        puzzle_name = puzzle_data['name']
        print(f"\n🔄 Processing puzzle {i}/{len(puzzles)}: {puzzle_name}")
        
        try:
            # Query model
            puzzle_start_time = time.time()
            response = query_qwen25_model(
                processor, model, process_vision_info, 
                puzzle_data['prompt'], puzzle_data['image']
            )
            inference_time = time.time() - puzzle_start_time
            
            if response.startswith("ERROR:"):
                print(f"❌ Model query failed for {puzzle_name}: {response}")
                failed_puzzles.append(puzzle_name)
                # Still save the error result
                moves = None
            else:
                print(f"✅ Response received ({len(response)} chars, {inference_time:.2f}s)")
                # Parse solution
                moves = parse_solution(response)
                if moves:
                    print(f"✅ Parsed {len(moves)} moves")
                else:
                    print("⚠️ Could not parse moves from response")
            
            # Save individual result
            result = save_individual_result(puzzle_data, response, moves, inference_time, output_dir)
            all_results.append(result)
            
            # Print progress
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                estimated_total = avg_time * len(puzzles)
                remaining = estimated_total - elapsed
                print(f"📊 Progress: {i}/{len(puzzles)} ({i/len(puzzles)*100:.1f}%) - "
                      f"Avg: {avg_time:.1f}s/puzzle - ETA: {remaining/60:.1f}min")
            
        except Exception as e:
            print(f"❌ Critical error processing {puzzle_name}: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            failed_puzzles.append(puzzle_name)
            
            # Save error result
            error_result = {
                'puzzle_name': puzzle_name,
                'model': 'qwen2.5_vl_3b',
                'response': f"CRITICAL_ERROR: {str(e)}",
                'parsed_moves': None,
                'inference_time': 0,
                'timestamp': time.time()
            }
            all_results.append(error_result)
            
            # Continue with next puzzle
            continue
    
    total_time = time.time() - start_time
    
    # Save summary results
    print(f"\n📊 Saving results...")
    summary = save_summary_results(all_results, output_dir, total_time)
    
    # Display final summary
    print(f"\n🎉 Evaluation completed!")
    print(f"📊 Final Summary:")
    print(f"  Total puzzles processed: {summary['total_puzzles']}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average time per puzzle: {summary['average_time_per_puzzle']:.2f} seconds")
    print(f"  Successful parses: {summary['successful_parses']}")
    print(f"  Failed parses: {summary['failed_parses']}")
    print(f"  Error responses: {summary['error_responses']}")
    print(f"  Success rate: {summary['successful_parses']/summary['total_puzzles']*100:.1f}%")
    
    if failed_puzzles:
        print(f"\n⚠️ Puzzles with critical failures:")
        for puzzle_name in failed_puzzles[:10]:  # Show first 10
            print(f"  - {puzzle_name}")
        if len(failed_puzzles) > 10:
            print(f"  ... and {len(failed_puzzles) - 10} more")
    
    print(f"\n📁 All results saved in: {output_dir.absolute()}")
    print(f"  - Individual results: {len(all_results)} JSON files")
    print(f"  - Summary: evaluation_summary.json")
    print(f"  - Complete dataset: all_results.json")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())