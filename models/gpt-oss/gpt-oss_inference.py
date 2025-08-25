import os
import json
import csv
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from datetime import datetime


class GPTOSSInference:
    def __init__(self, model_name="openai/gpt-oss-20b", device="auto"):
        """
        Initialize GPT-OSS model for Rush Hour puzzle inference.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run inference on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # System prompt for Rush Hour puzzles
        self.system_prompt = """You are an expert puzzle solver specializing in Rush Hour puzzles. Your task is to find the optimal sequence of moves to move the car 'C' to the TARGET position.

Key Instructions:
1. A 1-indexed coordinate system is being used
2. Each piece (car or blocker) can only move UP, DOWN, LEFT, or RIGHT by exactly one square
3. Pieces CANNOT move outside the grid or into occupied squares at any instant
4. Analyze the puzzle step by step and provide your reasoning
5. Provide your solution in the exact format requested

Be precise with coordinates and piece movements. Think through the problem logically."""

    def load_model(self):
        """Load GPT-OSS model with mxfp4 optimization"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading GPT-OSS model: {self.model_name}")
            print("Installing required dependencies for mxfp4...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with mxfp4 optimization
            print("Loading model with mxfp4 quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",  # This enables mxfp4 when available
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Model loaded successfully")
            print(f"✅ Device: {self.model.device if hasattr(self.model, 'device') else 'distributed'}")
            print(f"✅ Model dtype: {self.model.dtype}")
            print(f"✅ Using mxfp4 quantization for memory efficiency")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Make sure you have installed:")
            print("pip install --upgrade transformers kernels accelerate \"triton>=3.4\"")
            raise

    def create_puzzle_prompt_from_json(self, puzzle_metadata: Dict[str, Any]) -> str:
        """
        Create puzzle prompt using grid from puzzle_state.json
        
        Args:
            puzzle_metadata: Loaded puzzle_state.json data
            
        Returns:
            Puzzle prompt string with JSON grid
        """
        grid = puzzle_metadata.get('grid', [])
        car_position = puzzle_metadata.get('car_position', [])
        exit_position = puzzle_metadata.get('exit_position', [])
        pieces = puzzle_metadata.get('pieces', {})
        
        if not grid or not car_position or not exit_position:
            raise ValueError("Missing required puzzle data in JSON")
        
        # Convert grid to JSON string
        grid_json = json.dumps(grid, separators=(',', ':'))
        
        # Find blockers
        blockers = []
        for piece_name, piece_data in pieces.items():
            if piece_name != 'C' and piece_data.get('type') == 'blocker':
                pos = piece_data.get('position', [])
                if pos:
                    blockers.append(f"{piece_name} at [{pos[0]},{pos[1]}]")
        
        prompt = f"""Task: Solve this 3x3 Rush Hour puzzle - move car "C" from position [{car_position[0]},{car_position[1]}] to the TARGET at position [{exit_position[0]},{exit_position[1]}] given the position of the blockers below.

Current Grid State (JSON format):
{grid_json}

Current Pieces:
- Car "C": Position [{car_position[0]},{car_position[1]}]
- Blockers: {', '.join(blockers) if blockers else 'None'}
- TARGET: Position [{exit_position[0]},{exit_position[1]}]

Rules:
- Any piece can move UP, DOWN, LEFT, or RIGHT by exactly one square
- Pieces cannot move outside the 3x3 grid
- Pieces cannot move into occupied squares
- No two pieces can occupy the same square at any instant
- Goal: Move car "C" to the TARGET position

Coordinate System: [row,col] format where [1,1] is top-left, [3,3] is bottom-right

Please analyze this puzzle step by step and provide your solution in this format:
<solution>
Step 1: [PIECE] [start_position] -> [end_position]
Step 2: [PIECE] [start_position] -> [end_position]
...
</solution>

Example response format:
<solution>
Step 1: B2 [2,3] -> [3,3]
Step 2: B1 [2,2] -> [1,2]
Step 3: C [2,1] -> [2,2]
Step 4: C [2,2] -> [2,3]
</solution>
"""
        return prompt

    def create_chat_messages(self, puzzle_prompt: str) -> List[Dict[str, str]]:
        """
        Create chat messages for GPT-OSS model.
        
        Args:
            puzzle_prompt: The puzzle-specific prompt
            
        Returns:
            List of chat messages
        """
        messages = [
            {"role": "developer", "content": self.system_prompt},  # GPT-OSS uses "developer" instead of "system"
            {"role": "user", "content": puzzle_prompt}
        ]
        return messages

    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 4096) -> str:
        """
        Generate response using GPT-OSS model.
        
        Args:
            messages: Chat messages
            max_new_tokens: Maximum tokens to generate (GPT-OSS needs large values for reasoning)
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            # Apply chat template with GPT-OSS specific settings
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort="medium"  # GPT-OSS specific: can be "low", "medium", or "high"
            ).to(self.model.device)
            
            # Generate response with GPT-OSS recommended parameters
            print("🤖 Generating response with reasoning...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,  # GPT-OSS default
                    top_p=1.0,        # GPT-OSS default
                    top_k=40,         # GPT-OSS default
                    min_p=0.0,        # GPT-OSS default
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Important: skip_special_tokens=False to capture reasoning traces
                )
            
            # Decode response with special tokens to capture reasoning
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"ERROR: {str(e)}"

    def parse_gpt_oss_response(self, response: str) -> Dict[str, str]:
        """
        Parse GPT-OSS response to extract reasoning and final answer.
        GPT-OSS uses channels: analysis (reasoning) and final (answer).
        
        Args:
            response: Raw model response
            
        Returns:
            Dictionary with 'reasoning', 'final_answer', and 'full_response'
        """
        result = {
            'reasoning': '',
            'final_answer': '',
            'full_response': response
        }
        
        # GPT-OSS structure: <|start|>assistant<|channel|>analysis<|message|>REASONING<|end|><|start|>assistant<|channel|>final<|message|>ANSWER
        
        # Extract reasoning (analysis channel)
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>assistant<\|channel\|>final<\|message\|>)'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL)
        if analysis_match:
            result['reasoning'] = analysis_match.group(1).strip()
        
        # Extract final answer (final channel)
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
        final_match = re.search(final_pattern, response, re.DOTALL)
        if final_match:
            result['final_answer'] = final_match.group(1).strip()
        else:
            # Fallback: if no channels found, treat entire response as final answer
            result['final_answer'] = response
        
        return result

    def parse_solution(self, final_answer: str) -> List[str]:
        """
        Parse solution steps from final answer.
        
        Args:
            final_answer: Final answer from GPT-OSS
            
        Returns:
            List of solution steps
        """
        solution_steps = []
        
        # Look for <solution> tags
        solution_match = re.search(r'<solution>(.*?)</solution>', final_answer, re.DOTALL | re.IGNORECASE)
        if solution_match:
            solution_text = solution_match.group(1).strip()
        else:
            # Try to find steps in the response directly
            solution_text = final_answer
        
        # Extract individual steps
        step_pattern = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
        steps = re.findall(step_pattern, solution_text, re.IGNORECASE)
        
        for i, (piece, start_row, start_col, end_row, end_col) in enumerate(steps):
            step_text = f"Step {i+1}: {piece} [{start_row},{start_col}] -> [{end_row},{end_col}]"
            solution_steps.append(step_text)
        
        return solution_steps

    def load_puzzle_metadata(self, puzzle_folder: str) -> Dict[str, Any]:
        """Load puzzle metadata from JSON file."""
        json_file = os.path.join(puzzle_folder, "puzzle_state.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error reading JSON file {json_file}: {e}")
                return {}
        else:
            print(f"❌ JSON file not found: {json_file}")
            return {}

    def save_puzzle_result(self, output_path: str, puzzle_num: int, 
                          prompt: str, raw_response: str, parsed_response: Dict[str, str],
                          parsed_solution: List[str], metadata: Dict[str, Any], processing_time: float):
        """Save individual puzzle result with reasoning traces"""
        puzzle_result_folder = os.path.join(output_path, f"puzzle{puzzle_num}")
        os.makedirs(puzzle_result_folder, exist_ok=True)
        
        result = {
            'model_info': {
                'model_name': self.model_name,
                'inference_method': 'transformers_mxfp4',
                'model_type': 'GPT-OSS-20B',
                'reasoning_effort': 'medium'
            },
            'puzzle_info': {
                'puzzle_num': puzzle_num,
                'difficulty': metadata.get('puzzle_info', {}).get('difficulty', 'unknown'),
                'num_blockers': metadata.get('puzzle_info', {}).get('num_blockers', 0),
                'optimal_solution_length': metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                'processing_time_seconds': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            },
            'prompt': prompt,
            'raw_response': raw_response,
            'reasoning_trace': parsed_response['reasoning'],
            'final_answer': parsed_response['final_answer'],
            'parsed_solution': parsed_solution,
            'analysis': {
                'predicted_solution_length': len(parsed_solution),
                'length_matches_optimal': len(parsed_solution) == metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                'solution_found': len(parsed_solution) > 0,
                'parsing_successful': len(parsed_solution) > 0,
                'has_reasoning_trace': len(parsed_response['reasoning']) > 0,
                'response_length_chars': len(raw_response),
                'reasoning_length_chars': len(parsed_response['reasoning']),
                'final_answer_length_chars': len(parsed_response['final_answer'])
            }
        }
        
        result_file = os.path.join(puzzle_result_folder, f"gpt_oss_puzzle{puzzle_num}_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result_file

    def run_inference_on_dataset(self, dataset_path: str = "../../data/3x3", 
                                output_path: str = "gpt_oss_results3x3", 
                                max_puzzles: Optional[int] = None,
                                start_puzzle: int = 1):
        """Run inference on all puzzles in the dataset."""
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Find puzzle folders
        puzzle_folders = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and item.startswith("puzzle"):
                try:
                    puzzle_num = int(item.replace("puzzle", ""))
                    if puzzle_num >= start_puzzle:
                        puzzle_folders.append((puzzle_num, item_path))
                except ValueError:
                    continue
        
        puzzle_folders.sort(key=lambda x: x[0])
        
        if max_puzzles:
            puzzle_folders = puzzle_folders[:max_puzzles]
        
        print(f"Found {len(puzzle_folders)} puzzles to process")
        print(f"Starting inference with GPT-OSS-20B model...")
        
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (puzzle_num, puzzle_folder) in enumerate(puzzle_folders):
            print(f"\n{'='*60}")
            print(f"Processing Puzzle {puzzle_num} ({i+1}/{len(puzzle_folders)})")
            print(f"Folder: {puzzle_folder}")
            
            start_time = time.time()
            
            try:
                metadata = self.load_puzzle_metadata(puzzle_folder)
                if not metadata:
                    print(f"❌ Skipping puzzle {puzzle_num} - no metadata found")
                    continue
                
                puzzle_prompt = self.create_puzzle_prompt_from_json(metadata)
                messages = self.create_chat_messages(puzzle_prompt)
                
                print("🤖 Generating solution with reasoning...")
                raw_response = self.generate_response(messages, max_new_tokens=4096)
                
                # Parse GPT-OSS response to extract reasoning and final answer
                parsed_response = self.parse_gpt_oss_response(raw_response)
                parsed_solution = self.parse_solution(parsed_response['final_answer'])
                
                processing_time = time.time() - start_time
                
                result_file = self.save_puzzle_result(
                    output_path=output_path,
                    puzzle_num=puzzle_num,
                    prompt=puzzle_prompt,
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    parsed_solution=parsed_solution,
                    metadata=metadata,
                    processing_time=processing_time
                )
                
                result_summary = {
                    'puzzle_num': puzzle_num,
                    'puzzle_folder': os.path.basename(puzzle_folder),
                    'difficulty': metadata.get('puzzle_info', {}).get('difficulty', 'unknown'),
                    'num_blockers': metadata.get('puzzle_info', {}).get('num_blockers', 0),
                    'optimal_solution_length': metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                    'predicted_solution_length': len(parsed_solution),
                    'length_matches_optimal': len(parsed_solution) == metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                    'solution_found': len(parsed_solution) > 0,
                    'has_reasoning_trace': len(parsed_response['reasoning']) > 0,
                    'reasoning_length_chars': len(parsed_response['reasoning']),
                    'processing_time_seconds': round(processing_time, 2),
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result_summary)
                
                print(f"✅ Generated solution with {len(parsed_solution)} steps")
                print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"🎯 Optimal solution: {metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0)} steps")
                print(f"🧠 Reasoning trace: {len(parsed_response['reasoning'])} characters")
                print(f"📁 Result saved to: {result_file}")
                
                if (i + 1) % 5 == 0:  # Save progress more frequently due to longer processing times
                    self.save_results_summary(results, output_path, timestamp)
            
            except Exception as e:
                print(f"❌ Error processing puzzle {puzzle_num}: {e}")
                continue
        
        self.save_results_summary(results, output_path, timestamp)
        
        print(f"\n🎉 Inference complete!")
        print(f"Processed {len(results)} puzzles")
        print(f"Results saved to: {output_path}")

    def save_results_summary(self, results: List[Dict], output_path: str, timestamp: str):
        """Save results summary in multiple formats"""
        json_file = os.path.join(output_path, f"gpt_oss_inference_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        csv_file = os.path.join(output_path, f"gpt_oss_inference_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = [
                    'puzzle_num', 'puzzle_folder', 'difficulty', 'num_blockers',
                    'optimal_solution_length', 'predicted_solution_length', 
                    'length_matches_optimal', 'solution_found', 'has_reasoning_trace',
                    'reasoning_length_chars', 'processing_time_seconds', 'timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
        
        print(f"📊 Results summary saved to: {csv_file}")


def main():
    """Main execution function"""
    print("🚀 Starting GPT-OSS-20B Rush Hour Inference with mxfp4")
    print("=" * 70)
    
    # Initialize inference system
    inference = GPTOSSInference(
        model_name="openai/gpt-oss-20b",
        device="auto"
    )
    
    # Load model
    try:
        inference.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Make sure you have installed the required dependencies:")
        print("pip install --upgrade transformers kernels accelerate \"triton>=3.4\"")
        return
    
    # Configuration
    dataset_path = "/root/rushhoureval/data/3x3"  # Update this path as needed
    output_path = "gpt_oss_results3x3"
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n📁 Dataset path: {dataset_path}")
    print(f"📁 Output path: {output_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please update the dataset_path variable to point to your dataset.")
        return
    
    # Run inference
    inference.run_inference_on_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        max_puzzles=None,  # Process all puzzles
        start_puzzle=1
    )
    
    print("✅ Inference pipeline completed!")
    print("\n📋 Key features:")
    print("1. Uses mxfp4 quantization for memory efficiency")
    print("2. Captures full reasoning traces in analysis channel")
    print("3. Separates reasoning from final answers")
    print("4. Uses GPT-OSS recommended generation parameters")
    print("5. Saves detailed results with reasoning analysis")


if __name__ == "__main__":
    main()