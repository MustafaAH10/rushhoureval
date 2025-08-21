import os
import json
import csv
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


class QwenRushHourInference:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="auto"):
        """
        Initialize Qwen model for Rush Hour puzzle inference.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to run inference on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # System prompt for Rush Hour puzzles
        self.system_prompt = """You are an expert puzzle solver specializing in Rush Hour puzzles. Your task is to find the optimal solution to move the car 'C' to the TARGET position.

Key Instructions:
1. Analyze the current grid state carefully
2. Think step-by-step about which pieces need to move
3. Each piece (car or blocker) can only move UP, DOWN, LEFT, or RIGHT by exactly one square
4. Pieces cannot move outside the grid or into occupied squares
5. Provide your solution in the exact format requested

Be precise with coordinates and piece movements. Think logically about the sequence of moves needed."""

    def load_model(self):
        """Load Qwen model and tokenizer"""
        print(f"Loading Qwen model: {self.model_name}")
        print("This may take a few minutes...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Model loaded successfully on device: {self.model.device}")
            print(f"✅ Model dtype: {self.model.dtype}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
        
        prompt = f"""Task: Solve this 3x3 Rush Hour puzzle. Move car "C" from position [{car_position[0]},{car_position[1]}] to the TARGET at position [{exit_position[0]},{exit_position[1]}].

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
- No two pieces can occupy the same square
- Goal: Move car "C" to the TARGET position

Coordinate System: [row,col] format where [1,1] is top-left, [3,3] is bottom-right

Provide your solution as:
<solution>
Step 1: [PIECE] [start_position] -> [end_position]
Step 2: [PIECE] [start_position] -> [end_position]
...
</solution>

Example response format:
<solution>
Step 1: C [2,1] -> [2,2]
Step 2: B1 [1,3] -> [1,2]  
Step 3: C [2,2] -> [1,2]
</solution>
"""
        return prompt

    def create_chat_prompt(self, puzzle_prompt: str) -> str:
        """
        Create a chat-formatted prompt for Qwen model.
        
        Args:
            puzzle_prompt: The puzzle-specific prompt
            
        Returns:
            Formatted chat prompt string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": puzzle_prompt}
        ]
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback manual format
            chat_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            chat_prompt += f"<|im_start|>user\n{puzzle_prompt}<|im_end|>\n"
            chat_prompt += "<|im_start|>assistant\n"
            return chat_prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Generate response from Qwen model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response (remove input prompt)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"ERROR: {str(e)}"

    def parse_solution(self, response: str) -> List[str]:
        """
        Parse solution steps from model response.
        
        Args:
            response: Raw model response
            
        Returns:
            List of solution steps
        """
        solution_steps = []
        
        # Look for <solution> tags
        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
        
        if solution_match:
            solution_text = solution_match.group(1).strip()
            
            # Extract individual steps
            step_pattern = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
            steps = re.findall(step_pattern, solution_text, re.IGNORECASE)
            
            for i, (piece, start_row, start_col, end_row, end_col) in enumerate(steps):
                step_text = f"Step {i+1}: {piece} [{start_row},{start_col}] -> [{end_row},{end_col}]"
                solution_steps.append(step_text)
        
        return solution_steps

    def load_puzzle_metadata(self, puzzle_folder: str) -> Dict[str, Any]:
        """
        Load puzzle metadata from JSON file.
        
        Args:
            puzzle_folder: Path to puzzle folder
            
        Returns:
            Puzzle metadata dictionary
        """
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
                          prompt: str, raw_response: str, parsed_solution: List[str],
                          metadata: Dict[str, Any], processing_time: float):
        """
        Save individual puzzle result with all essential data in dedicated puzzle folder.
        
        Args:
            output_path: Base results output path
            puzzle_num: Puzzle number
            prompt: The prompt sent to model
            raw_response: Raw model response
            parsed_solution: Parsed solution steps
            metadata: Puzzle metadata
            processing_time: Processing time in seconds
        """
        # Create individual puzzle result folder
        puzzle_result_folder = os.path.join(output_path, f"puzzle{puzzle_num}")
        os.makedirs(puzzle_result_folder, exist_ok=True)
        
        result = {
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
            'parsed_solution': parsed_solution,
            'analysis': {
                'predicted_solution_length': len(parsed_solution),
                'length_matches_optimal': len(parsed_solution) == metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                'solution_found': len(parsed_solution) > 0,
                'parsing_successful': len(parsed_solution) > 0
            }
        }
        
        # Save to individual result file in puzzle folder
        result_file = os.path.join(puzzle_result_folder, f"qwen_puzzle{puzzle_num}_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result_file

    def run_inference_on_dataset(self, dataset_path: str = "data/3x3", 
                                output_path: str = "models/qwen/results", 
                                max_puzzles: Optional[int] = None,
                                start_puzzle: int = 1):
        """
        Run inference on all puzzles in the dataset.
        
        Args:
            dataset_path: Path to dataset folder
            output_path: Path to save results
            max_puzzles: Maximum number of puzzles to process (None for all)
            start_puzzle: Starting puzzle number
        """
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return

        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Find all puzzle folders
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
        
        # Sort by puzzle number
        puzzle_folders.sort(key=lambda x: x[0])
        
        # Limit number of puzzles if specified
        if max_puzzles:
            puzzle_folders = puzzle_folders[:max_puzzles]
        
        print(f"Found {len(puzzle_folders)} puzzles to process")
        print(f"Starting inference with Qwen model...")
        
        # Results storage
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process each puzzle
        for i, (puzzle_num, puzzle_folder) in enumerate(puzzle_folders):
            print(f"\n{'='*60}")
            print(f"Processing Puzzle {puzzle_num} ({i+1}/{len(puzzle_folders)})")
            print(f"Folder: {puzzle_folder}")
            
            start_time = time.time()
            
            try:
                # Load puzzle metadata from JSON
                metadata = self.load_puzzle_metadata(puzzle_folder)
                if not metadata:
                    print(f"❌ Skipping puzzle {puzzle_num} - no metadata found")
                    continue
                
                # Create puzzle prompt from JSON data
                puzzle_prompt = self.create_puzzle_prompt_from_json(metadata)
                
                # Create chat prompt
                chat_prompt = self.create_chat_prompt(puzzle_prompt)
                
                # Generate response
                print("🤖 Generating solution...")
                raw_response = self.generate_response(chat_prompt)
                
                # Parse solution
                parsed_solution = self.parse_solution(raw_response)
                
                processing_time = time.time() - start_time
                
                # Save individual puzzle result with all essential data
                result_file = self.save_puzzle_result(
                    output_path=output_path,
                    puzzle_num=puzzle_num,
                    prompt=puzzle_prompt,
                    raw_response=raw_response,
                    parsed_solution=parsed_solution,
                    metadata=metadata,
                    processing_time=processing_time
                )
                
                # Store for summary
                result_summary = {
                    'puzzle_num': puzzle_num,
                    'puzzle_folder': os.path.basename(puzzle_folder),
                    'difficulty': metadata.get('puzzle_info', {}).get('difficulty', 'unknown'),
                    'num_blockers': metadata.get('puzzle_info', {}).get('num_blockers', 0),
                    'optimal_solution_length': metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                    'predicted_solution_length': len(parsed_solution),
                    'length_matches_optimal': len(parsed_solution) == metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                    'solution_found': len(parsed_solution) > 0,
                    'processing_time_seconds': round(processing_time, 2),
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result_summary)
                
                # Print summary
                print(f"✅ Generated solution with {len(parsed_solution)} steps")
                print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"🎯 Optimal solution: {metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0)} steps")
                print(f"📁 Result saved to: {result_file}")
                
                # Save progress periodically
                if (i + 1) % 10 == 0:
                    self.save_results_summary(results, output_path, timestamp)
            
            except Exception as e:
                print(f"❌ Error processing puzzle {puzzle_num}: {e}")
                continue
        
        # Save final results summary
        self.save_results_summary(results, output_path, timestamp)
        
        print(f"\n🎉 Inference complete!")
        print(f"Processed {len(results)} puzzles")
        print(f"Results saved to: {output_path}")
        print(f"Individual results saved in: {output_path}/puzzle[N]/qwen_puzzle[N]_result.json")

    def save_results_summary(self, results: List[Dict], output_path: str, timestamp: str):
        """Save results summary in multiple formats"""
        
        # Save comprehensive JSON results
        json_file = os.path.join(output_path, f"qwen_inference_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_file = os.path.join(output_path, f"qwen_inference_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = [
                    'puzzle_num', 'puzzle_folder', 'difficulty', 'num_blockers',
                    'optimal_solution_length', 'predicted_solution_length', 
                    'length_matches_optimal', 'solution_found', 'processing_time_seconds', 'timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
        
        print(f"📊 Results summary saved to: {csv_file}")

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze inference results and compute metrics"""
        if not results:
            return {}
        
        total_puzzles = len(results)
        optimal_length_matches = sum(1 for r in results if r.get('length_matches_optimal', False))
        solutions_found = sum(1 for r in results if r.get('solution_found', False))
        
        # Difficulty breakdown
        difficulty_stats = {}
        for result in results:
            diff = result['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'optimal_matches': 0, 'solutions_found': 0}
            difficulty_stats[diff]['total'] += 1
            if result.get('length_matches_optimal', False):
                difficulty_stats[diff]['optimal_matches'] += 1
            if result.get('solution_found', False):
                difficulty_stats[diff]['solutions_found'] += 1
        
        # Processing time stats
        times = [r['processing_time_seconds'] for r in results]
        avg_time = sum(times) / len(times) if times else 0
        
        analysis = {
            'total_puzzles_processed': total_puzzles,
            'solutions_found': solutions_found,
            'solution_rate': solutions_found / total_puzzles if total_puzzles > 0 else 0,
            'optimal_length_matches': optimal_length_matches,
            'optimal_length_accuracy': optimal_length_matches / total_puzzles if total_puzzles > 0 else 0,
            'difficulty_breakdown': difficulty_stats,
            'average_processing_time_seconds': round(avg_time, 2),
            'total_processing_time_seconds': round(sum(times), 2)
        }
        
        return analysis


def main():
    """Main execution function"""
    print("🚀 Starting Qwen 2.5-7B Rush Hour Inference")
    print("=" * 60)
    
    # Initialize inference system
    inference = QwenRushHourInference(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="auto"
    )
    
    # Load model
    try:
        inference.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Configuration
    dataset_path = "data/3x3"
    output_path = "models/qwen/results"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n📁 Dataset path: {dataset_path}")
    print(f"📁 Output path: {output_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run the puzzle generator first to create the dataset.")
        return
    
    # Run inference on all puzzles
    inference.run_inference_on_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        max_puzzles=None,  # Process all puzzles
        start_puzzle=1
    )
    
    print("✅ Inference pipeline completed!")


if __name__ == "__main__":
    main()(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback manual format
            chat_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            chat_prompt += f"<|im_start|>user\n{puzzle_prompt}<|im_end|>\n"
            chat_prompt += "<|im_start|>assistant\n"
            return chat_prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Generate response from Qwen model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response (remove input prompt)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"ERROR: {str(e)}"

    def parse_solution(self, response: str) -> List[str]:
        """
        Parse solution steps from model response.
        
        Args:
            response: Raw model response
            
        Returns:
            List of solution steps
        """
        solution_steps = []
        
        # Look for <solution> tags
        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
        
        if solution_match:
            solution_text = solution_match.group(1).strip()
            
            # Extract individual steps
            step_pattern = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
            steps = re.findall(step_pattern, solution_text, re.IGNORECASE)
            
            for i, (piece, start_row, start_col, end_row, end_col) in enumerate(steps):
                step_text = f"Step {i+1}: {piece} [{start_row},{start_col}] -> [{end_row},{end_col}]"
                solution_steps.append(step_text)
        
        return solution_steps

    def load_puzzle_prompt(self, puzzle_folder: str) -> Optional[str]:
        """
        Load puzzle prompt from folder.
        
        Args:
            puzzle_folder: Path to puzzle folder
            
        Returns:
            Puzzle prompt text or None if not found
        """
        prompt_file = os.path.join(puzzle_folder, "prompt.txt")
        
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"❌ Error reading prompt file {prompt_file}: {e}")
                return None
        else:
            print(f"❌ Prompt file not found: {prompt_file}")
            return None

    def load_puzzle_metadata(self, puzzle_folder: str) -> Dict[str, Any]:
        """
        Load puzzle metadata from JSON file.
        
        Args:
            puzzle_folder: Path to puzzle folder
            
        Returns:
            Puzzle metadata dictionary
        """
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

    def run_inference_on_dataset(self, dataset_path: str = "data/3x3", 
                                output_path: str = "models/qwen/results", 
                                max_puzzles: Optional[int] = None,
                                start_puzzle: int = 1):
        """
        Run inference on all puzzles in the dataset.
        
        Args:
            dataset_path: Path to dataset folder
            output_path: Path to save results
            max_puzzles: Maximum number of puzzles to process (None for all)
            start_puzzle: Starting puzzle number
        """
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return

        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Find all puzzle folders
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
        
        # Sort by puzzle number
        puzzle_folders.sort(key=lambda x: x[0])
        
        # Limit number of puzzles if specified
        if max_puzzles:
            puzzle_folders = puzzle_folders[:max_puzzles]
        
        print(f"Found {len(puzzle_folders)} puzzles to process")
        print(f"Starting inference with Qwen model...")
        
        # Results storage
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process each puzzle
        for i, (puzzle_num, puzzle_folder) in enumerate(puzzle_folders):
            print(f"\n{'='*60}")
            print(f"Processing Puzzle {puzzle_num} ({i+1}/{len(puzzle_folders)})")
            print(f"Folder: {puzzle_folder}")
            
            start_time = time.time()
            
            # Load puzzle prompt
            puzzle_prompt = self.load_puzzle_prompt(puzzle_folder)
            if puzzle_prompt is None:
                print(f"❌ Skipping puzzle {puzzle_num} - no prompt found")
                continue
            
            # Load puzzle metadata
            metadata = self.load_puzzle_metadata(puzzle_folder)
            
            # Create chat prompt
            chat_prompt = self.create_chat_prompt(puzzle_prompt)
            
            # Generate response
            print("🤖 Generating solution...")
            response = self.generate_response(chat_prompt)
            
            # Parse solution
            solution_steps = self.parse_solution(response)
            
            processing_time = time.time() - start_time
            
            # Store results
            result = {
                'puzzle_num': puzzle_num,
                'puzzle_folder': os.path.basename(puzzle_folder),
                'difficulty': metadata.get('puzzle_info', {}).get('difficulty', 'unknown'),
                'num_blockers': metadata.get('puzzle_info', {}).get('num_blockers', 0),
                'optimal_solution_length': metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                'model_response': response,
                'parsed_solution_steps': solution_steps,
                'predicted_solution_length': len(solution_steps),
                'processing_time_seconds': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Print summary
            print(f"✅ Generated solution with {len(solution_steps)} steps")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"🎯 Optimal solution: {metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0)} steps")
            
            # Save individual result
            puzzle_result_file = os.path.join(output_path, f"puzzle{puzzle_num}_result.json")
            with open(puzzle_result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save progress periodically
            if (i + 1) % 10 == 0:
                self.save_results_summary(results, output_path, timestamp)
        
        # Save final results
        self.save_results_summary(results, output_path, timestamp)
        
        print(f"\n🎉 Inference complete!")
        print(f"Processed {len(results)} puzzles")
        print(f"Results saved to: {output_path}")

    def save_results_summary(self, results: List[Dict], output_path: str, timestamp: str):
        """Save results summary in multiple formats"""
        
        # Save comprehensive JSON results
        json_file = os.path.join(output_path, f"qwen_inference_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_file = os.path.join(output_path, f"qwen_inference_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = [
                    'puzzle_num', 'puzzle_folder', 'difficulty', 'num_blockers',
                    'optimal_solution_length', 'predicted_solution_length', 
                    'length_match', 'processing_time_seconds', 'timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow({
                        'puzzle_num': result['puzzle_num'],
                        'puzzle_folder': result['puzzle_folder'],
                        'difficulty': result['difficulty'],
                        'num_blockers': result['num_blockers'],
                        'optimal_solution_length': result['optimal_solution_length'],
                        'predicted_solution_length': result['predicted_solution_length'],
                        'length_match': result['optimal_solution_length'] == result['predicted_solution_length'],
                        'processing_time_seconds': result['processing_time_seconds'],
                        'timestamp': result['timestamp']
                    })
        
        print(f"📊 Results summary saved to: {csv_file}")

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze inference results and compute metrics"""
        if not results:
            return {}
        
        total_puzzles = len(results)
        optimal_length_matches = sum(1 for r in results 
                                   if r['optimal_solution_length'] == r['predicted_solution_length'])
        
        # Difficulty breakdown
        difficulty_stats = {}
        for result in results:
            diff = result['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'optimal_matches': 0}
            difficulty_stats[diff]['total'] += 1
            if result['optimal_solution_length'] == result['predicted_solution_length']:
                difficulty_stats[diff]['optimal_matches'] += 1
        
        # Processing time stats
        times = [r['processing_time_seconds'] for r in results]
        avg_time = sum(times) / len(times) if times else 0
        
        analysis = {
            'total_puzzles_processed': total_puzzles,
            'optimal_length_matches': optimal_length_matches,
            'optimal_length_accuracy': optimal_length_matches / total_puzzles if total_puzzles > 0 else 0,
            'difficulty_breakdown': difficulty_stats,
            'average_processing_time_seconds': round(avg_time, 2),
            'total_processing_time_seconds': round(sum(times), 2)
        }
        
        return analysis


def main():
    """Main execution function"""
    print("🚀 Starting Qwen 2.5-7B Rush Hour Inference")
    print("=" * 60)
    
    # Initialize inference system
    inference = QwenRushHourInference(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="auto"
    )
    
    # Load model
    try:
        inference.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Configuration
    dataset_path = "data/3x3"
    output_path = "models/qwen/results"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n📁 Dataset path: {dataset_path}")
    print(f"📁 Output path: {output_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run the puzzle generator first to create the dataset.")
        return
    
    # Run inference on all puzzles
    inference.run_inference_on_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        max_puzzles=None,  # Process all puzzles
        start_puzzle=1
    )
    
    print("✅ Inference pipeline completed!")


if __name__ == "__main__":
    main()