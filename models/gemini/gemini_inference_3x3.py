import os
import json
import csv
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv


class GeminiRushHourInference:
    def __init__(self, model_name="gemini-2.5-pro"):
        """
        Initialize Gemini model for Rush Hour puzzle inference.
        
        Args:
            model_name: Gemini model name
        """
        self.model_name = model_name
        self.client = None
        
        # Load environment variables
        load_dotenv()
        
        # System prompt for Rush Hour puzzles
        self.system_prompt = """You are an expert puzzle solver specializing in Rush Hour puzzles. Your task is to find the optimal sequence of moves to move the car 'C' to the TARGET position.

Key Instructions:
1. A 1-indexed coordinate system is being used
2. Each piece (car or blocker) can only move UP, DOWN, LEFT, or RIGHT by exactly one square
3. Pieces CANNOT move outside the grid or into occupied squares at any instant
4. Provide your solution in the exact format requested

Be precise with coordinates and piece movements. Think logically about the sequence of moves needed."""

    def load_model(self):
        """Load Gemini client"""
        print(f"Loading Gemini model: {self.model_name}")
        
        try:
            # Get API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Create client instance with API key
            self.client = genai.Client(api_key=api_key)
            
            print(f"✅ Gemini client loaded successfully: {self.model_name}")
            
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

Provide your solution as:
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

    def generate_response(self, prompt: str) -> str:
        """
        Generate response from Gemini model with dynamic thinking.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        if self.client is None:
            raise ValueError("Client not loaded. Call load_model() first.")
        
        try:
            # Create full prompt with system instruction
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            # Generate response with dynamic thinking enabled
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),  # Dynamic thinking
                    temperature=0.6,
                    max_output_tokens=32768,
                    top_p=0.95,
                    top_k=40
                )
            )
            
            return response.text.strip() if response.text else "No response generated"
            
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
        result_file = os.path.join(puzzle_result_folder, f"gemini_puzzle{puzzle_num}_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result_file

    def run_inference_on_dataset(self, dataset_path: str = "../../data/3x3", 
                                output_path: str = "gemini_results3x3", 
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
        print(f"Starting inference with Gemini model...")
        
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
                
                # Generate response
                print("🤖 Generating solution...")
                raw_response = self.generate_response(puzzle_prompt)
                
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
                
                # Rate limiting to avoid API limits
                time.sleep(1)
            
            except Exception as e:
                print(f"❌ Error processing puzzle {puzzle_num}: {e}")
                continue
        
        # Save final results summary
        self.save_results_summary(results, output_path, timestamp)
        
        print(f"\n🎉 Inference complete!")
        print(f"Processed {len(results)} puzzles")
        print(f"Results saved to: {output_path}")
        print(f"Individual results saved in: {output_path}/puzzle[N]/gemini_puzzle[N]_result.json")

    def save_results_summary(self, results: List[Dict], output_path: str, timestamp: str):
        """Save results summary in multiple formats"""
        
        # Save comprehensive JSON results
        json_file = os.path.join(output_path, f"gemini_inference_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_file = os.path.join(output_path, f"gemini_inference_summary_{timestamp}.csv")
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
    print("🚀 Starting Gemini 2.5 Pro Rush Hour Inference")
    print("=" * 60)
    
    # Initialize inference system
    inference = GeminiRushHourInference(
        model_name="gemini-2.5-pro"
    )
    
    # Load model
    try:
        inference.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Configuration
    dataset_path = "/home/mustafaah/rushhoureval/data/3x3"
    output_path = "results3x3"
    
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