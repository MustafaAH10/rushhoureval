import os
import json
import csv
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime


class DeepSeekReasoner4x4Inference:
    def __init__(self, api_key: str = None, model_name: str = "deepseek-reasoner"):
        """
        Initialize DeepSeek Reasoner model for 4x4 Rush Hour puzzle inference.
        
        Args:
            api_key: DeepSeek API key (if None, will use DEEPSEEK_API_KEY env var)
            model_name: DeepSeek model to use ("deepseek-reasoner" for thinking mode, "deepseek-chat" for non-thinking)
        """
        self.model_name = model_name
        
        # Initialize DeepSeek client (OpenAI-compatible)
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            # Will use DEEPSEEK_API_KEY environment variable
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("Please set DEEPSEEK_API_KEY environment variable or provide api_key")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # System prompt for 4x4 Rush Hour puzzles
        self.system_prompt = """You are an expert puzzle solver specializing in Rush Hour puzzles. Your task is to find the optimal sequence of moves to move the car 'C' to the TARGET position.

Key Instructions:
1. A 1-indexed coordinate system is being used
2. Each piece (car or blocker) can only move UP, DOWN, LEFT, or RIGHT by exactly one square
3. For 1x1 pieces (car C and blockers B1, B2, etc.): Use single coordinate format
4. For 2x1 pieces (blockers H1, H2, etc.): Use double coordinate format with both occupied positions
5. Pieces CANNOT move outside the 4x4 grid or into occupied squares at any instant
6. Think through the problem step by step to find the optimal solution
7. Provide your solution in the exact format requested

Be precise with coordinates and piece movements. Analyze the puzzle logically to find the shortest path."""

    def create_puzzle_prompt_from_json(self, puzzle_metadata: Dict[str, Any]) -> str:
        """
        Create puzzle prompt using grid from puzzle_state.json for 4x4 puzzles
        
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
        
        # Find blockers and categorize them
        blockers_1x1 = []
        blockers_2x1 = []
        
        for piece_name, piece_data in pieces.items():
            if piece_name != 'C':
                piece_type = piece_data.get('type', '')
                if piece_type == '1x1_blocker':
                    pos = piece_data.get('position', [])
                    if pos:
                        blockers_1x1.append(f"{piece_name} at [{pos[0]},{pos[1]}]")
                elif '2x1' in piece_type:
                    positions = piece_data.get('positions', [])
                    if positions:
                        positions_str = ", ".join([f"[{p[0]},{p[1]}]" for p in positions])
                        orientation = "horizontal" if "horizontal" in piece_type else "vertical"
                        blockers_2x1.append(f"{piece_name} ({orientation}) at {positions_str}")
        
        prompt = f"""Task: Solve this 4x4 Rush Hour puzzle - move car "C" from position [{car_position[0]},{car_position[1]}] to the TARGET at position [{exit_position[0]},{exit_position[1]}] given the position of the blockers below.

Current Grid State (JSON format):
{grid_json}

Current Pieces:
- Car "C": Position [{car_position[0]},{car_position[1]}]
- 1x1 Blockers (B1, B2, etc.): Single-cell obstacles that can be moved to clear a path
  {chr(10).join(f"  - {blocker}" for blocker in blockers_1x1) if blockers_1x1 else "  - None present"}
- 2x1 Blockers (H1, H2, etc.): Two-cell obstacles that move as a single unit
  {chr(10).join(f"  - {blocker}" for blocker in blockers_2x1) if blockers_2x1 else "  - None present"}
- TARGET: Position [{exit_position[0]},{exit_position[1]}]

Movement Rules:
- Any piece (car "C", 1x1 blockers "B1, B2, etc.", or 2x1 blockers "H1, H2, etc.") can move UP, DOWN, LEFT, or RIGHT
- Each move is exactly ONE square in any direction for the entire piece
- For 2x1 blockers: The entire piece moves together as a unit (both cells move simultaneously)
- Pieces strictly CANNOT move outside the 4x4 grid
- Pieces strictly CANNOT move into occupied squares (i.e. squares that already have another piece)
- At ANY instant, there CANNOT be two pieces occupying the same square
- The same piece can move multiple times in a row if needed
- You win when car "C" reaches the TARGET cell

Coordinate System:
- Use [row,col] format where [1,1] is top-left, [4,4] is bottom-right
- Each cell shows its coordinates in black text: (row,col)
- For 2x1 blockers, both occupied cells are shown in the piece description

Expected Output Format:
Wrap your solution in <solution> tags and provide it as a numbered list of moves in this exact format:

<solution>
Step 1: [PIECE] [start_position] -> [end_position]
Step 2: [PIECE] [start_position] -> [end_position]
...
</solution>

For 1x1 pieces (car "C" and blockers "B1", "B2", etc.):
- Use single coordinate: C [2,1] -> [2,2]

For 2x1 pieces (blockers "H1", "H2", etc.):
- List both coordinates: H1 [[1,1],[1,2]] -> [[2,1],[2,2]]

Example response format:
<solution>
Step 1: B2 [3,2] -> [4,2]
Step 2: H1 [(2,3), (3,3)] -> [(1,3), (2,3)]
Step 3: B2 [2,4] -> [1,4]
Step 4: C [3,1] -> [3,2]
Step 5: C [3,2] -> [3,3]
Step 6: C [3,3] -> [3,4]
Step 7: C [3,4] -> [2,4]
</solution>"""
        return prompt

    def generate_response(self, prompt: str, max_tokens: int = 32000) -> tuple[str, str, Dict]:
        """
        Generate response from DeepSeek Reasoner model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (including reasoning)
            
        Returns:
            Tuple of (content, reasoning_content, usage_info)
        """
        try:
            # Create messages for DeepSeek API
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            print(f"🧠 Generating solution with {self.model_name}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Extract response components
            content = response.choices[0].message.content or ""
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', "") or ""
            
            # Extract usage info
            usage_info = {}
            if hasattr(response, 'usage'):
                usage_info = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                    'model': self.model_name
                }
            
            # Print token usage
            if usage_info:
                print(f"📊 Token usage: {usage_info.get('prompt_tokens', 0)} prompt + {usage_info.get('completion_tokens', 0)} completion = {usage_info.get('total_tokens', 0)} total")
            
            return content.strip(), reasoning_content.strip(), usage_info
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"ERROR: {str(e)}", "", {}

    def parse_solution(self, response: str) -> List[str]:
        """
        Parse solution steps from model response for 4x4 puzzles.
        Handles both 1x1 and 2x1 piece formats.
        
        Args:
            response: Raw model response (content)
            
        Returns:
            List of solution steps
        """
        solution_steps = []
        
        # Look for <solution> tags
        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
        
        if solution_match:
            solution_text = solution_match.group(1).strip()
            
            # Pattern for 1x1 pieces: Step N: PIECE [r,c] -> [r,c]
            pattern_1x1 = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
            
            # Pattern for 2x1 pieces: Step N: PIECE [[r,c],[r,c]] -> [[r,c],[r,c]]
            pattern_2x1 = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*->\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]'
            
            # Find all step lines
            step_lines = re.findall(r'Step\s+\d+:.*', solution_text, re.IGNORECASE)
            
            for i, line in enumerate(step_lines):
                # Try 2x1 pattern first (more specific)
                match_2x1 = re.search(pattern_2x1, line, re.IGNORECASE)
                if match_2x1:
                    piece, sr1, sc1, sr2, sc2, er1, ec1, er2, ec2 = match_2x1.groups()
                    step_text = f"Step {i+1}: {piece} [[{sr1},{sc1}],[{sr2},{sc2}]] -> [[{er1},{ec1}],[{er2},{ec2}]]"
                    solution_steps.append(step_text)
                    continue
                
                # Try 1x1 pattern
                match_1x1 = re.search(pattern_1x1, line, re.IGNORECASE)
                if match_1x1:
                    piece, start_row, start_col, end_row, end_col = match_1x1.groups()
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
                          prompt: str, content: str, reasoning_content: str, 
                          parsed_solution: List[str], usage_info: Dict, 
                          metadata: Dict[str, Any], processing_time: float):
        """Save individual puzzle result with reasoning traces"""
        puzzle_result_folder = os.path.join(output_path, f"puzzle{puzzle_num}")
        os.makedirs(puzzle_result_folder, exist_ok=True)
        
        result = {
            'puzzle_info': {
                'puzzle_num': puzzle_num,
                'difficulty': metadata.get('puzzle_info', {}).get('difficulty', 'unknown'),
                'num_1x1_blockers': metadata.get('puzzle_info', {}).get('num_1x1_blockers', 0),
                'num_2x1_blockers': metadata.get('puzzle_info', {}).get('num_2x1_blockers', 0),
                'optimal_solution_length': metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                'processing_time_seconds': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            },
            'prompt': prompt,
            'deepseek_content': content,
            'deepseek_reasoning_content': reasoning_content,
            'parsed_solution': parsed_solution,
            'api_usage': usage_info,
            'analysis': {
                'predicted_solution_length': len(parsed_solution),
                'length_matches_optimal': len(parsed_solution) == metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                'solution_found': len(parsed_solution) > 0,
                'parsing_successful': len(parsed_solution) > 0,
                'has_reasoning': len(reasoning_content) > 0,
                'reasoning_length_chars': len(reasoning_content),
                'content_length_chars': len(content)
            }
        }
        
        result_file = os.path.join(puzzle_result_folder, f"deepseek_4x4_puzzle{puzzle_num}_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result_file

    def run_inference_on_dataset(self, dataset_path: str = "/root/rushhoureval/data/4x4", 
                                output_path: str = "deepseek_results4x4", 
                                max_puzzles: Optional[int] = None,
                                start_puzzle: int = 1,
                                delay_between_requests: float = 1.0):
        """Run inference on all 4x4 puzzles in the dataset using DeepSeek Reasoner."""
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Find all puzzle folders
        puzzle_folders = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and item.startswith("puzzle"):
                try:
                    # Extract base puzzle number (handle both "puzzle123" and "puzzle123_transformation" formats)
                    if "_" in item:
                        base_num = int(item.split("_")[0].replace("puzzle", ""))
                    else:
                        base_num = int(item.replace("puzzle", ""))
                    
                    if base_num >= start_puzzle:
                        puzzle_folders.append((base_num, item, item_path))
                except ValueError:
                    continue
        
        # Sort by puzzle number, then by folder name for consistent ordering
        puzzle_folders.sort(key=lambda x: (x[0], x[1]))
        
        # Limit number of puzzles if specified
        if max_puzzles:
            puzzle_folders = puzzle_folders[:max_puzzles]
        
        print(f"Found {len(puzzle_folders)} puzzles to process")
        print(f"Starting 4x4 inference with {self.model_name}...")
        print(f"Rate limit delay: {delay_between_requests}s between requests")
        
        # Results storage
        results = []
        total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process each puzzle
        for i, (puzzle_num, folder_name, puzzle_folder) in enumerate(puzzle_folders):
            print(f"\n{'='*60}")
            print(f"Processing Puzzle {folder_name} ({i+1}/{len(puzzle_folders)})")
            print(f"Folder: {puzzle_folder}")
            
            start_time = time.time()
            
            try:
                metadata = self.load_puzzle_metadata(puzzle_folder)
                if not metadata:
                    print(f"❌ Skipping puzzle {folder_name} - no metadata found")
                    continue
                
                puzzle_prompt = self.create_puzzle_prompt_from_json(metadata)
                
                print(f"🤖 Generating solution with {self.model_name}...")
                content, reasoning_content, usage_info = self.generate_response(puzzle_prompt)
                
                parsed_solution = self.parse_solution(content)
                processing_time = time.time() - start_time
                
                # Update total usage
                for key in total_usage:
                    total_usage[key] += usage_info.get(key, 0)
                
                result_file = self.save_puzzle_result(
                    output_path=output_path,
                    puzzle_num=i+1,  # Use sequential numbering for results
                    prompt=puzzle_prompt,
                    content=content,
                    reasoning_content=reasoning_content,
                    parsed_solution=parsed_solution,
                    usage_info=usage_info,
                    metadata=metadata,
                    processing_time=processing_time
                )
                
                # Store for summary
                result_summary = {
                    'puzzle_num': i+1,
                    'puzzle_folder': folder_name,
                    'difficulty': metadata.get('puzzle_info', {}).get('difficulty', 'unknown'),
                    'num_1x1_blockers': metadata.get('puzzle_info', {}).get('num_1x1_blockers', 0),
                    'num_2x1_blockers': metadata.get('puzzle_info', {}).get('num_2x1_blockers', 0),
                    'optimal_solution_length': metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                    'predicted_solution_length': len(parsed_solution),
                    'length_matches_optimal': len(parsed_solution) == metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0),
                    'solution_found': len(parsed_solution) > 0,
                    'processing_time_seconds': round(processing_time, 2),
                    'prompt_tokens': usage_info.get('prompt_tokens', 0),
                    'completion_tokens': usage_info.get('completion_tokens', 0),
                    'total_tokens': usage_info.get('total_tokens', 0),
                    'has_reasoning': len(reasoning_content) > 0,
                    'reasoning_length_chars': len(reasoning_content),
                    'content_length_chars': len(content),
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result_summary)
                
                # Print summary
                print(f"✅ Generated solution with {len(parsed_solution)} steps")
                print(f"🧠 Reasoning: {len(reasoning_content)} chars")
                print(f"📝 Content: {len(content)} chars")
                print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"🎯 Optimal solution: {metadata.get('puzzle_info', {}).get('total_moves_in_solution', 0)} steps")
                print(f"📁 Result saved to: {result_file}")
                
                if (i + 1) % 10 == 0:
                    self.save_results_summary(results, output_path, timestamp, total_usage)
                
                # Rate limiting delay
                if i < len(puzzle_folders) - 1:  # Don't delay after last request
                    time.sleep(delay_between_requests)
            
            except Exception as e:
                print(f"❌ Error processing puzzle {folder_name}: {e}")
                continue
        
        self.save_results_summary(results, output_path, timestamp, total_usage)
        
        print(f"\n🎉 4x4 Inference complete!")
        print(f"Processed {len(results)} puzzles")
        print(f"Total API usage: {total_usage}")
        print(f"Results saved to: {output_path}")

    def save_results_summary(self, results: List[Dict], output_path: str, timestamp: str, total_usage: Dict):
        """Save results summary in multiple formats"""
        json_file = os.path.join(output_path, f"deepseek_4x4_inference_results_{timestamp}.json")
        summary_data = {
            'model_info': {
                'model_name': self.model_name,
                'total_puzzles_processed': len(results),
                'timestamp': timestamp
            },
            'total_api_usage': total_usage,
            'results': results
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        csv_file = os.path.join(output_path, f"deepseek_4x4_inference_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = [
                    'puzzle_num', 'puzzle_folder', 'difficulty', 
                    'num_1x1_blockers', 'num_2x1_blockers',
                    'optimal_solution_length', 'predicted_solution_length', 
                    'length_matches_optimal', 'solution_found', 'processing_time_seconds',
                    'prompt_tokens', 'completion_tokens', 'total_tokens',
                    'has_reasoning', 'reasoning_length_chars', 'content_length_chars', 'timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
        
        print(f"📊 Results summary saved to: {csv_file}")

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze inference results and compute metrics for 4x4 puzzles with reasoning analysis"""
        if not results:
            return {}
        
        total_puzzles = len(results)
        optimal_length_matches = sum(1 for r in results if r.get('length_matches_optimal', False))
        solutions_found = sum(1 for r in results if r.get('solution_found', False))
        reasoning_available = sum(1 for r in results if r.get('has_reasoning', False))
        
        # Difficulty breakdown
        difficulty_stats = {}
        for result in results:
            diff = result['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'optimal_matches': 0, 'solutions_found': 0, 'with_reasoning': 0}
            difficulty_stats[diff]['total'] += 1
            if result.get('length_matches_optimal', False):
                difficulty_stats[diff]['optimal_matches'] += 1
            if result.get('solution_found', False):
                difficulty_stats[diff]['solutions_found'] += 1
            if result.get('has_reasoning', False):
                difficulty_stats[diff]['with_reasoning'] += 1
        
        # Blocker complexity analysis
        blocker_stats = {}
        for result in results:
            num_1x1 = result.get('num_1x1_blockers', 0)
            num_2x1 = result.get('num_2x1_blockers', 0)
            complexity_key = f"{num_1x1}x1+{num_2x1}x2"
            
            if complexity_key not in blocker_stats:
                blocker_stats[complexity_key] = {'total': 0, 'optimal_matches': 0, 'solutions_found': 0, 'with_reasoning': 0}
            blocker_stats[complexity_key]['total'] += 1
            if result.get('length_matches_optimal', False):
                blocker_stats[complexity_key]['optimal_matches'] += 1
            if result.get('solution_found', False):
                blocker_stats[complexity_key]['solutions_found'] += 1
            if result.get('has_reasoning', False):
                blocker_stats[complexity_key]['with_reasoning'] += 1
        
        # Processing time and reasoning stats
        times = [r['processing_time_seconds'] for r in results]
        completion_tokens = [r.get('completion_tokens', 0) for r in results]
        total_tokens = [r.get('total_tokens', 0) for r in results]
        reasoning_chars = [r.get('reasoning_length_chars', 0) for r in results]
        
        avg_time = sum(times) / len(times) if times else 0
        avg_completion_tokens = sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0
        avg_total_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0
        avg_reasoning_chars = sum(reasoning_chars) / len(reasoning_chars) if reasoning_chars else 0
        
        analysis = {
            'total_puzzles_processed': total_puzzles,
            'solutions_found': solutions_found,
            'solution_rate': solutions_found / total_puzzles if total_puzzles > 0 else 0,
            'optimal_length_matches': optimal_length_matches,
            'optimal_length_accuracy': optimal_length_matches / total_puzzles if total_puzzles > 0 else 0,
            'reasoning_available': reasoning_available,
            'reasoning_rate': reasoning_available / total_puzzles if total_puzzles > 0 else 0,
            'difficulty_breakdown': difficulty_stats,
            'blocker_complexity_breakdown': blocker_stats,
            'average_processing_time_seconds': round(avg_time, 2),
            'total_processing_time_seconds': round(sum(times), 2),
            'average_completion_tokens': round(avg_completion_tokens, 0),
            'total_completion_tokens': sum(completion_tokens),
            'average_total_tokens': round(avg_total_tokens, 0),
            'total_tokens_used': sum(total_tokens),
            'average_reasoning_length_chars': round(avg_reasoning_chars, 0),
            'max_reasoning_length_chars': max(reasoning_chars) if reasoning_chars else 0
        }
        
        return analysis


def main():
    """Main execution function"""
    print("🚀 Starting DeepSeek Reasoner 4x4 Rush Hour Inference")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ Please set your DEEPSEEK_API_KEY environment variable")
        print("   export DEEPSEEK_API_KEY='your-api-key-here'")
        return
    
    # Initialize inference system
    inference = DeepSeekReasoner4x4Inference(
        model_name="deepseek-reasoner",  # Use deepseek-reasoner for thinking mode
        api_key=api_key
    )
    
    # Configuration
    dataset_path = "/home/mustafaah/rushhoureval/data/4x4"
    output_path = "results4x4"
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n📁 Dataset path: {dataset_path}")
    print(f"📁 Output path: {output_path}")
    print(f"🤖 Model: {inference.model_name}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run the 4x4 puzzle generator first to create the dataset.")
        return
    
    # Run inference on all 4x4 puzzles
    inference.run_inference_on_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        max_puzzles=None,  # Process all puzzles
        start_puzzle=1,
        delay_between_requests=1.0  # 1 second delay for API rate limits
    )
    
    print("✅ Inference pipeline completed!")


if __name__ == "__main__":
    main()