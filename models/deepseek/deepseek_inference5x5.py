import os
import json
import csv
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime


class DeepSeekReasoner5x5Inference:
    def __init__(self, api_key: str = None, model_name: str = "deepseek-reasoner"):
        """
        Initialize DeepSeek Reasoner model for 5x5 Rush Hour puzzle inference.
        
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

        # System prompt for 5x5 Rush Hour puzzles
        self.system_prompt = """You are an expert puzzle solver specializing in Rush Hour puzzles. Your task is to find the optimal sequence of moves to move the horizontal 2x1 car 'C' so that it exactly covers the 2-cell TARGET zone.

Key Instructions:
1. A 1-indexed coordinate system is being used where [1,1] is top-left, [5,5] is bottom-right
2. The car 'C' is always a horizontal 2x1 piece that must exactly cover both cells of the TARGET zone
3. Each piece (car or blocker) can only move UP, DOWN, LEFT, or RIGHT by exactly one square
4. For all pieces (car C, 2x1 blockers, 3x1 blockers), the entire piece moves as a unit
5. Pieces CANNOT move outside the 5x5 grid or into occupied squares at any instant
6. Provide your solution in the exact format requested with all coordinates for multi-cell pieces

Be precise with coordinates and piece movements. Think logically about the sequence of moves needed."""

    def create_puzzle_prompt_from_json(self, puzzle_metadata: Dict[str, Any]) -> str:
        """
        Create puzzle prompt using grid from puzzle_state.json for 5x5 puzzles
        
        Args:
            puzzle_metadata: Loaded puzzle_state.json data
            
        Returns:
            Puzzle prompt string with JSON grid
        """
        grid = puzzle_metadata.get('grid', [])
        car_positions = puzzle_metadata.get('car_positions', [])
        target_zone = puzzle_metadata.get('target_zone', [])
        pieces = puzzle_metadata.get('pieces', {})
        
        if not grid or not car_positions or not target_zone:
            raise ValueError("Missing required puzzle data in JSON")
        
        # Convert grid to JSON string
        grid_json = json.dumps(grid, separators=(',', ':'))
        
        # Find blockers and categorize them
        blockers_2x1 = []
        blockers_3x1 = []
        
        for piece_name, piece_data in pieces.items():
            if piece_name != 'C':
                piece_type = piece_data.get('type', '')
                positions = piece_data.get('positions', [])
                
                if '2x1' in piece_type and positions:
                    positions_str = ", ".join([f"[{p[0]},{p[1]}]" for p in positions])
                    orientation = "horizontal" if "horizontal" in piece_type else "vertical"
                    blockers_2x1.append(f"{piece_name} (2x1 {orientation}) at {positions_str}")
                elif '3x1' in piece_type and positions:
                    positions_str = ", ".join([f"[{p[0]},{p[1]}]" for p in positions])
                    orientation = "horizontal" if "horizontal" in piece_type else "vertical"
                    blockers_3x1.append(f"{piece_name} (3x1 {orientation}) at {positions_str}")
        
        # Format car positions and target zone
        car_positions_str = ", ".join([f"[{p[0]},{p[1]}]" for p in car_positions])
        target_zone_str = f"[{target_zone[0][0]},{target_zone[0][1]}] and [{target_zone[1][0]},{target_zone[1][1]}]"
        
        prompt = f"""Task: Solve this 5x5 Rush Hour puzzle - move the horizontal 2x1 car "C" so that it exactly covers the 2-cell TARGET zone at positions {target_zone_str}.

Current Grid State (JSON format):
{grid_json}

Current Pieces:
- Car "C" (horizontal 2x1): Currently at positions {car_positions_str}
- 2x1 Blockers: Two-cell obstacles that can be horizontal or vertical
  {chr(10).join(f"  - {blocker}" for blocker in blockers_2x1) if blockers_2x1 else "  - None present"}
- 3x1 Blockers: Three-cell obstacles that can be horizontal or vertical
  {chr(10).join(f"  - {blocker}" for blocker in blockers_3x1) if blockers_3x1 else "  - None present"}
- TARGET Zone: Positions {target_zone_str} (horizontal 2-cell zone)

Movement Rules:
- Any piece can move UP, DOWN, LEFT, or RIGHT by exactly ONE square
- For multi-cell pieces, the entire piece moves together as a single unit
- All cells of a piece move simultaneously in the same direction
- Pieces strictly CANNOT move outside the 5x5 grid boundaries
- Pieces strictly CANNOT move into occupied squares (collision detection)
- At ANY instant, there CANNOT be two pieces occupying the same square
- The same piece can move multiple times consecutively if needed
- Victory condition: Car "C" must exactly cover BOTH cells of the TARGET zone

Coordinate System:
- Use [row,col] format where [1,1] is top-left corner, [5,5] is bottom-right corner
- Each cell in the grid shows its coordinates as (row,col)
- For multi-cell pieces, list ALL occupied cell coordinates

Expected Output Format:
Wrap your solution in <solution> tags and provide it as a numbered sequence:

<solution>
Step 1: [PIECE] [start_positions] -> [end_positions]
Step 2: [PIECE] [start_positions] -> [end_positions]
...
</solution>

For ALL pieces, always list complete coordinate sets e.g.:
- Car "C" (2x1): C [[2,1],[2,2]] -> [[2,2],[2,3]]
- 2x1 blockers: B1 [[1,1],[1,2]] -> [[1,2],[1,3]]
- 3x1 blockers: L1 [[3,1],[3,2],[3,3]] -> [[2,1],[2,2],[2,3]]

Example response format:
<solution>
Step 1: B2 [[2,2],[2,3]] -> [[2,1],[2,2]]
Step 2: B3 [[3,2],[3,3]] -> [[3,1],[3,2]]
Step 3: L1 [[3,4],[4,4],[5,4]] -> [[3,3],[4,3],[5,3]]
Step 4: L1 [[3,3],[4,3],[5,3]] -> [[2,3],[3,3],[4,3]]
Step 5: L1 [[2,3],[3,3],[4,3]] -> [[1,3],[2,3],[3,3]]
Step 6: C [[4,1],[4,2]] -> [[4,2],[4,3]]
Step 7: C [[4,2],[4,3]] -> [[4,3],[4,4]]
Step 8: C [[4,3],[4,4]] -> [[4,4],[4,5]]
Step 9: C [[4,4],[4,5]] -> [[3,4],[3,5]]
</solution>

Remember: The goal is to position car "C" so it exactly covers both target cells {target_zone_str}."""
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
        Parse solution steps from model response for 5x5 puzzles.
        Handles 2x1 car, 2x1 blockers, and 3x1 blockers.
        
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
            
            # Pattern for 2x1 pieces: Step N: PIECE [[r,c],[r,c]] -> [[r,c],[r,c]]
            pattern_2x1 = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*->\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]'
            
            # Pattern for 3x1 pieces: Step N: PIECE [[r,c],[r,c],[r,c]] -> [[r,c],[r,c],[r,c]]
            pattern_3x1 = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*->\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]'
            
            # Find all step lines
            step_lines = re.findall(r'Step\s+\d+:.*', solution_text, re.IGNORECASE)
            
            for i, line in enumerate(step_lines):
                # Try 3x1 pattern first (most specific)
                match_3x1 = re.search(pattern_3x1, line, re.IGNORECASE)
                if match_3x1:
                    piece, sr1, sc1, sr2, sc2, sr3, sc3, er1, ec1, er2, ec2, er3, ec3 = match_3x1.groups()
                    step_text = f"Step {i+1}: {piece} [[{sr1},{sc1}],[{sr2},{sc2}],[{sr3},{sc3}]] -> [[{er1},{ec1}],[{er2},{ec2}],[{er3},{ec3}]]"
                    solution_steps.append(step_text)
                    continue
                
                # Try 2x1 pattern
                match_2x1 = re.search(pattern_2x1, line, re.IGNORECASE)
                if match_2x1:
                    piece, sr1, sc1, sr2, sc2, er1, ec1, er2, ec2 = match_2x1.groups()
                    step_text = f"Step {i+1}: {piece} [[{sr1},{sc1}],[{sr2},{sc2}]] -> [[{er1},{ec1}],[{er2},{ec2}]]"
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
                'num_2x1_blockers': metadata.get('puzzle_info', {}).get('num_2x1_blockers', 0),
                'num_3x1_blockers': metadata.get('puzzle_info', {}).get('num_3x1_blockers', 0),
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
        
        result_file = os.path.join(puzzle_result_folder, f"deepseek_5x5_puzzle{puzzle_num}_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result_file

    def run_inference_on_dataset(self, dataset_path: str = "/root/rushhoureval/data/5x5", 
                                output_path: str = "deepseek_results5x5", 
                                max_puzzles: Optional[int] = None,
                                start_puzzle: int = 1,
                                delay_between_requests: float = 1.0):
        """Run inference on all 5x5 puzzles in the dataset using DeepSeek Reasoner."""
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
                    # Extract puzzle number
                    base_num = int(item.replace("puzzle", ""))
                    
                    if base_num >= start_puzzle:
                        puzzle_folders.append((base_num, item, item_path))
                except ValueError:
                    continue
        
        # Sort by puzzle number
        puzzle_folders.sort(key=lambda x: x[0])
        
        # Limit number of puzzles if specified
        if max_puzzles:
            puzzle_folders = puzzle_folders[:max_puzzles]
        
        print(f"Found {len(puzzle_folders)} puzzles to process")
        print(f"Starting 5x5 inference with {self.model_name}...")
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
                    'num_2x1_blockers': metadata.get('puzzle_info', {}).get('num_2x1_blockers', 0),
                    'num_3x1_blockers': metadata.get('puzzle_info', {}).get('num_3x1_blockers', 0),
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
        
        print(f"\n🎉 5x5 Inference complete!")
        print(f"Processed {len(results)} puzzles")
        print(f"Total API usage: {total_usage}")
        print(f"Results saved to: {output_path}")

    def save_results_summary(self, results: List[Dict], output_path: str, timestamp: str, total_usage: Dict):
        """Save results summary in multiple formats"""
        json_file = os.path.join(output_path, f"deepseek_5x5_inference_results_{timestamp}.json")
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
        
        csv_file = os.path.join(output_path, f"deepseek_5x5_inference_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = [
                    'puzzle_num', 'puzzle_folder', 'difficulty', 
                    'num_2x1_blockers', 'num_3x1_blockers',
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
        """Analyze inference results and compute metrics for 5x5 puzzles with reasoning analysis"""
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
            num_2x1 = result.get('num_2x1_blockers', 0)
            num_3x1 = result.get('num_3x1_blockers', 0)
            complexity_key = f"{num_2x1}x2+{num_3x1}x3"
            
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
    print("🚀 Starting DeepSeek Reasoner 5x5 Rush Hour Inference")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ Please set your DEEPSEEK_API_KEY environment variable")
        print("   export DEEPSEEK_API_KEY='your-api-key-here'")
        return
    
    # Initialize inference system
    inference = DeepSeekReasoner5x5Inference(
        model_name="deepseek-reasoner",  # Use deepseek-reasoner for thinking mode
        api_key=api_key
    )
    
    # Configuration
    dataset_path = "/home/mustafaah/rushhoureval/data/5x5"
    output_path = "results5x5"
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n📁 Dataset path: {dataset_path}")
    print(f"📁 Output path: {output_path}")
    print(f"🤖 Model: {inference.model_name}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run the 5x5 puzzle generator first to create the dataset.")
        return
    
    # Run inference on all 5x5 puzzles
    inference.run_inference_on_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        max_puzzles=None,  # Process all puzzles
        start_puzzle=1,
        delay_between_requests=1.0  # 1 second delay for API rate limits
    )
    
    print("✅ 5x5 Inference pipeline completed!")


if __name__ == "__main__":
    main()