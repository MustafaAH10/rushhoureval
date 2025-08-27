import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import copy
from collections import defaultdict


class RushHour3x3Validator:
    def __init__(self):
        self.max_row, self.max_col = 3, 3
    
    def parse_move(self, move_str: str) -> Optional[Tuple[str, List[int], List[int]]]:
        """Parse a move string for 3x3 puzzles (1x1 pieces only)."""
        pattern = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
        match = re.search(pattern, move_str, re.IGNORECASE)
        if match:
            piece, start_row, start_col, end_row, end_col = match.groups()
            start_pos = [int(start_row), int(start_col)]
            end_pos = [int(end_row), int(end_col)]
            return piece, start_pos, end_pos
        return None
    
    def is_valid_position(self, pos: List[int]) -> bool:
        """Check if position is within 3x3 grid bounds."""
        return 1 <= pos[0] <= self.max_row and 1 <= pos[1] <= self.max_col
    
    def is_adjacent_move(self, start_pos: List[int], end_pos: List[int]) -> bool:
        """Check if move is exactly one square in cardinal direction."""
        row_diff = abs(end_pos[0] - start_pos[0])
        col_diff = abs(end_pos[1] - start_pos[1])
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
    
    def apply_move(self, pieces_state: Dict[str, Any], piece_name: str, 
                   start_pos: List[int], end_pos: List[int]) -> Tuple[bool, str]:
        """Apply and validate a move for 3x3 puzzles."""
        if piece_name not in pieces_state:
            return False, f"Piece {piece_name} not found"
        
        piece_data = pieces_state[piece_name]
        current_pos = piece_data['position']
        
        if current_pos != start_pos:
            return False, f"Start position {start_pos} doesn't match current position {current_pos}"
        
        if not self.is_valid_position(end_pos):
            return False, f"End position {end_pos} is out of bounds"
        
        if not self.is_adjacent_move(start_pos, end_pos):
            return False, f"Move from {start_pos} to {end_pos} is not adjacent"
        
        # Check collisions with other pieces
        occupied_cells = set()
        for name, data in pieces_state.items():
            if name == piece_name:
                continue
            occupied_cells.add(tuple(data['position']))
        
        if tuple(end_pos) in occupied_cells:
            return False, f"End position {end_pos} is occupied"
        
        # Apply the move
        piece_data['position'] = end_pos
        return True, "Valid move"
    
    def validate_solution(self, initial_state: Dict[str, Any], 
                         moves: List[str], target_pos: List[int]) -> Dict[str, Any]:
        """Validate complete solution for 3x3 puzzles."""
        current_state = copy.deepcopy(initial_state['pieces'])
        
        validation_result = {
            'moves_are_legal': True,
            'reaches_target': False,
            'illegal_move_details': [],
            'total_moves': len(moves),
            'final_car_position': None
        }
        
        if len(moves) == 0:
            validation_result['moves_are_legal'] = False
            validation_result['illegal_move_details'].append({
                'move_number': 0,
                'error': 'Empty solution'
            })
            return validation_result
        
        for i, move_str in enumerate(moves):
            move_data = self.parse_move(move_str)
            if not move_data:
                validation_result['moves_are_legal'] = False
                validation_result['illegal_move_details'].append({
                    'move_number': i + 1,
                    'move_string': move_str,
                    'error': 'Failed to parse move'
                })
                break
            
            piece_name, start_pos, end_pos = move_data
            is_valid, error_msg = self.apply_move(current_state, piece_name, start_pos, end_pos)
            
            if not is_valid:
                validation_result['moves_are_legal'] = False
                validation_result['illegal_move_details'].append({
                    'move_number': i + 1,
                    'move_string': move_str,
                    'piece_name': piece_name,
                    'error': error_msg
                })
                break
        
        # Check if car reached target
        if validation_result['moves_are_legal'] and 'C' in current_state:
            final_car_pos = current_state['C']['position']
            validation_result['final_car_position'] = final_car_pos
            validation_result['reaches_target'] = (final_car_pos == target_pos)
        
        return validation_result


class ModelPerformanceAnalyzer3x3:
    def __init__(self, data_path: str, results_path: str):
        self.data_path = data_path
        self.results_path = results_path
        self.validator = RushHour3x3Validator()
    
    def load_puzzle_data(self, puzzle_num: int) -> Dict[str, Any]:
        puzzle_folder = os.path.join(self.data_path, f"puzzle{puzzle_num}")
        puzzle_state_file = os.path.join(puzzle_folder, "puzzle_state.json")
        with open(puzzle_state_file, 'r') as f:
            return json.load(f)
    
    def load_model_result(self, model_name: str, puzzle_num: int) -> Dict[str, Any]:
        result_folder = os.path.join(self.results_path, model_name, "results3x3", f"puzzle{puzzle_num}")
        result_files = [f for f in os.listdir(result_folder) if f.endswith('_result.json')]
        result_file = os.path.join(result_folder, result_files[0])
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def analyze_model_performance(self, model_name: str, max_puzzles: int = 150) -> Dict[str, Any]:
        results = {
            'model_name': model_name,
            'grid_size': '3x3',
            'total_puzzles': 0,
            'illegal_moves': 0,
            'legal_no_target': 0,
            'legal_suboptimal_target': 0,
            'legal_optimal_target': 0,
            'parsing_failures': 0,
            'detailed_results': []
        }
        
        for puzzle_num in range(1, max_puzzles + 1):
            try:
                puzzle_data = self.load_puzzle_data(puzzle_num)
                model_result = self.load_model_result(model_name, puzzle_num)
                results['total_puzzles'] += 1
                
                parsed_moves = model_result.get('parsed_solution', [])
                target_pos = puzzle_data['exit_position']
                optimal_length = puzzle_data['puzzle_info']['total_moves_in_solution']
                
                if not parsed_moves:
                    results['parsing_failures'] += 1
                    results['illegal_moves'] += 1
                    continue
                
                validation = self.validator.validate_solution(puzzle_data, parsed_moves, target_pos)
                moves_are_legal = validation['moves_are_legal']
                reaches_target = validation['reaches_target']
                move_count = len(parsed_moves)
                is_optimal = reaches_target and (move_count == optimal_length)
                
                # Categorize
                if not moves_are_legal:
                    results['illegal_moves'] += 1
                elif moves_are_legal and not reaches_target:
                    results['legal_no_target'] += 1
                elif moves_are_legal and reaches_target and not is_optimal:
                    results['legal_suboptimal_target'] += 1
                else:
                    results['legal_optimal_target'] += 1
                
                if puzzle_num % 10 == 0:
                    print(f"Processed {puzzle_num}/{max_puzzles} puzzles for {model_name}")
                    
            except Exception as e:
                print(f"Error processing puzzle {puzzle_num}: {e}")
                continue
        
        # Calculate percentages
        total = results['total_puzzles']
        if total > 0:
            results['illegal_percentage'] = (results['illegal_moves'] / total) * 100
            results['legal_no_target_percentage'] = (results['legal_no_target'] / total) * 100
            results['legal_suboptimal_percentage'] = (results['legal_suboptimal_target'] / total) * 100
            results['legal_optimal_percentage'] = (results['legal_optimal_target'] / total) * 100
            results['legal_accomplishes_target_percentage'] = ((results['legal_suboptimal_target'] + results['legal_optimal_target']) / total) * 100
            results['legal_optimal_target_percentage'] = (results['legal_optimal_target'] / total) * 100
        
        return results
    
    def analyze_all_models(self, model_names: List[str], max_puzzles: int = 150) -> Dict[str, Any]:
        all_results = {'grid_size': '3x3', 'models': {}}
        
        for model_name in model_names:
            print(f"\n=== Analyzing {model_name} (3x3) ===")
            try:
                model_results = self.analyze_model_performance(model_name, max_puzzles)
                all_results['models'][model_name] = model_results
                
                print(f"Legal & Accomplishes Target: {model_results.get('legal_accomplishes_target_percentage', 0):.1f}%")
                print(f"Legal & Optimal: {model_results.get('legal_optimal_target_percentage', 0):.1f}%")
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

def main():
    DATA_PATH = "/home/mustafaah/rushhoureval/data/3x3"
    RESULTS_PATH = "/home/mustafaah/Desktop/FINAL_DATA"
    MODEL_NAMES = [
        "gpt5",
        "gemini", 
        "qwen3-8b",
        "qwen14b",
        "qwen7b",
        "olmo2-13b",
        "llama3.1-8b",
        "deepseekv3"
    ]  # Update as needed
    MAX_PUZZLES = 150
    
    analyzer = ModelPerformanceAnalyzer3x3(DATA_PATH, RESULTS_PATH)
    all_results = analyzer.analyze_all_models(MODEL_NAMES, MAX_PUZZLES)
    analyzer.save_results(all_results, "3x3_validation_results.json")
    
    print("\n" + "="*60)
    print("3x3 BENCHMARK SUMMARY")
    print("="*60)
    for model_name, model_results in all_results['models'].items():
        print(f"\n{model_name}:")
        print(f"  Legal & Optimal: {model_results.get('legal_optimal_target', 0)} ({model_results.get('legal_optimal_target_percentage', 0):.1f}%)")
        print(f"  Legal & Accomplishes Target: {model_results.get('legal_suboptimal_target', 0) + model_results.get('legal_optimal_target', 0)} ({model_results.get('legal_accomplishes_target_percentage', 0):.1f}%)")


if __name__ == "__main__":
    main()