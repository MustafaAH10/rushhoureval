import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import copy
from collections import defaultdict


class RushHourValidator:
    def __init__(self, grid_size: str = "3x3"):
        """
        Initialize the Rush Hour validator.
        
        Args:
            grid_size: "3x3", "4x4", or "5x5"
        """
        self.grid_size = grid_size
        self.max_row, self.max_col = self._parse_grid_size(grid_size)
    
    def _parse_grid_size(self, grid_size: str) -> Tuple[int, int]:
        """Parse grid size string to dimensions."""
        if grid_size == "3x3":
            return 3, 3
        elif grid_size == "4x4":
            return 4, 4
        elif grid_size == "5x5":
            return 5, 5
        else:
            raise ValueError(f"Unsupported grid size: {grid_size}")
    
    def parse_move(self, move_str: str) -> Optional[Tuple[str, List[int], List[int]]]:
        """
        Parse a move string into piece name, start position, and end position.
        
        Args:
            move_str: Move string like "Step 1: B1 [2,3] -> [3,3]"
            
        Returns:
            Tuple of (piece_name, start_pos, end_pos) or None if parsing fails
        """
        # Pattern for 1x1 pieces: Step N: PIECE [r,c] -> [r,c]
        pattern_1x1 = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
        
        # Pattern for 2x1 pieces: Step N: PIECE [[r,c],[r,c]] -> [[r,c],[r,c]]
        pattern_2x1 = r'Step\s+\d+:\s*([A-Z0-9]+)\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*->\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]'
        
        # Try 2x1 pattern first (more specific)
        match_2x1 = re.search(pattern_2x1, move_str, re.IGNORECASE)
        if match_2x1:
            piece, sr1, sc1, sr2, sc2, er1, ec1, er2, ec2 = match_2x1.groups()
            start_pos = [[int(sr1), int(sc1)], [int(sr2), int(sc2)]]
            end_pos = [[int(er1), int(ec1)], [int(er2), int(ec2)]]
            return piece, start_pos, end_pos
        
        # Try 1x1 pattern
        match_1x1 = re.search(pattern_1x1, move_str, re.IGNORECASE)
        if match_1x1:
            piece, start_row, start_col, end_row, end_col = match_1x1.groups()
            start_pos = [int(start_row), int(start_col)]
            end_pos = [int(end_row), int(end_col)]
            return piece, start_pos, end_pos
        
        return None
    
    def is_valid_position(self, pos: List[int]) -> bool:
        """Check if a position is within grid bounds."""
        if isinstance(pos[0], list):  # 2x1 piece
            for p in pos:
                if p[0] < 1 or p[0] > self.max_row or p[1] < 1 or p[1] > self.max_col:
                    return False
        else:  # 1x1 piece
            if pos[0] < 1 or pos[0] > self.max_row or pos[1] < 1 or pos[1] > self.max_col:
                return False
        return True
    
    def is_adjacent_move(self, start_pos: List[int], end_pos: List[int]) -> bool:
        """Check if the move is exactly one square in a cardinal direction."""
        if isinstance(start_pos[0], list):  # 2x1 piece
            # For 2x1 pieces, both cells must move in the same direction
            if len(start_pos) != 2 or len(end_pos) != 2:
                return False
            
            # Calculate movement vectors for both cells
            move1 = [end_pos[0][0] - start_pos[0][0], end_pos[0][1] - start_pos[0][1]]
            move2 = [end_pos[1][0] - start_pos[1][0], end_pos[1][1] - start_pos[1][1]]
            
            # Both cells must move in the same direction
            if move1 != move2:
                return False
            
            # Must be exactly one square in a cardinal direction
            return abs(move1[0]) + abs(move1[1]) == 1
        else:  # 1x1 piece
            # Must be exactly one square in a cardinal direction
            row_diff = abs(end_pos[0] - start_pos[0])
            col_diff = abs(end_pos[1] - start_pos[1])
            return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
    
    def get_occupied_cells(self, pieces_state: Dict[str, Any]) -> set:
        """Get all currently occupied cells."""
        occupied = set()
        for piece_name, piece_data in pieces_state.items():
            if 'position' in piece_data:
                pos = piece_data['position']
                if isinstance(pos[0], list):  # 2x1 piece
                    for p in pos:
                        occupied.add(tuple(p))
                else:  # 1x1 piece
                    occupied.add(tuple(pos))
            elif 'positions' in piece_data:  # Alternative format for 2x1 pieces
                for pos in piece_data['positions']:
                    occupied.add(tuple(pos))
        return occupied
    
    def apply_move(self, pieces_state: Dict[str, Any], piece_name: str, 
                   start_pos: List[int], end_pos: List[int]) -> Tuple[bool, str]:
        """
        Apply a move to the game state and validate it.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if piece_name not in pieces_state:
            return False, f"Piece {piece_name} not found"
        
        piece_data = pieces_state[piece_name]
        
        # Get current position
        if 'position' in piece_data:
            current_pos = piece_data['position']
        elif 'positions' in piece_data:
            current_pos = piece_data['positions']
        else:
            return False, f"No position data for piece {piece_name}"
        
        # Verify start position matches current position
        if isinstance(current_pos[0], list):  # 2x1 piece
            # Sort both positions for comparison
            current_sorted = sorted([tuple(p) for p in current_pos])
            start_sorted = sorted([tuple(p) for p in start_pos])
            if current_sorted != start_sorted:
                return False, f"Start position {start_pos} doesn't match current position {current_pos}"
        else:  # 1x1 piece
            if current_pos != start_pos:
                return False, f"Start position {start_pos} doesn't match current position {current_pos}"
        
        # Check if end position is within bounds
        if not self.is_valid_position(end_pos):
            return False, f"End position {end_pos} is out of bounds"
        
        # Check if move is adjacent (exactly one square)
        if not self.is_adjacent_move(start_pos, end_pos):
            return False, f"Move from {start_pos} to {end_pos} is not adjacent or not cardinal"
        
        # Get all occupied cells except for the moving piece
        occupied_cells = set()
        for name, data in pieces_state.items():
            if name == piece_name:
                continue  # Skip the moving piece
            
            if 'position' in data:
                pos = data['position']
                if isinstance(pos[0], list):  # 2x1 piece
                    for p in pos:
                        occupied_cells.add(tuple(p))
                else:  # 1x1 piece
                    occupied_cells.add(tuple(pos))
            elif 'positions' in data:
                for pos in data['positions']:
                    occupied_cells.add(tuple(pos))
        
        # Check if end position conflicts with other pieces
        if isinstance(end_pos[0], list):  # 2x1 piece
            for pos in end_pos:
                if tuple(pos) in occupied_cells:
                    return False, f"End position {pos} is occupied by another piece"
        else:  # 1x1 piece
            if tuple(end_pos) in occupied_cells:
                return False, f"End position {end_pos} is occupied by another piece"
        
        # Apply the move
        if 'position' in piece_data:
            piece_data['position'] = end_pos
        elif 'positions' in piece_data:
            piece_data['positions'] = end_pos
        
        return True, "Valid move"
    
    def validate_solution(self, initial_state: Dict[str, Any], 
                         moves: List[str], target_pos: List[int]) -> Dict[str, Any]:
        """
        Validate a complete solution.
        
        CATEGORIES:
        1. Illegal moves (invalid moves, parsing failures)
        2. Legal moves that don't reach target
        3. Legal moves that reach target but are suboptimal
        4. Legal moves that reach target and are optimal
        
        Returns:
            Dictionary with validation results
        """
        # Create a copy of the initial state
        current_state = copy.deepcopy(initial_state['pieces'])
        
        validation_result = {
            'moves_are_legal': True,  # All moves are valid
            'reaches_target': False,  # Car reaches target position
            'illegal_move_details': [],
            'total_moves': len(moves),
            'final_car_position': None,
            'move_by_move_states': []
        }
        
        # If no moves provided, mark moves as illegal
        if len(moves) == 0:
            validation_result['moves_are_legal'] = False
            validation_result['illegal_move_details'].append({
                'move_number': 0,
                'move_string': 'No moves provided',
                'error': 'Empty solution'
            })
            return validation_result
        
        # Process each move sequentially
        for i, move_str in enumerate(moves):
            move_data = self.parse_move(move_str)
            if not move_data:
                validation_result['moves_are_legal'] = False
                validation_result['illegal_move_details'].append({
                    'move_number': i + 1,
                    'move_string': move_str,
                    'error': 'Failed to parse move'
                })
                break  # Stop on first parsing failure
            
            piece_name, start_pos, end_pos = move_data
            
            # Apply and validate the move
            is_valid, error_msg = self.apply_move(current_state, piece_name, start_pos, end_pos)
            
            if not is_valid:
                validation_result['moves_are_legal'] = False
                validation_result['illegal_move_details'].append({
                    'move_number': i + 1,
                    'move_string': move_str,
                    'piece_name': piece_name,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'error': error_msg
                })
                break  # Stop processing on first illegal move
            
            # Store state after this move for debugging
            validation_result['move_by_move_states'].append(copy.deepcopy(current_state))
        
        # Check if car C reached the target (only check if all moves were legal)
        if validation_result['moves_are_legal'] and 'C' in current_state:
            final_car_pos = current_state['C'].get('position', current_state['C'].get('positions'))
            validation_result['final_car_position'] = final_car_pos
            
            # Check if final position exactly matches target
            if isinstance(final_car_pos, list) and len(final_car_pos) == 2:
                validation_result['reaches_target'] = (final_car_pos == target_pos)
        
        return validation_result


class ModelPerformanceAnalyzer:
    def __init__(self, data_path: str, results_path: str, grid_size: str = "3x3"):
        """
        Initialize the model performance analyzer.
        
        Args:
            data_path: Path to the puzzle data directory
            results_path: Path to the model results directory
            grid_size: Grid size ("3x3", "4x4", "5x5")
        """
        self.data_path = data_path
        self.results_path = results_path
        self.grid_size = grid_size
        self.validator = RushHourValidator(grid_size)
    
    def load_puzzle_data(self, puzzle_num: int) -> Dict[str, Any]:
        """Load puzzle data from the dataset."""
        puzzle_folder = os.path.join(self.data_path, f"puzzle{puzzle_num}")
        puzzle_state_file = os.path.join(puzzle_folder, "puzzle_state.json")
        
        if not os.path.exists(puzzle_state_file):
            raise FileNotFoundError(f"Puzzle state file not found: {puzzle_state_file}")
        
        with open(puzzle_state_file, 'r') as f:
            return json.load(f)
    
    def load_model_result(self, model_name: str, puzzle_num: int) -> Dict[str, Any]:
        """Load model result for a specific puzzle."""
        result_folder = os.path.join(self.results_path, model_name, f"results{self.grid_size}", f"puzzle{puzzle_num}")
        
        # Find the result JSON file
        result_files = [f for f in os.listdir(result_folder) if f.endswith('_result.json')]
        if not result_files:
            raise FileNotFoundError(f"No result file found in {result_folder}")
        
        result_file = os.path.join(result_folder, result_files[0])
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def analyze_model_performance(self, model_name: str, max_puzzles: int = 150) -> Dict[str, Any]:
        """
        Analyze performance of a single model across all puzzles.
        
        FOUR DISTINCT CATEGORIES:
        1. Illegal moves (parsing failures, invalid moves)
        2. Legal moves that don't reach target
        3. Legal moves that reach target but are suboptimal
        4. Legal moves that reach target and are optimal
        """
        results = {
            'model_name': model_name,
            'grid_size': self.grid_size,
            'total_puzzles': 0,
            'illegal_moves': 0,  # Category 1: Invalid moves/parsing failures
            'legal_no_target': 0,  # Category 2: Legal moves, no target reached
            'legal_suboptimal_target': 0,  # Category 3: Legal, reaches target, suboptimal
            'legal_optimal_target': 0,  # Category 4: Legal, reaches target, optimal
            'parsing_failures': 0,
            'detailed_results': []
        }
        
        for puzzle_num in range(1, max_puzzles + 1):
            try:
                # Load puzzle data and model result
                puzzle_data = self.load_puzzle_data(puzzle_num)
                model_result = self.load_model_result(model_name, puzzle_num)
                
                results['total_puzzles'] += 1
                
                # Get the parsed solution
                parsed_moves = model_result.get('parsed_solution', [])
                target_pos = puzzle_data['exit_position']
                optimal_length = puzzle_data['puzzle_info']['total_moves_in_solution']
                
                # Handle parsing failures
                if not parsed_moves:
                    results['parsing_failures'] += 1
                    results['illegal_moves'] += 1  # Parsing failure = illegal
                    results['detailed_results'].append({
                        'puzzle_num': puzzle_num,
                        'category': 'parsing_failure',
                        'moves_are_legal': False,
                        'reaches_target': False,
                        'is_optimal': False,
                        'move_count': 0,
                        'optimal_length': optimal_length
                    })
                    continue
                
                # Validate the solution
                validation = self.validator.validate_solution(puzzle_data, parsed_moves, target_pos)
                
                moves_are_legal = validation['moves_are_legal']
                reaches_target = validation['reaches_target']
                move_count = len(parsed_moves)
                is_optimal = reaches_target and (move_count == optimal_length)
                
                # Categorize the solution
                if not moves_are_legal:
                    # Category 1: Illegal moves
                    results['illegal_moves'] += 1
                    category = 'illegal_moves'
                elif moves_are_legal and not reaches_target:
                    # Category 2: Legal moves but doesn't reach target
                    results['legal_no_target'] += 1
                    category = 'legal_no_target'
                elif moves_are_legal and reaches_target and not is_optimal:
                    # Category 3: Legal, reaches target, but suboptimal
                    results['legal_suboptimal_target'] += 1
                    category = 'legal_suboptimal_target'
                else:
                    # Category 4: Legal, reaches target, and optimal
                    results['legal_optimal_target'] += 1
                    category = 'legal_optimal_target'
                
                # Store detailed result
                puzzle_result = {
                    'puzzle_num': puzzle_num,
                    'category': category,
                    'moves_are_legal': moves_are_legal,
                    'reaches_target': reaches_target,
                    'is_optimal': is_optimal,
                    'move_count': move_count,
                    'optimal_length': optimal_length,
                    'illegal_move_details': validation.get('illegal_move_details', []),
                    'final_car_position': validation.get('final_car_position'),
                    'target_position': target_pos
                }
                results['detailed_results'].append(puzzle_result)
                
                # Print progress
                if puzzle_num % 10 == 0:
                    print(f"Processed {puzzle_num}/{max_puzzles} puzzles for {model_name}")
            
            except Exception as e:
                print(f"Error processing puzzle {puzzle_num} for {model_name}: {e}")
                continue
        
        # Calculate percentages
        total = results['total_puzzles']
        if total > 0:
            # Individual category percentages
            results['illegal_percentage'] = (results['illegal_moves'] / total) * 100
            results['legal_no_target_percentage'] = (results['legal_no_target'] / total) * 100
            results['legal_suboptimal_percentage'] = (results['legal_suboptimal_target'] / total) * 100
            results['legal_optimal_percentage'] = (results['legal_optimal_target'] / total) * 100
            
            # Combined percentages for charts
            results['legal_accomplishes_target_percentage'] = ((results['legal_suboptimal_target'] + results['legal_optimal_target']) / total) * 100
            results['legal_optimal_target_percentage'] = (results['legal_optimal_target'] / total) * 100
            
            results['parsing_failure_percentage'] = (results['parsing_failures'] / total) * 100
        
        return results
    
    def analyze_all_models(self, model_names: List[str], max_puzzles: int = 150) -> Dict[str, Any]:
        """Analyze performance of all models."""
        all_results = {
            'grid_size': self.grid_size,
            'total_puzzles_analyzed': max_puzzles,
            'models': {}
        }
        
        for model_name in model_names:
            print(f"\n=== Analyzing {model_name} ===")
            try:
                model_results = self.analyze_model_performance(model_name, max_puzzles)
                all_results['models'][model_name] = model_results
                
                # Print summary
                print(f"Legal solutions: {model_results['legal_solutions']}/{model_results['total_puzzles']} "
                      f"({model_results.get('legal_percentage', 0):.1f}%)")
                print(f"Optimal solutions: {model_results['legal_optimal_solutions']}/{model_results['total_puzzles']} "
                      f"({model_results.get('optimal_percentage', 0):.1f}%)")
            
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")
                continue
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run the analysis."""
    
    # Configuration - CHANGE THESE PATHS AS NEEDED
    DATA_BASE_PATH = "/home/mustafaah/rushhoureval/data"  # Base path to your data folder
    RESULTS_BASE_PATH = "/home/mustafaah/Desktop/FINAL_DATA/"  # Base path to your FINAL_DATA folder
    GRID_SIZE = "4x4"  # Change to "4x4" for 4x4 analysis
    MAX_PUZZLES = 150
    
    # List of model names - UPDATE THIS LIST BASED ON YOUR MODEL FOLDERS
    MODEL_NAMES = [
        "gpt5",
        "gemini", 
        "qwen3-8b",
        "qwen14b",
        "qwen7b",
        "olmo2-13b",
        "llama3.1-8b",
        "deepseekv3"
    ]
    
    # Construct full paths
    data_path = os.path.join(DATA_BASE_PATH, GRID_SIZE)
    results_path = RESULTS_BASE_PATH
    
    print(f"Rush Hour Solution Validator and Analyzer")
    print(f"Grid size: {GRID_SIZE}")
    print(f"Data path: {data_path}")
    print(f"Results path: {results_path}")
    print(f"Models to analyze: {MODEL_NAMES}")
    print(f"Max puzzles per model: {MAX_PUZZLES}")
    
    # Initialize analyzer
    analyzer = ModelPerformanceAnalyzer(data_path, results_path, GRID_SIZE)
    
    # Run analysis
    all_results = analyzer.analyze_all_models(MODEL_NAMES, MAX_PUZZLES)
    
    # Save results
    output_file = f"rush_hour_analysis_{GRID_SIZE}_results.json"
    analyzer.save_results(all_results, output_file)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY FOR {GRID_SIZE} PUZZLES")
    print(f"{'='*60}")
    
    for model_name, model_results in all_results['models'].items():
        print(f"\n{model_name}:")
        print(f"  Total puzzles: {model_results['total_puzzles']}")
        print(f"  Legal & Accomplishes Target: {model_results.get('legal_suboptimal_target', 0) + model_results.get('legal_optimal_target', 0)} ({model_results.get('legal_accomplishes_target_percentage', 0):.1f}%)")
        print(f"  Legal & Optimal: {model_results.get('legal_optimal_target', 0)} ({model_results.get('legal_optimal_target_percentage', 0):.1f}%)")
        print(f"  Legal & Suboptimal: {model_results.get('legal_suboptimal_target', 0)} ({model_results.get('legal_suboptimal_percentage', 0):.1f}%)")
        print(f"  Legal but No Target: {model_results.get('legal_no_target', 0)} ({model_results.get('legal_no_target_percentage', 0):.1f}%)")
        print(f"  Illegal Moves: {model_results.get('illegal_moves', 0)} ({model_results.get('illegal_percentage', 0):.1f}%)")
        print(f"  Parsing failures: {model_results.get('parsing_failures', 0)} ({model_results.get('parsing_failure_percentage', 0):.1f}%)")


if __name__ == "__main__":
    main()












