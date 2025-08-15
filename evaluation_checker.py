#!/usr/bin/env python3
"""
Evaluation Checker for Rush Hour Puzzle Model Responses

This module evaluates model responses using the comprehensive checking logic
from the original checker.ipynb.
"""

import re
import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

GRID_SIZE = 3

class RushHourResponseChecker:
    """Comprehensive checker for evaluating model responses on Rush Hour puzzles"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
    
    def parse_solution_from_response(self, response: str) -> Optional[List[Dict]]:
        """Extract solution from model response with multiple parsing strategies"""
        
        # Strategy 1: Look for <solution> tags
        solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
        if solution_match:
            solution_text = solution_match.group(1).strip()
        else:
            # Strategy 2: Look for step patterns in the entire response
            solution_text = response
        
        # Extract step lines with flexible parsing
        step_patterns = [
            r'Step\s+\d+:\s*([A-Z]\d*)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]',
            r'(\d+)\.\s*([A-Z]\d*)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]',
            r'Move\s+\d+:\s*([A-Z]\d*)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]'
        ]
        
        moves = []
        for pattern in step_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if len(match) == 5:  # Pattern includes step number
                        _, piece, start_row, start_col, end_row, end_col = match
                    else:  # Pattern without step number
                        piece, start_row, start_col, end_row, end_col = match
                    
                    moves.append({
                        'piece': piece.upper(),
                        'start': (int(start_row), int(start_col)),
                        'end': (int(end_row), int(end_col))
                    })
                break  # Use first successful pattern
        
        return moves if moves else None
    
    def load_puzzle_metadata(self, puzzle_folder: Path) -> Dict:
        """Load puzzle metadata from solution.txt"""
        solution_file = puzzle_folder / "solution.txt"
        
        if not solution_file.exists():
            return {}
        
        metadata = {}
        with open(solution_file, 'r') as f:
            content = f.read()
        
        # Extract exit position
        exit_match = re.search(r'Exit position: \[(\d+),(\d+)\]', content)
        if exit_match:
            metadata['exit_pos'] = (int(exit_match.group(1)), int(exit_match.group(2)))
        
        # Extract transformation info
        transform_match = re.search(r'Transformation: (.+)', content)
        if transform_match:
            metadata['transformation'] = transform_match.group(1).strip()
        
        # Extract reference solution
        reference_moves = []
        in_solution = False
        for line in content.split('\n'):
            if line.strip() == "Solution:":
                in_solution = True
                continue
            if in_solution and line.strip().startswith("Step"):
                step_match = re.search(r'Step\s+\d+:\s*([A-Z]\d*)\s*\[(\d+),(\d+)\]\s*->\s*\[(\d+),(\d+)\]', line)
                if step_match:
                    piece, start_row, start_col, end_row, end_col = step_match.groups()
                    reference_moves.append({
                        'piece': piece.upper(),
                        'start': (int(start_row), int(start_col)),
                        'end': (int(end_row), int(end_col))
                    })
        
        metadata['reference_moves'] = reference_moves
        metadata['optimal_length'] = len(reference_moves)
        
        return metadata
    
    def reconstruct_initial_grid(self, puzzle_metadata: Dict) -> Optional[List[List[str]]]:
        """Reconstruct initial grid state from reference solution"""
        if not puzzle_metadata.get('reference_moves') or not puzzle_metadata.get('exit_pos'):
            return None
        
        # Start with final state (car at exit)
        grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        exit_pos = puzzle_metadata['exit_pos']
        exit_row, exit_col = exit_pos[0] - 1, exit_pos[1] - 1  # Convert to 0-indexed
        grid[exit_row][exit_col] = 'C'
        
        # Work backwards through moves
        moves = puzzle_metadata['reference_moves'][::-1]  # Reverse order
        pieces_seen = {'C'}
        
        for move in moves:
            piece = move['piece']
            pieces_seen.add(piece)
            
            # In reverse: move from end back to start
            end_pos = (move['end'][0] - 1, move['end'][1] - 1)  # Convert to 0-indexed
            start_pos = (move['start'][0] - 1, move['start'][1] - 1)
            
            # Clear end position and place at start position
            grid[end_pos[0]][end_pos[1]] = '.'
            grid[start_pos[0]][start_pos[1]] = piece
        
        return grid
    
    def simulate_moves(self, initial_grid: List[List[str]], moves: List[Dict], 
                      exit_pos: Tuple[int, int]) -> Tuple[bool, str, List[List[List[str]]]]:
        """Simulate moves and validate correctness"""
        
        if not initial_grid or not moves:
            return False, "No grid or moves provided", []
        
        grid = [row[:] for row in initial_grid]  # Deep copy
        states = [[row[:] for row in grid]]  # Track all states
        exit_row, exit_col = exit_pos[0] - 1, exit_pos[1] - 1  # Convert to 0-indexed
        
        for i, move in enumerate(moves):
            piece = move['piece']
            start_row, start_col = move['start'][0] - 1, move['start'][1] - 1  # Convert to 0-indexed
            end_row, end_col = move['end'][0] - 1, move['end'][1] - 1
            
            # Validate move bounds
            if not (0 <= start_row < GRID_SIZE and 0 <= start_col < GRID_SIZE):
                return False, f"Step {i+1}: Start position [{move['start'][0]},{move['start'][1]}] out of bounds", states
            
            if not (0 <= end_row < GRID_SIZE and 0 <= end_col < GRID_SIZE):
                return False, f"Step {i+1}: End position [{move['end'][0]},{move['end'][1]}] out of bounds", states
            
            # Validate piece at start position
            if grid[start_row][start_col] != piece:
                actual_piece = grid[start_row][start_col]
                return False, f"Step {i+1}: Expected {piece} at [{move['start'][0]},{move['start'][1]}], found '{actual_piece}'", states
            
            # Validate end position is empty
            if grid[end_row][end_col] != '.':
                occupying_piece = grid[end_row][end_col]
                return False, f"Step {i+1}: Position [{move['end'][0]},{move['end'][1]}] occupied by '{occupying_piece}'", states
            
            # Validate move is adjacent (Manhattan distance = 1)
            if abs(start_row - end_row) + abs(start_col - end_col) != 1:
                return False, f"Step {i+1}: Move is not adjacent (must be exactly 1 square)", states
            
            # Apply move
            grid[start_row][start_col] = '.'
            grid[end_row][end_col] = piece
            states.append([row[:] for row in grid])
        
        # Check if solved
        if grid[exit_row][exit_col] == 'C':
            return True, "Solution correct!", states
        else:
            # Find current car position
            car_pos = None
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid[r][c] == 'C':
                        car_pos = (r+1, c+1)  # Convert to 1-indexed
                        break
            return False, f"Car at [{car_pos[0]},{car_pos[1]}], target at [{exit_pos[0]},{exit_pos[1]}]", states
    
    def calculate_progress_metrics(self, initial_grid: List[List[str]], moves: List[Dict], 
                                 exit_pos: Tuple[int, int]) -> Dict:
        """Calculate comprehensive progress metrics"""
        
        if not initial_grid:
            return {"progress_score": 0, "error": "No initial grid"}
        
        # Find initial car position
        initial_car_pos = None
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if initial_grid[r][c] == 'C':
                    initial_car_pos = (r, c)
                    break
        
        if not initial_car_pos:
            return {"progress_score": 0, "error": "No car found in grid"}
        
        exit_row, exit_col = exit_pos[0] - 1, exit_pos[1] - 1  # Convert to 0-indexed
        initial_distance = abs(initial_car_pos[0] - exit_row) + abs(initial_car_pos[1] - exit_col)
        
        if not moves:
            return {
                "progress_score": 0.0,
                "initial_distance": initial_distance,
                "final_distance": initial_distance,
                "valid_moves": 0,
                "total_moves": 0,
                "error": "No moves provided"
            }
        
        # Simulate valid moves
        grid = [row[:] for row in initial_grid]
        valid_moves = 0
        
        for i, move in enumerate(moves):
            piece = move['piece']
            start_row, start_col = move['start'][0] - 1, move['start'][1] - 1
            end_row, end_col = move['end'][0] - 1, move['end'][1] - 1
            
            # Check validity
            valid = (
                0 <= start_row < GRID_SIZE and 0 <= start_col < GRID_SIZE and
                0 <= end_row < GRID_SIZE and 0 <= end_col < GRID_SIZE and
                grid[start_row][start_col] == piece and
                grid[end_row][end_col] == '.' and
                abs(start_row - end_row) + abs(start_col - end_col) == 1
            )
            
            if valid:
                grid[start_row][start_col] = '.'
                grid[end_row][end_col] = piece
                valid_moves += 1
            else:
                break  # Stop at first invalid move
        
        # Find final car position
        final_car_pos = initial_car_pos
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c] == 'C':
                    final_car_pos = (r, c)
                    break
        
        final_distance = abs(final_car_pos[0] - exit_row) + abs(final_car_pos[1] - exit_col)
        
        # Calculate progress score
        if final_distance == 0:
            progress_score = 1.0
        elif initial_distance == 0:
            progress_score = 1.0
        else:
            distance_improvement = initial_distance - final_distance
            base_score = max(0, distance_improvement / initial_distance)
            
            # Bonus for valid moves
            move_bonus = 0.1 * (valid_moves / len(moves)) if len(moves) > 0 else 0
            progress_score = min(1.0, base_score + move_bonus)
        
        return {
            "progress_score": progress_score,
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "valid_moves": valid_moves,
            "total_moves": len(moves),
            "move_validity_rate": valid_moves / len(moves) if len(moves) > 0 else 0
        }
    
    def evaluate_single_response(self, puzzle_folder: Path, response: str, 
                                model_name: str = "unknown") -> Dict:
        """Comprehensive evaluation of a single model response"""
        
        puzzle_name = puzzle_folder.name
        
        # Load puzzle metadata
        metadata = self.load_puzzle_metadata(puzzle_folder)
        if not metadata:
            return {
                "puzzle": puzzle_name,
                "model": model_name,
                "status": "ERROR",
                "error": "Could not load puzzle metadata"
            }
        
        # Parse response
        moves = self.parse_solution_from_response(response)
        
        result = {
            "puzzle": puzzle_name,
            "model": model_name,
            "response_length": len(response),
            "moves_found": len(moves) if moves else 0,
            "optimal_length": metadata.get('optimal_length', 0),
            "exit_position": metadata.get('exit_pos', (0, 0)),
            "transformation": metadata.get('transformation', 'unknown')
        }
        
        if not moves:
            result.update({
                "status": "PARSE_ERROR", 
                "correctness": "INVALID",
                "progress_score": 0.0,
                "error": "Could not parse solution from response"
            })
            return result
        
        # Reconstruct initial grid
        initial_grid = self.reconstruct_initial_grid(metadata)
        if not initial_grid:
            result.update({
                "status": "GRID_ERROR",
                "correctness": "UNKNOWN", 
                "error": "Could not reconstruct initial grid"
            })
            return result
        
        # Simulate moves
        is_correct, message, states = self.simulate_moves(
            initial_grid, moves, metadata['exit_pos']
        )
        
        # Calculate progress metrics
        progress_metrics = self.calculate_progress_metrics(
            initial_grid, moves, metadata['exit_pos']
        )
        
        # Determine correctness classification
        if is_correct:
            if len(moves) == metadata['optimal_length']:
                correctness = "OPTIMAL"
            else:
                correctness = "CORRECT_SUBOPTIMAL"
        else:
            if progress_metrics['valid_moves'] == 0:
                correctness = "INVALID"
            elif progress_metrics['progress_score'] > 0.7:
                correctness = "PARTIAL_GOOD"
            elif progress_metrics['progress_score'] > 0.3:
                correctness = "PARTIAL_MODERATE"
            else:
                correctness = "PARTIAL_POOR"
        
        # Calculate efficiency
        efficiency = len(moves) / metadata['optimal_length'] if metadata['optimal_length'] > 0 else float('inf')
        
        result.update({
            "status": "EVALUATED",
            "correctness": correctness,
            "is_solved": is_correct,
            "message": message,
            "efficiency": efficiency,
            **progress_metrics
        })
        
        return result
    
    def evaluate_model_results(self, results_file: Path, model_name: str = None) -> pd.DataFrame:
        """Evaluate all responses for a model"""
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        evaluated_results = []
        
        for result in results_data:
            puzzle_name = result.get('puzzle_name', 'unknown')
            response = result.get('response', '')
            model_name_result = result.get('model_name', model_name or 'unknown')
            
            puzzle_folder = self.data_path / puzzle_name
            
            if puzzle_folder.exists():
                evaluation = self.evaluate_single_response(
                    puzzle_folder, response, model_name_result
                )
                
                # Add timing information if available
                if 'inference_time' in result:
                    evaluation['inference_time'] = result['inference_time']
                
                evaluated_results.append(evaluation)
            else:
                evaluated_results.append({
                    "puzzle": puzzle_name,
                    "model": model_name_result,
                    "status": "ERROR",
                    "error": f"Puzzle folder {puzzle_name} not found"
                })
        
        return pd.DataFrame(evaluated_results)
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive evaluation report with detailed metrics"""
        
        report = {
            "summary": {},
            "correctness_analysis": {},
            "progress_analysis": {},
            "efficiency_analysis": {},
            "transformation_analysis": {},
            "model_comparison": {}
        }
        
        # Summary statistics
        total_puzzles = len(results_df)
        report["summary"] = {
            "total_puzzles": total_puzzles,
            "models_evaluated": results_df['model'].nunique(),
            "unique_models": results_df['model'].unique().tolist()
        }
        
        # Correctness analysis
        correctness_counts = results_df['correctness'].value_counts()
        report["correctness_analysis"] = {
            "distribution": correctness_counts.to_dict(),
            "optimal_rate": (correctness_counts.get('OPTIMAL', 0) / total_puzzles * 100),
            "correct_rate": ((correctness_counts.get('OPTIMAL', 0) + 
                            correctness_counts.get('CORRECT_SUBOPTIMAL', 0)) / total_puzzles * 100),
            "partial_rate": ((correctness_counts.get('PARTIAL_GOOD', 0) + 
                            correctness_counts.get('PARTIAL_MODERATE', 0) + 
                            correctness_counts.get('PARTIAL_POOR', 0)) / total_puzzles * 100)
        }
        
        # Progress analysis
        valid_progress = results_df[results_df['progress_score'].notna()]
        if len(valid_progress) > 0:
            report["progress_analysis"] = {
                "mean_progress": float(valid_progress['progress_score'].mean()),
                "median_progress": float(valid_progress['progress_score'].median()),
                "progress_std": float(valid_progress['progress_score'].std()),
                "progress_distribution": {
                    "high_progress_>0.7": len(valid_progress[valid_progress['progress_score'] > 0.7]),
                    "medium_progress_0.3-0.7": len(valid_progress[(valid_progress['progress_score'] >= 0.3) & 
                                                                (valid_progress['progress_score'] <= 0.7)]),
                    "low_progress_<0.3": len(valid_progress[valid_progress['progress_score'] < 0.3])
                }
            }
        
        # Efficiency analysis (for solved puzzles)
        solved_puzzles = results_df[results_df['is_solved'] == True]
        if len(solved_puzzles) > 0:
            report["efficiency_analysis"] = {
                "mean_efficiency": float(solved_puzzles['efficiency'].mean()),
                "median_efficiency": float(solved_puzzles['efficiency'].median()),
                "optimal_solutions": len(solved_puzzles[solved_puzzles['efficiency'] == 1.0]),
                "efficiency_distribution": {
                    "optimal_1.0": len(solved_puzzles[solved_puzzles['efficiency'] == 1.0]),
                    "good_1.0-1.5": len(solved_puzzles[(solved_puzzles['efficiency'] > 1.0) & 
                                                      (solved_puzzles['efficiency'] <= 1.5)]),
                    "acceptable_1.5-2.0": len(solved_puzzles[(solved_puzzles['efficiency'] > 1.5) & 
                                                            (solved_puzzles['efficiency'] <= 2.0)]),
                    "poor_>2.0": len(solved_puzzles[solved_puzzles['efficiency'] > 2.0])
                }
            }
        
        # Transformation analysis
        if 'transformation' in results_df.columns:
            transform_performance = results_df.groupby('transformation').agg({
                'is_solved': 'mean',
                'progress_score': 'mean',
                'correctness': lambda x: (x == 'OPTIMAL').mean()
            }).round(3)
            report["transformation_analysis"] = transform_performance.to_dict()
        
        # Model comparison
        if results_df['model'].nunique() > 1:
            model_performance = results_df.groupby('model').agg({
                'is_solved': 'mean',
                'progress_score': 'mean', 
                'efficiency': 'mean',
                'correctness': lambda x: (x == 'OPTIMAL').mean(),
                'inference_time': 'mean' if 'inference_time' in results_df.columns else lambda x: None
            }).round(3)
            report["model_comparison"] = model_performance.to_dict()
        
        return report
    
    def create_visualizations(self, results_df: pd.DataFrame, output_dir: Path):
        """Create comprehensive visualizations of results"""
        
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Correctness Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        correctness_order = ['OPTIMAL', 'CORRECT_SUBOPTIMAL', 'PARTIAL_GOOD', 
                           'PARTIAL_MODERATE', 'PARTIAL_POOR', 'INVALID']
        correctness_counts = results_df['correctness'].value_counts()
        
        bars = ax.bar(range(len(correctness_order)), 
                     [correctness_counts.get(level, 0) for level in correctness_order])
        ax.set_xticks(range(len(correctness_order)))
        ax.set_xticklabels(correctness_order, rotation=45, ha='right')
        ax.set_title('Solution Correctness Distribution')
        ax.set_ylabel('Number of Puzzles')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correctness_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Progress Score Distribution  
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_progress = results_df[results_df['progress_score'].notna()]
        ax.hist(valid_progress['progress_score'], bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Progress Score Distribution')
        ax.set_xlabel('Progress Score')
        ax.set_ylabel('Frequency')
        ax.axvline(valid_progress['progress_score'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {valid_progress["progress_score"].mean():.3f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'progress_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Comparison (if multiple models)
        if results_df['model'].nunique() > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Success rate by model
            model_success = results_df.groupby('model')['is_solved'].mean()
            axes[0, 0].bar(model_success.index, model_success.values)
            axes[0, 0].set_title('Success Rate by Model')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Progress score by model
            results_df.boxplot(column='progress_score', by='model', ax=axes[0, 1])
            axes[0, 1].set_title('Progress Score by Model')
            axes[0, 1].set_xlabel('Model')
            
            # Efficiency by model (solved puzzles only)
            solved_df = results_df[results_df['is_solved'] == True]
            if len(solved_df) > 0:
                solved_df.boxplot(column='efficiency', by='model', ax=axes[1, 0])
                axes[1, 0].set_title('Efficiency by Model (Solved Puzzles)')
                axes[1, 0].set_xlabel('Model')
                axes[1, 0].set_ylim(0, 3)  # Cap for readability
            
            # Correctness breakdown by model
            correctness_by_model = results_df.groupby(['model', 'correctness']).size().unstack(fill_value=0)
            correctness_by_model.plot(kind='bar', stacked=True, ax=axes[1, 1])
            axes[1, 1].set_title('Correctness Breakdown by Model')
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Transformation Performance (if available)
        if 'transformation' in results_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Success rate by transformation
            transform_success = results_df.groupby('transformation')['is_solved'].mean()
            axes[0].bar(transform_success.index, transform_success.values)
            axes[0].set_title('Success Rate by Transformation')
            axes[0].set_ylabel('Success Rate')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Progress by transformation
            results_df.boxplot(column='progress_score', by='transformation', ax=axes[1])
            axes[1].set_title('Progress Score by Transformation')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'transformation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")

def main():
    """Example usage of the checker"""
    checker = RushHourResponseChecker()
    
    # Example: Check results for a specific model
    results_file = Path("results/qwen_vl_05b/all_responses.json")
    
    if results_file.exists():
        # Evaluate results
        results_df = checker.evaluate_model_results(results_file, "qwen_vl_05b")
        
        # Generate report
        report = checker.generate_comprehensive_report(results_df)
        
        # Save report
        report_file = results_file.parent / "evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualizations
        viz_dir = results_file.parent / "visualizations"
        checker.create_visualizations(results_df, viz_dir)
        
        # Print summary
        print("Evaluation Summary:")
        print(f"Total puzzles: {report['summary']['total_puzzles']}")
        print(f"Optimal solutions: {report['correctness_analysis']['optimal_rate']:.1f}%")
        print(f"Correct solutions: {report['correctness_analysis']['correct_rate']:.1f}%")
        print(f"Average progress: {report['progress_analysis']['mean_progress']:.3f}")
    else:
        print(f"Results file not found: {results_file}")

if __name__ == "__main__":
    main()