# QwenVL Model Evaluation

This directory contains the QwenVL model evaluation implementation for Rush Hour puzzles.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run evaluation:
```bash
python qwen_evaluator.py
```

## Features

- **Model Loading**: Automatically loads Qwen/Qwen-VL from HuggingFace
- **Batch Evaluation**: Processes multiple puzzles efficiently  
- **Response Analysis**: Parses and validates model solutions
- **Progress Tracking**: Saves intermediate results for each puzzle
- **Comprehensive Reporting**: Generates detailed evaluation metrics

## Output Structure

```
results/qwen_vl/
├── model_config.json          # Model configuration
├── sample_response.json       # Example response format
├── puzzle1_response.json      # Individual puzzle results
├── puzzle2_response.json      # ...
├── all_responses.json         # Complete results
├── detailed_evaluation.csv    # Analysis results
└── evaluation_report.json     # Summary report
```

## Configuration

Key parameters in `qwen_evaluator.py`:
- `model_id`: HuggingFace model identifier
- `max_tokens`: Maximum response length
- `temperature`: Sampling temperature
- `device`: GPU/CPU selection (automatic)

## Usage Examples

### Test Run (3 puzzles)
```python
from qwen_evaluator import QwenVLEvaluator

evaluator = QwenVLEvaluator()
results = evaluator.evaluate_puzzles(puzzle_subset=["puzzle1", "puzzle2", "puzzle3"])
```

### Full Evaluation
```python
evaluator = QwenVLEvaluator()
results = evaluator.evaluate_puzzles()  # All puzzles
```

## Performance Notes

- GPU recommended for faster inference
- Memory usage scales with model size and batch processing
- Typical inference time: 1-5 seconds per puzzle
- Progress saved incrementally to handle interruptions