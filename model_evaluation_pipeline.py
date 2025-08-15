#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Pipeline for Rush Hour Puzzles

This pipeline evaluates multimodal language models on rush hour puzzle solving tasks.
It supports both HuggingFace models and API-based models (Claude, GPT-4).
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import requests
from dataclasses import dataclass
import concurrent.futures
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for different model types"""
    name: str
    model_type: str  # 'huggingface' or 'api'
    model_id: str = None  # For HF models
    api_endpoint: str = None  # For API models
    api_key: str = None
    max_tokens: int = 1000
    temperature: float = 0.1

class ModelEvaluationPipeline:
    """Main pipeline for evaluating models on rush hour puzzles"""
    
    def __init__(self, data_path: str = "data", results_path: str = "results"):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Thread lock for concurrent execution
        self.lock = Lock()
        
        # Load puzzle list
        self.puzzle_folders = self._get_puzzle_folders()
        logger.info(f"Found {len(self.puzzle_folders)} puzzles")
    
    def _get_puzzle_folders(self) -> List[Path]:
        """Get all puzzle folders sorted by name"""
        folders = []
        for folder in self.data_path.iterdir():
            if folder.is_dir() and folder.name.startswith('puzzle'):
                folders.append(folder)
        return sorted(folders)
    
    def load_puzzle_data(self, puzzle_folder: Path) -> Dict:
        """Load puzzle data including prompt and image"""
        puzzle_data = {}
        
        # Load prompt
        prompt_file = puzzle_folder / "prompt.txt"
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                puzzle_data['prompt'] = f.read().strip()
        
        # Load image
        image_file = puzzle_folder / "initial_state.png"
        if image_file.exists():
            puzzle_data['image'] = Image.open(image_file)
        
        # Load reference solution for evaluation
        solution_file = puzzle_folder / "solution.txt"
        if solution_file.exists():
            with open(solution_file, 'r') as f:
                puzzle_data['solution'] = f.read().strip()
        
        puzzle_data['name'] = puzzle_folder.name
        return puzzle_data
    
    def setup_huggingface_model(self, model_config: ModelConfig):
        """Initialize HuggingFace model"""
        logger.info(f"Loading HuggingFace model: {model_config.model_id}")
        
        try:
            processor = AutoProcessor.from_pretrained(model_config.model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_config.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            return {'processor': processor, 'model': model}
        except Exception as e:
            logger.error(f"Failed to load model {model_config.model_id}: {e}")
            return None
    
    def query_huggingface_model(self, model_components: Dict, prompt: str, image: Image.Image, 
                               config: ModelConfig) -> str:
        """Query HuggingFace model with image and text"""
        try:
            processor = model_components['processor']
            model = model_components['model']
            
            # Process inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    do_sample=True if config.temperature > 0 else False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying HuggingFace model: {e}")
            return f"ERROR: {str(e)}"
    
    def query_api_model(self, prompt: str, image: Image.Image, config: ModelConfig) -> str:
        """Query API-based model (Claude, GPT-4)"""
        try:
            if "claude" in config.name.lower():
                return self._query_claude_api(prompt, image, config)
            elif "gpt" in config.name.lower():
                return self._query_openai_api(prompt, image, config)
            else:
                return "ERROR: Unsupported API model"
                
        except Exception as e:
            logger.error(f"Error querying API model: {e}")
            return f"ERROR: {str(e)}"
    
    def _query_claude_api(self, prompt: str, image: Image.Image, config: ModelConfig) -> str:
        """Query Claude API"""
        import base64
        from io import BytesIO
        
        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        }
        
        response = requests.post(config.api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def _query_openai_api(self, prompt: str, image: Image.Image, config: ModelConfig) -> str:
        """Query OpenAI API"""
        import base64
        from io import BytesIO
        
        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }
        
        data = {
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }]
        }
        
        response = requests.post(config.api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def evaluate_single_puzzle(self, puzzle_data: Dict, model_config: ModelConfig, 
                             model_components: Optional[Dict] = None) -> Dict:
        """Evaluate model on a single puzzle"""
        
        start_time = time.time()
        
        # Query model
        if model_config.model_type == 'huggingface':
            if model_components is None:
                return {'error': 'Model components not provided for HuggingFace model'}
            response = self.query_huggingface_model(
                model_components, puzzle_data['prompt'], puzzle_data['image'], model_config
            )
        else:  # API model
            response = self.query_api_model(
                puzzle_data['prompt'], puzzle_data['image'], model_config
            )
        
        inference_time = time.time() - start_time
        
        # Store result
        result = {
            'puzzle_name': puzzle_data['name'],
            'model_name': model_config.name,
            'response': response,
            'inference_time': inference_time,
            'timestamp': time.time()
        }
        
        return result
    
    def evaluate_model(self, model_config: ModelConfig, puzzle_subset: Optional[List[str]] = None,
                      max_workers: int = 1, save_intermediate: bool = True) -> Dict:
        """Evaluate a model on all puzzles"""
        
        logger.info(f"Starting evaluation for model: {model_config.name}")
        
        # Setup model if HuggingFace
        model_components = None
        if model_config.model_type == 'huggingface':
            model_components = self.setup_huggingface_model(model_config)
            if model_components is None:
                return {'error': 'Failed to load HuggingFace model'}
        
        # Determine which puzzles to evaluate
        puzzle_folders = self.puzzle_folders
        if puzzle_subset:
            puzzle_folders = [f for f in puzzle_folders if f.name in puzzle_subset]
        
        logger.info(f"Evaluating {len(puzzle_folders)} puzzles")
        
        # Create output directory for this model
        model_output_dir = self.results_path / model_config.name
        model_output_dir.mkdir(exist_ok=True)
        
        results = []
        
        # Process puzzles
        if max_workers > 1 and model_config.model_type == 'api':
            # Use threading for API models (I/O bound)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_puzzle = {}
                
                for puzzle_folder in puzzle_folders:
                    puzzle_data = self.load_puzzle_data(puzzle_folder)
                    future = executor.submit(
                        self.evaluate_single_puzzle, puzzle_data, model_config, model_components
                    )
                    future_to_puzzle[future] = puzzle_data['name']
                
                # Collect results
                for future in tqdm(concurrent.futures.as_completed(future_to_puzzle), 
                                 total=len(puzzle_folders), desc="Evaluating"):
                    puzzle_name = future_to_puzzle[future]
                    try:
                        result = future.get(timeout=60)  # 60 second timeout
                        results.append(result)
                        
                        # Save intermediate results
                        if save_intermediate:
                            with self.lock:
                                intermediate_file = model_output_dir / f"{puzzle_name}_response.json"
                                with open(intermediate_file, 'w') as f:
                                    json.dump(result, f, indent=2, default=str)
                                    
                    except Exception as e:
                        logger.error(f"Failed to evaluate {puzzle_name}: {e}")
                        results.append({
                            'puzzle_name': puzzle_name,
                            'model_name': model_config.name,
                            'error': str(e),
                            'timestamp': time.time()
                        })
        else:
            # Sequential processing for HuggingFace models or single-threaded
            for puzzle_folder in tqdm(puzzle_folders, desc="Evaluating"):
                puzzle_data = self.load_puzzle_data(puzzle_folder)
                
                try:
                    result = self.evaluate_single_puzzle(puzzle_data, model_config, model_components)
                    results.append(result)
                    
                    # Save intermediate results
                    if save_intermediate:
                        intermediate_file = model_output_dir / f"{puzzle_data['name']}_response.json"
                        with open(intermediate_file, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                            
                except Exception as e:
                    logger.error(f"Failed to evaluate {puzzle_data['name']}: {e}")
                    results.append({
                        'puzzle_name': puzzle_data['name'],
                        'model_name': model_config.name,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                
                # Add delay for API rate limiting
                if model_config.model_type == 'api':
                    time.sleep(1)  # 1 second delay between requests
        
        # Save complete results
        output_file = model_output_dir / "all_responses.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Completed evaluation for {model_config.name}. Results saved to {output_file}")
        
        return {
            'model_name': model_config.name,
            'total_puzzles': len(puzzle_folders),
            'results_file': str(output_file),
            'results': results
        }
    
    def run_evaluation_suite(self, model_configs: List[ModelConfig], 
                           puzzle_subset: Optional[List[str]] = None) -> Dict:
        """Run evaluation on multiple models"""
        
        logger.info(f"Starting evaluation suite with {len(model_configs)} models")
        
        all_results = {}
        
        for model_config in model_configs:
            try:
                results = self.evaluate_model(model_config, puzzle_subset)
                all_results[model_config.name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_config.name}: {e}")
                all_results[model_config.name] = {'error': str(e)}
        
        # Save combined results
        combined_file = self.results_path / "evaluation_summary.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation suite completed. Summary saved to {combined_file}")
        
        return all_results

# Predefined model configurations
def get_model_configs():
    """Get predefined model configurations"""
    
    configs = [
        # HuggingFace Models
        ModelConfig(
            name="qwen_vl_05b",
            model_type="huggingface",
            model_id="Qwen/Qwen-VL",
            max_tokens=500,
            temperature=0.1
        ),
        
        ModelConfig(
            name="llava_7b",
            model_type="huggingface", 
            model_id="llava-hf/llava-1.5-7b-hf",
            max_tokens=500,
            temperature=0.1
        ),
        
        # API Models (require API keys)
        ModelConfig(
            name="claude_35_sonnet",
            model_type="api",
            model_id="claude-3-5-sonnet-20241022",
            api_endpoint="https://api.anthropic.com/v1/messages",
            max_tokens=1000,
            temperature=0.1
        ),
        
        ModelConfig(
            name="gpt4_vision",
            model_type="api",
            model_id="gpt-4-vision-preview",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            max_tokens=1000,
            temperature=0.1
        ),
    ]
    
    return configs

if __name__ == "__main__":
    # Example usage
    pipeline = ModelEvaluationPipeline()
    
    # Start with QwenVL as requested
    qwen_config = ModelConfig(
        name="qwen_vl_05b",
        model_type="huggingface",
        model_id="Qwen/Qwen-VL", 
        max_tokens=500,
        temperature=0.1
    )
    
    # Test on a small subset first
    test_puzzles = ["puzzle1", "puzzle2", "puzzle3"]
    
    logger.info("Starting evaluation with QwenVL-0.5B")
    results = pipeline.evaluate_model(qwen_config, puzzle_subset=test_puzzles)
    
    print("Evaluation completed!")
    print(f"Results: {results}")