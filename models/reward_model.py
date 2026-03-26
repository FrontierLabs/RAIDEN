# -*- coding: utf-8 -*-
# @Time    : 2024/6/25
# @Author  : kailisun
# @Email   : kailisun@tencent.com
# @FileName: reward_model
"""
Evaluation model class
Supports Hugging Face model calling
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict, Any

from models.basic import BasicModel


class RewardModel():
    """Evaluation model class, supports Hugging Face model calling"""

    def __init__(self, model_path: str, device: str = "auto", max_tokens: int = 300):
        """Initialize evaluation model
        
        Args:
            model_path: Model path or Hugging Face model ID
            device: Device setting (auto, cuda:0, etc.)
            max_tokens: Maximum generation tokens
        """
        self.max_tokens = max_tokens
        
        print(f"Loading evaluation model: {model_path}")
        print(f"Device setting: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            padding_side="left"
        )
        
        # Special handling: add pad_token for some models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            # Specify specific device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
        
        # Set generation configuration
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Evaluation model loading completed: {model_path}")
        print(f"Model device: {self.model.device}")
    
    def call_model(self, query: str) -> str:
        """Call model for inference
        
        Args:
            query: Input query text
            
        Returns:
            Model-generated response text
        """
        # Encode input
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            add_special_tokens=False, 
            padding=True,
            truncation=True, 
            max_length=8192, 
            return_token_type_ids=False
        )
        
        # Move to model device
        inputs = {key: inputs[key].to(self.model.device) for key in inputs}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode response
        response = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        # Clean response text
        response_text = self._clean_response(response_text)
        
        return response_text
    
    def _clean_response(self, response: str) -> str:
        """Clean model response text
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove common special tokens
        special_tokens = [
            '<|im_end|>', '<|endoftext|>', '</s>', 
            '<reserved_103>', '<|im_start|>', '[DLG]'
        ]
        
        for token in special_tokens:
            response = response.replace(token, '')
        
        # Remove extra spaces and line breaks
        response = response.strip()
        
        return response
    
    def batch_call_model(self, queries: List[str]) -> List[str]:
        """Batch call model for inference
        
        Args:
            queries: Input query text list
            
        Returns:
            Model-generated response text list
        """
        responses = []
        
        for query in queries:
            try:
                response = self.call_model(query)
                responses.append(response)
            except Exception as e:
                print(f"Error processing query: {e}")
                responses.append("")
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Model information dictionary
        """
        return {
            "model_path": self.model.config.name_or_path,
            "device": str(self.model.device),
            "max_tokens": self.max_tokens,
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__
        }


# Helper functions
def load_results(file_path: str) -> Dict[str, str]:
    """Load model result file
    
    Args:
        file_path: Result file path
        
    Returns:
        Result dictionary {sample ID: model reply}
    """
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Failed to load result file: {e}")
        return {}


def save_results(results: Dict[str, str], file_path: str):
    """Save model result file
    
    Args:
        results: Result dictionary
        file_path: Save path
    """
    import json
    import os
    
    # Create directory
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {file_path}")


# Usage example
if __name__ == "__main__":
    # Example: initialize evaluation model
    model = RewardModel(
        model_path="your-huggingface-model-id",
        device="auto",
        max_tokens=300
    )
    
    # Example: call model
    query = "请对以下两个回复进行排序..."
    response = model.call_model(query)
    print(f"Model response: {response}")
    
    # Get model information
    info = model.get_model_info()
    print(f"Model information: {info}")