# -*- coding: utf-8 -*-
# @Time    : 2024/6/25
# @Author  : kailisun
# @Email   : kailisun@tencent.com
# @FileName: test_business_model_release
"""
Role-playing benchmark evaluation data generation
Supports multiple common model interface calls
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.loader import DataLoader
from models.hf_chat_model import HFChatModel


def main():
    parser = argparse.ArgumentParser(description='Role-playing benchmark evaluation data generation')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name: qwen, qwen2, qwen2.5, chatglm, chatglm2, chatglm3, etc.')    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Model path or Hugging Face model ID')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Test dataset path')
    parser.add_argument('--result_path', type=str, required=True,
                       help='Result save path')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device setting: auto, cuda:0, etc.')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum generation tokens')
    
    args = parser.parse_args()
    
    # Load data
    data_loader = DataLoader(args.data_path)
    
    # Initialize model
    llm = HFChatModel(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        max_tokens=args.max_tokens
    )
    
    # Run evaluation data generation
    llm.run(data_loader, args.result_path)
    
    print(f"Evaluation data generation completed, results saved at: {args.result_path}")


if __name__ == "__main__":
    main()