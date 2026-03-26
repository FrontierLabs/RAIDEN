# -*- coding: utf-8 -*-
# @Time    : 2024/6/25
# @Author  : kailisun
# @Email   : kailisun@tencent.com
# @FileName: hf_chat_model
"""
General model interface class
Supports common models like Qwen, Llama, ChatGLM, Baichuan, etc.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict, Any

# from models.basic import BasicModel
from data.loader import DataLoaderOutput
from data.generic import Role

class HFChatModel:
    """General model interface class"""
    def __init__(self, model_name, **kwargs):
        self.init_model(model_name, **kwargs)
    
    
    def init_model(self, model_name: str, model_path: str, device: str = "auto", 
                  max_tokens: int = 500, **kwargs):
        """Initialize model
        
        Args:
            model_name: Model name (qwen, llama, chatglm, baichuan, etc.)
            model_path: Model path or Hugging Face model ID
            device: Device setting
            max_tokens: Maximum generation tokens
        """
        self.model_name = model_name.lower()
        self.max_tokens = max_tokens
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            
            # Special handling: add pad_token for some models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Set generation configuration
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            
            print(f"Model initialization completed: {model_name} on {device}")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _is_qwen_family(self) -> bool:
        """Determine if it's a Qwen series model"""
        return self.model_name in ['qwen', 'qwen2', 'qwen2.5']
    
    def _is_chatglm_family(self) -> bool:
        """Determine if it's a ChatGLM series model"""
        return self.model_name in ['chatglm', 'chatglm2', 'chatglm3']
    
    def _format_messages_for_model(self, data: DataLoaderOutput) -> List[Dict[str, str]]:
        """Format messages into model input format
        
        Args:
            data: Data loader output
            
        Returns:
            Formatted message list
        """
        messages = []
        
            # System prompt
            system_content = f"Please role-play {data.npc_name} and converse with me. Here is his personal introduction:\n{data.npc_setting}"
        
        if self._is_qwen_family():
            # Qwen family model format
            messages.append({"role": "system", "content": system_content})
        elif self._is_chatglm_family():
            # ChatGLM family model format
            messages.append({"role": "system", "content": system_content})
        else:
            # Default format
            messages.append({"role": "system", "content": system_content})

        # Data normalization, user first, then bot
        if data.messages[0]["role"] == Role.ASSISTANT:
            messages.append({
                "role": "user",
                "content": ""
            })
        
        # Add dialogue history
        for message in data.messages:
            if message["role"] == Role.USER:
                messages.append({"role": "user", "content": message["text"]})
            else:
                messages.append({"role": "assistant", "content": message["text"]})
        
        return messages
    
    def _get_model_specific_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Generate model-specific prompt format based on model type
        
        Args:
            messages: Formatted message list
            
        Returns:
            Model-specific prompt string
        """
        try:
            # Use tokenizer's apply_chat_template method to automatically generate prompt
            # add_generation_prompt=True adds assistant's start marker at the end of prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except (AttributeError, KeyError, ValueError) as e:
            # If tokenizer doesn't support apply_chat_template, fall back to original logic
            print(f"Warning: tokenizer doesn't support apply_chat_template, fall back to manual concatenation: {e}")
            prompt = ""
            
            # if self._is_qwen_family():
            #     # Qwen格式: <|im_start|>role\ncontent<|im_end|>
            #     for msg in messages:
            #         prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            #     prompt += "<|im_start|>assistant\n"
            #     # 注意: assistant的<|im_end|>会在模型生成回复后自动添加
                
            # elif self._is_chatglm_family():
            #     # ChatGLM格式: <|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>\n
            #     for msg in messages:
            #         if msg['role'] == 'system':
            #             prompt += f"<|system|>\n{msg['content']}\n"
            #         elif msg['role'] == 'user':
            #             prompt += f"<|user|>\n{msg['content']}\n<|assistant|>\n"
            # else:
            #     # 默认格式: 简单的角色: 内容格式
            #     for msg in messages:
            #         prompt += f"{msg['role']}: {msg['content']}\n"
            #     prompt += "assistant: "
            
            return prompt
    
    def get_response(self, data: DataLoaderOutput) -> str:
        """Get model response
        
        Args:
            data: Data loader output
            
        Returns:
            Model-generated response text
        """
        try:
            # Format messages
            messages = self._format_messages_for_model(data)
            
            # Generate model-specific prompt
            prompt = self._get_model_specific_prompt(messages)
            
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(response, skip_special_tokens=True)
            
            # Clean response text
            response_text = self._clean_response(response_text)
            
            print(f"Model response: {response_text}")
            return response_text
        except Exception as e:
            print(f"Model inference failed: {e}")
            return ""
    
    def _clean_response(self, response: str) -> str:
        """Clean model response text
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove extra spaces and line breaks
        response = response.strip()
        
        # If using apply_chat_template, tokenizer usually handles special token cleanup automatically
        # But retain some common cleanup logic for safety
        
        return response
    
    # def run(self, data_loader, result_path: str):
    #     """运行评测数据生成
        
    #     Args:
    #         data_loader: 数据加载器
    #         result_path: 结果保存路径
    #     """
    #     import json
    #     from tqdm import tqdm
        
    #     results = []
        
    #     # print(f"开始生成评测数据，共{len(data_loader)}条样本...")
        
    #     for data in tqdm(data_loader):
    #         try:
    #             response = self.get_response(data)
    #             result_item =  {
    #                 "ID": data.ID,
    #                 "response": response,
    #             }
    #             results.append(result_item)
    #         except Exception as e:
    #             print(f"处理样本 {data.ID} 时出错: {e}")
        
    #     # 保存结果
    #     with open(result_path, 'w', encoding='utf-8') as f:
    #         json.dump(results, f, ensure_ascii=False, indent=2)
        
    #     print(f"评测数据生成完成，结果保存在: {result_path}")

    def run(self, data_loader, result_path: str):
        """Run evaluation data generation (real-time storage by line)
        
        Args:
            data_loader: Data loader
            result_path: Result save path (recommended suffix .jsonl)
        """
        import json
        from tqdm import tqdm
        
        # Use 'a' (append) mode or 'w' mode to open file
        # Use 'w' if you want to overwrite old files each time
        with open(result_path, 'w', encoding='utf-8') as f:
            for data in tqdm(data_loader):
                try:
                    response = self.get_response(data)
                    result_item = {
                        "ID": data.ID,
                        "response": response,
                    }
                    
                    # Convert single result to JSON string and write line by line
                    line = json.dumps(result_item, ensure_ascii=False)
                    f.write(line + '\n')
                    
                    # Optional: force flush buffer to ensure real-time writing to disk
                    f.flush() 
                    
                except Exception as e:
                    print(f"\nError processing sample {data.ID}: {e}")
        
        print(f"Evaluation data generation completed, results saved line by line at: {result_path}")


# Qwen family model format specifications:
# Correct format example:
#   <|im_start|>system
#   {system_prompt}<|im_end|>
#   <|im_start|>user
#   {user_message}<|im_end|>
#   <|im_start|>assistant
#   {assistant_message}<|im_end|>
# 
# Code implementation:
# 1. _get_model_specific_prompt method adds <|im_end|> for system and user messages
# 2. Assistant message only generates start marker <|im_start|>assistant\n
#   Model automatically adds <|im_end|> after generating response
# 3. _clean_response method cleans <|im_end|> markers from response

# ChatGLM family model format specifications:
# Correct format example:
#   <|system|>
#   {system_prompt}
#   <|user|>
#   {user_message}
#   <|assistant|>
#   {assistant_message}
# 
# Code implementation:
# 1. _get_model_specific_prompt method uses <|system|>, <|user|>, <|assistant|> tags
# 2. Assistant message only generates start marker <|assistant|>\n
#   Model doesn't include special markers after generating response
# 3. _clean_response method cleans <|system|>, <|user|>, <|assistant|> markers from response