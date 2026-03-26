# -*- coding: utf-8 -*-

"""
Role-playing benchmark pairwise evaluation
Using Hugging Face model calling method
"""

import json
import os
import sys
import re
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.loader import DataLoader
from postprocess.loader import load_results
from data.generic import Role
from models.reward_model import RewardModel


TEMPLATE = """<reserved_102> 
请你扮演一个角色扮演对话模型评测人员，对两个对话模型生产的结果进行排序并给出理由。

以下是要扮演的角色{npc_name}的介绍:
{info}

这是对话历史内容：
{history}

这是正确的参考回复: {reference}
===============================
【模型1的回复: {result1}】
【模型2的回复: {result2}】
===============================

以上是来自两个模型的结果，它们已经被随机化顺序。请严格根据评测标准进行评估和排序。
这是评测标准：{demand}

格式如下：
排序结果： 模型1>模型2 / 模型1<模型2 / 模型1=模型2
理由：
<reserved_103>"""


# Evaluation metrics definition
# Note: The keys below (A, B, C, ...) are the original dimension codes used in data files.
# They are mapped to public dimension names via `dimension_mapping` below (for display/reporting only).
# Do NOT change the keys here, as data files still use the original codes.
metrics_dict = {
    "A": "属性一致性",
    "B": "幻觉与拒答 - 知识边界", 
    "C": "幻觉与拒答 - 人设虚假属性",
    "D": "人设外知识",
    "E": "语言风格一致性",
    "F": "情绪价值",
    "G": "话题推进 - 抛出新话题",
    "H": "话题推进 - 推动话题",
    "I": "给出符合当前轮次的动作（强调动作本身）",
    "J": "回复上轮动作（强调回应）",
    "K1": "K1记忆能力-信息源",
    "K2": "记忆能力-问询",
    "L": "闲聊",
}

# Dimension name mapping: original internal codes -> public dimension names used in the paper.
# This mapping is for display/reporting purposes only.
# The actual data files still use the original codes (A, B, C, ...), so do NOT use these
# public names when reading or processing data.
dimension_mapping = {
    "A": "SBK",
    "B": "RCB",
    "C": "SCK",
    "D": "SAK",
    "E": "PLS",
    "F": "ER",
    "G": "TS",
    "H": "TA",
    "I": "null",
    "J": "PB",
    "K1": "CM1",
    "K2": "CM2",
    "K1-1": "CM1-1",
    "K1-2": "CM1-2",
    "K1-3": "CM1-3",
    "K1-4": "CM1-4",
    "K2-1": "CM2-1",
    "K2-2": "CM2-2",
    "K2-3": "CM2-3",
    "K2-4": "CM2-4",
    "L": "CC",
}

# Evaluation criteria description
demands = {
    "属性一致性": "评测模型能否根据人设信息正确回答用户的问题。\n排序标准：【信息完全正确且全面】 优于 【信息完全正确但不全面】 优于 【信息部分正确，存在部分错误】 优于 信息完全不正确。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "幻觉与拒答 - 知识边界": "评测模型能否对角色人设边界外的知识进行拒答（如古代人物被问及现代话题等）。\n排序标准：【正确拒绝回答 】 优于 【告知用户不了解该话题，但仍给出了该话题的细节】 优于 【生成与该话题相关的细节，但与人设信息强关联】 优于 【生成与人设无关的该话题细节】。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "幻觉与拒答 - 人设虚假属性": "评测模型能否对用户错误的诱导性提问进行更正。\n排序标准：【能更正信息且更合理的回复 】 优于 【告知对方说错了，但没有给出正确信息的回复】 优于 【肯定对方话题，但后面给出了正确信息的回复】 优于 【完全被对方误导的回复】。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "人设外知识": "评测模型能否正确回答人设外的问题。人设外问题指角色的人设描述未给出，但真实存在的信息。\n排序标准：【信息完全正确且全面】 优于 【信息完全正确但不全面】 优于 【信息部分正确，存在部分错误】 优于 信息完全不正确。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "语言风格一致性": "评测模型生成回复的语言风格是否符合人设要求的风格。生成结果的风格与人设越接近，模型得分越高。\n排序标准：【回复与人设要求风格一致，恰当的使用了口头禅】 优于 【回复与人设要求风格一致，未使用口头禅】 优于 【回复与人设要求风格一致，使用了人设中不恰当的口头禅导致整个回复不通顺】 优于 【风格不一致】。\n 符合上述标准情况下，有致命伤的模型（认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "情绪价值": "评测模型生成的结果能否为用户提供情绪价值，在用户表现出积极情绪时，模型要给予肯定；用户表现出消极情绪时，模型要进行安慰、鼓励。提供的情绪价值越高，模型的评分越高。",
    "话题推进 - 抛出新话题": "评测模型是否具有开启新话题的能力。当对话的话题无法推进或用户表明不想继续当前话题时，模型应主动抛出新的相关话题。\n排序标准：【流畅、合理地给出新话题】 优于 【生硬的给出新话题】 优于 【委婉、不明显地提到新话题】 优于 【没给新话题】。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "话题推进 - 推动话题": "评测模型是否具有推进话题进行的能力。在用户当前对话表示的信息量比较少导致话题停滞时，模型要主动推动话题进行。推进的话题越能激发用户继续聊天的欲望，模型评分越高。",
    "回复上轮动作（强调回应）": "评测模型能否给出合理的连续性动作描述。动作描述是指对话中用括号括起来的部分。\n排序标准：【括号中的动作描述与上文连贯，且语言内容与动作一致】 优于 【括号中的动作描述与上文连贯，但语言内容与动作不一致】 优于 【括号中的动作描述与上文不连贯】 优于 【未给出括号中的动作描述】。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "记忆能力-问询": "评测模型是否能正确记忆历史对话中的内容。请根据参考答案和历史对话，评测模型回复是否正确。\n排序标准：【正确、全面地回答用户问题】 优于 【正确回答用户问题，但不全面】 优于 【模型表示记得上文信息，但没有给出正确答案】 优于 【拒绝回答】 优于 【错误回答】。\n 符合上述标准情况下，有致命伤的模型（风格明显不符合人设，认为自己是AI模型，非常啰嗦，逻辑错误）更差；如果两个模型正确率一致且没有致命伤，可以标为 模型1=模型2。",
    "闲聊": "综合评测模型的回复质量。模型回复的内容逻辑越连贯、对话越流畅、越符合人类自然交流习惯，模型评分越高。\n排序标准：【与上文相关、逻辑正确，符合人类自然交流习惯，话题深入，语言风格与人设一致的回复】 优于 【话题不深入的回复】 优于 【语言风格与人设不一致的回复】 优于 【不符合人类自然交流习惯的回复】优于 【逻辑错误的回复】优于 【上下文不相关的回复】。"
}


def combine_message(messages, npc_name):
    """Merge message history"""
    data = []
    for message in messages:
        if message["role"] == Role.USER:
            data.append("用户：" + message["text"])
        else:
            data.append(npc_name + "：" + message["text"])
    return "\n".join(data)


class RewardModelEvaluate:
    """Evaluation model class"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """Initialize evaluation model
        
        Args:
            model_path: Evaluation model path or Hugging Face model ID
            device: Device setting
        """
        if model_path is None:
            raise ValueError("Must provide evaluation model path")
        
        self.llm = RewardModel(
            model_path=model_path,
            device=device
        )

        # self.llm.init_model(
        #     model_name=model_name if model_name else "reward_model",
        #     model_path=model_path,
        #     device=device
        # )
        
        print(f"Evaluation model initialization completed: {model_path}")
    
    def parse_output(self, output: str):
        """Parse model output, extract ranking results and reasons
        
        Args:
            output: Model original output
            
        Returns:
            win_model: Winning model identifier
            reason: Ranking reason
        """
        rank_result = re.findall(r"排序结果[:：](.*)", output)
        if not rank_result:
            return None, None

        rank_result = rank_result[0]
        if "=" in rank_result:
            win_model = "tie"
        elif re.findall(r"1\s*<\s*.*2", rank_result) or re.findall(r"2\s*>\s*.*1", rank_result):
            win_model = 1
        elif re.findall(r"1\s*>\s*.*2", rank_result) or re.findall(r"2\s*<\s*.*1", rank_result):
            win_model = 0
        else:
            print(f"Unable to parse ranking result: {rank_result}")
            return None, None

        reason = re.findall(r"理由[:：]([\s\S]*)$", output)
        if reason:
            reason = reason[0].strip()
        else:
            reason = None

        return win_model, reason
    
    def evaluate_one_case(self, npc_name: str, info: str, history: str, reference: str, 
                         result1: str, result2: str, demand: str):
        """Evaluate single sample
        
        Args:
            npc_name: Character name
            info: Character introduction
            history: Dialogue history
            reference: Reference response
            result1: Model1 response
            result2: Model2 response
            demand: Evaluation criteria
            
        Returns:
            win_model: Winning model identifier
            reason: Ranking reason
            output: Model original output
        """
        prompt = TEMPLATE.format(
            npc_name=npc_name, 
            info=info.strip(), 
            history=history.strip(),
            reference=reference, 
            result1=result1, 
            result2=result2, 
            demand=demand
        )

        output = self.llm.call_model(prompt.strip())
        win_model, reason = self.parse_output(output)
        
        return win_model, reason, output
    
    def evaluate(self, model1: str, model2: str, model1_result_file: str, save_path: str,
data_paths: list = None, baseline_result_files: list = None):
        """Execute pairwise evaluation
        
        Args:
            model1: Model to be evaluated name
            model2: Comparison model name
            model1_result_file: Model1 result file path
            save_path: Evaluation result save path
            data_paths: List of (dialogue_type, data_path) tuples for evaluation data.
                        e.g. [("short", "/path/to/short/"), ("long", "/path/to/long/")]
                        If None, defaults to [("default", "./data/")]
            baseline_result_files: List of baseline result file paths.
                        e.g. ["/path/to/baseline1.json", "/path/to/baseline2.json"]
                        If None, defaults to [model2 name + ".json" in current directory]
        """
        # Load data path configuration
        if data_paths is None:
            data_paths = [("default", "./data/")]

        # Load model2 baseline results (support multiple baseline files)
        if baseline_result_files is None:
            baseline_result_files = [f"{model2}.json"]

        model2_results = {}
        for f in baseline_result_files:
            if not os.path.exists(f):
                print(f"Model2 result file does not exist: {f}")
                continue
            partial = load_results(f)
            model2_results.update(partial)

        if not model2_results:
            print("Warning: No model2 results loaded, evaluation may be skipped.")

        # Load model1 results (load once, reuse across all data paths)
        model1_results = load_results(model1_result_file)

        # Create output directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as writer:
            
            for dialogue_type, data_path in data_paths:
                if not os.path.exists(data_path):
                    print(f"Data path does not exist: {data_path}")
                    continue
                    
                print(f"Processing {dialogue_type} dialogue data...")
                
                # Load data
                data_loader = DataLoader(data_path)
                
                for data in tqdm(data_loader, desc=f"Evaluating {dialogue_type} dialogue"):
                    ID = data.ID
                    
                    # Process evaluation metrics
                    metrics_to_evaluate = []
                    for metric in data.metrics:
                        if metric == "K1":  # Skip K1 metric
                            continue
                        try:
                            metrics_to_evaluate.append(metrics_dict[metric])
                        except KeyError:
                            # Process composite metrics
                            metrics = re.split(r"[、\s]", metric)
                            for m in metrics:
                                if m == "K1":
                                    continue
                                m = m.split("-")[0].strip()
                                try:
                                    metrics_to_evaluate.append(metrics_dict[m])
                                except KeyError:
                                    pass
                    
                    for metric in metrics_to_evaluate:
                        if metric not in demands:
                            continue
                            
                        # Get model responses
                        result1 = model1_results.get(ID, "").replace('[DLG]', '')
                        result2 = model2_results.get(ID, "")
                        
                        if not result1 or not result2:
                            continue
                        
                        # Merge dialogue history
                        history = combine_message(data.messages, data.npc_name)
                        
                        # Execute evaluation
                        try:
                            win_model, reason, output = self.evaluate_one_case(
                                data.npc_name, data.npc_setting, history, data.reference,
                                result1, result2, demands[metric]
                            )
                        except Exception as e:
                            print(f"Error evaluating sample {ID} metric {metric}: {e}")
                            continue
                        
                        # Build result record
                        result = {
                            "dialogue_type": dialogue_type,
                            "new_ID": ID,
                            "npc_name": data.npc_name,
                            "history": history,
                            "metrics": metric,
                            "response": data.reference,
                            "model1": model1,
                            "model2": model2,
                            "result1": result1,
                            "result2": result2,
                            "score": [{
                                "win_model": model1 if win_model == 0 else model2 if win_model == 1 else "tie",
                                "reason": reason,
                                "output": output
                            }]
                        }
                        
                        # Write result
                        writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                        writer.flush()
        
        print(f"Evaluation completed, results saved at: {save_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Role-playing benchmark pairwise evaluation')
    parser.add_argument('--model1', type=str, required=True, help='Model to be evaluated name')
    parser.add_argument('--model2', type=str, required=True, help='Comparison model name')
    parser.add_argument('--model1_result_file', type=str, required=True, 
                       help='Model1 evaluation result file path')
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='Evaluation result output folder')
    parser.add_argument('--reward_model_path', type=str, required=True,
                       help='Evaluation model path or Hugging Face model ID')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device setting: auto, cuda:0, etc')
    parser.add_argument('--data_paths', type=str, nargs='+', default=None,
                       help='Evaluation data paths in "type:path" format, e.g. short:/data/short/ long:/data/long/')
    parser.add_argument('--baseline_result_files', type=str, nargs='+', default=None,
                       help='Model2 baseline result file paths (support multiple files), e.g. /path/baseline1.json /path/baseline2.json')
    
    args = parser.parse_args()

    # Parse data_paths argument: "type:path" -> [(type, path), ...]
    data_paths = None
    if args.data_paths:
        data_paths = []
        for item in args.data_paths:
            if ":" in item:
                dtype, dpath = item.split(":", 1)
                data_paths.append((dtype.strip(), dpath.strip()))
            else:
                data_paths.append((os.path.basename(item.rstrip("/")), item))
    
    # Initialize evaluator
    evaluator = RewardModelEvaluate(
        model_path=args.reward_model_path,
        device=args.device
    )
    
    # Build save path
    save_path = os.path.join(args.output_folder, f"{args.model1}_{args.model2}.json")
    
    # Execute evaluation
    evaluator.evaluate(
        args.model1, args.model2, args.model1_result_file, save_path,
        data_paths=data_paths,
        baseline_result_files=args.baseline_result_files
    )


if __name__ == "__main__":
    main()