# -*- coding: utf-8 -*-

import json
import os
import argparse
from collections import defaultdict

def save_json(obj, save_path, indent=4):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_dirty_json(input_file_path):
    datas = []
    if not os.path.exists(input_file_path): return datas
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                datas.append(json.loads(line.rstrip(';')))
            except: continue
    return datas

def run_statistics(args):
    # 解析从 Shell 传来的列表
    eval_models = args.eval_models.split(',')
    baseline_models = args.baseline_models.split(',')
    
    self_awareness = ["属性一致性", "幻觉与拒答 - 知识边界", "幻觉与拒答 - 人设虚假属性", "人设外知识", "语言风格一致性"]
    conversation_ability = ["情绪价值", "话题推进 - 抛出新话题", "话题推进 - 推动话题", "回复上轮动作（强调回应）", "记忆能力-问询", "闲聊"]
    all_metrics = self_awareness + conversation_ability

    # 1. Data deduplication and aggregation
    no_dup_dir = os.path.join(args.data_folder, 'no_duplication')
    os.makedirs(no_dup_dir, exist_ok=True)
    all_datas = []
    
    for file_name in os.listdir(args.data_folder):
        if file_name.endswith('.json') and 'all_data' not in file_name and file_name != 'no_duplication':
            path = os.path.join(args.data_folder, file_name)
            datas = load_dirty_json(path)
            quchong_datas = []
            seen = set()
            for d in datas:
                key = f"{d.get('new_ID')}_{d.get('metrics')}"
                if key not in seen:
                    seen.add(key)
                    quchong_datas.append(d)
            all_datas.extend(quchong_datas)

    # 2. Calculate win rate
    res_matrix = {m1: {m2: {mt: {'win':0, 'tie':0, 'fail':0} for mt in all_metrics} for m2 in baseline_models} for m1 in eval_models}
    for item in all_datas:
        m1, m2, mt = item.get('model1'), item.get('model2'), item.get('metrics')
        if m1 in eval_models and m2 in baseline_models and mt in all_metrics:
            win_model = item['score'][0]['win_model']
            res_matrix[m1][m2][mt]['tie' if win_model == 'tie' else ('win' if win_model == m1 else 'fail')] += 1

    # 3. Print core statistics table (Overall/SA/CA)
    print("\n" + "="*85)
    print(f"{'Model Name':<50} | {'Overall':>8} | {'SA':>8} | {'CA':>8}")
    print("-" * 85)

    summary_results = {}
    for m1 in eval_models:
        metric_scores = {}
        for mt in all_metrics:
            w, total = 0, 0
            for m2 in baseline_models:
                w += res_matrix[m1][m2][mt]['win']
                total += sum(res_matrix[m1][m2][mt].values())
            metric_scores[mt] = w / total if total > 0 else 0.0
        
        sa = sum(metric_scores[mt] for mt in self_awareness) / len(self_awareness)
        ca = sum(metric_scores[mt] for mt in conversation_ability) / len(conversation_ability)
        ov = sum(metric_scores.values()) / len(all_metrics)
        summary_results[m1] = {'overall': ov, 'sa': sa, 'ca': ca, 'details': metric_scores}
        print(f"{m1:<50} | {ov:>8.4f} | {sa:>8.4f} | {ca:>8.4f}")

    # 4. Print horizontal comparison table across all dimensions
    print("\n" + "="*30 + " Win Rate Matrix by Dimension " + "="*30)
    # Header: dimension names
    metric_header = f"{'Model':<40}"
    for mt in all_metrics: metric_header += f" | {mt[:4]}" # Take first 4 characters as abbreviation
    print(metric_header)
    
    for m1 in eval_models:
        row = f"{m1[:40]:<40}"
        for mt in all_metrics:
            row += f" | {summary_results[m1]['details'][mt]:.3f}"
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--eval_models", type=str, required=True, help="Comma separated list")
    parser.add_argument("--baseline_models", type=str, required=True, help="Comma separated list")
    run_statistics(parser.parse_args())