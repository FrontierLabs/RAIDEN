# -*- coding: utf-8 -*-
# @Time    : 2024/4/25
# @Author  : ziweibai
# @Email   : ziweibai@tencent.com
# @FileName: loader
"""
    Description:
        
"""

import json
import os

def load_results(file_path):
    data = {}
    with open(file_path) as reader:
        for line in reader:
            line = json.loads(line)
            data[line["ID"]] = line["response"]
    return data

def load_all_results(dir_path):
    all_data = {}
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if not file_path.endswith("json"):
            continue
        model_name = filename.replace(".json", "")
        all_data[model_name] = load_results(file_path)
    return all_data
