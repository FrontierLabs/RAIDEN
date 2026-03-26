#!/bin/bash

# Role-playing Benchmark Evaluation Script
# Supports general model interfaces and Hugging Face model calls

# Configuration Parameters
# Target Model Configuration
MODEL1_NAME="qwen2.5"  # Target model to be evaluated. Model type: qwen, qwen2, qwen2.5, chatglm, chatglm2, chatglm3, etc.
MODEL1_PATH="Qwen/Qwen2.5-7B-Instruct"  # Hugging Face model ID or local path

# Comparison Model Configuration
MODEL2_NAME="minimax-abab6-chat"  # Baseline model name
BASELINE_RESULT_FILES=("./baseline_results/minimax-abab6-chat.json")  # Baseline result files (support multiple files)

# Data Path Configuration
DATA_DIR="./release_data"
# Evaluation data paths for reward model, in "type:path" format (support multiple entries)
EVAL_DATA_PATHS=("default:${DATA_DIR}")
MODEL1_RESULT_FILE_PATH="./results/${MODEL1_NAME}_test_results.json"  # Model 1 evaluation result path
OUTPUT_FOLDER_PATH="./evaluate_results"  # Evaluation result output folder

# Evaluation Model Configuration (for pairwise comparison)
REWARD_MODEL_PATH="FrontierLab/RPCAJudger"  # Evaluation model's Hugging Face ID or local path

# Device Configuration
DEVICE="auto"  # Device setting: auto, cuda:0, cuda:1, etc.
MAX_TOKENS=500  # Maximum generation tokens

export TERM=xterm-256color

# Step 1: Generate evaluation data for the target model
echo "=== Step 1: Generate evaluation data for the target model ==="
python tests/test_business_model_release.py \
    --model_name "${MODEL1_NAME}" \
    --model_path "${MODEL1_PATH}" \
    --data_path "${DATA_DIR}" \
    --result_path "${MODEL1_RESULT_FILE_PATH}" \
    --device "${DEVICE}" \
    --max_tokens "${MAX_TOKENS}"

# Check if Step 1 succeeded
if [ $? -ne 0 ]; then
    echo "Step 1 execution failed, please check configuration and model path"
    exit 1
fi

echo "Step 1 completed, evaluation data generated: ${MODEL1_RESULT_FILE_PATH}"

# Step 2: Call evaluation model for pairwise comparison
echo "=== Step 2: Call evaluation model for pairwise comparison ==="
python evaluate/reward_model_evaluate.py \
    --model1 "${MODEL1_NAME}" \
    --model2 "${MODEL2_NAME}" \
    --model1_result_file "${MODEL1_RESULT_FILE_PATH}" \
    --output_folder "${OUTPUT_FOLDER_PATH}" \
    --reward_model_path "${REWARD_MODEL_PATH}" \
    --device "${DEVICE}" \
    --data_paths "${EVAL_DATA_PATHS[@]}" \
    --baseline_result_files "${BASELINE_RESULT_FILES[@]}"

# Check if Step 2 succeeded
if [ $? -ne 0 ]; then
    echo "Step 2 execution failed, please check evaluation model configuration"
    exit 1
fi

echo "Step 2 completed, pairwise comparison results generated"

# --- Step 3: Result Statistics ---
echo "=== Step 3: Result Statistics and Deduplication Analysis ==="
# python evaluate/stat_results.py --data_folder "${OUTPUT_FOLDER_PATH}"
# --- 1. Configuration Parameters (managed here uniformly) ---

# Target model list (space separated)
# EVAL_MODELS=("model1" "model2" "model3"..)
EVAL_MODELS=("${MODEL1_NAME}")


# Baseline model list (space separated)
BASELINE_MODELS=("minimax-abab6-chat" "character_glm" "Atom-7B-Chat")

# Environment variable to resolve TTY error
export TERM=xterm-256color

# --- 2. Result Statistics ---
echo "=== Step 3: Result Statistics and Deduplication Analysis ==="

# Convert Shell arrays to comma-separated strings for Python
EVAL_MODELS_STR=$(IFS=,; echo "${EVAL_MODELS[*]}")
BASELINE_MODELS_STR=$(IFS=,; echo "${BASELINE_MODELS[*]}")

python evaluate/stat_results.py \
    --data_folder "${OUTPUT_FOLDER_PATH}" \
    --eval_models "${EVAL_MODELS_STR}" \
    --baseline_models "${BASELINE_MODELS_STR}"

echo "=== Evaluation process completed ==="
