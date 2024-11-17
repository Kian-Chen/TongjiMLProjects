#!/bin/bash

# 定义模型和数据集列表
MODELS=("bayesian" "svc" "knn" "logistic_regression")
DATASETS=("mnist" "cifar-10" "cifar-100")

# 循环所有模型和数据集的组合
for MODEL in "${MODELS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        # 构造日志文件名
        LOG_FILE="${MODEL}.txt"
        
        # 执行 Python 脚本
        python run.py --model "$MODEL" --data "$DATA" --log_file "$LOG_FILE"
    done
done
