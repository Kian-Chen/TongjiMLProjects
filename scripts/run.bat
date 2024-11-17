@echo off
:: 激活 conda 环境
call conda activate llm4sst

:: 定义模型和数据集列表
set MODELS=bayesian svc knn logistic_regression
set DATASETS=mnist cifar-10 cifar-100

:: 启用延迟变量扩展
setlocal enabledelayedexpansion

:: 跳转到 run.py 所在目录
cd ..

:: 循环所有模型和数据集的组合
for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (
        :: 构造日志文件名
        set LOG_FILE=%%M.txt
        python run.py --model %%M --data %%D --log_file !LOG_FILE!
    )
)

:: 禁用延迟变量扩展
endlocal
