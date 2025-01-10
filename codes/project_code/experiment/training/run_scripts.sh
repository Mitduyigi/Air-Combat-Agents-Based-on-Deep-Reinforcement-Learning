#!/bin/bash

# 执行第一个 Python 程序
echo "Running experiment_learning_rate.py"
python experiment_learning_rate.py

# 执行第二个 Python 程序
echo "Running experiment_hidden_dim.py"
python experiment_hidden_dim.py

# 执行第三个 Python 程序
echo "Running experiment_improvement.py"
python experiment_improvement.py

# 执行第四个 Python 程序
echo "Running experiment_improvement_ablation.py"
python experiment_improvement_ablation.py

# 脚本结束
echo "All scripts have been executed!"